#!/usr/bin/env python3
"""Experiment 07: PaddleOCR-VL vision-prefill reference and candidate benchmark.

This file is intentionally self-contained apart from the local model/config files in
this experiment folder. It does not import helpers from older experiments.

Subcommands:

  make-baseline
    Select real OmniDocBench GT crops, run the safe eager reference path, and
    store per-crop tensors on disk.

  compare
    Re-run the same crops with a candidate path and compare visual features,
    projected image embeddings, and prefill logits against the stored baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image
from tokenizers import Tokenizer

from local_modeling_paddleocr_vl import (
    LocalPaddleOCRVLForConditionalGeneration,
    _resolve_model_dir,
    apply_rotary_pos_emb_vision,
    attention_softmax,
    get_vision_attention_impl,
    get_vision_prompt_fa_layout,
    get_vision_softmax_dtype_mode,
    vision_prompt_flash_attention_bnsd,
)


Image.MAX_IMAGE_PIXELS = None

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
AOE_ROOT = REPO_ROOT.parent

IMAGE_TOKEN = "<|IMAGE_PLACEHOLDER|>"
IMAGE_START = "<|IMAGE_START|>"
IMAGE_END = "<|IMAGE_END|>"
BOS = "<|begin_of_sentence|>"

NPU_JIT_COMPILE_CHOICES = ("default", "off", "on")
DTYPE_CHOICES = ("fp16", "float16", "fp32", "float32", "bf16", "bfloat16")
VISION_ATTENTION_CHOICES = ("manual", "prompt_flash_attention")
TIMING_MODE_CHOICES = ("phase_sync", "e2e", "both")
VISION_COMPILE_BACKEND_CHOICES = ("none", "default", "aot_eager", "inductor", "torchair")
CANDIDATE_VISION_PATH_CHOICES = ("eager_visual", "static_visual")
STATIC_VISUAL_PAD_MODE_CHOICES = ("none", "mask_pad_one", "mask_pad_to_128")

LAYOUT_PROMPT_BY_LABEL = {
    "formula": "Formula Recognition:",
    "formula_number": "OCR:",
    "equation": "Formula Recognition:",
    "equation_isolated": "Formula Recognition:",
    "equation_semantic": "Formula Recognition:",
    "table": "Table Recognition:",
    "table_title": "OCR:",
    "table_caption": "OCR:",
    "chart": "Chart Recognition:",
    "chart_title": "OCR:",
    "spotting": "Spotting:",
    "seal": "Seal Recognition:",
}

DEFAULT_DATASET_CANDIDATES = (
    AOE_ROOT / "remote_artifacts/aos_research_remote_shutdown_20260531/glm_ocr_portable_bundle/data/OmniDocBench",
    Path("/home/lukaiv/datasets/OmniDocBench_current"),
    Path("/home/lukaiv/datasets/OmniDocBench"),
    Path("/home/lukaiv/data/OmniDocBench_current"),
    Path("/home/lukaiv/data/OmniDocBench"),
    Path("/workspace/data/OmniDocBench"),
)


@dataclass(frozen=True)
class PageInput:
    idx: int
    dataset_index: int
    image_path: Path
    image_rel: str
    page_info: dict[str, Any]
    gt_layout_dets: list[dict[str, Any]]


@dataclass(frozen=True)
class LayoutCrop:
    entry: dict[str, Any]
    image: Image.Image


@dataclass(frozen=True)
class PrefillInput:
    entry: dict[str, Any]
    prompt: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    timing_s: dict[str, float]


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "sum": float(arr.sum()),
        "avg": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(clean_json(value), ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_dataset_dir(path: str | Path | None) -> Path:
    if path:
        candidate = Path(path).expanduser()
        if candidate.exists():
            return candidate.resolve()
        repo_candidate = REPO_ROOT / candidate
        if repo_candidate.exists():
            return repo_candidate.resolve()
        return candidate.resolve()
    for candidate in DEFAULT_DATASET_CANDIDATES:
        if (candidate / "OmniDocBench.json").is_file() and (candidate / "images").is_dir():
            return candidate.resolve()
    return DEFAULT_DATASET_CANDIDATES[0].resolve()


def u_escape_path_component(value: str, *, uppercase_hex: bool = False) -> str:
    escaped: list[str] = []
    for char in str(value):
        code = ord(char)
        if code < 128:
            escaped.append(char)
            continue
        fmt = "04X" if uppercase_hex else "04x"
        escaped.append(f"#U{code:{fmt}}")
    return "".join(escaped)


def u_escape_relative_path(rel: str, *, uppercase_hex: bool = False) -> Path:
    path = Path(str(rel))
    return Path(*(u_escape_path_component(part, uppercase_hex=uppercase_hex) for part in path.parts))


def resolve_page_image_path(images_dir: Path, rel: str) -> tuple[Path | None, str, list[Path]]:
    candidates = [
        images_dir / rel,
        images_dir / u_escape_relative_path(rel, uppercase_hex=False),
        images_dir / u_escape_relative_path(rel, uppercase_hex=True),
    ]
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    for idx, candidate in enumerate(unique):
        try:
            if candidate.is_file():
                return candidate.resolve(), "json_path" if idx == 0 else "u_escape_fallback", unique
        except OSError:
            continue
    return None, "missing", unique


def load_pages(dataset_dir: Path, *, page_start: int, num_pages: int) -> tuple[list[PageInput], dict[str, Any]]:
    dataset_dir = resolve_dataset_dir(dataset_dir)
    json_path = dataset_dir / "OmniDocBench.json"
    images_dir = dataset_dir / "images"
    if not json_path.exists():
        raise FileNotFoundError(f"OmniDocBench.json not found: {json_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"OmniDocBench images directory not found: {images_dir}")
    dataset = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(dataset, list):
        raise ValueError(f"expected OmniDocBench JSON list, got {type(dataset).__name__}")
    start = int(page_start)
    end = start + int(num_pages)
    if start < 0 or end > len(dataset) or end <= start:
        raise ValueError(f"invalid page slice start={start} end={end} dataset_len={len(dataset)}")
    pages: list[PageInput] = []
    escaped_path_fallback_count = 0
    for dataset_index in range(start, end):
        record = dataset[dataset_index]
        page_info = dict(record.get("page_info", {}) or {})
        rel = str(page_info.get("image_path", ""))
        image_path, path_mode, candidates = resolve_page_image_path(images_dir, rel)
        if image_path is None:
            raise FileNotFoundError(
                f"page image not found for dataset index {dataset_index}: "
                + ", ".join(str(path) for path in candidates)
            )
        if path_mode == "u_escape_fallback":
            escaped_path_fallback_count += 1
        pages.append(
            PageInput(
                idx=len(pages),
                dataset_index=int(dataset_index),
                image_path=image_path,
                image_rel=rel,
                page_info=page_info,
                gt_layout_dets=clean_json(record.get("layout_dets", []) or []),
            )
        )
    return pages, {
        "dataset_dir": str(dataset_dir),
        "json_path": str(json_path),
        "json_sha256": sha256_file(json_path),
        "images_dir": str(images_dir),
        "dataset_len": int(len(dataset)),
        "page_start": int(start),
        "num_pages": int(num_pages),
        "loaded_page_count": int(len(pages)),
        "escaped_path_fallback_count": int(escaped_path_fallback_count),
        "missing_image_policy": "fatal",
    }


def gt_layout_box_sort_key(box: dict[str, Any]) -> tuple[int, int, int]:
    order = safe_int(box.get("gt_order"), default=10**9)
    has_order = 0 if box.get("gt_order") is not None else 1
    det_index = safe_int(box.get("gt_det_index"), default=10**9)
    return has_order, order, det_index


def safe_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def gt_text_for_det(det: dict[str, Any]) -> tuple[str, str]:
    for key in ("text", "latex", "html"):
        value = det.get(key)
        if isinstance(value, str) and value.strip():
            return value, key
    return "", ""


def prompt_for_label(label: str) -> str:
    return LAYOUT_PROMPT_BY_LABEL.get(str(label), "OCR:")


def clamp_box_xyxy(coordinate: Any, width: int, height: int, pad: int) -> tuple[int, int, int, int] | None:
    if not isinstance(coordinate, (list, tuple)) or len(coordinate) < 4:
        return None
    values = [float(value) for value in coordinate]
    if len(values) == 4:
        xs = [values[0], values[2]]
        ys = [values[1], values[3]]
    else:
        xs = values[0::2]
        ys = values[1::2]
    left = max(0, int(math.floor(min(xs))) - int(pad))
    top = max(0, int(math.floor(min(ys))) - int(pad))
    right = min(int(width), int(math.ceil(max(xs))) + int(pad))
    bottom = min(int(height), int(math.ceil(max(ys))) + int(pad))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def build_gt_layout_pages(
    pages: list[PageInput],
    *,
    include_ignored: bool,
    include_empty_gt: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    start = time.perf_counter()
    rows: list[dict[str, Any]] = []
    raw_count = 0
    skipped_ignored = 0
    skipped_empty = 0
    for page in pages:
        boxes: list[dict[str, Any]] = []
        for det_idx, det in enumerate(page.gt_layout_dets):
            raw_count += 1
            poly = det.get("poly")
            if not isinstance(poly, list) or len(poly) < 8:
                continue
            gt_text, gt_source = gt_text_for_det(det)
            ignored = bool(det.get("ignore", False))
            if ignored and not include_ignored:
                skipped_ignored += 1
                continue
            if not gt_text and not include_empty_gt:
                skipped_empty += 1
                continue
            boxes.append(
                {
                    "label": str(det.get("category_type", "unknown")),
                    "score": 1.0,
                    "coordinate": clean_json(poly),
                    "gt_det_index": int(det_idx),
                    "gt_order": det.get("order"),
                    "gt_anno_id": det.get("anno_id"),
                    "gt_ignore": ignored,
                    "gt_text_source": gt_source,
                    "ground_truth": gt_text,
                    "ground_truth_source": gt_source,
                }
            )
        boxes.sort(key=gt_layout_box_sort_key)
        rows.append(
            {
                "selected_page_idx": int(page.idx),
                "dataset_index": int(page.dataset_index),
                "image_rel": page.image_rel,
                "boxes": boxes,
            }
        )
    return rows, {
        "gt_layout_build_s": float(time.perf_counter() - start),
        "gt_layout_raw_box_count": int(raw_count),
        "gt_layout_skipped_ignored_count": int(skipped_ignored),
        "gt_layout_skipped_empty_gt_count": int(skipped_empty),
        "uses_ground_truth_boxes": True,
    }


def build_crops(
    *,
    pages: list[PageInput],
    layout_pages: list[dict[str, Any]],
    crop_padding: int,
    min_crop_side: int,
    skip_labels: str,
) -> tuple[list[LayoutCrop], dict[str, Any]]:
    start = time.perf_counter()
    crops: list[LayoutCrop] = []
    skipped: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    prompt_counts: Counter[str] = Counter()
    per_page_counts: list[dict[str, Any]] = []
    page_by_dataset_idx = {int(page.dataset_index): page for page in pages}
    page_by_selected_idx = {int(page.idx): page for page in pages}
    skip_set = {label.strip() for label in str(skip_labels or "").split(",") if label.strip()}
    for page_result in layout_pages:
        selected_idx = int(page_result.get("selected_page_idx", page_result.get("page_index", 0)) or 0)
        dataset_idx = int(page_result.get("dataset_index", selected_idx))
        page = page_by_dataset_idx.get(dataset_idx) or page_by_selected_idx.get(selected_idx)
        if page is None:
            raise ValueError(f"layout page result does not match selected pages: {page_result.keys()}")
        boxes = page_result.get("boxes", [])
        with Image.open(page.image_path).convert("RGB") as image:
            width, height = image.size
            kept = 0
            for box_idx, box in enumerate(boxes):
                label = str(box.get("label", "unknown"))
                label_counts[label] += 1
                if label in skip_set:
                    skipped.append({"page": int(page.idx), "box": int(box_idx), "label": label, "reason": "skip_label"})
                    continue
                bbox = clamp_box_xyxy(box.get("coordinate"), width, height, int(crop_padding))
                if bbox is None:
                    skipped.append({"page": int(page.idx), "box": int(box_idx), "label": label, "reason": "invalid_coordinate"})
                    continue
                crop_w = int(bbox[2] - bbox[0])
                crop_h = int(bbox[3] - bbox[1])
                if crop_w < int(min_crop_side) or crop_h < int(min_crop_side):
                    skipped.append(
                        {
                            "page": int(page.idx),
                            "box": int(box_idx),
                            "label": label,
                            "reason": "too_small",
                            "crop_size": [crop_w, crop_h],
                        }
                    )
                    continue
                prompt = prompt_for_label(label)
                prompt_counts[prompt] += 1
                source_box_idx = safe_int(box.get("gt_det_index"), default=box_idx)
                crop_id = f"page{page.idx:04d}_box{source_box_idx:04d}_{label}"
                entry = {
                    "id": crop_id,
                    "source_image": str(page.image_path),
                    "image_rel": page.image_rel,
                    "page_index": int(page.idx),
                    "dataset_index": int(page.dataset_index),
                    "layout_box_index": int(box_idx),
                    "layout_box_source_index": int(source_box_idx),
                    "layout_gt_order": clean_json(box.get("gt_order")),
                    "layout_gt_anno_id": clean_json(box.get("gt_anno_id")),
                    "category_type": label,
                    "layout_label": label,
                    "bbox_xyxy": list(bbox),
                    "crop_size": [crop_w, crop_h],
                    "suggested_prompt": prompt,
                    "ground_truth": str(box.get("ground_truth", "") or ""),
                    "ground_truth_source": str(box.get("ground_truth_source", "") or ""),
                }
                crops.append(LayoutCrop(entry=entry, image=image.crop(bbox).copy()))
                kept += 1
            per_page_counts.append(
                {
                    "selected_page_idx": int(page.idx),
                    "dataset_index": int(page.dataset_index),
                    "image_rel": page.image_rel,
                    "layout_box_count": int(len(boxes)),
                    "recognizer_crop_count": int(kept),
                }
            )
    return crops, {
        "crop_extract_s": float(time.perf_counter() - start),
        "layout_box_count": int(sum(row["layout_box_count"] for row in per_page_counts)),
        "recognizer_crop_count": int(len(crops)),
        "skipped_count": int(len(skipped)),
        "skipped_samples": skipped[:16],
        "label_counts": dict(sorted(label_counts.items())),
        "prompt_counts": dict(sorted(prompt_counts.items())),
        "per_page_counts": per_page_counts,
        "crop_padding": int(crop_padding),
        "min_crop_side": int(min_crop_side),
        "skip_labels": sorted(skip_set),
    }


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    if height < factor:
        width = round((width * factor) / height)
        height = factor
    if width < factor:
        height = round((height * factor) / width)
        width = factor
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return int(h_bar), int(w_bar)


def load_preprocessor_config(model_dir: Path) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "do_convert_rgb": True,
        "do_normalize": True,
        "do_rescale": True,
        "do_resize": True,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "max_pixels": 1003520,
        "merge_size": 2,
        "min_pixels": 112896,
        "patch_size": 14,
        "resample": 3,
        "rescale_factor": 1.0 / 255.0,
        "temporal_patch_size": 1,
    }
    path = model_dir / "preprocessor_config.json"
    if path.exists():
        defaults.update(json.loads(path.read_text(encoding="utf-8")))
    return defaults


def preprocess_pil_crop(image: Image.Image, cfg: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    timing: dict[str, float] = {"image_read_s": 0.0}
    start = time.perf_counter()
    if cfg["do_convert_rgb"]:
        image = image.convert("RGB")
    width, height = image.size
    patch_size = int(cfg["patch_size"])
    merge_size = int(cfg["merge_size"])
    temporal_patch_size = int(cfg["temporal_patch_size"])
    if temporal_patch_size != 1:
        raise ValueError(f"temporal_patch_size must be 1 for this recognizer path, got {temporal_patch_size}")
    resized_height, resized_width = height, width
    if cfg["do_resize"]:
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=patch_size * merge_size,
            min_pixels=int(cfg["min_pixels"]),
            max_pixels=int(cfg["max_pixels"]),
        )
        image = image.resize((resized_width, resized_height), Image.Resampling(int(cfg["resample"])))
    array = np.asarray(image)
    if cfg["do_rescale"]:
        array = array.astype(np.float32) * float(cfg["rescale_factor"])
    else:
        array = array.astype(np.float32)
    if cfg["do_normalize"]:
        mean = np.array(cfg["image_mean"], dtype=np.float32)
        std = np.array(cfg["image_std"], dtype=np.float32)
        array = (array - mean) / std
    patches = array.transpose(2, 0, 1)[None, ...]
    channel = patches.shape[1]
    grid_t = patches.shape[0] // temporal_patch_size
    grid_h = resized_height // patch_size
    grid_w = resized_width // patch_size
    patches = patches.reshape(grid_t, temporal_patch_size, channel, grid_h, patch_size, grid_w, patch_size)
    patches = patches.transpose(0, 3, 5, 2, 1, 4, 6)
    flatten_patches = patches.reshape(grid_t * grid_h * grid_w, channel, patch_size, patch_size)
    timing["preprocess_cpu_s"] = float(time.perf_counter() - start)
    return torch.from_numpy(flatten_patches), torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.long), timing


def build_paddleocr_vl_prompt(prompt: str, *, image_token_count: int) -> str:
    template = f"{BOS}User: {IMAGE_START}{IMAGE_TOKEN}{IMAGE_END}{prompt}\nAssistant:\n"
    placeholder = "<|placeholder|>"
    return template.replace(IMAGE_TOKEN, placeholder * int(image_token_count), 1).replace(placeholder, IMAGE_TOKEN)


def build_inputs(
    tokenizer: Tokenizer,
    image_grid_thw: torch.Tensor,
    prompt: str,
    merge_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    image_token_count = int(image_grid_thw[0].prod().item()) // int(merge_size) // int(merge_size)
    text = build_paddleocr_vl_prompt(prompt, image_token_count=image_token_count)
    ids = tokenizer.encode(text).ids
    input_ids = torch.tensor([ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def build_prefill_inputs(
    *,
    crops: list[LayoutCrop],
    tokenizer: Tokenizer,
    pre_cfg: dict[str, Any],
    prompt_override: str | None,
) -> tuple[list[PrefillInput], dict[str, Any]]:
    inputs: list[PrefillInput] = []
    total_start = time.perf_counter()
    timing_rows: list[dict[str, float]] = []
    for crop in crops:
        prompt = str(prompt_override if prompt_override is not None else crop.entry.get("suggested_prompt", "OCR:"))
        pixel_values, image_grid_thw, timing = preprocess_pil_crop(crop.image, pre_cfg)
        token_start = time.perf_counter()
        input_ids, attention_mask = build_inputs(tokenizer, image_grid_thw, prompt, merge_size=int(pre_cfg["merge_size"]))
        timing["token_build_s"] = float(time.perf_counter() - token_start)
        timing["input_build_s"] = timing["image_read_s"] + timing["preprocess_cpu_s"] + timing["token_build_s"]
        timing_rows.append(timing)
        inputs.append(
            PrefillInput(
                entry=crop.entry,
                prompt=prompt,
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                timing_s=timing,
            )
        )
    keys = sorted({key for row in timing_rows for key in row})
    return inputs, {
        "input_build_wall_s": float(time.perf_counter() - total_start),
        **{key: stats([float(row[key]) for row in timing_rows if key in row]) for key in keys},
    }


def parse_dtype(name: str) -> torch.dtype:
    normalized = str(name).lower()
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def npu_is_available() -> bool:
    try:
        import torch_npu  # noqa: F401
    except ModuleNotFoundError:
        return False
    except Exception as exc:
        raise RuntimeError(f"torch_npu is installed but failed to initialize: {exc.__class__.__name__}: {exc}") from exc
    return hasattr(torch, "npu") and torch.npu.is_available()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if npu_is_available():
            return torch.device("npu:0")
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name.startswith("npu"):
        try:
            import torch_npu  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("NPU device requested, but torch_npu is not importable in this environment.") from exc
    return torch.device(name)


def configure_npu_jit_compile(mode: str, device: torch.device, *, verbose: bool = True) -> None:
    if mode not in NPU_JIT_COMPILE_CHOICES:
        raise ValueError(f"unsupported npu_jit_compile={mode!r}")
    if mode == "default" or device.type != "npu":
        return
    try:
        import torch_npu  # noqa: F401

        requested = mode == "on"
        torch.npu.set_compile_mode(jit_compile=requested)
        if verbose:
            print(f"[npu] set torch.npu compile mode: jit_compile={requested}", file=sys.stderr, flush=True)
    except Exception as exc:
        raise RuntimeError(f"failed to set NPU jit_compile={mode}: {exc.__class__.__name__}: {exc}") from exc


def maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "npu":
        import torch_npu

        torch_npu.npu.synchronize()


def build_vision_cu_seqlens(image_grid_thw: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    grid = image_grid_thw.to(device)
    cu_seqlens = torch.repeat_interleave(grid[:, 1] * grid[:, 2], grid[:, 0]).cumsum(dim=0, dtype=torch.int32)
    return torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)


def build_single_crop_vision_cu_seqlens(image_grid_thw: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    grid = image_grid_thw.detach().cpu().reshape(-1, 3)
    lengths: list[int] = []
    for t, h, w in grid.tolist():
        lengths.extend([int(h) * int(w)] * int(t))
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + int(length))
    return torch.tensor(cu, device=device, dtype=torch.int32)


def single_crop_grid_ints(image_grid_thw: torch.Tensor) -> tuple[int, int, int]:
    grid = image_grid_thw.detach().cpu().reshape(-1, 3)
    if int(grid.shape[0]) != 1:
        raise ValueError(f"static single-crop vision expects exactly one image grid, got {tuple(grid.shape)}")
    t, h, w = grid[0].tolist()
    return int(t), int(h), int(w)


def build_static_abs_pos_embed(
    model: LocalPaddleOCRVLForConditionalGeneration,
    image_grid_thw: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    t, h, w = single_crop_grid_ints(image_grid_thw)
    embeddings_module = model.visual.vision_model.embeddings
    dtype = embeddings_module.patch_embedding.weight.dtype
    dummy = torch.empty((int(t) * int(h) * int(w), embeddings_module.embed_dim), device=device, dtype=dtype)
    with torch.inference_mode():
        pos = embeddings_module.interpolate_pos_encoding(dummy, int(h), int(w)).squeeze(0).repeat(int(t), 1)
    return pos.contiguous()


def build_static_vision_rope(
    model: LocalPaddleOCRVLForConditionalGeneration,
    image_grid_thw: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    _t, h, w = single_crop_grid_ints(image_grid_thw)
    encoder = model.visual.vision_model.encoder
    image_pids = torch.arange(int(image_grid_thw.prod().item()), device=device, dtype=torch.int64) % int(h * w)
    pids = torch.stack((image_pids // int(w), image_pids % int(w)), dim=-1)
    rotary_max = encoder.rotary_pos_emb(max(int(h), int(w)))
    rotary_embeddings = rotary_max[pids].flatten(1).repeat(1, 2)
    return rotary_embeddings.cos().contiguous(), rotary_embeddings.sin().contiguous()


def build_static_pad_attention_mask(real_seq_len: int, pad_tokens: int, *, device: torch.device) -> torch.Tensor:
    if int(pad_tokens) <= 0:
        return torch.zeros((1, 1, 1, 1), device=device, dtype=torch.bool)
    physical_seq_len = int(real_seq_len) + int(pad_tokens)
    mask = torch.zeros((1, 1, physical_seq_len, physical_seq_len), device=device, dtype=torch.bool)
    real = int(real_seq_len)
    mask[..., :real, real:physical_seq_len] = True
    mask[..., real:physical_seq_len, :real] = True
    return mask.contiguous()


def static_visual_pad_tokens(real_seq_len: int, mode: str) -> int:
    if mode == "none":
        return 0
    if mode == "mask_pad_one":
        return 1 if int(real_seq_len) % 16 == 0 else 0
    if mode != "mask_pad_to_128":
        raise ValueError(f"unsupported static visual pad mode: {mode!r}; choices={STATIC_VISUAL_PAD_MODE_CHOICES}")
    real_seq_len = int(real_seq_len)
    if real_seq_len % 16 != 0:
        return 0
    min_physical_seq_len = real_seq_len + 1
    if min_physical_seq_len <= 128:
        return 1
    physical_seq_len = ((min_physical_seq_len + 127) // 128) * 128
    return int(physical_seq_len - real_seq_len)


class SingleCropStaticVisualModule(torch.nn.Module):
    """Shape-specialized visual encoder wrapper for fullgraph static compilation."""

    def __init__(
        self,
        model: LocalPaddleOCRVLForConditionalGeneration,
        image_grid_thw: torch.Tensor,
        *,
        device: torch.device,
        static_visual_pad_mode: str,
    ):
        super().__init__()
        self.model = model
        self.static_visual_pad_mode = str(static_visual_pad_mode)
        if self.static_visual_pad_mode not in STATIC_VISUAL_PAD_MODE_CHOICES:
            raise ValueError(
                f"unsupported static_visual_pad_mode={self.static_visual_pad_mode!r}; "
                f"choices={STATIC_VISUAL_PAD_MODE_CHOICES}"
            )
        self.register_buffer("image_grid_thw_const", image_grid_thw.detach().clone(), persistent=False)
        self.register_buffer(
            "cu_seqlens_const",
            build_single_crop_vision_cu_seqlens(image_grid_thw, device=device),
            persistent=False,
        )
        self.static_real_seq_len = int(image_grid_thw.prod().item())
        self.static_pad_tokens = 0
        if self.static_visual_pad_mode != "none":
            grid_t, _grid_h, _grid_w = single_crop_grid_ints(image_grid_thw)
            if int(grid_t) != 1:
                raise ValueError(
                    f"static_visual {self.static_visual_pad_mode} currently supports single-image crop grids "
                    "with T=1 only"
                )
            self.static_pad_tokens = static_visual_pad_tokens(self.static_real_seq_len, self.static_visual_pad_mode)
        self.static_physical_seq_len = self.static_real_seq_len + self.static_pad_tokens

        abs_pos_embed = build_static_abs_pos_embed(model, image_grid_thw, device=device)
        rope_cos, rope_sin = build_static_vision_rope(model, image_grid_thw, device=device)
        if self.static_pad_tokens:
            abs_pos_embed = torch.cat(
                [
                    abs_pos_embed,
                    torch.zeros(
                        self.static_pad_tokens,
                        abs_pos_embed.shape[-1],
                        device=device,
                        dtype=abs_pos_embed.dtype,
                    ),
                ],
                dim=0,
            ).contiguous()
            rope_cos = torch.cat(
                [
                    rope_cos,
                    torch.ones(self.static_pad_tokens, rope_cos.shape[-1], device=device, dtype=rope_cos.dtype),
                ],
                dim=0,
            ).contiguous()
            rope_sin = torch.cat(
                [
                    rope_sin,
                    torch.zeros(self.static_pad_tokens, rope_sin.shape[-1], device=device, dtype=rope_sin.dtype),
                ],
                dim=0,
            ).contiguous()
        self.register_buffer("abs_pos_embed_const", abs_pos_embed, persistent=False)
        self.register_buffer("vision_rope_cos_const", rope_cos, persistent=False)
        self.register_buffer("vision_rope_sin_const", rope_sin, persistent=False)
        self.register_buffer(
            "static_pad_attention_mask",
            build_static_pad_attention_mask(self.static_real_seq_len, self.static_pad_tokens, device=device),
            persistent=False,
        )

    def _zero_static_pad_rows(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.static_pad_tokens <= 0:
            return hidden_states
        return torch.cat(
            [
                hidden_states[: self.static_real_seq_len],
                torch.zeros_like(hidden_states[self.static_real_seq_len : self.static_physical_seq_len]),
            ],
            dim=0,
        )

    def _static_mask_padded_attention(self, attention: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states = attention.q_proj(hidden_states).view(seq_length, attention.num_heads, attention.head_dim)
        key_states = attention.k_proj(hidden_states).view(seq_length, attention.num_heads, attention.head_dim)
        value_states = attention.v_proj(hidden_states).view(seq_length, attention.num_heads, attention.head_dim)
        query_states, key_states = apply_rotary_pos_emb_vision(
            query_states,
            key_states,
            self.vision_rope_cos_const,
            self.vision_rope_sin_const,
        )
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)
        attention_impl = get_vision_attention_impl()
        if attention_impl == "prompt_flash_attention":
            if get_vision_prompt_fa_layout() != "bnsd":
                raise ValueError(
                    f"static_visual {self.static_visual_pad_mode} currently supports PromptFA layout bnsd only"
                )
            attn_output = vision_prompt_flash_attention_bnsd(
                query_states,
                key_states,
                value_states,
                num_heads=int(attention.num_heads),
                scale=float(attention.scaling),
                atten_mask=self.static_pad_attention_mask,
            )
        elif attention_impl == "manual":
            scores = torch.matmul(query_states, key_states.transpose(2, 3)) * attention.scaling
            scores = scores.masked_fill(self.static_pad_attention_mask, torch.finfo(scores.dtype).min)
            probs = attention_softmax(
                scores,
                dim=-1,
                output_dtype=query_states.dtype,
                mode=get_vision_softmax_dtype_mode(),
            )
            attn_output = torch.matmul(probs, value_states)
        else:
            raise ValueError(f"unknown vision attention implementation: {attention_impl!r}")
        attn_output = attn_output.transpose(1, 2).contiguous().view(seq_length, -1)
        return attention.out_proj(attn_output)

    def _static_mask_padded_encoder_layer(self, encoder_layer: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        attn_input = encoder_layer.layer_norm1(hidden_states)
        hidden_states = hidden_states + self._static_mask_padded_attention(encoder_layer.self_attn, attn_input)
        hidden_states = self._zero_static_pad_rows(hidden_states)
        hidden_states = hidden_states + encoder_layer.mlp(encoder_layer.layer_norm2(hidden_states))
        return self._zero_static_pad_rows(hidden_states)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        transformer = self.model.visual.vision_model
        embeddings_module = transformer.embeddings
        pixel_values = pixel_values.to(dtype=embeddings_module.patch_embedding.weight.dtype)
        patch_embeds = embeddings_module.patch_embedding(pixel_values)
        hidden_states = patch_embeds.flatten(-2).squeeze(-1)
        if self.static_pad_tokens:
            hidden_states = torch.cat(
                [
                    hidden_states,
                    torch.zeros(
                        self.static_pad_tokens,
                        hidden_states.shape[-1],
                        device=hidden_states.device,
                        dtype=hidden_states.dtype,
                    ),
                ],
                dim=0,
            )
        hidden_states = hidden_states + self.abs_pos_embed_const
        position_embeddings = (self.vision_rope_cos_const, self.vision_rope_sin_const)
        for encoder_layer in transformer.encoder.layers:
            if self.static_pad_tokens:
                hidden_states = self._static_mask_padded_encoder_layer(encoder_layer, hidden_states)
            else:
                hidden_states = encoder_layer(hidden_states, self.cu_seqlens_const, position_embeddings)
        hidden_states = transformer.post_layernorm(hidden_states)
        return hidden_states[: self.static_real_seq_len]


def import_torchair():
    try:
        from torch_npu.dynamo import torchair
        from torch_npu.dynamo.torchair.configs.compiler_config import CompilerConfig

        return torchair, CompilerConfig
    except Exception:
        import torchair
        from torchair.configs.compiler_config import CompilerConfig

        return torchair, CompilerConfig


def vision_compile_backend(name: str, device: torch.device):
    if name == "default":
        return None
    if name == "torchair":
        if device.type != "npu":
            raise ValueError("--vision-compile-backend torchair requires --device npu:0")
        torchair, CompilerConfig = import_torchair()
        config = CompilerConfig()
        return torchair.get_npu_backend(compiler_config=config)
    return name


def maybe_compile_static_visual(
    *,
    model: LocalPaddleOCRVLForConditionalGeneration,
    item: PrefillInput,
    device: torch.device,
    backend_name: str,
    static_visual_pad_mode: str,
) -> tuple[Callable[[torch.Tensor], torch.Tensor] | None, dict[str, Any]]:
    if backend_name not in VISION_COMPILE_BACKEND_CHOICES:
        raise ValueError(f"unsupported vision compile backend={backend_name!r}; choices={VISION_COMPILE_BACKEND_CHOICES}")
    wrapper = SingleCropStaticVisualModule(
        model,
        item.image_grid_thw,
        device=device,
        static_visual_pad_mode=static_visual_pad_mode,
    ).eval()
    meta: dict[str, Any] = {
        "candidate_vision_path": "static_visual",
        "backend": str(backend_name),
        "static_visual_pad_mode": str(static_visual_pad_mode),
        "static_visual_pad_tokens": int(wrapper.static_pad_tokens),
        "static_visual_real_seq_len": int(wrapper.static_real_seq_len),
        "static_visual_real_seq_mod16": int(wrapper.static_real_seq_len % 16),
        "static_visual_physical_seq_len": int(wrapper.static_physical_seq_len),
        "static_visual_physical_seq_mod16": int(wrapper.static_physical_seq_len % 16),
        "static_visual_physical_seq_mod128": int(wrapper.static_physical_seq_len % 128),
        "fullgraph": bool(backend_name != "none"),
        "dynamic": False,
        "image_grid_thw": [int(value) for value in item.image_grid_thw.flatten().tolist()],
        "cu_seqlens": [int(value) for value in wrapper.cu_seqlens_const.detach().cpu().reshape(-1).tolist()],
        "static_abs_pos_embed_shape": [int(dim) for dim in wrapper.abs_pos_embed_const.shape],
        "static_vision_rope_shape": [int(dim) for dim in wrapper.vision_rope_cos_const.shape],
    }
    if backend_name == "none":
        meta.update({"enabled": False, "compile_api": None})
        return wrapper, meta

    import torch._dynamo

    old_capture_scalar_outputs = bool(torch._dynamo.config.capture_scalar_outputs)
    torch._dynamo.config.capture_scalar_outputs = True
    compile_kwargs: dict[str, Any] = {"fullgraph": True, "dynamic": False}
    backend = vision_compile_backend(backend_name, device)
    if backend is not None:
        compile_kwargs["backend"] = backend
    maybe_sync(device)
    start = time.perf_counter()
    compiled = torch.compile(wrapper, **compile_kwargs)
    maybe_sync(device)
    meta.update(
        {
            "enabled": True,
            "compile_api": "torch.compile",
            "compile_wrapper_s": float(time.perf_counter() - start),
            "capture_scalar_outputs": True,
            "capture_scalar_outputs_previous": old_capture_scalar_outputs,
        }
    )
    return compiled, meta


@torch.inference_mode()
def prepare_candidate_vision_forward(
    *,
    args: argparse.Namespace,
    model: LocalPaddleOCRVLForConditionalGeneration,
    item: PrefillInput,
    device: torch.device,
) -> tuple[Callable[[torch.Tensor], torch.Tensor] | None, dict[str, Any]]:
    if args.candidate_vision_path == "eager_visual":
        return None, {
            "candidate_vision_path": "eager_visual",
            "enabled": False,
            "backend": "none",
            "compile_api": None,
            "note": "Normal model.visual path with runtime cu_seqlens; not a compile-compatible static boundary.",
        }
    if args.candidate_vision_path != "static_visual":
        raise ValueError(
            f"unsupported candidate_vision_path={args.candidate_vision_path!r}; "
            f"choices={CANDIDATE_VISION_PATH_CHOICES}"
        )
    vision_forward, meta = maybe_compile_static_visual(
        model=model,
        item=item,
        device=device,
        backend_name=str(args.vision_compile_backend),
        static_visual_pad_mode=str(args.static_visual_pad_mode),
    )
    if vision_forward is None:
        raise RuntimeError("static_visual candidate did not produce a callable vision_forward")
    if str(args.vision_compile_backend) != "none":
        pixel_values = item.pixel_values.to(device=device, dtype=model.visual.dtype)
        maybe_sync(device)
        start = time.perf_counter()
        first_output = vision_forward(pixel_values)
        maybe_sync(device)
        meta["compiled_first_call_s"] = float(time.perf_counter() - start)
        meta["first_output_shape"] = [int(dim) for dim in first_output.shape]
        meta["first_output_nonfinite_count"] = int((~torch.isfinite(first_output.float())).sum().item())
    return vision_forward, meta


def scatter_projected_image_embeds(
    *,
    model: LocalPaddleOCRVLForConditionalGeneration,
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    image_embeds: torch.Tensor,
) -> torch.Tensor:
    projected = image_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
    positions = torch.nonzero(input_ids[0] == int(model.config.image_token_id), as_tuple=False).flatten()
    if int(positions.numel()) != int(projected.shape[0]):
        raise ValueError(
            "image features and image tokens do not match: "
            f"tokens={int(positions.numel())} features={int(projected.shape[0])}"
        )
    flat_embeds = inputs_embeds[0].clone()
    flat_embeds.index_copy_(0, positions.to(flat_embeds.device), projected)
    return flat_embeds.unsqueeze(0)


@torch.inference_mode()
def compute_vision_prefill(
    *,
    model: LocalPaddleOCRVLForConditionalGeneration,
    item: PrefillInput,
    device: torch.device,
    cache_length: int,
    sync: bool,
    record_phase_timings: bool = True,
    vision_forward: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    timing: dict[str, float] = {}

    def measure(name: str, fn):
        if sync:
            maybe_sync(device)
        start = time.perf_counter()
        value = fn()
        if sync:
            maybe_sync(device)
        if record_phase_timings:
            timing[name] = timing.get(name, 0.0) + float(time.perf_counter() - start)
        return value

    input_ids, attention_mask, pixel_values = measure(
        "device_transfer",
        lambda: (
            item.input_ids.to(device),
            item.attention_mask.to(device),
            item.pixel_values.to(device=device, dtype=model.visual.dtype),
        ),
    )
    image_grid_thw = item.image_grid_thw
    if int(input_ids.shape[1]) > int(cache_length):
        raise ValueError(
            f"input length {int(input_ids.shape[1])} exceeds cache_length={int(cache_length)} for item {item.entry.get('id')}"
        )
    if vision_forward is None:
        cu_seqlens = measure("vision_cu_seqlens", lambda: build_vision_cu_seqlens(image_grid_thw, device=device))
        visual_features = measure(
            "visual_features",
            lambda: model.visual(
                pixel_values=pixel_values.unsqueeze(0),
                image_grid_thw=image_grid_thw,
                cu_seqlens=cu_seqlens,
            ),
        )
    else:
        visual_features = measure("visual_features", lambda: vision_forward(pixel_values))
    image_embeds = measure("adaptive_mlp_projector", lambda: model.mlp_AR(visual_features, image_grid_thw))
    inputs_embeds = measure("text_token_embedding", lambda: model.model.embed_tokens(input_ids))
    inputs_embeds = measure(
        "image_embed_scatter",
        lambda: scatter_projected_image_embeds(
            model=model,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            image_embeds=image_embeds,
        ),
    )
    position_ids_cpu, _rope_deltas_cpu = measure(
        "mrope_index_cpu",
        lambda: model.get_rope_index(item.input_ids, item.image_grid_thw, item.attention_mask),
    )
    position_ids = measure("mrope_index_transfer", lambda: position_ids_cpu.to(device))
    cache = measure(
        "static_cache_alloc",
        lambda: model.allocate_static_cache(
            batch_size=int(inputs_embeds.shape[0]),
            cache_length=int(cache_length),
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
            init_mode="zeros",
        ),
    )
    hidden_states = measure(
        "text_prefill",
        lambda: model.model.forward_prefill_static(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
        ),
    )
    prefill_logits = measure("prefill_lm_head", lambda: model.lm_head(hidden_states[:, -1:, :]))
    if record_phase_timings:
        timing["phase_sync_sum_s"] = float(sum(timing.values()))
    return {
        "visual_features": visual_features.detach(),
        "image_embeds": image_embeds.detach(),
        "prefill_logits": prefill_logits.detach(),
        "prefill_hidden_last": hidden_states[:, -1:, :].detach(),
        "input_ids": input_ids.detach(),
        "attention_mask": attention_mask.detach(),
        "image_grid_thw": image_grid_thw.detach(),
        "prefill_argmax": torch.argmax(prefill_logits[:, -1, :].float(), dim=-1).detach(),
    }, timing


@torch.inference_mode()
def run_prefill_measurement(
    *,
    model: LocalPaddleOCRVLForConditionalGeneration,
    item: PrefillInput,
    device: torch.device,
    cache_length: int,
    timing_mode: str,
    vision_forward: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    if timing_mode not in TIMING_MODE_CHOICES:
        raise ValueError(f"unsupported timing_mode={timing_mode!r}; choices={TIMING_MODE_CHOICES}")
    if timing_mode == "phase_sync":
        return compute_vision_prefill(
            model=model,
            item=item,
            device=device,
            cache_length=cache_length,
            sync=True,
            record_phase_timings=True,
            vision_forward=vision_forward,
        )
    if timing_mode == "e2e":
        maybe_sync(device)
        start = time.perf_counter()
        tensors, _timing = compute_vision_prefill(
            model=model,
            item=item,
            device=device,
            cache_length=cache_length,
            sync=False,
            record_phase_timings=False,
            vision_forward=vision_forward,
        )
        maybe_sync(device)
        return tensors, {"e2e_wall_s": float(time.perf_counter() - start)}

    phase_tensors, phase_timing = compute_vision_prefill(
        model=model,
        item=item,
        device=device,
        cache_length=cache_length,
        sync=True,
        record_phase_timings=True,
        vision_forward=vision_forward,
    )
    maybe_sync(device)
    start = time.perf_counter()
    e2e_tensors, _timing = compute_vision_prefill(
        model=model,
        item=item,
        device=device,
        cache_length=cache_length,
        sync=False,
        record_phase_timings=False,
        vision_forward=vision_forward,
    )
    maybe_sync(device)
    phase_timing["e2e_wall_s"] = float(time.perf_counter() - start)
    phase_timing["phase_vs_e2e_prefill_logits_max_abs_diff"] = float(
        torch.max(torch.abs(phase_tensors["prefill_logits"].float() - e2e_tensors["prefill_logits"].float())).item()
    )
    return e2e_tensors, phase_timing


def select_stratified_inputs(inputs: list[PrefillInput], *, count: int, bucket_count: int) -> tuple[list[PrefillInput], dict[str, Any]]:
    if int(count) <= 0 or int(count) >= len(inputs):
        selected = list(inputs)
        return selected, {"strategy": "all", "selected_count": int(len(selected)), "available_count": int(len(inputs))}
    sorted_inputs = sorted(inputs, key=lambda item: (vision_tokens(item), str(item.entry.get("layout_label", "")), str(item.entry.get("id", ""))))
    bucket_count = max(1, min(int(bucket_count), int(count), len(sorted_inputs)))
    buckets = np.array_split(np.arange(len(sorted_inputs)), bucket_count)
    per_bucket = int(math.ceil(int(count) / bucket_count))
    selected: list[PrefillInput] = []
    selected_ids: set[str] = set()
    bucket_summaries: list[dict[str, Any]] = []
    for bucket_idx, indices in enumerate(buckets):
        bucket_items = [sorted_inputs[int(idx)] for idx in indices.tolist()]
        by_label: dict[str, list[PrefillInput]] = defaultdict(list)
        for item in bucket_items:
            by_label[str(item.entry.get("layout_label", ""))].append(item)
        for label_items in by_label.values():
            label_items.sort(key=lambda item: (vision_tokens(item), str(item.entry.get("id", ""))))
        labels = sorted(by_label, key=lambda label: (-len(by_label[label]), label))
        added = 0
        cursor = 0
        while added < per_bucket and labels:
            progressed = False
            for label in labels:
                label_items = by_label[label]
                if cursor >= len(label_items):
                    continue
                item = label_items[cursor]
                item_id = str(item.entry.get("id"))
                if item_id not in selected_ids:
                    selected.append(item)
                    selected_ids.add(item_id)
                    added += 1
                    progressed = True
                    if added >= per_bucket:
                        break
            cursor += 1
            if not progressed:
                break
        bucket_summaries.append(
            {
                "bucket_idx": int(bucket_idx),
                "available_count": int(len(bucket_items)),
                "selected_count": int(added),
                "vision_tokens": stats([float(vision_tokens(item)) for item in bucket_items]),
                "labels": dict(sorted(Counter(str(item.entry.get("layout_label", "")) for item in bucket_items).items())),
            }
        )
    if len(selected) < int(count):
        for item in sorted_inputs:
            item_id = str(item.entry.get("id"))
            if item_id in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(item_id)
            if len(selected) >= int(count):
                break
    selected = selected[: int(count)]
    selected.sort(key=lambda item: (vision_tokens(item), str(item.entry.get("id", ""))))
    return selected, {
        "strategy": "token_stratified_label_round_robin",
        "available_count": int(len(inputs)),
        "requested_count": int(count),
        "selected_count": int(len(selected)),
        "bucket_count": int(bucket_count),
        "per_bucket_target": int(per_bucket),
        "buckets": bucket_summaries,
    }


def vision_tokens(item: PrefillInput) -> int:
    return int(item.image_grid_thw.prod().item())


def projected_tokens(item: PrefillInput, *, merge_size: int) -> int:
    return int(vision_tokens(item) // int(merge_size) // int(merge_size))


def input_row(item: PrefillInput, *, merge_size: int) -> dict[str, Any]:
    return {
        "id": str(item.entry.get("id")),
        "source_image": str(item.entry.get("source_image", "")),
        "image_rel": str(item.entry.get("image_rel", "")),
        "page_index": int(item.entry.get("page_index", 0)),
        "dataset_index": int(item.entry.get("dataset_index", 0)),
        "layout_label": str(item.entry.get("layout_label", "")),
        "crop_size": clean_json(item.entry.get("crop_size", [0, 0])),
        "bbox_xyxy": clean_json(item.entry.get("bbox_xyxy", [])),
        "prompt": str(item.prompt),
        "ground_truth_source": str(item.entry.get("ground_truth_source", "")),
        "ground_truth_sample": str(item.entry.get("ground_truth", ""))[:240],
        "image_grid_thw": [int(value) for value in item.image_grid_thw.flatten().tolist()],
        "vision_tokens": int(vision_tokens(item)),
        "projected_image_tokens": int(projected_tokens(item, merge_size=merge_size)),
        "input_tokens": int(item.input_ids.shape[1]),
    }


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    detached = tensor.detach()
    return {
        "shape": [int(dim) for dim in detached.shape],
        "dtype": str(detached.dtype),
        "numel": int(detached.numel()),
        "nonfinite_count": int((~torch.isfinite(detached.float())).sum().item()),
    }


def topk_summary(logits: torch.Tensor, *, k: int = 8) -> dict[str, Any]:
    values, indices = torch.topk(logits[:, -1, :].float(), k=min(int(k), int(logits.shape[-1])), dim=-1)
    return {
        "topk_indices": [int(value) for value in indices[0].detach().cpu().tolist()],
        "topk_values": [float(value) for value in values[0].detach().cpu().tolist()],
        "argmax": int(indices[0, 0].item()),
    }


def diff_stats(lhs: torch.Tensor, rhs: torch.Tensor) -> dict[str, Any]:
    if tuple(lhs.shape) != tuple(rhs.shape):
        return {
            "shape_match": False,
            "lhs_shape": [int(dim) for dim in lhs.shape],
            "rhs_shape": [int(dim) for dim in rhs.shape],
        }
    lhs_f = lhs.detach().float()
    rhs_f = rhs.detach().float()
    finite = torch.isfinite(lhs_f) & torch.isfinite(rhs_f)
    diff = torch.abs(lhs_f - rhs_f)
    finite_diff = diff[finite]
    if finite_diff.numel() == 0:
        max_abs = mean_abs = rms_abs = p99_abs = p999_abs = None
    else:
        max_abs = float(finite_diff.max().item())
        mean_abs = float(finite_diff.mean().item())
        rms_abs = float(torch.sqrt(torch.mean(finite_diff * finite_diff)).item())
        p99_abs = float(torch.quantile(finite_diff, 0.99).item())
        p999_abs = float(torch.quantile(finite_diff, 0.999).item())
    return {
        "shape_match": True,
        "shape": [int(dim) for dim in lhs.shape],
        "numel": int(lhs.numel()),
        "lhs_nonfinite_count": int((~torch.isfinite(lhs_f)).sum().item()),
        "rhs_nonfinite_count": int((~torch.isfinite(rhs_f)).sum().item()),
        "diff_nonfinite_count": int((~torch.isfinite(diff)).sum().item()),
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "rms_abs_diff": rms_abs,
        "p99_abs_diff": p99_abs,
        "p999_abs_diff": p999_abs,
        "allclose_atol_5e_2_rtol_5e_2": bool(torch.allclose(lhs_f, rhs_f, atol=5e-2, rtol=5e-2)),
        "allclose_atol_1e_1_rtol_1e_1": bool(torch.allclose(lhs_f, rhs_f, atol=1e-1, rtol=1e-1)),
        "allclose_atol_1e_0_rtol_1e_0": bool(torch.allclose(lhs_f, rhs_f, atol=1.0, rtol=1.0)),
    }


def aggregate_diff(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [row["diffs"][key] for row in rows if row.get("diffs", {}).get(key, {}).get("shape_match", False)]
    return {
        "count": int(len(values)),
        "max_abs_diff": stats([float(row["max_abs_diff"]) for row in values if row.get("max_abs_diff") is not None]),
        "mean_abs_diff": stats([float(row["mean_abs_diff"]) for row in values if row.get("mean_abs_diff") is not None]),
        "allclose_5e_2_pass_count": int(sum(bool(row.get("allclose_atol_5e_2_rtol_5e_2")) for row in values)),
        "allclose_1e_1_pass_count": int(sum(bool(row.get("allclose_atol_1e_1_rtol_1e_1")) for row in values)),
        "allclose_1e_0_pass_count": int(sum(bool(row.get("allclose_atol_1e_0_rtol_1e_0")) for row in values)),
    }


def build_inputs_for_args(args: argparse.Namespace, *, model_dir: Path, tokenizer: Tokenizer) -> tuple[list[PrefillInput], dict[str, Any]]:
    pages, page_summary = load_pages(resolve_dataset_dir(args.dataset_dir), page_start=int(args.page_start), num_pages=int(args.num_pages))
    layout_pages, layout_summary = build_gt_layout_pages(
        pages,
        include_ignored=bool(args.include_ignored_gt),
        include_empty_gt=bool(args.include_empty_gt),
    )
    crops, crop_summary = build_crops(
        pages=pages,
        layout_pages=layout_pages,
        crop_padding=int(args.crop_padding),
        min_crop_side=int(args.min_crop_side),
        skip_labels=str(args.skip_labels or ""),
    )
    pre_cfg = load_preprocessor_config(model_dir)
    inputs, input_summary = build_prefill_inputs(
        crops=crops,
        tokenizer=tokenizer,
        pre_cfg=pre_cfg,
        prompt_override=args.prompt,
    )
    return inputs, {
        "page": page_summary,
        "layout": layout_summary,
        "crop": crop_summary,
        "input_build": input_summary,
        "preprocessor": pre_cfg,
    }


def apply_runtime_env(args: argparse.Namespace) -> None:
    os.environ["PADDLE_OCR_VL_VISION_ATTENTION"] = str(args.vision_attention)
    os.environ["PADDLE_OCR_VL_VISION_PROMPT_FA_LAYOUT"] = str(args.vision_prompt_fa_layout)
    os.environ["PADDLE_OCR_VL_VISION_PROMPT_FA_MASK_SPARSE_MODE"] = str(args.vision_prompt_fa_mask_sparse_mode)


def load_model_for_args(args: argparse.Namespace) -> tuple[LocalPaddleOCRVLForConditionalGeneration, Path, torch.device, torch.dtype]:
    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype)
    configure_npu_jit_compile(args.npu_jit_compile, device)
    model_dir = _resolve_model_dir(args.model)
    maybe_sync(device)
    start = time.perf_counter()
    model = LocalPaddleOCRVLForConditionalGeneration.from_pretrained(model_dir, dtype=dtype, device=device)
    maybe_sync(device)
    print(
        f"MODEL_LOAD model={model_dir} device={device} dtype={dtype} elapsed_s={time.perf_counter() - start:.3f}",
        file=sys.stderr,
        flush=True,
    )
    return model, model_dir, device, dtype


def save_reference_tensor(path: Path, tensors: dict[str, torch.Tensor], *, item: PrefillInput, merge_size: int) -> dict[str, Any]:
    payload = {
        "item": input_row(item, merge_size=merge_size),
        "tensors": {
            "visual_features": tensors["visual_features"].detach().cpu(),
            "image_embeds": tensors["image_embeds"].detach().cpu(),
            "prefill_logits": tensors["prefill_logits"].detach().cpu(),
            "input_ids": tensors["input_ids"].detach().cpu(),
            "attention_mask": tensors["attention_mask"].detach().cpu(),
            "image_grid_thw": tensors["image_grid_thw"].detach().cpu(),
        },
        "topk": topk_summary(tensors["prefill_logits"]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {
        "tensor_file": str(path.name),
        "tensor_sha256": sha256_file(path),
        "tensor_summaries": {
            "visual_features": tensor_summary(payload["tensors"]["visual_features"]),
            "image_embeds": tensor_summary(payload["tensors"]["image_embeds"]),
            "prefill_logits": tensor_summary(payload["tensors"]["prefill_logits"]),
        },
        "topk": payload["topk"],
    }


@torch.inference_mode()
def make_baseline(args: argparse.Namespace) -> None:
    apply_runtime_env(args)
    baseline_dir = Path(args.baseline_dir).expanduser().resolve()
    manifest_path = baseline_dir / "reference_manifest.json"
    tensor_dir = baseline_dir / "tensors"
    if manifest_path.exists() and not bool(args.force):
        raise FileExistsError(f"baseline manifest already exists: {manifest_path}; pass --force to overwrite")
    if bool(args.force) and baseline_dir.exists():
        for old_file in baseline_dir.glob("tensors/*.pt"):
            old_file.unlink()
    baseline_dir.mkdir(parents=True, exist_ok=True)
    tensor_dir.mkdir(parents=True, exist_ok=True)

    model, model_dir, device, dtype = load_model_for_args(args)
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    inputs, build_summary = build_inputs_for_args(args, model_dir=model_dir, tokenizer=tokenizer)
    selected, selection_summary = select_stratified_inputs(
        inputs,
        count=int(args.crop_count),
        bucket_count=int(args.selection_buckets),
    )
    merge_size = int(build_summary["preprocessor"]["merge_size"])
    rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, float]] = []
    for idx, item in enumerate(selected):
        print(
            f"BASELINE_ITEM {idx + 1}/{len(selected)} id={item.entry.get('id')} "
            f"vision_tokens={vision_tokens(item)} label={item.entry.get('layout_label')}",
            file=sys.stderr,
            flush=True,
        )
        tensors, timing = run_prefill_measurement(
            model=model,
            item=item,
            device=device,
            cache_length=int(args.cache_length),
            timing_mode=str(args.timing_mode),
        )
        timing_rows.append(timing)
        tensor_name = f"{idx:04d}_{item.entry.get('id')}.pt"
        tensor_meta = save_reference_tensor(tensor_dir / tensor_name, tensors, item=item, merge_size=merge_size)
        rows.append(
            {
                "index": int(idx),
                **input_row(item, merge_size=merge_size),
                **tensor_meta,
                "timing_s": timing,
            }
        )

    phase_keys = sorted({key for row in timing_rows for key in row})
    manifest = {
        "schema_version": 1,
        "experiment": "07_vision_prefill_optimization",
        "kind": "vision_prefill_reference_baseline",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reference_contract": {
            "dtype": str(dtype),
            "device": str(device),
            "compiled": False,
            "vision_attention": get_vision_attention_impl(),
            "vision_prompt_fa_layout": get_vision_prompt_fa_layout(),
            "vision_prompt_fa_mask_sparse_mode": int(args.vision_prompt_fa_mask_sparse_mode),
            "authoritative_npu_reference": bool(
                device.type == "npu"
                and dtype == torch.float16
                and get_vision_attention_impl() == "prompt_flash_attention"
            ),
            "stored_tensors": ["visual_features", "image_embeds", "prefill_logits"],
            "decode_in_scope": False,
            "timing_mode": str(args.timing_mode),
            "timing_mode_note": (
                "phase_sync uses one device synchronize before and after every named phase; e2e uses one "
                "device synchronize around the whole prefill call; both runs both contracts and returns e2e tensors."
            ),
        },
        "model": str(model_dir),
        "baseline_dir": str(baseline_dir),
        "tensor_dir": "tensors",
        "cache_length": int(args.cache_length),
        "selection": selection_summary,
        "build_summary": build_summary,
        "item_count": int(len(rows)),
        "label_counts": dict(sorted(Counter(row["layout_label"] for row in rows).items())),
        "vision_tokens": stats([float(row["vision_tokens"]) for row in rows]),
        "projected_image_tokens": stats([float(row["projected_image_tokens"]) for row in rows]),
        "input_tokens": stats([float(row["input_tokens"]) for row in rows]),
        "phase_timing_s": {key: stats([float(row[key]) for row in timing_rows if key in row]) for key in phase_keys},
        "items": rows,
    }
    manifest["manifest_sha256_without_self"] = sha256_json(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    print(json.dumps({"baseline_manifest": str(manifest_path), "item_count": len(rows)}, indent=2), flush=True)


def load_baseline_manifest(path: Path) -> dict[str, Any]:
    if path.is_dir():
        path = path / "reference_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"baseline manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def crop_from_manifest_item(item: dict[str, Any], *, dataset_dir: Path) -> LayoutCrop:
    images_dir = dataset_dir / "images"
    image_rel = str(item["image_rel"]) if "image_rel" in item else ""
    if not image_rel:
        # Older or hand-written manifests can fall back to source_image.
        image_path = Path(str(item.get("source_image", "")))
    else:
        image_path, _mode, candidates = resolve_page_image_path(images_dir, image_rel)
        if image_path is None:
            raise FileNotFoundError(
                f"could not resolve image for baseline item {item.get('id')}: "
                + ", ".join(str(path) for path in candidates)
            )
    bbox = item.get("bbox_xyxy")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"baseline item {item.get('id')} lacks bbox_xyxy")
    with Image.open(image_path).convert("RGB") as image:
        crop = image.crop(tuple(int(value) for value in bbox)).copy()
    entry = {
        "id": item["id"],
        "source_image": str(image_path),
        "image_rel": image_rel,
        "page_index": int(item.get("page_index", 0)),
        "dataset_index": int(item.get("dataset_index", 0)),
        "layout_label": str(item.get("layout_label", "")),
        "category_type": str(item.get("layout_label", "")),
        "bbox_xyxy": bbox,
        "crop_size": item.get("crop_size", [0, 0]),
        "suggested_prompt": str(item.get("prompt", "OCR:")),
        "ground_truth": str(item.get("ground_truth_sample", "")),
        "ground_truth_source": str(item.get("ground_truth_source", "")),
    }
    return LayoutCrop(entry=entry, image=crop)


def build_inputs_from_manifest(
    *,
    manifest: dict[str, Any],
    model_dir: Path,
    tokenizer: Tokenizer,
    dataset_dir: Path,
) -> list[PrefillInput]:
    pre_cfg = load_preprocessor_config(model_dir)
    crops = [crop_from_manifest_item(row, dataset_dir=dataset_dir) for row in manifest["items"]]
    inputs, _summary = build_prefill_inputs(crops=crops, tokenizer=tokenizer, pre_cfg=pre_cfg, prompt_override=None)
    by_id = {str(item.entry["id"]): item for item in inputs}
    ordered = [by_id[str(row["id"])] for row in manifest["items"]]
    return ordered


@torch.inference_mode()
def compare_candidate(args: argparse.Namespace) -> None:
    apply_runtime_env(args)
    baseline_path = Path(args.baseline).expanduser().resolve()
    manifest = load_baseline_manifest(baseline_path)
    baseline_dir = baseline_path if baseline_path.is_dir() else baseline_path.parent
    tensor_dir = baseline_dir / str(manifest.get("tensor_dir", "tensors"))
    model, model_dir, device, dtype = load_model_for_args(args)
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    dataset_dir = resolve_dataset_dir(args.dataset_dir or manifest["build_summary"]["page"]["dataset_dir"])
    inputs = build_inputs_from_manifest(manifest=manifest, model_dir=model_dir, tokenizer=tokenizer, dataset_dir=dataset_dir)
    merge_size = int(load_preprocessor_config(model_dir)["merge_size"])

    max_items = int(args.max_items)
    if max_items > 0:
        inputs = inputs[:max_items]
    rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, float]] = []
    for idx, item in enumerate(inputs):
        baseline_item = manifest["items"][idx]
        tensor_path = tensor_dir / str(baseline_item["tensor_file"])
        if sha256_file(tensor_path) != str(baseline_item["tensor_sha256"]):
            raise RuntimeError(f"baseline tensor sha256 mismatch: {tensor_path}")
        baseline_payload = torch.load(tensor_path, map_location="cpu")
        vision_forward, vision_compile = prepare_candidate_vision_forward(
            args=args,
            model=model,
            item=item,
            device=device,
        )
        for _ in range(int(args.warmup_repeats)):
            run_prefill_measurement(
                model=model,
                item=item,
                device=device,
                cache_length=int(args.cache_length),
                timing_mode=str(args.timing_mode),
                vision_forward=vision_forward,
            )
        repeat_timings: list[dict[str, float]] = []
        candidate_tensors: dict[str, torch.Tensor] | None = None
        for _ in range(int(args.repeats)):
            candidate_tensors, timing = run_prefill_measurement(
                model=model,
                item=item,
                device=device,
                cache_length=int(args.cache_length),
                timing_mode=str(args.timing_mode),
                vision_forward=vision_forward,
            )
            repeat_timings.append(timing)
            timing_rows.append(timing)
        assert candidate_tensors is not None
        baseline_tensors = baseline_payload["tensors"]
        diffs = {
            "visual_features": diff_stats(candidate_tensors["visual_features"].cpu(), baseline_tensors["visual_features"]),
            "image_embeds": diff_stats(candidate_tensors["image_embeds"].cpu(), baseline_tensors["image_embeds"]),
            "prefill_logits": diff_stats(candidate_tensors["prefill_logits"].cpu(), baseline_tensors["prefill_logits"]),
        }
        candidate_topk = topk_summary(candidate_tensors["prefill_logits"])
        baseline_topk = baseline_payload["topk"]
        baseline_argmax = int(baseline_topk["argmax"])
        rows.append(
            {
                "index": int(idx),
                **input_row(item, merge_size=merge_size),
                "diffs": diffs,
                "baseline_topk": baseline_topk,
                "candidate_topk": candidate_topk,
                "argmax_match": bool(int(candidate_topk["argmax"]) == baseline_argmax),
                "candidate_top8_contains_baseline_argmax": bool(baseline_argmax in set(candidate_topk["topk_indices"])),
                "vision_compile": clean_json(vision_compile),
                "timing_s": {key: stats([float(row[key]) for row in repeat_timings if key in row]) for key in sorted({key for row in repeat_timings for key in row})},
            }
        )
        print(
            f"COMPARE_ITEM {idx + 1}/{len(inputs)} id={item.entry.get('id')} "
            f"logit_max_abs={diffs['prefill_logits'].get('max_abs_diff')} argmax_match={rows[-1]['argmax_match']}",
            file=sys.stderr,
            flush=True,
        )

    output = {
        "schema_version": 1,
        "experiment": "07_vision_prefill_optimization",
        "kind": "vision_prefill_candidate_compare",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": {
            "name": str(args.candidate_name),
            "dtype": str(dtype),
            "device": str(device),
            "compiled": bool(
                str(args.candidate_vision_path) == "static_visual"
                and str(args.vision_compile_backend) != "none"
            ),
            "compile_api": (
                "torch.compile"
                if str(args.candidate_vision_path) == "static_visual"
                and str(args.vision_compile_backend) != "none"
                else None
            ),
            "vision_attention": get_vision_attention_impl(),
            "vision_prompt_fa_layout": get_vision_prompt_fa_layout(),
            "vision_prompt_fa_mask_sparse_mode": int(args.vision_prompt_fa_mask_sparse_mode),
            "repeats": int(args.repeats),
            "warmup_repeats": int(args.warmup_repeats),
            "timing_mode": str(args.timing_mode),
            "candidate_vision_path": str(args.candidate_vision_path),
            "vision_compile_backend": str(args.vision_compile_backend),
            "static_visual_pad_mode": str(args.static_visual_pad_mode),
            "timing_mode_note": (
                "e2e is the real candidate latency metric: one synchronize before and after the whole prefill. "
                "phase_sync is diagnostic and intentionally syncs around every named phase."
            ),
        },
        "baseline": {
            "path": str(baseline_path),
            "reference_contract": manifest.get("reference_contract", {}),
            "item_count": int(manifest.get("item_count", len(manifest.get("items", [])))),
        },
        "cache_length": int(args.cache_length),
        "compared_count": int(len(rows)),
        "summary": {
            "argmax_match_count": int(sum(bool(row["argmax_match"]) for row in rows)),
            "top8_contains_baseline_argmax_count": int(sum(bool(row["candidate_top8_contains_baseline_argmax"]) for row in rows)),
            "visual_features": aggregate_diff(rows, "visual_features"),
            "image_embeds": aggregate_diff(rows, "image_embeds"),
            "prefill_logits": aggregate_diff(rows, "prefill_logits"),
            "phase_timing_s": {
                key: stats([float(row[key]) for row in timing_rows if key in row])
                for key in sorted({key for row in timing_rows for key in row})
            },
            "vision_tokens": stats([float(row["vision_tokens"]) for row in rows]),
            "projected_image_tokens": stats([float(row["projected_image_tokens"]) for row in rows]),
        },
        "items": rows,
    }
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    print(json.dumps({"compare_output": str(output_path), "summary": output["summary"]}, indent=2, default=json_default), flush=True)


def add_common_args(parser: argparse.ArgumentParser, *, timing_default: str) -> None:
    parser.add_argument("--model", default="PaddlePaddle/PaddleOCR-VL-1.6")
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="fp16", choices=DTYPE_CHOICES)
    parser.add_argument("--npu-jit-compile", default="off", choices=NPU_JIT_COMPILE_CHOICES)
    parser.add_argument("--vision-attention", default="prompt_flash_attention", choices=VISION_ATTENTION_CHOICES)
    parser.add_argument("--vision-prompt-fa-layout", default="bnsd", choices=("bnsd", "bsnd", "bsh"))
    parser.add_argument("--vision-prompt-fa-mask-sparse-mode", type=int, default=1, choices=(0, 1))
    parser.add_argument("--cache-length", type=int, default=2048)
    parser.add_argument("--timing-mode", default=str(timing_default), choices=TIMING_MODE_CHOICES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    make_parser = subparsers.add_parser("make-baseline", help="Create the stored reference baseline bundle.")
    add_common_args(make_parser, timing_default="phase_sync")
    make_parser.add_argument("--baseline-dir", default=str(SCRIPT_DIR / "baselines" / "promptfa_fp16_eager_64"))
    make_parser.add_argument("--page-start", type=int, default=0)
    make_parser.add_argument("--num-pages", type=int, default=64)
    make_parser.add_argument("--crop-count", type=int, default=64)
    make_parser.add_argument("--selection-buckets", type=int, default=8)
    make_parser.add_argument("--crop-padding", type=int, default=0)
    make_parser.add_argument("--min-crop-side", type=int, default=4)
    make_parser.add_argument("--skip-labels", default="")
    make_parser.add_argument("--include-ignored-gt", action="store_true")
    make_parser.add_argument("--include-empty-gt", action="store_true")
    make_parser.add_argument("--prompt", default=None)
    make_parser.add_argument("--force", action="store_true")

    compare_parser = subparsers.add_parser("compare", help="Compare a candidate path against a stored baseline.")
    add_common_args(compare_parser, timing_default="e2e")
    compare_parser.add_argument("--baseline", default=str(SCRIPT_DIR / "baselines" / "promptfa_fp16_eager_64"))
    compare_parser.add_argument("--candidate-name", default="candidate")
    compare_parser.add_argument("--output", default=str(SCRIPT_DIR / "outputs" / "candidate_compare.json"))
    compare_parser.add_argument("--repeats", type=int, default=1)
    compare_parser.add_argument("--warmup-repeats", type=int, default=0)
    compare_parser.add_argument("--max-items", type=int, default=0)
    compare_parser.add_argument("--candidate-vision-path", default="eager_visual", choices=CANDIDATE_VISION_PATH_CHOICES)
    compare_parser.add_argument("--vision-compile-backend", default="none", choices=VISION_COMPILE_BACKEND_CHOICES)
    compare_parser.add_argument("--static-visual-pad-mode", default="none", choices=STATIC_VISUAL_PAD_MODE_CHOICES)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "make-baseline":
        make_baseline(args)
    elif args.command == "compare":
        compare_candidate(args)
    else:
        raise RuntimeError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
