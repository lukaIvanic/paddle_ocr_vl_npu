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

  probe-promptfa-mask
    Synthetic NPU-only check for PromptFA atten_mask semantics.

  probe-promptfa-compile
    Synthetic NPU-only eager-vs-compiled check for PromptFA masking and TorchAir
    lowering, without loading the OCR model.
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
import torch.nn.functional as F
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
TIMING_MODE_CHOICES = ("standard", "phase_sync")
VISION_COMPILE_BACKEND_CHOICES = ("none", "default", "aot_eager", "inductor", "torchair")
TORCHAIR_MODE_CHOICES = ("default", "max-autotune")
TORCHAIR_GRAPH_DUMP_TYPE_CHOICES = ("none", "txt", "pbtxt")
TORCHAIR_MSIT_DUMP_KIND_CHOICES = ("none", "ge", "fx")
TORCHAIR_MSIT_DUMP_MODE_CHOICES = ("input", "output", "all")
LAYERNORM_PROBE_IMPL_CHOICES = ("nn", "functional", "manual", "manual_fp16_reduce", "npu_eval")
VISUAL_PREFIX_STAGE_CHOICES = (
    "patch_conv",
    "patch_flat",
    "patch_pad",
    "patch_pos",
    "ln1",
    "qkv",
    "qk_rope_v",
    "attn_out",
    "layer0_out",
)
QKV_LINEAR_PROBE_IMPL_CHOICES = (
    "module_three",
    "module_q",
    "functional_three",
    "functional_q",
    "matmul_three",
    "matmul_q",
    "functional_single",
    "matmul_single",
    "functional_q_no_bias",
    "matmul_q_no_bias",
    "addmm_q",
    "mm_q",
    "bmm_q",
    "matmul_3d_q",
    "einsum_q",
    "conv1d_q",
    "npu_bmm_v2_q",
    "npu_linear_q",
    "npu_linear_three",
    "npu_linear_single",
    "npu_grouped_matmul_q",
)
QKV_LINEAR_PROBE_SOURCE_CHOICES = ("ln1", "patch_pos")
QKV_LINEAR_PROBE_LN_IMPL_CHOICES = ("module", "functional", "manual_fp32", "manual_fp16")
QKV_LINEAR_PROBE_BRIDGE_CHOICES = (
    "none",
    "contiguous",
    "clone",
    "add_zero",
    "mul_one",
    "reshape_contiguous",
    "format_cast_nd",
    "format_cast_nz_then_nd",
    "transpose_roundtrip",
)
STATIC_VISUAL_PAD_POLICY = "always_mask_pad_to_atlas_inference_alignment"

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
DEFAULT_MODEL = os.environ.get("MODEL", "/home/lukaiv/models/paddle_ocr_0_9b_v_1_6")


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


def summarize_tree(path: str | Path | None, *, max_files: int = 24) -> dict[str, Any]:
    if path is None or not str(path).strip():
        return {"path": None, "exists": False, "file_count": 0, "sample_files": []}
    root = Path(path).expanduser().resolve()
    if not root.exists():
        return {"path": str(root), "exists": False, "file_count": 0, "sample_files": []}
    files = [item for item in root.rglob("*") if item.is_file()]
    sample_files = []
    for item in sorted(files)[: int(max_files)]:
        try:
            rel_path = str(item.relative_to(root))
        except ValueError:
            rel_path = str(item)
        sample_files.append({"path": rel_path, "size": int(item.stat().st_size)})
    return {
        "path": str(root),
        "exists": True,
        "file_count": int(len(files)),
        "sample_files": sample_files,
    }


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
    physical_seq_len = int(real_seq_len) + int(pad_tokens)
    mask = torch.zeros((1, 1, physical_seq_len, physical_seq_len), device=device, dtype=torch.bool)
    real = int(real_seq_len)
    if int(pad_tokens) > 0:
        mask[..., :real, real:physical_seq_len] = True
        mask[..., real:physical_seq_len, :real] = True
    return mask.contiguous()


def static_visual_pad_tokens(
    real_seq_len: int,
    *,
    debug_no_padding: bool = False,
    debug_min_pad_tokens: int = 0,
    debug_pad_to_multiple: int = 0,
) -> int:
    """Return static visual padding for the masked padded candidate path.

    Normal experiment-07 candidates always add at least one masked dummy row. For Atlas inference
    cards, PromptFA requires S > 128 to be 128-aligned when using an attention mask, so larger crops
    round to the next 128 boundary. Shorter shapes round to a 16 boundary. The no-padding case is
    available only through the explicit diagnostic flag.
    """
    real_seq_len = int(real_seq_len)
    if bool(debug_no_padding):
        return 0
    debug_min_pad_tokens = max(0, int(debug_min_pad_tokens))
    debug_pad_to_multiple = max(0, int(debug_pad_to_multiple))

    minimum_physical_seq_len = real_seq_len + 1
    alignment = 128 if minimum_physical_seq_len > 128 else 16
    remainder = minimum_physical_seq_len % alignment
    physical_seq_len = (
        minimum_physical_seq_len
        if remainder == 0
        else minimum_physical_seq_len + (alignment - remainder)
    )
    pad_tokens = int(physical_seq_len - real_seq_len)

    if debug_min_pad_tokens:
        pad_tokens = max(int(pad_tokens), int(debug_min_pad_tokens))
    if debug_pad_to_multiple:
        physical_seq_len = real_seq_len + int(pad_tokens)
        remainder = physical_seq_len % int(debug_pad_to_multiple)
        if remainder:
            pad_tokens += int(debug_pad_to_multiple) - int(remainder)
    return int(pad_tokens)


def slice_visual_features_to_real(visual_features: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
    real_seq_len = int(image_grid_thw.prod().item())
    return visual_features[:real_seq_len]


class SingleCropStaticVisualModule(torch.nn.Module):
    """Shape-specialized visual encoder wrapper for fullgraph static compilation."""

    def __init__(
        self,
        model: LocalPaddleOCRVLForConditionalGeneration,
        image_grid_thw: torch.Tensor,
        *,
        device: torch.device,
        debug_no_padding: bool = False,
        debug_min_pad_tokens: int = 0,
        debug_pad_to_multiple: int = 0,
    ):
        super().__init__()
        self.model = model
        self.static_visual_pad_policy = STATIC_VISUAL_PAD_POLICY
        self.register_buffer("image_grid_thw_const", image_grid_thw.detach().clone(), persistent=False)
        self.register_buffer(
            "cu_seqlens_const",
            build_single_crop_vision_cu_seqlens(image_grid_thw, device=device),
            persistent=False,
        )
        self.static_real_seq_len = int(image_grid_thw.prod().item())
        grid_t, _grid_h, _grid_w = single_crop_grid_ints(image_grid_thw)
        if int(grid_t) != 1:
            raise ValueError("static_visual padding currently supports single-image crop grids with T=1 only")
        self.debug_no_padding = bool(debug_no_padding)
        self.debug_min_pad_tokens = max(0, int(debug_min_pad_tokens))
        self.debug_pad_to_multiple = max(0, int(debug_pad_to_multiple))
        self.static_pad_tokens = static_visual_pad_tokens(
            self.static_real_seq_len,
            debug_no_padding=self.debug_no_padding,
            debug_min_pad_tokens=self.debug_min_pad_tokens,
            debug_pad_to_multiple=self.debug_pad_to_multiple,
        )
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
        if self.static_pad_tokens:
            pad_mask = build_static_pad_attention_mask(self.static_real_seq_len, self.static_pad_tokens, device=device)
        else:
            pad_mask = None
        self.register_buffer("static_pad_attention_mask", pad_mask, persistent=False)

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
                raise ValueError("static_visual currently supports PromptFA layout bnsd only")
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
            if self.static_pad_attention_mask is not None:
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
        return hidden_states + encoder_layer.mlp(encoder_layer.layer_norm2(hidden_states))

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
        for encoder_layer in transformer.encoder.layers:
            hidden_states = self._static_mask_padded_encoder_layer(encoder_layer, hidden_states)
        hidden_states = transformer.post_layernorm(hidden_states)
        return hidden_states


class SingleCropStaticVisualPrefixModule(torch.nn.Module):
    """Shape-specialized prefix boundary for localizing static visual GE drift."""

    def __init__(
        self,
        model: LocalPaddleOCRVLForConditionalGeneration,
        image_grid_thw: torch.Tensor,
        *,
        device: torch.device,
        stage: str,
        debug_no_padding: bool = False,
        debug_min_pad_tokens: int = 0,
        debug_pad_to_multiple: int = 0,
    ):
        super().__init__()
        if stage not in VISUAL_PREFIX_STAGE_CHOICES:
            raise ValueError(f"unsupported static visual prefix stage={stage!r}")
        self.stage = str(stage)
        self.static_visual = SingleCropStaticVisualModule(
            model,
            image_grid_thw,
            device=device,
            debug_no_padding=debug_no_padding,
            debug_min_pad_tokens=debug_min_pad_tokens,
            debug_pad_to_multiple=debug_pad_to_multiple,
        ).eval()

    @property
    def static_real_seq_len(self) -> int:
        return int(self.static_visual.static_real_seq_len)

    @property
    def static_pad_tokens(self) -> int:
        return int(self.static_visual.static_pad_tokens)

    @property
    def static_physical_seq_len(self) -> int:
        return int(self.static_visual.static_physical_seq_len)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        transformer = self.static_visual.model.visual.vision_model
        embeddings_module = transformer.embeddings
        pixel_values = pixel_values.to(dtype=embeddings_module.patch_embedding.weight.dtype)
        patch_embeds = embeddings_module.patch_embedding(pixel_values)
        if self.stage == "patch_conv":
            return patch_embeds

        hidden_states = patch_embeds.flatten(-2).squeeze(-1)
        if self.stage == "patch_flat":
            return hidden_states

        if self.static_visual.static_pad_tokens:
            hidden_states = torch.cat(
                [
                    hidden_states,
                    torch.zeros(
                        self.static_visual.static_pad_tokens,
                        hidden_states.shape[-1],
                        device=hidden_states.device,
                        dtype=hidden_states.dtype,
                    ),
                ],
                dim=0,
            )
        if self.stage == "patch_pad":
            return hidden_states

        hidden_states = hidden_states + self.static_visual.abs_pos_embed_const
        if self.stage == "patch_pos":
            return hidden_states

        encoder_layer = transformer.encoder.layers[0]
        ln1_hidden = encoder_layer.layer_norm1(hidden_states)
        if self.stage == "ln1":
            return ln1_hidden

        attention = encoder_layer.self_attn
        seq_length = ln1_hidden.shape[0]
        query_states = attention.q_proj(ln1_hidden).view(seq_length, attention.num_heads, attention.head_dim)
        key_states = attention.k_proj(ln1_hidden).view(seq_length, attention.num_heads, attention.head_dim)
        value_states = attention.v_proj(ln1_hidden).view(seq_length, attention.num_heads, attention.head_dim)
        if self.stage == "qkv":
            return torch.cat(
                [
                    query_states.reshape(seq_length, -1),
                    key_states.reshape(seq_length, -1),
                    value_states.reshape(seq_length, -1),
                ],
                dim=-1,
            )

        query_states, key_states = apply_rotary_pos_emb_vision(
            query_states,
            key_states,
            self.static_visual.vision_rope_cos_const,
            self.static_visual.vision_rope_sin_const,
        )
        if self.stage == "qk_rope_v":
            return torch.cat(
                [
                    query_states.reshape(seq_length, -1),
                    key_states.reshape(seq_length, -1),
                    value_states.reshape(seq_length, -1),
                ],
                dim=-1,
            )

        attn_output = self.static_visual._static_mask_padded_attention(attention, ln1_hidden)
        if self.stage == "attn_out":
            return attn_output

        hidden_states = hidden_states + attn_output
        hidden_states = hidden_states + encoder_layer.mlp(encoder_layer.layer_norm2(hidden_states))
        if self.stage == "layer0_out":
            return hidden_states
        raise RuntimeError(f"unreachable static visual prefix stage={self.stage!r}")


class VisionQKVLinearProbeModule(torch.nn.Module):
    """Small QKV projection variants for isolating TorchAir linear lowering."""

    def __init__(self, attention: torch.nn.Module, *, impl: str, bridge: str = "none"):
        super().__init__()
        if impl not in QKV_LINEAR_PROBE_IMPL_CHOICES:
            raise ValueError(f"unsupported QKV linear probe impl={impl!r}")
        if bridge not in QKV_LINEAR_PROBE_BRIDGE_CHOICES:
            raise ValueError(f"unsupported QKV linear probe bridge={bridge!r}")
        self.impl = str(impl)
        self.bridge = str(bridge)
        self.q_proj = attention.q_proj
        self.k_proj = attention.k_proj
        self.v_proj = attention.v_proj
        q_weight = attention.q_proj.weight.detach().clone().contiguous()
        k_weight = attention.k_proj.weight.detach().clone().contiguous()
        v_weight = attention.v_proj.weight.detach().clone().contiguous()
        q_bias = None if attention.q_proj.bias is None else attention.q_proj.bias.detach().clone().contiguous()
        k_bias = None if attention.k_proj.bias is None else attention.k_proj.bias.detach().clone().contiguous()
        v_bias = None if attention.v_proj.bias is None else attention.v_proj.bias.detach().clone().contiguous()
        self.register_buffer("q_weight", q_weight, persistent=False)
        self.register_buffer("k_weight", k_weight, persistent=False)
        self.register_buffer("v_weight", v_weight, persistent=False)
        self.register_buffer("q_bias", q_bias, persistent=False)
        self.register_buffer("k_bias", k_bias, persistent=False)
        self.register_buffer("v_bias", v_bias, persistent=False)
        self.register_buffer("qkv_weight", torch.cat([q_weight, k_weight, v_weight], dim=0).contiguous(), persistent=False)
        if q_bias is None or k_bias is None or v_bias is None:
            qkv_bias = None
        else:
            qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0).contiguous()
        self.register_buffer("qkv_bias", qkv_bias, persistent=False)

    @staticmethod
    def _linear_matmul(hidden_states: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        output = torch.matmul(hidden_states, weight.transpose(0, 1))
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    def _linear_addmm(hidden_states: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        if bias is None:
            return torch.mm(hidden_states, weight.transpose(0, 1))
        return torch.addmm(bias, hidden_states, weight.transpose(0, 1))

    @staticmethod
    def _linear_bmm(hidden_states: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        output = torch.bmm(hidden_states.unsqueeze(0), weight.transpose(0, 1).unsqueeze(0)).squeeze(0)
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    def _linear_npu_bmm_v2(hidden_states: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        import torch_npu

        output = torch_npu.npu_bmmV2(
            hidden_states.unsqueeze(0),
            weight.transpose(0, 1).unsqueeze(0),
            [],
        ).squeeze(0)
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    def _linear_matmul_3d(hidden_states: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        output = torch.matmul(hidden_states.unsqueeze(0), weight.transpose(0, 1).unsqueeze(0)).squeeze(0)
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    def _linear_einsum(hidden_states: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        output = torch.einsum("sh,oh->so", hidden_states, weight)
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    def _linear_conv1d(hidden_states: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        output = F.conv1d(hidden_states.transpose(0, 1).unsqueeze(0), weight.unsqueeze(-1), bias)
        return output.squeeze(0).transpose(0, 1)

    @staticmethod
    def _linear_npu(hidden_states: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        import torch_npu

        return torch_npu.npu_linear(hidden_states, weight, bias)

    @staticmethod
    def _linear_grouped_matmul(
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        import torch_npu

        group_list = torch.full((1,), hidden_states.shape[0], dtype=torch.int64, device=hidden_states.device)
        bias_arg = None if bias is None else [bias]
        weight_3d = weight.transpose(0, 1).contiguous().unsqueeze(0)
        return torch_npu.npu_grouped_matmul(
            [hidden_states],
            [weight_3d],
            bias=bias_arg,
            group_list=group_list,
            split_item=2,
            group_type=0,
            group_list_type=1,
        )[0]

    def _bridge_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.bridge == "none":
            return hidden_states
        if self.bridge == "contiguous":
            return hidden_states.contiguous()
        if self.bridge == "clone":
            return hidden_states.clone()
        if self.bridge == "add_zero":
            return hidden_states + torch.zeros_like(hidden_states)
        if self.bridge == "mul_one":
            return hidden_states * torch.ones((), device=hidden_states.device, dtype=hidden_states.dtype)
        if self.bridge == "reshape_contiguous":
            return hidden_states.reshape(hidden_states.shape[0], hidden_states.shape[-1]).contiguous()
        if self.bridge == "format_cast_nd":
            import torch_npu

            return torch_npu.npu_format_cast(hidden_states, 2)
        if self.bridge == "format_cast_nz_then_nd":
            import torch_npu

            return torch_npu.npu_format_cast(torch_npu.npu_format_cast(hidden_states, 29), 2)
        if self.bridge == "transpose_roundtrip":
            return hidden_states.transpose(0, 1).contiguous().transpose(0, 1).contiguous()
        raise RuntimeError(f"unreachable QKV linear bridge={self.bridge!r}")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self._bridge_hidden_states(hidden_states)
        if self.impl == "module_three":
            return torch.cat(
                [
                    self.q_proj(hidden_states),
                    self.k_proj(hidden_states),
                    self.v_proj(hidden_states),
                ],
                dim=-1,
            )
        if self.impl == "module_q":
            return self.q_proj(hidden_states)
        if self.impl == "functional_three":
            return torch.cat(
                [
                    F.linear(hidden_states, self.q_weight, self.q_bias),
                    F.linear(hidden_states, self.k_weight, self.k_bias),
                    F.linear(hidden_states, self.v_weight, self.v_bias),
                ],
                dim=-1,
            )
        if self.impl == "functional_q":
            return F.linear(hidden_states, self.q_weight, self.q_bias)
        if self.impl == "matmul_three":
            return torch.cat(
                [
                    self._linear_matmul(hidden_states, self.q_weight, self.q_bias),
                    self._linear_matmul(hidden_states, self.k_weight, self.k_bias),
                    self._linear_matmul(hidden_states, self.v_weight, self.v_bias),
                ],
                dim=-1,
            )
        if self.impl == "matmul_q":
            return self._linear_matmul(hidden_states, self.q_weight, self.q_bias)
        if self.impl == "functional_single":
            return F.linear(hidden_states, self.qkv_weight, self.qkv_bias)
        if self.impl == "matmul_single":
            return self._linear_matmul(hidden_states, self.qkv_weight, self.qkv_bias)
        if self.impl == "functional_q_no_bias":
            return F.linear(hidden_states, self.q_weight, None)
        if self.impl == "matmul_q_no_bias":
            return self._linear_matmul(hidden_states, self.q_weight, None)
        if self.impl == "addmm_q":
            return self._linear_addmm(hidden_states, self.q_weight, self.q_bias)
        if self.impl == "mm_q":
            return self._linear_addmm(hidden_states, self.q_weight, None) if self.q_bias is None else (
                torch.mm(hidden_states, self.q_weight.transpose(0, 1)) + self.q_bias
            )
        if self.impl == "bmm_q":
            return self._linear_bmm(hidden_states, self.q_weight, self.q_bias)
        if self.impl == "matmul_3d_q":
            return self._linear_matmul_3d(hidden_states, self.q_weight, self.q_bias)
        if self.impl == "einsum_q":
            return self._linear_einsum(hidden_states, self.q_weight, self.q_bias)
        if self.impl == "conv1d_q":
            return self._linear_conv1d(hidden_states, self.q_weight, self.q_bias)
        if self.impl == "npu_bmm_v2_q":
            return self._linear_npu_bmm_v2(hidden_states, self.q_weight, self.q_bias)
        if self.impl == "npu_linear_q":
            return self._linear_npu(hidden_states, self.q_weight, self.q_bias)
        if self.impl == "npu_linear_three":
            return torch.cat(
                [
                    self._linear_npu(hidden_states, self.q_weight, self.q_bias),
                    self._linear_npu(hidden_states, self.k_weight, self.k_bias),
                    self._linear_npu(hidden_states, self.v_weight, self.v_bias),
                ],
                dim=-1,
            )
        if self.impl == "npu_linear_single":
            return self._linear_npu(hidden_states, self.qkv_weight, self.qkv_bias)
        if self.impl == "npu_grouped_matmul_q":
            return self._linear_grouped_matmul(hidden_states, self.q_weight, self.q_bias)
        raise RuntimeError(f"unreachable QKV linear probe impl={self.impl!r}")


class VisionLayerNormQKVLinearProbeModule(torch.nn.Module):
    """LayerNorm plus QKV projection as the smallest producer-consumer graph."""

    def __init__(
        self,
        layer_norm: torch.nn.Module,
        attention: torch.nn.Module,
        *,
        impl: str,
        bridge: str = "none",
        ln_impl: str = "module",
    ):
        super().__init__()
        self.layer_norm = layer_norm
        self.ln_impl = str(ln_impl)
        self.qkv = VisionQKVLinearProbeModule(attention, impl=impl, bridge=bridge)

    def _layer_norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.ln_impl == "module":
            return self.layer_norm(hidden_states)
        weight = self.layer_norm.weight
        bias = self.layer_norm.bias
        eps = float(self.layer_norm.eps)
        if self.ln_impl == "functional":
            return F.layer_norm(hidden_states, (hidden_states.shape[-1],), weight, bias, eps)
        if self.ln_impl == "manual_fp32":
            x = hidden_states.float()
            mean = x.mean(dim=-1, keepdim=True)
            var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
            y = (x - mean) * torch.rsqrt(var + eps)
            y = y.to(dtype=hidden_states.dtype)
            if weight is not None:
                y = y * weight
            if bias is not None:
                y = y + bias
            return y
        if self.ln_impl == "manual_fp16":
            mean = hidden_states.mean(dim=-1, keepdim=True)
            var = (hidden_states - mean).pow(2).mean(dim=-1, keepdim=True)
            y = (hidden_states - mean) * torch.rsqrt(var + eps)
            if weight is not None:
                y = y * weight
            if bias is not None:
                y = y + bias
            return y
        raise RuntimeError(f"unreachable QKV LayerNorm impl={self.ln_impl!r}")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.qkv(self._layer_norm(hidden_states))


def import_torchair():
    try:
        import torchair
        from torchair.configs.compiler_config import CompilerConfig

        return torchair, CompilerConfig
    except Exception:
        from torch_npu.dynamo import torchair
        from torch_npu.dynamo.torchair.configs.compiler_config import CompilerConfig

        return torchair, CompilerConfig


def set_torchair_graph_dump_path(graph_dump: Any, dump_dir: Path) -> list[str]:
    configured_attrs: list[str] = []
    dump_dir.mkdir(parents=True, exist_ok=True)
    for attr in ("path", "_path"):
        if hasattr(graph_dump, attr):
            setattr(graph_dump, attr, str(dump_dir))
            configured_attrs.append(attr)
    return configured_attrs


def parse_csv_string_list(raw: str) -> list[str]:
    values: list[str] = []
    for chunk in str(raw).replace(";", ",").split(","):
        chunk = chunk.strip()
        if chunk:
            values.append(chunk)
    if not values:
        raise ValueError(f"expected at least one comma-separated value, got {raw!r}")
    return values


def ensure_top_level_torchair_importable() -> None:
    try:
        import torchair  # noqa: F401

        return
    except Exception:
        pass
    try:
        from torch_npu.dynamo import torchair as torchair_module
        from torch_npu.dynamo.torchair import configs as configs_module
        from torch_npu.dynamo.torchair.configs import compiler_config as compiler_config_module
    except Exception as exc:
        raise RuntimeError(
            "TorchAir is not importable as either top-level `torchair` or `torch_npu.dynamo.torchair`."
        ) from exc

    sys.modules.setdefault("torchair", torchair_module)
    sys.modules.setdefault("torchair.configs", configs_module)
    sys.modules.setdefault("torchair.configs.compiler_config", compiler_config_module)


def apply_msit_torchair_dump_config(
    config: Any,
    *,
    kind: str,
    dump_dir: str | Path | None,
    dump_mode: str,
    dump_token: str,
    dump_layer: str,
    fusion_switch_file: str | Path | None,
) -> dict[str, Any]:
    if kind not in TORCHAIR_MSIT_DUMP_KIND_CHOICES:
        raise ValueError(f"--torchair-msit-dump-kind must be one of {TORCHAIR_MSIT_DUMP_KIND_CHOICES}, got {kind!r}")
    if kind == "none":
        return {"enabled": False, "kind": "none"}
    if dump_mode not in TORCHAIR_MSIT_DUMP_MODE_CHOICES:
        raise ValueError(f"--torchair-msit-dump-mode must be one of {TORCHAIR_MSIT_DUMP_MODE_CHOICES}, got {dump_mode!r}")
    if dump_dir is None or not str(dump_dir).strip():
        raise ValueError("--torchair-msit-dump-dir is required when --torchair-msit-dump-kind is ge or fx")

    base_dir = Path(dump_dir).expanduser().resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    fusion_path = None
    if fusion_switch_file is not None and str(fusion_switch_file).strip():
        fusion_path = Path(fusion_switch_file).expanduser().resolve()
        if not fusion_path.is_file():
            raise FileNotFoundError(f"--torchair-msit-fusion-switch-file does not exist: {fusion_path}")
    dump_token_values = parse_int_list(dump_token) if str(dump_token).strip() else None
    dump_layer_values = parse_csv_string_list(dump_layer) if str(dump_layer).strip() else None

    config_source = "msit_llm.dump.torchair_dump"
    ensure_top_level_torchair_importable()
    try:
        from msit_llm.dump import torchair_dump
    except Exception as exc:
        torchair_dump = None
        config_source = f"local_compat_fallback_after_{exc.__class__.__name__}"

    if kind == "ge" and torchair_dump is not None:
        torchair_dump.get_ge_dump_config(
            dump_path=str(base_dir),
            dump_mode=str(dump_mode),
            fusion_switch_file=str(fusion_path) if fusion_path is not None else None,
            dump_token=dump_token_values,
            dump_layer=dump_layer_values,
            compiler_config=config,
        )
        expected_dir = base_dir / "msit_ge_dump"
    elif kind == "fx" and torchair_dump is not None:
        if fusion_path is not None or dump_token_values is not None or dump_layer_values is not None:
            raise ValueError("MSIT FX dump does not accept fusion-switch, dump-token, or dump-layer filters")
        torchair_dump.get_fx_dump_config(dump_path=str(base_dir), compiler_config=config)
        expected_dir = base_dir / "msit_fx_dump"
    elif kind == "ge":
        expected_dir = base_dir / "msit_ge_dump"
        expected_dir.mkdir(parents=True, exist_ok=True)
        config.debug.graph_dump.type = "txt"
        if hasattr(config.debug.graph_dump, "_path"):
            setattr(config.debug.graph_dump, "_path", str(expected_dir))
        elif hasattr(config.debug.graph_dump, "path"):
            config.debug.graph_dump.path = str(expected_dir)
        if fusion_path is not None:
            config.fusion_config.fusion_switch_file = str(fusion_path)
        config.dump_config.enable_dump = True
        config.dump_config.dump_mode = str(dump_mode)
        config.dump_config.dump_path = str(expected_dir)
        if dump_token_values is not None:
            config.dump_config.dump_step = "|".join(str(value) for value in dump_token_values)
        if dump_layer_values is not None:
            config.dump_config.dump_layer = " ".join(dump_layer_values)
    else:
        if fusion_path is not None or dump_token_values is not None or dump_layer_values is not None:
            raise ValueError("MSIT FX dump does not accept fusion-switch, dump-token, or dump-layer filters")
        expected_dir = base_dir / "msit_fx_dump"
        expected_dir.mkdir(parents=True, exist_ok=True)
        config.debug.data_dump.type = "npy"
        if hasattr(config.debug.data_dump, "path"):
            config.debug.data_dump.path = str(expected_dir)

    return {
        "enabled": True,
        "kind": str(kind),
        "config_source": str(config_source),
        "dump_base_dir": str(base_dir),
        "expected_dump_dir": str(expected_dir),
        "dump_mode": str(dump_mode) if kind == "ge" else None,
        "dump_token": dump_token_values,
        "dump_layer": dump_layer_values,
        "fusion_switch_file": str(fusion_path) if fusion_path is not None else None,
        "doc_expected_compare_role": "my_path_ge_target" if kind == "ge" else "golden_path_fx_reference",
    }


def vision_compile_backend(
    name: str,
    device: torch.device,
    *,
    torchair_mode: str = "default",
    torchair_run_eagerly: bool = False,
    torchair_graph_dump_type: str = "none",
    torchair_graph_dump_dir: str | Path | None = None,
    torchair_msit_dump_kind: str = "none",
    torchair_msit_dump_dir: str | Path | None = None,
    torchair_msit_dump_mode: str = "output",
    torchair_msit_dump_token: str = "",
    torchair_msit_dump_layer: str = "",
    torchair_msit_fusion_switch_file: str | Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    if name == "default":
        return None, {"backend_kind": "torch_default"}
    if name == "torchair":
        if device.type != "npu":
            raise ValueError("--vision-compile-backend torchair requires --device npu:0")
        if torchair_mode not in TORCHAIR_MODE_CHOICES:
            raise ValueError(f"--torchair-mode must be one of {TORCHAIR_MODE_CHOICES}, got {torchair_mode!r}")
        if torchair_graph_dump_type not in TORCHAIR_GRAPH_DUMP_TYPE_CHOICES:
            raise ValueError(
                f"--torchair-graph-dump-type must be one of {TORCHAIR_GRAPH_DUMP_TYPE_CHOICES}, "
                f"got {torchair_graph_dump_type!r}"
            )
        if torchair_msit_dump_kind not in TORCHAIR_MSIT_DUMP_KIND_CHOICES:
            raise ValueError(
                f"--torchair-msit-dump-kind must be one of {TORCHAIR_MSIT_DUMP_KIND_CHOICES}, "
                f"got {torchair_msit_dump_kind!r}"
            )
        if torchair_run_eagerly and torchair_msit_dump_kind != "none":
            raise ValueError("--torchair-run-eagerly cannot be combined with MSIT GE/FX dump collection")
        torchair, CompilerConfig = import_torchair()
        config = CompilerConfig()
        meta: dict[str, Any] = {
            "backend_kind": "torchair",
            "torchair_mode": str(torchair_mode),
            "torchair_run_eagerly": bool(torchair_run_eagerly),
            "torchair_graph_dump_type": str(torchair_graph_dump_type),
            "torchair_graph_dump_dir": None,
            "torchair_graph_dump_path_attrs": [],
            "torchair_msit_dump": {"enabled": False, "kind": "none"},
        }
        if torchair_mode == "max-autotune":
            config.mode = str(torchair_mode)
        if torchair_run_eagerly:
            config.debug.run_eagerly = True
        if torchair_graph_dump_type != "none":
            config.debug.graph_dump.type = str(torchair_graph_dump_type)
            if torchair_graph_dump_dir is not None and str(torchair_graph_dump_dir).strip():
                dump_dir = Path(torchair_graph_dump_dir).expanduser().resolve()
                meta["torchair_graph_dump_dir"] = str(dump_dir)
                meta["torchair_graph_dump_path_attrs"] = set_torchair_graph_dump_path(config.debug.graph_dump, dump_dir)
        if torchair_msit_dump_kind != "none":
            meta["torchair_msit_dump"] = apply_msit_torchair_dump_config(
                config,
                kind=str(torchair_msit_dump_kind),
                dump_dir=torchair_msit_dump_dir,
                dump_mode=str(torchair_msit_dump_mode),
                dump_token=str(torchair_msit_dump_token),
                dump_layer=str(torchair_msit_dump_layer),
                fusion_switch_file=torchair_msit_fusion_switch_file,
            )
        return torchair.get_npu_backend(compiler_config=config), meta
    return name, {"backend_kind": str(name)}


def maybe_compile_static_visual(
    *,
    model: LocalPaddleOCRVLForConditionalGeneration,
    item: PrefillInput,
    device: torch.device,
    backend_name: str,
    debug_no_padding: bool = False,
    debug_min_pad_tokens: int = 0,
    debug_pad_to_multiple: int = 0,
    torchair_mode: str = "default",
    torchair_run_eagerly: bool = False,
    torchair_graph_dump_type: str = "none",
    torchair_graph_dump_dir: str | Path | None = None,
    torchair_msit_dump_kind: str = "none",
    torchair_msit_dump_dir: str | Path | None = None,
    torchair_msit_dump_mode: str = "output",
    torchair_msit_dump_token: str = "",
    torchair_msit_dump_layer: str = "",
    torchair_msit_fusion_switch_file: str | Path | None = None,
) -> tuple[Callable[[torch.Tensor], torch.Tensor] | None, dict[str, Any]]:
    if backend_name not in VISION_COMPILE_BACKEND_CHOICES:
        raise ValueError(f"unsupported vision compile backend={backend_name!r}; choices={VISION_COMPILE_BACKEND_CHOICES}")
    wrapper = SingleCropStaticVisualModule(
        model,
        item.image_grid_thw,
        device=device,
        debug_no_padding=debug_no_padding,
        debug_min_pad_tokens=debug_min_pad_tokens,
        debug_pad_to_multiple=debug_pad_to_multiple,
    ).eval()
    mask_shape = (
        None
        if wrapper.static_pad_attention_mask is None
        else [int(dim) for dim in wrapper.static_pad_attention_mask.shape]
    )
    meta: dict[str, Any] = {
        "candidate_vision_path": "static_visual",
        "backend": str(backend_name),
        "static_visual_pad_policy": str(wrapper.static_visual_pad_policy),
        "debug_static_visual_no_padding": bool(wrapper.debug_no_padding),
        "debug_static_visual_min_pad_tokens": int(wrapper.debug_min_pad_tokens),
        "debug_static_visual_pad_to_multiple": int(wrapper.debug_pad_to_multiple),
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
        "static_pad_attention_mask_shape": mask_shape,
        "static_pad_attention_mask_enabled": bool(wrapper.static_pad_attention_mask is not None),
        "static_visual_encoder_path": "single_static_path_masked_padding_default",
    }
    if backend_name == "none":
        meta.update({"enabled": False, "compile_api": None})
        return wrapper, meta

    import torch._dynamo

    old_capture_scalar_outputs = bool(torch._dynamo.config.capture_scalar_outputs)
    torch._dynamo.config.capture_scalar_outputs = True
    compile_kwargs: dict[str, Any] = {"fullgraph": True, "dynamic": False}
    torch._dynamo.reset()
    backend, backend_meta = vision_compile_backend(
        backend_name,
        device,
        torchair_mode=torchair_mode,
        torchair_run_eagerly=torchair_run_eagerly,
        torchair_graph_dump_type=torchair_graph_dump_type,
        torchair_graph_dump_dir=torchair_graph_dump_dir,
        torchair_msit_dump_kind=torchair_msit_dump_kind,
        torchair_msit_dump_dir=torchair_msit_dump_dir,
        torchair_msit_dump_mode=torchair_msit_dump_mode,
        torchair_msit_dump_token=torchair_msit_dump_token,
        torchair_msit_dump_layer=torchair_msit_dump_layer,
        torchair_msit_fusion_switch_file=torchair_msit_fusion_switch_file,
    )
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
            "dynamo_reset_before_compile": True,
            "compile_backend_meta": backend_meta,
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
    vision_forward, meta = maybe_compile_static_visual(
        model=model,
        item=item,
        device=device,
        backend_name=str(args.vision_compile_backend),
        debug_no_padding=bool(args.debug_static_visual_no_padding),
        debug_min_pad_tokens=int(args.debug_static_visual_min_pad_tokens),
        debug_pad_to_multiple=int(args.debug_static_visual_pad_to_multiple),
        torchair_mode=str(getattr(args, "torchair_mode", "default")),
        torchair_run_eagerly=bool(getattr(args, "torchair_run_eagerly", False)),
        torchair_graph_dump_type=str(getattr(args, "torchair_graph_dump_type", "none")),
        torchair_graph_dump_dir=getattr(args, "torchair_graph_dump_dir", None),
        torchair_msit_dump_kind=str(getattr(args, "torchair_msit_dump_kind", "none")),
        torchair_msit_dump_dir=getattr(args, "torchair_msit_dump_dir", None),
        torchair_msit_dump_mode=str(getattr(args, "torchair_msit_dump_mode", "output")),
        torchair_msit_dump_token=str(getattr(args, "torchair_msit_dump_token", "")),
        torchair_msit_dump_layer=str(getattr(args, "torchair_msit_dump_layer", "")),
        torchair_msit_fusion_switch_file=getattr(args, "torchair_msit_fusion_switch_file", None),
    )
    if vision_forward is None:
        raise RuntimeError("static_visual candidate did not produce a callable vision_forward")
    if str(args.vision_compile_backend) != "none":
        pixel_values = item.pixel_values.to(device=device, dtype=model.visual.dtype)
        static_eager_output = None
        if bool(args.validate_compiled_against_static_eager):
            static_eager = SingleCropStaticVisualModule(
                model,
                item.image_grid_thw,
                device=device,
                debug_no_padding=bool(args.debug_static_visual_no_padding),
                debug_min_pad_tokens=int(args.debug_static_visual_min_pad_tokens),
                debug_pad_to_multiple=int(args.debug_static_visual_pad_to_multiple),
            ).eval()
            maybe_sync(device)
            eager_start = time.perf_counter()
            static_eager_output = static_eager(pixel_values)
            maybe_sync(device)
            meta["static_eager_validation_s"] = float(time.perf_counter() - eager_start)
        maybe_sync(device)
        start = time.perf_counter()
        first_output = vision_forward(pixel_values)
        maybe_sync(device)
        if bool(meta.get("capture_scalar_outputs", False)):
            import torch._dynamo

            torch._dynamo.config.capture_scalar_outputs = bool(
                meta.get("capture_scalar_outputs_previous", False)
            )
            meta["capture_scalar_outputs_restored_after_first_call"] = True
        first_real_output = slice_visual_features_to_real(first_output, item.image_grid_thw)
        meta["compiled_first_call_s"] = float(time.perf_counter() - start)
        meta["first_output_shape"] = [int(dim) for dim in first_output.shape]
        meta["first_output_nonfinite_count"] = int((~torch.isfinite(first_output.float())).sum().item())
        meta["first_real_output_shape"] = [int(dim) for dim in first_real_output.shape]
        meta["first_real_output_nonfinite_count"] = int((~torch.isfinite(first_real_output.float())).sum().item())
        if static_eager_output is not None:
            static_eager_real_output = slice_visual_features_to_real(static_eager_output, item.image_grid_thw)
            meta["compiled_vs_static_eager_validation"] = {
                "enabled": True,
                "same_wrapper_class": True,
                "separate_wrapper_instance": True,
                "physical": diff_stats(first_output.cpu(), static_eager_output.cpu()),
                "real_rows": diff_stats(first_real_output.cpu(), static_eager_real_output.cpu()),
                "static_eager_output_shape": [int(dim) for dim in static_eager_output.shape],
                "static_eager_real_output_shape": [int(dim) for dim in static_eager_real_output.shape],
                "static_eager_output_nonfinite_count": int((~torch.isfinite(static_eager_output.float())).sum().item()),
                "static_eager_real_output_nonfinite_count": int(
                    (~torch.isfinite(static_eager_real_output.float())).sum().item()
                ),
            }
        else:
            meta["compiled_vs_static_eager_validation"] = {"enabled": False}
        backend_meta = meta.get("compile_backend_meta", {})
        if isinstance(backend_meta, dict) and backend_meta.get("torchair_graph_dump_dir"):
            meta["torchair_graph_dump_summary_after_first_call"] = summarize_tree(
                str(backend_meta["torchair_graph_dump_dir"])
            )
        msit_dump_meta = backend_meta.get("torchair_msit_dump", {}) if isinstance(backend_meta, dict) else {}
        if isinstance(msit_dump_meta, dict) and msit_dump_meta.get("enabled"):
            meta["torchair_msit_dump_summary_after_first_call"] = summarize_tree(
                str(msit_dump_meta.get("expected_dump_dir", ""))
            )
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
def compute_visual_tower_only(
    *,
    model: LocalPaddleOCRVLForConditionalGeneration,
    item: PrefillInput,
    device: torch.device,
    vision_forward: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    pixel_values = item.pixel_values.to(device=device, dtype=model.visual.dtype)
    image_grid_thw = item.image_grid_thw
    cu_seqlens = None
    if vision_forward is None:
        cu_seqlens = build_vision_cu_seqlens(image_grid_thw, device=device)

    maybe_sync(device)
    start = time.perf_counter()
    if vision_forward is None:
        visual_features = model.visual(
            pixel_values=pixel_values.unsqueeze(0),
            image_grid_thw=image_grid_thw,
            cu_seqlens=cu_seqlens,
        )
    else:
        visual_features = vision_forward(pixel_values)
    maybe_sync(device)
    timing = {
        "visual_tower_e2e_s": float(time.perf_counter() - start),
        "visual_tower_physical_output_seq_len": float(int(visual_features.shape[0])),
        "visual_tower_real_output_seq_len": float(int(image_grid_thw.prod().item())),
    }
    return slice_visual_features_to_real(visual_features, image_grid_thw).detach(), timing


@torch.inference_mode()
def compute_prefill_tail_from_visual_features(
    *,
    model: LocalPaddleOCRVLForConditionalGeneration,
    item: PrefillInput,
    device: torch.device,
    cache_length: int,
    visual_features: torch.Tensor,
) -> dict[str, torch.Tensor]:
    input_ids = item.input_ids.to(device)
    attention_mask = item.attention_mask.to(device)
    image_grid_thw = item.image_grid_thw
    if int(input_ids.shape[1]) > int(cache_length):
        raise ValueError(
            f"input length {int(input_ids.shape[1])} exceeds cache_length={int(cache_length)} for item {item.entry.get('id')}"
        )
    image_embeds = model.mlp_AR(visual_features, image_grid_thw)
    inputs_embeds = model.model.embed_tokens(input_ids)
    inputs_embeds = scatter_projected_image_embeds(
        model=model,
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        image_embeds=image_embeds,
    )
    position_ids_cpu, _rope_deltas_cpu = model.get_rope_index(item.input_ids, item.image_grid_thw, item.attention_mask)
    position_ids = position_ids_cpu.to(device)
    cache = model.allocate_static_cache(
        batch_size=int(inputs_embeds.shape[0]),
        cache_length=int(cache_length),
        device=inputs_embeds.device,
        dtype=inputs_embeds.dtype,
        init_mode="zeros",
    )
    hidden_states = model.model.forward_prefill_static(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        cache=cache,
    )
    prefill_logits = model.lm_head(hidden_states[:, -1:, :])
    return {
        "visual_features": visual_features.detach(),
        "image_embeds": image_embeds.detach(),
        "prefill_logits": prefill_logits.detach(),
        "prefill_hidden_last": hidden_states[:, -1:, :].detach(),
        "input_ids": input_ids.detach(),
        "attention_mask": attention_mask.detach(),
        "image_grid_thw": image_grid_thw.detach(),
        "prefill_argmax": torch.argmax(prefill_logits[:, -1, :].float(), dim=-1).detach(),
    }


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
        visual_features = measure(
            "visual_features",
            lambda: slice_visual_features_to_real(vision_forward(pixel_values), image_grid_thw),
        )
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
    visual_features, timing = compute_visual_tower_only(
        model=model,
        item=item,
        device=device,
        vision_forward=vision_forward,
    )
    tensors = compute_prefill_tail_from_visual_features(
        model=model,
        item=item,
        device=device,
        cache_length=cache_length,
        visual_features=visual_features,
    )
    maybe_sync(device)
    full_start = time.perf_counter()
    full_tensors, _timing = compute_vision_prefill(
        model=model,
        item=item,
        device=device,
        cache_length=cache_length,
        sync=False,
        record_phase_timings=False,
        vision_forward=vision_forward,
    )
    maybe_sync(device)
    timing["full_prefill_e2e_s"] = float(time.perf_counter() - full_start)
    timing["vision_tower_vs_full_prefill_visual_features_max_abs_diff"] = float(
        torch.max(torch.abs(tensors["visual_features"].float() - full_tensors["visual_features"].float())).item()
    )
    timing["vision_tower_vs_full_prefill_prefill_logits_max_abs_diff"] = float(
        torch.max(torch.abs(tensors["prefill_logits"].float() - full_tensors["prefill_logits"].float())).item()
    )
    return tensors, timing


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


def tensor_probe_summary(tensor: torch.Tensor) -> dict[str, Any]:
    detached = tensor.detach()
    finite = torch.isfinite(detached.float())
    finite_values = detached.float()[finite]
    summary = tensor_summary(detached)
    summary.update(
        {
            "stride": [int(dim) for dim in detached.stride()],
            "is_contiguous": bool(detached.is_contiguous()),
            "device": str(detached.device),
        }
    )
    if finite_values.numel() > 0:
        summary.update(
            {
                "finite_min": float(finite_values.min().item()),
                "finite_max": float(finite_values.max().item()),
                "finite_mean_abs": float(finite_values.abs().mean().item()),
            }
        )
    return summary


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


def aggregate_timed_token_rate(rows: list[dict[str, Any]], *, token_key: str, time_key: str) -> dict[str, Any]:
    token_total = 0.0
    time_total = 0.0
    repeat_count = 0
    item_count = 0
    for row in rows:
        timing_stats = row.get("timing_s", {}).get(time_key, {})
        count = int(timing_stats.get("count", 0) or 0)
        time_sum = float(timing_stats.get("sum", 0.0) or 0.0)
        if count <= 0 or time_sum <= 0.0:
            continue
        token_total += float(row[token_key]) * float(count)
        time_total += time_sum
        repeat_count += count
        item_count += 1
    return {
        "item_count": int(item_count),
        "repeat_count": int(repeat_count),
        "tokens": float(token_total),
        "time_s": float(time_total),
        "tokens_per_s": (float(token_total) / float(time_total)) if time_total > 0.0 else None,
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


def manual_attention_bnsd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    scores = torch.matmul(q, k.transpose(-2, -1)) * float(scale)
    if mask is not None:
        scores = scores.masked_fill(mask.to(torch.bool), torch.finfo(scores.dtype).min)
    probs = attention_softmax(scores, dim=-1, output_dtype=q.dtype, mode="fp32")
    return torch.matmul(probs, v)


def parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for chunk in str(raw).replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(int(chunk))
    if not values:
        raise ValueError(f"expected at least one integer, got {raw!r}")
    return values


def parse_float_list(raw: str) -> list[float]:
    values: list[float] = []
    for chunk in str(raw).replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(float(chunk))
    if not values:
        raise ValueError(f"expected at least one float, got {raw!r}")
    return values


class LayerNormOnlyModule(torch.nn.Module):
    """Small module for isolating LayerNorm eager-vs-compiled behavior."""

    def __init__(
        self,
        *,
        hidden_size: int,
        eps: float,
        impl: str,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ):
        super().__init__()
        if impl not in LAYERNORM_PROBE_IMPL_CHOICES:
            raise ValueError(f"unsupported layernorm impl={impl!r}")
        self.hidden_size = int(hidden_size)
        self.eps = float(eps)
        self.impl = str(impl)
        if self.impl == "nn":
            self.layer_norm = torch.nn.LayerNorm(self.hidden_size, eps=self.eps, elementwise_affine=True).to(
                device=weight.device,
                dtype=weight.dtype,
            )
            with torch.no_grad():
                self.layer_norm.weight.copy_(weight)
                self.layer_norm.bias.copy_(bias)
        else:
            self.register_buffer("weight", weight.detach().clone().contiguous(), persistent=False)
            self.register_buffer("bias", bias.detach().clone().contiguous(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.impl == "nn":
            return self.layer_norm(x)
        if self.impl == "functional":
            return F.layer_norm(x, (self.hidden_size,), self.weight, self.bias, self.eps)
        if self.impl == "manual":
            x_f = x.float()
            mean = x_f.mean(dim=-1, keepdim=True)
            centered = x_f - mean
            variance = (centered * centered).mean(dim=-1, keepdim=True)
            normalized = centered * torch.rsqrt(variance + self.eps)
            return normalized.to(dtype=x.dtype) * self.weight + self.bias
        if self.impl == "manual_fp16_reduce":
            # Deliberately keep the reduction in input dtype. This is a diagnostic
            # lower-precision path for the fp16 accumulation/overflow hypothesis.
            reduce_dtype = x.dtype
            count = x.shape[-1]
            mean = x.sum(dim=-1, keepdim=True, dtype=reduce_dtype) / count
            centered = x - mean
            variance = (centered * centered).sum(dim=-1, keepdim=True, dtype=reduce_dtype) / count
            eps = torch.tensor(self.eps, device=x.device, dtype=reduce_dtype)
            normalized = centered * torch.rsqrt(variance + eps)
            return normalized * self.weight + self.bias
        if self.impl == "npu_eval":
            import torch_npu

            return torch_npu.npu_layer_norm_eval(x, [self.hidden_size], self.weight, self.bias, self.eps)
        raise RuntimeError(f"unreachable layernorm impl={self.impl!r}")


def build_synthetic_layernorm_case(
    *,
    seq_len: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    input_scale: float,
    affine: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + int(seq_len) * 17 + int(hidden_size))
    x = torch.randn((int(seq_len), int(hidden_size)), generator=generator, dtype=dtype) * float(input_scale)
    if affine == "identity":
        weight = torch.ones((int(hidden_size),), dtype=dtype)
        bias = torch.zeros((int(hidden_size),), dtype=dtype)
    elif affine == "random":
        weight = torch.randn((int(hidden_size),), generator=generator, dtype=dtype) * 0.05 + 1.0
        bias = torch.randn((int(hidden_size),), generator=generator, dtype=dtype) * 0.02
    else:
        raise ValueError(f"unsupported --synthetic-affine {affine!r}")
    return (
        x.to(device=device),
        weight.to(device=device),
        bias.to(device=device),
        {
            "source": "synthetic",
            "seq_len": int(seq_len),
            "hidden_size": int(hidden_size),
            "input_scale": float(input_scale),
            "affine": str(affine),
        },
    )


@torch.inference_mode()
def build_real_first_layernorm_case(
    *,
    args: argparse.Namespace,
    model: LocalPaddleOCRVLForConditionalGeneration,
    model_dir: Path,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, dict[str, Any]]:
    manifest_path = Path(args.baseline).expanduser().resolve()
    manifest = load_baseline_manifest(manifest_path)
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    dataset_dir = resolve_dataset_dir(args.dataset_dir or manifest["build_summary"]["page"]["dataset_dir"])
    inputs = build_inputs_from_manifest(manifest=manifest, model_dir=model_dir, tokenizer=tokenizer, dataset_dir=dataset_dir)
    if not inputs:
        raise ValueError(f"baseline manifest has no inputs: {manifest_path}")
    item_index = int(args.real_item_index)
    if item_index < 0 or item_index >= len(inputs):
        raise IndexError(f"--real-item-index {item_index} out of range for {len(inputs)} baseline items")
    item = inputs[item_index]
    wrapper = SingleCropStaticVisualModule(
        model,
        item.image_grid_thw,
        device=device,
        debug_no_padding=bool(args.debug_static_visual_no_padding),
        debug_min_pad_tokens=int(args.debug_static_visual_min_pad_tokens),
        debug_pad_to_multiple=int(args.debug_static_visual_pad_to_multiple),
    ).eval()
    transformer = model.visual.vision_model
    embeddings_module = transformer.embeddings
    pixel_values = item.pixel_values.to(device=device, dtype=embeddings_module.patch_embedding.weight.dtype)
    hidden_states = embeddings_module.patch_embedding(pixel_values).flatten(-2).squeeze(-1)
    if wrapper.static_pad_tokens:
        hidden_states = torch.cat(
            [
                hidden_states,
                torch.zeros(
                    wrapper.static_pad_tokens,
                    hidden_states.shape[-1],
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                ),
            ],
            dim=0,
        )
    hidden_states = (hidden_states + wrapper.abs_pos_embed_const).contiguous()
    layer_norm = transformer.encoder.layers[0].layer_norm1
    return (
        hidden_states.detach(),
        layer_norm.weight.detach().to(device=device, dtype=hidden_states.dtype),
        layer_norm.bias.detach().to(device=device, dtype=hidden_states.dtype),
        float(layer_norm.eps),
        {
            "source": "real_static_visual_layer0_layer_norm1_input",
            "item_index": int(item_index),
            "id": str(item.entry.get("id")),
            "layout_label": str(item.entry.get("layout_label")),
            "image_grid_thw": [int(value) for value in item.image_grid_thw.flatten().tolist()],
            "real_seq_len": int(wrapper.static_real_seq_len),
            "pad_tokens": int(wrapper.static_pad_tokens),
            "physical_seq_len": int(wrapper.static_physical_seq_len),
            "physical_seq_mod16": int(wrapper.static_physical_seq_len % 16),
            "physical_seq_mod128": int(wrapper.static_physical_seq_len % 128),
            "static_visual_pad_policy": str(wrapper.static_visual_pad_policy),
            "debug_static_visual_no_padding": bool(wrapper.debug_no_padding),
            "debug_static_visual_min_pad_tokens": int(wrapper.debug_min_pad_tokens),
            "debug_static_visual_pad_to_multiple": int(wrapper.debug_pad_to_multiple),
        },
    )


def layernorm_compiled_matches_eager(row: dict[str, Any]) -> bool:
    diff = row.get("compiled_second_vs_eager_before", {})
    return bool(
        diff.get("shape_match")
        and diff.get("allclose_atol_5e_2_rtol_5e_2")
        and diff.get("lhs_nonfinite_count") == 0
        and diff.get("rhs_nonfinite_count") == 0
    )


@torch.inference_mode()
def probe_layernorm_compile(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype)
    configure_npu_jit_compile(args.npu_jit_compile, device)
    backend_name = str(args.vision_compile_backend)
    impls = [impl.strip() for impl in str(args.impls).split(",") if impl.strip()]
    if not impls:
        raise ValueError("--impls must include at least one implementation")
    for impl in impls:
        if impl not in LAYERNORM_PROBE_IMPL_CHOICES:
            raise ValueError(f"unsupported --impls entry {impl!r}; choices={LAYERNORM_PROBE_IMPL_CHOICES}")
        if impl == "npu_eval" and device.type != "npu":
            raise ValueError("--impls npu_eval requires --device npu:0")

    backend = None
    backend_meta: dict[str, Any] = {"backend_kind": "none"}
    compile_kwargs: dict[str, Any] = {"fullgraph": True, "dynamic": False}
    old_capture_scalar_outputs: bool | None = None
    if backend_name != "none":
        if backend_name == "torchair" and device.type != "npu":
            raise ValueError("--vision-compile-backend torchair requires --device npu:0")
        import torch._dynamo

        old_capture_scalar_outputs = bool(torch._dynamo.config.capture_scalar_outputs)
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.reset()
        backend, backend_meta = vision_compile_backend(
            backend_name,
            device,
            torchair_mode=str(getattr(args, "torchair_mode", "default")),
            torchair_run_eagerly=bool(getattr(args, "torchair_run_eagerly", False)),
            torchair_graph_dump_type=str(getattr(args, "torchair_graph_dump_type", "none")),
            torchair_graph_dump_dir=getattr(args, "torchair_graph_dump_dir", None),
            torchair_msit_dump_kind=str(getattr(args, "torchair_msit_dump_kind", "none")),
            torchair_msit_dump_dir=getattr(args, "torchair_msit_dump_dir", None),
            torchair_msit_dump_mode=str(getattr(args, "torchair_msit_dump_mode", "output")),
            torchair_msit_dump_token=str(getattr(args, "torchair_msit_dump_token", "")),
            torchair_msit_dump_layer=str(getattr(args, "torchair_msit_dump_layer", "")),
            torchair_msit_fusion_switch_file=getattr(args, "torchair_msit_fusion_switch_file", None),
        )
        if backend is not None:
            compile_kwargs["backend"] = backend

    cases: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, dict[str, Any]]] = []
    for seq_len in parse_int_list(args.seq_lens):
        for input_scale in parse_float_list(args.synthetic_input_scales):
            x, weight, bias, case_meta = build_synthetic_layernorm_case(
                seq_len=int(seq_len),
                hidden_size=int(args.hidden_size),
                dtype=dtype,
                device=device,
                seed=int(args.seed),
                input_scale=float(input_scale),
                affine=str(args.synthetic_affine),
            )
            cases.append((x, weight, bias, float(args.eps), case_meta))

    model_dir: Path | None = None
    if bool(args.include_real_first_crop):
        model, model_dir, loaded_device, loaded_dtype = load_model_for_args(args)
        if loaded_device != device or loaded_dtype != dtype:
            raise RuntimeError("internal device/dtype mismatch while loading real LayerNorm probe model")
        cases.append(
            build_real_first_layernorm_case(
                args=args,
                model=model,
                model_dir=model_dir,
                device=device,
            )
        )

    rows: list[dict[str, Any]] = []
    try:
        for case_idx, (x, weight, bias, eps, case_meta) in enumerate(cases):
            input_summary = tensor_summary(x.detach().cpu())
            for impl in impls:
                row: dict[str, Any] = {
                    "case_index": int(case_idx),
                    "impl": str(impl),
                    "backend": str(backend_name),
                    "device": str(device),
                    "dtype": str(dtype),
                    "eps": float(eps),
                    "input_shape": [int(dim) for dim in x.shape],
                    "input_numel": int(x.numel()),
                    "input_summary": input_summary,
                    "case": case_meta,
                }
                try:
                    module = LayerNormOnlyModule(
                        hidden_size=int(x.shape[-1]),
                        eps=float(eps),
                        impl=str(impl),
                        weight=weight,
                        bias=bias,
                    ).eval()
                    maybe_sync(device)
                    start = time.perf_counter()
                    eager_before = module(x)
                    maybe_sync(device)
                    eager_before_s = float(time.perf_counter() - start)
                    row["eager_before_s"] = eager_before_s
                    row["eager_before_summary"] = tensor_summary(eager_before.detach().cpu())

                    if backend_name == "none":
                        row.update(
                            {
                                "ok": True,
                                "compiled": False,
                                "compiled_second_matches_eager": None,
                                "eager_after_vs_eager_before": diff_stats(eager_before.cpu(), eager_before.cpu()),
                            }
                        )
                    else:
                        import torch._dynamo

                        torch._dynamo.reset()
                        maybe_sync(device)
                        start = time.perf_counter()
                        compiled = torch.compile(module, **compile_kwargs)
                        maybe_sync(device)
                        compile_wrapper_s = float(time.perf_counter() - start)

                        maybe_sync(device)
                        start = time.perf_counter()
                        compiled_first = compiled(x)
                        maybe_sync(device)
                        compiled_first_s = float(time.perf_counter() - start)

                        maybe_sync(device)
                        start = time.perf_counter()
                        compiled_second = compiled(x)
                        maybe_sync(device)
                        compiled_second_s = float(time.perf_counter() - start)

                        maybe_sync(device)
                        start = time.perf_counter()
                        eager_after = module(x)
                        maybe_sync(device)
                        eager_after_s = float(time.perf_counter() - start)

                        row.update(
                            {
                                "ok": True,
                                "compiled": True,
                                "compile_wrapper_s": compile_wrapper_s,
                                "compiled_first_s": compiled_first_s,
                                "compiled_second_s": compiled_second_s,
                                "eager_after_s": eager_after_s,
                                "compiled_first_summary": tensor_summary(compiled_first.detach().cpu()),
                                "compiled_second_summary": tensor_summary(compiled_second.detach().cpu()),
                                "compiled_first_vs_eager_before": diff_stats(compiled_first.cpu(), eager_before.cpu()),
                                "compiled_second_vs_eager_before": diff_stats(compiled_second.cpu(), eager_before.cpu()),
                                "compiled_first_vs_second": diff_stats(compiled_first.cpu(), compiled_second.cpu()),
                                "eager_after_vs_eager_before": diff_stats(eager_after.cpu(), eager_before.cpu()),
                            }
                        )
                        row["compiled_second_matches_eager"] = layernorm_compiled_matches_eager(row)
                except Exception as exc:
                    row.update({"ok": False, "error": f"{exc.__class__.__name__}: {exc}"})
                rows.append(row)
    finally:
        if old_capture_scalar_outputs is not None:
            import torch._dynamo

            torch._dynamo.config.capture_scalar_outputs = old_capture_scalar_outputs

    ok_rows = [row for row in rows if row.get("ok")]
    compiled_rows = [row for row in ok_rows if row.get("compiled")]
    mismatch_rows = [row for row in compiled_rows if not bool(row.get("compiled_second_matches_eager"))]
    error_rows = [row for row in rows if not row.get("ok")]
    summary = {
        "total_cases": int(len(rows)),
        "ok_cases": int(len(ok_rows)),
        "error_cases": int(len(error_rows)),
        "compiled_cases": int(len(compiled_rows)),
        "compiled_second_matches_eager_count": int(
            sum(bool(row.get("compiled_second_matches_eager")) for row in compiled_rows)
        ),
        "compiled_second_matches_eager_all": bool(
            len(compiled_rows) > 0
            and all(bool(row.get("compiled_second_matches_eager")) for row in compiled_rows)
        ),
        "compiled_nonfinite_case_count": int(
            sum(
                int(row.get("compiled_second_vs_eager_before", {}).get("lhs_nonfinite_count", 0) or 0) > 0
                for row in compiled_rows
            )
        ),
        "failed_case_keys": [
            {
                "case_index": row.get("case_index"),
                "impl": row.get("impl"),
                "source": row.get("case", {}).get("source"),
                "error": row.get("error"),
            }
            for row in error_rows
        ],
        "mismatch_case_keys": [
            {
                "case_index": row.get("case_index"),
                "impl": row.get("impl"),
                "source": row.get("case", {}).get("source"),
                "shape": row.get("input_shape"),
                "max_abs_diff": row.get("compiled_second_vs_eager_before", {}).get("max_abs_diff"),
                "mean_abs_diff": row.get("compiled_second_vs_eager_before", {}).get("mean_abs_diff"),
                "compiled_nonfinite_count": row.get("compiled_second_vs_eager_before", {}).get("lhs_nonfinite_count"),
                "eager_nonfinite_count": row.get("compiled_second_vs_eager_before", {}).get("rhs_nonfinite_count"),
            }
            for row in mismatch_rows
        ],
    }
    output = {
        "schema_version": 1,
        "experiment": "07_vision_prefill_optimization",
        "kind": "layernorm_compile_probe",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "dtype": str(dtype),
        "backend": str(backend_name),
        "compile_api": None if backend_name == "none" else "torch.compile",
        "fullgraph": bool(backend_name != "none"),
        "dynamic": False,
        "capture_scalar_outputs": old_capture_scalar_outputs is not None,
        "capture_scalar_outputs_previous": old_capture_scalar_outputs,
        "dynamo_reset_before_compile": bool(backend_name != "none"),
        "compile_backend_meta": backend_meta,
        "uses_torchair_cache_compile": False,
        "explicit_cache_dir": None,
        "model": None if model_dir is None else str(model_dir),
        "baseline": str(Path(args.baseline).expanduser().resolve()) if bool(args.include_real_first_crop) else None,
        "summary": summary,
        "results": rows,
    }
    output_json = json.dumps(output, indent=2, default=json_default)
    output_path_raw = str(getattr(args, "output", "") or "").strip()
    if output_path_raw:
        output_path = Path(output_path_raw).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_json, encoding="utf-8")
        print(json.dumps({"layernorm_probe_output": str(output_path), "summary": summary}, indent=2, default=json_default), flush=True)
    else:
        print(output_json, flush=True)


def visual_prefix_compiled_matches_eager(row: dict[str, Any]) -> bool:
    diff = row.get("compiled_second_vs_eager_before", {})
    return bool(
        diff.get("shape_match")
        and diff.get("allclose_atol_5e_2_rtol_5e_2")
        and diff.get("lhs_nonfinite_count") == 0
        and diff.get("rhs_nonfinite_count") == 0
    )


def parse_visual_prefix_stages(raw: str) -> list[str]:
    stages = [stage.strip() for stage in str(raw).replace(";", ",").split(",") if stage.strip()]
    if not stages:
        raise ValueError("--stages must contain at least one stage")
    bad = [stage for stage in stages if stage not in VISUAL_PREFIX_STAGE_CHOICES]
    if bad:
        raise ValueError(f"unsupported --stages entries {bad}; choices={VISUAL_PREFIX_STAGE_CHOICES}")
    return stages


def parse_qkv_linear_probe_impls(raw: str) -> list[str]:
    impls = [impl.strip() for impl in str(raw).replace(";", ",").split(",") if impl.strip()]
    if not impls:
        raise ValueError("--impls must contain at least one implementation")
    bad = [impl for impl in impls if impl not in QKV_LINEAR_PROBE_IMPL_CHOICES]
    if bad:
        raise ValueError(f"unsupported --impls entries {bad}; choices={QKV_LINEAR_PROBE_IMPL_CHOICES}")
    return impls


def parse_qkv_linear_probe_sources(raw: str) -> list[str]:
    sources = [source.strip() for source in str(raw).replace(";", ",").split(",") if source.strip()]
    if not sources:
        raise ValueError("--sources must contain at least one source")
    bad = [source for source in sources if source not in QKV_LINEAR_PROBE_SOURCE_CHOICES]
    if bad:
        raise ValueError(f"unsupported --sources entries {bad}; choices={QKV_LINEAR_PROBE_SOURCE_CHOICES}")
    return sources


def parse_qkv_linear_probe_ln_impls(raw: str) -> list[str]:
    impls = [impl.strip() for impl in str(raw).replace(";", ",").split(",") if impl.strip()]
    if not impls:
        raise ValueError("--ln-impls must contain at least one implementation")
    bad = [impl for impl in impls if impl not in QKV_LINEAR_PROBE_LN_IMPL_CHOICES]
    if bad:
        raise ValueError(f"unsupported --ln-impls entries {bad}; choices={QKV_LINEAR_PROBE_LN_IMPL_CHOICES}")
    return impls


def parse_qkv_linear_probe_bridges(raw: str) -> list[str]:
    bridges = [bridge.strip() for bridge in str(raw).replace(";", ",").split(",") if bridge.strip()]
    if not bridges:
        raise ValueError("--bridges must contain at least one bridge")
    bad = [bridge for bridge in bridges if bridge not in QKV_LINEAR_PROBE_BRIDGE_CHOICES]
    if bad:
        raise ValueError(f"unsupported --bridges entries {bad}; choices={QKV_LINEAR_PROBE_BRIDGE_CHOICES}")
    return bridges


def qkv_linear_compiled_matches_eager(row: dict[str, Any]) -> bool:
    diff = row.get("compiled_second_vs_eager_before", {})
    return bool(
        diff.get("shape_match")
        and diff.get("allclose_atol_5e_2_rtol_5e_2")
        and diff.get("lhs_nonfinite_count") == 0
        and diff.get("rhs_nonfinite_count") == 0
    )


def configure_npu_mm_bmm_format_nd(mode: str, device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {"mode": str(mode), "applied": False}
    if str(mode) == "default":
        return info
    if device.type != "npu":
        return {**info, "error": "NPU-only option"}
    import torch_npu

    requested = str(mode) == "enable"
    option = {"MM_BMM_ND_ENABLE": "enable" if requested else "disable"}
    setter = getattr(torch_npu.npu, "set_option", None)
    if setter is None:
        raise RuntimeError("torch_npu.npu.set_option is not available; cannot set MM_BMM_ND_ENABLE")
    before = None
    getter = getattr(torch_npu.npu, "get_mm_bmm_format_nd", None)
    if getter is not None:
        try:
            before = bool(getter())
        except Exception:
            before = None
    setter(option)
    after = None
    if getter is not None:
        try:
            after = bool(getter())
        except Exception:
            after = None
    info.update({"applied": True, "requested": requested, "option": option, "before": before, "after": after})
    return info


@torch.inference_mode()
def probe_visual_prefix_compile(args: argparse.Namespace) -> None:
    apply_runtime_env(args)
    model, model_dir, device, dtype = load_model_for_args(args)
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    baseline_path = Path(args.baseline).expanduser().resolve()
    manifest = load_baseline_manifest(baseline_path)
    dataset_dir = resolve_dataset_dir(args.dataset_dir or manifest["build_summary"]["page"]["dataset_dir"])
    inputs = build_inputs_from_manifest(manifest=manifest, model_dir=model_dir, tokenizer=tokenizer, dataset_dir=dataset_dir)
    max_items = int(args.max_items)
    if max_items > 0:
        inputs = inputs[:max_items]
    if not inputs:
        raise ValueError(f"no inputs available from baseline {baseline_path}")

    stages = parse_visual_prefix_stages(args.stages)
    backend_name = str(args.vision_compile_backend)
    backend = None
    backend_meta: dict[str, Any] = {"backend_kind": "none"}
    compile_kwargs: dict[str, Any] = {"fullgraph": True, "dynamic": False}
    old_capture_scalar_outputs: bool | None = None
    if backend_name != "none":
        import torch._dynamo

        old_capture_scalar_outputs = bool(torch._dynamo.config.capture_scalar_outputs)
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.reset()
        backend, backend_meta = vision_compile_backend(
            backend_name,
            device,
            torchair_mode=str(getattr(args, "torchair_mode", "default")),
            torchair_run_eagerly=bool(getattr(args, "torchair_run_eagerly", False)),
            torchair_graph_dump_type=str(getattr(args, "torchair_graph_dump_type", "none")),
            torchair_graph_dump_dir=getattr(args, "torchair_graph_dump_dir", None),
            torchair_msit_dump_kind=str(getattr(args, "torchair_msit_dump_kind", "none")),
            torchair_msit_dump_dir=getattr(args, "torchair_msit_dump_dir", None),
            torchair_msit_dump_mode=str(getattr(args, "torchair_msit_dump_mode", "output")),
            torchair_msit_dump_token=str(getattr(args, "torchair_msit_dump_token", "")),
            torchair_msit_dump_layer=str(getattr(args, "torchair_msit_dump_layer", "")),
            torchair_msit_fusion_switch_file=getattr(args, "torchair_msit_fusion_switch_file", None),
        )
        if backend is not None:
            compile_kwargs["backend"] = backend

    rows: list[dict[str, Any]] = []
    try:
        for item_index, item in enumerate(inputs):
            pixel_values = item.pixel_values.to(device=device, dtype=model.visual.dtype)
            for stage in stages:
                row: dict[str, Any] = {
                    "item_index": int(item_index),
                    "id": str(item.entry.get("id")),
                    "layout_label": str(item.entry.get("layout_label")),
                    "stage": str(stage),
                    "backend": str(backend_name),
                    "device": str(device),
                    "dtype": str(dtype),
                    "input_pixel_values_shape": [int(dim) for dim in pixel_values.shape],
                    "image_grid_thw": [int(value) for value in item.image_grid_thw.flatten().tolist()],
                }
                try:
                    module = SingleCropStaticVisualPrefixModule(
                        model,
                        item.image_grid_thw,
                        device=device,
                        stage=str(stage),
                        debug_no_padding=bool(args.debug_static_visual_no_padding),
                        debug_min_pad_tokens=int(args.debug_static_visual_min_pad_tokens),
                        debug_pad_to_multiple=int(args.debug_static_visual_pad_to_multiple),
                    ).eval()
                    row.update(
                        {
                            "static_real_seq_len": int(module.static_real_seq_len),
                            "static_pad_tokens": int(module.static_pad_tokens),
                            "static_physical_seq_len": int(module.static_physical_seq_len),
                            "static_physical_seq_mod16": int(module.static_physical_seq_len % 16),
                            "static_physical_seq_mod128": int(module.static_physical_seq_len % 128),
                        }
                    )
                    maybe_sync(device)
                    start = time.perf_counter()
                    eager_before = module(pixel_values)
                    maybe_sync(device)
                    eager_before_s = float(time.perf_counter() - start)
                    row["eager_before_s"] = eager_before_s
                    row["eager_before_summary"] = tensor_summary(eager_before.detach().cpu())

                    if backend_name == "none":
                        row.update(
                            {
                                "ok": True,
                                "compiled": False,
                                "compiled_second_matches_eager": None,
                                "eager_after_vs_eager_before": diff_stats(eager_before.cpu(), eager_before.cpu()),
                            }
                        )
                    else:
                        import torch._dynamo

                        torch._dynamo.reset()
                        maybe_sync(device)
                        start = time.perf_counter()
                        compiled = torch.compile(module, **compile_kwargs)
                        maybe_sync(device)
                        compile_wrapper_s = float(time.perf_counter() - start)

                        maybe_sync(device)
                        start = time.perf_counter()
                        compiled_first = compiled(pixel_values)
                        maybe_sync(device)
                        compiled_first_s = float(time.perf_counter() - start)

                        maybe_sync(device)
                        start = time.perf_counter()
                        compiled_second = compiled(pixel_values)
                        maybe_sync(device)
                        compiled_second_s = float(time.perf_counter() - start)

                        maybe_sync(device)
                        start = time.perf_counter()
                        eager_after = module(pixel_values)
                        maybe_sync(device)
                        eager_after_s = float(time.perf_counter() - start)
                        row.update(
                            {
                                "ok": True,
                                "compiled": True,
                                "compile_wrapper_s": compile_wrapper_s,
                                "compiled_first_s": compiled_first_s,
                                "compiled_second_s": compiled_second_s,
                                "eager_after_s": eager_after_s,
                                "compiled_first_summary": tensor_summary(compiled_first.detach().cpu()),
                                "compiled_second_summary": tensor_summary(compiled_second.detach().cpu()),
                                "compiled_first_vs_eager_before": diff_stats(compiled_first.cpu(), eager_before.cpu()),
                                "compiled_second_vs_eager_before": diff_stats(compiled_second.cpu(), eager_before.cpu()),
                                "compiled_first_vs_second": diff_stats(compiled_first.cpu(), compiled_second.cpu()),
                                "eager_after_vs_eager_before": diff_stats(eager_after.cpu(), eager_before.cpu()),
                            }
                        )
                        row["compiled_second_matches_eager"] = visual_prefix_compiled_matches_eager(row)
                except Exception as exc:
                    row.update({"ok": False, "error": f"{exc.__class__.__name__}: {exc}"})
                rows.append(row)
                print(
                    f"VISUAL_PREFIX item={item_index} stage={stage} ok={row.get('ok')} "
                    f"match={row.get('compiled_second_matches_eager')} "
                    f"max_abs={row.get('compiled_second_vs_eager_before', {}).get('max_abs_diff')}",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        if old_capture_scalar_outputs is not None:
            import torch._dynamo

            torch._dynamo.config.capture_scalar_outputs = old_capture_scalar_outputs

    ok_rows = [row for row in rows if row.get("ok")]
    compiled_rows = [row for row in ok_rows if row.get("compiled")]
    mismatch_rows = [row for row in compiled_rows if not bool(row.get("compiled_second_matches_eager"))]
    error_rows = [row for row in rows if not row.get("ok")]
    first_mismatch = mismatch_rows[0] if mismatch_rows else None
    summary = {
        "total_cases": int(len(rows)),
        "ok_cases": int(len(ok_rows)),
        "error_cases": int(len(error_rows)),
        "compiled_cases": int(len(compiled_rows)),
        "compiled_second_matches_eager_count": int(
            sum(bool(row.get("compiled_second_matches_eager")) for row in compiled_rows)
        ),
        "compiled_second_matches_eager_all": bool(
            len(compiled_rows) > 0
            and all(bool(row.get("compiled_second_matches_eager")) for row in compiled_rows)
        ),
        "compiled_nonfinite_case_count": int(
            sum(
                int(row.get("compiled_second_vs_eager_before", {}).get("lhs_nonfinite_count", 0) or 0) > 0
                for row in compiled_rows
            )
        ),
        "first_mismatch": None
        if first_mismatch is None
        else {
            "item_index": first_mismatch.get("item_index"),
            "id": first_mismatch.get("id"),
            "stage": first_mismatch.get("stage"),
            "shape": first_mismatch.get("compiled_second_vs_eager_before", {}).get("shape"),
            "max_abs_diff": first_mismatch.get("compiled_second_vs_eager_before", {}).get("max_abs_diff"),
            "mean_abs_diff": first_mismatch.get("compiled_second_vs_eager_before", {}).get("mean_abs_diff"),
            "compiled_nonfinite_count": first_mismatch.get("compiled_second_vs_eager_before", {}).get(
                "lhs_nonfinite_count"
            ),
            "eager_nonfinite_count": first_mismatch.get("compiled_second_vs_eager_before", {}).get(
                "rhs_nonfinite_count"
            ),
        },
        "mismatch_case_keys": [
            {
                "item_index": row.get("item_index"),
                "id": row.get("id"),
                "stage": row.get("stage"),
                "max_abs_diff": row.get("compiled_second_vs_eager_before", {}).get("max_abs_diff"),
                "mean_abs_diff": row.get("compiled_second_vs_eager_before", {}).get("mean_abs_diff"),
                "compiled_nonfinite_count": row.get("compiled_second_vs_eager_before", {}).get("lhs_nonfinite_count"),
            }
            for row in mismatch_rows
        ],
        "failed_case_keys": [
            {
                "item_index": row.get("item_index"),
                "id": row.get("id"),
                "stage": row.get("stage"),
                "error": row.get("error"),
            }
            for row in error_rows
        ],
    }
    output = {
        "schema_version": 1,
        "experiment": "07_vision_prefill_optimization",
        "kind": "visual_prefix_compile_probe",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "dtype": str(dtype),
        "model": str(model_dir),
        "baseline": str(baseline_path),
        "backend": str(backend_name),
        "compile_api": None if backend_name == "none" else "torch.compile",
        "fullgraph": bool(backend_name != "none"),
        "dynamic": False,
        "capture_scalar_outputs": old_capture_scalar_outputs is not None,
        "capture_scalar_outputs_previous": old_capture_scalar_outputs,
        "dynamo_reset_before_compile": bool(backend_name != "none"),
        "compile_backend_meta": backend_meta,
        "stages": stages,
        "summary": summary,
        "results": rows,
    }
    output_json = json.dumps(output, indent=2, default=json_default)
    output_path_raw = str(getattr(args, "output", "") or "").strip()
    if output_path_raw:
        output_path = Path(output_path_raw).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_json, encoding="utf-8")
        print(json.dumps({"visual_prefix_probe_output": str(output_path), "summary": summary}, indent=2, default=json_default), flush=True)
    else:
        print(output_json, flush=True)


@torch.inference_mode()
def probe_qkv_linear_compile(args: argparse.Namespace) -> None:
    apply_runtime_env(args)
    model, model_dir, device, dtype = load_model_for_args(args)
    mm_bmm_format_nd_info = configure_npu_mm_bmm_format_nd(str(args.npu_mm_bmm_format_nd), device)
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    baseline_path = Path(args.baseline).expanduser().resolve()
    manifest = load_baseline_manifest(baseline_path)
    dataset_dir = resolve_dataset_dir(args.dataset_dir or manifest["build_summary"]["page"]["dataset_dir"])
    inputs = build_inputs_from_manifest(manifest=manifest, model_dir=model_dir, tokenizer=tokenizer, dataset_dir=dataset_dir)
    item_index = int(args.item_index)
    if item_index < 0 or item_index >= len(inputs):
        raise ValueError(f"--item-index {item_index} is out of range for {len(inputs)} baseline inputs")
    item = inputs[item_index]
    impls = parse_qkv_linear_probe_impls(args.impls)
    sources = parse_qkv_linear_probe_sources(args.sources)
    bridges = parse_qkv_linear_probe_bridges(args.bridges)
    ln_impls = parse_qkv_linear_probe_ln_impls(args.ln_impls)

    backend_name = str(args.vision_compile_backend)
    backend = None
    backend_meta: dict[str, Any] = {"backend_kind": "none"}
    compile_kwargs: dict[str, Any] = {"fullgraph": True, "dynamic": False}
    old_capture_scalar_outputs: bool | None = None
    if backend_name != "none":
        import torch._dynamo

        old_capture_scalar_outputs = bool(torch._dynamo.config.capture_scalar_outputs)
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.reset()
        backend, backend_meta = vision_compile_backend(
            backend_name,
            device,
            torchair_mode=str(getattr(args, "torchair_mode", "default")),
            torchair_run_eagerly=bool(getattr(args, "torchair_run_eagerly", False)),
            torchair_graph_dump_type=str(getattr(args, "torchair_graph_dump_type", "none")),
            torchair_graph_dump_dir=getattr(args, "torchair_graph_dump_dir", None),
            torchair_msit_dump_kind=str(getattr(args, "torchair_msit_dump_kind", "none")),
            torchair_msit_dump_dir=getattr(args, "torchair_msit_dump_dir", None),
            torchair_msit_dump_mode=str(getattr(args, "torchair_msit_dump_mode", "output")),
            torchair_msit_dump_token=str(getattr(args, "torchair_msit_dump_token", "")),
            torchair_msit_dump_layer=str(getattr(args, "torchair_msit_dump_layer", "")),
            torchair_msit_fusion_switch_file=getattr(args, "torchair_msit_fusion_switch_file", None),
        )
        if backend is not None:
            compile_kwargs["backend"] = backend

    pixel_values = item.pixel_values.to(device=device, dtype=model.visual.dtype)
    patch_pos_module = SingleCropStaticVisualPrefixModule(
        model,
        item.image_grid_thw,
        device=device,
        stage="patch_pos",
        debug_no_padding=bool(args.debug_static_visual_no_padding),
        debug_min_pad_tokens=int(args.debug_static_visual_min_pad_tokens),
        debug_pad_to_multiple=int(args.debug_static_visual_pad_to_multiple),
    ).eval()
    ln1_module = SingleCropStaticVisualPrefixModule(
        model,
        item.image_grid_thw,
        device=device,
        stage="ln1",
        debug_no_padding=bool(args.debug_static_visual_no_padding),
        debug_min_pad_tokens=int(args.debug_static_visual_min_pad_tokens),
        debug_pad_to_multiple=int(args.debug_static_visual_pad_to_multiple),
    ).eval()
    maybe_sync(device)
    patch_pos_hidden = patch_pos_module(pixel_values).detach()
    ln1_hidden = ln1_module(pixel_values).detach()
    maybe_sync(device)

    transformer = model.visual.vision_model
    encoder_layer = transformer.encoder.layers[0]
    attention = encoder_layer.self_attn

    rows: list[dict[str, Any]] = []
    try:
        for source in sources:
            if source == "ln1":
                input_tensor = ln1_hidden
                source_ln_impls = ("precomputed",)
            elif source == "patch_pos":
                input_tensor = patch_pos_hidden
                source_ln_impls = tuple(ln_impls)
            else:
                raise RuntimeError(f"unreachable QKV source={source!r}")
            for ln_impl in source_ln_impls:
                for bridge in bridges:
                    for impl in impls:
                        row: dict[str, Any] = {
                            "item_index": int(item_index),
                            "id": str(item.entry.get("id")),
                            "layout_label": str(item.entry.get("layout_label")),
                            "source": str(source),
                            "ln_impl": str(ln_impl),
                            "bridge": str(bridge),
                            "impl": str(impl),
                            "backend": str(backend_name),
                            "device": str(device),
                            "dtype": str(dtype),
                            "image_grid_thw": [int(value) for value in item.image_grid_thw.flatten().tolist()],
                            "static_real_seq_len": int(ln1_module.static_real_seq_len),
                            "static_pad_tokens": int(ln1_module.static_pad_tokens),
                            "static_physical_seq_len": int(ln1_module.static_physical_seq_len),
                            "static_physical_seq_mod16": int(ln1_module.static_physical_seq_len % 16),
                            "static_physical_seq_mod128": int(ln1_module.static_physical_seq_len % 128),
                            "input_summary": tensor_probe_summary(input_tensor),
                            "q_weight_summary": tensor_probe_summary(attention.q_proj.weight),
                            "q_bias_summary": None
                            if attention.q_proj.bias is None
                            else tensor_probe_summary(attention.q_proj.bias),
                        }
                        try:
                            if source == "ln1":
                                module = VisionQKVLinearProbeModule(
                                    attention,
                                    impl=str(impl),
                                    bridge=str(bridge),
                                ).eval()
                            else:
                                module = VisionLayerNormQKVLinearProbeModule(
                                    encoder_layer.layer_norm1,
                                    attention,
                                    impl=str(impl),
                                    bridge=str(bridge),
                                    ln_impl=str(ln_impl),
                                ).eval()

                            maybe_sync(device)
                            start = time.perf_counter()
                            eager_before = module(input_tensor)
                            maybe_sync(device)
                            eager_before_s = float(time.perf_counter() - start)
                            row["eager_before_s"] = eager_before_s
                            row["eager_before_summary"] = tensor_probe_summary(eager_before)

                            if backend_name == "none":
                                row.update(
                                    {
                                        "ok": True,
                                        "compiled": False,
                                        "compiled_second_matches_eager": None,
                                        "eager_after_vs_eager_before": diff_stats(
                                            eager_before.cpu(),
                                            eager_before.cpu(),
                                        ),
                                    }
                                )
                            else:
                                import torch._dynamo

                                torch._dynamo.reset()
                                maybe_sync(device)
                                start = time.perf_counter()
                                compiled = torch.compile(module, **compile_kwargs)
                                maybe_sync(device)
                                compile_wrapper_s = float(time.perf_counter() - start)

                                maybe_sync(device)
                                start = time.perf_counter()
                                compiled_first = compiled(input_tensor)
                                maybe_sync(device)
                                compiled_first_s = float(time.perf_counter() - start)

                                maybe_sync(device)
                                start = time.perf_counter()
                                compiled_second = compiled(input_tensor)
                                maybe_sync(device)
                                compiled_second_s = float(time.perf_counter() - start)

                                maybe_sync(device)
                                start = time.perf_counter()
                                eager_after = module(input_tensor)
                                maybe_sync(device)
                                eager_after_s = float(time.perf_counter() - start)
                                row.update(
                                    {
                                        "ok": True,
                                        "compiled": True,
                                        "compile_wrapper_s": compile_wrapper_s,
                                        "compiled_first_s": compiled_first_s,
                                        "compiled_second_s": compiled_second_s,
                                        "eager_after_s": eager_after_s,
                                        "compiled_first_summary": tensor_probe_summary(compiled_first),
                                        "compiled_second_summary": tensor_probe_summary(compiled_second),
                                        "compiled_first_vs_eager_before": diff_stats(
                                            compiled_first.cpu(),
                                            eager_before.cpu(),
                                        ),
                                        "compiled_second_vs_eager_before": diff_stats(
                                            compiled_second.cpu(),
                                            eager_before.cpu(),
                                        ),
                                        "compiled_first_vs_second": diff_stats(
                                            compiled_first.cpu(),
                                            compiled_second.cpu(),
                                        ),
                                        "eager_after_vs_eager_before": diff_stats(
                                            eager_after.cpu(),
                                            eager_before.cpu(),
                                        ),
                                    }
                                )
                                row["compiled_second_matches_eager"] = qkv_linear_compiled_matches_eager(row)
                        except Exception as exc:
                            row.update({"ok": False, "error": f"{exc.__class__.__name__}: {exc}"})
                        rows.append(row)
                        print(
                            f"QKV_LINEAR source={source} ln_impl={ln_impl} bridge={bridge} impl={impl} "
                            f"ok={row.get('ok')} match={row.get('compiled_second_matches_eager')} "
                            f"max_abs={row.get('compiled_second_vs_eager_before', {}).get('max_abs_diff')} "
                            f"compiled_max={row.get('compiled_second_summary', {}).get('finite_max')}",
                            file=sys.stderr,
                            flush=True,
                        )
    finally:
        if old_capture_scalar_outputs is not None:
            import torch._dynamo

            torch._dynamo.config.capture_scalar_outputs = old_capture_scalar_outputs

    ok_rows = [row for row in rows if row.get("ok")]
    compiled_rows = [row for row in ok_rows if row.get("compiled")]
    mismatch_rows = [row for row in compiled_rows if not bool(row.get("compiled_second_matches_eager"))]
    error_rows = [row for row in rows if not row.get("ok")]
    first_mismatch = mismatch_rows[0] if mismatch_rows else None
    by_source_impl = [
        {
            "source": row.get("source"),
            "ln_impl": row.get("ln_impl"),
            "bridge": row.get("bridge"),
            "impl": row.get("impl"),
            "ok": row.get("ok"),
            "compiled_second_matches_eager": row.get("compiled_second_matches_eager"),
            "max_abs_diff": row.get("compiled_second_vs_eager_before", {}).get("max_abs_diff"),
            "mean_abs_diff": row.get("compiled_second_vs_eager_before", {}).get("mean_abs_diff"),
            "compiled_nonfinite_count": row.get("compiled_second_vs_eager_before", {}).get("lhs_nonfinite_count"),
            "compiled_finite_min": row.get("compiled_second_summary", {}).get("finite_min"),
            "compiled_finite_max": row.get("compiled_second_summary", {}).get("finite_max"),
        }
        for row in rows
    ]
    summary = {
        "total_cases": int(len(rows)),
        "ok_cases": int(len(ok_rows)),
        "error_cases": int(len(error_rows)),
        "compiled_cases": int(len(compiled_rows)),
        "compiled_second_matches_eager_count": int(
            sum(bool(row.get("compiled_second_matches_eager")) for row in compiled_rows)
        ),
        "compiled_second_matches_eager_all": bool(
            len(compiled_rows) > 0
            and all(bool(row.get("compiled_second_matches_eager")) for row in compiled_rows)
        ),
        "compiled_nonfinite_case_count": int(
            sum(
                int(row.get("compiled_second_vs_eager_before", {}).get("lhs_nonfinite_count", 0) or 0) > 0
                for row in compiled_rows
            )
        ),
        "first_mismatch": None
        if first_mismatch is None
        else {
            "source": first_mismatch.get("source"),
            "ln_impl": first_mismatch.get("ln_impl"),
            "bridge": first_mismatch.get("bridge"),
            "impl": first_mismatch.get("impl"),
            "shape": first_mismatch.get("compiled_second_vs_eager_before", {}).get("shape"),
            "max_abs_diff": first_mismatch.get("compiled_second_vs_eager_before", {}).get("max_abs_diff"),
            "mean_abs_diff": first_mismatch.get("compiled_second_vs_eager_before", {}).get("mean_abs_diff"),
            "compiled_nonfinite_count": first_mismatch.get("compiled_second_vs_eager_before", {}).get(
                "lhs_nonfinite_count"
            ),
            "compiled_finite_min": first_mismatch.get("compiled_second_summary", {}).get("finite_min"),
            "compiled_finite_max": first_mismatch.get("compiled_second_summary", {}).get("finite_max"),
        },
        "by_source_impl": by_source_impl,
        "failed_case_keys": [
            {
                "source": row.get("source"),
                "ln_impl": row.get("ln_impl"),
                "bridge": row.get("bridge"),
                "impl": row.get("impl"),
                "error": row.get("error"),
            }
            for row in error_rows
        ],
    }
    output = {
        "schema_version": 1,
        "experiment": "07_vision_prefill_optimization",
        "kind": "qkv_linear_compile_probe",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "dtype": str(dtype),
        "model": str(model_dir),
        "baseline": str(baseline_path),
        "backend": str(backend_name),
        "compile_api": None if backend_name == "none" else "torch.compile",
        "fullgraph": bool(backend_name != "none"),
        "dynamic": False,
        "capture_scalar_outputs": old_capture_scalar_outputs is not None,
        "capture_scalar_outputs_previous": old_capture_scalar_outputs,
        "dynamo_reset_before_compile": bool(backend_name != "none"),
        "compile_backend_meta": backend_meta,
        "npu_mm_bmm_format_nd": mm_bmm_format_nd_info,
        "sources": sources,
        "ln_impls": ln_impls,
        "bridges": bridges,
        "impls": impls,
        "source_meaning": {
            "ln1": "QKV-only graph fed by the materialized eager layer0 layer_norm1 output.",
            "patch_pos": "LayerNorm->QKV graph fed by the materialized eager patch+position output.",
        },
        "bridge_meaning": {
            "format_cast_nd": "Explicit torch_npu.npu_format_cast(tensor, 2) after LayerNorm and before QKV.",
            "format_cast_nz_then_nd": "Explicit tensor -> FRACTAL_NZ(29) -> ND(2) format transition after LayerNorm.",
            "transpose_roundtrip": "Known-good diagnostic barrier that forces materialization through transpose/contiguous.",
        },
        "summary": summary,
        "results": rows,
    }
    output_json = json.dumps(output, indent=2, default=json_default)
    output_path_raw = str(getattr(args, "output", "") or "").strip()
    if output_path_raw:
        output_path = Path(output_path_raw).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_json, encoding="utf-8")
        print(json.dumps({"qkv_linear_probe_output": str(output_path), "summary": summary}, indent=2, default=json_default), flush=True)
    else:
        print(output_json, flush=True)


class PromptFAOnlyModule(torch.nn.Module):
    """Tiny PromptFA module for isolating eager-vs-compile behavior."""

    def __init__(
        self,
        *,
        num_heads: int,
        scale: float,
        sparse_mode: int,
        atten_mask: torch.Tensor | None,
    ):
        super().__init__()
        self.num_heads = int(num_heads)
        self.scale = float(scale)
        self.sparse_mode = int(sparse_mode)
        self.register_buffer(
            "atten_mask",
            None if atten_mask is None else atten_mask.detach().clone().to(torch.bool).contiguous(),
            persistent=False,
        )

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return vision_prompt_flash_attention_bnsd(
            q,
            k,
            v,
            num_heads=self.num_heads,
            scale=self.scale,
            atten_mask=self.atten_mask,
            sparse_mode_override=self.sparse_mode,
        )


def build_promptfa_probe_mask(case: str, seq_len: int, *, device: torch.device) -> tuple[torch.Tensor | None, int, str]:
    if case == "no_mask":
        return None, 0, "none"
    if case == "all_false_mask":
        return torch.zeros((1, 1, seq_len, seq_len), device=device, dtype=torch.bool), 1, "all_false"
    if case == "block_mask":
        mask = torch.zeros((1, 1, seq_len, seq_len), device=device, dtype=torch.bool)
        block_start = max(1, int(seq_len * 3 // 4))
        mask[..., :, block_start:seq_len] = True
        return mask, 1, f"block_last_{seq_len - block_start}_keys"
    if case == "block_mask_mode0":
        mask = torch.zeros((1, 1, seq_len, seq_len), device=device, dtype=torch.bool)
        block_start = max(1, int(seq_len * 3 // 4))
        mask[..., :, block_start:seq_len] = True
        return mask, 0, f"diagnostic_mode0_block_last_{seq_len - block_start}_keys"
    raise ValueError(f"unsupported --cases entry {case!r}")


@torch.inference_mode()
def probe_promptfa_compile(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    if device.type != "npu":
        raise RuntimeError("probe-promptfa-compile is NPU-only because it tests TorchAir lowering of PromptFA")
    dtype = parse_dtype(args.dtype)
    configure_npu_jit_compile(args.npu_jit_compile, device)
    os.environ["PADDLE_OCR_VL_VISION_PROMPT_FA_LAYOUT"] = "bnsd"

    seq_lens = parse_int_list(args.seq_lens)
    cases = [case.strip() for case in str(args.cases).split(",") if case.strip()]
    if not cases:
        raise ValueError("--cases must include at least one case")
    heads = int(args.heads)
    head_dim = int(args.head_dim)
    if heads <= 0 or head_dim <= 0:
        raise ValueError("--heads and --head-dim must be positive")
    backend_name = str(args.vision_compile_backend)
    if backend_name == "none":
        raise ValueError("probe-promptfa-compile requires a real compile backend, not --vision-compile-backend none")
    import torch._dynamo

    torch._dynamo.reset()
    backend, backend_meta = vision_compile_backend(
        backend_name,
        device,
        torchair_mode=str(getattr(args, "torchair_mode", "default")),
        torchair_run_eagerly=bool(getattr(args, "torchair_run_eagerly", False)),
        torchair_graph_dump_type=str(getattr(args, "torchair_graph_dump_type", "none")),
        torchair_graph_dump_dir=getattr(args, "torchair_graph_dump_dir", None),
        torchair_msit_dump_kind=str(getattr(args, "torchair_msit_dump_kind", "none")),
        torchair_msit_dump_dir=getattr(args, "torchair_msit_dump_dir", None),
        torchair_msit_dump_mode=str(getattr(args, "torchair_msit_dump_mode", "output")),
        torchair_msit_dump_token=str(getattr(args, "torchair_msit_dump_token", "")),
        torchair_msit_dump_layer=str(getattr(args, "torchair_msit_dump_layer", "")),
        torchair_msit_fusion_switch_file=getattr(args, "torchair_msit_fusion_switch_file", None),
    )

    old_capture_scalar_outputs = bool(torch._dynamo.config.capture_scalar_outputs)
    torch._dynamo.config.capture_scalar_outputs = True
    compile_kwargs: dict[str, Any] = {"fullgraph": True, "dynamic": False}
    if backend is not None:
        compile_kwargs["backend"] = backend

    rows: list[dict[str, Any]] = []
    try:
        for seq_len in seq_lens:
            if int(seq_len) <= 1:
                raise ValueError(f"seq_len must be >1, got {seq_len}")
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(args.seed) + int(seq_len))
            q = torch.randn((1, heads, int(seq_len), head_dim), generator=generator, dtype=dtype).to(device)
            k = torch.randn((1, heads, int(seq_len), head_dim), generator=generator, dtype=dtype).to(device)
            v = torch.randn((1, heads, int(seq_len), head_dim), generator=generator, dtype=dtype).to(device)
            scale = float(head_dim) ** -0.5

            for case in cases:
                atten_mask, sparse_mode, mask_description = build_promptfa_probe_mask(case, int(seq_len), device=device)
                module = PromptFAOnlyModule(
                    num_heads=heads,
                    scale=scale,
                    sparse_mode=sparse_mode,
                    atten_mask=atten_mask,
                ).eval()
                ref_mask = atten_mask if atten_mask is not None else None
                row: dict[str, Any] = {
                    "seq_len": int(seq_len),
                    "case": str(case),
                    "mask_description": str(mask_description),
                    "sparse_mode": int(sparse_mode),
                    "mask_shape": None if atten_mask is None else [int(dim) for dim in atten_mask.shape],
                    "physical_seq_mod16": int(seq_len % 16),
                    "physical_seq_mod128": int(seq_len % 128),
                    "backend": backend_name,
                }
                try:
                    maybe_sync(device)
                    start = time.perf_counter()
                    eager_out = module(q, k, v)
                    maybe_sync(device)
                    eager_s = float(time.perf_counter() - start)

                    maybe_sync(device)
                    start = time.perf_counter()
                    compiled = torch.compile(module, **compile_kwargs)
                    maybe_sync(device)
                    compile_wrapper_s = float(time.perf_counter() - start)

                    maybe_sync(device)
                    start = time.perf_counter()
                    compiled_first = compiled(q, k, v)
                    maybe_sync(device)
                    compiled_first_s = float(time.perf_counter() - start)

                    maybe_sync(device)
                    start = time.perf_counter()
                    compiled_second = compiled(q, k, v)
                    maybe_sync(device)
                    compiled_second_s = float(time.perf_counter() - start)

                    manual_ref = manual_attention_bnsd(q, k, v, scale=scale, mask=ref_mask)
                    row.update(
                        {
                            "ok": True,
                            "eager_s": eager_s,
                            "compile_wrapper_s": compile_wrapper_s,
                            "compiled_first_s": compiled_first_s,
                            "compiled_second_s": compiled_second_s,
                            "eager_vs_manual": diff_stats(eager_out.cpu(), manual_ref.cpu()),
                            "compiled_first_vs_eager": diff_stats(compiled_first.cpu(), eager_out.cpu()),
                            "compiled_second_vs_eager": diff_stats(compiled_second.cpu(), eager_out.cpu()),
                            "compiled_first_vs_second": diff_stats(compiled_first.cpu(), compiled_second.cpu()),
                        }
                    )
                except Exception as exc:
                    row.update({"ok": False, "error": f"{exc.__class__.__name__}: {exc}"})
                rows.append(row)
    finally:
        torch._dynamo.config.capture_scalar_outputs = old_capture_scalar_outputs

    ok_rows = [row for row in rows if row.get("ok")]
    failed_rows = [row for row in rows if not row.get("ok")]
    def compiled_matches_eager(row: dict[str, Any]) -> bool:
        diff = row.get("compiled_second_vs_eager", {})
        return bool(
            diff.get("allclose_atol_5e_2_rtol_5e_2")
            and diff.get("lhs_nonfinite_count") == 0
            and diff.get("rhs_nonfinite_count") == 0
        )

    compiled_match_count = int(sum(compiled_matches_eager(row) for row in ok_rows))
    summary = {
        "total_cases": int(len(rows)),
        "ok_cases": int(len(ok_rows)),
        "error_cases": int(len(failed_rows)),
        "compiled_second_matches_eager_count": int(compiled_match_count),
        "compiled_second_matches_eager_all": bool(len(rows) > 0 and compiled_match_count == len(rows)),
        "failed_case_keys": [
            {"seq_len": row.get("seq_len"), "case": row.get("case"), "error": row.get("error")}
            for row in failed_rows
        ],
        "mismatch_case_keys": [
            {
                "seq_len": row.get("seq_len"),
                "case": row.get("case"),
                "max_abs_diff": row.get("compiled_second_vs_eager", {}).get("max_abs_diff"),
                "mean_abs_diff": row.get("compiled_second_vs_eager", {}).get("mean_abs_diff"),
                "lhs_nonfinite_count": row.get("compiled_second_vs_eager", {}).get("lhs_nonfinite_count"),
            }
            for row in ok_rows
            if not compiled_matches_eager(row)
        ],
    }

    output = {
        "schema_version": 1,
        "experiment": "07_vision_prefill_optimization",
        "kind": "promptfa_compile_probe",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "dtype": str(dtype),
        "backend": backend_name,
        "compile_api": "torch.compile",
        "fullgraph": True,
        "dynamic": False,
        "uses_torchair_cache_compile": False,
        "explicit_cache_dir": None,
        "capture_scalar_outputs": True,
        "capture_scalar_outputs_previous": old_capture_scalar_outputs,
        "dynamo_reset_before_compile": True,
        "compile_backend_meta": backend_meta,
        "torchair_graph_dump_summary": summarize_tree(backend_meta.get("torchair_graph_dump_dir"))
        if isinstance(backend_meta, dict)
        else {"path": None, "exists": False, "file_count": 0, "sample_files": []},
        "torchair_msit_dump_summary": summarize_tree(
            backend_meta.get("torchair_msit_dump", {}).get("expected_dump_dir")
        )
        if isinstance(backend_meta, dict) and isinstance(backend_meta.get("torchair_msit_dump"), dict)
        else {"path": None, "exists": False, "file_count": 0, "sample_files": []},
        "shape": {
            "batch": 1,
            "heads": int(heads),
            "seq_lens": [int(value) for value in seq_lens],
            "head_dim": int(head_dim),
            "layout": "BNSD",
        },
        "summary": summary,
        "results": rows,
    }
    output_json = json.dumps(output, indent=2, default=json_default)
    output_path_raw = str(getattr(args, "output", "") or "").strip()
    if output_path_raw:
        output_path = Path(output_path_raw).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_json, encoding="utf-8")
        print(json.dumps({"probe_output": str(output_path), "summary": summary}, indent=2, default=json_default), flush=True)
    else:
        print(output_json, flush=True)


@torch.inference_mode()
def probe_promptfa_mask(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    if device.type != "npu":
        raise RuntimeError("probe-promptfa-mask is NPU-only because it tests torch_npu.npu_prompt_flash_attention")
    dtype = parse_dtype(args.dtype)
    configure_npu_jit_compile(args.npu_jit_compile, device)
    os.environ["PADDLE_OCR_VL_VISION_PROMPT_FA_LAYOUT"] = "bnsd"

    batch = 1
    heads = int(args.heads)
    seq_len = int(args.seq_len)
    head_dim = int(args.head_dim)
    if seq_len <= 1 or heads <= 0 or head_dim <= 0:
        raise ValueError("--seq-len, --heads, and --head-dim must be positive, with seq-len > 1")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed))
    q = torch.randn((batch, heads, seq_len, head_dim), generator=generator, dtype=dtype).to(device)
    k = torch.randn((batch, heads, seq_len, head_dim), generator=generator, dtype=dtype).to(device)
    v = torch.randn((batch, heads, seq_len, head_dim), generator=generator, dtype=dtype).to(device)
    scale = float(head_dim) ** -0.5

    block_mask = torch.zeros((1, 1, seq_len, seq_len), device=device, dtype=torch.bool)
    block_start = max(1, int(seq_len * 3 // 4))
    block_mask[..., :, block_start:seq_len] = True
    causal_mask = torch.triu(torch.ones((1, 1, seq_len, seq_len), device=device, dtype=torch.bool), diagonal=1)

    ref_unmasked = manual_attention_bnsd(q, k, v, scale=scale, mask=None)
    ref_block = manual_attention_bnsd(q, k, v, scale=scale, mask=block_mask)
    ref_causal = manual_attention_bnsd(q, k, v, scale=scale, mask=causal_mask)

    results: list[dict[str, Any]] = []
    for sparse_mode in [0, 1]:
        os.environ["PADDLE_OCR_VL_VISION_PROMPT_FA_MASK_SPARSE_MODE"] = str(sparse_mode)
        row: dict[str, Any] = {"sparse_mode": int(sparse_mode)}
        try:
            maybe_sync(device)
            start = time.perf_counter()
            out_no_mask = vision_prompt_flash_attention_bnsd(
                q,
                k,
                v,
                num_heads=heads,
                scale=scale,
                atten_mask=None,
            )
            maybe_sync(device)
            no_mask_s = float(time.perf_counter() - start)

            maybe_sync(device)
            start = time.perf_counter()
            out_block = vision_prompt_flash_attention_bnsd(
                q,
                k,
                v,
                num_heads=heads,
                scale=scale,
                atten_mask=block_mask,
            )
            maybe_sync(device)
            block_s = float(time.perf_counter() - start)

            row.update(
                {
                    "ok": True,
                    "no_mask_elapsed_s": no_mask_s,
                    "block_mask_elapsed_s": block_s,
                    "no_mask_vs_ref_unmasked": diff_stats(out_no_mask.cpu(), ref_unmasked.cpu()),
                    "block_mask_vs_ref_block": diff_stats(out_block.cpu(), ref_block.cpu()),
                    "block_mask_vs_ref_unmasked": diff_stats(out_block.cpu(), ref_unmasked.cpu()),
                    "block_mask_vs_ref_causal": diff_stats(out_block.cpu(), ref_causal.cpu()),
                    "block_output_vs_no_mask_output": diff_stats(out_block.cpu(), out_no_mask.cpu()),
                }
            )
        except Exception as exc:
            row.update({"ok": False, "error": f"{exc.__class__.__name__}: {exc}"})
        results.append(row)

    by_mode = {int(row["sparse_mode"]): row for row in results}
    mode0 = by_mode.get(0, {})
    mode1 = by_mode.get(1, {})
    summary = {
        "mode0_mask_matches_block": bool(
            mode0.get("block_mask_vs_ref_block", {}).get("allclose_atol_5e_2_rtol_5e_2")
        ),
        "mode0_mask_differs_from_unmasked": bool(
            mode0.get("ok") and not mode0.get("block_mask_vs_ref_unmasked", {}).get("allclose_atol_5e_2_rtol_5e_2")
        ),
        "mode1_mask_matches_block": bool(
            mode1.get("block_mask_vs_ref_block", {}).get("allclose_atol_5e_2_rtol_5e_2")
        ),
        "mode1_mask_differs_from_unmasked": bool(
            mode1.get("ok") and not mode1.get("block_mask_vs_ref_unmasked", {}).get("allclose_atol_5e_2_rtol_5e_2")
        ),
        "mode0_error": mode0.get("error"),
        "mode1_error": mode1.get("error"),
    }
    summary["mode0_full_mask_semantics_passed"] = bool(
        summary["mode0_mask_matches_block"] and summary["mode0_mask_differs_from_unmasked"]
    )
    summary["mode1_full_mask_semantics_passed"] = bool(
        summary["mode1_mask_matches_block"] and summary["mode1_mask_differs_from_unmasked"]
    )
    summary["recommended_mask_sparse_mode"] = 1
    summary["recommended_full_mask_semantics_passed"] = bool(summary["mode1_full_mask_semantics_passed"])

    output = {
        "schema_version": 1,
        "experiment": "07_vision_prefill_optimization",
        "kind": "promptfa_mask_semantics_probe",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "dtype": str(dtype),
        "shape": {
            "batch": int(batch),
            "heads": int(heads),
            "seq_len": int(seq_len),
            "head_dim": int(head_dim),
            "layout": "BNSD",
        },
        "mask": {
            "block_mask_shape": [int(dim) for dim in block_mask.shape],
            "blocked_key_start": int(block_start),
            "blocked_key_count": int(seq_len - block_start),
            "true_means_masked": True,
        },
        "summary": summary,
        "results": results,
    }
    print(json.dumps(output, indent=2, default=json_default), flush=True)


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
                "standard records visual_tower_e2e_s plus full_prefill_e2e_s as separate measurements; "
                "phase_sync synchronizes around every named phase and is diagnostic only."
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
        real_vision_tokens = int(vision_tokens(item))
        candidate_physical_vision_tokens = int(
            vision_compile.get("static_visual_physical_seq_len", real_vision_tokens)
            if isinstance(vision_compile, dict)
            else real_vision_tokens
        )
        for _ in range(int(args.repeats)):
            candidate_tensors, timing = run_prefill_measurement(
                model=model,
                item=item,
                device=device,
                cache_length=int(args.cache_length),
                timing_mode=str(args.timing_mode),
                vision_forward=vision_forward,
            )
            visual_tower_s = float(timing.get("visual_tower_e2e_s", 0.0) or 0.0)
            if visual_tower_s > 0.0:
                timing["visual_tower_effective_tokens_per_s"] = float(real_vision_tokens) / visual_tower_s
                timing["visual_tower_physical_tokens_per_s"] = float(candidate_physical_vision_tokens) / visual_tower_s
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
                "candidate_physical_vision_tokens": int(candidate_physical_vision_tokens),
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
            "compiled": bool(str(args.vision_compile_backend) != "none"),
            "compile_api": (
                "torch.compile"
                if str(args.vision_compile_backend) != "none"
                else None
            ),
            "vision_attention": get_vision_attention_impl(),
            "vision_prompt_fa_layout": get_vision_prompt_fa_layout(),
            "vision_prompt_fa_mask_sparse_mode": int(args.vision_prompt_fa_mask_sparse_mode),
            "repeats": int(args.repeats),
            "warmup_repeats": int(args.warmup_repeats),
            "timing_mode": str(args.timing_mode),
            "candidate_vision_path": "static_visual",
            "vision_compile_backend": str(args.vision_compile_backend),
            "torchair_mode": str(getattr(args, "torchair_mode", "default")),
            "torchair_run_eagerly": bool(getattr(args, "torchair_run_eagerly", False)),
            "torchair_graph_dump_type": str(getattr(args, "torchair_graph_dump_type", "none")),
            "torchair_graph_dump_dir": str(getattr(args, "torchair_graph_dump_dir", "") or ""),
            "torchair_msit_dump_kind": str(getattr(args, "torchair_msit_dump_kind", "none")),
            "torchair_msit_dump_dir": str(getattr(args, "torchair_msit_dump_dir", "") or ""),
            "torchair_msit_dump_mode": str(getattr(args, "torchair_msit_dump_mode", "output")),
            "torchair_msit_dump_token": str(getattr(args, "torchair_msit_dump_token", "") or ""),
            "torchair_msit_dump_layer": str(getattr(args, "torchair_msit_dump_layer", "") or ""),
            "torchair_msit_fusion_switch_file": str(getattr(args, "torchair_msit_fusion_switch_file", "") or ""),
            "static_visual_pad_policy": STATIC_VISUAL_PAD_POLICY,
            "debug_static_visual_no_padding": bool(args.debug_static_visual_no_padding),
            "debug_static_visual_min_pad_tokens": int(args.debug_static_visual_min_pad_tokens),
            "debug_static_visual_pad_to_multiple": int(args.debug_static_visual_pad_to_multiple),
            "static_visual_encoder_path": "single_static_path_masked_padding_default",
            "timing_mode_note": (
                "standard records visual_tower_e2e_s plus full_prefill_e2e_s as separate measurements. "
                "The visual tower metric is the headline speed metric for experiment 07; full prefill e2e "
                "is secondary context; phase_sync is diagnostic only."
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
            "visual_tower_effective_tokens_per_s": aggregate_timed_token_rate(
                rows,
                token_key="vision_tokens",
                time_key="visual_tower_e2e_s",
            ),
            "visual_tower_physical_tokens_per_s": aggregate_timed_token_rate(
                rows,
                token_key="candidate_physical_vision_tokens",
                time_key="visual_tower_e2e_s",
            ),
            "vision_tokens": stats([float(row["vision_tokens"]) for row in rows]),
            "candidate_physical_vision_tokens": stats([float(row["candidate_physical_vision_tokens"]) for row in rows]),
            "projected_image_tokens": stats([float(row["projected_image_tokens"]) for row in rows]),
            "first_item_static_visual_shapes": [
                {
                    "index": int(row["index"]),
                    "id": str(row["id"]),
                    "image_grid_thw": list(row["image_grid_thw"]),
                    "vision_tokens": int(row["vision_tokens"]),
                    "static_visual_pad_tokens": int(row["vision_compile"].get("static_visual_pad_tokens", 0)),
                    "static_visual_physical_seq_len": int(
                        row["vision_compile"].get("static_visual_physical_seq_len", row["vision_tokens"])
                    ),
                    "static_visual_physical_seq_mod16": int(
                        row["vision_compile"].get("static_visual_physical_seq_mod16", row["vision_tokens"] % 16)
                    ),
                    "static_pad_attention_mask_enabled": bool(
                        row["vision_compile"].get("static_pad_attention_mask_enabled", False)
                    ),
                    "static_pad_attention_mask_shape": row["vision_compile"].get("static_pad_attention_mask_shape"),
                }
                for row in rows[:8]
            ],
        },
        "items": rows,
    }
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    print(json.dumps({"compare_output": str(output_path), "summary": output["summary"]}, indent=2, default=json_default), flush=True)


def add_common_args(parser: argparse.ArgumentParser, *, timing_default: str) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Local PaddleOCR-VL model directory. HF download is disabled.")
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="fp16", choices=DTYPE_CHOICES)
    parser.add_argument("--npu-jit-compile", default="off", choices=NPU_JIT_COMPILE_CHOICES)
    parser.add_argument("--vision-attention", default="prompt_flash_attention", choices=VISION_ATTENTION_CHOICES)
    parser.add_argument("--vision-prompt-fa-layout", default="bnsd", choices=("bnsd", "bsnd", "bsh"))
    parser.add_argument(
        "--vision-prompt-fa-mask-sparse-mode",
        type=int,
        default=1,
        choices=(0, 1),
        help="Sparse mode used only when passing a PromptFA atten_mask. Default: 1, validated on 310P3/CANN 8.2.RC1.",
    )
    parser.add_argument("--cache-length", type=int, default=2048)
    parser.add_argument("--timing-mode", default=str(timing_default), choices=TIMING_MODE_CHOICES)


def add_torchair_diagnostic_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--torchair-mode",
        default="default",
        choices=TORCHAIR_MODE_CHOICES,
        help=(
            "TorchAir compiler mode. Keep default on 310P; it is the max-autotune/Ascend-IR path used "
            "for this investigation."
        ),
    )
    parser.add_argument(
        "--torchair-run-eagerly",
        action="store_true",
        help=(
            "Diagnostic only: set compiler_config.debug.run_eagerly=True to execute the FX graph eagerly "
            "before graph execution, which helps separate FX semantics from GE/CANN graph execution."
        ),
    )
    parser.add_argument(
        "--torchair-graph-dump-type",
        default="none",
        choices=TORCHAIR_GRAPH_DUMP_TYPE_CHOICES,
        help="Diagnostic only: request TorchAir graph dumps, usually pbtxt for grep-friendly inspection.",
    )
    parser.add_argument(
        "--torchair-graph-dump-dir",
        default="",
        help="Diagnostic only: optional graph-dump directory. If omitted, TorchAir chooses its default dump location.",
    )
    parser.add_argument(
        "--torchair-msit-dump-kind",
        default="none",
        choices=TORCHAIR_MSIT_DUMP_KIND_CHOICES,
        help=(
            "Diagnostic only: attach MSIT TorchAir GE or FX dump config to the TorchAir CompilerConfig. "
            "Use separate dump dirs for ge and fx runs."
        ),
    )
    parser.add_argument(
        "--torchair-msit-dump-dir",
        default="",
        help="Diagnostic only: base directory passed to MSIT get_ge_dump_config/get_fx_dump_config.",
    )
    parser.add_argument(
        "--torchair-msit-dump-mode",
        default="output",
        choices=TORCHAIR_MSIT_DUMP_MODE_CHOICES,
        help="Diagnostic only: GE dump mode. Start with output to limit dump size; rerun with all if compare needs inputs.",
    )
    parser.add_argument(
        "--torchair-msit-dump-token",
        default="",
        help="Diagnostic only: optional comma-separated GE dump token filter, for example 0 or 0,1.",
    )
    parser.add_argument(
        "--torchair-msit-dump-layer",
        default="",
        help="Diagnostic only: optional comma-separated GE dump layer/op filter.",
    )
    parser.add_argument(
        "--torchair-msit-fusion-switch-file",
        default="",
        help="Diagnostic only: optional fusion switch JSON for GE dump, usually for GE-vs-GE fusion-off compare.",
    )


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
    add_common_args(compare_parser, timing_default="standard")
    compare_parser.add_argument("--baseline", default=str(SCRIPT_DIR / "baselines" / "promptfa_fp16_eager_64"))
    compare_parser.add_argument("--candidate-name", default="candidate")
    compare_parser.add_argument("--output", default=str(SCRIPT_DIR / "outputs" / "candidate_compare.json"))
    compare_parser.add_argument("--repeats", type=int, default=1)
    compare_parser.add_argument("--warmup-repeats", type=int, default=0)
    compare_parser.add_argument("--max-items", type=int, default=0)
    compare_parser.add_argument("--vision-compile-backend", default="none", choices=VISION_COMPILE_BACKEND_CHOICES)
    add_torchair_diagnostic_args(compare_parser)
    compare_parser.add_argument(
        "--debug-static-visual-no-padding",
        action="store_true",
        help=(
            "Diagnostic only: disable static visual dummy rows and padding masks. "
            "Use only as a no-mask control; normal experiment-07 candidates are padded."
        ),
    )
    compare_parser.add_argument(
        "--debug-static-visual-min-pad-tokens",
        type=int,
        default=0,
        help=(
            "Diagnostic only: require at least this many dummy visual rows in addition to the normal "
            "masked padded policy. Keep 0 for normal runs."
        ),
    )
    compare_parser.add_argument(
        "--debug-static-visual-pad-to-multiple",
        type=int,
        default=0,
        help=(
            "Diagnostic only: after normal/forced padding, round physical visual sequence length up "
            "to this multiple. Keep 0 for normal runs."
        ),
    )
    compare_parser.add_argument(
        "--validate-compiled-against-static-eager",
        action="store_true",
        help=(
            "Diagnostic only: for compiled candidates, run the same static visual wrapper eagerly once "
            "and record compiled-first-call vs static-eager physical and real-row diffs."
        ),
    )

    probe_parser = subparsers.add_parser(
        "probe-promptfa-mask",
        help="Synthetic NPU-only check of PromptFA custom atten_mask behavior for sparse modes 0 and 1.",
    )
    probe_parser.add_argument("--device", default="npu:0")
    probe_parser.add_argument("--dtype", default="fp16", choices=DTYPE_CHOICES)
    probe_parser.add_argument("--npu-jit-compile", default="off", choices=NPU_JIT_COMPILE_CHOICES)
    probe_parser.add_argument("--seq-len", type=int, default=640)
    probe_parser.add_argument("--heads", type=int, default=16)
    probe_parser.add_argument("--head-dim", type=int, default=72)
    probe_parser.add_argument("--seed", type=int, default=1234)

    compile_probe_parser = subparsers.add_parser(
        "probe-promptfa-compile",
        help="Synthetic NPU-only eager-vs-compiled PromptFA check for no-mask and masked cases.",
    )
    compile_probe_parser.add_argument("--device", default="npu:0")
    compile_probe_parser.add_argument("--dtype", default="fp16", choices=DTYPE_CHOICES)
    compile_probe_parser.add_argument("--npu-jit-compile", default="off", choices=NPU_JIT_COMPILE_CHOICES)
    compile_probe_parser.add_argument("--vision-compile-backend", default="torchair", choices=VISION_COMPILE_BACKEND_CHOICES)
    add_torchair_diagnostic_args(compile_probe_parser)
    compile_probe_parser.add_argument("--seq-lens", default="640,768")
    compile_probe_parser.add_argument("--cases", default="no_mask,all_false_mask,block_mask")
    compile_probe_parser.add_argument("--heads", type=int, default=16)
    compile_probe_parser.add_argument("--head-dim", type=int, default=72)
    compile_probe_parser.add_argument("--seed", type=int, default=1234)
    compile_probe_parser.add_argument("--output", default="")

    layernorm_probe_parser = subparsers.add_parser(
        "probe-layernorm-compile",
        help="Synthetic plus real-crop eager-vs-compiled LayerNorm check for TorchAir GE lowering.",
    )
    layernorm_probe_parser.add_argument("--model", default=DEFAULT_MODEL, help="Local model directory; required for --include-real-first-crop.")
    layernorm_probe_parser.add_argument("--dataset-dir", default=None)
    layernorm_probe_parser.add_argument("--baseline", default=str(SCRIPT_DIR / "baselines" / "promptfa_fp16_eager_64"))
    layernorm_probe_parser.add_argument("--device", default="npu:0")
    layernorm_probe_parser.add_argument("--dtype", default="fp16", choices=DTYPE_CHOICES)
    layernorm_probe_parser.add_argument("--npu-jit-compile", default="off", choices=NPU_JIT_COMPILE_CHOICES)
    layernorm_probe_parser.add_argument("--vision-compile-backend", default="torchair", choices=VISION_COMPILE_BACKEND_CHOICES)
    add_torchair_diagnostic_args(layernorm_probe_parser)
    layernorm_probe_parser.add_argument("--impls", default="nn,functional,manual,manual_fp16_reduce,npu_eval")
    layernorm_probe_parser.add_argument("--seq-lens", default="580,640,768")
    layernorm_probe_parser.add_argument("--hidden-size", type=int, default=1152)
    layernorm_probe_parser.add_argument("--eps", type=float, default=1e-6)
    layernorm_probe_parser.add_argument("--seed", type=int, default=1234)
    layernorm_probe_parser.add_argument(
        "--synthetic-input-scales",
        default="1.0",
        help=(
            "Comma-separated synthetic input scales. Use values like 1,64,128 to stress fp16 "
            "LayerNorm reduction overflow separately from real-crop inputs."
        ),
    )
    layernorm_probe_parser.add_argument("--synthetic-affine", default="random", choices=("identity", "random"))
    layernorm_probe_parser.add_argument("--include-real-first-crop", action="store_true")
    layernorm_probe_parser.add_argument("--real-item-index", type=int, default=0)
    layernorm_probe_parser.add_argument(
        "--debug-static-visual-no-padding",
        action="store_true",
        help="Diagnostic only: build the real-crop LayerNorm input with the no-padding static visual control.",
    )
    layernorm_probe_parser.add_argument(
        "--debug-static-visual-min-pad-tokens",
        type=int,
        default=0,
        help="Diagnostic only: force additional real-crop static visual dummy rows.",
    )
    layernorm_probe_parser.add_argument(
        "--debug-static-visual-pad-to-multiple",
        type=int,
        default=0,
        help="Diagnostic only: round real-crop static visual physical rows to this multiple.",
    )
    layernorm_probe_parser.add_argument("--output", default="")

    prefix_probe_parser = subparsers.add_parser(
        "probe-visual-prefix-compile",
        help="Real-crop compiled-prefix bisection for static visual GE drift.",
    )
    prefix_probe_parser.add_argument("--model", default=DEFAULT_MODEL, help="Local model directory.")
    prefix_probe_parser.add_argument("--dataset-dir", default=None)
    prefix_probe_parser.add_argument("--baseline", default=str(SCRIPT_DIR / "baselines" / "promptfa_fp16_eager_64"))
    prefix_probe_parser.add_argument("--device", default="npu:0")
    prefix_probe_parser.add_argument("--dtype", default="fp16", choices=DTYPE_CHOICES)
    prefix_probe_parser.add_argument("--npu-jit-compile", default="off", choices=NPU_JIT_COMPILE_CHOICES)
    prefix_probe_parser.add_argument("--vision-attention", default="prompt_flash_attention", choices=VISION_ATTENTION_CHOICES)
    prefix_probe_parser.add_argument("--vision-prompt-fa-layout", default="bnsd", choices=("bnsd", "bsnd", "bsh"))
    prefix_probe_parser.add_argument("--vision-prompt-fa-mask-sparse-mode", type=int, default=1, choices=(0, 1))
    prefix_probe_parser.add_argument("--vision-compile-backend", default="torchair", choices=VISION_COMPILE_BACKEND_CHOICES)
    add_torchair_diagnostic_args(prefix_probe_parser)
    prefix_probe_parser.add_argument(
        "--stages",
        default="patch_conv,patch_flat,patch_pad,patch_pos,ln1",
        help=f"Comma-separated prefix stages. Choices: {','.join(VISUAL_PREFIX_STAGE_CHOICES)}",
    )
    prefix_probe_parser.add_argument("--max-items", type=int, default=1)
    prefix_probe_parser.add_argument(
        "--debug-static-visual-no-padding",
        action="store_true",
        help="Diagnostic only: disable static visual padding/mask in the prefix module.",
    )
    prefix_probe_parser.add_argument("--debug-static-visual-min-pad-tokens", type=int, default=0)
    prefix_probe_parser.add_argument("--debug-static-visual-pad-to-multiple", type=int, default=0)
    prefix_probe_parser.add_argument("--output", default="")

    qkv_probe_parser = subparsers.add_parser(
        "probe-qkv-linear-compile",
        help="Real-crop isolated QKV linear/LN->QKV compile probe for the first vision layer.",
    )
    qkv_probe_parser.add_argument("--model", default=DEFAULT_MODEL, help="Local model directory.")
    qkv_probe_parser.add_argument("--dataset-dir", default=None)
    qkv_probe_parser.add_argument("--baseline", default=str(SCRIPT_DIR / "baselines" / "promptfa_fp16_eager_64"))
    qkv_probe_parser.add_argument("--device", default="npu:0")
    qkv_probe_parser.add_argument("--dtype", default="fp16", choices=DTYPE_CHOICES)
    qkv_probe_parser.add_argument("--npu-jit-compile", default="off", choices=NPU_JIT_COMPILE_CHOICES)
    qkv_probe_parser.add_argument("--vision-attention", default="prompt_flash_attention", choices=VISION_ATTENTION_CHOICES)
    qkv_probe_parser.add_argument("--vision-prompt-fa-layout", default="bnsd", choices=("bnsd", "bsnd", "bsh"))
    qkv_probe_parser.add_argument("--vision-prompt-fa-mask-sparse-mode", type=int, default=1, choices=(0, 1))
    qkv_probe_parser.add_argument("--vision-compile-backend", default="torchair", choices=VISION_COMPILE_BACKEND_CHOICES)
    add_torchair_diagnostic_args(qkv_probe_parser)
    qkv_probe_parser.add_argument(
        "--npu-mm-bmm-format-nd",
        default="default",
        choices=("default", "enable", "disable"),
        help="Diagnostic only: set torch_npu MM_BMM_ND_ENABLE before compiling the QKV probe.",
    )
    qkv_probe_parser.add_argument("--item-index", type=int, default=0)
    qkv_probe_parser.add_argument(
        "--sources",
        default="ln1,patch_pos",
        help=f"Comma-separated source tensors. Choices: {','.join(QKV_LINEAR_PROBE_SOURCE_CHOICES)}",
    )
    qkv_probe_parser.add_argument(
        "--bridges",
        default="none",
        help=f"Comma-separated post-LayerNorm/pre-QKV bridge ops. Choices: {','.join(QKV_LINEAR_PROBE_BRIDGE_CHOICES)}",
    )
    qkv_probe_parser.add_argument(
        "--ln-impls",
        default="module",
        help=(
            "Comma-separated LayerNorm implementations for source=patch_pos. "
            f"Choices: {','.join(QKV_LINEAR_PROBE_LN_IMPL_CHOICES)}"
        ),
    )
    qkv_probe_parser.add_argument(
        "--impls",
        default="module_three,functional_three,matmul_three,functional_single,matmul_single,module_q,functional_q,matmul_q,functional_q_no_bias,matmul_q_no_bias",
        help=f"Comma-separated implementations. Choices: {','.join(QKV_LINEAR_PROBE_IMPL_CHOICES)}",
    )
    qkv_probe_parser.add_argument(
        "--debug-static-visual-no-padding",
        action="store_true",
        help="Diagnostic only: build source tensors with static visual padding disabled.",
    )
    qkv_probe_parser.add_argument("--debug-static-visual-min-pad-tokens", type=int, default=0)
    qkv_probe_parser.add_argument("--debug-static-visual-pad-to-multiple", type=int, default=0)
    qkv_probe_parser.add_argument("--output", default="")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "make-baseline":
        make_baseline(args)
    elif args.command == "compare":
        compare_candidate(args)
    elif args.command == "probe-promptfa-mask":
        probe_promptfa_mask(args)
    elif args.command == "probe-promptfa-compile":
        probe_promptfa_compile(args)
    elif args.command == "probe-layernorm-compile":
        probe_layernorm_compile(args)
    elif args.command == "probe-visual-prefix-compile":
        probe_visual_prefix_compile(args)
    elif args.command == "probe-qkv-linear-compile":
        probe_qkv_linear_compile(args)
    else:
        raise RuntimeError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
