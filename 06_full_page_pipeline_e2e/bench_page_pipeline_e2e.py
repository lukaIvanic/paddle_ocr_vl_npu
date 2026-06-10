#!/usr/bin/env python3
"""Benchmark the page-to-GT-crop PaddleOCR-VL serving shape.

Experiment 6 measures the path we actually care about for page serving:

1. Select real OmniDocBench page images.
2. Select layout regions, defaulting to OmniDocBench GT layout boxes.
3. Crop selected layout regions from those pages in memory.
4. Build PaddleOCR-VL recognizer prefills one crop at a time.
5. Decode all crop states through the existing hot-swap batch scheduler.

Validation, when enabled, compares the hot-swap recognizer output against the
same local static recognizer run independently per detected crop. It is not an
OmniDocBench accuracy scorer. The optional official layout path remains as a
separate baseline hook for machines where PaddleOCR/PaddleX can run.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable
import unicodedata

import numpy as np
import torch
from PIL import Image
from tokenizers import Tokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
AOE_ROOT = REPO_ROOT.parent
EXP5_DIR = REPO_ROOT / "05_full_recognizer_optimizations"
if str(EXP5_DIR) not in sys.path:
    sys.path.insert(0, str(EXP5_DIR))

from bench_recognizer_queue import (  # noqa: E402
    BACKEND_CHOICES,
    DECODE_SCHEDULE_CHOICES,
    EOS_MODE_CHOICES,
    QueueInput,
    build_ready_bank_incremental,
    cast_decode_linear_weights_to_nz,
    json_default,
    materialize_hotswap_item,
    prompt_token_summary,
    static_hotswap_decode_loop,
    stats,
    token_range_summary,
    validate_outputs,
    vision_token_bucket_summary,
)
from local_modeling_paddleocr_vl import (  # noqa: E402
    DECODE_ATTENTION,
    DECODE_CACHE_UPDATE,
    LocalPaddleOCRVLForConditionalGeneration,
    _resolve_model_dir,
)
from probe_static_compile import DEFAULT_TORCHAIR_CACHE_DIR, compile_decode_module, maybe_sync  # noqa: E402
from run_local_recognition import (  # noqa: E402
    NPU_JIT_COMPILE_CHOICES,
    build_inputs,
    configure_npu_jit_compile,
    load_preprocessor_config,
    parse_dtype,
    resolve_device,
    smart_resize,
)


Image.MAX_IMAGE_PIXELS = None

DEFAULT_DATASET_DIR = (
    AOE_ROOT
    / "remote_artifacts/aos_research_remote_shutdown_20260531"
    / "glm_ocr_portable_bundle/data/OmniDocBench"
)

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

TEXT_EDIT_LABELS = {"text_block"}
TEXT_DIAGNOSTIC_LABELS = {"text_block", "title", "code_txt"}
FORMULA_LABELS = {"formula", "formula_number", "equation", "equation_isolated", "equation_semantic"}
TABLE_LABELS = {"table"}
READING_ORDER_LABELS = {"text_block", "title", "code_txt"}


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


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def tok_per_s(count: int | float, seconds: float) -> float | None:
    seconds = float(seconds)
    if seconds <= 0:
        return None
    return float(count) / seconds


def resolve_dataset_dir(path: Path) -> Path:
    path = path.expanduser()
    if path.exists():
        return path.resolve()
    candidate = REPO_ROOT / path
    if candidate.exists():
        return candidate.resolve()
    return path.resolve()


def load_pages(dataset_dir: Path, *, page_start: int, num_pages: int) -> list[PageInput]:
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
    end = min(len(dataset), start + int(num_pages))
    if start < 0 or start >= len(dataset):
        raise ValueError(f"--page-start {start} outside dataset with {len(dataset)} pages")
    if end <= start:
        raise ValueError(f"--num-pages must select at least one page, got {num_pages}")

    pages: list[PageInput] = []
    for selected_idx, dataset_index in enumerate(range(start, end)):
        record = dataset[dataset_index]
        page_info = dict(record.get("page_info", {}) or {})
        rel = str(page_info.get("image_path", ""))
        image_path = images_dir / rel
        if not image_path.exists():
            raise FileNotFoundError(f"page image not found for dataset index {dataset_index}: {image_path}")
        pages.append(
            PageInput(
                idx=int(selected_idx),
                dataset_index=int(dataset_index),
                image_path=image_path.resolve(),
                image_rel=rel,
                page_info=page_info,
                gt_layout_dets=clean_json(record.get("layout_dets", []) or []),
            )
        )
    return pages


def create_layout_predictor(args: argparse.Namespace) -> Any:
    kwargs = {
        "model_name": args.layout_model_name,
        "model_dir": str(args.layout_model_dir) if args.layout_model_dir else None,
        "device": args.layout_device,
    }
    optional = {
        "threshold": args.layout_threshold,
        "layout_nms": args.layout_nms,
        "layout_unclip_ratio": args.layout_unclip_ratio,
        "layout_merge_bboxes_mode": args.layout_merge_bboxes_mode,
    }
    kwargs.update({key: value for key, value in optional.items() if value is not None})
    kwargs = {key: value for key, value in kwargs.items() if value is not None}

    try:
        from paddleocr import LayoutDetection

        return LayoutDetection(**kwargs)
    except Exception as first_exc:
        try:
            from paddlex import create_predictor

            fallback_kwargs = dict(kwargs)
            model_name = fallback_kwargs.pop("model_name")
            model_dir = fallback_kwargs.pop("model_dir", None)
            return create_predictor(model_name=model_name, model_dir=model_dir, **fallback_kwargs)
        except Exception as second_exc:
            raise RuntimeError(
                "Could not create the official PaddleOCR/PaddleX layout detector. "
                "Install paddleocr/paddlex in the work environment, or pass a valid "
                "--layout-model-dir. First error: "
                f"{type(first_exc).__name__}: {first_exc}. Fallback error: "
                f"{type(second_exc).__name__}: {second_exc}"
            ) from second_exc


def predictor_predict(predictor: Any, image_paths: list[Path]) -> list[Any]:
    paths = [str(path) for path in image_paths]
    output = predictor.predict(paths)
    return list(output)


def result_to_dict(result: Any) -> dict[str, Any]:
    raw = result
    if hasattr(raw, "json"):
        attr = getattr(raw, "json")
        raw = attr() if callable(attr) else attr
    if isinstance(raw, str):
        raw = json.loads(raw)
    if isinstance(raw, dict):
        return clean_json(raw)

    if hasattr(result, "save_to_json"):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            result.save_to_json(tmp_path)
            json_files = sorted(tmp_path.glob("*.json"))
            if not json_files:
                raise ValueError(f"save_to_json produced no JSON files for layout result {type(result).__name__}")
            return clean_json(json.loads(json_files[0].read_text(encoding="utf-8")))

    raise TypeError(f"cannot normalize layout result object of type {type(result).__name__}")


def extract_layout_boxes(result_dict: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = result_dict.get("res", result_dict)
    candidates = [
        root.get("boxes") if isinstance(root, dict) else None,
        (root.get("layout_det_res", {}) or {}).get("boxes") if isinstance(root, dict) else None,
        result_dict.get("boxes"),
        (result_dict.get("layout_det_res", {}) or {}).get("boxes"),
    ]
    for boxes in candidates:
        if isinstance(boxes, list):
            return [clean_json(box) for box in boxes], clean_json(root)
    raise ValueError(
        "layout result has no boxes list; expected result['boxes'] or "
        "result['res']['layout_det_res']['boxes']. Keys: "
        f"{sorted(result_dict.keys())}"
    )


def run_layout_detection(
    *,
    pages: list[PageInput],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    timings: dict[str, float] = {}
    start = time.perf_counter()
    predictor = create_layout_predictor(args)
    timings["layout_model_init_s"] = time.perf_counter() - start

    start = time.perf_counter()
    raw_results = predictor_predict(predictor, [page.image_path for page in pages])
    timings["layout_detection_s"] = time.perf_counter() - start
    if len(raw_results) != len(pages):
        raise ValueError(f"layout returned {len(raw_results)} results for {len(pages)} pages")

    normalized = []
    for page, result in zip(pages, raw_results):
        result_dict = result_to_dict(result)
        boxes, root = extract_layout_boxes(result_dict)
        normalized.append(
            {
                "selected_page_idx": int(page.idx),
                "dataset_index": int(page.dataset_index),
                "image_path": str(page.image_path),
                "image_rel": page.image_rel,
                "page_info": clean_json(page.page_info),
                "boxes": boxes,
                "raw_result_root": root,
            }
        )

    if hasattr(predictor, "close"):
        close_start = time.perf_counter()
        predictor.close()
        timings["layout_model_close_s"] = time.perf_counter() - close_start
    else:
        timings["layout_model_close_s"] = 0.0
    return normalized, timings


def load_layout_cache(path: Path, pages: list[PageInput]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    start = time.perf_counter()
    data = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("pages"), list):
        rows = data["pages"]
    elif isinstance(data, list):
        rows = data
    else:
        raise ValueError(f"unsupported layout cache JSON shape in {path}")
    if len(rows) < len(pages):
        raise ValueError(f"layout cache has {len(rows)} pages but benchmark selected {len(pages)}")
    selected = rows[: len(pages)]
    return selected, {"layout_cache_load_s": time.perf_counter() - start, "layout_detection_s": 0.0}


def write_layout_cache(path: Path, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": clean_json(metadata), "pages": clean_json(rows)}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_omnidocbench_gt_layout_pages(
    pages: list[PageInput],
    *,
    include_ignored: bool,
    include_empty_gt: bool,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    start = time.perf_counter()
    rows = []
    total_raw = 0
    total_skipped_ignored = 0
    total_skipped_empty_gt = 0
    for page in pages:
        boxes = []
        for det_idx, det in enumerate(page.gt_layout_dets):
            total_raw += 1
            poly = det.get("poly")
            if not isinstance(poly, list) or len(poly) < 8:
                continue
            gt_text, gt_source = gt_text_for_det(det)
            gt_ignore = bool(det.get("ignore", False))
            if gt_ignore and not include_ignored:
                total_skipped_ignored += 1
                continue
            if not gt_text and not include_empty_gt:
                total_skipped_empty_gt += 1
                continue
            boxes.append(
                {
                    "cls_id": None,
                    "label": str(det.get("category_type", "unknown")),
                    "score": 1.0,
                    "coordinate": clean_json(poly),
                    "gt_det_index": int(det_idx),
                    "gt_order": det.get("order"),
                    "gt_anno_id": det.get("anno_id"),
                    "gt_category_type": det.get("category_type"),
                    "gt_ignore": gt_ignore,
                    "gt_has_text": bool(gt_text),
                    "gt_text_source": gt_source,
                }
            )
        boxes.sort(key=gt_layout_box_sort_key)
        rows.append(
            {
                "selected_page_idx": int(page.idx),
                "dataset_index": int(page.dataset_index),
                "image_path": str(page.image_path),
                "image_rel": page.image_rel,
                "page_info": clean_json(page.page_info),
                "boxes": boxes,
                "raw_result_root": {
                    "source": "OmniDocBench.json layout_dets",
                    "uses_ground_truth_boxes": True,
                    "layout_det_count": int(len(boxes)),
                    "raw_layout_det_count": int(len(page.gt_layout_dets)),
                    "include_ignored": bool(include_ignored),
                    "include_empty_gt": bool(include_empty_gt),
                    "sorted_by_gt_order": True,
                },
            }
        )
    return rows, {
        "gt_layout_build_s": time.perf_counter() - start,
        "layout_detection_s": 0.0,
        "gt_layout_raw_box_count": int(total_raw),
        "gt_layout_skipped_ignored_count": int(total_skipped_ignored),
        "gt_layout_skipped_empty_gt_count": int(total_skipped_empty_gt),
    }


def gt_layout_box_sort_key(box: dict[str, Any]) -> tuple[int, int, int]:
    order = safe_int(box.get("gt_order"), default=10**9)
    has_order = 0 if box.get("gt_order") is not None else 1
    det_index = safe_int(box.get("gt_det_index"), default=10**9)
    return has_order, order, det_index


def prompt_for_label(label: str) -> str:
    return LAYOUT_PROMPT_BY_LABEL.get(str(label), "OCR:")


def gt_text_for_det(det: dict[str, Any]) -> tuple[str, str]:
    for key in ("text", "latex", "html"):
        value = det.get(key)
        if isinstance(value, str) and value.strip():
            return value, key
    return "", ""


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


def gt_poly_to_xyxy(poly: Any, width: int, height: int) -> tuple[int, int, int, int] | None:
    if not isinstance(poly, (list, tuple)) or len(poly) < 8:
        return None
    values = [float(value) for value in poly]
    xs = values[0::2]
    ys = values[1::2]
    left = max(0, int(math.floor(min(xs))))
    top = max(0, int(math.floor(min(ys))))
    right = min(int(width), int(math.ceil(max(xs))))
    bottom = min(int(height), int(math.ceil(max(ys))))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def bbox_iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    ix1 = max(left[0], right[0])
    iy1 = max(left[1], right[1])
    ix2 = min(left[2], right[2])
    iy2 = min(left[3], right[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    left_area = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    right_area = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    union = left_area + right_area - inter
    return float(inter) / float(union) if union > 0 else 0.0


def best_gt_match(
    *,
    page: PageInput,
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
) -> dict[str, Any]:
    best: dict[str, Any] = {
        "matched": False,
        "iou": 0.0,
    }
    for gt_idx, det in enumerate(page.gt_layout_dets):
        gt_bbox = gt_poly_to_xyxy(det.get("poly"), width, height)
        if gt_bbox is None:
            continue
        iou = bbox_iou(bbox, gt_bbox)
        if iou <= float(best.get("iou", 0.0)):
            continue
        gt_text, gt_source = gt_text_for_det(det)
        best = {
            "matched": True,
            "iou": float(iou),
            "gt_index": int(gt_idx),
            "gt_category_type": det.get("category_type"),
            "gt_order": det.get("order"),
            "gt_anno_id": det.get("anno_id"),
            "gt_bbox_xyxy": list(gt_bbox),
            "ground_truth": gt_text,
            "ground_truth_source": gt_source,
        }
    return best


def build_detected_crops(
    *,
    pages: list[PageInput],
    layout_pages: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[LayoutCrop], dict[str, Any], dict[str, float]]:
    timings: dict[str, float] = {}
    start = time.perf_counter()
    crops: list[LayoutCrop] = []
    skipped: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    prompt_counts: Counter[str] = Counter()
    per_page_counts: list[dict[str, Any]] = []
    page_by_dataset_idx = {int(page.dataset_index): page for page in pages}
    page_by_selected_idx = {int(page.idx): page for page in pages}
    skip_labels = {label.strip() for label in str(args.skip_labels or "").split(",") if label.strip()}
    uses_ground_truth_boxes = str(getattr(args, "layout_source", "")) == "omnidocbench_gt"

    for page_result in layout_pages:
        selected_idx = int(page_result.get("selected_page_idx", page_result.get("page_index", 0)) or 0)
        dataset_idx = int(page_result.get("dataset_index", selected_idx))
        page = page_by_dataset_idx.get(dataset_idx) or page_by_selected_idx.get(selected_idx)
        if page is None:
            raise ValueError(f"layout page result does not match selected pages: {page_result.keys()}")
        boxes = page_result.get("boxes", [])
        if not isinstance(boxes, list):
            raise ValueError(f"layout boxes for page {page.idx} are not a list")
        with Image.open(page.image_path).convert("RGB") as image:
            width, height = image.size
            kept_for_page = 0
            for box_idx, box in enumerate(boxes):
                label = str(box.get("label", "unknown"))
                label_counts[label] += 1
                if label in skip_labels:
                    skipped.append(
                        {
                            "page": int(page.idx),
                            "box": int(box_idx),
                            "label": label,
                            "reason": "skip_label",
                        }
                    )
                    continue
                bbox = clamp_box_xyxy(box.get("coordinate"), width, height, int(args.crop_padding))
                if bbox is None:
                    skipped.append(
                        {
                            "page": int(page.idx),
                            "box": int(box_idx),
                            "label": label,
                            "reason": "invalid_coordinate",
                        }
                    )
                    continue
                crop_w = int(bbox[2] - bbox[0])
                crop_h = int(bbox[3] - bbox[1])
                if crop_w < int(args.min_crop_side) or crop_h < int(args.min_crop_side):
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
                crop_image = image.crop(bbox)
                gt_match = best_gt_match(page=page, bbox=bbox, width=width, height=height)
                entry = {
                    "id": crop_id,
                    "file": f"<memory>/{crop_id}.png",
                    "source_image": str(page.image_path),
                    "image_rel": page.image_rel,
                    "page_index": int(page.idx),
                    "dataset_index": int(page.dataset_index),
                    "page_no": page.page_info.get("page_no"),
                    "page_attribute": clean_json(page.page_info.get("page_attribute", {})),
                    "layout_box_index": int(box_idx),
                    "layout_box_source_index": int(source_box_idx),
                    "layout_gt_order": clean_json(box.get("gt_order")),
                    "category_type": label,
                    "layout_label": label,
                    "layout_cls_id": clean_json(box.get("cls_id")),
                    "layout_score": clean_json(box.get("score")),
                    "layout_coordinate": clean_json(box.get("coordinate")),
                    "bbox_xyxy": list(bbox),
                    "crop_size": [crop_w, crop_h],
                    "suggested_prompt": prompt,
                    "prompt_source": "experiment6_layout_label_map",
                    "gt_layout_match": clean_json(gt_match),
                    "ground_truth_source": gt_match.get("ground_truth_source", ""),
                    "ground_truth": gt_match.get("ground_truth", ""),
                }
                crops.append(LayoutCrop(entry=entry, image=crop_image.copy()))
                kept_for_page += 1
            per_page_counts.append(
                {
                    "selected_page_idx": int(page.idx),
                    "dataset_index": int(page.dataset_index),
                    "image_rel": page.image_rel,
                    "layout_box_count": int(len(boxes)),
                    "recognizer_crop_count": int(kept_for_page),
                }
            )
    timings["crop_extract_s"] = time.perf_counter() - start
    summary = {
        "layout_box_count": int(sum(row["layout_box_count"] for row in per_page_counts)),
        "recognizer_crop_count": int(len(crops)),
        "skipped_count": int(len(skipped)),
        "skipped_samples": skipped[:16],
        "label_counts": dict(sorted(label_counts.items())),
        "prompt_counts": dict(sorted(prompt_counts.items())),
        "per_page_counts": per_page_counts,
        "crop_padding": int(args.crop_padding),
        "min_crop_side": int(args.min_crop_side),
        "skip_labels": sorted(skip_labels),
        "prompt_map": dict(sorted(LAYOUT_PROMPT_BY_LABEL.items())),
        "default_prompt_for_unmapped_labels": "OCR:",
        "uses_ground_truth_boxes": bool(uses_ground_truth_boxes),
        "gt_box_matching_for_diagnostics_only": True,
    }
    return crops, summary, timings


def preprocess_pil_crop_timed(image: Image.Image, cfg: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
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
    patches = patches.reshape(
        grid_t,
        temporal_patch_size,
        channel,
        grid_h,
        patch_size,
        grid_w,
        patch_size,
    )
    patches = patches.transpose(0, 3, 5, 2, 1, 4, 6)
    flatten_patches = patches.reshape(grid_t * grid_h * grid_w, channel, patch_size, patch_size)
    pixel_values = torch.from_numpy(flatten_patches)
    image_grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.long)
    timing["preprocess_cpu_s"] = time.perf_counter() - start
    return pixel_values, image_grid_thw, timing


def build_queue_inputs_from_crops(
    *,
    crops: list[LayoutCrop],
    tokenizer: Tokenizer,
    pre_cfg: dict[str, Any],
    prompt_override: str | None,
) -> tuple[list[QueueInput], dict[str, Any]]:
    inputs: list[QueueInput] = []
    total_start = time.perf_counter()
    for crop in crops:
        prompt = str(prompt_override if prompt_override is not None else crop.entry.get("suggested_prompt", "OCR:"))
        pixel_values, image_grid_thw, timing = preprocess_pil_crop_timed(crop.image, pre_cfg)
        token_start = time.perf_counter()
        input_ids, attention_mask = build_inputs(
            tokenizer,
            image_grid_thw,
            prompt,
            merge_size=int(pre_cfg["merge_size"]),
        )
        timing["token_build_s"] = time.perf_counter() - token_start
        timing["input_build_s"] = timing["image_read_s"] + timing["preprocess_cpu_s"] + timing["token_build_s"]
        inputs.append(
            QueueInput(
                entry=crop.entry,
                crop_path=Path(str(crop.entry["file"])),
                prompt=prompt,
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                timing_s=timing,
            )
        )
    summary = aggregate_timing_dicts([item.timing_s for item in inputs])
    summary["input_build_wall_s"] = float(time.perf_counter() - total_start)
    return inputs, summary


def aggregate_timing_dicts(rows: list[dict[str, float]]) -> dict[str, Any]:
    keys: set[str] = set()
    for row in rows:
        keys.update(row.keys())
    return {key: stats([float(row[key]) for row in rows if key in row]) for key in sorted(keys)}


def compile_decode_for_batch(
    *,
    model: LocalPaddleOCRVLForConditionalGeneration,
    model_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    backend_name: str,
    torchair_cache_dir: Path,
    batch_size: int,
    cache_length: int,
    warm_position: int,
) -> tuple[Callable, dict[str, Any], dict[str, float]]:
    flat_decode = model.make_flat_static_decode_module().eval()
    timings: dict[str, float] = {}
    maybe_sync(device)
    start = time.perf_counter()
    decode_fn, compile_meta = compile_decode_module(
        flat_decode,
        backend_name=str(backend_name),
        device=device,
        cache_root=torchair_cache_dir,
        batch_size=int(batch_size),
        cache_length=int(cache_length),
        dtype=dtype,
        model_dir=model_dir,
    )
    maybe_sync(device)
    timings["compile_wrapper_s"] = time.perf_counter() - start

    warm_cache = model.allocate_static_cache(
        batch_size=int(batch_size),
        cache_length=int(cache_length),
        device=device,
        dtype=dtype,
        init_mode="zeros",
    )
    warm_input = torch.zeros((int(batch_size), 1), device=device, dtype=torch.int64)
    warm_cache_position = torch.full(
        (int(batch_size),),
        min(int(warm_position), int(cache_length) - 1),
        device=device,
        dtype=torch.int64,
    )
    warm_rope = torch.zeros((int(batch_size), 1), device=device, dtype=torch.int64)
    maybe_sync(device)
    start = time.perf_counter()
    decode_fn(warm_input, warm_cache_position, warm_rope, *warm_cache.flat_tensors())
    maybe_sync(device)
    timings["compile_first_call_s"] = time.perf_counter() - start
    return decode_fn, compile_meta, timings


def decode_warmup_summary(compile_meta: dict[str, Any], compile_timing: dict[str, float]) -> dict[str, Any]:
    first_call_s = float(compile_timing.get("compile_first_call_s", 0.0) or 0.0)
    backend = str(compile_meta.get("backend", "unknown"))
    compile_api = str(compile_meta.get("compile_api", "unknown"))
    persistent_cache = bool(compile_meta.get("torchair_ge_cache", False))
    cache_dir = compile_meta.get("torchair_cache_dir")

    if compile_api == "none":
        state = "not_applicable_raw_eager"
        threshold_note = "No compile/cache warmup is used for raw eager decode."
    elif persistent_cache:
        if first_call_s <= 5.0:
            state = "persistent_cache_hit_or_already_warm"
        else:
            state = "persistent_cache_cold_compile_or_cache_miss"
        threshold_note = "TorchAir cache_compile is treated as warm/cache-hit when first decode call is <=5s."
    else:
        if first_call_s <= 2.0:
            state = "torch_compile_warm_or_disk_cache_hit"
        else:
            state = "torch_compile_cold_or_disk_cache_miss"
        threshold_note = "CUDA torch.compile/Inductor cache is not explicitly managed by this harness; state is inferred from first-call latency."

    return {
        "measured_decode_starts_after_warmup": True,
        "warmup_call_before_measured_decode": True,
        "warmup_input": "dummy static decode tensors with selected active_batch_size/cache_length",
        "compile_first_call_s": first_call_s,
        "cache_state": state,
        "cache_state_threshold_note": threshold_note,
        "backend": backend,
        "compile_api": compile_api,
        "persistent_cache_managed_by_harness": persistent_cache,
        "persistent_cache_dir": str(cache_dir) if cache_dir is not None else None,
    }


def page_output_summary(decoded_items: list[Any]) -> list[dict[str, Any]]:
    by_page: dict[int, list[Any]] = defaultdict(list)
    for item in decoded_items:
        by_page[int(item.item.input_item.entry.get("page_index", 0))].append(item)
    output = []
    for page_idx, items in sorted(by_page.items()):
        output.append(
            {
                "page_index": int(page_idx),
                "crop_count": int(len(items)),
                "generated_tokens_trimmed": int(sum(len(item.trimmed_token_ids) for item in items)),
                "eos_hit_count": int(sum(1 for item in items if item.eos_hit)),
                "length_cap_hit_count": int(sum(1 for item in items if item.length_cap_hit)),
                "label_counts": dict(sorted(Counter(str(item.item.input_item.entry.get("layout_label")) for item in items).items())),
                "text_samples": [item.generated_text for item in items[:3]],
            }
        )
    return output


def normalize_text_for_rough_match(text: str) -> str:
    return "".join(str(text).split()).lower()


def edit_distance(left: str | list[Any], right: str | list[Any]) -> int:
    left_seq = list(left)
    right_seq = list(right)
    if not left_seq:
        return len(right_seq)
    if not right_seq:
        return len(left_seq)
    previous = list(range(len(right_seq) + 1))
    for i, left_value in enumerate(left_seq, start=1):
        current = [i]
        for j, right_value in enumerate(right_seq, start=1):
            cost = 0 if left_value == right_value else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + cost,
                )
            )
        previous = current
    return int(previous[-1])


def normalized_edit_distance(left: str | list[Any], right: str | list[Any]) -> float:
    left_len = len(left)
    right_len = len(right)
    if left_len == 0 and right_len == 0:
        return 0.0
    if left_len == 0 or right_len == 0:
        return 1.0
    return float(edit_distance(left, right)) / float(max(left_len, right_len))


def normalize_omnidocbench_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = text.replace("\\t", "").replace("\\n", "")
    text = text.replace("\t", "").replace("\n", "")
    text = text.replace("/t", "").replace("/n", "")
    return re.sub(r"[^\w\u4e00-\u9fff]", "", text)


def strip_formula_delimiters_local(text: str) -> str:
    text = str(text or "").strip()
    pairs = [("$$", "$$"), ("\\[", "\\]"), ("\\(", "\\)"), ("$", "$")]
    changed = True
    while text and changed:
        changed = False
        for left, right in pairs:
            if text.startswith(left) and text.endswith(right) and len(text) >= len(left) + len(right):
                text = text[len(left) : len(text) - len(right)].strip()
                changed = True
                break
    return text


def normalize_omnidocbench_formula(text: str) -> str:
    text = strip_formula_delimiters_local(text)
    text = re.sub(r"\\tag\s*\{[^{}]*\}", "", text)
    text = re.sub(r"\\(?:notag|nonumber)\b", "", text)
    filters = [
        "\\mathbf",
        "\\mathrm",
        "\\mathnormal",
        "\\mathit",
        "\\mathbb",
        "\\mathcal",
        "\\mathscr",
        "\\mathfrak",
        "\\mathsf",
        "\\mathtt",
        "\\textbf",
        "\\text",
        "\\boldmath",
        "\\boldsymbol",
        "\\operatorname",
        "\\bm",
        "\\left",
        "\\right",
        "\\displaystyle",
        "\\quad",
        "\\qquad",
        "\\enspace",
        "\\space",
        "\\thinspace",
        "\\medspace",
        "\\thickspace",
        "$$",
    ]
    for token in filters:
        text = text.replace(token, "")
    text = re.sub(r"\\[!,;:]", "", text)
    text = re.sub(r"(?<!\\)&", "", text)
    text = text.replace("\\mid", "|").replace("\\vert", "|")
    text = text.replace("\\{", "").replace("\\}", "")
    text = re.sub(r"\\hspace\{.*?\}", "", text)
    text = re.sub(r"\\begin\{.*?\}", "", text)
    text = re.sub(r"\\end\{.*?\}", "", text)
    text = re.sub(r"\{[lcr| ]+\}", "", text)
    text = text.strip(".")
    previous = None
    simple_group = re.compile(r"\{([A-Za-z0-9.+\-]+)\}")
    while text != previous:
        previous = text
        text = simple_group.sub(r"\1", text)
        text = text.replace("{}", "")
        text = re.sub(r"\{\{+", "{", text)
        text = re.sub(r"}}+", "}", text)
    return re.sub(r"\s+", "", text.lower())


def tokenize_formula_for_bleu(text: str) -> list[str]:
    text = normalize_omnidocbench_formula(text)
    return re.findall(r"\\[a-zA-Z]+|\\.|\d+(?:\.\d+)?|[A-Za-z]+|.", text)


def ngram_counts(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[idx : idx + n]) for idx in range(0, len(tokens) - n + 1))


def bleu_score(prediction: str, reference: str, *, max_n: int = 4) -> float:
    pred_tokens = tokenize_formula_for_bleu(prediction)
    ref_tokens = tokenize_formula_for_bleu(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        pred_counts = ngram_counts(pred_tokens, n)
        ref_counts = ngram_counts(ref_tokens, n)
        total = sum(pred_counts.values())
        if total == 0:
            precisions.append(1.0 if len(pred_tokens) < n else 0.0)
            continue
        overlap = sum(min(count, ref_counts[gram]) for gram, count in pred_counts.items())
        precisions.append((overlap + 1.0) / (total + 1.0))
    log_precision = sum(math.log(max(value, 1e-12)) for value in precisions) / float(max_n)
    brevity = 1.0 if len(pred_tokens) > len(ref_tokens) else math.exp(1.0 - float(len(ref_tokens)) / float(len(pred_tokens)))
    return float(brevity * math.exp(log_precision))


@dataclass
class TableNode:
    tag: str
    colspan: int = 1
    rowspan: int = 1
    text: str = ""
    children: list["TableNode"] | None = None


class SimpleTableParser(HTMLParser):
    TABLE_TAGS = {"table", "thead", "tbody", "tfoot", "tr", "td", "th"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[TableNode] = []
        self.roots: list[TableNode] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag not in self.TABLE_TAGS:
            return
        attr_map = {key.lower(): value for key, value in attrs}
        if tag == "th":
            tag = "td"
        node = TableNode(
            tag=tag,
            colspan=safe_int(attr_map.get("colspan"), default=1),
            rowspan=safe_int(attr_map.get("rowspan"), default=1),
            text="",
            children=[],
        )
        if self.stack:
            self.stack[-1].children = self.stack[-1].children or []
            self.stack[-1].children.append(node)
        else:
            self.roots.append(node)
        self.stack.append(node)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "th":
            tag = "td"
        if tag not in self.TABLE_TAGS:
            return
        for idx in range(len(self.stack) - 1, -1, -1):
            if self.stack[idx].tag == tag:
                del self.stack[idx:]
                return

    def handle_data(self, data: str) -> None:
        if not self.stack:
            return
        self.stack[-1].text += data


def safe_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def normalize_table_html_for_parse(text: str) -> str:
    text = html.unescape(unicodedata.normalize("NFKC", str(text or "")))
    text = re.sub(r"```(?:html|markdown|latex)?", "", text)
    text = text.replace("```", "")
    text = re.sub(r"\s+", " ", text).strip()
    if "<table" not in text.lower() and "<fcel>" in text:
        text = fcel_table_to_html(text)
    return text


def paddle_pipeline_postprocess_text(label: str, text: str) -> str:
    result = str(text or "")
    if ("\\(" in result and "\\)" in result) or ("\\[" in result and "\\]" in result):
        result = result.replace("$", "")
        result = (
            result.replace("\\(", " $ ")
            .replace("\\)", " $")
            .replace("\\[\\[", "\\[")
            .replace("\\]\\]", "\\]")
            .replace("\\[", " $$ ")
            .replace("\\]", " $$ ")
        )
        if str(label) == "formula_number":
            result = result.replace("$", "")
    if str(label) == "table":
        normalized = normalize_table_html_for_parse(result)
        if "<table" in normalized.lower():
            result = normalized
    return result


def fcel_table_to_html(text: str) -> str:
    rows = []
    for raw_row in re.split(r"<nl\s*/?>", str(text or ""), flags=re.IGNORECASE):
        raw_row = raw_row.strip()
        if not raw_row:
            continue
        cells = [cell.strip() for cell in re.split(r"<fcel\s*/?>", raw_row, flags=re.IGNORECASE)]
        cells = [cell for cell in cells if cell]
        if cells:
            rows.append(cells)
    if not rows:
        return str(text or "")
    row_html = []
    for row in rows:
        row_html.append("<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>")
    return "<table>" + "".join(row_html) + "</table>"


def extract_table_root(text: str) -> TableNode | None:
    normalized = normalize_table_html_for_parse(text)
    if "<table" not in normalized.lower():
        return None
    parser = SimpleTableParser()
    try:
        parser.feed(normalized)
        parser.close()
    except Exception:
        return None
    for root in parser.roots:
        if root.tag == "table":
            return root
    return parser.roots[0] if parser.roots else None


def normalize_table_cell_text_local(text: str) -> str:
    text = html.unescape(unicodedata.normalize("NFKC", str(text or "")))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def table_tree_size(node: TableNode | None) -> int:
    if node is None:
        return 0
    return 1 + sum(table_tree_size(child) for child in (node.children or []))


def table_rename_cost(left: TableNode, right: TableNode, *, structure_only: bool) -> float:
    if left.tag != right.tag or left.colspan != right.colspan or left.rowspan != right.rowspan:
        return 1.0
    if structure_only or left.tag != "td":
        return 0.0
    left_text = normalize_table_cell_text_local(left.text)
    right_text = normalize_table_cell_text_local(right.text)
    return normalized_edit_distance(left_text, right_text)


def table_tree_distance(left: TableNode | None, right: TableNode | None, *, structure_only: bool) -> float:
    if left is None:
        return float(table_tree_size(right))
    if right is None:
        return float(table_tree_size(left))
    left_children = left.children or []
    right_children = right.children or []
    previous = [float(sum(table_tree_size(child) for child in right_children[:j])) for j in range(len(right_children) + 1)]
    for i, left_child in enumerate(left_children, start=1):
        current = [float(sum(table_tree_size(child) for child in left_children[:i]))]
        for j, right_child in enumerate(right_children, start=1):
            current.append(
                min(
                    previous[j] + float(table_tree_size(left_child)),
                    current[j - 1] + float(table_tree_size(right_child)),
                    previous[j - 1] + table_tree_distance(left_child, right_child, structure_only=structure_only),
                )
            )
        previous = current
    return table_rename_cost(left, right, structure_only=structure_only) + previous[-1]


def lightweight_teds(prediction: str, reference: str, *, structure_only: bool) -> float | None:
    pred_root = extract_table_root(prediction)
    ref_root = extract_table_root(reference)
    if pred_root is None or ref_root is None:
        return None
    denom = max(table_tree_size(pred_root), table_tree_size(ref_root))
    if denom <= 0:
        return None
    distance = table_tree_distance(pred_root, ref_root, structure_only=structure_only)
    return float(max(0.0, 1.0 - distance / float(denom)))


def summarize_metric_rows(rows: list[dict[str, Any]], key: str, *, lower_is_better: bool) -> dict[str, Any]:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return {
            "count": 0,
            "avg": None,
            "score": None,
            "lower_is_better": bool(lower_is_better),
        }
    avg = float(sum(values) / float(len(values)))
    return {
        "count": int(len(values)),
        "avg": avg,
        "score": float(1.0 - avg) if lower_is_better else avg,
        "lower_is_better": bool(lower_is_better),
    }


def page_average(rows: list[dict[str, Any]], key: str) -> float | None:
    page_values: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if row.get(key) is not None:
            page_values[int(row.get("page_index", 0))].append(float(row[key]))
    averages = [sum(values) / float(len(values)) for values in page_values.values() if values]
    if not averages:
        return None
    return float(sum(averages) / float(len(averages)))


def omnidocbench_metrics_without_cdm(decoded_items: list[Any], *, min_iou: float) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for idx, decoded in enumerate(decoded_items):
        entry = decoded.item.input_item.entry
        gt_match = entry.get("gt_layout_match", {}) or {}
        gt_text = str(entry.get("ground_truth", "") or "")
        if not gt_match.get("matched") or float(gt_match.get("iou", 0.0) or 0.0) < float(min_iou) or not gt_text:
            continue
        label = str(entry.get("layout_label"))
        raw_pred = str(decoded.generated_text or "")
        pred = paddle_pipeline_postprocess_text(label, raw_pred)
        base = {
            "idx": int(idx),
            "id": str(entry.get("id")),
            "page_index": int(entry.get("page_index", 0)),
            "layout_label": label,
            "gt_category_type": str(gt_match.get("gt_category_type", label)),
            "gt_order": entry.get("gt_layout_match", {}).get("gt_order", gt_match.get("gt_order")),
            "layout_box_index": int(entry.get("layout_box_index", idx)),
            "pred_sample": pred[:240],
            "raw_pred_sample": raw_pred[:240],
            "gt_sample": gt_text[:240],
        }
        if label in TEXT_DIAGNOSTIC_LABELS:
            norm_pred = normalize_omnidocbench_text(pred)
            norm_gt = normalize_omnidocbench_text(gt_text)
            rows.append(
                {
                    **base,
                    "metric_family": "text",
                    "official_component": bool(label in TEXT_EDIT_LABELS),
                    "edit_dist": normalized_edit_distance(norm_gt, norm_pred),
                    "norm_pred_len": int(len(norm_pred)),
                    "norm_gt_len": int(len(norm_gt)),
                }
            )
        elif label in FORMULA_LABELS:
            norm_pred = normalize_omnidocbench_formula(pred)
            norm_gt = normalize_omnidocbench_formula(gt_text)
            rows.append(
                {
                    **base,
                    "metric_family": "formula",
                    "official_component": True,
                    "edit_dist": normalized_edit_distance(norm_gt, norm_pred),
                    "bleu_1_4": bleu_score(pred, gt_text),
                    "cdm": None,
                    "cdm_unavailable_reason": "skipped_by_request_no_cdm_dependencies",
                    "norm_pred_len": int(len(norm_pred)),
                    "norm_gt_len": int(len(norm_gt)),
                }
            )
        elif label in TABLE_LABELS:
            norm_pred = normalize_table_html_for_parse(pred)
            norm_gt = normalize_table_html_for_parse(gt_text)
            teds = lightweight_teds(pred, gt_text, structure_only=False)
            teds_s = lightweight_teds(pred, gt_text, structure_only=True)
            rows.append(
                {
                    **base,
                    "metric_family": "table",
                    "official_component": True,
                    "edit_dist": normalized_edit_distance(norm_gt, norm_pred),
                    "teds": teds,
                    "teds_structure_only": teds_s,
                    "teds_implementation": "local_lightweight_html_tree_edit_no_apted_lxml",
                    "norm_pred_len": int(len(norm_pred)),
                    "norm_gt_len": int(len(norm_gt)),
                }
            )

    text_rows = [row for row in rows if row["metric_family"] == "text" and row["official_component"]]
    text_diag_rows = [row for row in rows if row["metric_family"] == "text"]
    formula_rows = [row for row in rows if row["metric_family"] == "formula"]
    table_rows = [row for row in rows if row["metric_family"] == "table"]
    reading_order = reading_order_edit(decoded_items, min_iou=float(min_iou))

    text_edit = summarize_metric_rows(text_rows, "edit_dist", lower_is_better=True)
    formula_edit = summarize_metric_rows(formula_rows, "edit_dist", lower_is_better=True)
    formula_bleu = summarize_metric_rows(formula_rows, "bleu_1_4", lower_is_better=False)
    table_edit = summarize_metric_rows(table_rows, "edit_dist", lower_is_better=True)
    table_teds = summarize_metric_rows(table_rows, "teds", lower_is_better=False)
    table_teds_s = summarize_metric_rows(table_rows, "teds_structure_only", lower_is_better=False)

    conclusion_scores = []
    if text_edit["score"] is not None:
        conclusion_scores.append(float(text_edit["score"]))
    if table_teds["score"] is not None:
        conclusion_scores.append(float(table_teds["score"]))
    conclusion_mean_percent = (
        None
        if not conclusion_scores
        else float(sum(conclusion_scores) / float(len(conclusion_scores)) * 100.0)
    )

    return {
        "enabled": True,
        "is_official_omnidocbench_metric": False,
        "scope": "gt_crop_predictions_vs_omnidocbench_gt_metric_families_without_cdm_or_mgam",
        "min_iou": float(min_iou),
        "matched_scored_items": int(len(rows)),
        "official_comparison_warning": (
            "Not leaderboard-comparable: this uses GT crops, no layout detector, no MGAM prediction-side segmentation, "
            "no CDM, and a lightweight local TEDS implementation."
        ),
        "reported_paddleocr_vl_1_6_reference": {
            "overall": 96.33,
            "text_edit": 0.033,
            "formula_cdm": 97.49,
            "table_teds": 94.76,
            "table_teds_structure_only": 97.11,
            "reading_order_edit": 0.127,
        },
        "leaderboard_overall": None,
        "leaderboard_overall_unavailable_reason": "CDM and official MGAM/TEDS evaluator are intentionally not run.",
        "available_non_cdm_component_mean_score_percent": conclusion_mean_percent,
        "available_non_cdm_component_mean_note": (
            "Deprecated compatibility field. Formula edit/BLEU are diagnostics only because PaddleOCR-VL "
            "reports Formula CDM; this mean now includes only text Edit_dist score and table TEDS when available."
        ),
        "text_table_conclusion_mean_score_percent": conclusion_mean_percent,
        "text_table_conclusion_components": {
            "text_block_Edit_dist_score": text_edit["score"],
            "table_TEDS_score": table_teds["score"],
            "formula_excluded_reason": "Formula edit/BLEU are not comparable to PaddleOCR-VL's reported Formula CDM.",
        },
        "text_block_Edit_dist": {
            **text_edit,
            "page_avg": page_average(text_rows, "edit_dist"),
            "score_percent": None if text_edit["score"] is None else float(text_edit["score"] * 100.0),
        },
        "text_diagnostic_Edit_dist_including_title_code": {
            **summarize_metric_rows(text_diag_rows, "edit_dist", lower_is_better=True),
            "page_avg": page_average(text_diag_rows, "edit_dist"),
        },
        "display_formula_Edit_dist": {
            **formula_edit,
            "page_avg": page_average(formula_rows, "edit_dist"),
            "score_percent": None if formula_edit["score"] is None else float(formula_edit["score"] * 100.0),
        },
        "display_formula_BLEU_1_4": {
            **formula_bleu,
            "page_avg": page_average(formula_rows, "bleu_1_4"),
            "score_percent": None if formula_bleu["score"] is None else float(formula_bleu["score"] * 100.0),
        },
        "display_formula_CDM": {
            "count": int(len(formula_rows)),
            "avg": None,
            "score": None,
            "score_percent": None,
            "available": False,
            "unavailable_reason": "skipped_by_request_no_cdm_dependencies",
        },
        "table_Edit_dist": {
            **table_edit,
            "page_avg": page_average(table_rows, "edit_dist"),
            "score_percent": None if table_edit["score"] is None else float(table_edit["score"] * 100.0),
        },
        "table_TEDS": {
            **table_teds,
            "page_avg": page_average(table_rows, "teds"),
            "score_percent": None if table_teds["score"] is None else float(table_teds["score"] * 100.0),
            "implementation": "local_lightweight_html_tree_edit_no_apted_lxml",
        },
        "table_TEDS_structure_only": {
            **table_teds_s,
            "page_avg": page_average(table_rows, "teds_structure_only"),
            "score_percent": None if table_teds_s["score"] is None else float(table_teds_s["score"] * 100.0),
            "implementation": "local_lightweight_html_tree_edit_no_apted_lxml",
        },
        "reading_order_Edit_dist": reading_order,
        "by_layout_label": summarize_rows_by_label(rows),
        "samples": rows[:24],
    }


def summarize_rows_by_label(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label in sorted(set(str(row["layout_label"]) for row in rows)):
        label_rows = [row for row in rows if str(row["layout_label"]) == label]
        metric_keys = sorted(
            {
                key
                for row in label_rows
                for key, value in row.items()
                if key in {"edit_dist", "bleu_1_4", "teds", "teds_structure_only"} and value is not None
            }
        )
        out[label] = {"count": int(len(label_rows))}
        for key in metric_keys:
            vals = [float(row[key]) for row in label_rows if row.get(key) is not None]
            out[label][f"avg_{key}"] = float(sum(vals) / float(len(vals))) if vals else None
    return out


def reading_order_edit(decoded_items: list[Any], *, min_iou: float) -> dict[str, Any]:
    page_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for idx, decoded in enumerate(decoded_items):
        entry = decoded.item.input_item.entry
        label = str(entry.get("layout_label"))
        if label not in READING_ORDER_LABELS:
            continue
        gt_match = entry.get("gt_layout_match", {}) or {}
        if not gt_match.get("matched") or float(gt_match.get("iou", 0.0) or 0.0) < float(min_iou):
            continue
        raw_order = gt_match.get("gt_order", entry.get("gt_order"))
        if raw_order is None:
            continue
        page_rows[int(entry.get("page_index", 0))].append(
            {
                "decoded_index": int(idx),
                "layout_box_index": int(entry.get("layout_box_index", idx)),
                "gt_order": safe_int(raw_order, default=int(entry.get("layout_box_index", idx))),
                "label": label,
            }
        )
    page_scores = []
    for page_index, rows in sorted(page_rows.items()):
        if not rows:
            continue
        pred_sequence = [int(row["gt_order"]) for row in sorted(rows, key=lambda row: row["decoded_index"])]
        gt_sequence = sorted(pred_sequence)
        edit = normalized_edit_distance(gt_sequence, pred_sequence)
        page_scores.append(
            {
                "page_index": int(page_index),
                "count": int(len(rows)),
                "edit_dist": float(edit),
                "gt_order": gt_sequence,
                "pred_order": pred_sequence,
            }
        )
    if not page_scores:
        return {
            "count": 0,
            "avg": None,
            "score": None,
            "lower_is_better": True,
            "scope": "gt_crop_text_component_order",
            "page_scores": [],
        }
    avg = float(sum(float(row["edit_dist"]) for row in page_scores) / float(len(page_scores)))
    return {
        "count": int(len(page_scores)),
        "avg": avg,
        "score": float(1.0 - avg),
        "lower_is_better": True,
        "scope": "gt_crop_text_component_order",
        "official_full_page_reading_order": False,
        "page_scores": page_scores[:32],
    }


def rough_ground_truth_accuracy(decoded_items: list[Any], *, min_iou: float) -> dict[str, Any]:
    rows = []
    for idx, decoded in enumerate(decoded_items):
        entry = decoded.item.input_item.entry
        gt_match = entry.get("gt_layout_match", {}) or {}
        gt_text = str(entry.get("ground_truth", "") or "")
        if not gt_match.get("matched") or float(gt_match.get("iou", 0.0) or 0.0) < float(min_iou) or not gt_text:
            continue
        label = str(entry.get("layout_label"))
        raw_pred = str(decoded.generated_text or "")
        pred = paddle_pipeline_postprocess_text(label, raw_pred)
        norm_pred = normalize_text_for_rough_match(pred)
        norm_gt = normalize_text_for_rough_match(gt_text)
        ratio = SequenceMatcher(None, norm_pred, norm_gt).ratio() if norm_pred or norm_gt else 1.0
        rows.append(
            {
                "idx": int(idx),
                "id": str(entry.get("id")),
                "page_index": int(entry.get("page_index", 0)),
                "layout_label": label,
                "gt_category_type": gt_match.get("gt_category_type"),
                "iou": float(gt_match.get("iou", 0.0) or 0.0),
                "normalized_exact": bool(norm_pred == norm_gt),
                "sequence_ratio": float(ratio),
                "pred_sample": pred[:240],
                "raw_pred_sample": raw_pred[:240],
                "gt_sample": gt_text[:240],
            }
        )
    if not rows:
        return {
            "enabled": True,
            "is_official_omnidocbench_metric": False,
            "min_iou": float(min_iou),
            "matched_text_items": 0,
            "normalized_exact_count": 0,
            "normalized_exact_rate": None,
            "avg_sequence_ratio": None,
            "samples": [],
        }
    exact = sum(1 for row in rows if row["normalized_exact"])
    return {
        "enabled": True,
        "is_official_omnidocbench_metric": False,
        "scope": "detected_crop_text_vs_best_iou_ground_truth_region",
        "min_iou": float(min_iou),
        "matched_text_items": int(len(rows)),
        "normalized_exact_count": int(exact),
        "normalized_exact_rate": float(exact) / float(len(rows)),
        "avg_sequence_ratio": float(sum(float(row["sequence_ratio"]) for row in rows) / float(len(rows))),
        "by_layout_label": {
            label: {
                "count": int(len(label_rows)),
                "normalized_exact_rate": (
                    float(sum(1 for row in label_rows if row["normalized_exact"])) / float(len(label_rows))
                    if label_rows
                    else None
                ),
                "avg_sequence_ratio": (
                    float(sum(float(row["sequence_ratio"]) for row in label_rows) / float(len(label_rows)))
                    if label_rows
                    else None
                ),
            }
            for label, label_rows in sorted(
                {
                    label: [row for row in rows if row["layout_label"] == label]
                    for label in sorted(set(str(row["layout_label"]) for row in rows))
                }.items()
            )
        },
        "samples": rows[:16],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="PaddlePaddle/PaddleOCR-VL-1.6")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--page-start", type=int, default=0)
    parser.add_argument("--num-pages", type=int, default=64)
    parser.add_argument("--layout-model-name", default="PP-DocLayoutV3")
    parser.add_argument("--layout-model-dir", type=Path, default=None)
    parser.add_argument("--layout-device", default=None)
    parser.add_argument("--layout-threshold", type=float, default=None)
    parser.add_argument("--layout-nms", type=str, choices=["true", "false"], default=None)
    parser.add_argument("--layout-unclip-ratio", type=float, default=None)
    parser.add_argument("--layout-merge-bboxes-mode", default=None)
    parser.add_argument(
        "--layout-source",
        default="omnidocbench_gt",
        choices=["omnidocbench_gt", "official", "cache"],
        help=(
            "Region source for page crops. omnidocbench_gt uses OmniDocBench.json layout_dets and imports no Paddle; "
            "official runs PaddleOCR/PaddleX layout detection; cache reads a prior layout JSON."
        ),
    )
    parser.add_argument("--layout-cache-json", type=Path, default=None)
    parser.add_argument("--reuse-layout-cache", action="store_true")
    parser.add_argument("--crop-padding", type=int, default=0)
    parser.add_argument("--min-crop-side", type=int, default=4)
    parser.add_argument("--skip-labels", default="")
    parser.add_argument(
        "--include-ignored-gt",
        action="store_true",
        help="With --layout-source omnidocbench_gt, include layout_dets marked ignore=true.",
    )
    parser.add_argument(
        "--include-empty-gt",
        action="store_true",
        help="With --layout-source omnidocbench_gt, include layout_dets that have no text/latex/html target.",
    )
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--cache-length", type=int, default=2048)
    parser.add_argument("--active-batch-size", type=int, default=8)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "float16", "bf16", "bfloat16"])
    parser.add_argument("--decode-backend", default="torchair", choices=BACKEND_CHOICES)
    parser.add_argument("--decode-schedule", default="hotswap", choices=DECODE_SCHEDULE_CHOICES)
    parser.add_argument("--eos-mode", default="overlap_event_flags", choices=EOS_MODE_CHOICES)
    parser.add_argument("--npu-jit-compile", default="off", choices=NPU_JIT_COMPILE_CHOICES)
    parser.add_argument("--torchair-cache-dir", type=Path, default=DEFAULT_TORCHAIR_CACHE_DIR)
    parser.add_argument("--validation-items", type=int, default=-1)
    parser.add_argument("--rough-gt-min-iou", type=float, default=0.5)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if int(args.num_pages) <= 0:
        raise ValueError("--num-pages must be positive")
    if int(args.active_batch_size) <= 0:
        raise ValueError("--active-batch-size must be positive")
    if str(args.decode_schedule) != "hotswap":
        raise ValueError("experiment 6 currently requires --decode-schedule hotswap")
    if str(args.eos_mode) != "overlap_event_flags":
        raise ValueError("experiment 6 hot-swap page benchmark requires --eos-mode overlap_event_flags")
    if args.layout_nms is not None:
        args.layout_nms = args.layout_nms == "true"

    model_dir = _resolve_model_dir(args.model)
    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype, device)
    configure_npu_jit_compile(args.npu_jit_compile, device)

    pages = load_pages(args.dataset_dir, page_start=int(args.page_start), num_pages=int(args.num_pages))

    layout_source = "cache" if args.reuse_layout_cache else str(args.layout_source)
    if layout_source == "cache":
        if args.layout_cache_json is None:
            raise ValueError("--layout-source cache/--reuse-layout-cache requires --layout-cache-json")
        layout_pages, layout_timing = load_layout_cache(args.layout_cache_json, pages)
        layout_cache_mode = "read_existing"
    elif layout_source == "omnidocbench_gt":
        layout_pages, layout_timing = build_omnidocbench_gt_layout_pages(
            pages,
            include_ignored=bool(args.include_ignored_gt),
            include_empty_gt=bool(args.include_empty_gt),
        )
        layout_cache_mode = "omnidocbench_gt_layout_dets"
        if args.layout_cache_json is not None:
            write_layout_cache(
                args.layout_cache_json,
                layout_pages,
                {
                    "layout_source": layout_source,
                    "page_start": int(args.page_start),
                    "num_pages": int(args.num_pages),
                    "dataset_dir": str(resolve_dataset_dir(args.dataset_dir)),
                    "uses_ground_truth_boxes": True,
                    "include_ignored_gt": bool(args.include_ignored_gt),
                    "include_empty_gt": bool(args.include_empty_gt),
                },
            )
    elif layout_source == "official":
        args.layout_source = "official"
        layout_pages, layout_timing = run_layout_detection(pages=pages, args=args)
        layout_cache_mode = "fresh_official_layout_detection"
        if args.layout_cache_json is not None:
            write_layout_cache(
                args.layout_cache_json,
                layout_pages,
                {
                    "layout_source": layout_source,
                    "layout_model_name": args.layout_model_name,
                    "layout_model_dir": str(args.layout_model_dir) if args.layout_model_dir else None,
                    "layout_device": args.layout_device,
                    "page_start": int(args.page_start),
                    "num_pages": int(args.num_pages),
                    "dataset_dir": str(resolve_dataset_dir(args.dataset_dir)),
                    "uses_ground_truth_boxes": False,
                },
            )
    else:
        raise ValueError(f"unsupported layout_source={layout_source!r}")
    args.layout_source = layout_source

    crops, crop_summary, crop_timing = build_detected_crops(pages=pages, layout_pages=layout_pages, args=args)
    if not crops:
        raise RuntimeError("layout detection produced zero recognizer crops")
    if int(args.active_batch_size) > len(crops):
        raise ValueError(
            f"--active-batch-size {args.active_batch_size} exceeds detected crops {len(crops)}; "
            "experiment 6 does not create fake decode rows"
        )

    pre_cfg = load_preprocessor_config(model_dir)
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    queue_inputs, input_build_summary = build_queue_inputs_from_crops(
        crops=crops,
        tokenizer=tokenizer,
        pre_cfg=pre_cfg,
        prompt_override=args.prompt,
    )

    cache_preflight = prompt_token_summary(
        queue_inputs,
        max_new_tokens=int(args.max_new_tokens),
        cache_length=int(args.cache_length),
    )
    if int(cache_preflight["overflow_count"]) > 0:
        output = {
            "experiment": "06_full_page_pipeline_e2e",
            "error": "cache_length_too_small",
            "page_count": int(len(pages)),
            "recognizer_crop_count": int(len(queue_inputs)),
            "active_batch_size": int(args.active_batch_size),
            "cache_preflight": cache_preflight,
            "layout_cache_mode": layout_cache_mode,
            "layout_timing_s": layout_timing,
            "crop_summary": crop_summary,
            "input_build_summary_s": input_build_summary,
        }
        print(json.dumps(output, indent=2, sort_keys=True, default=json_default))
        return

    setup_timing: dict[str, float] = {}
    maybe_sync(device)
    start = time.perf_counter()
    model = LocalPaddleOCRVLForConditionalGeneration.from_pretrained(model_dir, dtype=dtype, device=device)
    maybe_sync(device)
    setup_timing["recognizer_model_load_s"] = time.perf_counter() - start

    maybe_sync(device)
    start = time.perf_counter()
    weight_format_meta = cast_decode_linear_weights_to_nz(model)
    maybe_sync(device)
    setup_timing["decode_weight_format_s"] = time.perf_counter() - start

    decode_fn, compile_meta, compile_timing = compile_decode_for_batch(
        model=model,
        model_dir=model_dir,
        device=device,
        dtype=dtype,
        backend_name=str(args.decode_backend),
        torchair_cache_dir=args.torchair_cache_dir,
        batch_size=int(args.active_batch_size),
        cache_length=int(args.cache_length),
        warm_position=int(cache_preflight["input_tokens"]["max"]),
    )
    setup_timing.update(compile_timing)

    ready_start = time.perf_counter()
    ready_items, ready_bank, ready_details = build_ready_bank_incremental(
        model=model,
        queue_inputs=queue_inputs,
        cache_length=int(args.cache_length),
        device=device,
        vision_prefill_batch_size=1,
    )
    maybe_sync(device)
    ready_bank_build_s = time.perf_counter() - ready_start

    hotswap_external_overlap_buffer_setup_s = 0.0
    overlap_cpu_tokens = None
    overlap_copy_stream = None
    if device.type == "npu":
        import torch_npu

        start = time.perf_counter()
        overlap_cpu_tokens = torch.empty((1, int(args.active_batch_size)), dtype=ready_bank.next_token.dtype, pin_memory=True)
        overlap_copy_stream = torch_npu.npu.Stream(device=device)
        hotswap_external_overlap_buffer_setup_s = time.perf_counter() - start

    eos_token_id = int(model.config.eos_token_id)
    decode_start = time.perf_counter()
    hotswap_result = static_hotswap_decode_loop(
        decode_fn=decode_fn,
        ready=ready_bank,
        batch_size=int(args.active_batch_size),
        eos_token_id=eos_token_id,
        max_new_tokens=int(args.max_new_tokens),
        eos_mode=str(args.eos_mode),
        overlap_cpu_tokens=overlap_cpu_tokens,
        overlap_copy_stream=overlap_copy_stream,
    )
    maybe_sync(device)
    decode_queue_s = time.perf_counter() - decode_start

    postprocess_start = time.perf_counter()
    hotswap_rows = [[int(value) for value in row] for row in hotswap_result.ids.tolist()]
    hotswap_lengths = [int(value) for value in hotswap_result.lengths.tolist()]
    decoded_items = [
        materialize_hotswap_item(
            ready_item=ready_item,
            token_ids=hotswap_rows[idx],
            length=hotswap_lengths[idx],
            tokenizer=tokenizer,
            eos_token_id=eos_token_id,
            max_new_tokens=int(args.max_new_tokens),
        )
        for idx, ready_item in enumerate(ready_items)
    ]
    decode_output_postprocess_s = time.perf_counter() - postprocess_start

    validation = validate_outputs(
        model=model,
        decoded=decoded_items,
        device=device,
        cache_length=int(args.cache_length),
        max_new_tokens=int(args.max_new_tokens),
        eos_token_id=eos_token_id,
        max_items=int(args.validation_items),
    )
    trimmed_rows = [item.trimmed_token_ids for item in decoded_items]
    token_summary = token_range_summary(trimmed_rows, vocab_size=int(model.config.text_config.vocab_size))
    length_cap_hit_count = int(sum(1 for item in decoded_items if item.length_cap_hit))
    rough_accuracy = rough_ground_truth_accuracy(decoded_items, min_iou=float(args.rough_gt_min_iou))
    omnidocbench_metrics = omnidocbench_metrics_without_cdm(decoded_items, min_iou=float(args.rough_gt_min_iou))

    layout_detection_s = float(layout_timing.get("layout_detection_s", 0.0) or 0.0)
    crop_extract_s = float(crop_timing.get("crop_extract_s", 0.0) or 0.0)
    input_build_wall_s = float(input_build_summary.get("input_build_wall_s", 0.0) or 0.0)
    measured_e2e_s = (
        layout_detection_s
        + crop_extract_s
        + input_build_wall_s
        + float(ready_bank_build_s)
        + float(decode_queue_s)
        + float(decode_output_postprocess_s)
    )
    total_decode_calls = int(hotswap_result.decode_calls)
    raw_decode_token_calls = int(total_decode_calls) * int(args.active_batch_size)
    effective_decode_token_calls = int(sum(max(0, len(item.trimmed_token_ids) - 1) for item in decoded_items))
    generated_tokens_including_prefill_first = int(sum(len(item.trimmed_token_ids) for item in decoded_items))

    output = {
        "experiment": "06_full_page_pipeline_e2e",
        "page_pipeline_scope": (
            "full pages -> selected layout source, default OmniDocBench GT boxes -> crops -> "
            "local PaddleOCR-VL recognizer prefill per crop -> hotswap batched text decode"
        ),
        "uses_ground_truth_layout_boxes": bool(layout_source == "omnidocbench_gt"),
        "doc_layout_model_measured": bool(layout_source == "official"),
        "omnidocbench_scoring": False,
        "model": str(model_dir),
        "dataset_dir": str(resolve_dataset_dir(args.dataset_dir)),
        "device": str(device),
        "dtype": str(dtype),
        "npu_jit_compile": str(args.npu_jit_compile),
        "page_start": int(args.page_start),
        "page_count": int(len(pages)),
        "recognizer_crop_count": int(len(decoded_items)),
        "active_batch_size": int(args.active_batch_size),
        "prefill_batch_size": 1,
        "decode_schedule": "hotswap",
        "decode_backend": str(args.decode_backend),
        "decode_attention": DECODE_ATTENTION if device.type == "npu" else "manual",
        "decode_cache_update": DECODE_CACHE_UPDATE if device.type == "npu" else "per_row_copy",
        "eos_mode": str(args.eos_mode),
        "max_new_tokens": int(args.max_new_tokens),
        "cache_length": int(args.cache_length),
        "layout": {
            "source": layout_source,
            "cache_mode": layout_cache_mode,
            "model_name": str(args.layout_model_name),
            "model_dir": str(args.layout_model_dir) if args.layout_model_dir else None,
            "device": args.layout_device,
            "threshold": args.layout_threshold,
            "layout_nms": args.layout_nms,
            "layout_unclip_ratio": args.layout_unclip_ratio,
            "layout_merge_bboxes_mode": args.layout_merge_bboxes_mode,
            "cache_json": str(args.layout_cache_json) if args.layout_cache_json else None,
            "include_ignored_gt": bool(args.include_ignored_gt),
            "include_empty_gt": bool(args.include_empty_gt),
        },
        "setup_timing_s": {
            **layout_timing,
            **setup_timing,
        },
        "linear_weight_format": weight_format_meta,
        "compile": compile_meta,
        "decode_warmup": decode_warmup_summary(compile_meta, compile_timing),
        "cache_preflight": cache_preflight,
        "crop_summary": crop_summary,
        "input_build_summary_s": input_build_summary,
        "ready_bank_build_details": ready_details,
        "ready_item_timing_summary_s": aggregate_timing_dicts([item.timing_s for item in ready_items]),
        "vision_shape_bucket_summary": vision_token_bucket_summary(ready_items),
        "phase_timing_s": {
            "layout_detection": layout_detection_s,
            "crop_extract": crop_extract_s,
            "recognizer_cpu_input_build": input_build_wall_s,
            "recognizer_ready_bank_build": float(ready_bank_build_s),
            "hotswap_external_overlap_buffer_setup": float(hotswap_external_overlap_buffer_setup_s),
            "text_decode_queue": float(decode_queue_s),
            "decode_output_postprocess": float(decode_output_postprocess_s),
            "measured_e2e_page_pipeline_excluding_setup_and_validation": float(measured_e2e_s),
            "validation": float(validation.get("elapsed_s", 0.0) or 0.0),
        },
        "throughput": {
            "pages_per_s_measured_e2e": tok_per_s(len(pages), measured_e2e_s),
            "seconds_per_page_measured_e2e": None if not pages else measured_e2e_s / float(len(pages)),
            "crops_per_s_measured_e2e": tok_per_s(len(decoded_items), measured_e2e_s),
            "layout_pages_per_s": tok_per_s(len(pages), layout_detection_s),
            "layout_boxes_per_s": tok_per_s(int(crop_summary["layout_box_count"]), layout_detection_s),
            "crop_extract_crops_per_s": tok_per_s(len(decoded_items), crop_extract_s),
            "prefill_crops_per_s": tok_per_s(len(decoded_items), ready_bank_build_s),
            "decode_crops_per_s": tok_per_s(len(decoded_items), decode_queue_s),
            "decode_calls_per_s": tok_per_s(total_decode_calls, decode_queue_s),
            "raw_decode_token_calls_per_s": tok_per_s(raw_decode_token_calls, decode_queue_s),
            "effective_decode_tokens_per_s": tok_per_s(effective_decode_token_calls, decode_queue_s),
        },
        "decode_summary": {
            "decode_calls": int(total_decode_calls),
            "raw_decode_token_calls": int(raw_decode_token_calls),
            "effective_decode_token_calls": int(effective_decode_token_calls),
            "generated_tokens_including_prefill_first": int(generated_tokens_including_prefill_first),
            "swap_event_count": int(len(hotswap_result.swap_events)),
            "total_swapped_in_items": int(
                sum(len(event.get("swapped_in_item_ids", [])) for event in hotswap_result.swap_events)
            ),
            "stopped_all_items": bool(hotswap_result.stopped_all_items),
            "eos_hit_count": int(sum(1 for item in decoded_items if item.eos_hit)),
            "length_cap_hit_count": int(length_cap_hit_count),
            "trimmed_new_tokens": stats([float(len(item.trimmed_token_ids)) for item in decoded_items]),
            "hotswap_phase_timing_s": hotswap_result.phase_timing_s,
            "hotswap_diagnostics": hotswap_result.diagnostics,
        },
        "token_id_range": token_summary,
        "correctness": {
            **validation,
            "invalid_token_count": int(token_summary["invalid_count"]),
            "length_cap_hit_count": int(length_cap_hit_count),
            "length_cap_is_required_failure": True,
            "required_checks_scope": "hotswap_detected_crop_outputs_vs_same_local_static_reference_and_token_id_range",
            "ground_truth_checked": False,
            "all_required_checks_passed": bool(
                validation.get("all_required_checks_passed", False)
                and int(token_summary["invalid_count"]) == 0
                and int(length_cap_hit_count) == 0
            ),
        },
        "omnidocbench_metrics_without_cdm": omnidocbench_metrics,
        "rough_ground_truth_accuracy": rough_accuracy,
        "pages": page_output_summary(decoded_items),
        "items_sample": [
            {
                "idx": int(idx),
                "id": str(decoded.item.input_item.entry.get("id")),
                "page_index": int(decoded.item.input_item.entry.get("page_index", 0)),
                "layout_box_index": int(decoded.item.input_item.entry.get("layout_box_index", 0)),
                "layout_label": str(decoded.item.input_item.entry.get("layout_label")),
                "layout_score": decoded.item.input_item.entry.get("layout_score"),
                "bbox_xyxy": decoded.item.input_item.entry.get("bbox_xyxy"),
                "crop_size": decoded.item.input_item.entry.get("crop_size"),
                "prompt": decoded.item.input_item.prompt,
                "input_tokens": int(decoded.item.input_item.input_ids.shape[1]),
                "vision_tokens": int(decoded.item.vision_tokens),
                "projected_image_tokens": int(decoded.item.projected_image_tokens),
                "generated_tokens_trimmed": int(len(decoded.trimmed_token_ids)),
                "decode_calls": int(decoded.decode_calls),
                "eos_hit": bool(decoded.eos_hit),
                "length_cap_hit": bool(decoded.length_cap_hit),
                "generated_text": decoded.generated_text,
                "paddle_pipeline_postprocessed_text": paddle_pipeline_postprocess_text(
                    str(decoded.item.input_item.entry.get("layout_label")),
                    decoded.generated_text,
                ),
                "gt_layout_match": decoded.item.input_item.entry.get("gt_layout_match"),
                "ground_truth_source": decoded.item.input_item.entry.get("ground_truth_source"),
                "ground_truth_sample": str(decoded.item.input_item.entry.get("ground_truth", ""))[:240],
            }
            for idx, decoded in enumerate(decoded_items[:32])
        ],
        "texts": {
            "sample": [item.generated_text for item in decoded_items[: min(8, len(decoded_items))]],
            "paddle_pipeline_postprocessed_sample": [
                paddle_pipeline_postprocess_text(str(item.item.input_item.entry.get("layout_label")), item.generated_text)
                for item in decoded_items[: min(8, len(decoded_items))]
            ],
        },
        "stage_notes": {
            "layout_detection": (
                "When layout_source=official, this is official PaddleOCR/PaddleX LayoutDetection over full page images. "
                "When layout_source=omnidocbench_gt, no document layout model is run and layout_detection is zero; "
                "ignored and empty-target GT boxes are skipped unless explicitly included."
            ),
            "crop_extract": (
                "Boxes are clamped to page bounds and cropped from page images. "
                "With layout_source=omnidocbench_gt these boxes come from OmniDocBench.json layout_dets."
            ),
            "prompt_mapping": "Because prompt_label is only effective when official layout detection is disabled, this harness maps layout labels to recognizer prompts explicitly and reports the map.",
            "prefill": "Recognizer CPU preprocessing plus vision/projector/text prefill are per detected crop; prefill batch size is fixed at 1 in experiment 6.",
            "decode": "All detected crop ready states are decoded by the experiment-5 hot-swap scheduler with one active compiled batch; no fake rows are added.",
            "measured_e2e": "Excludes layout/model init, recognizer model load, decode weight format conversion, torch compile/cache warm, and validation. Includes layout inference only when layout_source=official; with layout_source=omnidocbench_gt it measures crop extraction, recognizer input build, prefill, decode, and output postprocess.",
            "decode_warmup": "The decode callable is invoked once on dummy static-cache inputs before the measured decode queue. measured_e2e_page_pipeline_excluding_setup_and_validation therefore reports the post-warmup decode path; setup_timing_s.compile_first_call_s and decode_warmup.cache_state label whether that warmup looked cold or cache-hot.",
            "validation": "Validation is outside the timing window and checks hot-swap output against the same local static recognizer per crop. It is not an OCR quality metric.",
            "quality_metrics": (
                "omnidocbench_metrics_without_cdm reports local GT-crop metrics aligned with OmniDocBench metric families "
                "where practical: text Edit_dist, table Edit_dist and lightweight TEDS/TEDS-S, and GT-crop reading-order "
                "Edit_dist. Formula Edit_dist/BLEU are retained as diagnostics only and are excluded from conclusion-style "
                "means because PaddleOCR-VL reports Formula CDM. This harness intentionally does not compute CDM or "
                "official MGAM matching. GT-crop metrics use lightweight Paddle-style output postprocessing before scoring."
            ),
        },
    }

    print(json.dumps(output, indent=2, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
