#!/usr/bin/env python3
"""Benchmark only the PaddleOCR-VL vision prefill path on real page crops.

This is an experiment-6 side harness. It intentionally reuses the same
OmniDocBench page loading, GT layout crop extraction, crop preprocessing, and
prompt/input construction as the full page pipeline, then stops after the
native-resolution vision encoder plus adaptive MLP projector.

Measured vision call per crop:

    CPU preprocessed crop tensor -> device transfer -> model.get_image_features()

No text-token embedding, image embedding scatter, KV prefill, LM head, decode,
or output validation is run here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
EXP5_DIR = REPO_ROOT / "05_full_recognizer_optimizations"
if str(EXP5_DIR) not in sys.path:
    sys.path.insert(0, str(EXP5_DIR))

from bench_page_pipeline_e2e import (  # noqa: E402
    DEFAULT_DATASET_DIR,
    aggregate_timing_dicts,
    build_detected_crops,
    build_omnidocbench_gt_layout_pages,
    build_queue_inputs_from_crops,
    clean_json,
    load_pages_result,
    page_load_summary,
    resolve_dataset_dir,
    tok_per_s,
)
from bench_recognizer_queue import QueueInput, json_default, stats  # noqa: E402
from local_modeling_paddleocr_vl import (  # noqa: E402
    VISION_ATTENTION_CHOICES,
    VISION_ATTENTION_ENV,
    VISION_PROMPT_FA_LAYOUT_CHOICES,
    VISION_PROMPT_FA_LAYOUT_ENV,
    LocalPaddleOCRVLForConditionalGeneration,
    _resolve_model_dir,
    get_vision_attention_impl,
    get_vision_prompt_fa_layout,
)
from probe_static_compile import maybe_sync  # noqa: E402
from run_local_recognition import (  # noqa: E402
    NPU_JIT_COMPILE_CHOICES,
    configure_npu_jit_compile,
    load_preprocessor_config,
    parse_dtype,
    resolve_device,
)


MODE_CHOICES = ("sync_per_crop", "unsynced_loop")


def parse_modes(raw: str) -> list[str]:
    modes = [item.strip() for item in str(raw).replace(",", " ").split() if item.strip()]
    if not modes:
        raise ValueError("--modes must select at least one mode")
    bad = [mode for mode in modes if mode not in MODE_CHOICES]
    if bad:
        raise ValueError(f"unsupported mode(s): {bad}; choices={MODE_CHOICES}")
    deduped: list[str] = []
    for mode in modes:
        if mode not in deduped:
            deduped.append(mode)
    return deduped


def tensor_grid(item: QueueInput) -> list[int]:
    return [int(value) for value in item.image_grid_thw.reshape(-1).tolist()]


def vision_tokens(item: QueueInput) -> int:
    return int(item.image_grid_thw.prod().item())


def projected_tokens(item: QueueInput, *, merge_size: int) -> int:
    return int(vision_tokens(item) // int(merge_size) // int(merge_size))


def summarize_inputs(inputs: list[QueueInput], *, merge_size: int) -> dict[str, Any]:
    vision_counts = [vision_tokens(item) for item in inputs]
    projected_counts = [projected_tokens(item, merge_size=merge_size) for item in inputs]
    input_counts = [int(item.input_ids.shape[1]) for item in inputs]
    grids = [tuple(tensor_grid(item)) for item in inputs]
    grid_counts = Counter(grids)
    label_counts = Counter(str(item.entry.get("layout_label", "unknown")) for item in inputs)
    prompt_counts = Counter(str(item.prompt) for item in inputs)
    crop_sizes = [
        [int(value) for value in clean_json(item.entry.get("crop_size", [0, 0]))]
        for item in inputs
    ]
    return {
        "count": int(len(inputs)),
        "total_vision_tokens": int(sum(vision_counts)),
        "total_projected_image_tokens": int(sum(projected_counts)),
        "total_input_tokens": int(sum(input_counts)),
        "vision_tokens": {
            "stats": stats([float(value) for value in vision_counts]),
            "unique_count": int(len(set(vision_counts))),
        },
        "projected_image_tokens": {
            "stats": stats([float(value) for value in projected_counts]),
            "unique_count": int(len(set(projected_counts))),
        },
        "input_tokens": {
            "stats": stats([float(value) for value in input_counts]),
            "unique_count": int(len(set(input_counts))),
        },
        "image_grid_thw": {
            "unique_count": int(len(grid_counts)),
            "top_buckets": [
                {"grid": [int(value) for value in grid], "count": int(count)}
                for grid, count in sorted(grid_counts.items(), key=lambda item: (-item[1], item[0]))[:16]
            ],
        },
        "label_counts": dict(sorted(label_counts.items())),
        "prompt_counts": dict(sorted(prompt_counts.items())),
        "crop_size_samples": crop_sizes[:16],
    }


@torch.inference_mode()
def run_vision_one(
    *,
    model: LocalPaddleOCRVLForConditionalGeneration,
    item: QueueInput,
    device: torch.device,
) -> torch.Tensor:
    pixel_values = item.pixel_values.to(device=device, dtype=model.visual.dtype)
    return model.get_image_features(pixel_values, item.image_grid_thw)


@torch.inference_mode()
def warmup_vision(
    *,
    model: LocalPaddleOCRVLForConditionalGeneration,
    inputs: list[QueueInput],
    device: torch.device,
    warmup_items: int,
) -> dict[str, Any]:
    count = min(max(0, int(warmup_items)), len(inputs))
    if count <= 0:
        return {"count": 0, "elapsed_s": 0.0, "item_ids": []}
    maybe_sync(device)
    start = time.perf_counter()
    outputs = []
    for item in inputs[:count]:
        outputs.append(run_vision_one(model=model, item=item, device=device))
    maybe_sync(device)
    elapsed = time.perf_counter() - start
    return {
        "count": int(count),
        "elapsed_s": float(elapsed),
        "item_ids": [str(item.entry.get("id")) for item in inputs[:count]],
        "projected_shapes": [[int(dim) for dim in output.shape] for output in outputs],
    }


@torch.inference_mode()
def run_sync_per_crop(
    *,
    model: LocalPaddleOCRVLForConditionalGeneration,
    inputs: list[QueueInput],
    device: torch.device,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    total_start = time.perf_counter()
    for idx, item in enumerate(inputs):
        maybe_sync(device)
        start = time.perf_counter()
        output = run_vision_one(model=model, item=item, device=device)
        maybe_sync(device)
        elapsed = time.perf_counter() - start
        rows.append(
            {
                "idx": int(idx),
                "id": str(item.entry.get("id")),
                "page_index": int(item.entry.get("page_index", 0)),
                "layout_label": str(item.entry.get("layout_label", "")),
                "vision_tokens": int(vision_tokens(item)),
                "projected_image_tokens": int(output.shape[0]),
                "input_tokens": int(item.input_ids.shape[1]),
                "image_grid_thw": tensor_grid(item),
                "elapsed_s": float(elapsed),
            }
        )
    total_s = time.perf_counter() - total_start
    return summarize_mode("sync_per_crop", rows=rows, total_s=total_s)


@torch.inference_mode()
def run_unsynced_loop(
    *,
    model: LocalPaddleOCRVLForConditionalGeneration,
    inputs: list[QueueInput],
    device: torch.device,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    maybe_sync(device)
    start = time.perf_counter()
    for idx, item in enumerate(inputs):
        output = run_vision_one(model=model, item=item, device=device)
        rows.append(
            {
                "idx": int(idx),
                "id": str(item.entry.get("id")),
                "page_index": int(item.entry.get("page_index", 0)),
                "layout_label": str(item.entry.get("layout_label", "")),
                "vision_tokens": int(vision_tokens(item)),
                "projected_image_tokens": int(output.shape[0]),
                "input_tokens": int(item.input_ids.shape[1]),
                "image_grid_thw": tensor_grid(item),
            }
        )
    maybe_sync(device)
    total_s = time.perf_counter() - start
    return summarize_mode("unsynced_loop", rows=rows, total_s=total_s)


def summarize_mode(mode: str, *, rows: list[dict[str, Any]], total_s: float) -> dict[str, Any]:
    total_vision_tokens = int(sum(int(row.get("vision_tokens", 0)) for row in rows))
    total_projected_tokens = int(sum(int(row.get("projected_image_tokens", 0)) for row in rows))
    input_tokens = int(sum(int(row.get("input_tokens", 0)) for row in rows))
    elapsed_rows = [float(row["elapsed_s"]) for row in rows if "elapsed_s" in row]
    return {
        "mode": mode,
        "measurement_scope": (
            "per crop: CPU preprocessed pixel tensor -> device transfer -> native-resolution visual encoder "
            "+ post layernorm + adaptive MLP projector"
        ),
        "sync_policy": (
            "device synchronize before and after every crop"
            if mode == "sync_per_crop"
            else "one device synchronize before the loop and one after the full loop; no per-crop sync"
        ),
        "count": int(len(rows)),
        "total_s": float(total_s),
        "items_per_s": tok_per_s(len(rows), total_s),
        "vision_tokens_per_s": tok_per_s(total_vision_tokens, total_s),
        "projected_image_tokens_per_s": tok_per_s(total_projected_tokens, total_s),
        "input_tokens_per_s": tok_per_s(input_tokens, total_s),
        "total_vision_tokens": int(total_vision_tokens),
        "total_projected_image_tokens": int(total_projected_tokens),
        "total_input_tokens": int(input_tokens),
        "per_crop_elapsed_s": stats(elapsed_rows) if elapsed_rows else None,
        "samples": rows[:16],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="PaddlePaddle/PaddleOCR-VL-1.6")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--page-start", type=int, default=0)
    parser.add_argument("--num-pages", type=int, default=8)
    parser.add_argument("--layout-source", default="omnidocbench_gt", choices=["omnidocbench_gt"])
    parser.add_argument("--crop-padding", type=int, default=0)
    parser.add_argument("--min-crop-side", type=int, default=4)
    parser.add_argument("--skip-labels", default="")
    parser.add_argument("--include-ignored-gt", action="store_true")
    parser.add_argument("--include-empty-gt", action="store_true")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "float16", "bf16", "bfloat16"])
    parser.add_argument("--npu-jit-compile", default="off", choices=NPU_JIT_COMPILE_CHOICES)
    parser.add_argument("--vision-attention", default=os.environ.get(VISION_ATTENTION_ENV, "manual"), choices=VISION_ATTENTION_CHOICES)
    parser.add_argument(
        "--vision-prompt-fa-layout",
        default=os.environ.get(VISION_PROMPT_FA_LAYOUT_ENV, "bnsd"),
        choices=VISION_PROMPT_FA_LAYOUT_CHOICES,
    )
    parser.add_argument(
        "--modes",
        default="sync_per_crop,unsynced_loop",
        help="Comma/space separated modes. Choices: sync_per_crop, unsynced_loop.",
    )
    parser.add_argument("--warmup-items", type=int, default=1)
    parser.add_argument(
        "--max-crops",
        type=int,
        default=0,
        help="Optional development cap after crop extraction/input preprocessing. 0 means all selected crops.",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if int(args.num_pages) <= 0:
        raise ValueError("--num-pages must be positive")
    modes = parse_modes(args.modes)

    os.environ[VISION_ATTENTION_ENV] = str(args.vision_attention)
    os.environ[VISION_PROMPT_FA_LAYOUT_ENV] = str(args.vision_prompt_fa_layout)

    model_dir = _resolve_model_dir(args.model)
    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype, device)
    configure_npu_jit_compile(args.npu_jit_compile, device)

    page_load = load_pages_result(
        args.dataset_dir,
        page_start=int(args.page_start),
        num_pages=int(args.num_pages),
    )
    pages = page_load.pages

    layout_pages, layout_timing = build_omnidocbench_gt_layout_pages(
        pages,
        include_ignored=bool(args.include_ignored_gt),
        include_empty_gt=bool(args.include_empty_gt),
    )
    crops, crop_summary, crop_timing = build_detected_crops(pages=pages, layout_pages=layout_pages, args=args)
    if not crops:
        raise RuntimeError("OmniDocBench GT layout produced zero recognizer crops")
    raw_extracted_crop_count = int(len(crops))
    if int(args.max_crops) > 0:
        crops = crops[: int(args.max_crops)]
    if not crops:
        raise RuntimeError("zero crops after --max-crops")

    pre_cfg = load_preprocessor_config(model_dir)
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    queue_inputs, input_build_summary = build_queue_inputs_from_crops(
        crops=crops,
        tokenizer=tokenizer,
        pre_cfg=pre_cfg,
        prompt_override=args.prompt,
    )
    if not queue_inputs:
        raise RuntimeError("zero queue inputs after --max-crops")

    setup_timing: dict[str, float] = {}
    maybe_sync(device)
    start = time.perf_counter()
    model = LocalPaddleOCRVLForConditionalGeneration.from_pretrained(model_dir, dtype=dtype, device=device)
    maybe_sync(device)
    setup_timing["recognizer_model_load_s"] = time.perf_counter() - start

    warmup = warmup_vision(
        model=model,
        inputs=queue_inputs,
        device=device,
        warmup_items=int(args.warmup_items),
    )

    mode_results: dict[str, Any] = {}
    for mode in modes:
        if mode == "sync_per_crop":
            mode_results[mode] = run_sync_per_crop(model=model, inputs=queue_inputs, device=device)
        elif mode == "unsynced_loop":
            mode_results[mode] = run_unsynced_loop(model=model, inputs=queue_inputs, device=device)
        else:
            raise AssertionError(mode)

    comparisons: dict[str, Any] = {}
    if "sync_per_crop" in mode_results and "unsynced_loop" in mode_results:
        sync_s = float(mode_results["sync_per_crop"]["total_s"])
        unsynced_s = float(mode_results["unsynced_loop"]["total_s"])
        comparisons["unsynced_vs_sync_per_crop"] = {
            "speedup": (sync_s / unsynced_s) if unsynced_s > 0 else None,
            "saved_s": float(sync_s - unsynced_s),
            "sync_per_crop_total_s": sync_s,
            "unsynced_loop_total_s": unsynced_s,
            "note": (
                "This isolates per-crop device synchronization overhead for the same crop/preprocess/model path. "
                "Run order and warmup are reported because first-use kernel behavior can still affect small runs."
            ),
        }

    output = {
        "experiment": "06_vision_prefill_only",
        "scope": (
            "full pages -> OmniDocBench GT layout boxes -> real OCR crops -> PaddleOCR-VL crop preprocessing "
            "-> vision device transfer + native-resolution visual encoder + adaptive MLP projector only"
        ),
        "not_run": [
            "document layout detector",
            "text token embedding",
            "image embedding scatter into text sequence",
            "mRoPE index construction",
            "KV cache allocation/prefill",
            "LM head prefill",
            "text decode/hot-swap",
            "OCR text validation or accuracy scoring",
        ],
        "model": str(model_dir),
        "dataset_dir": str(resolve_dataset_dir(args.dataset_dir)),
        "device": str(device),
        "dtype": str(dtype),
        "npu_jit_compile": str(args.npu_jit_compile),
        "vision_attention": get_vision_attention_impl(),
        "vision_prompt_fa_layout": get_vision_prompt_fa_layout(),
        "page_start": int(args.page_start),
        "page_count": int(len(pages)),
        "page_load": page_load_summary(page_load),
        "layout": {
            "source": "omnidocbench_gt",
            "uses_ground_truth_boxes": True,
            "include_ignored_gt": bool(args.include_ignored_gt),
            "include_empty_gt": bool(args.include_empty_gt),
        },
        "recognizer_crop_count": int(len(queue_inputs)),
        "raw_extracted_crop_count_before_max_crops": int(raw_extracted_crop_count),
        "max_crops": None if int(args.max_crops) <= 0 else int(args.max_crops),
        "prompt_override": args.prompt,
        "crop_summary": crop_summary,
        "input_summary": summarize_inputs(queue_inputs, merge_size=int(pre_cfg["merge_size"])),
        "layout_timing_s": layout_timing,
        "crop_timing_s": crop_timing,
        "input_build_summary_s": input_build_summary,
        "queue_input_timing_summary_s": aggregate_timing_dicts([item.timing_s for item in queue_inputs]),
        "setup_timing_s": setup_timing,
        "warmup": warmup,
        "mode_order": modes,
        "mode_order_note": (
            "Modes run sequentially in the listed order after warmup. For tiny runs, rerun once or set "
            "--warmup-items high enough to avoid first-use effects."
        ),
        "modes": mode_results,
        "comparisons": comparisons,
    }

    if args.json:
        print(json.dumps(output, indent=2, sort_keys=True, default=json_default))
    else:
        summary = {
            "experiment": output["experiment"],
            "device": output["device"],
            "vision_attention": output["vision_attention"],
            "page_count": output["page_count"],
            "recognizer_crop_count": output["recognizer_crop_count"],
            "comparisons": comparisons,
            "modes": {
                key: {
                    "total_s": value["total_s"],
                    "items_per_s": value["items_per_s"],
                    "vision_tokens_per_s": value["vision_tokens_per_s"],
                }
                for key, value in mode_results.items()
            },
        }
        print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
