#!/usr/bin/env python3
"""Run real PaddleOCR-VL generation through a selected decode boundary.

This is a text-decode lab, not a synthetic operator probe.  It obtains one
real recognition crop from the owned layout frontend, duplicates that exact
request into every decode slot, and then runs the production recognizer from
image preprocessing through vision/text prefill and autoregressive decoding.
The repeated cohort deliberately gives every slot the same prompt length and
generation path so a B16 boundary is exercised uniformly.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import torch

HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
REPO_ROOT = EXPERIMENT_ROOT.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

from paddleocr_vl.model.text_decode import decode_optimization_names
from paddleocr_vl.serving.engine import ContinuousRecognizer
from paddleocr_vl.serving.runtime_defaults import (
    OPTIMIZED_TEXT_BUCKETS,
    OPTIMIZED_VISION_BUCKETS,
)
from paddleocr_vl.serving.types import RecognitionRequest, RecognitionResult
from utils.timing import synchronize


DEFAULT_PAGE_IMAGE = Path(
    "/workspace/datasets/OmniDocBench/images/"
    "page-573c437e-c309-4483-a038-ef2f440b104a.png"
)
DEFAULT_LAYOUT_MODEL = Path("/workspace/models/PP-DocLayoutV3_safetensors")
DEFAULT_RECOGNIZER_MODEL = Path("/workspace/models/PaddleOCR-VL-1.6")
DEFAULT_CACHE_ROOT = (
    REPO_ROOT / ".runtime_cache/09_text_decode_real_generation"
)
BOUNDARY_PROGRESS_EVENTS = (
    "diagnostic_pending_state",
    "diagnostic_compute_sync_begin",
    "diagnostic_compute_sync_end",
    "diagnostic_compute_sync_error",
    "diagnostic_d2h_sync_begin",
    "diagnostic_d2h_sync_end",
    "diagnostic_d2h_sync_error",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page-image", type=Path, default=DEFAULT_PAGE_IMAGE)
    parser.add_argument("--layout-model", type=Path, default=DEFAULT_LAYOUT_MODEL)
    parser.add_argument(
        "--layout-device",
        choices=("cpu", "npu"),
        default="cpu",
        help="Layout only selects the crop and is excluded from generation timing.",
    )
    parser.add_argument(
        "--layout-model-backend",
        choices=("owned", "transformers"),
        default="owned",
    )
    parser.add_argument("--block-index", type=int, default=3)
    parser.add_argument("--expected-prompt", default="Table Recognition:")
    parser.add_argument("--expected-crop-width", type=int, default=1022)
    parser.add_argument("--expected-crop-height", type=int, default=772)
    parser.add_argument("--expected-input-tokens", type=int, default=1021)
    parser.add_argument("--recognizer-model", type=Path, default=DEFAULT_RECOGNIZER_MODEL)
    parser.add_argument("--decode-cache-dir", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument(
        "--decode-optimization",
        choices=decode_optimization_names(),
        default="combined_apply_mha_repeat",
    )
    parser.add_argument(
        "--decode-backend",
        choices=("raw_eager", "torchair"),
        default="torchair",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--replicas", type=int, default=16)
    parser.add_argument("--cache-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--target-effective-length", type=int, default=1280)
    parser.add_argument("--min-pixels", type=int, default=28224)
    parser.add_argument(
        "--vision-backend",
        choices=("raw_eager", "torchair"),
        default="raw_eager",
    )
    parser.add_argument(
        "--text-backend",
        choices=("raw_eager", "torchair"),
        default="raw_eager",
    )
    parser.add_argument(
        "--vision-attention",
        choices=("manual", "prompt_flash_attention"),
        default="prompt_flash_attention",
    )
    parser.add_argument(
        "--vision-promptfa-align-128",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.batch_size & (args.batch_size - 1):
        parser.error("--batch-size must be a positive power of two")
    if args.replicas != args.batch_size:
        parser.error("this uniform-boundary test requires replicas == batch-size")
    if args.cache_length <= args.target_effective_length:
        parser.error("--cache-length must exceed --target-effective-length")
    if args.max_new_tokens <= 1:
        parser.error("--max-new-tokens must exceed one")
    if args.vision_promptfa_align_128 and args.vision_attention != "prompt_flash_attention":
        parser.error("--vision-promptfa-align-128 requires prompt_flash_attention")
    return args


def _token_hash(token_ids: list[int]) -> str:
    payload = json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _first_divergence(left: list[int], right: list[int]) -> dict[str, Any] | None:
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return {
                "index": index,
                "actual": left_token,
                "reference": right_token,
            }
    if len(left) != len(right):
        index = min(len(left), len(right))
        return {
            "index": index,
            "actual": left[index] if index < len(left) else None,
            "reference": right[index] if index < len(right) else None,
        }
    return None


def _reference_comparison(
    reference_path: Path | None,
    rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if reference_path is None:
        return None
    path = reference_path.expanduser().resolve()
    reference = json.loads(path.read_text(encoding="utf-8"))
    if reference.get("kind") != "text_decode_real_generation":
        raise ValueError(f"not a real-generation reference: {path}")
    reference_rows = {
        str(row["request_id"]): row for row in reference.get("results", ())
    }
    comparisons: list[dict[str, Any]] = []
    for row in rows:
        request_id = str(row["request_id"])
        if request_id not in reference_rows:
            raise ValueError(f"reference is missing request {request_id}")
        expected = reference_rows[request_id]
        actual_tokens = [int(value) for value in row["token_ids"]]
        expected_tokens = [int(value) for value in expected["token_ids"]]
        comparisons.append(
            {
                "request_id": request_id,
                "token_ids_exact": actual_tokens == expected_tokens,
                "text_exact": row["text"] == expected["text"],
                "stop_reason_exact": (
                    row["stop_reason"] == expected["stop_reason"]
                ),
                "first_token_divergence": _first_divergence(
                    actual_tokens,
                    expected_tokens,
                ),
            }
        )
    return {
        "path": str(path),
        "requests": len(comparisons),
        "all_token_ids_exact": all(
            row["token_ids_exact"] for row in comparisons
        ),
        "all_text_exact": all(row["text_exact"] for row in comparisons),
        "all_stop_reasons_exact": all(
            row["stop_reason_exact"] for row in comparisons
        ),
        "per_request": comparisons,
    }


def _select_real_request(args: argparse.Namespace) -> tuple[RecognitionRequest, dict[str, Any]]:
    from pipeline.layout_frontend import OwnedLayoutFrontend

    page_image = args.page_image.expanduser().resolve()
    layout_model = args.layout_model.expanduser().resolve()
    if not page_image.is_file():
        raise FileNotFoundError(page_image)
    if not layout_model.is_dir():
        raise FileNotFoundError(layout_model)
    device = torch.device(args.layout_device if args.layout_device == "cpu" else "npu:0")
    frontend = OwnedLayoutFrontend(
        layout_model,
        device,
        graph_capture=False,
        device_stage_timing=device.type == "npu",
        model_backend=args.layout_model_backend,
        model_dtype=torch.float32 if device.type == "cpu" else torch.float16,
    )
    started = time.perf_counter()
    prepared = frontend.prepare_page(
        page_image,
        0,
        min_pixels=args.min_pixels,
    )
    layout_wall_s = time.perf_counter() - started
    matches = [
        request
        for request, block_index in zip(
            prepared.requests,
            prepared.request_block_indices,
            strict=True,
        )
        if int(block_index) == args.block_index
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one layout block {args.block_index}, got {len(matches)}"
        )
    request = matches[0]
    crop_size = (request.crop.width, request.crop.height)
    expected_size = (args.expected_crop_width, args.expected_crop_height)
    if request.prompt != args.expected_prompt:
        raise AssertionError(
            f"prompt drift: expected {args.expected_prompt!r}, got {request.prompt!r}"
        )
    if crop_size != expected_size:
        raise AssertionError(
            f"crop-size drift: expected {expected_size}, got {crop_size}"
        )
    selected = RecognitionRequest(
        request_id="boundary_real_source",
        crop=request.crop.copy(),
        prompt=request.prompt,
        skip_special_tokens=request.skip_special_tokens,
        min_pixels=request.min_pixels,
        max_pixels=request.max_pixels,
    )
    metadata = {
        "page_image": str(page_image),
        "layout_model": str(layout_model),
        "layout_device": str(device),
        "layout_model_backend": args.layout_model_backend,
        "layout_wall_s": layout_wall_s,
        "layout_request_count": len(prepared.requests),
        "selected_block_index": args.block_index,
        "selected_prompt": request.prompt,
        "selected_crop_size": list(crop_size),
    }
    del prepared, frontend
    gc.collect()
    if device.type == "npu":
        synchronize(device)
        torch.npu.empty_cache()
    return selected, metadata


def _result_row(
    result: RecognitionResult,
    target_effective_length: int,
) -> dict[str, Any]:
    row = asdict(result)
    maximum_written_effective_length = (
        int(result.input_tokens) + int(result.decode_calls_executed)
    )
    boundary_token_offset = target_effective_length - int(result.input_tokens)
    start = max(0, boundary_token_offset - 4)
    end = min(len(result.token_ids), boundary_token_offset + 5)
    row["token_sha256"] = _token_hash(result.token_ids)
    row["maximum_written_effective_length"] = maximum_written_effective_length
    row["crossed_target_effective_length"] = (
        maximum_written_effective_length >= target_effective_length
    )
    row["target_token_window"] = {
        "target_offset_from_first_generated_token": boundary_token_offset,
        "start": start,
        "end": end,
        "token_ids": result.token_ids[start:end],
    }
    return row


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    from pipeline.layout_mask_guard import install_layout_mask_guard

    install_layout_mask_guard()

    import torch_npu  # noqa: F401

    if not torch.npu.is_available():
        raise RuntimeError("real text-decode generation requires an available NPU")
    torch.npu.set_compile_mode(jit_compile=False)
    device = torch.device("npu:0")

    source_request, source = _select_real_request(args)
    requests = [
        RecognitionRequest(
            request_id=f"boundary_real_{index:02d}",
            crop=source_request.crop.copy(),
            prompt=source_request.prompt,
            skip_special_tokens=source_request.skip_special_tokens,
            min_pixels=source_request.min_pixels,
            max_pixels=source_request.max_pixels,
        )
        for index in range(args.replicas)
    ]

    setup_started = time.perf_counter()
    recognizer = ContinuousRecognizer(
        model=str(args.recognizer_model.expanduser().resolve()),
        dtype="fp16",
        decode_backend=args.decode_backend,
        decode_optimization=args.decode_optimization,
        batch_size=args.batch_size,
        cache_length=args.cache_length,
        max_new_tokens=args.max_new_tokens,
        torchair_cache_dir=args.decode_cache_dir.expanduser().resolve(),
        vision_backend=args.vision_backend,
        vision_attention=args.vision_attention,
        vision_buckets=OPTIMIZED_VISION_BUCKETS,
        vision_padding="auto",
        vision_promptfa_align_128=args.vision_promptfa_align_128,
        vision_packing="off",
        text_backend=args.text_backend,
        text_buckets=OPTIMIZED_TEXT_BUCKETS,
        text_padding="auto",
        text_packing="off",
        preprocessor_min_pixels=args.min_pixels,
        scheduler_progress=True,
        scheduler_progress_events=BOUNDARY_PROGRESS_EVENTS,
        diagnostic_decode_effective_length=args.target_effective_length,
    )
    setup_s = time.perf_counter() - setup_started
    configuration = recognizer.configuration()
    if configuration["decode_optimization"] != args.decode_optimization:
        raise AssertionError("recognizer did not retain selected decode optimization")

    synchronize(device)
    torch.npu.reset_peak_memory_stats(device)
    memory_before = int(torch.npu.memory_allocated(device))
    results: list[RecognitionResult] = []
    run_started = time.perf_counter()
    schedule = recognizer.run(
        requests,
        schedule_id="real_b16_decode_boundary",
        emit_result=results.append,
    )
    synchronize(device)
    run_wall_s = time.perf_counter() - run_started
    peak_memory = int(torch.npu.max_memory_allocated(device))
    memory_after = int(torch.npu.memory_allocated(device))

    results.sort(key=lambda result: result.request_id)
    rows = [
        _result_row(result, args.target_effective_length)
        for result in results
    ]
    if len(rows) != args.replicas:
        raise AssertionError(f"expected {args.replicas} results, got {len(rows)}")
    input_lengths = {int(row["input_tokens"]) for row in rows}
    if input_lengths != {args.expected_input_tokens}:
        raise AssertionError(
            f"input-token drift: expected {args.expected_input_tokens}, got {input_lengths}"
        )
    if not all(row["crossed_target_effective_length"] for row in rows):
        failed = [
            row["request_id"]
            for row in rows
            if not row["crossed_target_effective_length"]
        ]
        raise AssertionError(
            f"real generations did not cross effective length "
            f"{args.target_effective_length}: {failed}"
        )

    token_hashes = {str(row["token_sha256"]) for row in rows}
    texts = {str(row["text"]) for row in rows}
    stops = {str(row["stop_reason"]) for row in rows}
    reference = _reference_comparison(args.reference, rows)
    payload = {
        "schema_version": 1,
        "kind": "text_decode_real_generation",
        "configuration": {
            "batch_size": args.batch_size,
            "replicas": args.replicas,
            "cache_length": args.cache_length,
            "max_new_tokens": args.max_new_tokens,
            "target_effective_length": args.target_effective_length,
            "decode_backend": args.decode_backend,
            "decode_optimization": args.decode_optimization,
            "vision_backend": args.vision_backend,
            "vision_attention": args.vision_attention,
            "vision_promptfa_align_128": args.vision_promptfa_align_128,
            "text_backend": args.text_backend,
            "min_pixels": args.min_pixels,
        },
        "source": source,
        "recognizer_configuration": configuration,
        "setup_s": setup_s,
        "run_wall_s": run_wall_s,
        "schedule": asdict(schedule),
        "memory_bytes": {
            "before_run": memory_before,
            "after_run": memory_after,
            "peak": peak_memory,
            "peak_delta_from_before_run": peak_memory - memory_before,
        },
        "cohort": {
            "requests": len(rows),
            "input_tokens": sorted(input_lengths),
            "generated_token_lengths": [
                int(row["generated_tokens_including_eos"]) for row in rows
            ],
            "all_crossed_target": True,
            "all_token_ids_identical": len(token_hashes) == 1,
            "all_text_identical": len(texts) == 1,
            "stop_reasons": sorted(stops),
            "token_hashes": sorted(token_hashes),
        },
        "reference_comparison": reference,
        "results": rows,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(
        "REAL_DECODE_GENERATION "
        f"optimization={args.decode_optimization} "
        f"requests={len(rows)} input_tokens={sorted(input_lengths)} "
        f"generated={payload['cohort']['generated_token_lengths'][0]} "
        f"crossed={payload['cohort']['all_crossed_target']} "
        f"identical={payload['cohort']['all_token_ids_identical']} "
        f"raw_tok_s={schedule.rates['raw_decode_tok_per_s']:.1f} "
        f"effective_tok_s={schedule.rates['effective_decode_tok_per_s']:.1f} "
        f"run_wall_s={run_wall_s:.3f}",
        flush=True,
    )
    print(f"output={output}", flush=True)


if __name__ == "__main__":
    main()
