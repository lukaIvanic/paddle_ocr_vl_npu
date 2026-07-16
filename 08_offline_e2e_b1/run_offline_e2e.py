#!/usr/bin/env python3
"""Run real layout and run-scoped cross-page PaddleOCR-VL recognition."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path

import torch

from engine import ContinuousRecognizer
from layout import PPDocLayoutV3Runtime
from pipeline import OfflinePagePipeline, aggregate_pages
from run_local_recognition import NPU_JIT_COMPILE_CHOICES, resolve_device
from runtime_defaults import (
    DECODE_BACKEND_CHOICES,
    DEFAULT_CACHE_LENGTH,
    DEFAULT_DECODE_BACKEND,
    DEFAULT_DECODE_BATCH_SIZE,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEXT_BACKEND,
    DEFAULT_VISION_BACKEND,
    OPTIMIZED_TEXT_BUCKETS,
    OPTIMIZED_VISION_BUCKETS,
)
from schema import RunResult
from text_compile import TEXT_BACKEND_CHOICES, parse_text_buckets
from vision_compile import VISION_BACKEND_CHOICES, parse_vision_buckets


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "tmp" / "08_offline_e2e_b1"
DEFAULT_CACHE_ROOT = REPO_ROOT / ".runtime_cache" / "08_offline_e2e_b1_torchair"
DEFAULT_VISION_CACHE_ROOT = REPO_ROOT / ".runtime_cache" / "08_offline_e2e_b1_vision_torchair"
DEFAULT_TEXT_CACHE_ROOT = REPO_ROOT / ".runtime_cache" / "08_offline_e2e_b1_text_torchair"
DEFAULT_LOCAL_RECOGNIZER = Path("/workspace/models/PaddleOCR-VL-1.6")
DEFAULT_RECOGNIZER = str(DEFAULT_LOCAL_RECOGNIZER) if DEFAULT_LOCAL_RECOGNIZER.is_dir() else "PaddlePaddle/PaddleOCR-VL-1.6"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, action="append", required=True)
    parser.add_argument("--layout-model", type=Path, default=Path("/workspace/models/PP-DocLayoutV3_safetensors"))
    parser.add_argument("--recognizer-model", default=DEFAULT_RECOGNIZER)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", default="fp16", choices=("fp16", "float16", "bf16", "bfloat16"))
    parser.add_argument("--layout-threshold", type=float, default=0.3)
    parser.add_argument(
        "--decode-backend",
        default=DEFAULT_DECODE_BACKEND,
        choices=DECODE_BACKEND_CHOICES,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_DECODE_BATCH_SIZE,
        help="Persistent decode-arena capacity; must be a power of two.",
    )
    parser.add_argument("--cache-length", type=int, default=DEFAULT_CACHE_LENGTH)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
    )
    parser.add_argument(
        "--preprocessor-min-pixels",
        type=int,
        default=None,
        help="Override only the recognition-crop min_pixels resize floor; preserve the model's max_pixels.",
    )
    parser.add_argument("--npu-jit-compile", default="off", choices=NPU_JIT_COMPILE_CHOICES)
    parser.add_argument("--torchair-cache-dir", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument(
        "--vision-backend",
        default=DEFAULT_VISION_BACKEND,
        choices=VISION_BACKEND_CHOICES,
        help="Use eager vision, or one static TorchAir encoder graph per configured token bucket.",
    )
    parser.add_argument(
        "--vision-compile-buckets",
        default=",".join(str(bucket) for bucket in OPTIMIZED_VISION_BUCKETS),
        help="Strictly increasing comma-separated positive sequence lengths used by compiled B=1 vision.",
    )
    parser.add_argument(
        "--vision-torchair-cache-dir",
        type=Path,
        default=DEFAULT_VISION_CACHE_ROOT,
    )
    parser.add_argument(
        "--text-backend",
        default=DEFAULT_TEXT_BACKEND,
        choices=TEXT_BACKEND_CHOICES,
        help="Use eager text prefill, or one static TorchAir transformer graph per token bucket.",
    )
    parser.add_argument(
        "--text-compile-buckets",
        default=",".join(str(bucket) for bucket in OPTIMIZED_TEXT_BUCKETS),
        help="Strictly increasing comma-separated positive sequence lengths used by compiled B=1 text prefill.",
    )
    parser.add_argument(
        "--text-torchair-cache-dir",
        type=Path,
        default=DEFAULT_TEXT_CACHE_ROOT,
    )
    parser.add_argument("--recognize-chart", action="store_true")
    parser.add_argument("--recognize-seal", action="store_true")
    parser.add_argument("--recognize-image", action="store_true")
    parser.add_argument("--max-regions", type=int, default=None, help="Debug-only cap; omitted means the full page.")
    parser.add_argument("--save-crops", action="store_true")
    parser.add_argument("--no-save-annotated", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    images = [path.expanduser().resolve() for path in args.image]
    missing = [str(path) for path in images if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"input images not found: {missing}")
    if args.max_regions is not None and args.max_regions <= 0:
        raise ValueError("--max-regions must be positive when supplied")
    if args.batch_size <= 0 or args.batch_size & (args.batch_size - 1):
        raise ValueError("--batch-size must be a positive power of two")
    vision_buckets = parse_vision_buckets(args.vision_compile_buckets)
    text_buckets = parse_text_buckets(args.text_compile_buckets)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else DEFAULT_OUTPUT_ROOT / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_started = time.perf_counter()
    device_started = time.perf_counter()
    device = resolve_device(args.device)
    device_runtime_init_s = time.perf_counter() - device_started
    layout = PPDocLayoutV3Runtime(args.layout_model, device, threshold=args.layout_threshold)
    recognizer = ContinuousRecognizer(
        model=args.recognizer_model,
        device=args.device,
        dtype=args.dtype,
        decode_backend=args.decode_backend,
        batch_size=args.batch_size,
        cache_length=args.cache_length,
        max_new_tokens=args.max_new_tokens,
        torchair_cache_dir=args.torchair_cache_dir.expanduser().resolve(),
        vision_backend=args.vision_backend,
        vision_buckets=vision_buckets,
        vision_torchair_cache_dir=args.vision_torchair_cache_dir.expanduser().resolve(),
        text_backend=args.text_backend,
        text_buckets=text_buckets,
        text_torchair_cache_dir=args.text_torchair_cache_dir.expanduser().resolve(),
        npu_jit_compile=args.npu_jit_compile,
        preprocessor_min_pixels=args.preprocessor_min_pixels,
    )
    setup_total_s = time.perf_counter() - setup_started

    pipeline = OfflinePagePipeline(
        layout=layout,
        recognizer=recognizer,
        recognize_chart=args.recognize_chart,
        recognize_seal=args.recognize_seal,
        recognize_image=args.recognize_image,
        max_regions=args.max_regions,
        artifact_dir=output_dir,
        save_crops=args.save_crops,
        save_annotated=not args.no_save_annotated,
    )
    pipeline_run = pipeline.run_pages(
        images,
        on_page_completed=lambda page: print(
            f"page_completed={page.page_id} "
            f"recognized_regions={len(page.recognized_regions)} "
            f"latency_s={page.timing_s['page_total']:.6f}"
        ),
    )
    pages = pipeline_run.pages
    configuration = {
        **recognizer.configuration(),
        "layout_model": str(args.layout_model.expanduser().resolve()),
        "layout_threshold": float(args.layout_threshold),
        "layout_runtime": "transformers.AutoModelForObjectDetection",
        "layout_source": "real_pp_doclayout_v3_inference",
        "region_execution": "lazy_sequential_prefill_run_scoped_continuous_decode",
        "cross_page_decode": True,
        "page_completion": "emit_when_all_page_regions_complete",
        "recognize_chart": bool(args.recognize_chart),
        "recognize_seal": bool(args.recognize_seal),
        "recognize_image": bool(args.recognize_image),
        "max_regions_debug_limit": args.max_regions,
    }
    result = RunResult(
        experiment="08_offline_e2e_b1",
        configuration=configuration,
        setup_timing_s={
            "device_runtime_init": float(device_runtime_init_s),
            **layout.setup_timing_s,
            **recognizer.setup_timing_s,
            "total": float(setup_total_s),
        },
        pages=pages,
        decode_schedule=pipeline_run.decode_schedule,
        aggregate=aggregate_pages(
            pages,
            pipeline_run.decode_schedule,
            run_wall_s=pipeline_run.run_wall_s,
        ),
        metric_definitions={
            "layout_inference": "Synchronized wall time for the real PP-DocLayoutV3 model call only.",
            "page_total": "Per-page latency from beginning that page's image load through its immediate completion emission; pages can overlap in the run-scoped scheduler.",
            "run_wall_s": "Wall time from starting the first page until every page has emitted; this is the E2E throughput denominator.",
            "page_total_including_artifacts": "page_total plus page text and optional annotated-image/crop writes.",
            "request_total": "In-memory crop preprocessing through prefill, compiled decode, D2H token transfer, and detokenization.",
            "device_stage_s": "Accelerator event time for vision/text-prefill sub-stages; it is not host wall time.",
            "vision_encoder_post_layernorm_compiled": "NPU event time for the selected static padded vision-encoder graph plus its final LayerNorm; input preparation is reported separately.",
            "vision_useful_token_fraction": "Sum of real vision tokens divided by sum of physical bucket tokens; eager and overflow requests have no padding.",
            "text_prefill_compiled": "NPU event time for the selected static padded text-transformer graph, including in-place KV cache population.",
            "text_useful_token_fraction": "Sum of real text-prefill tokens divided by sum of physical bucket tokens; eager and overflow requests have no padding.",
            "continuous_decode_wall": "Decode throughput denominator: the larger of exclusive host-control wall time and serialized decode-plus-admission device time, preventing overlapped producer synchronization from understating decode cost.",
            "run_scoped_scheduler_wall": "Full recognition scheduler wall including lazy page production, eager prefill, decode, detokenization, and page-completion callbacks.",
            "raw_decode_tok_per_s": "Every persistent-arena token slot executed by the compiled graph divided by continuous-decode wall time, including idle slots and one-step completion look-ahead.",
            "effective_decode_tok_per_s": "Real generated tokens after each prefill-produced first token, including EOS but excluding idle and look-ahead slots, divided by the same decode-control wall time.",
            "active_slot_fraction": "Slots assigned to a real request when an iteration launched divided by all fixed-arena slots. It includes the one delayed look-ahead iteration.",
            "e2e_output_tok_per_s": "All generated tokens including each first token and EOS divided by run wall time.",
        },
    )
    result.aggregate["page_completion_order"] = pipeline_run.completion_order
    output_path = output_dir / "run.json"
    output_path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    aggregate = result.aggregate
    print(f"output={output_path}")
    print(
        f"pages={aggregate['pages']} layout_regions={aggregate['layout_regions']} "
        f"recognized_regions={aggregate['recognized_regions']} partial_pages={aggregate['partial_pages']}"
    )
    print(
        f"run_wall_s={aggregate['run_wall_s']:.6f} "
        f"decode_wall_s={aggregate['continuous_decode_wall_s']:.6f} "
        f"raw_decode_tok_per_s={aggregate['rates']['raw_decode_tok_per_s']} "
        f"effective_decode_tok_per_s={aggregate['rates']['effective_decode_tok_per_s']} "
        f"e2e_output_tok_per_s={aggregate['rates']['e2e_output_tok_per_s']}"
    )


if __name__ == "__main__":
    main()
