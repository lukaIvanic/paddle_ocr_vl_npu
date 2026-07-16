#!/usr/bin/env python3
"""Run real PP-DocLayoutV3 plus sequential B=1 PaddleOCR-VL recognition."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from engine import SequentialRecognizer
from layout import PPDocLayoutV3Runtime
from pipeline import OfflinePagePipeline, aggregate_pages
from run_local_recognition import NPU_JIT_COMPILE_CHOICES, resolve_device
from schema import RunResult


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "tmp" / "08_offline_e2e_b1"
DEFAULT_CACHE_ROOT = REPO_ROOT / ".runtime_cache" / "08_offline_e2e_b1_torchair"
DEFAULT_LOCAL_RECOGNIZER = Path("/workspace/models/PaddleOCR-VL-1.6")
DEFAULT_RECOGNIZER = str(DEFAULT_LOCAL_RECOGNIZER) if DEFAULT_LOCAL_RECOGNIZER.is_dir() else "PaddlePaddle/PaddleOCR-VL-1.6"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, action="append", required=True)
    parser.add_argument("--layout-model", type=Path, default=Path("/workspace/models/PP-DocLayoutV3_safetensors"))
    parser.add_argument("--recognizer-model", default=DEFAULT_RECOGNIZER)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", default="fp16", choices=("fp16", "float16", "bf16", "bfloat16"))
    parser.add_argument("--layout-threshold", type=float, default=0.3)
    parser.add_argument("--decode-backend", default="torchair", choices=("raw_eager", "eager", "default", "torchair"))
    parser.add_argument("--cache-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--npu-jit-compile", default="off", choices=NPU_JIT_COMPILE_CHOICES)
    parser.add_argument("--torchair-cache-dir", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--recognize-chart", action="store_true")
    parser.add_argument("--recognize-seal", action="store_true")
    parser.add_argument("--recognize-image", action="store_true")
    parser.add_argument("--max-regions", type=int, default=None, help="Debug-only cap; omitted means the full page.")
    parser.add_argument("--save-crops", action="store_true")
    parser.add_argument("--no-save-annotated", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    images = [path.expanduser().resolve() for path in args.image]
    missing = [str(path) for path in images if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"input images not found: {missing}")
    if args.max_regions is not None and args.max_regions <= 0:
        raise ValueError("--max-regions must be positive when supplied")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else DEFAULT_OUTPUT_ROOT / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_started = time.perf_counter()
    device = resolve_device(args.device)
    layout = PPDocLayoutV3Runtime(args.layout_model, device, threshold=args.layout_threshold)
    recognizer = SequentialRecognizer(
        model=args.recognizer_model,
        device=args.device,
        dtype=args.dtype,
        decode_backend=args.decode_backend,
        cache_length=args.cache_length,
        max_new_tokens=args.max_new_tokens,
        torchair_cache_dir=args.torchair_cache_dir.expanduser().resolve(),
        npu_jit_compile=args.npu_jit_compile,
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
    pages = [pipeline.run_page(path, index) for index, path in enumerate(images)]
    configuration = {
        **recognizer.configuration(),
        "layout_model": str(args.layout_model.expanduser().resolve()),
        "layout_threshold": float(args.layout_threshold),
        "layout_runtime": "transformers.AutoModelForObjectDetection",
        "layout_source": "real_pp_doclayout_v3_inference",
        "region_execution": "strict_sequential_b1",
        "recognize_chart": bool(args.recognize_chart),
        "recognize_seal": bool(args.recognize_seal),
        "recognize_image": bool(args.recognize_image),
        "max_regions_debug_limit": args.max_regions,
    }
    result = RunResult(
        experiment="08_offline_e2e_b1",
        configuration=configuration,
        setup_timing_s={
            **layout.setup_timing_s,
            **recognizer.setup_timing_s,
            "total": float(setup_total_s),
        },
        pages=pages,
        aggregate=aggregate_pages(pages),
        metric_definitions={
            "layout_inference": "Synchronized wall time for the real PP-DocLayoutV3 model call only.",
            "page_total": "Image open through real layout, all sequential recognitions, and reading-order text assembly; setup and diagnostic artifact writes are excluded.",
            "page_total_including_artifacts": "page_total plus page text and optional annotated-image/crop writes.",
            "request_total": "In-memory crop preprocessing through eager prefill, compiled decode, D2H token transfer, and detokenization.",
            "device_stage_s": "Accelerator event time for eager vision/text-prefill sub-stages; it is not host wall time.",
            "decode_effective_tok_per_s": "Generated tokens after the prefill-produced first token, including EOS, divided by compiled decode wall time.",
            "decode_executed_calls_per_s": "All executed compiled graph calls divided by decode wall time; one look-ahead call can occur while asynchronously detecting EOS.",
            "e2e_output_tok_per_s": "All generated tokens including each first token and EOS divided by summed page wall time.",
        },
    )
    output_path = output_dir / "run.json"
    output_path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    aggregate = result.aggregate
    print(f"output={output_path}")
    print(
        f"pages={aggregate['pages']} layout_regions={aggregate['layout_regions']} "
        f"recognized_regions={aggregate['recognized_regions']} partial_pages={aggregate['partial_pages']}"
    )
    print(
        f"page_wall_s={aggregate['sum_page_wall_s']:.6f} "
        f"decode_wall_s={aggregate['sum_compiled_decode_wall_s']:.6f} "
        f"decode_tok_per_s={aggregate['rates']['decode_effective_tok_per_s']} "
        f"e2e_output_tok_per_s={aggregate['rates']['e2e_output_tok_per_s']}"
    )


if __name__ == "__main__":
    main()
