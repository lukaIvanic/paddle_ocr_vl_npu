#!/usr/bin/env python3
"""Run the single-file optimized UniRec path on the six smoke crops."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from modeling_optimized_unirec import OptimizedUniRecRunner


EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parent
DEFAULT_IMAGES = [
    PROJECT_ROOT / "crops/crop_01_text_block_en.png",
    PROJECT_ROOT / "crops/crop_02_equation_matrix.png",
    PROJECT_ROOT / "crops/crop_03_code_block.png",
    PROJECT_ROOT / "crops/crop_04_handwritten_title_zh.png",
    PROJECT_ROOT / "crops/crop_05_table_rwkv_dims.png",
    PROJECT_ROOT / "crops/crop_06_chart_cubic_spline.png",
]


def configure_npu_jit_compile(mode: str, device: str) -> None:
    if mode == "default" or not str(device).startswith("npu"):
        return
    try:
        import torch
        import torch_npu  # noqa: F401

        requested = mode == "on"
        torch.npu.set_compile_mode(jit_compile=requested)
        print(f"[npu] set torch.npu compile mode: jit_compile={requested}", flush=True)
    except Exception as exc:
        raise RuntimeError(f"failed to set NPU jit_compile={mode}: {exc.__class__.__name__}: {exc}") from exc


def clean_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): clean_json(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(inner) for inner in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--image", type=Path, action="append", default=None)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--decode-mode", choices=("eager", "compiled", "compiled_ifa"), default="compiled")
    parser.add_argument(
        "--compile-backend",
        choices=("torchair", "inductor"),
        default="torchair",
        help="Use torchair on NPU, or inductor for CUDA-only local checks.",
    )
    parser.add_argument("--compile-dynamic", action="store_true")
    parser.add_argument(
        "--compile-cache-dir",
        type=Path,
        default=PROJECT_ROOT / ".runtime_cache/12_unirec_0_1b_inference/torchair_decode",
        help="TorchAir cache_compile cache root for NPU compiled decode.",
    )
    parser.add_argument("--npu-jit-compile", choices=("off", "on", "default"), default="off")
    parser.add_argument("--warmup-first-image", action="store_true")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "tmp/12_unirec_0_1b_inference/optimized_summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images = [path.expanduser().resolve() for path in (args.image or DEFAULT_IMAGES)]
    configure_npu_jit_compile(args.npu_jit_compile, args.device)
    runner = OptimizedUniRecRunner(
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        compile_cache_dir=args.compile_cache_dir if args.compile_backend == "torchair" else None,
    )

    warmup = None
    if args.warmup_first_image and images:
        print(f"[warmup] optimized {args.decode_mode}: {images[0].name}", flush=True)
        warmup = runner.generate_one(
            images[0],
            max_length=args.max_length,
            decode_mode=args.decode_mode,
            compile_backend=args.compile_backend,
            compile_dynamic=args.compile_dynamic,
        )
        print(f"[warmup] total_latency_s={warmup['total_latency_s']:.4f}", flush=True)

    results = []
    for index, image_path in enumerate(images, start=1):
        print(f"\n[{index}/{len(images)}] optimized {args.decode_mode}: {image_path.name}", flush=True)
        result = runner.generate_one(
            image_path,
            max_length=args.max_length,
            decode_mode=args.decode_mode,
            compile_backend=args.compile_backend,
            compile_dynamic=args.compile_dynamic,
        )
        print(f"ttft_s={result['ttft_s']:.4f}", flush=True)
        print(f"decode_s={result['decode_s']:.4f}", flush=True)
        print(f"total_latency_s={result['total_latency_s']:.4f}", flush=True)
        print(f"decode_tokens_per_s={result['decode_tokens_per_s']}", flush=True)
        print(f"overall_tokens_per_s={result['overall_tokens_per_s']}", flush=True)
        print("generation:")
        print(result["text"])
        results.append(result)

    payload = {
        "experiment": "12_unirec_optimized_per_image",
        "status": "ok",
        "model_path": str(args.model_path.expanduser().resolve()),
        "device": args.device,
        "dtype": args.dtype,
        "max_length": int(args.max_length),
        "decode_mode": args.decode_mode,
        "compile_backend": args.compile_backend if args.decode_mode.startswith("compiled") else None,
        "compile_dynamic": bool(args.compile_dynamic),
        "compile_cache_dir": str(args.compile_cache_dir.expanduser().resolve()) if args.compile_backend == "torchair" else None,
        "npu_jit_compile": args.npu_jit_compile,
        "model_load_s": runner.model_load_s,
        "warmup": warmup,
        "images": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(clean_json(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {args.output_json}")
    print(json.dumps(clean_json(payload), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
