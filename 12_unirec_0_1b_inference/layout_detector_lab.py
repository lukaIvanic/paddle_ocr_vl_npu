#!/usr/bin/env python3
"""Profile the production PP-DocLayoutV2 NPU detector one page at a time.

The default ``current_production`` contract uses the exact production page
decoder, BGR materialization, adapter, optimized model configuration, threshold,
and OpenDoc image ordering. Recognition, crop construction, and output assembly
are intentionally excluded. Select ``--contract custom`` explicitly for a
historical or experimental model configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from layout_page_input import decode_page_rgb, materialize_layout_rgb
from opendoc_layout_npu import (
    LAYOUT_DEPTHWISE_REWRITE_CHOICES,
    LAYOUT_WEIGHT_FORMAT_CHOICES,
    PPDocLayoutV2NpuAdapter,
)


CURRENT_PRODUCTION_CONTRACT = {
    "execution": "torchair",
    "dtype": "float16",
    "reading_order_dtype": "float16",
    "threshold": 0.4,
    "weight_format": "torchair_internal",
    "freeze_parameters": False,
    "depthwise_rewrite": "group16",
    "fuse_frozen_bn": False,
    "fuse_eval_bn": False,
    "precompute_frozen_bn_affine": False,
    "preformat_frozen_bn_buffers": True,
    "input_color_order": "rgb",
}

CUSTOM_DEFAULTS = {
    "execution": "eager",
    "dtype": "float32",
    "reading_order_dtype": None,
    "threshold": 0.4,
    "weight_format": "native",
    "freeze_parameters": False,
    "depthwise_rewrite": "native",
    "fuse_frozen_bn": False,
    "fuse_eval_bn": False,
    "precompute_frozen_bn_affine": False,
    "preformat_frozen_bn_buffers": False,
    "input_color_order": "bgr",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        choices=("custom", "current_production"),
        default="current_production",
        help=(
            "Select current_production to enforce the exact optimized W1/B1 "
            "layout configuration used by the active prefill producer"
        ),
    )
    parser.add_argument("--openocr-root", type=Path, required=True)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/workspace/models/PP-DocLayoutV2_safetensors"),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/workspace/datasets/OmniDocBench/images"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--execution",
        choices=("eager", "torchair"),
        default=None,
        help="Run the model eagerly or through the static fullgraph TorchAir path",
    )
    parser.add_argument(
        "--compile-cache-dir",
        type=Path,
        default=Path(
            ".runtime_cache/12_unirec_0_1b_inference/layout_detector_lab"
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "float32"),
        default=None,
    )
    parser.add_argument(
        "--reading-order-dtype",
        choices=("float16", "float32"),
        default=None,
        help="Override only the learned reading-order head dtype",
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--weight-format",
        choices=LAYOUT_WEIGHT_FORMAT_CHOICES,
        default=None,
    )
    parser.add_argument(
        "--freeze-parameters",
        action="store_true",
        default=None,
        help="Let TorchAir treat model parameters as immutable graph data",
    )
    parser.add_argument(
        "--depthwise-rewrite",
        choices=LAYOUT_DEPTHWISE_REWRITE_CHOICES,
        default=None,
        help="Exact block-diagonal replacement for depthwise 5x5 convolutions",
    )
    parser.add_argument(
        "--fuse-frozen-bn",
        action="store_true",
        default=None,
        help="Fold inference-only backbone FrozenBatchNorm2d into Conv2d",
    )
    parser.add_argument(
        "--fuse-eval-bn",
        action="store_true",
        default=None,
        help="Fold evaluation FPN/PAN BatchNorm2d into Conv2d",
    )
    parser.add_argument(
        "--precompute-frozen-bn-affine",
        action="store_true",
        default=None,
        help="Precompute exact FrozenBN scale/bias on NPU and store NC1HWC0",
    )
    parser.add_argument(
        "--preformat-frozen-bn-buffers",
        action="store_true",
        default=None,
        help="Keep FrozenBN math but store its four buffers as NC1HWC0",
    )
    parser.add_argument(
        "--input-color-order",
        choices=("bgr", "rgb"),
        default=None,
        help="Channel order of each HWC page passed to the layout adapter",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument(
        "--warmup-pages",
        type=int,
        default=1,
        help="Warmup calls on the first selected page; excluded from results",
    )
    parser.add_argument(
        "--torch-cpu-threads",
        type=int,
        default=0,
        help=(
            "Diagnostic override for PyTorch intra-op CPU threads; zero keeps "
            "the production process default unchanged"
        ),
    )
    args = parser.parse_args()
    _resolve_contract(parser, args)
    if args.torch_cpu_threads < 0:
        parser.error("--torch-cpu-threads must be non-negative")
    return args


def _resolve_contract(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    expected = (
        CURRENT_PRODUCTION_CONTRACT
        if args.contract == "current_production"
        else CUSTOM_DEFAULTS
    )
    for name, value in expected.items():
        supplied = getattr(args, name)
        if (
            args.contract == "current_production"
            and supplied is not None
            and supplied != value
        ):
            parser.error(
                f"--contract current_production requires {name}={value!r}, "
                f"got {supplied!r}"
            )
        if supplied is None:
            setattr(args, name, value)


def decode_page_image(path: Path, *, color_order: str) -> tuple[np.ndarray, dict[str, float]]:
    rgb, timing = decode_page_rgb(path)
    started = time.perf_counter()
    if color_order == "rgb":
        image = materialize_layout_rgb(rgb)
        rgb_materialize_s = time.perf_counter() - started
        rgb_to_bgr_s = 0.0
    elif color_order == "bgr":
        from layout_page_input import materialize_layout_bgr

        image = materialize_layout_bgr(rgb)
        rgb_to_bgr_s = time.perf_counter() - started
        rgb_materialize_s = 0.0
    else:
        raise ValueError(f"unsupported layout input color order: {color_order}")
    return image, {
        "page_file_read_s": timing["file_read_s"],
        "page_image_decode_s": timing["direct_rgb_decode_s"],
        "page_rgb_materialize_s": rgb_materialize_s,
        "page_rgb_to_bgr_s": rgb_to_bgr_s,
    }


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def result_digest(result: dict[str, Any]) -> str:
    payload = json.dumps(
        result,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def summarize(records: list[dict[str, Any]], setup_s: float) -> dict[str, Any]:
    stage_names = sorted(
        {
            name
            for record in records
            for name in record["stage_s"]
        }
    )
    stages: dict[str, Any] = {}
    for name in stage_names:
        values = [float(record["stage_s"].get(name, 0.0)) for record in records]
        stages[name] = {
            "total_s": sum(values),
            "mean_ms": statistics.fmean(values) * 1000.0,
            "median_ms": statistics.median(values) * 1000.0,
            "p90_ms": percentile(values, 0.90) * 1000.0,
            "min_ms": min(values) * 1000.0,
            "max_ms": max(values) * 1000.0,
        }

    page_wall = [float(record["page_wall_s"]) for record in records]
    measured_wall_s = sum(page_wall)
    detector_total_s = stages.get("detector_total_s", {}).get("total_s", 0.0)
    for name, stage in stages.items():
        stage["page_wall_share_pct"] = (
            100.0 * float(stage["total_s"]) / measured_wall_s
            if measured_wall_s
            else 0.0
        )
        if name not in {
            "page_file_read_s",
            "page_image_decode_s",
            "page_rgb_materialize_s",
            "page_rgb_to_bgr_s",
        }:
            stage["detector_share_pct"] = (
                100.0 * float(stage["total_s"]) / detector_total_s
                if detector_total_s
                else 0.0
            )

    return {
        "setup_s": setup_s,
        "page_count": len(records),
        "measured_page_wall_s": measured_wall_s,
        "pages_per_s": len(records) / measured_wall_s if measured_wall_s else 0.0,
        "page_wall_mean_ms": statistics.fmean(page_wall) * 1000.0,
        "page_wall_median_ms": statistics.median(page_wall) * 1000.0,
        "page_wall_p90_ms": percentile(page_wall, 0.90) * 1000.0,
        "stages": stages,
    }


def main() -> None:
    args = parse_args()
    if args.limit < 1:
        raise ValueError("--limit must be >= 1")
    if args.warmup_pages < 0:
        raise ValueError("--warmup-pages must be >= 0")

    if args.device.startswith("npu"):
        import torch_npu

        torch_npu.npu.set_compile_mode(jit_compile=False)

    if args.torch_cpu_threads:
        torch.set_num_threads(args.torch_cpu_threads)
        torch.set_num_interop_threads(args.torch_cpu_threads)

    openocr_root = args.openocr_root.expanduser().resolve()
    sys.path.insert(0, str(openocr_root))
    from tools.utils.utility import get_image_file_list

    input_path = args.input.expanduser().resolve()
    image_paths = [
        Path(path).resolve()
        for path in sorted(get_image_file_list(str(input_path)))
    ][args.offset : args.offset + args.limit]
    if not image_paths:
        raise ValueError(f"No images found under {input_path}")

    print(
        f"LAYOUT_LAB setup contract={args.contract} "
        f"pages={len(image_paths)} dtype={args.dtype} "
        f"reading_order_dtype={args.reading_order_dtype or args.dtype} "
        f"device={args.device} execution={args.execution}",
        flush=True,
    )
    print("LAYOUT_LAB phase=model_setup_begin", flush=True)
    detector = PPDocLayoutV2NpuAdapter(
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        reading_order_dtype=args.reading_order_dtype,
        threshold=args.threshold,
        profile_stages=True,
        execution=args.execution,
        compile_cache_dir=args.compile_cache_dir,
        batch_size=1,
        weight_format=args.weight_format,
        freeze_parameters=args.freeze_parameters,
        depthwise_rewrite=args.depthwise_rewrite,
        fuse_frozen_bn=args.fuse_frozen_bn,
        fuse_eval_bn=args.fuse_eval_bn,
        precompute_frozen_bn_affine=args.precompute_frozen_bn_affine,
        preformat_frozen_bn_buffers=args.preformat_frozen_bn_buffers,
        input_color_order=args.input_color_order,
    )
    print(
        f"LAYOUT_LAB phase=model_setup_end setup_s={detector.setup_s:.3f}",
        flush=True,
    )

    warmup_rgb, _ = decode_page_rgb(image_paths[0])
    for index in range(args.warmup_pages):
        print(
            f"LAYOUT_LAB phase=warmup_call_begin "
            f"call={index + 1}/{args.warmup_pages}",
            flush=True,
        )
        # Match the production worker warmup. The current production contract
        # keeps the decoded page in RGB; custom BGR runs retain the old path.
        warmup_image = (
            materialize_layout_rgb(warmup_rgb)
            if args.input_color_order == "rgb"
            else warmup_rgb[..., ::-1]
        )
        detector([warmup_image], threshold=args.threshold)
        print(
            f"LAYOUT_LAB phase=warmup_call_end "
            f"call={index + 1}/{args.warmup_pages}",
            flush=True,
        )
    detector.reset_timing()

    records: list[dict[str, Any]] = []
    for page_index, image_path in enumerate(image_paths):
        page_started = time.perf_counter()
        image, decode_timing = decode_page_image(
            image_path,
            color_order=args.input_color_order,
        )
        before = dict(detector.stage_s)
        result = detector([image], threshold=args.threshold)[0]
        stage_s = {
            name: float(seconds) - float(before.get(name, 0.0))
            for name, seconds in detector.stage_s.items()
        }
        stage_s.update(decode_timing)
        record = {
            "page_index": page_index + args.offset,
            "image": str(image_path),
            "height": int(image.shape[0]),
            "width": int(image.shape[1]),
            "box_count": len(result["boxes"]),
            "result": result,
            "result_digest": result_digest(result),
            "page_wall_s": time.perf_counter() - page_started,
            "stage_s": stage_s,
        }
        records.append(record)
        print(
            f"LAYOUT_LAB page={page_index + 1}/{len(image_paths)} "
            f"wall_ms={record['page_wall_s'] * 1000.0:.1f} "
            f"forward_ms={stage_s.get('model_forward_s', 0.0) * 1000.0:.1f} "
            f"boxes={record['box_count']} "
            f"digest={record['result_digest'][:12]}",
            flush=True,
        )

    resolved_contract = {
        name: getattr(args, name) for name in CURRENT_PRODUCTION_CONTRACT
    }
    report = {
        "config": {
            "contract": args.contract,
            "production_contract_verified": (
                args.contract == "current_production"
                and resolved_contract == CURRENT_PRODUCTION_CONTRACT
            ),
            "resolved_model_contract": resolved_contract,
            "page_input": {
                "decoder": "shared_production_decode_page_rgb",
                "color_order": args.input_color_order,
                "materialization": "shared_production_contiguous_rgb",
            },
            "openocr_root": str(openocr_root),
            "model_path": str(args.model_path.expanduser().resolve()),
            "input": str(input_path),
            "device": args.device,
            "dtype": args.dtype,
            "reading_order_dtype": args.reading_order_dtype,
            "execution": args.execution,
            "compile_cache_dir": str(args.compile_cache_dir.expanduser().resolve()),
            "threshold": args.threshold,
            "offset": args.offset,
            "limit": args.limit,
            "warmup_pages": args.warmup_pages,
            "cpu_runtime": {
                "architecture": platform.machine(),
                "logical_cpu_count": os.cpu_count(),
                "torch_intraop_threads": torch.get_num_threads(),
                "torch_interop_threads": torch.get_num_interop_threads(),
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
                "OPENBLAS_NUM_THREADS": os.environ.get(
                    "OPENBLAS_NUM_THREADS"
                ),
            },
            "weight_format": args.weight_format,
            "weight_format_summary": detector.weight_format_summary,
            "freeze_parameters": args.freeze_parameters,
            "depthwise_rewrite": args.depthwise_rewrite,
            "depthwise_rewrite_summary": detector.depthwise_rewrite_summary,
            "fuse_frozen_bn": args.fuse_frozen_bn,
            "frozen_bn_fusion_summary": detector.frozen_bn_fusion_summary,
            "fuse_eval_bn": args.fuse_eval_bn,
            "eval_bn_fusion_summary": detector.eval_bn_fusion_summary,
            "precompute_frozen_bn_affine": args.precompute_frozen_bn_affine,
            "frozen_bn_affine_summary": detector.frozen_bn_affine_summary,
            "preformat_frozen_bn_buffers": args.preformat_frozen_bn_buffers,
            "frozen_bn_buffer_format_summary": (
                detector.frozen_bn_buffer_format_summary
            ),
            "scheduling": "sequential_b1_same_process",
        },
        "summary": summarize(records, detector.setup_s),
        "pages": records,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")

    print("LAYOUT_LAB summary", flush=True)
    for name, stage in sorted(
        report["summary"]["stages"].items(),
        key=lambda item: item[1]["total_s"],
        reverse=True,
    ):
        if name == "detector_total_s":
            continue
        print(
            f"  {name}: total={stage['total_s']:.3f}s "
            f"mean={stage['mean_ms']:.2f}ms p90={stage['p90_ms']:.2f}ms",
            flush=True,
        )
    print(
        f"LAYOUT_LAB done pages_per_s={report['summary']['pages_per_s']:.3f} "
        f"output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
