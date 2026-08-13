#!/usr/bin/env python3
"""A/B one full eager UniRec vision B1 forward with grouped-FZ weights."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch_npu

from modeling_optimized_unirec import OptimizedUniRecRunner, synchronize_device
from profile_stock_eager_vision_b1 import (
    _difference,
    _measure_controls,
    _parse_profile,
    _profile_once,
)
from vision_focal_depthwise import rewrite_eager_stage2_7x7_grouped_fz


TARGET_LOGICAL_TO_FZ1 = 'TransData | "384,1,7,7" -> "49,24,16,16"'
TARGET_GROUP_REPACK = 'TransData | "49,24,16,16" -> "1176,1,16,16"'
TARGET_CONV = 'Conv2D | "1,24,4,60,16;1176,1,16,16"'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--control-repeats", type=int, default=20)
    parser.add_argument("--parser-topn", type=int, default=200)
    parser.add_argument(
        "--profile-metric",
        choices=("pipe", "memory", "l2", "memory_access"),
        default="pipe",
    )
    args = parser.parse_args()
    if (args.width, args.height) != (960, 64):
        parser.error("the first model gate is fixed at 960x64")
    if args.warmups < 1 or args.control_repeats < 2:
        parser.error("use at least one warmup and two control repeats")
    if args.parser_topn < 100:
        parser.error("parser-topn must be at least 100")
    if not args.device.startswith("npu"):
        parser.error("this model gate requires an NPU")
    return args


def _physical_devices() -> list[int]:
    raw = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    devices = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if len(devices) != 1:
        raise RuntimeError(f"expected one visible physical NPU, got {devices}")
    if devices[0] in {5, 6}:
        raise RuntimeError(f"physical NPU {devices[0]} is excluded")
    return devices


def _warm(
    run: Callable[[], torch.Tensor],
    *,
    device: str,
    repeats: int,
) -> list[float]:
    wall_ms = []
    for _ in range(repeats):
        synchronize_device(device)
        started = time.perf_counter()
        run()
        synchronize_device(device)
        wall_ms.append((time.perf_counter() - started) * 1000.0)
    return wall_ms


def _aggregate(rows: list[dict[str, Any]], needle: str) -> dict[str, Any]:
    matches = [row for row in rows if needle in str(row.get("name", ""))]
    return {
        "count": sum(int(row.get("count", 0)) for row in matches),
        "duration_us": sum(float(row.get("duration_us", 0.0)) for row in matches),
        "rows": matches,
    }


def _target_operations(parsed: dict[str, Any]) -> dict[str, Any]:
    kernel = parsed["summary"]["runs"][0]["kernel_details"]
    transdata = kernel.get("top_transdata_shape_signatures", [])
    shapes = kernel.get("top_shape_signatures", [])
    return {
        "logical_weight_to_fz1": _aggregate(transdata, TARGET_LOGICAL_TO_FZ1),
        "fz1_to_grouped_fz384": _aggregate(transdata, TARGET_GROUP_REPACK),
        "physical_conv2d": _aggregate(shapes, TARGET_CONV),
    }


def main() -> None:
    args = parse_args()
    devices = _physical_devices()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    torch_npu.npu.set_compile_mode(jit_compile=False)

    setup_started = time.perf_counter()
    runner = OptimizedUniRecRunner(
        model_path=args.model_path.expanduser().resolve(),
        device=args.device,
        dtype="float16",
    )
    generator = torch.Generator(device="cpu").manual_seed(20260813)
    pixels = torch.rand(
        (1, 3, args.height, args.width),
        generator=generator,
        dtype=torch.float32,
    )
    pixels.mul_(2.0).sub_(1.0)
    pixels = pixels.to(device=args.device, dtype=runner.dtype)
    setup_s = time.perf_counter() - setup_started

    def run() -> torch.Tensor:
        return runner.model.forward_encoder(pixels)

    with torch.inference_mode():
        native_warmup_ms = _warm(
            run, device=args.device, repeats=args.warmups
        )
        native_before, native_output = _measure_controls(
            run, device=args.device, repeats=args.control_repeats
        )
        native_profile_timing, native_profile_output = _profile_once(
            run,
            device=args.device,
            profile_dir=output_dir / f"native_profile_{args.profile_metric}",
            metric=args.profile_metric,
        )
        native_after, native_after_output = _measure_controls(
            run, device=args.device, repeats=args.control_repeats
        )

        rewrite_started = time.perf_counter()
        rewrite = rewrite_eager_stage2_7x7_grouped_fz(
            runner.model.encoder.vision_encoder
        )
        synchronize_device(args.device)
        rewrite_wall_s = time.perf_counter() - rewrite_started

        grouped_warmup_ms = _warm(
            run, device=args.device, repeats=args.warmups
        )
        grouped_before, grouped_output = _measure_controls(
            run, device=args.device, repeats=args.control_repeats
        )
        grouped_profile_timing, grouped_profile_output = _profile_once(
            run,
            device=args.device,
            profile_dir=output_dir / f"grouped_profile_{args.profile_metric}",
            metric=args.profile_metric,
        )
        grouped_after, grouped_after_output = _measure_controls(
            run, device=args.device, repeats=args.control_repeats
        )

    native_parsed = _parse_profile(
        output_dir / f"native_profile_{args.profile_metric}",
        topn=args.parser_topn,
    )
    grouped_parsed = _parse_profile(
        output_dir / f"grouped_profile_{args.profile_metric}",
        topn=args.parser_topn,
    )
    native_targets = _target_operations(native_parsed)
    grouped_targets = _target_operations(grouped_parsed)
    parity = {
        "native_profile_vs_native_control": _difference(
            native_profile_output, native_output
        ),
        "native_after_vs_native_control": _difference(
            native_after_output, native_output
        ),
        "grouped_vs_native": {
            "exact": bool(torch.equal(grouped_output, native_output)),
            **_difference(grouped_output, native_output),
        },
        "grouped_profile_vs_native": _difference(
            grouped_profile_output, native_output
        ),
        "grouped_after_vs_native": _difference(
            grouped_after_output, native_output
        ),
    }

    status = "ok"
    if rewrite["rewritten_count"] != 9:
        status = "rewrite_count_failed"
    if not parity["grouped_vs_native"]["exact"]:
        status = "model_parity_failed"
    if (
        native_targets["logical_weight_to_fz1"]["count"] != 9
        or native_targets["fz1_to_grouped_fz384"]["count"] != 9
        or native_targets["physical_conv2d"]["count"] != 9
    ):
        status = "native_target_count_failed"
    if (
        grouped_targets["logical_weight_to_fz1"]["count"] != 0
        or grouped_targets["fz1_to_grouped_fz384"]["count"] != 0
        or grouped_targets["physical_conv2d"]["count"] != 9
    ):
        status = "grouped_target_count_failed"

    native_ms = float(native_after["device_event"]["median_ms"])
    grouped_ms = float(grouped_after["device_event"]["median_ms"])
    native_step = native_parsed["summary"]["runs"][0]["step_trace_time"][
        "totals_us"
    ]
    grouped_step = grouped_parsed["summary"]["runs"][0]["step_trace_time"][
        "totals_us"
    ]
    report = {
        "status": status,
        "device_name": torch.npu.get_device_name(0),
        "physical_devices": devices,
        "dtype": "float16",
        "npu_jit_compile": False,
        "input_shape": [1, 3, args.height, args.width],
        "input": "deterministic_uniform_minus1_plus1",
        "model_setup_s": setup_s,
        "rewrite_wall_s": rewrite_wall_s,
        "rewrite": rewrite,
        "native": {
            "warmup_wall_ms": native_warmup_ms,
            "control_before": native_before,
            "profile_timing": native_profile_timing,
            "control_after": native_after,
            "target_operations": native_targets,
            "parsed_profile": native_parsed,
        },
        "grouped_fz": {
            "warmup_wall_ms": grouped_warmup_ms,
            "control_before": grouped_before,
            "profile_timing": grouped_profile_timing,
            "control_after": grouped_after,
            "target_operations": grouped_targets,
            "parsed_profile": grouped_parsed,
        },
        "parity": parity,
        "clean_latency_comparison": {
            "native_post_profile_median_ms": native_ms,
            "grouped_post_profile_median_ms": grouped_ms,
            "speedup": native_ms / grouped_ms,
            "saved_ms": native_ms - grouped_ms,
        },
        "profile_step_comparison_ms": {
            "native_computing": float(native_step["Computing"]) / 1000.0,
            "grouped_computing": float(grouped_step["Computing"]) / 1000.0,
            "native_free": float(native_step["Free"]) / 1000.0,
            "grouped_free": float(grouped_step["Free"]) / 1000.0,
            "native_stage": float(native_step["Stage"]) / 1000.0,
            "grouped_stage": float(grouped_step["Stage"]) / 1000.0,
        },
    }
    output_json = output_dir / "result.json"
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    torch.save(native_output.detach().cpu(), output_dir / "native_output.pt")
    torch.save(grouped_output.detach().cpu(), output_dir / "grouped_output.pt")

    print(
        "UNIREC_EAGER_GROUPED_FZ_VISION_B1 "
        f"status={status} device={report['device_name']} "
        f"native_ms={native_ms:.6f} grouped_ms={grouped_ms:.6f} "
        f"speedup={native_ms / grouped_ms:.3f} "
        f"saved_ms={native_ms - grouped_ms:.6f} "
        f"exact={parity['grouped_vs_native']['exact']} "
        f"max_abs={parity['grouped_vs_native']['max_abs']:.9g} "
        f"mean_abs={parity['grouped_vs_native']['mean_abs']:.9g} "
        f"rewritten={rewrite['rewritten_count']}",
        flush=True,
    )
    print(
        "UNIREC_EAGER_GROUPED_FZ_VISION_KERNELS "
        f"native_logical_fz1={native_targets['logical_weight_to_fz1']['count']}/"
        f"{native_targets['logical_weight_to_fz1']['duration_us'] / 1000.0:.6f}ms "
        f"native_group_repack={native_targets['fz1_to_grouped_fz384']['count']}/"
        f"{native_targets['fz1_to_grouped_fz384']['duration_us'] / 1000.0:.6f}ms "
        f"native_conv={native_targets['physical_conv2d']['count']}/"
        f"{native_targets['physical_conv2d']['duration_us'] / 1000.0:.6f}ms "
        f"grouped_logical_fz1={grouped_targets['logical_weight_to_fz1']['count']}/"
        f"{grouped_targets['logical_weight_to_fz1']['duration_us'] / 1000.0:.6f}ms "
        f"grouped_group_repack={grouped_targets['fz1_to_grouped_fz384']['count']}/"
        f"{grouped_targets['fz1_to_grouped_fz384']['duration_us'] / 1000.0:.6f}ms "
        f"grouped_conv={grouped_targets['physical_conv2d']['count']}/"
        f"{grouped_targets['physical_conv2d']['duration_us'] / 1000.0:.6f}ms",
        flush=True,
    )
    print(
        "UNIREC_EAGER_GROUPED_FZ_VISION_PROFILE_STEP "
        f"native_compute_ms={float(native_step['Computing']) / 1000.0:.6f} "
        f"grouped_compute_ms={float(grouped_step['Computing']) / 1000.0:.6f} "
        f"native_free_ms={float(native_step['Free']) / 1000.0:.6f} "
        f"grouped_free_ms={float(grouped_step['Free']) / 1000.0:.6f} "
        f"native_stage_ms={float(native_step['Stage']) / 1000.0:.6f} "
        f"grouped_stage_ms={float(grouped_step['Stage']) / 1000.0:.6f}",
        flush=True,
    )
    print(f"OUTPUT_JSON={output_json}", flush=True)
    if status != "ok":
        raise RuntimeError(status)


if __name__ == "__main__":
    main()
