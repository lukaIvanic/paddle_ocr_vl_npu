#!/usr/bin/env python3
"""Compile/profile one UniRec vision B1 lane with exact grouped-FZ gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch_npu

from modeling_optimized_unirec import (
    OptimizedUniRecRunner,
    import_torchair_cache_compile,
    synchronize_device,
)
from profile_eager_grouped_fz_vision_b1 import _target_operations
from profile_stock_eager_vision_b1 import (
    _difference,
    _measure_controls,
    _parse_profile,
    _profile_once,
)
from vision_compile_batch_matrix import _new_stock_encoder_module
from vision_focal_depthwise import rewrite_eager_stage23_5x5_7x7_grouped_fz
from vision_full_batch import VisionBucketSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--lane",
        choices=("native", "grouped_fz"),
        required=True,
    )
    parser.add_argument(
        "--reference-output",
        type=Path,
        help="Optional compiled CPU output from the native lane.",
    )
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--control-repeats", type=int, default=20)
    parser.add_argument("--parser-topn", type=int, default=200)
    parser.add_argument(
        "--profile-metric",
        choices=("pipe", "memory", "l2", "memory_access"),
        default="pipe",
    )
    parser.add_argument(
        "--frozen-parameter",
        action="store_true",
        help="Embed weights into the compiled graph instead of runtime binding.",
    )
    args = parser.parse_args()
    if args.warmups < 1 or args.control_repeats < 2:
        parser.error("use at least one warmup and two control repeats")
    if args.parser_topn < 100:
        parser.error("parser-topn must be at least 100")
    if not args.device.startswith("npu"):
        parser.error("this compile gate requires an NPU")
    if args.lane == "native" and args.reference_output is not None:
        parser.error("reference-output is only valid for the grouped_fz lane")
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


def _comparison(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    return {
        "exact": bool(torch.equal(left, right)),
        **_difference(left, right),
    }


def _source_hash(lane: str, frozen_parameter: bool) -> str:
    here = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    digest.update(Path(__file__).read_bytes())
    digest.update((here / "vision_focal_depthwise.py").read_bytes())
    digest.update((here / "grouped_fz_descriptor_bridge.cpp").read_bytes())
    digest.update((here / "vision_compile_batch_matrix.py").read_bytes())
    digest.update(lane.encode("utf-8"))
    digest.update(str(bool(frozen_parameter)).encode("utf-8"))
    return digest.hexdigest()[:12]


def main() -> None:
    args = parse_args()
    devices = _physical_devices()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    cache_root = args.cache_root.expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    torch_npu.npu.set_compile_mode(jit_compile=False)

    setup_started = time.perf_counter()
    runner = OptimizedUniRecRunner(
        model_path=args.model_path.expanduser().resolve(),
        device=args.device,
        dtype="float16",
        compile_cache_dir=cache_root,
    )
    generator = torch.Generator(device="cpu").manual_seed(20260813)
    pixels = torch.rand(
        (1, 3, 64, 960),
        generator=generator,
        dtype=torch.float32,
    )
    pixels.mul_(2.0).sub_(1.0)
    pixels = pixels.to(device=args.device, dtype=runner.dtype)
    setup_s = time.perf_counter() - setup_started

    rewrite: dict[str, Any] = {
        "requested": "native",
        "rewritten_count": 0,
    }
    if args.lane == "grouped_fz":
        rewrite = rewrite_eager_stage23_5x5_7x7_grouped_fz(
            runner.model.encoder.vision_encoder
        )
        synchronize_device(args.device)

    spec = VisionBucketSpec(width=960, height=64, batch_size=1)
    module = _new_stock_encoder_module(runner, spec)
    source_hash = _source_hash(args.lane, args.frozen_parameter)
    cache_dir = cache_root / (
        f"vision_stock_fixed_{spec.key}_float16_{args.lane}_"
        f"frozen{int(args.frozen_parameter)}_src{source_hash}"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    from torch_npu.dynamo.torchair.configs.compiler_config import CompilerConfig

    config = CompilerConfig()
    config.mode.value = "max-autotune"
    if args.frozen_parameter:
        config.experimental_config.frozen_parameter.value = True
    cache_compile, compile_api = import_torchair_cache_compile()
    registration_started = time.perf_counter()
    compiled = cache_compile(
        module.forward,
        config=config,
        dynamic=False,
        cache_dir=str(cache_dir),
        ge_cache=True,
        fullgraph=True,
    )
    registration_s = time.perf_counter() - registration_started

    def run_eager() -> torch.Tensor:
        return module(pixels)

    def run_compiled() -> torch.Tensor:
        return compiled(pixels)

    with torch.inference_mode():
        eager_output = run_eager()
        synchronize_device(args.device)
        first_call_started = time.perf_counter()
        first_output = run_compiled()
        synchronize_device(args.device)
        first_call_s = time.perf_counter() - first_call_started
        warmup_wall_ms = _warm(
            run_compiled,
            device=args.device,
            repeats=args.warmups,
        )
        control_before, control_output = _measure_controls(
            run_compiled,
            device=args.device,
            repeats=args.control_repeats,
        )
        profile_timing, profile_output = _profile_once(
            run_compiled,
            device=args.device,
            profile_dir=output_dir / f"compiled_profile_{args.profile_metric}",
            metric=args.profile_metric,
        )
        control_after, after_output = _measure_controls(
            run_compiled,
            device=args.device,
            repeats=args.control_repeats,
        )

    parsed = _parse_profile(
        output_dir / f"compiled_profile_{args.profile_metric}",
        topn=args.parser_topn,
    )
    targets = _target_operations(parsed)
    parity: dict[str, Any] = {
        "compiled_first_vs_eager_same_lane": _comparison(
            first_output, eager_output
        ),
        "compiled_control_vs_eager_same_lane": _comparison(
            control_output, eager_output
        ),
        "compiled_profile_vs_control": _comparison(
            profile_output, control_output
        ),
        "compiled_after_vs_control": _comparison(after_output, control_output),
    }
    if args.reference_output is not None:
        native_compiled = torch.load(
            args.reference_output.expanduser().resolve(),
            map_location="cpu",
            weights_only=True,
        )
        parity["grouped_compiled_vs_native_compiled"] = _comparison(
            control_output.detach().cpu(), native_compiled
        )

    status = "ok"
    if args.lane == "grouped_fz":
        if rewrite["rewritten_count"] != 22:
            status = "rewrite_count_failed"
        if (
            targets["logical_weight_to_fz1"]["count"] != 0
            or targets["fz1_to_grouped_fz"]["count"] != 0
            or targets["physical_conv2d"]["count"] != 22
        ):
            status = "grouped_target_count_failed"
        reference = parity.get("grouped_compiled_vs_native_compiled")
        if reference is not None and not reference["exact"]:
            status = "compiled_reference_parity_failed"

    report = {
        "status": status,
        "lane": args.lane,
        "device_name": torch.npu.get_device_name(0),
        "physical_devices": devices,
        "dtype": "float16",
        "npu_jit_compile": False,
        "input_shape": [1, 3, 64, 960],
        "input": "deterministic_uniform_minus1_plus1",
        "compile_api": compile_api,
        "frozen_parameter": bool(args.frozen_parameter),
        "cache_dir": str(cache_dir),
        "source_hash": source_hash,
        "setup_s": setup_s,
        "registration_s": registration_s,
        "compiled_first_call_s": first_call_s,
        "warmup_wall_ms": warmup_wall_ms,
        "control_before": control_before,
        "profile_timing": profile_timing,
        "control_after": control_after,
        "rewrite": rewrite,
        "target_operations": targets,
        "parity": parity,
        "parsed_profile": parsed,
        "measurement_scope": (
            "one fixed-shape full UniRec vision encoder forward; no H2D, "
            "preprocessing, layout, text prefill, or decode"
        ),
    }
    output_json = output_dir / "result.json"
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    torch.save(eager_output.detach().cpu(), output_dir / "eager_output.pt")
    torch.save(control_output.detach().cpu(), output_dir / "compiled_output.pt")

    reference = parity.get("grouped_compiled_vs_native_compiled")
    print(
        "UNIREC_COMPILED_GROUPED_FZ_VISION_B1 "
        f"status={status} lane={args.lane} device={report['device_name']} "
        f"first_call_s={first_call_s:.6f} "
        f"median_ms={control_after['device_event']['median_ms']:.6f} "
        f"same_lane_exact={parity['compiled_control_vs_eager_same_lane']['exact']} "
        f"reference_exact={None if reference is None else reference['exact']} "
        f"reference_max_abs={None if reference is None else reference['max_abs']}",
        flush=True,
    )
    print(
        "UNIREC_COMPILED_GROUPED_FZ_VISION_KERNELS "
        f"logical_fz1={targets['logical_weight_to_fz1']['count']}/"
        f"{targets['logical_weight_to_fz1']['duration_us'] / 1000.0:.6f}ms "
        f"group_repack={targets['fz1_to_grouped_fz']['count']}/"
        f"{targets['fz1_to_grouped_fz']['duration_us'] / 1000.0:.6f}ms "
        f"conv={targets['physical_conv2d']['count']}/"
        f"{targets['physical_conv2d']['duration_us'] / 1000.0:.6f}ms",
        flush=True,
    )
    print(f"OUTPUT_JSON={output_json}", flush=True)
    if status != "ok":
        raise RuntimeError(status)


if __name__ == "__main__":
    main()
