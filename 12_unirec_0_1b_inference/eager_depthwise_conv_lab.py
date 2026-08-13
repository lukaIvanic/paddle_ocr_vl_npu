#!/usr/bin/env python3
"""Isolate the UniRec 384-channel 7x7 eager depthwise Conv2D path."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
TARGET_LOGICAL_TO_FZ1 = (
    'TransData | "384,1,7,7" -> "49,24,16,16"'
)
TARGET_GROUP_REPACK = (
    'TransData | "49,24,16,16" -> "1176,1,16,16"'
)
TARGET_CONV = (
    'Conv2D | "1,24,4,60,16;1176,1,16,16"'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--lane",
        choices=("matrix", "native", "fractal_z_1", "grouped_fz_384"),
        default="matrix",
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--control-repeats", type=int, default=50)
    parser.add_argument("--parser-topn", type=int, default=200)
    parser.add_argument(
        "--profile-metric",
        choices=("pipe", "memory", "l2", "memory_access"),
        default="pipe",
    )
    args = parser.parse_args()
    if args.warmups < 1 or args.control_repeats < 2:
        parser.error("use at least one warmup and two control repeats")
    if args.parser_topn < 20:
        parser.error("parser-topn must be at least 20")
    if not args.device.startswith("npu"):
        parser.error("this lab requires an NPU device")
    return args


def _physical_devices() -> list[int]:
    raw = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    devices = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if len(devices) != 1:
        raise RuntimeError(f"expected one visible physical NPU, got {devices}")
    if devices[0] in {5, 6}:
        raise RuntimeError(f"physical NPU {devices[0]} is excluded")
    return devices


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values: Sequence[float]) -> dict[str, Any]:
    samples = [float(value) for value in values]
    return {
        "samples_ms": samples,
        "min_ms": min(samples),
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "p90_ms": _percentile(samples, 0.90),
        "max_ms": max(samples),
    }


def _synchronize(device: str) -> None:
    import torch

    torch.npu.synchronize(torch.device(device))


def _measure_controls(
    run: Callable[[], Any],
    *,
    device: str,
    repeats: int,
) -> tuple[dict[str, Any], Any]:
    import torch_npu

    event_ms: list[float] = []
    wall_ms: list[float] = []
    output = None
    for _ in range(repeats):
        _synchronize(device)
        start = torch_npu.npu.Event(enable_timing=True)
        end = torch_npu.npu.Event(enable_timing=True)
        wall_started = time.perf_counter()
        start.record()
        output = run()
        end.record()
        end.synchronize()
        event_ms.append(float(start.elapsed_time(end)))
        wall_ms.append((time.perf_counter() - wall_started) * 1000.0)
    if output is None:
        raise RuntimeError("control forward produced no output")
    return {
        "device_event": _summary(event_ms),
        "synchronized_wall": _summary(wall_ms),
    }, output


def _profiler_config(metric: str) -> Any:
    import torch_npu.profiler as npu_prof

    metrics = {
        "pipe": npu_prof.AiCMetrics.PipeUtilization,
        "memory": npu_prof.AiCMetrics.Memory,
        "l2": npu_prof.AiCMetrics.L2Cache,
        "memory_access": npu_prof.AiCMetrics.MemoryAccess,
    }
    return npu_prof._ExperimentalConfig(
        profiler_level=npu_prof.ProfilerLevel.Level1,
        aic_metrics=metrics[metric],
        l2_cache=metric == "l2",
        export_type=npu_prof.ExportType.Text,
        data_simplification=False,
    )


def _profile_once(
    run: Callable[[], Any],
    *,
    device: str,
    profile_dir: Path,
    metric: str,
) -> tuple[dict[str, float], Any]:
    import torch
    import torch_npu
    import torch_npu.profiler as npu_prof

    profile_dir.mkdir(parents=True, exist_ok=False)
    schedule = npu_prof.schedule(wait=0, warmup=0, active=1, repeat=1)
    _synchronize(device)
    event_start = torch_npu.npu.Event(enable_timing=True)
    event_end = torch_npu.npu.Event(enable_timing=True)
    context_started = time.perf_counter()
    output = None
    with npu_prof.profile(
        activities=[
            npu_prof.ProfilerActivity.CPU,
            npu_prof.ProfilerActivity.NPU,
        ],
        schedule=schedule,
        experimental_config=_profiler_config(metric),
        on_trace_ready=npu_prof.tensorboard_trace_handler(
            str(profile_dir), analyse_flag=True
        ),
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
    ) as profiler:
        _synchronize(device)
        forward_started = time.perf_counter()
        event_start.record()
        with torch.profiler.record_function("unirec.eager_depthwise_conv_lab"):
            output = run()
        event_end.record()
        event_end.synchronize()
        forward_wall_ms = (time.perf_counter() - forward_started) * 1000.0
        profiler.step()
    if output is None:
        raise RuntimeError("profiled forward produced no output")
    return {
        "device_event_ms": float(event_start.elapsed_time(event_end)),
        "synchronized_forward_wall_ms": forward_wall_ms,
        "profile_context_wall_s": time.perf_counter() - context_started,
    }, output


def _parse_profile(profile_dir: Path, *, topn: int) -> dict[str, Any]:
    parser_path = REPO_ROOT / "11_mineru_2_5_pro_inference/parse_npu_profile.py"
    parsed_json = profile_dir / "profile_parse_summary.json"
    parsed_md = profile_dir / "profile_parse_summary.md"
    command = [
        sys.executable,
        str(parser_path),
        "--profile-dir",
        str(profile_dir),
        "--topn",
        str(topn),
        "--out-json",
        str(parsed_json),
        "--out-md",
        str(parsed_md),
        "--skip-trace",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return {
        "command": command,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "summary": json.loads(parsed_json.read_text(encoding="utf-8")),
        "markdown": str(parsed_md),
    }


def _matching_rows(rows: list[dict[str, Any]], needle: str) -> list[dict[str, Any]]:
    return [row for row in rows if needle in str(row.get("name", ""))]


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": sum(int(row.get("count", 0)) for row in rows),
        "duration_us": sum(float(row.get("duration_us", 0.0)) for row in rows),
        "rows": rows,
    }


def _load_grouped_fz_bridge() -> tuple[Any, float]:
    import torch_npu
    from torch.utils.cpp_extension import load

    torch_npu_root = Path(torch_npu.__file__).resolve().parent
    if shutil.which("ninja") is None:
        interpreter_ninja = Path(sys.executable).resolve().parent / "ninja"
        if interpreter_ninja.is_file() and os.access(interpreter_ninja, os.X_OK):
            os.environ["PATH"] = (
                f"{interpreter_ninja.parent}{os.pathsep}"
                f"{os.environ.get('PATH', '')}"
            )
    if shutil.which("ninja") is None:
        raise RuntimeError(
            "ninja is required to build the grouped-FZ descriptor bridge; "
            "put an existing ninja executable on PATH"
        )
    library_dir = torch_npu_root / "lib"
    source = HERE / "grouped_fz_descriptor_bridge.cpp"
    started = time.perf_counter()
    bridge = load(
        name="unirec_grouped_fz_descriptor_bridge_v3",
        sources=[str(source)],
        extra_include_paths=[str(torch_npu_root / "include")],
        extra_cflags=["-O2"],
        extra_ldflags=[
            f"-L{library_dir}",
            "-ltorch_npu",
            f"-Wl,-rpath,{library_dir}",
        ],
        verbose=True,
    )
    return bridge, time.perf_counter() - started


def _run_lane(args: argparse.Namespace) -> None:
    import torch
    import torch.nn.functional as functional
    import torch_npu

    from vision_focal_depthwise import pack_grouped_fz_host

    devices = _physical_devices()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    # torch-npu 2.10 requires this before the first NPU allocation if an
    # internal-format tensor is requested explicitly.
    torch.npu.config.allow_internal_format = args.lane in {
        "fractal_z_1",
        "grouped_fz_384",
    }
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.manual_seed(20260813)
    host_input = torch.randn((1, 384, 4, 60), dtype=torch.float16)
    host_weight = torch.randn((384, 1, 7, 7), dtype=torch.float16) * 0.01
    inputs = host_input.to(args.device)
    weight = host_weight.to(args.device)
    weight_format_before = int(torch_npu.get_npu_format(weight))
    extension_build_s = 0.0
    descriptor = None
    if args.lane == "fractal_z_1":
        weight = torch_npu.npu_format_cast(weight, 4)
    elif args.lane == "grouped_fz_384":
        bridge, extension_build_s = _load_grouped_fz_bridge()
        packed_host = pack_grouped_fz_host(
            host_weight.numpy(), groups=384
        )
        packed_storage = torch.from_numpy(packed_host).to(args.device)
        weight = bridge.wrap_grouped_fz(
            packed_storage,
            list(host_weight.shape),
            384,
        )
        origin_format, storage_format, base_shape, storage_shape = (
            bridge.describe_npu_storage(weight)
        )
        descriptor = {
            "origin_format": int(origin_format),
            "storage_format": int(storage_format),
            "base_shape": [int(value) for value in base_shape],
            "storage_shape": [int(value) for value in storage_shape],
            "physical_bytes": int(packed_host.nbytes),
        }
    weight_format_after = (
        int(descriptor["storage_format"])
        if descriptor is not None
        else int(torch_npu.get_npu_format(weight))
    )

    def run() -> torch.Tensor:
        return functional.conv2d(
            inputs,
            weight,
            bias=None,
            stride=1,
            padding=3,
            dilation=1,
            groups=384,
        )

    with torch.inference_mode():
        warmup_wall_ms: list[float] = []
        for _ in range(args.warmups):
            _synchronize(args.device)
            started = time.perf_counter()
            run()
            _synchronize(args.device)
            warmup_wall_ms.append((time.perf_counter() - started) * 1000.0)
        control_before, before_output = _measure_controls(
            run, device=args.device, repeats=args.control_repeats
        )
        profile_dir = output_dir / f"profile_{args.profile_metric}"
        profile_timing, profiled_output = _profile_once(
            run,
            device=args.device,
            profile_dir=profile_dir,
            metric=args.profile_metric,
        )
        control_after, after_output = _measure_controls(
            run, device=args.device, repeats=args.control_repeats
        )

    parsed = _parse_profile(profile_dir, topn=args.parser_topn)
    kernel = parsed["summary"]["runs"][0]["kernel_details"]
    shape_rows = kernel.get("top_shape_signatures", [])
    transdata_rows = kernel.get("top_transdata_shape_signatures", [])
    logical_to_fz1 = _aggregate(
        _matching_rows(transdata_rows, TARGET_LOGICAL_TO_FZ1)
    )
    group_repack = _aggregate(
        _matching_rows(transdata_rows, TARGET_GROUP_REPACK)
    )
    native_convolution = _aggregate(_matching_rows(shape_rows, TARGET_CONV))
    any_convolution = _aggregate(
        [
            row
            for row in shape_rows
            if str(row.get("name", "")).startswith("Conv2D |")
        ]
    )

    before_event = float(control_before["device_event"]["median_ms"])
    after_event = float(control_after["device_event"]["median_ms"])
    profile_event = float(profile_timing["device_event_ms"])
    combined_control = (before_event + after_event) / 2.0
    profile_delta = (profiled_output - before_output).abs()
    after_delta = (after_output - before_output).abs()
    report = {
        "status": "ok",
        "lane": args.lane,
        "device_name": torch.npu.get_device_name(0),
        "physical_devices": devices,
        "dtype": "float16",
        "npu_jit_compile": False,
        "input_shape": [1, 384, 4, 60],
        "logical_weight_shape": [384, 1, 7, 7],
        "conv_contract": {
            "stride": 1,
            "padding": 3,
            "dilation": 1,
            "groups": 384,
            "bias": False,
        },
        "weight_format": {
            "before": weight_format_before,
            "after": weight_format_after,
            "requested": args.lane,
        },
        "grouped_fz_descriptor": descriptor,
        "extension_build_s": extension_build_s,
        "warmup_wall_ms": warmup_wall_ms,
        "control_before": control_before,
        "profile_timing": profile_timing,
        "control_after": control_after,
        "profiler_overhead": {
            "profiled_event_vs_combined_control": profile_event
            / combined_control,
            "control_after_vs_before": after_event / before_event,
        },
        "profiled_vs_control": {
            "max_abs": float(profile_delta.max().item()),
            "mean_abs": float(profile_delta.mean().item()),
        },
        "control_after_vs_before": {
            "max_abs": float(after_delta.max().item()),
            "mean_abs": float(after_delta.mean().item()),
        },
        "target_operations": {
            "logical_weight_to_fz1": logical_to_fz1,
            "fz1_to_grouped_fz384": group_repack,
            "native_physical_conv2d": native_convolution,
            "conv2d_any_physical_signature": any_convolution,
        },
        "parsed_profile": parsed,
    }
    if args.lane == "native" and group_repack["count"] != 1:
        report["status"] = "target_repack_not_reproduced"
    if args.lane == "native" and native_convolution["count"] != 1:
        report["status"] = "target_conv_not_reproduced"
    if args.lane == "grouped_fz_384":
        expected_descriptor = {
            "origin_format": 0,
            "storage_format": 4,
            "base_shape": [384, 1, 7, 7],
            "storage_shape": [1176, 1, 16, 16],
            "physical_bytes": 1176 * 16 * 16 * 2,
        }
        if descriptor != expected_descriptor:
            report["status"] = "grouped_descriptor_mismatch"
        if logical_to_fz1["count"] != 0 or group_repack["count"] != 0:
            report["status"] = "prepacked_weight_was_repacked"
    if any_convolution["count"] != 1:
        report["status"] = "single_conv_not_observed"

    output_json = output_dir / "result.json"
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    torch.save(before_output.detach().cpu(), output_dir / "output.pt")
    print(
        "UNIREC_EAGER_DEPTHWISE_CONV_LANE "
        f"lane={args.lane} status={report['status']} "
        f"weight_format={weight_format_before}->{weight_format_after} "
        f"control_before_ms={before_event:.6f} "
        f"profiled_ms={profile_event:.6f} "
        f"control_after_ms={after_event:.6f} "
        f"profile_overhead={profile_event / combined_control:.3f} "
        f"logical_to_fz1={logical_to_fz1['count']}/"
        f"{logical_to_fz1['duration_us'] / 1000.0:.6f}ms "
        f"group_repack={group_repack['count']}/"
        f"{group_repack['duration_us'] / 1000.0:.6f}ms "
        f"native_conv={native_convolution['count']}/"
        f"{native_convolution['duration_us'] / 1000.0:.6f}ms "
        f"any_conv={any_convolution['count']}/"
        f"{any_convolution['duration_us'] / 1000.0:.6f}ms",
        flush=True,
    )
    print(f"OUTPUT_JSON={output_json}", flush=True)
    if report["status"] != "ok":
        raise RuntimeError(report["status"])


def _run_matrix(args: argparse.Namespace) -> None:
    import torch

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    reports: dict[str, dict[str, Any]] = {}
    for lane in ("native", "fractal_z_1", "grouped_fz_384"):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--output-dir",
            str(output_dir / lane),
            "--lane",
            lane,
            "--device",
            args.device,
            "--warmups",
            str(args.warmups),
            "--control-repeats",
            str(args.control_repeats),
            "--parser-topn",
            str(args.parser_topn),
            "--profile-metric",
            args.profile_metric,
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        (output_dir / f"{lane}.stdout.log").write_text(
            completed.stdout + completed.stderr, encoding="utf-8"
        )
        print(completed.stdout, end="", flush=True)
        if completed.returncode != 0:
            print(completed.stderr, end="", file=sys.stderr, flush=True)
            raise RuntimeError(f"{lane} failed with exit {completed.returncode}")
        reports[lane] = json.loads(
            (output_dir / lane / "result.json").read_text(encoding="utf-8")
        )
        if lane == "native":
            target = reports[lane]["target_operations"]
            if target["fz1_to_grouped_fz384"]["count"] != 1:
                raise RuntimeError(
                    "native gate failed: target FZ:1 -> FZ:384 repack absent"
                )

    native_output = torch.load(
        output_dir / "native/output.pt", map_location="cpu", weights_only=True
    )
    fz1_output = torch.load(
        output_dir / "fractal_z_1/output.pt",
        map_location="cpu",
        weights_only=True,
    )
    grouped_output = torch.load(
        output_dir / "grouped_fz_384/output.pt",
        map_location="cpu",
        weights_only=True,
    )
    delta = (native_output - fz1_output).abs()
    grouped_delta = (native_output - grouped_output).abs()
    comparison = {
        "status": "ok",
        "native_replication_gate": True,
        "lanes": reports,
        "native_vs_fractal_z_1": {
            "exact": bool(torch.equal(native_output, fz1_output)),
            "allclose_atol_5e_2_rtol_5e_2": bool(
                torch.allclose(native_output, fz1_output, atol=5e-2, rtol=5e-2)
            ),
            "max_abs": float(delta.max().item()),
            "mean_abs": float(delta.mean().item()),
        },
        "native_vs_grouped_fz_384": {
            "exact": bool(torch.equal(native_output, grouped_output)),
            "allclose_atol_5e_2_rtol_5e_2": bool(
                torch.allclose(native_output, grouped_output, atol=5e-2, rtol=5e-2)
            ),
            "max_abs": float(grouped_delta.max().item()),
            "mean_abs": float(grouped_delta.mean().item()),
        },
    }
    comparison["fractal_z_1_expected_semantics_failure"] = not comparison[
        "native_vs_fractal_z_1"
    ]["allclose_atol_5e_2_rtol_5e_2"]
    grouped = reports["grouped_fz_384"]
    if not comparison["native_vs_grouped_fz_384"][
        "allclose_atol_5e_2_rtol_5e_2"
    ]:
        comparison["status"] = "grouped_candidate_semantics_failed"
    if grouped["target_operations"]["fz1_to_grouped_fz384"]["count"] != 0:
        comparison["status"] = "grouped_candidate_repacked"
    summary_path = output_dir / "matrix_summary.json"
    summary_path.write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
    )
    native = reports["native"]
    fz1 = reports["fractal_z_1"]
    print(
        "UNIREC_EAGER_DEPTHWISE_CONV_MATRIX "
        f"status={comparison['status']} "
        f"parity_exact={comparison['native_vs_fractal_z_1']['exact']} "
        f"native_ms={native['control_before']['device_event']['median_ms']:.6f} "
        f"fz1_ms={fz1['control_before']['device_event']['median_ms']:.6f} "
        f"native_repack={native['target_operations']['fz1_to_grouped_fz384']['count']} "
        f"fz1_repack={fz1['target_operations']['fz1_to_grouped_fz384']['count']} "
        f"grouped_ms={grouped['control_before']['device_event']['median_ms']:.6f} "
        f"grouped_repack={grouped['target_operations']['fz1_to_grouped_fz384']['count']} "
        f"grouped_exact={comparison['native_vs_grouped_fz_384']['exact']} "
        f"grouped_max_abs={comparison['native_vs_grouped_fz_384']['max_abs']:.9g} "
        f"grouped_mean_abs={comparison['native_vs_grouped_fz_384']['mean_abs']:.9g}",
        flush=True,
    )
    print(f"MATRIX_SUMMARY_JSON={summary_path}", flush=True)
    # A numerically invalid candidate is an experiment result, not a harness
    # failure. The per-lane structural gates still exit nonzero when profiling
    # did not observe the intended operation.


def main() -> None:
    args = parse_args()
    if args.lane == "matrix":
        _run_matrix(args)
    else:
        _run_lane(args)


if __name__ == "__main__":
    main()
