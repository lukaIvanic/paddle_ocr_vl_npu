#!/usr/bin/env python3
"""Profile the fixed graphs used by the UniRec prefill-only producer.

The suite measures graph replay with NPU events before and after profiling, then
captures one already-warmed replay window with ``torch_npu.profiler``.  It uses
the production compiled PP-DocLayoutV2 graph, all five production full-vision
graphs, and the packed S1024 cross-KV graph.  Page decode, crop preprocessing,
H2D, D2H, IPC, and graph compilation are intentionally outside these lanes.

Every graph has a static shape.  Synthetic values are therefore sufficient for
kernel and latency comparison: the same compiled operators and shapes execute
regardless of the image values.  The first-128 weighting is the exact W1/T16
call histogram measured on OmniDocBench v1.6 at offset 0.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import torch


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from modeling_optimized_unirec import (  # noqa: E402
    OptimizedUniRecRunner,
    synchronize_device,
)
from opendoc_layout_npu import (  # noqa: E402
    LAYOUT_DEPTHWISE_REWRITE_CHOICES,
    LAYOUT_WEIGHT_FORMAT_CHOICES,
    PPDocLayoutV2NpuAdapter,
)
from vision_full_batch import (  # noqa: E402
    DEFAULT_VISION_BUCKETS,
    BucketedFullVisionRuntime,
)


FIRST128_LAYOUT_CALLS = 128
FIRST128_TEXT_PREFILL_CALLS = 126
FIRST128_VISION_CALLS = {
    "960x64_b16": 63,
    "512x256_b16": 8,
    "960x256_b4": 52,
    "512x512_b8": 4,
    "960x512_b4": 12,
}
FIRST128_VISION_FALLBACK_ROWS = 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--layout-model", type=Path, required=True)
    parser.add_argument("--layout-cache-dir", type=Path, required=True)
    parser.add_argument("--recognition-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--control-repeats", type=int, default=10)
    parser.add_argument("--profile-steps", type=int, default=1)
    parser.add_argument("--parser-topn", type=int, default=50)
    parser.add_argument("--lane", choices=("all", "layout"), default="all")
    parser.add_argument(
        "--layout-dtype", choices=("float16", "float32"), default="float32"
    )
    parser.add_argument(
        "--layout-depthwise-rewrite",
        choices=LAYOUT_DEPTHWISE_REWRITE_CHOICES,
        default="native",
    )
    parser.add_argument(
        "--layout-weight-format",
        choices=LAYOUT_WEIGHT_FORMAT_CHOICES,
        default="native",
    )
    parser.add_argument("--layout-fuse-frozen-bn", action="store_true")
    parser.add_argument(
        "--profile-metric",
        choices=("pipe", "memory", "l2", "memory_access"),
        default="pipe",
    )
    args = parser.parse_args(argv)
    if args.warmup < 1:
        parser.error("--warmup must be at least 1 to exclude cold graph loading")
    if args.control_repeats < 1:
        parser.error("--control-repeats must be positive")
    if args.profile_steps < 1:
        parser.error("--profile-steps must be positive")
    if args.parser_topn < 1:
        parser.error("--parser-topn must be positive")
    return args


def _physical_devices() -> list[int]:
    raw = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    devices = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not devices:
        raise RuntimeError("ASCEND_RT_VISIBLE_DEVICES must select one physical NPU")
    if len(devices) != 1:
        raise RuntimeError(f"profile suite requires one visible NPU, got {devices}")
    if 5 in devices:
        raise RuntimeError("physical NPU 5 is excluded from UniRec experiments")
    return devices


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _sample_summary(values: Sequence[float]) -> dict[str, Any]:
    samples = [float(value) for value in values]
    return {
        "samples_ms": samples,
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "p05_ms": _percentile(samples, 0.05),
        "p95_ms": _percentile(samples, 0.95),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def _measure_replays(
    run: Callable[[], Any],
    *,
    device: str,
    repeats: int,
) -> dict[str, Any]:
    import torch_npu

    event_ms: list[float] = []
    wall_ms: list[float] = []
    output = None
    for _ in range(repeats):
        synchronize_device(device)
        start = torch_npu.npu.Event(enable_timing=True)
        end = torch_npu.npu.Event(enable_timing=True)
        wall_started = time.perf_counter()
        start.record()
        output = run()
        end.record()
        end.synchronize()
        wall_ms.append((time.perf_counter() - wall_started) * 1000.0)
        event_ms.append(float(start.elapsed_time(end)))
    if output is None:
        raise RuntimeError("graph replay produced no output")
    return {
        "device_event": _sample_summary(event_ms),
        "synchronized_wall": _sample_summary(wall_ms),
    }


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


def _profile_lane(
    name: str,
    run: Callable[[], Any],
    *,
    output_root: Path,
    device: str,
    warmup: int,
    control_repeats: int,
    profile_steps: int,
    profile_metric: str,
    parser_topn: int,
    first128_calls: int,
    input_contract: dict[str, Any],
) -> dict[str, Any]:
    import torch_npu.profiler as npu_prof

    lane_dir = output_root / name
    lane_dir.mkdir(parents=True, exist_ok=False)
    warmup_wall_ms = []
    for _ in range(warmup):
        synchronize_device(device)
        started = time.perf_counter()
        run()
        synchronize_device(device)
        warmup_wall_ms.append((time.perf_counter() - started) * 1000.0)

    before = _measure_replays(run, device=device, repeats=control_repeats)
    profile_dir = lane_dir / f"profile_{profile_metric}"
    profile_dir.mkdir(parents=True, exist_ok=False)
    schedule = npu_prof.schedule(wait=0, warmup=0, active=1, repeat=1)
    synchronize_device(device)
    profile_started = time.perf_counter()
    with npu_prof.profile(
        activities=[
            npu_prof.ProfilerActivity.CPU,
            npu_prof.ProfilerActivity.NPU,
        ],
        schedule=schedule,
        experimental_config=_profiler_config(profile_metric),
        on_trace_ready=npu_prof.tensorboard_trace_handler(
            str(profile_dir), analyse_flag=True
        ),
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
    ) as profiler:
        with torch.profiler.record_function(f"unirec.prefill_graph.{name}"):
            for _ in range(profile_steps):
                run()
        synchronize_device(device)
        profiler.step()
    profile_wall_s = time.perf_counter() - profile_started
    after = _measure_replays(run, device=device, repeats=control_repeats)
    parsed = _parse_profile(profile_dir, topn=parser_topn)

    steady_event_ms = (
        float(before["device_event"]["mean_ms"])
        + float(after["device_event"]["mean_ms"])
    ) / 2.0
    return {
        "name": name,
        "input_contract": input_contract,
        "first128_calls": int(first128_calls),
        "warmup_wall_ms": warmup_wall_ms,
        "control_before": before,
        "control_after": after,
        "steady_device_event_mean_ms": steady_event_ms,
        "weighted_first128_device_s": steady_event_ms * first128_calls / 1000.0,
        "profile_metric": profile_metric,
        "profile_steps": profile_steps,
        "profile_wall_s": profile_wall_s,
        "profile_dir": str(profile_dir),
        "parsed_profile": parsed,
    }


def _layout_lane(
    args: argparse.Namespace,
    output_root: Path,
) -> dict[str, Any]:
    detector = PPDocLayoutV2NpuAdapter(
        model_path=args.layout_model.expanduser().resolve(),
        device=args.device,
        dtype=args.layout_dtype,
        threshold=0.4,
        profile_stages=False,
        execution="torchair",
        compile_cache_dir=args.layout_cache_dir.expanduser().resolve(),
        batch_size=1,
        depthwise_rewrite=args.layout_depthwise_rewrite,
        weight_format=args.layout_weight_format,
        fuse_frozen_bn=args.layout_fuse_frozen_bn,
    )
    if detector.compiled_runtime is None:
        raise RuntimeError("layout profiler requires the compiled runtime")
    pixel_values = torch.zeros(
        (1, 3, 800, 800),
        dtype={"float16": torch.float16, "float32": torch.float32}[
            args.layout_dtype
        ],
        device=args.device,
    )
    run = lambda: detector.compiled_runtime(pixel_values)
    result = _profile_lane(
        f"layout_b1_800x800_{args.layout_dtype}_{args.layout_depthwise_rewrite}_"
        f"{args.layout_weight_format}_frozenbn{int(args.layout_fuse_frozen_bn)}",
        run,
        output_root=output_root,
        device=args.device,
        warmup=args.warmup,
        control_repeats=args.control_repeats,
        profile_steps=args.profile_steps,
        profile_metric=args.profile_metric,
        parser_topn=args.parser_topn,
        first128_calls=FIRST128_LAYOUT_CALLS,
        input_contract={
            "pixel_values": [1, 3, 800, 800],
            "dtype": args.layout_dtype,
            "depthwise_rewrite": args.layout_depthwise_rewrite,
            "weight_format": args.layout_weight_format,
            "fuse_frozen_bn": args.layout_fuse_frozen_bn,
            "execution": "compiled_fullgraph",
        },
    )
    del pixel_values, detector
    synchronize_device(args.device)
    torch.npu.empty_cache()
    return result


def _recognition_lanes(
    args: argparse.Namespace,
    output_root: Path,
) -> list[dict[str, Any]]:
    runner = OptimizedUniRecRunner(
        model_path=args.model_path.expanduser().resolve(),
        device=args.device,
        dtype="float16",
        compile_cache_dir=args.recognition_cache_dir.expanduser().resolve(),
    )
    vision = BucketedFullVisionRuntime(runner, specs=DEFAULT_VISION_BUCKETS)
    lanes = []
    for spec in DEFAULT_VISION_BUCKETS:
        pixels = torch.zeros(
            (spec.batch_size, 3, spec.height, spec.width),
            dtype=runner.dtype,
            device=args.device,
        )
        masks = tuple(
            torch.ones(
                (
                    spec.batch_size,
                    1,
                    spec.height // factor,
                    spec.width // factor,
                ),
                dtype=runner.dtype,
                device=args.device,
            )
            for factor in (2, 4, 8, 16, 32)
        )
        compiled = vision.compiled[spec.key]
        run = lambda compiled=compiled, pixels=pixels, masks=masks: compiled(
            pixels, *masks
        )
        lanes.append(
            _profile_lane(
                f"vision_{spec.key}_fp16",
                run,
                output_root=output_root,
                device=args.device,
                warmup=args.warmup,
                control_repeats=args.control_repeats,
                profile_steps=args.profile_steps,
                profile_metric=args.profile_metric,
                parser_topn=args.parser_topn,
                first128_calls=FIRST128_VISION_CALLS[spec.key],
                input_contract={
                    "pixel_values": [
                        spec.batch_size,
                        3,
                        spec.height,
                        spec.width,
                    ],
                    "mask_shapes": [
                        list(mask.shape) for mask in masks
                    ],
                    "dtype": "float16",
                    "execution": "compiled_masked_full_vision",
                },
            )
        )
        del pixels, masks

    text_runtime = runner._get_compiled_packed_text_prefill_runtime()
    packed = torch.zeros(
        (1, text_runtime.bucket, int(runner.config.d_model)),
        dtype=runner.dtype,
        device=args.device,
    )
    text_run = lambda: text_runtime.compiled(packed)
    lanes.append(
        _profile_lane(
            "cross_kv_packed_b1_s1024_fp16",
            text_run,
            output_root=output_root,
            device=args.device,
            warmup=args.warmup,
            control_repeats=args.control_repeats,
            profile_steps=args.profile_steps,
            profile_metric=args.profile_metric,
            parser_topn=args.parser_topn,
            first128_calls=FIRST128_TEXT_PREFILL_CALLS,
            input_contract={
                "encoder_hidden_states": [
                    1,
                    text_runtime.bucket,
                    int(runner.config.d_model),
                ],
                "dtype": "float16",
                "execution": "compiled_packed_cross_kv",
            },
        )
    )
    del packed, vision, runner
    synchronize_device(args.device)
    torch.npu.empty_cache()
    return lanes


def _environment(physical_devices: list[int]) -> dict[str, Any]:
    import torch_npu

    try:
        import importlib.metadata as metadata

        torchair_version = metadata.version("torchair")
    except Exception:
        torchair_version = None
    return {
        "physical_devices": physical_devices,
        "logical_device": 0,
        "device_name": torch.npu.get_device_name(0),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "torchair": torchair_version,
        "python": sys.version.replace("\n", " "),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    physical_devices = _physical_devices()
    import torch_npu

    if not torch.npu.is_available():
        raise RuntimeError("UniRec prefill graph profiling requires an NPU")
    torch_npu.npu.set_compile_mode(jit_compile=False)
    output_root = args.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=False)

    started = time.perf_counter()
    layout = _layout_lane(args, output_root)
    recognition = (
        _recognition_lanes(args, output_root) if args.lane == "all" else []
    )
    lanes = [layout, *recognition]
    vision_weighted_s = sum(
        float(lane["weighted_first128_device_s"])
        for lane in recognition
        if lane["name"].startswith("vision_")
    )
    text_weighted_s = next(
        (
            float(lane["weighted_first128_device_s"])
            for lane in recognition
            if lane["name"].startswith("cross_kv_")
        ),
        0.0,
    )
    report = {
        "format": "unirec_prefill_graph_profile_suite_v1",
        "environment": _environment(physical_devices),
        "config": {
            "model_path": str(args.model_path.expanduser().resolve()),
            "layout_model": str(args.layout_model.expanduser().resolve()),
            "layout_cache_dir": str(args.layout_cache_dir.expanduser().resolve()),
            "recognition_cache_dir": str(
                args.recognition_cache_dir.expanduser().resolve()
            ),
            "device": args.device,
            "warmup": args.warmup,
            "control_repeats": args.control_repeats,
            "profile_steps": args.profile_steps,
            "profile_metric": args.profile_metric,
            "parser_topn": args.parser_topn,
            "lane": args.lane,
            "layout_dtype": args.layout_dtype,
            "layout_depthwise_rewrite": args.layout_depthwise_rewrite,
            "layout_weight_format": args.layout_weight_format,
            "layout_fuse_frozen_bn": args.layout_fuse_frozen_bn,
        },
        "first128_workload": {
            "layout_calls": FIRST128_LAYOUT_CALLS,
            "vision_calls": FIRST128_VISION_CALLS,
            "vision_fallback_rows_not_profiled": FIRST128_VISION_FALLBACK_ROWS,
            "text_prefill_calls": FIRST128_TEXT_PREFILL_CALLS,
        },
        "lanes": lanes,
        "weighted_first128_device_s": {
            "layout_graph": float(layout["weighted_first128_device_s"]),
            "vision_graphs": vision_weighted_s,
            "cross_kv_graph": text_weighted_s,
            "recognition_graphs_total": vision_weighted_s + text_weighted_s,
        },
        "suite_wall_s": time.perf_counter() - started,
        "measurement_scope": {
            "included": "already-warmed fixed-shape compiled graph replay",
            "excluded": [
                "graph compilation and cache loading",
                "page file IO and layout image processor",
                "crop preprocessing",
                "host-to-device compact input transfer and normalization",
                "cross-KV device-to-host export",
                "IPC and packing",
                "one eager fallback vision crop",
            ],
        },
    }
    summary_path = output_root / "profile_suite_summary.json"
    summary_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "UNIREC_PREFILL_PROFILE_HEADLINE "
        f"device={report['environment']['device_name']} "
        f"layout_weighted_s={report['weighted_first128_device_s']['layout_graph']:.6f} "
        f"vision_weighted_s={vision_weighted_s:.6f} "
        f"cross_kv_weighted_s={text_weighted_s:.6f} "
        f"recognition_weighted_s={vision_weighted_s + text_weighted_s:.6f}",
        flush=True,
    )
    for lane in lanes:
        print(
            "UNIREC_PREFILL_PROFILE_LANE "
            f"name={lane['name']} calls={lane['first128_calls']} "
            f"device_ms={lane['steady_device_event_mean_ms']:.6f} "
            f"weighted_s={lane['weighted_first128_device_s']:.6f}",
            flush=True,
        )
    print(f"UNIREC_PREFILL_PROFILE_OUTPUT {summary_path}", flush=True)


if __name__ == "__main__":
    main()
