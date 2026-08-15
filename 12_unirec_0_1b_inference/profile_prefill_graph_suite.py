#!/usr/bin/env python3
"""Profile the fixed forwards used by the UniRec prefill-only producer.

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
    DEFAULT_LAYOUT_DEPTHWISE_REWRITE,
    DEFAULT_LAYOUT_WEIGHT_FORMAT,
    LAYOUT_DEPTHWISE_REWRITE_CHOICES,
    LAYOUT_WEIGHT_FORMAT_CHOICES,
    PPDocLayoutV2NpuAdapter,
    prepare_layout_resized_uint8_exact,
)
from vision_full_batch import (  # noqa: E402
    DEFAULT_VISION_BUCKETS,
    VISION_FOCAL_DEPTHWISE_REWRITE_CHOICES,
    VISION_WEIGHT_FORMAT_CHOICES,
    BucketedFullVisionRuntime,
    _new_masked_full_encoder_module,
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
    parser.add_argument(
        "--layout-input-image",
        type=Path,
        help=(
            "Real page used to construct the exact production 800x800 input. "
            "This is required for representative eager reading-order work."
        ),
    )
    parser.add_argument("--layout-cache-dir", type=Path, required=True)
    parser.add_argument("--recognition-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--control-repeats", type=int, default=10)
    parser.add_argument("--profile-steps", type=int, default=1)
    parser.add_argument(
        "--torch-cpu-threads",
        type=int,
        default=1,
        help=(
            "PyTorch intra-op CPU threads used while launching the NPU "
            "forward. One matches the representative W1/T1 layout lab."
        ),
    )
    parser.add_argument(
        "--skip-profiler",
        action="store_true",
        help=(
            "Measure warmed graph replays without capturing or parsing an "
            "NPU profile. Use this for low-wall-time latency A/B gates."
        ),
    )
    parser.add_argument("--parser-topn", type=int, default=50)
    parser.add_argument(
        "--lane", choices=("all", "layout", "vision"), default="all"
    )
    parser.add_argument(
        "--vision-depthwise-rewrite",
        choices=VISION_FOCAL_DEPTHWISE_REWRITE_CHOICES,
        default="native",
    )
    parser.add_argument(
        "--vision-bucket",
        action="append",
        choices=tuple(spec.key for spec in DEFAULT_VISION_BUCKETS),
        help="profile only this vision bucket; repeat to select more than one",
    )
    parser.add_argument(
        "--vision-weight-format",
        choices=VISION_WEIGHT_FORMAT_CHOICES,
        default="native",
    )
    parser.add_argument(
        "--allow-vision-parity-drift",
        action="store_true",
        help=(
            "Record same-process native tensor differences without rejecting "
            "the candidate. This is a performance-experiment control, not a "
            "quality acceptance claim."
        ),
    )
    parser.add_argument(
        "--save-vision-outputs-dir",
        type=Path,
        help="Save one warmed compiled output per selected vision bucket.",
    )
    parser.add_argument(
        "--reference-vision-outputs-dir",
        type=Path,
        help=(
            "Require every selected compiled vision output to be bit-exact "
            "to a matching output saved by a native compiled run."
        ),
    )
    parser.add_argument(
        "--layout-execution",
        choices=("eager", "torchair"),
        default="torchair",
        help="Profile the faithful eager model or its static TorchAir graph.",
    )
    parser.add_argument(
        "--layout-dtype", choices=("float16", "float32"), default="float32"
    )
    parser.add_argument(
        "--layout-reading-order-dtype",
        choices=("float16", "float32"),
        default=None,
        help=(
            "Override only the learned reading-order module dtype. By default "
            "it follows --layout-dtype."
        ),
    )
    parser.add_argument(
        "--layout-depthwise-rewrite",
        choices=LAYOUT_DEPTHWISE_REWRITE_CHOICES,
        default=DEFAULT_LAYOUT_DEPTHWISE_REWRITE,
    )
    parser.add_argument(
        "--layout-weight-format",
        choices=LAYOUT_WEIGHT_FORMAT_CHOICES,
        default=DEFAULT_LAYOUT_WEIGHT_FORMAT,
    )
    parser.add_argument("--layout-fuse-frozen-bn", action="store_true")
    parser.add_argument("--layout-fuse-eval-bn", action="store_true")
    parser.add_argument(
        "--layout-precompute-frozen-bn-affine", action="store_true"
    )
    parser.add_argument(
        "--layout-preformat-frozen-bn-buffers", action="store_true"
    )
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
    if args.torch_cpu_threads < 1:
        parser.error("--torch-cpu-threads must be positive")
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
    excluded = sorted(set(devices) & {5, 6})
    if excluded:
        raise RuntimeError(
            f"physical NPUs 5 and 6 are excluded from UniRec experiments: {excluded}"
        )
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
    skip_profiler: bool,
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
    profile_wall_s = 0.0
    parsed = None
    if not skip_profiler:
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
        parsed = _parse_profile(profile_dir, topn=parser_topn)
    after = _measure_replays(run, device=device, repeats=control_repeats)

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
        "profile_dir": str(profile_dir) if not skip_profiler else None,
        "parsed_profile": parsed,
    }


def _layout_lane(
    args: argparse.Namespace,
    output_root: Path,
) -> dict[str, Any]:
    reading_order_dtype = (
        args.layout_reading_order_dtype or args.layout_dtype
    )
    detector = PPDocLayoutV2NpuAdapter(
        model_path=args.layout_model.expanduser().resolve(),
        device=args.device,
        dtype=args.layout_dtype,
        reading_order_dtype=reading_order_dtype,
        threshold=0.4,
        profile_stages=False,
        execution=args.layout_execution,
        compile_cache_dir=args.layout_cache_dir.expanduser().resolve(),
        batch_size=1,
        depthwise_rewrite=args.layout_depthwise_rewrite,
        weight_format=args.layout_weight_format,
        fuse_frozen_bn=args.layout_fuse_frozen_bn,
        fuse_eval_bn=args.layout_fuse_eval_bn,
        precompute_frozen_bn_affine=(
            args.layout_precompute_frozen_bn_affine
        ),
        preformat_frozen_bn_buffers=(
            args.layout_preformat_frozen_bn_buffers
        ),
    )
    if args.layout_input_image is not None:
        from layout_page_input import decode_page_rgb, materialize_layout_rgb

        input_image = args.layout_input_image.expanduser().resolve()
        rgb, _ = decode_page_rgb(input_image)
        layout_rgb = materialize_layout_rgb(rgb)
        host_pixels = prepare_layout_resized_uint8_exact([layout_rgb])[
            "pixel_values"
        ]
        pixel_values = host_pixels.to(device=args.device)
        pixel_values = pixel_values.to(dtype=torch.float32).div_(255.0)
        if args.layout_dtype != "float32":
            pixel_values = pixel_values.to(dtype=torch.float16)
        synchronize_device(args.device)
        input_source = str(input_image)
    else:
        pixel_values = torch.zeros(
            (1, 3, 800, 800),
            dtype={"float16": torch.float16, "float32": torch.float32}[
                args.layout_dtype
            ],
            device=args.device,
        )
        input_source = "synthetic_zeros"
    if args.layout_execution == "torchair":
        if detector.compiled_runtime is None:
            raise RuntimeError("compiled layout profiler has no compiled runtime")
        run = lambda: detector.compiled_runtime(pixel_values)
        execution_contract = "compiled_fullgraph"
    else:
        if detector.compiled_runtime is not None:
            raise RuntimeError("eager layout profiler unexpectedly compiled a graph")
        run = lambda: detector.model(pixel_values=pixel_values)
        execution_contract = "raw_eager_model_forward"
    result = _profile_lane(
        f"layout_b1_800x800_{args.layout_execution}_{args.layout_dtype}_"
        f"readingorder_{reading_order_dtype}_{args.layout_depthwise_rewrite}_"
        f"{args.layout_weight_format}_frozenbn{int(args.layout_fuse_frozen_bn)}_"
        f"evalbn{int(args.layout_fuse_eval_bn)}_"
        f"precomputedfrozenbn{int(args.layout_precompute_frozen_bn_affine)}_"
        f"formattedfrozenbnbuffers{int(args.layout_preformat_frozen_bn_buffers)}",
        run,
        output_root=output_root,
        device=args.device,
        warmup=args.warmup,
        control_repeats=args.control_repeats,
        profile_steps=args.profile_steps,
        profile_metric=args.profile_metric,
        parser_topn=args.parser_topn,
        skip_profiler=args.skip_profiler,
        first128_calls=FIRST128_LAYOUT_CALLS,
        input_contract={
            "pixel_values": [1, 3, 800, 800],
            "source": input_source,
            "dtype": args.layout_dtype,
            "reading_order_dtype": reading_order_dtype,
            "depthwise_rewrite": args.layout_depthwise_rewrite,
            "weight_format": args.layout_weight_format,
            "fuse_frozen_bn": args.layout_fuse_frozen_bn,
            "fuse_eval_bn": args.layout_fuse_eval_bn,
            "precompute_frozen_bn_affine": (
                args.layout_precompute_frozen_bn_affine
            ),
            "preformat_frozen_bn_buffers": (
                args.layout_preformat_frozen_bn_buffers
            ),
            "execution": execution_contract,
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
    save_outputs_dir = (
        args.save_vision_outputs_dir.expanduser().resolve()
        if args.save_vision_outputs_dir is not None
        else None
    )
    reference_outputs_dir = (
        args.reference_vision_outputs_dir.expanduser().resolve()
        if args.reference_vision_outputs_dir is not None
        else None
    )
    if save_outputs_dir is not None:
        save_outputs_dir.mkdir(parents=True, exist_ok=False)
    if reference_outputs_dir is not None and not reference_outputs_dir.is_dir():
        raise FileNotFoundError(reference_outputs_dir)
    selected_keys = set(args.vision_bucket or ())
    vision_specs = tuple(
        spec
        for spec in DEFAULT_VISION_BUCKETS
        if not selected_keys or spec.key in selected_keys
    )
    baseline_outputs: dict[str, torch.Tensor] = {}
    validation_inputs: dict[
        str, tuple[torch.Tensor, tuple[torch.Tensor, ...]]
    ] = {}
    if (
        args.vision_depthwise_rewrite != "native"
        or args.vision_weight_format != "native"
    ):
        for spec in vision_specs:
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
            native_module = _new_masked_full_encoder_module(runner, spec)
            with torch.inference_mode():
                baseline_outputs[spec.key] = native_module(
                    pixels, *masks
                ).detach().clone()
            validation_inputs[spec.key] = (pixels, masks)
            del native_module
        synchronize_device(args.device)
    vision = BucketedFullVisionRuntime(
        runner,
        specs=vision_specs,
        focal_depthwise_rewrite=args.vision_depthwise_rewrite,
        weight_format=args.vision_weight_format,
    )
    lanes = []
    for spec in vision_specs:
        if spec.key in validation_inputs:
            pixels, masks = validation_inputs[spec.key]
        else:
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
        with torch.inference_mode():
            compiled_output = run().detach()
        synchronize_device(args.device)
        compiled_output_cpu = compiled_output.cpu()
        del compiled_output
        compiled_reference_validation = {
            "reference": "not_requested",
            "exact": True,
            "max_abs": 0.0,
            "mean_abs": 0.0,
        }
        if reference_outputs_dir is not None:
            reference_path = reference_outputs_dir / f"{spec.key}.pt"
            reference_output = torch.load(
                reference_path,
                map_location="cpu",
                weights_only=True,
            )
            difference = (compiled_output_cpu - reference_output).abs()
            compiled_reference_validation = {
                "reference": str(reference_path),
                "exact": bool(torch.equal(compiled_output_cpu, reference_output)),
                "max_abs": float(difference.max().item()),
                "mean_abs": float(difference.mean().item()),
            }
            if not compiled_reference_validation["exact"]:
                raise RuntimeError(
                    "compiled vision output is not bit-exact to native "
                    f"compiled reference for {spec.key}: "
                    f"{compiled_reference_validation}"
                )
            del reference_output, difference
        if save_outputs_dir is not None:
            torch.save(compiled_output_cpu, save_outputs_dir / f"{spec.key}.pt")
        del compiled_output_cpu
        validation = {
            "rewrite": args.vision_depthwise_rewrite,
            "weight_format": args.vision_weight_format,
            "reference": "not_required_for_native",
            "allclose_atol_5e_2_rtol_5e_2": True,
            "max_abs": 0.0,
            "mean_abs": 0.0,
        }
        if spec.key in baseline_outputs:
            with torch.inference_mode():
                candidate = run()
            synchronize_device(args.device)
            difference = (candidate - baseline_outputs[spec.key]).abs()
            validation = {
                "rewrite": args.vision_depthwise_rewrite,
                "weight_format": args.vision_weight_format,
                "reference": "same_process_native_masked_full_encoder",
                "allclose_atol_5e_2_rtol_5e_2": bool(
                    torch.allclose(
                        candidate,
                        baseline_outputs[spec.key],
                        atol=5e-2,
                        rtol=5e-2,
                    )
                ),
                "max_abs": float(difference.max().item()),
                "mean_abs": float(difference.mean().item()),
                "parity_policy": (
                    "report" if args.allow_vision_parity_drift else "enforce"
                ),
            }
            if (
                not validation["allclose_atol_5e_2_rtol_5e_2"]
                and not args.allow_vision_parity_drift
            ):
                raise RuntimeError(
                    f"vision rewrite parity failed for {spec.key}: {validation}"
                )
            del candidate, difference
        lane_name = f"vision_{spec.key}_fp16"
        if args.vision_depthwise_rewrite != "native":
            lane_name += f"_dw{args.vision_depthwise_rewrite}"
        if args.vision_weight_format != "native":
            lane_name += f"_w{args.vision_weight_format}"
        lane = _profile_lane(
            lane_name,
            run,
            output_root=output_root,
            device=args.device,
            warmup=args.warmup,
            control_repeats=args.control_repeats,
            profile_steps=args.profile_steps,
            profile_metric=args.profile_metric,
            parser_topn=args.parser_topn,
            skip_profiler=args.skip_profiler,
            first128_calls=FIRST128_VISION_CALLS[spec.key],
            input_contract={
                "pixel_values": [
                    spec.batch_size,
                    3,
                    spec.height,
                    spec.width,
                ],
                "mask_shapes": [list(mask.shape) for mask in masks],
                "dtype": "float16",
                "execution": "compiled_masked_full_vision",
                "focal_depthwise_rewrite": args.vision_depthwise_rewrite,
                "weight_format": args.vision_weight_format,
            },
        )
        lane["focal_depthwise_rewrite_summary"] = (
            vision.focal_depthwise_rewrite_summary
        )
        lane["rewrite_validation"] = validation
        lane["compiled_reference_validation"] = (
            compiled_reference_validation
        )
        lane["weight_format_summary"] = vision.weight_format_summary
        lanes.append(lane)
        del pixels, masks

    if args.lane != "vision":
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
                skip_profiler=args.skip_profiler,
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
        del packed
    del vision, runner
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
    torch.set_num_threads(args.torch_cpu_threads)
    torch.set_num_interop_threads(args.torch_cpu_threads)
    physical_devices = _physical_devices()
    import torch_npu

    if not torch.npu.is_available():
        raise RuntimeError("UniRec prefill graph profiling requires an NPU")
    torch_npu.npu.set_compile_mode(jit_compile=False)
    output_root = args.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=False)

    started = time.perf_counter()
    layout = _layout_lane(args, output_root) if args.lane != "vision" else None
    recognition = (
        _recognition_lanes(args, output_root) if args.lane != "layout" else []
    )
    if args.lane == "vision":
        recognition = [
            lane for lane in recognition if lane["name"].startswith("vision_")
        ]
    lanes = [*([layout] if layout is not None else []), *recognition]
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
            "layout_input_image": (
                str(args.layout_input_image.expanduser().resolve())
                if args.layout_input_image is not None
                else None
            ),
            "layout_cache_dir": str(args.layout_cache_dir.expanduser().resolve()),
            "recognition_cache_dir": str(
                args.recognition_cache_dir.expanduser().resolve()
            ),
            "device": args.device,
            "warmup": args.warmup,
            "control_repeats": args.control_repeats,
            "profile_steps": args.profile_steps,
            "torch_cpu_threads": args.torch_cpu_threads,
            "torch_intraop_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "skip_profiler": args.skip_profiler,
            "profile_metric": args.profile_metric,
            "parser_topn": args.parser_topn,
            "lane": args.lane,
            "vision_depthwise_rewrite": args.vision_depthwise_rewrite,
            "vision_weight_format": args.vision_weight_format,
            "vision_buckets": args.vision_bucket,
            "save_vision_outputs_dir": (
                str(args.save_vision_outputs_dir.expanduser().resolve())
                if args.save_vision_outputs_dir is not None
                else None
            ),
            "reference_vision_outputs_dir": (
                str(args.reference_vision_outputs_dir.expanduser().resolve())
                if args.reference_vision_outputs_dir is not None
                else None
            ),
            "layout_dtype": args.layout_dtype,
            "layout_reading_order_dtype": (
                args.layout_reading_order_dtype or args.layout_dtype
            ),
            "layout_execution": args.layout_execution,
            "layout_depthwise_rewrite": args.layout_depthwise_rewrite,
            "layout_weight_format": args.layout_weight_format,
            "layout_fuse_frozen_bn": args.layout_fuse_frozen_bn,
            "layout_fuse_eval_bn": args.layout_fuse_eval_bn,
            "layout_precompute_frozen_bn_affine": (
                args.layout_precompute_frozen_bn_affine
            ),
            "layout_preformat_frozen_bn_buffers": (
                args.layout_preformat_frozen_bn_buffers
            ),
        },
        "first128_workload": {
            "layout_calls": FIRST128_LAYOUT_CALLS,
            "vision_calls": FIRST128_VISION_CALLS,
            "vision_fallback_rows_not_profiled": FIRST128_VISION_FALLBACK_ROWS,
            "text_prefill_calls": FIRST128_TEXT_PREFILL_CALLS,
        },
        "lanes": lanes,
        "weighted_first128_device_s": {
            "layout_graph": (
                float(layout["weighted_first128_device_s"])
                if layout is not None
                else 0.0
            ),
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
