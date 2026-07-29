#!/usr/bin/env python3
"""Profile one production compiled PromptFA B1 vision graph.

The experiment deliberately has one selected shape and one graph boundary:

    VisionPrefillRuntime.compiled[sequence_length](
        prefix_hidden_states,
        rope_cos,
        rope_sin,
        attention_mask,
    )

It measures the same graph with NPU events before, during, and after native
torch_npu profiling.  The pre/post controls quantify profiler perturbation;
only those controls are throughput measurements.  The profiled pass is a
diagnostic capture of the already-warmed graph replay, not compilation or
model setup.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
REPO_ROOT = EXPERIMENT_ROOT.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXPERIMENT_ROOT))

from paddleocr_vl.model.modeling import (  # noqa: E402
    LocalPaddleOCRVLForConditionalGeneration,
)
from paddleocr_vl.model.vision_prefill import (  # noqa: E402
    VisionPrefillRuntime,
    vision_cache_dir_for_bucket,
)
from utils.timing import DeviceTimeline, synchronize  # noqa: E402
from vision_lab import DEFAULT_MODEL, _environment  # noqa: E402


BATCH_SIZE = 1
SUPPORTED_SEQUENCE_LENGTHS = (512, 2048)
DEFAULT_CACHE_ROOT = (
    REPO_ROOT / ".runtime_cache/09_persistent_page_engine_vision_torchair"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "tmp/09_persistent_page_engine/vision_lab"
)
DEFAULT_PROFILE_ROOT = (
    REPO_ROOT / ".runtime_cache/09_persistent_page_engine_profiles"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence-length",
        type=int,
        choices=SUPPORTED_SEQUENCE_LENGTHS,
        required=True,
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument(
        "--allow-compile-if-missing",
        action="store_true",
        help=(
            "Permit cache preparation to compile the selected graph. "
            "The final profiler run should omit this flag."
        ),
    )
    parser.add_argument(
        "--prepare-cache-only",
        action="store_true",
        help=(
            "Load or compile the selected graph, write cache_preparation.json, "
            "and exit before throughput measurement or profiling."
        ),
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--control-repeats", type=int, default=20)
    parser.add_argument("--profile-warmup-steps", type=int, default=1)
    parser.add_argument("--profile-steps", type=int, default=5)
    parser.add_argument("--parser-topn", type=int, default=30)
    args = parser.parse_args(argv)
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.control_repeats <= 0:
        parser.error("--control-repeats must be positive")
    if args.profile_warmup_steps <= 0:
        parser.error("--profile-warmup-steps must be positive")
    if args.profile_steps <= 0:
        parser.error("--profile-steps must be positive")
    if args.parser_topn <= 0:
        parser.error("--parser-topn must be positive")
    return args


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _duration_summary(samples_ms: Sequence[float]) -> dict[str, Any]:
    values = [float(value) for value in samples_ms]
    mean_ms = statistics.mean(values)
    median_ms = statistics.median(values)
    return {
        "samples_ms": values,
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "p05_ms": _percentile(values, 0.05),
        "p95_ms": _percentile(values, 0.95),
    }


def _replay_summary(
    samples_ms: Sequence[float],
    *,
    physical_tokens: int,
) -> dict[str, Any]:
    summary = _duration_summary(samples_ms)
    mean_ms = float(summary["mean_ms"])
    median_ms = float(summary["median_ms"])
    summary.update(
        {
            "physical_tokens_per_s_mean": physical_tokens
            / (mean_ms / 1000.0),
            "physical_tokens_per_s_median": physical_tokens
            / (median_ms / 1000.0),
        }
    )
    return summary


def _measurement_summary(
    device_ms: Sequence[float],
    synchronized_wall_ms: Sequence[float],
    *,
    physical_tokens: int,
) -> dict[str, Any]:
    return {
        "device_event": _replay_summary(
            device_ms,
            physical_tokens=physical_tokens,
        ),
        "synchronized_host_wall": _replay_summary(
            synchronized_wall_ms,
            physical_tokens=physical_tokens,
        ),
    }


def _run_measured(
    run: Callable[..., torch.Tensor],
    inputs: tuple[torch.Tensor, ...],
    *,
    device: torch.device,
    repeats: int,
    physical_tokens: int,
) -> tuple[dict[str, Any], torch.Tensor]:
    device_ms: list[float] = []
    synchronized_wall_ms: list[float] = []
    output: torch.Tensor | None = None
    for _ in range(repeats):
        timeline = DeviceTimeline(device)
        wall_started = time.perf_counter()
        output = timeline.measure("graph_replay", lambda: run(*inputs))
        spans = timeline.resolve_spans()
        synchronized_wall_ms.append(
            (time.perf_counter() - wall_started) * 1000.0
        )
        device_ms.append(
            float(spans["graph_replay"]["seconds"]) * 1000.0
        )
    if output is None:
        raise RuntimeError("measurement produced no output")
    return (
        _measurement_summary(
            device_ms,
            synchronized_wall_ms,
            physical_tokens=physical_tokens,
        ),
        output,
    )


def _cache_populated(path: Path) -> bool:
    return path.is_dir() and any(item.is_file() for item in path.rglob("*"))


def _profiler_config() -> Any:
    import torch_npu.profiler as npu_prof

    return npu_prof._ExperimentalConfig(
        profiler_level=npu_prof.ProfilerLevel.Level1,
        aic_metrics=npu_prof.AiCMetrics.PipeUtilization,
        l2_cache=False,
        export_type=npu_prof.ExportType.Text,
        data_simplification=False,
    )


def _run_parser(profile_dir: Path, *, topn: int) -> dict[str, Any]:
    parser_path = (
        REPO_ROOT
        / "07_vision_prefill_optimization"
        / "parse_static_visual_encoder_profile.py"
    )
    command = [
        sys.executable,
        str(parser_path),
        "--profile-dir",
        str(profile_dir),
        "--topn",
        str(topn),
        "--skip-trace",
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        "command": command,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "parsed_json": str(profile_dir / "parsed_profile_summary.json"),
        "parsed_markdown": str(profile_dir / "parsed_profile_summary.md"),
    }


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0 else None


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    import torch_npu  # noqa: F401
    import torch_npu.profiler as npu_prof

    device = torch.device("npu:0")
    if not torch.npu.is_available():
        raise RuntimeError("the B1 vision profiler requires an NPU")
    torch.npu.set_compile_mode(jit_compile=False)
    dtype = torch.float16
    sequence_length = int(args.sequence_length)
    physical_tokens = BATCH_SIZE * sequence_length
    model_dir = args.model.expanduser().resolve()
    cache_root = args.cache_dir.expanduser().resolve()
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else DEFAULT_OUTPUT_ROOT
        / f"vision_s{sequence_length}_npu_profile"
    ).expanduser().resolve()
    profile_dir = (
        args.profile_dir
        if args.profile_dir is not None
        else DEFAULT_PROFILE_ROOT
        / f"vision_s{sequence_length}_npu_profile"
    ).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"output directory already exists and is non-empty: {output_dir}"
        )
    if (
        not args.prepare_cache_only
        and profile_dir.exists()
        and any(profile_dir.iterdir())
    ):
        raise RuntimeError(
            f"profile directory already exists and is non-empty: {profile_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.prepare_cache_only:
        shutil.rmtree(profile_dir, ignore_errors=True)
        profile_dir.mkdir(parents=True, exist_ok=True)

    synchronize(device)
    setup_started = time.perf_counter()
    model = LocalPaddleOCRVLForConditionalGeneration.from_pretrained(
        model_dir,
        dtype=dtype,
        device=device,
    )
    hidden_size = int(model.config.vision_config.hidden_size)
    head_dim = hidden_size // int(
        model.config.vision_config.num_attention_heads
    )
    exact_cache_dir = vision_cache_dir_for_bucket(
        cache_root,
        bucket=sequence_length,
        dtype=dtype,
        device=device,
        model_dir=model_dir,
        attention_impl="prompt_flash_attention",
        head_dim=head_dim,
    )
    cache_existed_before_run = _cache_populated(exact_cache_dir)
    if not cache_existed_before_run and not args.allow_compile_if_missing:
        raise RuntimeError(
            f"the exact production B1xS{sequence_length} cache is absent; "
            "refusing to compile "
            f"without --allow-compile-if-missing: {exact_cache_dir}"
        )
    runtime = VisionPrefillRuntime(
        model,
        backend="torchair",
        buckets=(sequence_length,),
        cache_root=cache_root,
        device=device,
        dtype=dtype,
        model_dir=model_dir,
        attention_impl="prompt_flash_attention",
        padding="bucket",
    )
    run = runtime.compiled[sequence_length]
    setup_s = time.perf_counter() - setup_started
    cache_populated_after_setup = _cache_populated(exact_cache_dir)
    if not cache_populated_after_setup:
        raise RuntimeError(
            "VisionPrefillRuntime returned without populating its exact cache: "
            f"{exact_cache_dir}"
        )
    if args.prepare_cache_only:
        preparation = {
            "schema_version": 1,
            "purpose": (
                f"prepare the exact B1xS{sequence_length} compiled PromptFA "
                "vision cache outside the profiler run"
            ),
            "shape": {
                "batch_size": BATCH_SIZE,
                "sequence_length": sequence_length,
                "physical_tokens_per_replay": physical_tokens,
                "hidden_size": hidden_size,
                "head_dim": head_dim,
            },
            "cache_dir": str(exact_cache_dir),
            "cache_existed_before_run": cache_existed_before_run,
            "compile_was_permitted": bool(args.allow_compile_if_missing),
            "cache_populated_after_setup": cache_populated_after_setup,
            "setup_s": setup_s,
            "runtime_metadata": runtime.metadata,
            "environment": _environment(device),
        }
        preparation_path = output_dir / "cache_preparation.json"
        preparation_path.write_text(
            json.dumps(preparation, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "cache_preparation": str(preparation_path),
                    "cache_dir": str(exact_cache_dir),
                    "cache_existed_before_run": cache_existed_before_run,
                    "setup_s": setup_s,
                },
                indent=2,
            ),
            flush=True,
        )
        return

    inputs = (
        torch.zeros(
            (BATCH_SIZE, sequence_length, hidden_size),
            device=device,
            dtype=dtype,
        ),
        torch.ones(
            (BATCH_SIZE, sequence_length, head_dim),
            device=device,
            dtype=torch.float32,
        ),
        torch.zeros(
            (BATCH_SIZE, sequence_length, head_dim),
            device=device,
            dtype=torch.float32,
        ),
        torch.zeros(
            (
                BATCH_SIZE,
                1,
                sequence_length,
                sequence_length,
            ),
            device=device,
            dtype=torch.bool,
        ),
    )
    for _ in range(args.warmup):
        warm_output = run(*inputs)
        if tuple(warm_output.shape[:2]) != (
            BATCH_SIZE,
            sequence_length,
        ):
            raise RuntimeError(
                "cached graph returned the wrong warmup shape: "
                f"{tuple(warm_output.shape)}"
            )
    synchronize(device)

    pre_profile, output = _run_measured(
        run,
        inputs,
        device=device,
        repeats=args.control_repeats,
        physical_tokens=physical_tokens,
    )

    schedule = npu_prof.schedule(
        wait=0,
        warmup=args.profile_warmup_steps,
        active=args.profile_steps,
        repeat=1,
    )
    profiler_step_ms: list[float] = []
    profile_warmup_device_ms: list[float] = []
    profile_warmup_wall_ms: list[float] = []
    profiled_device_ms: list[float] = []
    profiled_wall_ms: list[float] = []
    synchronize(device)
    profile_wall_started = time.perf_counter()
    with npu_prof.profile(
        activities=[
            npu_prof.ProfilerActivity.CPU,
            npu_prof.ProfilerActivity.NPU,
        ],
        schedule=schedule,
        experimental_config=_profiler_config(),
        on_trace_ready=npu_prof.tensorboard_trace_handler(
            str(profile_dir),
            analyse_flag=True,
        ),
        record_shapes=True,
        profile_memory=False,
        with_stack=True,
        with_modules=True,
        with_flops=False,
    ) as profiler:
        profiled_output: torch.Tensor | None = None
        total_profile_loop_steps = (
            args.profile_warmup_steps + args.profile_steps
        )
        for step in range(total_profile_loop_steps):
            is_warmup = step < args.profile_warmup_steps
            phase = "warmup" if is_warmup else "active"
            phase_step = (
                step
                if is_warmup
                else step - args.profile_warmup_steps
            )
            profile_label = (
                f"paddleocr_vl.vision_prefill.B1.S{sequence_length}."
                f"{phase}.step{phase_step}"
            )
            with torch.profiler.record_function(profile_label):
                timeline = DeviceTimeline(device)
                wall_started = time.perf_counter()
                profiled_output = timeline.measure(
                    "graph_replay",
                    lambda: run(*inputs),
                )
                spans = timeline.resolve_spans()
                wall_ms = (
                    time.perf_counter() - wall_started
                ) * 1000.0
                device_ms = (
                    float(spans["graph_replay"]["seconds"]) * 1000.0
                )
                if is_warmup:
                    profile_warmup_wall_ms.append(wall_ms)
                    profile_warmup_device_ms.append(device_ms)
                else:
                    profiled_wall_ms.append(wall_ms)
                    profiled_device_ms.append(device_ms)
            step_started = time.perf_counter()
            profiler.step()
            profiler_step_ms.append(
                (time.perf_counter() - step_started) * 1000.0
            )
    synchronize(device)
    profile_context_wall_s = time.perf_counter() - profile_wall_started
    if profiled_output is None:
        raise RuntimeError("profiled measurement produced no output")
    profiled = _measurement_summary(
        profiled_device_ms,
        profiled_wall_ms,
        physical_tokens=physical_tokens,
    )

    post_profile, output = _run_measured(
        run,
        inputs,
        device=device,
        repeats=args.control_repeats,
        physical_tokens=physical_tokens,
    )
    if tuple(output.shape[:2]) != (BATCH_SIZE, sequence_length):
        raise RuntimeError(
            "cached graph returned the wrong output shape: "
            f"{tuple(output.shape)}"
        )

    pre_device = pre_profile["device_event"]
    post_device = post_profile["device_event"]
    profiled_device = profiled["device_event"]
    control_center_device_ms = (
        float(pre_device["median_ms"])
        + float(post_device["median_ms"])
    ) / 2.0
    control_center_tok_s = physical_tokens / (
        control_center_device_ms / 1000.0
    )
    slowdown = _ratio(
        float(profiled_device["median_ms"]),
        control_center_device_ms,
    )

    summary = {
        "schema_version": 1,
        "purpose": (
            f"B1xS{sequence_length} production compiled PromptFA "
            "vision-prefill NPU "
            "bottleneck profile"
        ),
        "shape": {
            "batch_size": BATCH_SIZE,
            "sequence_length": sequence_length,
            "physical_tokens_per_replay": physical_tokens,
            "hidden_size": hidden_size,
            "head_dim": head_dim,
        },
        "boundary": (
            f"VisionPrefillRuntime.compiled[{sequence_length}]: vision "
            "encoder layers plus "
            "post LayerNorm; patch embedding, projector, text prefill, and "
            "decode excluded"
        ),
        "attention": "prompt_flash_attention",
        "backend": "torchair.inference.cache_compile",
        "cache_only": cache_existed_before_run,
        "cache_dir": str(exact_cache_dir),
        "cache_existed_before_run": cache_existed_before_run,
        "compile_was_permitted": bool(args.allow_compile_if_missing),
        "cache_populated_after_setup": cache_populated_after_setup,
        "setup_s": setup_s,
        "warmup_steps_outside_profiler": args.warmup,
        "control_repeats_each_side": args.control_repeats,
        "profile_active_steps": args.profile_steps,
        "profiler_step_contract": (
            "one profiler.step after exactly one synchronized compiled "
            f"B1xS{sequence_length} graph replay"
        ),
        "profiler": {
            "activities": ["CPU", "NPU"],
            "level": "Level1",
            "aic_metric": "PipeUtilization",
            "record_shapes": True,
            "profile_memory": False,
            "with_stack": True,
            "with_modules": True,
            "profile_dir": str(profile_dir),
            "context_wall_s": profile_context_wall_s,
            "scheduled_warmup_steps": args.profile_warmup_steps,
            "scheduled_warmup_measurement": _measurement_summary(
                profile_warmup_device_ms,
                profile_warmup_wall_ms,
                physical_tokens=physical_tokens,
            ),
            "profiler_step_ms": _duration_summary(profiler_step_ms),
            "throughput_measurement": False,
        },
        "measurements": {
            "pre_profile_control": pre_profile,
            "profiled_diagnostic": profiled,
            "post_profile_control": post_profile,
            "control_center_device_median_ms": control_center_device_ms,
            "control_center_physical_tokens_per_s": control_center_tok_s,
            "profiled_device_slowdown_ratio": slowdown,
            "profiled_device_slowdown_pct": (
                (slowdown - 1.0) * 100.0
                if slowdown is not None
                else None
            ),
            "post_vs_pre_device_median_ratio": _ratio(
                float(post_device["median_ms"]),
                float(pre_device["median_ms"]),
            ),
        },
        "output": {
            "shape": [int(value) for value in output.shape],
            "finite": bool(torch.isfinite(output.float()).all().item()),
        },
        "runtime_metadata": runtime.metadata,
        "environment": _environment(device),
    }
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    parser_result = _run_parser(
        profile_dir,
        topn=args.parser_topn,
    )
    parsed_json = output_dir / "parsed_profile_summary.json"
    parsed_markdown = output_dir / "parsed_profile_summary.md"
    shutil.copyfile(
        profile_dir / "parsed_profile_summary.json",
        parsed_json,
    )
    shutil.copyfile(
        profile_dir / "parsed_profile_summary.md",
        parsed_markdown,
    )
    parser_result.update(
        {
            "copied_parsed_json": str(parsed_json),
            "copied_parsed_markdown": str(parsed_markdown),
        }
    )
    summary["parsed_profile"] = parser_result
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "profile_dir": str(profile_dir),
                "pre_profile_physical_tok_s": pre_device[
                    "physical_tokens_per_s_median"
                ],
                "profiled_physical_tok_s": profiled_device[
                    "physical_tokens_per_s_median"
                ],
                "post_profile_physical_tok_s": post_device[
                    "physical_tokens_per_s_median"
                ],
                "control_center_physical_tok_s": control_center_tok_s,
                "profiled_device_slowdown_pct": summary["measurements"][
                    "profiled_device_slowdown_pct"
                ],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
