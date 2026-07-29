#!/usr/bin/env python3
"""Profile the production compiled PromptFA vision graph at B1 x S512.

The experiment deliberately has one shape and one graph boundary:

    VisionPrefillRuntime.compiled[512](
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
SEQUENCE_LENGTH = 512
PHYSICAL_TOKENS = BATCH_SIZE * SEQUENCE_LENGTH
DEFAULT_CACHE_ROOT = (
    REPO_ROOT / ".runtime_cache/09_persistent_page_engine_vision_torchair"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "tmp/09_persistent_page_engine/vision_lab"
    / "vision_s512_npu_profile"
)
DEFAULT_PROFILE_DIR = (
    REPO_ROOT
    / ".runtime_cache/09_persistent_page_engine_profiles"
    / "vision_s512_npu_profile"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--profile-dir", type=Path, default=DEFAULT_PROFILE_DIR)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--control-repeats", type=int, default=20)
    parser.add_argument("--profile-steps", type=int, default=5)
    parser.add_argument("--parser-topn", type=int, default=30)
    args = parser.parse_args(argv)
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.control_repeats <= 0:
        parser.error("--control-repeats must be positive")
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


def _replay_summary(samples_ms: Sequence[float]) -> dict[str, Any]:
    summary = _duration_summary(samples_ms)
    mean_ms = float(summary["mean_ms"])
    median_ms = float(summary["median_ms"])
    summary.update(
        {
            "physical_tokens_per_s_mean": PHYSICAL_TOKENS
            / (mean_ms / 1000.0),
            "physical_tokens_per_s_median": PHYSICAL_TOKENS
            / (median_ms / 1000.0),
        }
    )
    return summary


def _measurement_summary(
    device_ms: Sequence[float],
    synchronized_wall_ms: Sequence[float],
) -> dict[str, Any]:
    return {
        "device_event": _replay_summary(device_ms),
        "synchronized_host_wall": _replay_summary(synchronized_wall_ms),
    }


def _run_measured(
    run: Callable[..., torch.Tensor],
    inputs: tuple[torch.Tensor, ...],
    *,
    device: torch.device,
    repeats: int,
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
    return _measurement_summary(device_ms, synchronized_wall_ms), output


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
        raise RuntimeError("the S512 profiler requires an NPU")
    torch.npu.set_compile_mode(jit_compile=False)
    dtype = torch.float16
    model_dir = args.model.expanduser().resolve()
    cache_root = args.cache_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    profile_dir = args.profile_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"output directory already exists and is non-empty: {output_dir}"
        )
    if profile_dir.exists() and any(profile_dir.iterdir()):
        raise RuntimeError(
            f"profile directory already exists and is non-empty: {profile_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
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
        bucket=SEQUENCE_LENGTH,
        dtype=dtype,
        device=device,
        model_dir=model_dir,
        attention_impl="prompt_flash_attention",
        head_dim=head_dim,
    )
    if not _cache_populated(exact_cache_dir):
        raise RuntimeError(
            "the exact production B1xS512 cache is absent; refusing to compile "
            f"during a profiling run: {exact_cache_dir}"
        )
    runtime = VisionPrefillRuntime(
        model,
        backend="torchair",
        buckets=(SEQUENCE_LENGTH,),
        cache_root=cache_root,
        device=device,
        dtype=dtype,
        model_dir=model_dir,
        attention_impl="prompt_flash_attention",
        padding="bucket",
    )
    run = runtime.compiled[SEQUENCE_LENGTH]
    setup_s = time.perf_counter() - setup_started

    inputs = (
        torch.zeros(
            (BATCH_SIZE, SEQUENCE_LENGTH, hidden_size),
            device=device,
            dtype=dtype,
        ),
        torch.ones(
            (BATCH_SIZE, SEQUENCE_LENGTH, head_dim),
            device=device,
            dtype=torch.float32,
        ),
        torch.zeros(
            (BATCH_SIZE, SEQUENCE_LENGTH, head_dim),
            device=device,
            dtype=torch.float32,
        ),
        torch.zeros(
            (
                BATCH_SIZE,
                1,
                SEQUENCE_LENGTH,
                SEQUENCE_LENGTH,
            ),
            device=device,
            dtype=torch.bool,
        ),
    )
    for _ in range(args.warmup):
        warm_output = run(*inputs)
        if tuple(warm_output.shape[:2]) != (
            BATCH_SIZE,
            SEQUENCE_LENGTH,
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
    )

    schedule = npu_prof.schedule(
        wait=0,
        warmup=0,
        active=args.profile_steps,
        repeat=1,
    )
    profiler_step_ms: list[float] = []
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
        profiled_device_ms: list[float] = []
        profiled_wall_ms: list[float] = []
        profiled_output: torch.Tensor | None = None
        for step in range(args.profile_steps):
            with torch.profiler.record_function(
                f"paddleocr_vl.vision_prefill.B1.S512.step{step}"
            ):
                timeline = DeviceTimeline(device)
                wall_started = time.perf_counter()
                profiled_output = timeline.measure(
                    "graph_replay",
                    lambda: run(*inputs),
                )
                spans = timeline.resolve_spans()
                profiled_wall_ms.append(
                    (time.perf_counter() - wall_started) * 1000.0
                )
                profiled_device_ms.append(
                    float(spans["graph_replay"]["seconds"]) * 1000.0
                )
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
    )

    post_profile, output = _run_measured(
        run,
        inputs,
        device=device,
        repeats=args.control_repeats,
    )
    if tuple(output.shape[:2]) != (BATCH_SIZE, SEQUENCE_LENGTH):
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
    control_center_tok_s = PHYSICAL_TOKENS / (
        control_center_device_ms / 1000.0
    )
    slowdown = _ratio(
        float(profiled_device["median_ms"]),
        control_center_device_ms,
    )

    summary = {
        "schema_version": 1,
        "purpose": (
            "B1xS512 production compiled PromptFA vision-prefill NPU "
            "bottleneck profile"
        ),
        "shape": {
            "batch_size": BATCH_SIZE,
            "sequence_length": SEQUENCE_LENGTH,
            "physical_tokens_per_replay": PHYSICAL_TOKENS,
            "hidden_size": hidden_size,
            "head_dim": head_dim,
        },
        "boundary": (
            "VisionPrefillRuntime.compiled[512]: vision encoder layers plus "
            "post LayerNorm; patch embedding, projector, text prefill, and "
            "decode excluded"
        ),
        "attention": "prompt_flash_attention",
        "backend": "torchair.inference.cache_compile",
        "cache_only": True,
        "cache_dir": str(exact_cache_dir),
        "cache_existed_before_run": True,
        "setup_s": setup_s,
        "warmup_steps_outside_profiler": args.warmup,
        "control_repeats_each_side": args.control_repeats,
        "profile_active_steps": args.profile_steps,
        "profiler_step_contract": (
            "one profiler.step after exactly one synchronized compiled "
            "B1xS512 graph replay"
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
