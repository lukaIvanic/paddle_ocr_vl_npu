#!/usr/bin/env python3
"""Profile one warmed MinerU vision-prefill shape with torch_npu.profiler.

The profiled boundary is the full ``model.get_image_features`` call: patch
embedding, position preparation, the eager or TorchAir transformer blocks, and
the patch merger. Image decode/resize, processor work, H2D, cache restore or
compile, and model setup remain outside the profile window.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

from local_modeling_mineru import LocalMinerU2_5ForConditionalGeneration
from run_transformers_recognition_smoke import configure_npu, synchronize
from vision_prefill_compile import (
    DEFAULT_VISION_BUCKETS,
    MinerUVisionPrefillRuntime,
    parse_vision_buckets,
    select_vision_bucket,
)
from vision_prefill_lab import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DATASET_JSON,
    DEFAULT_IMAGES_DIR,
    DEFAULT_MODEL,
    _dataset_images,
    _prepare_pages,
    _summary,
)


PROFILE_METRICS = ("pipe", "memory", "l2", "memory_access")
DEFAULT_PROFILE_ROOT = Path(
    ".runtime_cache/11_mineru_2_5_pro_inference/vision_prefill_profiles"
)
DEFAULT_OUTPUT_DIR = Path(
    "tmp/11_mineru_2_5_pro_inference/vision_prefill_profile/layout_1036"
)


def _csv_metrics(value: str) -> tuple[str, ...]:
    values = tuple(piece.strip() for piece in value.split(",") if piece.strip())
    unknown = sorted(set(values) - set(PROFILE_METRICS))
    if not values or unknown:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated values from {PROFILE_METRICS}; unknown={unknown}"
        )
    return tuple(dict.fromkeys(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset-json", type=Path, default=DEFAULT_DATASET_JSON)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--image", type=Path, action="append", default=[])
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--layout-size", type=int, nargs=2, default=(1036, 1036))
    parser.add_argument("--execution", choices=("eager", "torchair"), default="torchair")
    parser.add_argument(
        "--buckets",
        default=",".join(str(value) for value in DEFAULT_VISION_BUCKETS),
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--metrics", type=_csv_metrics, default=("pipe", "memory"))
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--baseline-steps", type=int, default=10)
    parser.add_argument("--profile-steps", type=int, default=3)
    parser.add_argument("--profile-root", type=Path, default=DEFAULT_PROFILE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--parser-topn", type=int, default=30)
    args = parser.parse_args()
    args.buckets = parse_vision_buckets(args.buckets)
    if args.offset < 0:
        parser.error("--offset must be non-negative")
    if args.warmup_steps < 0 or args.baseline_steps <= 0 or args.profile_steps <= 0:
        parser.error("warmup must be non-negative; baseline/profile steps must be positive")
    if any(value <= 0 for value in args.layout_size):
        parser.error("--layout-size values must be positive")
    return args


def npu_profiler_config(metric: str):
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
    )


def _forward(
    model: LocalMinerU2_5ForConditionalGeneration,
    page: dict[str, Any],
) -> torch.Tensor:
    return model.get_image_features(page["pixel_values"], page["grid"])


def _measure_forward(
    fn: Callable[[], torch.Tensor],
    *,
    steps: int,
) -> dict[str, Any]:
    samples: list[float] = []
    output = None
    for _ in range(steps):
        synchronize()
        started = time.perf_counter()
        output = fn()
        synchronize()
        samples.append(time.perf_counter() - started)
    if output is None:
        raise RuntimeError("measurement produced no output")
    return {
        "samples_s": samples,
        "summary_s": _summary(samples),
        "output_shape": list(output.shape),
        "nonfinite": int((~torch.isfinite(output)).sum().item()),
    }


def _run_parser(
    profile_dir: Path,
    output_dir: Path,
    *,
    topn: int,
) -> dict[str, Any]:
    parser = Path(__file__).resolve().parent / "parse_npu_profile.py"
    output_json = output_dir / "parsed_profile_summary.json"
    output_md = output_dir / "parsed_profile_summary.md"
    command = [
        sys.executable,
        str(parser),
        "--profile-dir",
        str(profile_dir),
        "--topn",
        str(topn),
        "--out-json",
        str(output_json),
        "--out-md",
        str(output_md),
        "--skip-trace",
    ]
    print(f"[parser] {' '.join(command)}", flush=True)
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return {
        "command": command,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "json": str(output_json),
        "markdown": str(output_md),
    }


def _profile_metric(
    *,
    metric: str,
    model: LocalMinerU2_5ForConditionalGeneration,
    page: dict[str, Any],
    baseline_steps: int,
    profile_steps: int,
    profile_root: Path,
    output_dir: Path,
    parser_topn: int,
) -> dict[str, Any]:
    import torch_npu.profiler as npu_prof

    fn = lambda: _forward(model, page)
    print(f"[{metric}] unprofiled baseline before capture", flush=True)
    before = _measure_forward(fn, steps=baseline_steps)

    metric_profile_dir = profile_root / metric
    metric_output_dir = output_dir / metric
    metric_profile_dir.mkdir(parents=True, exist_ok=True)
    metric_output_dir.mkdir(parents=True, exist_ok=True)
    schedule = npu_prof.schedule(wait=0, warmup=0, active=profile_steps, repeat=1)
    forward_times: list[float] = []
    step_times: list[float] = []
    print(f"[{metric}] profiling {profile_steps} full vision forwards", flush=True)
    synchronize()
    context_started = time.perf_counter()
    active_started = None
    active_ended = None
    with npu_prof.profile(
        activities=[npu_prof.ProfilerActivity.CPU, npu_prof.ProfilerActivity.NPU],
        schedule=schedule,
        experimental_config=npu_profiler_config(metric),
        on_trace_ready=npu_prof.tensorboard_trace_handler(
            str(metric_profile_dir),
            analyse_flag=True,
        ),
        record_shapes=True,
        profile_memory=False,
        with_stack=True,
    ) as profiler:
        active_started = time.perf_counter()
        for step in range(profile_steps):
            with torch.profiler.record_function(
                f"mineru.vision_prefill.{metric}.step{step}"
            ):
                synchronize()
                started = time.perf_counter()
                output = fn()
                synchronize()
                forward_times.append(time.perf_counter() - started)
            step_started = time.perf_counter()
            profiler.step()
            step_times.append(time.perf_counter() - step_started)
        active_ended = time.perf_counter()
    synchronize()
    context_wall_s = time.perf_counter() - context_started
    if int((~torch.isfinite(output)).sum().item()):
        raise RuntimeError(f"{metric} profile produced nonfinite output")

    print(f"[{metric}] unprofiled baseline after capture", flush=True)
    after = _measure_forward(fn, steps=baseline_steps)
    parser = _run_parser(
        metric_profile_dir,
        metric_output_dir,
        topn=parser_topn,
    )
    unprofiled_reference_s = statistics.mean(
        [before["summary_s"]["mean"], after["summary_s"]["mean"]]
    )
    profiled_forward_mean_s = statistics.mean(forward_times)
    active_loop_s = float((active_ended or context_started) - (active_started or context_started))
    return {
        "metric": metric,
        "profile_dir": str(metric_profile_dir),
        "unprofiled_before": before,
        "unprofiled_after": after,
        "unprofiled_reference_mean_s": unprofiled_reference_s,
        "profiled_forward_samples_s": forward_times,
        "profiled_forward_summary_s": _summary(forward_times),
        "profiled_forward_slowdown": profiled_forward_mean_s / unprofiled_reference_s,
        "profiler_step_samples_s": step_times,
        "profiler_step_summary_s": _summary(step_times),
        "profile_context_wall_s": context_wall_s,
        "profile_active_loop_wall_s": active_loop_s,
        "profile_context_non_active_s": max(0.0, context_wall_s - active_loop_s),
        "parser": parser,
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    print("[setup] configure NPU and load processor/model", flush=True)
    configure_npu()
    import torch_npu  # noqa: F401
    from transformers import AutoProcessor

    model_dir = args.model.expanduser().resolve()
    setup_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        model_dir,
        use_fast=True,
        local_files_only=True,
    )
    model = LocalMinerU2_5ForConditionalGeneration.from_pretrained(
        model_dir,
        dtype=torch.float16,
        device="npu:0",
    )
    model.set_vision_attention_impl("prompt_flash_attention")
    synchronize()
    setup_s = time.perf_counter() - setup_started
    print(f"[setup] complete in {setup_s:.3f}s", flush=True)

    # The corpus helper takes --limit from the namespace. The profiler always
    # selects exactly one real page.
    args.limit = 1
    pages, preparation = _prepare_pages(
        processor,
        _dataset_images(args),
        layout_size=tuple(args.layout_size),
        device=model.device,
        dtype=model.dtype,
    )
    page = pages[0]
    real_tokens = int(page["real_tokens"])
    bucket = select_vision_bucket(real_tokens, args.buckets)
    if args.execution == "torchair" and bucket is None:
        raise ValueError(
            f"real tokens {real_tokens} exceed largest compiled bucket {args.buckets[-1]}"
        )
    runtime = None
    if args.execution == "torchair":
        runtime = MinerUVisionPrefillRuntime(
            model.visual,
            buckets=args.buckets,
            cache_root=args.cache_dir,
            model_dir=model_dir,
            device=model.device,
            dtype=model.dtype,
        )
    model.set_vision_prefill_runtime(runtime)
    print(
        f"[shape] execution={args.execution} page={page['name']} "
        f"layout={tuple(args.layout_size)} real_tokens={real_tokens} bucket={bucket}",
        flush=True,
    )

    print(f"[warmup] {args.warmup_steps} steps", flush=True)
    for _ in range(args.warmup_steps):
        synchronize()
        _forward(model, page)
        synchronize()
    if runtime is not None:
        runtime.route_counts.clear()
        runtime.real_tokens = 0
        runtime.physical_tokens = 0

    output_dir = args.output_dir.expanduser().resolve()
    profile_root = args.profile_root.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_root.mkdir(parents=True, exist_ok=True)
    metric_results = {}
    for metric in args.metrics:
        metric_results[metric] = _profile_metric(
            metric=metric,
            model=model,
            page=page,
            baseline_steps=args.baseline_steps,
            profile_steps=args.profile_steps,
            profile_root=profile_root,
            output_dir=output_dir,
            parser_topn=args.parser_topn,
        )
        print(
            f"[{metric}] profiler slowdown="
            f"{metric_results[metric]['profiled_forward_slowdown']:.3f}x",
            flush=True,
        )

    result = {
        "schema_version": 1,
        "kind": "mineru_vision_prefill_torch_npu_profile",
        "profile_scope": "full get_image_features: patch + positions + transformer blocks + merger",
        "excluded_scope": "model setup + image decode/resize + processor + H2D + graph cache restore/compile",
        "device": "Ascend NPU",
        "dtype": "fp16",
        "attention": "prompt_flash_attention",
        "execution": args.execution,
        "setup_s": setup_s,
        "model": str(model_dir),
        "page": {
            key: value
            for key, value in page.items()
            if key not in {"pixel_values", "grid"}
        },
        "layout_size_wh": list(args.layout_size),
        "real_tokens": real_tokens,
        "physical_tokens": bucket if bucket is not None else real_tokens,
        "preparation": preparation,
        "warmup_steps": args.warmup_steps,
        "baseline_steps": args.baseline_steps,
        "profile_steps": args.profile_steps,
        "metrics": metric_results,
        "runtime": runtime.metadata() if runtime is not None else None,
    }
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    headline = {
        metric: {
            "unprofiled_reference_mean_ms": value["unprofiled_reference_mean_s"] * 1000.0,
            "profiled_forward_mean_ms": value["profiled_forward_summary_s"]["mean"] * 1000.0,
            "profiled_forward_slowdown": value["profiled_forward_slowdown"],
            "profiler_step_mean_ms": value["profiler_step_summary_s"]["mean"] * 1000.0,
        }
        for metric, value in metric_results.items()
    }
    print(json.dumps({"headline": headline, "result": str(result_path)}, indent=2))


if __name__ == "__main__":
    main()
