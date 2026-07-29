#!/usr/bin/env python3
"""Full-stack PaddleOCR-VL vision MatMul format/alignment laboratory.

The measured boundary preserves all 27 vision encoder layers, including both
LayerNorms, residuals, Q/K/V/output projections, FC1/GELU/FC2, and the final
post-LayerNorm.  Only attention itself is replaced by a cheap token-local
surrogate:

    out_proj((q_proj(x) + k_proj(x) + v_proj(x)) / 3)

This keeps every production Linear weight and its surrounding operations in the
graph while removing attention as a competing bottleneck.  It is a performance
laboratory, not an OCR-correctness path.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
REPO_ROOT = EXPERIMENT_ROOT.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))
sys.path.insert(0, str(HERE))

from paddleocr_vl.model.compile_utils import (  # noqa: E402
    TORCHAIR_EXECUTION_MODE,
    cache_key_part,
    import_torchair,
    short_file_hash,
    torch_npu_version_label,
    torchair_version_label,
)
from paddleocr_vl.model.modeling import (  # noqa: E402
    LocalPaddleOCRVLForConditionalGeneration,
)
from paddleocr_vl.model.vision_prefill import _activation  # noqa: E402
from utils.timing import DeviceTimeline, synchronize  # noqa: E402
from vision_lab import DEFAULT_MODEL, _environment  # noqa: E402


FRACTAL_NZ = 29
DEFAULT_CACHE_ROOT = (
    REPO_ROOT / ".runtime_cache/09_persistent_page_engine_vision_matmul_lab"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "tmp/09_persistent_page_engine/vision_matmul_lab"
)
DEFAULT_PROFILE_ROOT = (
    REPO_ROOT
    / ".runtime_cache/09_persistent_page_engine_vision_matmul_profiles"
)
SEQUENCE_LENGTHS = (512, 2048)
INTERMEDIATE_SIZES = (4304, 4352)
WEIGHT_FORMATS = ("native", "fractal_nz")
EXECUTIONS = ("raw_eager", "torchair")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence-length",
        type=int,
        choices=SEQUENCE_LENGTHS,
        required=True,
    )
    parser.add_argument(
        "--intermediate-size",
        type=int,
        choices=INTERMEDIATE_SIZES,
        required=True,
    )
    parser.add_argument(
        "--weight-format",
        choices=WEIGHT_FORMATS,
        required=True,
    )
    parser.add_argument(
        "--execution",
        choices=EXECUTIONS,
        default="torchair",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument(
        "--allow-compile-if-missing",
        action="store_true",
        help="Permit creation of a missing TorchAir graph cache.",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument(
        "--calls-per-sample",
        type=int,
        default=5,
        help=(
            "Full 27-layer replays inside one NPU-event sample. Values above "
            "one amortize host synchronization and launch overhead."
        ),
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-warmup-steps", type=int, default=1)
    parser.add_argument("--profile-steps", type=int, default=3)
    parser.add_argument("--parser-topn", type=int, default=200)
    args = parser.parse_args(argv)
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.samples <= 0 or args.calls_per_sample <= 0:
        parser.error("--samples and --calls-per-sample must be positive")
    if args.profile_warmup_steps <= 0 or args.profile_steps <= 0:
        parser.error("profiler step counts must be positive")
    return args


class VisionMatmulLayer(nn.Module):
    """One real vision block with attention replaced by a Linear-only surrogate."""

    def __init__(self, source: nn.Module, *, intermediate_size: int):
        super().__init__()
        self.layer_norm1 = source.layer_norm1
        self.q_proj = source.self_attn.q_proj
        self.k_proj = source.self_attn.k_proj
        self.v_proj = source.self_attn.v_proj
        self.out_proj = source.self_attn.out_proj
        self.layer_norm2 = source.layer_norm2
        self.hidden_act = str(source.mlp.hidden_act)
        source_intermediate = int(source.mlp.fc1.out_features)
        if intermediate_size == source_intermediate:
            self.fc1 = source.mlp.fc1
            self.fc2 = source.mlp.fc2
        else:
            self.fc1, self.fc2 = _zero_extended_mlp(
                source.mlp.fc1,
                source.mlp.fc2,
                target_intermediate=intermediate_size,
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        attention_input = self.layer_norm1(hidden_states)
        q = self.q_proj(attention_input)
        k = self.k_proj(attention_input)
        v = self.v_proj(attention_input)
        surrogate = (q + k + v) * (1.0 / 3.0)
        hidden_states = hidden_states + self.out_proj(surrogate)
        mlp_input = self.layer_norm2(hidden_states)
        hidden_states = hidden_states + self.fc2(
            _activation(self.hidden_act, self.fc1(mlp_input))
        )
        return hidden_states


class VisionMatmulStack(nn.Module):
    """All 27 real vision blocks plus the production post-LayerNorm."""

    def __init__(self, model: nn.Module, *, intermediate_size: int):
        super().__init__()
        transformer = model.visual.vision_model
        source_layers = transformer.encoder.layers
        self.layers = nn.ModuleList(
            [
                VisionMatmulLayer(
                    layer,
                    intermediate_size=intermediate_size,
                )
                for layer in source_layers
            ]
        )
        self.post_layernorm = transformer.post_layernorm

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.post_layernorm(hidden_states)


def _zero_extended_mlp(
    source_fc1: nn.Linear,
    source_fc2: nn.Linear,
    *,
    target_intermediate: int,
) -> tuple[nn.Linear, nn.Linear]:
    source_intermediate = int(source_fc1.out_features)
    hidden_size = int(source_fc1.in_features)
    if target_intermediate <= source_intermediate:
        raise ValueError(
            "the alignment experiment only supports zero-extension: "
            f"{source_intermediate} -> {target_intermediate}"
        )
    if int(source_fc2.in_features) != source_intermediate:
        raise ValueError("source FC1/FC2 intermediate dimensions disagree")
    if int(source_fc2.out_features) != hidden_size:
        raise ValueError("source FC2 hidden dimension disagrees with FC1")
    device = source_fc1.weight.device
    dtype = source_fc1.weight.dtype
    fc1 = nn.Linear(
        hidden_size,
        target_intermediate,
        bias=source_fc1.bias is not None,
        device=device,
        dtype=dtype,
    )
    fc2 = nn.Linear(
        target_intermediate,
        hidden_size,
        bias=source_fc2.bias is not None,
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        fc1.weight.zero_()
        fc1.weight[:source_intermediate].copy_(source_fc1.weight)
        if fc1.bias is not None:
            fc1.bias.zero_()
            fc1.bias[:source_intermediate].copy_(source_fc1.bias)
        fc2.weight.zero_()
        fc2.weight[:, :source_intermediate].copy_(source_fc2.weight)
        if fc2.bias is not None:
            fc2.bias.copy_(source_fc2.bias)
    return fc1, fc2


def _linear_modules(stage: nn.Module) -> list[tuple[str, nn.Linear]]:
    modules = [
        (name, module)
        for name, module in stage.named_modules()
        if isinstance(module, nn.Linear)
    ]
    expected = len(stage.layers) * 6
    if len(modules) != expected:
        raise RuntimeError(
            f"expected {expected} Linear modules, found {len(modules)}"
        )
    return modules


def _format_histogram(
    modules: list[tuple[str, nn.Linear]],
    torch_npu: Any,
) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                str(int(torch_npu.get_npu_format(module.weight)))
                for _name, module in modules
            ).items()
        )
    )


def _prepare_weight_format(
    stage: nn.Module,
    *,
    requested: str,
    torch_npu: Any,
) -> dict[str, Any]:
    modules = _linear_modules(stage)
    before = _format_histogram(modules, torch_npu)
    metadata: dict[str, Any] = {
        "requested": requested,
        "target_format": (
            "FRACTAL_NZ" if requested == "fractal_nz" else "unchanged"
        ),
        "target_format_code": (
            FRACTAL_NZ if requested == "fractal_nz" else None
        ),
        "linear_weight_count": len(modules),
        "before_format_histogram": before,
        "converted_count": 0,
        "status": "ready",
        "failures": [],
    }
    if requested == "native":
        metadata["after_format_histogram"] = before
        metadata["all_after_are_nz"] = all(
            int(code) == FRACTAL_NZ
            for code in before
        )
        return metadata

    converted = 0
    failures: list[dict[str, Any]] = []
    for name, module in modules:
        before_code = int(torch_npu.get_npu_format(module.weight))
        if before_code == FRACTAL_NZ:
            continue
        try:
            module.weight.data = torch_npu.npu_format_cast(
                module.weight.data,
                FRACTAL_NZ,
            )
            after_code = int(torch_npu.get_npu_format(module.weight))
        except Exception as exc:
            failures.append(
                {
                    "module": name,
                    "before_format": before_code,
                    "error": repr(exc),
                }
            )
            break
        if after_code != FRACTAL_NZ:
            failures.append(
                {
                    "module": name,
                    "before_format": before_code,
                    "after_format": after_code,
                    "error": "format_cast_did_not_produce_fractal_nz",
                }
            )
            break
        converted += 1
    after = _format_histogram(modules, torch_npu)
    all_after_are_nz = all(
        int(torch_npu.get_npu_format(module.weight)) == FRACTAL_NZ
        for _name, module in modules
    )
    metadata.update(
        {
            "converted_count": converted,
            "after_format_histogram": after,
            "all_after_are_nz": all_after_are_nz,
            "failures": failures,
            "status": "ready" if all_after_are_nz else "unsupported",
        }
    )
    return metadata


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _sample_summary(values: Sequence[float]) -> dict[str, Any]:
    samples = [float(value) for value in values]
    return {
        "samples": samples,
        "mean": statistics.mean(samples),
        "median": statistics.median(samples),
        "p05": _percentile(samples, 0.05),
        "p95": _percentile(samples, 0.95),
    }


def _repeat(
    run: Callable[[torch.Tensor], torch.Tensor],
    hidden_states: torch.Tensor,
    calls: int,
) -> torch.Tensor:
    output: torch.Tensor | None = None
    for _ in range(calls):
        output = run(hidden_states)
    if output is None:
        raise AssertionError("repeat count must be positive")
    return output


def _measure(
    run: Callable[[torch.Tensor], torch.Tensor],
    hidden_states: torch.Tensor,
    *,
    device: torch.device,
    samples: int,
    calls_per_sample: int,
    sequence_length: int,
    flops_per_call: int,
) -> tuple[dict[str, Any], torch.Tensor]:
    device_block_ms: list[float] = []
    wall_block_ms: list[float] = []
    output: torch.Tensor | None = None
    for _ in range(samples):
        timeline = DeviceTimeline(device)
        wall_started = time.perf_counter()
        output = timeline.measure(
            "full_stack_replays",
            lambda: _repeat(run, hidden_states, calls_per_sample),
        )
        spans = timeline.resolve_spans()
        wall_block_ms.append((time.perf_counter() - wall_started) * 1000.0)
        device_block_ms.append(
            float(spans["full_stack_replays"]["seconds"]) * 1000.0
        )
    if output is None:
        raise RuntimeError("measurement produced no output")

    device_per_call_ms = [
        value / calls_per_sample for value in device_block_ms
    ]
    wall_per_call_ms = [
        value / calls_per_sample for value in wall_block_ms
    ]
    device = _sample_summary(device_per_call_ms)
    wall = _sample_summary(wall_per_call_ms)
    device_median_s = float(device["median"]) / 1000.0
    wall_median_s = float(wall["median"]) / 1000.0
    return (
        {
            "samples": samples,
            "calls_per_sample": calls_per_sample,
            "total_measured_full_stack_calls": samples * calls_per_sample,
            "device_event_per_call_ms": device,
            "synchronized_wall_per_call_ms": wall,
            "physical_tokens_per_s_device_median": (
                sequence_length / device_median_s
            ),
            "physical_tokens_per_s_wall_median": (
                sequence_length / wall_median_s
            ),
            "linear_tflop_per_s_device_median": (
                flops_per_call / device_median_s / 1e12
            ),
        },
        output,
    )


def _diff(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    left32 = left.float()
    right32 = right.float()
    delta = (left32 - right32).abs()
    return {
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
        "left_finite": bool(torch.isfinite(left32).all().item()),
        "right_finite": bool(torch.isfinite(right32).all().item()),
        "same_shape": tuple(left.shape) == tuple(right.shape),
    }


def _cache_dir(
    root: Path,
    *,
    sequence_length: int,
    intermediate_size: int,
    weight_format: str,
    dtype: torch.dtype,
    device: torch.device,
    model_dir: Path,
) -> Path:
    key = "_".join(
        [
            "vision_matmul_stack",
            "layers27",
            "attention_surrogate_qkvmean",
            f"seq{sequence_length}",
            f"intermediate{intermediate_size}",
            f"weights{weight_format}",
            f"dtype{cache_key_part(dtype)}",
            f"mode{cache_key_part(TORCHAIR_EXECUTION_MODE)}",
            f"model{short_file_hash(model_dir / 'config.json')}",
            f"torch{cache_key_part(torch.__version__)}",
            f"torchnpu{torch_npu_version_label(device)}",
            f"torchair{torchair_version_label(device)}",
            f"src{short_file_hash(Path(__file__).resolve())}",
        ]
    )
    return root.expanduser().resolve() / key


def _cache_populated(path: Path) -> bool:
    return path.is_dir() and any(item.is_file() for item in path.rglob("*"))


def _compile(
    stage: nn.Module,
    *,
    cache_dir: Path,
    allow_missing: bool,
    device: torch.device,
    example: torch.Tensor,
) -> tuple[Callable[[torch.Tensor], torch.Tensor], dict[str, Any]]:
    existed = _cache_populated(cache_dir)
    if not existed and not allow_missing:
        raise RuntimeError(
            "the exact graph cache is missing; pass "
            f"--allow-compile-if-missing to create it: {cache_dir}"
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    torchair, CompilerConfig = import_torchair()
    synchronize(device)
    wrapper_started = time.perf_counter()
    compiled = torchair.inference.cache_compile(
        stage.forward,
        config=CompilerConfig(),
        dynamic=False,
        cache_dir=str(cache_dir),
        ge_cache=True,
    )
    synchronize(device)
    wrapper_s = time.perf_counter() - wrapper_started
    synchronize(device)
    first_started = time.perf_counter()
    first_output = compiled(example)
    synchronize(device)
    first_call_s = time.perf_counter() - first_started
    del first_output
    if not _cache_populated(cache_dir):
        raise RuntimeError(f"TorchAir did not populate cache: {cache_dir}")
    return compiled, {
        "api": "torchair.inference.cache_compile",
        "dynamic": False,
        "fullgraph": True,
        "ge_cache": True,
        "cache_dir": str(cache_dir),
        "cache_existed_before": existed,
        "compile_was_permitted": allow_missing,
        "wrapper_s": wrapper_s,
        "first_call_s": first_call_s,
    }


def _profiler_config() -> Any:
    import torch_npu.profiler as npu_prof

    return npu_prof._ExperimentalConfig(
        profiler_level=npu_prof.ProfilerLevel.Level1,
        aic_metrics=npu_prof.AiCMetrics.PipeUtilization,
        l2_cache=False,
        export_type=npu_prof.ExportType.Text,
        data_simplification=False,
    )


def _profile(
    run: Callable[[torch.Tensor], torch.Tensor],
    hidden_states: torch.Tensor,
    *,
    profile_dir: Path,
    warmup_steps: int,
    active_steps: int,
    label: str,
) -> dict[str, Any]:
    import torch_npu.profiler as npu_prof

    if profile_dir.exists() and any(profile_dir.iterdir()):
        raise RuntimeError(
            f"profile directory already exists and is non-empty: {profile_dir}"
        )
    shutil.rmtree(profile_dir, ignore_errors=True)
    profile_dir.mkdir(parents=True, exist_ok=True)
    schedule = npu_prof.schedule(
        wait=0,
        warmup=warmup_steps,
        active=active_steps,
        repeat=1,
    )
    context_started = time.perf_counter()
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
        for step in range(warmup_steps + active_steps):
            phase = "warmup" if step < warmup_steps else "active"
            with torch.profiler.record_function(f"{label}.{phase}.step{step}"):
                output = run(hidden_states)
                torch.npu.synchronize()
            profiler.step()
    torch.npu.synchronize()
    del output
    return {
        "profile_dir": str(profile_dir),
        "scheduled_warmup_steps": warmup_steps,
        "active_steps": active_steps,
        "context_wall_s": time.perf_counter() - context_started,
        "throughput_measurement": False,
    }


def _parse_profile(
    profile_dir: Path,
    output_dir: Path,
    *,
    topn: int,
) -> dict[str, Any]:
    parser = (
        REPO_ROOT
        / "07_vision_prefill_optimization"
        / "parse_static_visual_encoder_profile.py"
    )
    command = [
        sys.executable,
        str(parser),
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
    source_json = profile_dir / "parsed_profile_summary.json"
    source_md = profile_dir / "parsed_profile_summary.md"
    destination_json = output_dir / "parsed_profile_summary.json"
    destination_md = output_dir / "parsed_profile_summary.md"
    shutil.copyfile(source_json, destination_json)
    shutil.copyfile(source_md, destination_md)
    parsed = json.loads(destination_json.read_text(encoding="utf-8"))
    dispatch: Counter[str] = Counter()
    dispatch_duration_us: Counter[str] = Counter()
    transdata_count = 0
    transdata_duration_us = 0.0
    weighted_cube_numerator = 0.0
    weighted_cube_denominator = 0.0
    shape_signatures: list[dict[str, Any]] = []
    for run in parsed.get("runs", []):
        kernel_details = run.get("kernel_details", {})
        for row in kernel_details.get("top_kernel_types", []):
            name = str(row.get("name", ""))
            lowered = name.lower()
            count = int(row.get("count", 0))
            duration = float(row.get("duration_us", 0.0))
            if "matmulv2" in lowered:
                dispatch["MatMulV2"] += count
                dispatch_duration_us["MatMulV2"] += duration
            elif "matmulv3" in lowered:
                dispatch["MatMulV3"] += count
                dispatch_duration_us["MatMulV3"] += duration
            elif "matmul" in lowered:
                dispatch[name] += count
                dispatch_duration_us[name] += duration
            if "transdata" in lowered:
                transdata_count += count
                transdata_duration_us += duration
        total_duration = float(kernel_details.get("total_duration_us", 0.0))
        cube = float(
            kernel_details.get("weighted_cube_utilization_pct", 0.0)
        )
        weighted_cube_numerator += cube * total_duration
        weighted_cube_denominator += total_duration
        for row in kernel_details.get("top_shape_format_signatures", []):
            if "matmul" in str(row.get("name", "")).lower():
                shape_signatures.append(row)
    return {
        "command": command,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "parsed_json": str(destination_json),
        "parsed_markdown": str(destination_md),
        "dispatch": {
            "counts": dict(dispatch),
            "duration_us": dict(dispatch_duration_us),
        },
        "transdata": {
            "count": transdata_count,
            "duration_us": transdata_duration_us,
        },
        "weighted_cube_utilization_pct": (
            weighted_cube_numerator / weighted_cube_denominator
            if weighted_cube_denominator
            else 0.0
        ),
        "matmul_shape_format_signatures": shape_signatures,
    }


def _linear_flops_per_call(
    *,
    sequence_length: int,
    hidden_size: int,
    intermediate_size: int,
    layers: int,
) -> int:
    # Four hidden->hidden projections and the two MLP projections per layer.
    per_token_per_layer = (
        4 * 2 * hidden_size * hidden_size
        + 4 * hidden_size * intermediate_size
    )
    return sequence_length * layers * per_token_per_layer


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    import torch_npu

    device = torch.device("npu:0")
    if not torch.npu.is_available():
        raise RuntimeError("vision MatMul lab requires an NPU")
    torch.npu.set_compile_mode(jit_compile=False)
    dtype = torch.float16
    model_dir = args.model.expanduser().resolve()
    sequence_length = int(args.sequence_length)
    intermediate_size = int(args.intermediate_size)
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else DEFAULT_OUTPUT_ROOT
        / (
            f"s{sequence_length}_i{intermediate_size}_"
            f"{args.weight_format}_{args.execution}"
        )
    ).expanduser().resolve()
    profile_dir = (
        args.profile_dir
        if args.profile_dir is not None
        else DEFAULT_PROFILE_ROOT
        / (
            f"s{sequence_length}_i{intermediate_size}_"
            f"{args.weight_format}_{args.execution}"
        )
    ).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"output directory already exists and is non-empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_started = time.perf_counter()
    model = LocalPaddleOCRVLForConditionalGeneration.from_pretrained(
        model_dir,
        dtype=dtype,
        device=device,
    )
    config = model.config.vision_config
    hidden_size = int(config.hidden_size)
    layers = int(config.num_hidden_layers)
    source_intermediate = int(config.intermediate_size)
    if source_intermediate != 4304 or layers != 27 or hidden_size != 1152:
        raise RuntimeError(
            "this lab is pinned to PaddleOCR-VL-1.6 vision dimensions; got "
            f"hidden={hidden_size}, intermediate={source_intermediate}, "
            f"layers={layers}"
        )
    torch.manual_seed(20260729)
    hidden_states = (
        torch.randn(
            (1, sequence_length, hidden_size),
            dtype=dtype,
            device="cpu",
        )
        * 0.02
    ).to(device)

    reference_stage = VisionMatmulStack(
        model,
        intermediate_size=source_intermediate,
    ).eval()
    synchronize(device)
    reference_output = reference_stage(hidden_states)
    synchronize(device)
    if intermediate_size == source_intermediate:
        candidate_stage = reference_stage
    else:
        candidate_stage = VisionMatmulStack(
            model,
            intermediate_size=intermediate_size,
        ).eval()

    format_metadata = _prepare_weight_format(
        candidate_stage,
        requested=args.weight_format,
        torch_npu=torch_npu,
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": format_metadata["status"],
        "purpose": (
            "full 27-layer vision MatMul format/alignment experiment with "
            "attention replaced by a Q/K/V-dependent token-local surrogate"
        ),
        "boundary": (
            "27 x (LayerNorm1 + Q/K/V/out Linear surrogate + residual + "
            "LayerNorm2 + FC1/GELU/FC2 + residual) + post-LayerNorm"
        ),
        "not_an_ocr_path": True,
        "shape": {
            "batch_size": 1,
            "sequence_length": sequence_length,
            "hidden_size": hidden_size,
            "source_intermediate_size": source_intermediate,
            "candidate_intermediate_size": intermediate_size,
            "layers": layers,
            "linear_calls_per_layer": 6,
            "linear_calls_per_full_stack": layers * 6,
        },
        "requested": {
            "weight_format": args.weight_format,
            "execution": args.execution,
        },
        "weight_format": format_metadata,
        "setup_s_through_format_preparation": time.perf_counter()
        - setup_started,
        "environment": _environment(device),
    }
    summary_path = output_dir / "run_summary.json"
    if format_metadata["status"] != "ready":
        summary["reason"] = (
            "requested FRACTAL_NZ could not be materialized by this "
            "torch_npu/runtime; no mislabeled fallback timing was run"
        )
        summary_path.write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2), flush=True)
        return

    synchronize(device)
    raw_candidate_output = candidate_stage(hidden_states)
    synchronize(device)
    raw_candidate_vs_native = _diff(
        raw_candidate_output,
        reference_output,
    )
    cache_dir = _cache_dir(
        args.cache_dir,
        sequence_length=sequence_length,
        intermediate_size=intermediate_size,
        weight_format=args.weight_format,
        dtype=dtype,
        device=device,
        model_dir=model_dir,
    )
    if args.execution == "torchair":
        run, compile_metadata = _compile(
            candidate_stage,
            cache_dir=cache_dir,
            allow_missing=bool(args.allow_compile_if_missing),
            device=device,
            example=hidden_states,
        )
    else:
        run = candidate_stage
        compile_metadata = {
            "api": None,
            "cache_dir": None,
            "cache_existed_before": None,
        }
    for _ in range(args.warmup):
        warm_output = run(hidden_states)
    synchronize(device)
    del warm_output

    flops_per_call = _linear_flops_per_call(
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        layers=layers,
    )
    measurements, output = _measure(
        run,
        hidden_states,
        device=device,
        samples=args.samples,
        calls_per_sample=args.calls_per_sample,
        sequence_length=sequence_length,
        flops_per_call=flops_per_call,
    )
    summary.update(
        {
            "status": "completed",
            "compile": compile_metadata,
            "linear_flops_per_full_stack_call": flops_per_call,
            "measurements": measurements,
            "numerics": {
                "raw_candidate_vs_native_4304": raw_candidate_vs_native,
                "measured_output_vs_raw_candidate": _diff(
                    output,
                    raw_candidate_output,
                ),
                "measured_output_finite": bool(
                    torch.isfinite(output.float()).all().item()
                ),
            },
        }
    )

    if args.profile:
        label = (
            "paddleocr_vl.vision_matmul_lab."
            f"S{sequence_length}.I{intermediate_size}."
            f"{args.weight_format}.{args.execution}"
        )
        summary["profiler"] = _profile(
            run,
            hidden_states,
            profile_dir=profile_dir,
            warmup_steps=args.profile_warmup_steps,
            active_steps=args.profile_steps,
            label=label,
        )
        summary["parsed_profile"] = _parse_profile(
            profile_dir,
            output_dir,
            topn=args.parser_topn,
        )

    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "status": summary["status"],
                "shape": summary["shape"],
                "weight_format": summary["weight_format"],
                "device_median_ms": measurements[
                    "device_event_per_call_ms"
                ]["median"],
                "physical_tokens_per_s": measurements[
                    "physical_tokens_per_s_device_median"
                ],
                "linear_tflop_per_s": measurements[
                    "linear_tflop_per_s_device_median"
                ],
                "dispatch": summary.get("parsed_profile", {}).get(
                    "dispatch"
                ),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
