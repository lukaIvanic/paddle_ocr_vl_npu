#!/usr/bin/env python3
"""Benchmark one locked PaddleOCR table-decoder Q/B anchor per process."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time
from typing import Any, Callable, Sequence

import torch


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
REPO_ROOT = EXPERIMENT_ROOT.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

MANUAL_SPEC_ATTENTION = (
    "manual_grouped_legal_scaled_masked_softmax_fp16_combined_qkv_post_rope"
)
# This is a locked benchmark, not an implementation selector. Ignore an
# inherited experimental override so every saved anchor has one meaning.
os.environ["SPEC_VERIFY_ATTENTION"] = MANUAL_SPEC_ATTENTION

from paddleocr_vl.model.compile_utils import import_torchair  # noqa: E402
from paddleocr_vl.model.modeling import (  # noqa: E402
    LocalPaddleOCRVLForConditionalGeneration,
    _resolve_model_dir,
)
from paddleocr_vl.model.text_decode import (  # noqa: E402
    TextDecodeStage,
    cast_decode_linear_weights_to_nz,
    compile_text_decode_stage,
    load_decode_vocab_token_ids,
    prepare_decode_compact_lm_head,
    prepare_decode_optimization_modules,
    prepare_decode_rope_factor_lut,
    prepare_decode_weight_prefetch,
    resolve_decode_optimization,
    torchair_cache_dir_for_shape,
)
from paddleocr_vl.model.text_mixed_q import (  # noqa: E402
    DEFAULT_MIXED_M16_ATTENTION_ORDER,
    DEFAULT_MIXED_M16_LAYOUT,
    DEFAULT_MIXED_M16_PREFETCH,
    DEFAULT_MIXED_M16_ROTARY_MODE,
    MIXED_M16_OPTIMIZATION,
    MIXED_M16_LAYOUTS,
    MIXED_M16_ATTENTION_ORDERS,
    MIXED_M16_PREFETCH_MODES,
    MIXED_M16_ROTARY_MODES,
    MIXED_LAYOUT_PACKED_BSND_PROMPTFA,
    PACKED_TOKEN_COUNT,
    TextMixedM16Stage,
    mixed_m16_contract,
    torchair_cache_dir_for_mixed_m16,
)
from paddleocr_vl.model.text_spec_verify import (  # noqa: E402
    SPEC_VERIFY_ATTENTION,
    TextSpecVerifyStage,
    _register_scaled_masked_softmax_torchair_converter,
    torchair_cache_dir_for_spec_shape,
    unique_spec_verify_forward,
)
from utils.timing import DeviceTimeline, synchronize  # noqa: E402


DEFAULT_MODEL = Path("/workspace/models/PaddleOCR-VL-1.6")
DEFAULT_CACHE_ROOT = (
    REPO_ROOT / ".runtime_cache/09_persistent_page_engine_torchair"
)
DEFAULT_COMPACT_VOCAB = (
    EXPERIMENT_ROOT
    / "presets/table_compact_vocab/b1_verifier_topfreq_16384.json"
)
DEFAULT_REFERENCE_ANCHORS = (
    REPO_ROOT
    / "tmp/09_persistent_page_engine/"
    "table_q_anchor_ab925dd5_20260904T0948Z"
)
DECODE_OPTIMIZATION = "combined_apply_complete_layer_prefetch1_rope_lut"
SPEC_OPTIMIZATION = "combined_apply_spec_prefetch_mrope"
MIXED_LANE = "mixed_m16"
DRAFT_POSITIONS = (128, 155, 173, 189, 205, 225, 270, 382)
VERIFIER_POSITION = 1249
WARMUPS = 10
REPEATS = 50


@dataclass(frozen=True)
class Lane:
    name: str
    kind: str
    batch_size: int
    query_length: int
    cache_length: int
    optimization: str


LANES = {
    "b8q1": Lane(
        name="b8q1",
        kind="decode_q1",
        batch_size=8,
        query_length=1,
        cache_length=768,
        optimization=DECODE_OPTIMIZATION,
    ),
    "b1q8": Lane(
        name="b1q8",
        kind="spec_verify",
        batch_size=1,
        query_length=8,
        cache_length=4096,
        optimization=SPEC_OPTIMIZATION,
    ),
    "b16q1": Lane(
        name="b16q1",
        kind="decode_q1",
        batch_size=16,
        query_length=1,
        cache_length=768,
        optimization=DECODE_OPTIMIZATION,
    ),
    "b1q16": Lane(
        name="b1q16",
        kind="spec_verify",
        batch_size=1,
        query_length=16,
        cache_length=4096,
        optimization=SPEC_OPTIMIZATION,
    ),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lane",
        required=True,
        choices=(*tuple(LANES), MIXED_LANE),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument(
        "--reference-anchor-dir",
        type=Path,
        default=DEFAULT_REFERENCE_ANCHORS,
    )
    parser.add_argument(
        "--decode-vocab-token-ids",
        type=Path,
        default=DEFAULT_COMPACT_VOCAB,
    )
    parser.add_argument("--warmups", type=int, default=WARMUPS)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument(
        "--mixed-layout",
        choices=MIXED_M16_LAYOUTS,
        default=DEFAULT_MIXED_M16_LAYOUT,
        help="QKV/lane layout used only by the mixed_m16 lane",
    )
    parser.add_argument(
        "--mixed-prefetch",
        choices=MIXED_M16_PREFETCH_MODES,
        default=DEFAULT_MIXED_M16_PREFETCH,
        help="prefetch schedule used only by the mixed_m16 lane",
    )
    parser.add_argument(
        "--mixed-attention-order",
        choices=MIXED_M16_ATTENTION_ORDERS,
        default=DEFAULT_MIXED_M16_ATTENTION_ORDER,
        help="attention branch order used only by the mixed_m16 lane",
    )
    parser.add_argument(
        "--mixed-rotary",
        choices=MIXED_M16_ROTARY_MODES,
        default=DEFAULT_MIXED_M16_ROTARY_MODE,
        help="shared or per-lane ApplyRotary used only by mixed_m16",
    )
    args = parser.parse_args(argv)
    if args.warmups < 1:
        parser.error("--warmups must be positive")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    return args


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _draft_positions(batch_size: int, device: torch.device) -> torch.Tensor:
    repeated = (
        DRAFT_POSITIONS
        * math.ceil(int(batch_size) / len(DRAFT_POSITIONS))
    )[: int(batch_size)]
    return torch.tensor(repeated, device=device, dtype=torch.int64)


def _measure(
    device: torch.device,
    fn: Callable[[], torch.Tensor],
    *,
    warmups: int,
    repeats: int,
) -> tuple[list[float], torch.Tensor, float, list[float]]:
    warmup_wall_s: list[float] = []
    output: torch.Tensor | None = None
    for _ in range(warmups):
        started = time.perf_counter()
        output = fn()
        synchronize(device)
        warmup_wall_s.append(time.perf_counter() - started)

    timeline = DeviceTimeline(device)
    wall_started = time.perf_counter()
    for index in range(repeats):
        output = timeline.measure(f"step_{index:04d}", fn)
    durations = list(timeline.resolve().values())
    host_wall_s = time.perf_counter() - wall_started
    if output is None:
        raise AssertionError("benchmark produced no output")
    return durations, output.clone(), host_wall_s, warmup_wall_s


def _timing(
    durations: list[float],
    *,
    host_wall_s: float,
    physical_positions_per_call: int,
) -> dict[str, Any]:
    total_s = sum(durations)
    calls = len(durations)
    return {
        "device_s": total_s,
        "host_wall_s": float(host_wall_s),
        "latency_ms": {
            "mean": statistics.mean(durations) * 1000.0,
            "median": statistics.median(durations) * 1000.0,
            "p95": _percentile(durations, 0.95) * 1000.0,
            "min": min(durations) * 1000.0,
            "max": max(durations) * 1000.0,
        },
        "graph_calls_per_s": calls / total_s,
        "host_graph_calls_per_s": calls / host_wall_s,
        "physical_positions_per_call": int(physical_positions_per_call),
        "physical_positions_per_s": (
            calls * int(physical_positions_per_call) / total_s
        ),
        "host_physical_positions_per_s": (
            calls * int(physical_positions_per_call) / host_wall_s
        ),
    }


def _profile_calls(
    device: torch.device,
    fn: Callable[[], torch.Tensor],
    profile_dir: Path,
    *,
    warmups: int,
    repeats: int,
) -> dict[str, Any]:
    import torch_npu.profiler as npu_prof

    profiler_warmup_calls = int(warmups)
    captured_calls = int(repeats)
    resolved = profile_dir.expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=False)
    schedule = npu_prof.schedule(
        wait=0,
        warmup=profiler_warmup_calls,
        active=captured_calls,
        repeat=1,
    )
    synchronize(device)
    started = time.perf_counter()
    with npu_prof.profile(
        activities=[
            npu_prof.ProfilerActivity.CPU,
            npu_prof.ProfilerActivity.NPU,
        ],
        schedule=schedule,
        on_trace_ready=npu_prof.tensorboard_trace_handler(
            str(resolved),
            analyse_flag=True,
        ),
        record_shapes=True,
        profile_memory=False,
        with_stack=True,
        with_modules=False,
        with_flops=False,
        experimental_config=npu_prof._ExperimentalConfig(
            profiler_level=npu_prof.ProfilerLevel.Level1,
            aic_metrics=npu_prof.AiCMetrics.PipeUtilization,
            l2_cache=False,
            export_type=npu_prof.ExportType.Text,
            data_simplification=False,
        ),
    ) as profiler:
        for index in range(profiler_warmup_calls + captured_calls):
            with torch.profiler.record_function(
                f"table_q_anchor.profiled_call_{index}"
            ):
                fn()
                synchronize(device)
            profiler.step()
    synchronize(device)
    return {
        "path": str(resolved),
        "wall_s": time.perf_counter() - started,
        "profiler_warmup_calls": profiler_warmup_calls,
        "captured_calls": captured_calls,
        "outside_measured_timing": True,
    }


def _anchor_warnings(lane: Lane, timing: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    median_ms = float(timing["latency_ms"]["median"])
    positions_s = float(timing["physical_positions_per_s"])
    if lane.name == "b1q8":
        if not 1.15 <= median_ms <= 1.50:
            warnings.append(
                "B1Q8 median is outside the historical 1.15-1.50 ms band"
            )
        if not 5300.0 <= positions_s <= 7000.0:
            warnings.append(
                "B1Q8 throughput is outside the historical 5.3k-7.0k band"
            )
    elif lane.name == "b8q1":
        if not 5500.0 <= positions_s <= 7600.0:
            warnings.append(
                "B8Q1 throughput is outside the expected 5.5k-7.6k band"
            )
    return warnings


def _mixed_reference_comparison(
    anchor_dir: Path,
    native_ids: list[int],
) -> tuple[dict[str, Any], list[str]]:
    paths = {
        "verifier": anchor_dir.expanduser().resolve() / "b1q8.json",
        "draft": anchor_dir.expanduser().resolve() / "b8q1.json",
    }
    comparison: dict[str, Any] = {
        "anchor_dir": str(anchor_dir.expanduser().resolve()),
        "available": all(path.is_file() for path in paths.values()),
        "branches": {},
    }
    warnings: list[str] = []
    if not comparison["available"]:
        comparison["missing"] = [
            str(path) for path in paths.values() if not path.is_file()
        ]
        warnings.append("mixed M16 reference anchor outputs are unavailable")
        return comparison, warnings

    actual_by_branch = {
        "verifier": native_ids[:8],
        "draft": native_ids[8:],
    }
    for branch, path in paths.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected = [
            int(value)
            for value in payload["result"]["native_output_token_ids"]
        ]
        actual = actual_by_branch[branch]
        mismatches = [
            {
                "index": index,
                "expected": int(expected_value),
                "actual": int(actual_value),
            }
            for index, (expected_value, actual_value) in enumerate(
                zip(expected, actual, strict=True)
            )
            if int(expected_value) != int(actual_value)
        ]
        comparison["branches"][branch] = {
            "path": str(path),
            "expected_native_ids": expected,
            "actual_native_ids": actual,
            "exact_match": not mismatches,
            "mismatches": mismatches,
        }
        if mismatches:
            warnings.append(
                f"mixed M16 {branch} IDs differ from its isolated anchor"
            )
    return comparison, warnings


def _build_decode_lane(
    model: LocalPaddleOCRVLForConditionalGeneration,
    lane: Lane,
    *,
    device: torch.device,
    dtype: torch.dtype,
    model_dir: Path,
    cache_root: Path,
    linear_weight_format: str,
) -> tuple[Callable[[], torch.Tensor], dict[str, Any], tuple[int, ...]]:
    optimization = resolve_decode_optimization(lane.optimization)
    prepare_decode_rope_factor_lut(
        model,
        optimization,
        cache_length=lane.cache_length,
        dtype=dtype,
    )
    prepare_decode_weight_prefetch(model, optimization)
    stage = TextDecodeStage(
        model,
        optimization=optimization,
        cache_length=lane.cache_length,
    ).eval()
    cache_dir = torchair_cache_dir_for_shape(
        cache_root,
        batch_size=lane.batch_size,
        cache_length=lane.cache_length,
        dtype=dtype,
        device=device,
        model_dir=model_dir,
        linear_weight_format=linear_weight_format,
        optimization=optimization,
    )
    cache_was_warm = cache_dir.is_dir() and any(cache_dir.iterdir())
    wrapper_started = time.perf_counter()
    fn, metadata = compile_text_decode_stage(
        stage,
        backend_name="torchair",
        device=device,
        cache_root=cache_root,
        batch_size=lane.batch_size,
        cache_length=lane.cache_length,
        dtype=dtype,
        model_dir=model_dir,
        linear_weight_format=linear_weight_format,
        optimization=optimization,
    )
    synchronize(device)
    compile_wrapper_s = time.perf_counter() - wrapper_started
    cache = model.allocate_static_cache(
        batch_size=lane.batch_size,
        cache_length=lane.cache_length,
        device=device,
        dtype=dtype,
        init_mode="zeros",
        num_key_value_heads=int(model.config.text_config.num_key_value_heads),
        packed_kv=optimization.packed_kv_scatter,
    )
    input_ids = torch.arange(
        1,
        lane.batch_size + 1,
        device=device,
        dtype=torch.int64,
    ).view(lane.batch_size, 1)
    cache_position = _draft_positions(lane.batch_size, device)
    rope_deltas = torch.zeros(
        (lane.batch_size, 1), device=device, dtype=torch.int64
    )

    def call() -> torch.Tensor:
        return fn(
            input_ids,
            cache_position,
            rope_deltas,
            *cache.flat_tensors(),
        )

    return call, {
        **metadata,
        "cache_was_warm_before_setup": bool(cache_was_warm),
        "compile_wrapper_s": float(compile_wrapper_s),
        "cache_allocated_bytes": sum(
            int(tensor.numel()) * int(tensor.element_size())
            for tensor in cache.flat_tensors()
        ),
    }, tuple(int(value) for value in cache_position.cpu().tolist())


def _build_spec_lane(
    model: LocalPaddleOCRVLForConditionalGeneration,
    lane: Lane,
    *,
    device: torch.device,
    dtype: torch.dtype,
    model_dir: Path,
    cache_root: Path,
    linear_weight_format: str,
) -> tuple[Callable[[], torch.Tensor], dict[str, Any], tuple[int, ...]]:
    optimization = resolve_decode_optimization(lane.optimization)
    prepare_decode_weight_prefetch(model, optimization)
    _register_scaled_masked_softmax_torchair_converter()
    draft_length = lane.query_length - 1
    stage = TextSpecVerifyStage(
        model,
        batch_size=lane.batch_size,
        draft_length=draft_length,
        optimization=optimization,
    ).eval()
    entrypoint = unique_spec_verify_forward(
        stage,
        lane.batch_size,
        draft_length,
    )
    cache_dir = torchair_cache_dir_for_spec_shape(
        cache_root,
        batch_size=lane.batch_size,
        draft_length=draft_length,
        cache_length=lane.cache_length,
        dtype=dtype,
        device=device,
        model_dir=model_dir,
        linear_weight_format=linear_weight_format,
        optimization=optimization,
    )
    cache_was_warm = cache_dir.is_dir() and any(cache_dir.iterdir())
    cache_dir.mkdir(parents=True, exist_ok=True)
    torchair, CompilerConfig = import_torchair()
    wrapper_started = time.perf_counter()
    fn = torchair.inference.cache_compile(
        entrypoint,
        config=CompilerConfig(),
        dynamic=False,
        cache_dir=str(cache_dir),
        ge_cache=True,
    )
    synchronize(device)
    compile_wrapper_s = time.perf_counter() - wrapper_started
    cache = model.allocate_static_cache(
        batch_size=lane.batch_size,
        cache_length=lane.cache_length,
        device=device,
        dtype=dtype,
        init_mode="zeros",
    )
    input_ids = torch.arange(
        1,
        lane.query_length + 1,
        device=device,
        dtype=torch.int64,
    ).view(1, lane.query_length)
    cache_position = torch.tensor(
        (VERIFIER_POSITION,), device=device, dtype=torch.int64
    )
    rope_deltas = torch.zeros((1, 1), device=device, dtype=torch.int64)

    def call() -> torch.Tensor:
        return fn(
            input_ids,
            cache_position,
            rope_deltas,
            *cache.flat_tensors(),
        )

    return call, {
        "boundary": "token_embedding_text_transformer_lm_head_argmax",
        "attention": SPEC_VERIFY_ATTENTION,
        "cache_update": "npu_scatter",
        "optimization": optimization.name,
        "torchair_cache_dir": str(cache_dir),
        "cache_was_warm_before_setup": bool(cache_was_warm),
        "compile_wrapper_s": float(compile_wrapper_s),
        "cache_allocated_bytes": sum(
            int(tensor.numel()) * int(tensor.element_size())
            for tensor in cache.flat_tensors()
        ),
    }, (VERIFIER_POSITION,)


def _build_mixed_m16_lane(
    model: LocalPaddleOCRVLForConditionalGeneration,
    *,
    device: torch.device,
    dtype: torch.dtype,
    model_dir: Path,
    cache_root: Path,
    linear_weight_format: str,
    layout: str,
    prefetch_mode: str,
    attention_order: str,
    rotary_mode: str,
) -> tuple[Callable[[], torch.Tensor], dict[str, Any], tuple[int, ...]]:
    optimization = resolve_decode_optimization(MIXED_M16_OPTIMIZATION)
    prepare_decode_rope_factor_lut(
        model,
        optimization,
        cache_length=4096,
        dtype=dtype,
    )
    prepare_decode_weight_prefetch(model, optimization)
    _register_scaled_masked_softmax_torchair_converter()
    stage = TextMixedM16Stage(
        model,
        optimization=optimization,
        layout=layout,
        prefetch_mode=prefetch_mode,
        attention_order=attention_order,
        rotary_mode=rotary_mode,
    ).eval()
    cache_dir = torchair_cache_dir_for_mixed_m16(
        cache_root,
        dtype=dtype,
        device=device,
        model_dir=model_dir,
        linear_weight_format=linear_weight_format,
        layout=layout,
        prefetch_mode=prefetch_mode,
        attention_order=attention_order,
        rotary_mode=rotary_mode,
    )
    cache_was_warm = cache_dir.is_dir() and any(cache_dir.iterdir())
    cache_dir.mkdir(parents=True, exist_ok=True)
    torchair, CompilerConfig = import_torchair()
    wrapper_started = time.perf_counter()
    fn = torchair.inference.cache_compile(
        stage.forward,
        config=CompilerConfig(),
        dynamic=False,
        cache_dir=str(cache_dir),
        ge_cache=True,
    )
    synchronize(device)
    compile_wrapper_s = time.perf_counter() - wrapper_started

    verifier_cache = model.allocate_static_cache(
        batch_size=1,
        cache_length=4096,
        device=device,
        dtype=dtype,
        init_mode="zeros",
    )
    draft_cache = model.allocate_static_cache(
        batch_size=8,
        cache_length=768,
        device=device,
        dtype=dtype,
        init_mode="zeros",
    )
    verifier_input_ids = torch.arange(
        1, 9, device=device, dtype=torch.int64
    ).view(1, 8)
    verifier_cache_position = torch.tensor(
        (VERIFIER_POSITION,), device=device, dtype=torch.int64
    )
    verifier_rope_deltas = torch.zeros(
        (1, 1), device=device, dtype=torch.int64
    )
    draft_input_ids = torch.arange(
        1, 9, device=device, dtype=torch.int64
    ).view(8, 1)
    draft_cache_position = _draft_positions(8, device)
    draft_rope_deltas = torch.zeros(
        (8, 1), device=device, dtype=torch.int64
    )
    packed_cache = layout == MIXED_LAYOUT_PACKED_BSND_PROMPTFA
    if packed_cache:
        flat_caches = tuple(
            torch.zeros((1, 10240, 2, 128), device=device, dtype=dtype)
            for _ in range(2 * int(model.config.text_config.num_hidden_layers))
        )
    else:
        flat_caches = (*verifier_cache.flat_tensors(), *draft_cache.flat_tensors())

    def call() -> torch.Tensor:
        return fn(
            verifier_input_ids,
            verifier_cache_position,
            verifier_rope_deltas,
            draft_input_ids,
            draft_cache_position,
            draft_rope_deltas,
            *flat_caches,
        )

    def validate_full_forward() -> dict[str, Any]:
        # Outside all timing/profile windows. The reference sees the same
        # deterministic IDs, positions and zero historical prefix as the anchors.
        reference = TextMixedM16Stage(
            model, optimization=optimization, prefetch_mode=prefetch_mode,
            attention_order="draft_then_verifier", rotary_mode="per_lane",
        ).eval()
        reference_ids = reference(
            verifier_input_ids, verifier_cache_position, verifier_rope_deltas,
            draft_input_ids, draft_cache_position, draft_rope_deltas,
            *verifier_cache.flat_tensors(), *draft_cache.flat_tensors(),
        )
        candidate_ids = call()
        synchronize(device)
        differences = []
        layers = int(model.config.text_config.num_hidden_layers)
        for i in range(2 * layers):
            expected_v = verifier_cache.flat_tensors()[i]
            expected_d = draft_cache.flat_tensors()[i]
            actual = flat_caches[i]
            actual_v = actual[:, :4096].transpose(1, 2)
            actual_d = actual[:, 4096:].reshape(8, 768, 2, 128).transpose(1, 2)
            differences.append({
                "tensor": i,
                "verifier_max_abs": float((actual_v - expected_v).abs().max().item()),
                "draft_max_abs": float((actual_d - expected_d).abs().max().item()),
                "verifier_finite": bool(torch.isfinite(actual_v).all().item()),
                "draft_finite": bool(torch.isfinite(actual_d).all().item()),
            })
        return {
            "reference": "full_model_eager_two_attention_same_inputs_zero_prefix",
            "candidate_ids": candidate_ids.cpu().tolist(),
            "reference_ids": reference_ids.cpu().tolist(),
            "matching_ids": int((candidate_ids == reference_ids).sum().item()),
            "total_ids": 16,
            "cache_differences": differences,
        }

    if packed_cache:
        call.validate_full_forward = validate_full_forward

    verifier_cache_bytes = sum(
        int(tensor.numel()) * int(tensor.element_size())
        for tensor in verifier_cache.flat_tensors()
    )
    draft_cache_bytes = sum(
        int(tensor.numel()) * int(tensor.element_size())
        for tensor in draft_cache.flat_tensors()
    )
    return call, {
        "boundary": (
            "two_embeddings_one_transformer_one_attention_one_lm_head"
            if packed_cache else
            "two_embeddings_one_transformer_two_attentions_one_lm_head"
        ),
        "optimization": optimization.name,
        "contract": mixed_m16_contract(
            layout,
            prefetch_mode,
            attention_order,
            rotary_mode,
        ),
        "torchair_cache_dir": str(cache_dir),
        "cache_was_warm_before_setup": bool(cache_was_warm),
        "compile_wrapper_s": float(compile_wrapper_s),
        "verifier_cache_allocated_bytes": verifier_cache_bytes,
        "draft_cache_allocated_bytes": draft_cache_bytes,
        "cache_allocated_bytes": verifier_cache_bytes + draft_cache_bytes,
    }, (
        VERIFIER_POSITION,
        *tuple(int(value) for value in draft_cache_position.cpu().tolist()),
    )


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    mixed_lane = args.lane == MIXED_LANE
    lane = None if mixed_lane else LANES[args.lane]
    lane_name = MIXED_LANE if mixed_lane else lane.name
    optimization_name = (
        MIXED_M16_OPTIMIZATION if mixed_lane else lane.optimization
    )
    physical_positions_per_call = (
        PACKED_TOKEN_COUNT
        if mixed_lane
        else lane.batch_size * lane.query_length
    )
    if mixed_lane:
        contract: dict[str, Any] = {
            "lane": lane_name,
            "kind": "mixed_m16",
            "physical_positions_per_call": PACKED_TOKEN_COUNT,
            **mixed_m16_contract(
                args.mixed_layout,
                args.mixed_prefetch,
                args.mixed_attention_order,
                args.mixed_rotary,
            ),
        }
    else:
        contract = {
            "lane": lane.name,
            "kind": lane.kind,
            "batch_size": lane.batch_size,
            "query_length": lane.query_length,
            "physical_positions_per_call": physical_positions_per_call,
            "cache_length": lane.cache_length,
            "optimization": lane.optimization,
            "spec_attention": MANUAL_SPEC_ATTENTION,
        }
    contract.update(
        {
            "dtype": "torch.float16",
            "linear_weight_format": "decode_nz",
            "compact_vocab": str(args.decode_vocab_token_ids.resolve()),
            "warmups": int(args.warmups),
            "repeats": int(args.repeats),
        }
    )
    output_stem = lane_name
    if mixed_lane and args.mixed_layout != DEFAULT_MIXED_M16_LAYOUT:
        output_stem = f"{lane_name}_{args.mixed_layout}"
    if mixed_lane and args.mixed_prefetch != DEFAULT_MIXED_M16_PREFETCH:
        output_stem = f"{output_stem}_prefetch_{args.mixed_prefetch}"
    if (
        mixed_lane
        and args.mixed_attention_order != DEFAULT_MIXED_M16_ATTENTION_ORDER
    ):
        output_stem = f"{output_stem}_{args.mixed_attention_order}"
    if mixed_lane and args.mixed_rotary != DEFAULT_MIXED_M16_ROTARY_MODE:
        output_stem = f"{output_stem}_rotary_{args.mixed_rotary}"
    output_path = args.output_dir.expanduser().resolve() / f"{output_stem}.json"
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "table_q_anchor_single_lane",
        "status": "setup",
        "contract": contract,
        "provenance": {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": socket.gethostname(),
            "git_commit": _git_commit(),
            "ascend_rt_visible_devices": os.environ.get(
                "ASCEND_RT_VISIBLE_DEVICES"
            ),
            "argv": list(sys.argv),
        },
        "setup": {},
        "result": None,
        "warnings": [],
    }
    _write(output_path, payload)

    import torch_npu  # noqa: F401

    torch.npu.config.allow_internal_format = True
    torch.npu.set_compile_mode(jit_compile=False)
    if not torch.npu.is_available():
        raise RuntimeError("table Q/B anchor benchmark requires an NPU")
    device = torch.device("npu:0")
    dtype = torch.float16
    model_dir = _resolve_model_dir(args.model)
    cache_root = args.cache_dir.expanduser().resolve()
    setup_started = time.perf_counter()

    print(f"TABLE_Q_ANCHOR lane={lane_name} stage=model_load", flush=True)
    started = time.perf_counter()
    model = LocalPaddleOCRVLForConditionalGeneration.from_pretrained(
        model_dir,
        dtype=dtype,
        device=device,
    ).eval()
    synchronize(device)
    model_load_s = time.perf_counter() - started

    token_ids, vocab_metadata = load_decode_vocab_token_ids(
        args.decode_vocab_token_ids,
        full_vocab_size=int(model.lm_head.weight.shape[0]),
    )
    prepare_decode_compact_lm_head(model, token_ids)
    optimization = prepare_decode_optimization_modules(model, optimization_name)
    prepare_decode_weight_prefetch(model, optimization)
    started = time.perf_counter()
    weight_format = cast_decode_linear_weights_to_nz(model)
    synchronize(device)
    weight_format_s = time.perf_counter() - started
    linear_weight_format = str(weight_format["effective_mode"])

    print(f"TABLE_Q_ANCHOR lane={lane_name} stage=graph_wrapper", flush=True)
    if mixed_lane:
        call, runtime_metadata, positions = _build_mixed_m16_lane(
            model,
            device=device,
            dtype=dtype,
            model_dir=model_dir,
            cache_root=cache_root,
            linear_weight_format=linear_weight_format,
            layout=args.mixed_layout,
            prefetch_mode=args.mixed_prefetch,
            attention_order=args.mixed_attention_order,
            rotary_mode=args.mixed_rotary,
        )
    elif lane.kind == "decode_q1":
        call, runtime_metadata, positions = _build_decode_lane(
            model,
            lane,
            device=device,
            dtype=dtype,
            model_dir=model_dir,
            cache_root=cache_root,
            linear_weight_format=linear_weight_format,
        )
    else:
        call, runtime_metadata, positions = _build_spec_lane(
            model,
            lane,
            device=device,
            dtype=dtype,
            model_dir=model_dir,
            cache_root=cache_root,
            linear_weight_format=linear_weight_format,
        )

    payload["setup"] = {
        "model_dir": str(model_dir),
        "model_load_s": float(model_load_s),
        "weight_format_s": float(weight_format_s),
        "weight_format": weight_format,
        "compact_vocab": vocab_metadata,
        "runtime": runtime_metadata,
        "cache_positions": list(positions),
        "setup_before_warmups_s": time.perf_counter() - setup_started,
    }
    _write(output_path, payload)

    print(
        f"TABLE_Q_ANCHOR lane={lane_name} "
        f"warmups={args.warmups} repeats={args.repeats}",
        flush=True,
    )
    durations, output, host_wall_s, warmup_wall_s = _measure(
        device,
        call,
        warmups=args.warmups,
        repeats=args.repeats,
    )
    synchronize(device)
    native_ids = output.detach().cpu().reshape(-1).tolist()
    expected_count = physical_positions_per_call
    if len(native_ids) != expected_count:
        raise RuntimeError(
            f"lane {lane_name} produced {len(native_ids)} IDs, "
            f"expected {expected_count}"
        )
    full_vocab_size = int(model.config.text_config.vocab_size)
    if any(int(value) < 0 or int(value) >= full_vocab_size for value in native_ids):
        raise RuntimeError("lane produced an out-of-range native token ID")

    timing = _timing(
        durations,
        host_wall_s=host_wall_s,
        physical_positions_per_call=expected_count,
    )
    profile_metadata = (
        _profile_calls(
            device,
            call,
            args.profile_dir,
            warmups=args.warmups,
            repeats=args.repeats,
        )
        if args.profile_dir is not None
        else None
    )
    if mixed_lane:
        reference_comparison, warnings = _mixed_reference_comparison(
            args.reference_anchor_dir,
            [int(value) for value in native_ids],
        )
    else:
        reference_comparison = None
        warnings = _anchor_warnings(lane, timing)
    payload["status"] = "complete"
    payload["result"] = {
        "timing": timing,
        "warmup_wall_s": {
            "calls": len(warmup_wall_s),
            "first": warmup_wall_s[0],
            "remaining_mean": (
                statistics.mean(warmup_wall_s[1:])
                if len(warmup_wall_s) > 1
                else None
            ),
            "all": warmup_wall_s,
        },
        "native_output_token_ids": [int(value) for value in native_ids],
    }
    if reference_comparison is not None:
        payload["result"]["reference_comparison"] = reference_comparison
    if profile_metadata is not None:
        payload["result"]["profile"] = profile_metadata
    if hasattr(call, "validate_full_forward"):
        print("TABLE_Q_ANCHOR full-forward KV/ID comparison", flush=True)
        payload["result"]["full_forward_comparison"] = call.validate_full_forward()
    payload["warnings"] = warnings
    payload["setup"]["total_process_setup_and_benchmark_s"] = (
        time.perf_counter() - setup_started
    )
    _write(output_path, payload)
    print(
        "TABLE_Q_ANCHOR_RESULT "
        f"lane={lane_name} "
        f"median_ms={timing['latency_ms']['median']:.4f} "
        f"p95_ms={timing['latency_ms']['p95']:.4f} "
        f"iters_s={timing['graph_calls_per_s']:.1f} "
        f"physical_positions_s={timing['physical_positions_per_s']:.1f} "
        f"warnings={len(warnings)} "
        f"output={output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
