#!/usr/bin/env python3
"""Benchmark one locked PaddleOCR table-decoder Q/B anchor per process."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import statistics
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
DECODE_OPTIMIZATION = "combined_apply_complete_layer_prefetch1_rope_lut"
SPEC_OPTIMIZATION = "combined_apply_spec_prefetch_mrope"
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
    parser.add_argument("--lane", required=True, choices=tuple(LANES))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument(
        "--decode-vocab-token-ids",
        type=Path,
        default=DEFAULT_COMPACT_VOCAB,
    )
    parser.add_argument("--warmups", type=int, default=WARMUPS)
    parser.add_argument("--repeats", type=int, default=REPEATS)
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


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    lane = LANES[args.lane]
    output_path = args.output_dir.expanduser().resolve() / f"{lane.name}.json"
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "table_q_anchor_single_lane",
        "status": "setup",
        "contract": {
            "lane": lane.name,
            "kind": lane.kind,
            "batch_size": lane.batch_size,
            "query_length": lane.query_length,
            "physical_positions_per_call": (
                lane.batch_size * lane.query_length
            ),
            "cache_length": lane.cache_length,
            "optimization": lane.optimization,
            "spec_attention": MANUAL_SPEC_ATTENTION,
            "dtype": "torch.float16",
            "linear_weight_format": "decode_nz",
            "compact_vocab": str(args.decode_vocab_token_ids.resolve()),
            "warmups": int(args.warmups),
            "repeats": int(args.repeats),
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

    print(f"TABLE_Q_ANCHOR lane={lane.name} stage=model_load", flush=True)
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
    optimization = prepare_decode_optimization_modules(model, lane.optimization)
    prepare_decode_weight_prefetch(model, optimization)
    started = time.perf_counter()
    weight_format = cast_decode_linear_weights_to_nz(model)
    synchronize(device)
    weight_format_s = time.perf_counter() - started
    linear_weight_format = str(weight_format["effective_mode"])

    print(f"TABLE_Q_ANCHOR lane={lane.name} stage=graph_wrapper", flush=True)
    if lane.kind == "decode_q1":
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
        f"TABLE_Q_ANCHOR lane={lane.name} "
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
    expected_count = lane.batch_size * lane.query_length
    if len(native_ids) != expected_count:
        raise RuntimeError(
            f"lane {lane.name} produced {len(native_ids)} IDs, "
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
    payload["warnings"] = warnings
    payload["setup"]["total_process_setup_and_benchmark_s"] = (
        time.perf_counter() - setup_started
    )
    _write(output_path, payload)
    print(
        "TABLE_Q_ANCHOR_RESULT "
        f"lane={lane.name} "
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
