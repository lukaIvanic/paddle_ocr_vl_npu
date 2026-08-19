#!/usr/bin/env python3
"""Measure one cached UniRec decoder graph across realistic mask occupancies."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from modeling_optimized_unirec import (
    LOCAL_UNIREC_STATIC_CACHE_LEN,
    OptimizedUniRecRunner,
    synchronize_device,
)
from text_decode_lab import make_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--artifact-crops-jsonl", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--self-cache-length", type=int, required=True)
    parser.add_argument("--cross-cache-length", type=int, required=True)
    parser.add_argument("--active-rows", type=int, nargs="+", required=True)
    parser.add_argument("--cache-positions", type=int, nargs="+", required=True)
    parser.add_argument(
        "--source-modes",
        nargs="+",
        choices=("realistic", "full"),
        default=("realistic",),
        help=(
            "realistic samples source lengths from the artifact; full marks "
            "every active row valid through the static cross-KV capacity"
        ),
    )
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=30)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if any(not 1 <= value <= args.batch_size for value in args.active_rows):
        parser.error("--active-rows values must be within the batch")
    if any(
        not 0 <= value < args.self_cache_length
        for value in args.cache_positions
    ):
        parser.error("--cache-positions values must be inside self-KV")
    if max(args.cache_positions) + args.measure_steps >= args.self_cache_length:
        parser.error("cache position plus measured steps exceeds self-KV")
    return args


def read_source_lengths(path: Path, capacity: int) -> list[int]:
    lengths = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            length = int(row["cross_kv"]["source_length"])
            if length <= capacity:
                lengths.append(length)
    if not lengths:
        raise ValueError(f"no source lengths fit cross-KV capacity {capacity}")
    return sorted(lengths)


def quantile_sample(sorted_values: list[int], count: int) -> list[int]:
    """Choose a deterministic distribution-preserving sample."""
    if count < 1:
        return []
    size = len(sorted_values)
    indices = np.linspace(0, size - 1, num=count, dtype=np.float64)
    return [sorted_values[int(round(index))] for index in indices]


def distribution(values: list[int]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": int(array.min()),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p99": float(np.percentile(array, 99)),
        "max": int(array.max()),
    }


def configure_masks_(
    state: dict[str, Any],
    *,
    active_rows: int,
    active_position: int,
    source_lengths: list[int],
) -> None:
    batch = int(state["cache_position"].shape[0])
    cross_len = int(state["cross_mask"].shape[-1])
    if len(source_lengths) != active_rows:
        raise ValueError("source-length sample does not match active rows")
    lengths = torch.zeros(batch, dtype=torch.int64, device=state["cross_mask"].device)
    lengths[:active_rows] = torch.tensor(
        source_lengths, dtype=torch.int64, device=lengths.device
    )
    positions = torch.arange(
        cross_len, dtype=torch.int64, device=lengths.device
    ).view(1, 1, 1, cross_len)
    valid = positions < lengths.view(batch, 1, 1, 1)
    negative = torch.finfo(torch.float32).min
    state["cross_mask"].copy_(
        torch.where(
            valid,
            torch.zeros((), dtype=torch.float32, device=valid.device),
            torch.full((), negative, dtype=torch.float32, device=valid.device),
        )
    )
    state["cache_position"].fill_(1)
    state["cache_position"][:active_rows].fill_(active_position)


def allocate_pinned(shape: tuple[int, ...]) -> tuple[torch.Tensor, bool]:
    try:
        return torch.empty(shape, dtype=torch.int64, pin_memory=True), True
    except RuntimeError:
        return torch.empty(shape, dtype=torch.int64), False


@torch.inference_mode()
def measure_point(
    *,
    runner: OptimizedUniRecRunner,
    module: Any,
    state: dict[str, Any],
    active_rows: int,
    active_position: int,
    source_lengths: list[int],
    source_mode: str,
    warmup_steps: int,
    measure_steps: int,
) -> dict[str, Any]:
    batch = int(state["cache_position"].shape[0])
    configure_masks_(
        state,
        active_rows=active_rows,
        active_position=active_position,
        source_lengths=source_lengths,
    )
    next_host, next_pinned = allocate_pinned((batch,))
    position_host, position_pinned = allocate_pinned((batch,))
    next_array = next_host.numpy()
    position_array = position_host.numpy()
    eos = int(runner.config.eos_token_id)

    def reset_host() -> None:
        next_array[:] = int(runner.config.decoder_start_token_id)
        next_array[active_rows:] = eos
        position_array[:] = 1
        position_array[:active_rows] = active_position

    def input_copy() -> None:
        state["next_token"].view(-1).copy_(next_host, non_blocking=next_pinned)
        state["cache_position"].copy_(
            position_host, non_blocking=position_pinned
        )

    def forward_and_wait() -> list[int]:
        logits = module(
            state["next_token"],
            state["cache_position"],
            0,
            state["self_keys"],
            state["self_values"],
            state["cross_keys"],
            state["cross_values"],
            state["cross_mask"],
        )
        predicted = runner.model.select_next_token(logits)
        return [int(value) for value in predicted.detach().cpu().view(-1).tolist()]

    reset_host()
    for _ in range(warmup_steps):
        input_copy()
        predicted = forward_and_wait()
        next_array[:active_rows] = predicted[:active_rows]
        next_array[active_rows:] = eos
        position_array[:active_rows] += 1
    synchronize_device(runner.device)

    reset_host()
    input_build_s = 0.0
    decode_s = 0.0
    iteration_s = 0.0
    for _ in range(measure_steps):
        iteration_started = time.perf_counter()
        input_started = time.perf_counter()
        input_copy()
        input_build_s += time.perf_counter() - input_started
        decode_started = time.perf_counter()
        predicted = forward_and_wait()
        decode_s += time.perf_counter() - decode_started
        next_array[:active_rows] = predicted[:active_rows]
        next_array[active_rows:] = eos
        position_array[:active_rows] += 1
        iteration_s += time.perf_counter() - iteration_started
    synchronize_device(runner.device)
    raw_slots = batch * measure_steps
    effective = active_rows * measure_steps
    return {
        "active_rows": active_rows,
        "active_fraction": active_rows / batch,
        "initial_cache_position": active_position,
        "source_lengths": distribution(source_lengths),
        "source_mode": source_mode,
        "warmup_steps": warmup_steps,
        "measure_steps": measure_steps,
        "decode_s": decode_s,
        "decode_step_ms": decode_s * 1000.0 / measure_steps,
        "raw_tok_s": raw_slots / decode_s,
        "effective_tok_s": effective / decode_s,
        "input_build_step_ms": input_build_s * 1000.0 / measure_steps,
        "iteration_step_ms": iteration_s * 1000.0 / measure_steps,
        "host_pinned": bool(next_pinned and position_pinned),
    }


def main() -> None:
    args = parse_args()
    if args.self_cache_length != LOCAL_UNIREC_STATIC_CACHE_LEN:
        raise ValueError(
            "UNIREC_STATIC_CACHE_LEN does not match --self-cache-length: "
            f"{LOCAL_UNIREC_STATIC_CACHE_LEN} != {args.self_cache_length}"
        )
    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    runner = OptimizedUniRecRunner(
        model_path=args.model,
        device=args.device,
        dtype="float16",
        compile_cache_dir=args.cache_dir,
    )
    module, compile_meta = runner._compile_decode_module(
        backend="torchair",
        self_attention_backend="increfa_all",
        compile_dynamic=False,
        cross_cache_len=args.cross_cache_length,
        batch_size=args.batch_size,
    )
    state = make_state(
        runner,
        batch_size=args.batch_size,
        self_cache_length=args.self_cache_length,
        cross_cache_length=args.cross_cache_length,
        cache_position=min(args.cache_positions),
        seed=7,
    )
    eligible = read_source_lengths(
        args.artifact_crops_jsonl, args.cross_cache_length
    )
    points = []
    first_call_s = None
    for position in args.cache_positions:
        for active_rows in args.active_rows:
            for source_mode in args.source_modes:
                sample = (
                    quantile_sample(eligible, active_rows)
                    if source_mode == "realistic"
                    else [args.cross_cache_length] * active_rows
                )
                if first_call_s is None:
                    configure_masks_(
                        state,
                        active_rows=active_rows,
                        active_position=position,
                        source_lengths=sample,
                    )
                    started = time.perf_counter()
                    _ = module(
                        state["next_token"],
                        state["cache_position"],
                        0,
                        state["self_keys"],
                        state["self_values"],
                        state["cross_keys"],
                        state["cross_values"],
                        state["cross_mask"],
                    )
                    synchronize_device(runner.device)
                    first_call_s = time.perf_counter() - started
                point = measure_point(
                    runner=runner,
                    module=module,
                    state=state,
                    active_rows=active_rows,
                    active_position=position,
                    source_lengths=sample,
                    source_mode=source_mode,
                    warmup_steps=args.warmup_steps,
                    measure_steps=args.measure_steps,
                )
                points.append(point)
                print(
                    "UNIREC_DECODE_MASK_SWEEP_POINT "
                    + json.dumps(point, sort_keys=True),
                    flush=True,
                )
    payload = {
        "schema_version": 1,
        "kind": "unirec_decode_mask_occupancy_sweep",
        "status": "ok",
        "physical_devices": [
            int(value)
            for value in os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").split(",")
            if value.strip().isdigit()
        ],
        "shape": {
            "batch_size": args.batch_size,
            "self_cache_length": args.self_cache_length,
            "cross_cache_length": args.cross_cache_length,
        },
        "eligible_source_lengths": distribution(eligible),
        "first_call_s": first_call_s,
        "compile": compile_meta,
        "points": points,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        "UNIREC_DECODE_MASK_SWEEP: PASS output=" + str(args.output),
        flush=True,
    )


if __name__ == "__main__":
    main()
