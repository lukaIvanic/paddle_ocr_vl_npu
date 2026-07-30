#!/usr/bin/env python3
"""Analyze and isolate the packed-text final-token GatherV2 operation.

The Phase-21 trace already contains the physical/real sequence lengths,
segment lengths, segment offsets, packed-graph output shape, and graph-level
pass/fail result. Analyze mode reconstructs the exact ``last_token_indices``
input for every graph call. Run mode executes only
``torch.index_select(hidden_states, 1, last_token_indices)`` eagerly or
through TorchAir for one selected call.

Run every NPU lane in a fresh process. An AICore exception can poison the
runtime even when Python catches the resulting exception.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

import torch


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))


def gather_last_tokens(
    hidden_states: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """The exact packed-graph operation under investigation."""

    return torch.index_select(hidden_states, 1, indices)


def load_events(path: Path) -> list[dict[str, Any]]:
    resolved = path.expanduser().resolve()
    events = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                events.append(json.loads(stripped))
            except json.JSONDecodeError as exception:
                raise RuntimeError(
                    f"invalid JSONL record at {resolved}:{line_number}"
                ) from exception
    return events


def _last_indices(
    segment_lengths: Sequence[int],
    max_members: int,
) -> list[int]:
    indices = []
    offset = 0
    for raw_length in segment_lengths:
        length = int(raw_length)
        offset += length
        indices.append(offset - 1)
    if len(indices) > max_members:
        raise ValueError(
            "segment count exceeds packed graph max-members output: "
            f"segments={len(indices)} max_members={max_members}"
        )
    return indices + [0] * (max_members - len(indices))


def analyze_events(
    trace_path: Path,
) -> dict[str, Any]:
    events = load_events(trace_path)
    before: dict[int, dict[str, Any]] = {}
    enqueued: dict[int, dict[str, Any]] = {}
    status: dict[int, str] = {}
    for event in events:
        event_name = event.get("event")
        graph_call_raw = event.get("graph_call")
        if graph_call_raw is None:
            continue
        graph_call = int(graph_call_raw)
        if event_name == "packed_graph_before":
            before[graph_call] = event
        elif event_name == "packed_graph_enqueued":
            enqueued[graph_call] = event
            status[graph_call] = "enqueued"
        elif event_name == "packed_graph_sync_passed":
            status[graph_call] = "passed"
        elif event_name in {
            "packed_graph_enqueue_failed",
            "packed_graph_sync_failed",
        }:
            status[graph_call] = "failed"

    calls = []
    for graph_call in sorted(before):
        event = before[graph_call]
        prepared = event["prepared"]
        output = enqueued.get(graph_call, {}).get("output")
        if output is None:
            calls.append(
                {
                    "graph_call": graph_call,
                    "status": status.get(graph_call, "unknown"),
                    "prepared": prepared,
                    "output": None,
                    "analysis_error": (
                        "packed graph output descriptor is unavailable"
                    ),
                }
            )
            continue
        output_shape = [int(value) for value in output["shape"]]
        if len(output_shape) != 3 or output_shape[0] != 1:
            raise RuntimeError(
                "unexpected packed graph output shape: "
                f"call={graph_call} shape={output_shape}"
            )
        max_members = output_shape[1]
        hidden_size = output_shape[2]
        physical_seq_len = int(prepared["physical_seq_len"])
        real_seq_len = int(prepared["real_seq_len"])
        segment_lengths = [
            int(value) for value in prepared["segment_lengths"]
        ]
        segment_offsets = [
            int(value) for value in prepared["segment_offsets"]
        ]
        last_token_indices = _last_indices(
            segment_lengths,
            max_members,
        )
        expected_offsets = []
        offset = 0
        for length in segment_lengths:
            expected_offsets.append(offset)
            offset += length
        active_indices = last_token_indices[: len(segment_lengths)]
        calls.append(
            {
                "graph_call": graph_call,
                "status": status.get(graph_call, "unknown"),
                "physical_seq_len": physical_seq_len,
                "real_seq_len": real_seq_len,
                "padding_tokens": physical_seq_len - real_seq_len,
                "segment_count": len(segment_lengths),
                "max_members": max_members,
                "hidden_size": hidden_size,
                "dtype": output["dtype"],
                "segment_lengths": segment_lengths,
                "segment_offsets": segment_offsets,
                "last_token_indices": last_token_indices,
                "active_last_token_indices": active_indices,
                "last_active_index": (
                    active_indices[-1] if active_indices else None
                ),
                "last_index_hits_physical_boundary": (
                    bool(active_indices)
                    and active_indices[-1] == physical_seq_len - 1
                ),
                "indices_in_bounds": all(
                    0 <= index < physical_seq_len
                    for index in last_token_indices
                ),
                "offsets_match_lengths": (
                    segment_offsets == expected_offsets
                    and offset == real_seq_len
                ),
                "output": output,
            }
        )

    failed_calls = [
        call for call in calls if call["status"] == "failed"
    ]
    if not failed_calls:
        raise RuntimeError("trace has no packed_graph_sync_failed call")
    failed = failed_calls[-1]
    passing_before = [
        call
        for call in calls
        if call["status"] == "passed"
        and call["graph_call"] < failed["graph_call"]
    ]
    same_shape = [
        call
        for call in passing_before
        if call.get("physical_seq_len") == failed.get("physical_seq_len")
        and call.get("hidden_size") == failed.get("hidden_size")
        and call.get("max_members") == failed.get("max_members")
        and call.get("dtype") == failed.get("dtype")
    ]
    control = (
        same_shape[-1]
        if same_shape
        else (passing_before[-1] if passing_before else None)
    )
    differing_fields: dict[str, Any] = {}
    if control is not None:
        for name in (
            "physical_seq_len",
            "real_seq_len",
            "padding_tokens",
            "segment_count",
            "segment_lengths",
            "segment_offsets",
            "last_token_indices",
            "last_active_index",
            "last_index_hits_physical_boundary",
        ):
            if failed.get(name) != control.get(name):
                differing_fields[name] = {
                    "control": control.get(name),
                    "failed": failed.get(name),
                }
    return {
        "trace_path": str(trace_path.expanduser().resolve()),
        "event_count": len(events),
        "graph_call_count": len(calls),
        "calls": calls,
        "failed_call": failed["graph_call"],
        "control_call": (
            control["graph_call"] if control is not None else None
        ),
        "control_same_static_shape": (
            control is not None
            and control.get("physical_seq_len")
            == failed.get("physical_seq_len")
            and control.get("hidden_size") == failed.get("hidden_size")
            and control.get("max_members") == failed.get("max_members")
            and control.get("dtype") == failed.get("dtype")
        ),
        "failed_vs_control": differing_fields,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_suffix(resolved.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, resolved)


def _exception_record(exception: BaseException) -> dict[str, str]:
    return {
        "exception_type": type(exception).__name__,
        "exception": repr(exception),
        "traceback": traceback.format_exc(),
    }


def _dtype(label: str) -> Any:
    mapping = {
        "torch.float16": torch.float16,
        "torch.bfloat16": torch.bfloat16,
        "torch.float32": torch.float32,
    }
    try:
        return mapping[label]
    except KeyError as exception:
        raise ValueError(
            f"unsupported packed hidden-state dtype: {label}"
        ) from exception


def _select_call(
    analysis: dict[str, Any],
    *,
    selection: str,
    graph_call: int | None,
) -> dict[str, Any]:
    selected_call = graph_call
    if selected_call is None:
        selected_call = analysis[
            "failed_call" if selection == "failed" else "control_call"
        ]
    if selected_call is None:
        raise RuntimeError(f"trace has no {selection} graph call")
    for call in analysis["calls"]:
        if int(call["graph_call"]) == int(selected_call):
            if "analysis_error" in call:
                raise RuntimeError(call["analysis_error"])
            return call
    raise RuntimeError(f"graph call {selected_call} is absent from trace")


def _build_hidden(
    *,
    physical_seq_len: int,
    hidden_size: int,
    dtype: Any,
    device: Any,
) -> Any:
    rows = (
        torch.arange(
            physical_seq_len,
            device=device,
            dtype=torch.float32,
        )
        .remainder(257)
        .div(257)
        .to(dtype)
    )
    return (
        rows.view(1, physical_seq_len, 1)
        .expand(1, physical_seq_len, hidden_size)
        .contiguous()
    )


def _expected_output(
    indices: Sequence[int],
    *,
    hidden_size: int,
    dtype: Any,
) -> Any:
    rows = (
        torch.tensor(indices, dtype=torch.float32)
        .remainder(257)
        .div(257)
        .to(dtype)
    )
    return (
        rows.view(1, len(indices), 1)
        .expand(1, len(indices), hidden_size)
        .contiguous()
    )


def run_lane(args: argparse.Namespace) -> None:
    import torch_npu  # noqa: F401

    analysis = analyze_events(args.trace)
    call = _select_call(
        analysis,
        selection=args.selection,
        graph_call=args.graph_call,
    )
    effective_indices = list(call["last_token_indices"])
    if args.index_variant == "boundary_minus_one":
        active_position = int(call["segment_count"]) - 1
        if active_position < 0:
            raise RuntimeError("selected packed call has no active segments")
        if (
            effective_indices[active_position]
            != int(call["physical_seq_len"]) - 1
        ):
            raise RuntimeError(
                "boundary_minus_one requires the final active index to equal "
                "physical_seq_len - 1"
            )
        effective_indices[active_position] -= 1
    output_path = args.output.expanduser().resolve()
    summary: dict[str, Any] = {
        "mode": "run",
        "status": "starting",
        "backend": args.backend,
        "selection": args.selection,
        "index_variant": args.index_variant,
        "graph_call": call["graph_call"],
        "trace_path": str(args.trace.expanduser().resolve()),
        "call": call,
        "effective_last_token_indices": effective_indices,
        "device": args.device,
    }
    write_json(output_path, summary)

    device = torch.device(args.device)
    if device.type != "npu":
        raise ValueError("the packed GatherV2 probe requires an NPU")
    if not torch.npu.is_available():
        raise RuntimeError("NPU is unavailable")
    torch.npu.set_device(device)
    torch.npu.set_compile_mode(jit_compile=False)
    dtype = _dtype(call["dtype"])
    hidden = _build_hidden(
        physical_seq_len=int(call["physical_seq_len"]),
        hidden_size=int(call["hidden_size"]),
        dtype=dtype,
        device=device,
    )
    target_indices = torch.tensor(
        effective_indices,
        device=device,
        dtype=torch.int64,
    )
    warm_indices = torch.zeros_like(target_indices)

    executable = gather_last_tokens
    cache_was_warm = None
    if args.backend == "torchair":
        if args.cache_dir is None:
            raise ValueError("--cache-dir is required for TorchAir")
        from paddleocr_vl.model.compile_utils import import_torchair

        torchair, CompilerConfig = import_torchair()
        cache_dir = args.cache_dir.expanduser().resolve()
        cache_was_warm = cache_dir.is_dir() and any(cache_dir.iterdir())
        cache_dir.mkdir(parents=True, exist_ok=True)
        summary["cache_dir"] = str(cache_dir)
        summary["cache_was_warm"] = cache_was_warm
        write_json(output_path, summary)
        executable = torchair.inference.cache_compile(
            gather_last_tokens,
            config=CompilerConfig(),
            dynamic=False,
            cache_dir=str(cache_dir),
            ge_cache=True,
        )

    try:
        summary["status"] = "warm_call_starting"
        write_json(output_path, summary)
        warm_started = time.perf_counter()
        warm_output = executable(hidden, warm_indices)
        torch.npu.synchronize()
        summary["warm_call_s"] = time.perf_counter() - warm_started
        summary["warm_output_shape"] = [
            int(value) for value in warm_output.shape
        ]
        summary["status"] = "warm_call_passed"
        write_json(output_path, summary)

        summary["status"] = "target_call_starting"
        write_json(output_path, summary)
        target_started = time.perf_counter()
        target_output = executable(hidden, target_indices)
        torch.npu.synchronize()
        target_s = time.perf_counter() - target_started
        summary["status"] = "target_call_executed"
        summary["target_call_s"] = target_s
        write_json(output_path, summary)

        actual = target_output.cpu()
        expected = _expected_output(
            effective_indices,
            hidden_size=int(call["hidden_size"]),
            dtype=dtype,
        )
        exact = bool(torch.equal(actual, expected))
        maximum_error = float(
            (actual.float() - expected.float()).abs().max().item()
        )
        summary.update(
            {
                "status": "passed" if exact else "incorrect",
                "exact": exact,
                "max_abs_error": maximum_error,
                "output_shape": [int(value) for value in actual.shape],
                "output_sum": float(actual.float().sum().item()),
            }
        )
        write_json(output_path, summary)
        if not exact:
            raise RuntimeError(
                "isolated gather completed but returned incorrect values"
            )
    except BaseException as exception:
        summary.update(
            {
                "status": "failed",
                **_exception_record(exception),
            }
        )
        write_json(output_path, summary)
        raise
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("analyze", "run"),
    )
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--selection",
        choices=("failed", "control"),
        default="failed",
    )
    parser.add_argument("--graph-call", type=int)
    parser.add_argument(
        "--index-variant",
        choices=("recorded", "boundary_minus_one"),
        default="recorded",
        help=(
            "Optionally replace a final active index at physical_seq_len - 1 "
            "with physical_seq_len - 2 while keeping the static graph shape"
        ),
    )
    parser.add_argument(
        "--backend",
        choices=("eager", "torchair"),
        default="eager",
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="npu:0")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.mode == "analyze":
        analysis = analyze_events(args.trace)
        write_json(args.output, analysis)
        print(json.dumps(analysis, ensure_ascii=False, indent=2), flush=True)
        return
    run_lane(args)


if __name__ == "__main__":
    main()
