#!/usr/bin/env python3
"""Simulate a fair, profile-guided router over an archived vision crop stream.

This is deliberately post-hoc. It does not change the serving pipeline or run
an NPU. It answers whether the currently cached BxS graphs are sufficient to
justify a live router before that complexity is added to production.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from vision_lab_graph_profile import PINNED_910B2_PROFILE


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
REPO_ROOT = EXPERIMENT_ROOT.parent
DEFAULT_CORPUS = (
    REPO_ROOT
    / "tmp/09_persistent_page_engine/vision_lab"
    / "corpus_256p_minpixels_div4_ee29c91.json"
)
DEFAULT_TRACE = (
    REPO_ROOT
    / "tmp/09_persistent_page_engine"
    / "packed_integration_stageB_minpixels_div4_packed_256p_44c20b6"
    / "recognition_trace.jsonl"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "tmp/09_persistent_page_engine/vision_lab"
    / "router_sim_256p_minpixels_div4.json"
)
DEFAULT_VARIANT = "min_pixels_28224"
DEFAULT_LOOKAHEADS = (1, 4, 8, 16, 32)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument(
        "--trace",
        type=Path,
        default=DEFAULT_TRACE,
        help="Packed E2E trace used for the current-router baseline and eager costs.",
    )
    parser.add_argument(
        "--lookahead",
        type=int,
        action="append",
        default=[],
        help="Repeat for each bounded ready-window size to simulate.",
    )
    parser.add_argument(
        "--audit-lookahead",
        type=int,
        default=16,
        help="Store per-call routing details for this lookahead only.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    args.lookahead = tuple(dict.fromkeys(args.lookahead or DEFAULT_LOOKAHEADS))
    if any(value <= 0 for value in args.lookahead):
        parser.error("--lookahead must be positive")
    if args.audit_lookahead not in args.lookahead:
        parser.error("--audit-lookahead must also be included in --lookahead")
    return args


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
    if not records:
        raise ValueError(f"trace contains no records: {path}")
    return records


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
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


def _load_items(path: Path, variant: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("self_check", {}).get("passed"):
        raise ValueError("corpus regression self-check is not marked passed")
    try:
        raw_items = payload["variants"][variant]["items"]
    except KeyError as exc:
        raise KeyError(f"corpus does not contain variant {variant!r}") from exc
    items = [dict(item) for item in raw_items]
    for index, item in enumerate(items):
        item["source_index"] = index
        item["real_vision_tokens"] = int(item["real_vision_tokens"])
        item["name"] = str(item["name"])
    names = [item["name"] for item in items]
    if len(set(names)) != len(names):
        raise ValueError("corpus request names are not unique")
    return payload, items


def _eager_costs(records: Iterable[dict[str, Any]]) -> dict[str, float]:
    costs: dict[str, float] = {}
    for record in records:
        name = str(record["request_id"])
        cost_ms = float(record.get("device_stage_s", {}).get("vision_prefill", 0.0)) * 1000.0
        if cost_ms > 0.0:
            costs[name] = cost_ms
    return costs


def _pack_candidate(
    pending: Sequence[dict[str, Any]],
    *,
    batch_size: int,
    sequence_length: int,
) -> dict[str, Any] | None:
    """Fill B rows while requiring the oldest crop to participate.

    The oldest crop anchors row zero. Remaining visible crops are considered
    largest-first and placed in the tightest row where they still fit. This is
    bounded best-fit-decreasing, not an exhaustive subset search.
    """
    oldest = pending[0]
    oldest_tokens = int(oldest["real_vision_tokens"])
    if oldest_tokens > sequence_length:
        return None
    rows: list[list[dict[str, Any]]] = [[] for _ in range(batch_size)]
    row_tokens = [0] * batch_size
    rows[0].append(oldest)
    row_tokens[0] = oldest_tokens
    remaining = sorted(
        pending[1:],
        key=lambda item: (-int(item["real_vision_tokens"]), int(item["source_index"])),
    )
    for item in remaining:
        tokens = int(item["real_vision_tokens"])
        candidates = [
            row_index
            for row_index, used in enumerate(row_tokens)
            if used + tokens <= sequence_length
        ]
        if not candidates:
            continue
        selected_row = max(candidates, key=lambda row_index: row_tokens[row_index])
        rows[selected_row].append(item)
        row_tokens[selected_row] += tokens
    selected = [item for row in rows for item in row]
    real_tokens = sum(row_tokens)
    physical_tokens = batch_size * sequence_length
    return {
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "rows": rows,
        "row_real_tokens": row_tokens,
        "selected": selected,
        "real_tokens": real_tokens,
        "physical_tokens": physical_tokens,
        "padding_tokens": physical_tokens - real_tokens,
    }


def _best_candidate(pending: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for (batch_size, sequence_length), cost in PINNED_910B2_PROFILE["graphs"].items():
        candidate = _pack_candidate(
            pending,
            batch_size=int(batch_size),
            sequence_length=int(sequence_length),
        )
        if candidate is None:
            continue
        median_ms = float(cost["median_ms"])
        candidate["profiled_ms"] = median_ms
        candidate["effective_tokens_per_s"] = candidate["real_tokens"] / (median_ms / 1000.0)
        candidates.append(candidate)
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            item["effective_tokens_per_s"],
            item["real_tokens"],
            -item["profiled_ms"],
            -item["physical_tokens"],
        ),
    )


def _shape_label(batch_size: int, sequence_length: int) -> str:
    return f"b{batch_size}_s{sequence_length}"


def _simulate(
    items: Sequence[dict[str, Any]],
    *,
    lookahead: int,
    eager_ms_by_name: dict[str, float],
    keep_decisions: bool,
) -> dict[str, Any]:
    pending: list[dict[str, Any]] = []
    cursor = 0
    compiled_calls = 0
    compiled_real = 0
    compiled_physical = 0
    compiled_ms = 0.0
    eager_real = 0
    eager_ms = 0.0
    eager_names: list[str] = []
    router_ns: list[int] = []
    shapes: Counter[str] = Counter()
    decisions: list[dict[str, Any]] = []

    def refill() -> None:
        nonlocal cursor
        while len(pending) < lookahead and cursor < len(items):
            pending.append(items[cursor])
            cursor += 1

    refill()
    while pending:
        oldest = pending[0]
        route_started = time.perf_counter_ns()
        candidate = _best_candidate(pending)
        router_ns.append(time.perf_counter_ns() - route_started)
        if candidate is None:
            name = str(oldest["name"])
            try:
                cost_ms = eager_ms_by_name[name]
            except KeyError as exc:
                raise KeyError(f"no measured eager vision cost for overflow crop {name}") from exc
            eager_names.append(name)
            eager_real += int(oldest["real_vision_tokens"])
            eager_ms += cost_ms
            if keep_decisions:
                decisions.append(
                    {
                        "kind": "eager_overflow",
                        "oldest_source_index": int(oldest["source_index"]),
                        "crops": [name],
                        "real_tokens": int(oldest["real_vision_tokens"]),
                        "reference_ms": cost_ms,
                    }
                )
            pending.pop(0)
        else:
            selected_indices = {int(item["source_index"]) for item in candidate["selected"]}
            if int(oldest["source_index"]) not in selected_indices:
                raise AssertionError("router candidate omitted the oldest crop")
            label = _shape_label(candidate["batch_size"], candidate["sequence_length"])
            shapes[label] += 1
            compiled_calls += 1
            compiled_real += int(candidate["real_tokens"])
            compiled_physical += int(candidate["physical_tokens"])
            compiled_ms += float(candidate["profiled_ms"])
            if keep_decisions:
                decisions.append(
                    {
                        "kind": "compiled",
                        "shape": label,
                        "oldest_source_index": int(oldest["source_index"]),
                        "rows": [
                            [str(item["name"]) for item in row]
                            for row in candidate["rows"]
                        ],
                        "row_real_tokens": candidate["row_real_tokens"],
                        "real_tokens": candidate["real_tokens"],
                        "physical_tokens": candidate["physical_tokens"],
                        "profiled_ms": candidate["profiled_ms"],
                        "effective_tokens_per_s": candidate["effective_tokens_per_s"],
                    }
                )
            pending[:] = [
                item
                for item in pending
                if int(item["source_index"]) not in selected_indices
            ]
        refill()

    total_real = compiled_real + eager_real
    expected_real = sum(int(item["real_vision_tokens"]) for item in items)
    if total_real != expected_real:
        raise AssertionError(f"router lost tokens: expected={expected_real}, got={total_real}")
    hybrid_ms = compiled_ms + eager_ms
    router_us = [value / 1000.0 for value in router_ns]
    result = {
        "lookahead": lookahead,
        "crops": len(items),
        "total_real_tokens": total_real,
        "compiled": {
            "calls": compiled_calls,
            "crops": len(items) - len(eager_names),
            "real_tokens": compiled_real,
            "physical_tokens": compiled_physical,
            "padding_tokens": compiled_physical - compiled_real,
            "useful_token_fraction": compiled_real / compiled_physical,
            "profiled_device_s": compiled_ms / 1000.0,
            "effective_real_tokens_per_s": compiled_real / (compiled_ms / 1000.0),
            "raw_physical_tokens_per_s": compiled_physical / (compiled_ms / 1000.0),
            "shape_calls": dict(sorted(shapes.items())),
        },
        "eager_overflow": {
            "crops": len(eager_names),
            "real_tokens": eager_real,
            "recorded_reference_device_s": eager_ms / 1000.0,
            "names": eager_names,
        },
        "hybrid_projection": {
            "device_s": hybrid_ms / 1000.0,
            "effective_real_tokens_per_s": total_real / (hybrid_ms / 1000.0),
            "basis": (
                "profiled isolated medians for compiled graphs plus per-crop "
                "recorded E2E device spans for eager overflow"
            ),
        },
        "router_cpu": {
            "decisions": len(router_us),
            "total_ms": sum(router_us) / 1000.0,
            "mean_us": statistics.mean(router_us),
            "p50_us": _percentile(router_us, 0.50),
            "p95_us": _percentile(router_us, 0.95),
            "max_us": max(router_us),
            "scope": "candidate construction and selection only; excludes JSON I/O",
        },
    }
    if keep_decisions:
        result["decision_audit"] = decisions
    return result


def _baseline_from_trace(
    records: Sequence[dict[str, Any]],
    *,
    expected_names: set[str],
) -> dict[str, Any]:
    trace_names = {str(record["request_id"]) for record in records}
    if trace_names != expected_names:
        raise ValueError(
            "trace/corpus request mismatch: "
            f"missing={len(expected_names - trace_names)} extra={len(trace_names - expected_names)}"
        )
    compiled_groups: dict[int, list[dict[str, Any]]] = {}
    eager_records: list[dict[str, Any]] = []
    for record in records:
        vision = record["vision"]
        if str(vision["execution"]) == "compiled":
            compiled_groups.setdefault(int(vision["pack_group_id"]), []).append(record)
        else:
            eager_records.append(record)

    compiled_real = 0
    compiled_physical = 0
    compiled_ms = 0.0
    shape_calls: Counter[str] = Counter()
    for group_id, group in compiled_groups.items():
        first = group[0]["vision"]
        batch_size = 1
        sequence_length = int(first["pack_physical_vision_tokens"])
        shape = (batch_size, sequence_length)
        try:
            cost = PINNED_910B2_PROFILE["graphs"][shape]
        except KeyError as exc:
            raise KeyError(f"baseline group {group_id} uses unprofiled shape {shape}") from exc
        real_tokens = sum(int(record["vision"]["real_vision_tokens"]) for record in group)
        if real_tokens != int(first["pack_real_vision_tokens"]):
            raise AssertionError(f"baseline pack group {group_id} has inconsistent real tokens")
        compiled_real += real_tokens
        compiled_physical += sequence_length
        compiled_ms += float(cost["median_ms"])
        shape_calls[_shape_label(*shape)] += 1

    eager_real = sum(int(record["vision"]["real_vision_tokens"]) for record in eager_records)
    eager_ms = sum(
        float(record.get("device_stage_s", {}).get("vision_prefill", 0.0)) * 1000.0
        for record in eager_records
    )
    if eager_records and eager_ms <= 0.0:
        raise AssertionError("baseline eager records have no positive device timing")
    total_real = compiled_real + eager_real
    hybrid_ms = compiled_ms + eager_ms
    return {
        "name": "current production FIFO packed B1 router",
        "crops": len(records),
        "total_real_tokens": total_real,
        "compiled": {
            "calls": len(compiled_groups),
            "crops": len(records) - len(eager_records),
            "real_tokens": compiled_real,
            "physical_tokens": compiled_physical,
            "padding_tokens": compiled_physical - compiled_real,
            "useful_token_fraction": compiled_real / compiled_physical,
            "profiled_device_s": compiled_ms / 1000.0,
            "effective_real_tokens_per_s": compiled_real / (compiled_ms / 1000.0),
            "raw_physical_tokens_per_s": compiled_physical / (compiled_ms / 1000.0),
            "shape_calls": dict(sorted(shape_calls.items())),
        },
        "eager_overflow": {
            "crops": len(eager_records),
            "real_tokens": eager_real,
            "recorded_reference_device_s": eager_ms / 1000.0,
        },
        "hybrid_projection": {
            "device_s": hybrid_ms / 1000.0,
            "effective_real_tokens_per_s": total_real / (hybrid_ms / 1000.0),
            "basis": (
                "profiled isolated medians for compiled graphs plus per-crop "
                "recorded E2E device spans for eager overflow"
            ),
        },
    }


def _summary_row(result: dict[str, Any]) -> dict[str, Any]:
    router_name = result.get("name")
    if router_name is None:
        router_name = f"lookahead_{result['lookahead']}"
    return {
        "router": router_name,
        "compiled_calls": result["compiled"]["calls"],
        "eager_crops": result["eager_overflow"]["crops"],
        "padding_pct": 100.0 * (1.0 - result["compiled"]["useful_token_fraction"]),
        "compiled_s": result["compiled"]["profiled_device_s"],
        "eager_reference_s": result["eager_overflow"]["recorded_reference_device_s"],
        "hybrid_s": result["hybrid_projection"]["device_s"],
        "hybrid_effective_tokens_per_s": result["hybrid_projection"]["effective_real_tokens_per_s"],
        "router_cpu_total_ms": result.get("router_cpu", {}).get("total_ms"),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    corpus_path = args.corpus.expanduser().resolve()
    trace_path = args.trace.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    corpus, items = _load_items(corpus_path, args.variant)
    records = _read_jsonl(trace_path)
    eager_costs = _eager_costs(records)
    expected_names = {str(item["name"]) for item in items}
    baseline = _baseline_from_trace(records, expected_names=expected_names)
    if baseline["total_real_tokens"] != sum(
        int(item["real_vision_tokens"]) for item in items
    ):
        raise AssertionError("baseline trace and selected corpus variant token totals differ")

    simulations: list[dict[str, Any]] = []
    for lookahead in args.lookahead:
        simulations.append(
            _simulate(
                items,
                lookahead=lookahead,
                eager_ms_by_name=eager_costs,
                keep_decisions=lookahead == args.audit_lookahead,
            )
        )

    payload = {
        "schema_version": 1,
        "kind": "posthoc_profile_guided_vision_router_simulation",
        "inputs": {
            "corpus": str(corpus_path),
            "corpus_source": corpus["source"],
            "variant": args.variant,
            "trace": str(trace_path),
            "crops": len(items),
            "real_vision_tokens": sum(int(item["real_vision_tokens"]) for item in items),
        },
        "profile": {
            key: value
            for key, value in PINNED_910B2_PROFILE.items()
            if key != "graphs"
        },
        "routing_policy": {
            "fairness": "every compiled candidate includes the oldest visible crop",
            "availability_proxy": "ordered rolling crop window",
            "row_packing": "oldest anchors row 0; remaining visible crops use bounded best-fit-decreasing",
            "selection": "maximum useful real tokens divided by pinned median graph cost",
            "candidate_graphs": [
                _shape_label(batch_size, sequence_length)
                for batch_size, sequence_length in PINNED_910B2_PROFILE["graphs"]
            ],
            "not_modeled": [
                "live crop arrival timestamps",
                "layout and CPU preprocessing availability",
                "packing/materialization host overhead",
                "device contention with layout, text prefill, and decode",
            ],
        },
        "baseline": baseline,
        "simulations": simulations,
    }
    payload["headline"] = [_summary_row(baseline)] + [
        _summary_row(result) for result in simulations
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "headline": payload["headline"]}, indent=2))


if __name__ == "__main__":
    main()
