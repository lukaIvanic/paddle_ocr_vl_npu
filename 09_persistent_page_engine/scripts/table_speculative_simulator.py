#!/usr/bin/env python3
"""Simulate target-token acceptance from precomputed table-row OCR drafts."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
from typing import Any


DEFAULT_BLOCK_COSTS = {1: 1.0, 8: 1.05, 16: 1.10, 32: 1.30}


@dataclass
class MatcherState:
    cursor: int = 0


@dataclass(frozen=True)
class Candidate:
    start: int
    tokens: tuple[int, ...]
    anchor_tokens: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--drafts", type=Path, required=True)
    parser.add_argument("--baseline-records", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block-sizes", default="1,8,16,32")
    parser.add_argument(
        "--matchers",
        default=(
            "suffix_global_a1,suffix_monotonic_a1,suffix_monotonic_a2,"
            "suffix_monotonic_a4,oracle_global"
        ),
    )
    parser.add_argument("--max-anchor-tokens", type=int, default=64)
    parser.add_argument("--backtrack-tokens", type=int, default=8)
    parser.add_argument("--vision-tok-per-s", type=float, default=30_000.0)
    parser.add_argument("--draft-decode-tok-per-s", type=float, default=6_000.0)
    parser.add_argument("--target-decode-tok-per-s", type=float, default=750.0)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {name: None for name in ("min", "mean", "p50", "p75", "p90", "p99", "max")}
    return {
        "min": min(values),
        "mean": statistics.mean(values),
        "p50": quantile(values, 0.50),
        "p75": quantile(values, 0.75),
        "p90": quantile(values, 0.90),
        "p99": quantile(values, 0.99),
        "max": max(values),
    }


def target_tokens(record: dict[str, Any]) -> list[int]:
    rows = record.get("rows") or []
    if len(rows) != 1:
        raise ValueError(f"{record.get('request_id')}: whole target must contain one row")
    tokens = [int(value) for value in rows[0].get("token_ids") or ()]
    if not tokens:
        raise ValueError(f"{record.get('request_id')}: target token_ids are empty")
    return tokens


def flatten_drafts(record: dict[str, Any], eos_token: int) -> tuple[list[int], list[int]]:
    flat: list[int] = []
    row_for_token: list[int] = []
    for row_index, row in enumerate(sorted(record.get("rows") or [], key=lambda item: item["row_index"])):
        tokens = [int(value) for value in row.get("token_ids") or ()]
        if tokens and tokens[-1] == eos_token:
            tokens.pop()
        flat.extend(tokens)
        row_for_token.extend([row_index] * len(tokens))
    return flat, row_for_token


def preceding_match(
    prefix: list[int],
    draft: list[int],
    continuation: int,
    maximum: int,
) -> int:
    matched = 0
    while (
        matched < maximum
        and matched < len(prefix)
        and matched < continuation
        and prefix[-matched - 1] == draft[continuation - matched - 1]
    ):
        matched += 1
    return matched


def build_continuation_index(
    draft: list[int],
    maximum_anchor: int,
) -> dict[int, dict[tuple[int, ...], list[int]]]:
    lengths = []
    length = 1
    while length <= maximum_anchor:
        lengths.append(length)
        length *= 2
    index: dict[int, dict[tuple[int, ...], list[int]]] = {}
    for anchor_length in lengths:
        by_anchor: defaultdict[tuple[int, ...], list[int]] = defaultdict(list)
        for continuation in range(anchor_length, len(draft)):
            by_anchor[tuple(draft[continuation - anchor_length : continuation])].append(
                continuation
            )
        index[anchor_length] = dict(by_anchor)
    return index


def lcp(left: list[int] | tuple[int, ...], right: list[int] | tuple[int, ...]) -> int:
    matched = 0
    for lhs, rhs in zip(left, right):
        if lhs != rhs:
            break
        matched += 1
    return matched


def suffix_candidate(
    prefix: list[int],
    draft: list[int],
    continuation_index: dict[int, dict[tuple[int, ...], list[int]]],
    state: MatcherState,
    block_size: int,
    minimum_anchor: int,
    maximum_anchor: int,
    backtrack_tokens: int,
    monotonic: bool,
) -> Candidate | None:
    if not prefix or not draft:
        return None
    lower_bound = max(1, state.cursor - backtrack_tokens) if monotonic else 1
    usable_lengths = [
        length
        for length in continuation_index
        if minimum_anchor <= length <= len(prefix)
    ]
    for indexed_anchor in sorted(usable_lengths, reverse=True):
        positions = continuation_index[indexed_anchor].get(
            tuple(prefix[-indexed_anchor:]),
            (),
        )
        best: tuple[tuple[int, int, int], Candidate] | None = None
        for continuation in positions:
            if continuation < lower_bound:
                continue
            anchor = preceding_match(prefix, draft, continuation, maximum_anchor)
            candidate_tokens = tuple(draft[continuation : continuation + block_size])
            if not candidate_tokens:
                continue
            forward = int(continuation >= state.cursor)
            distance = abs(continuation - state.cursor)
            score = (anchor, forward, -distance)
            candidate = Candidate(continuation, candidate_tokens, anchor)
            if best is None or score > best[0]:
                best = (score, candidate)
        if best is not None:
            return best[1]
    return None


def oracle_start_matches(target: list[int], draft: list[int]) -> list[tuple[int, int]]:
    """Return (longest match, draft start) for every target start in linear time."""
    transitions: list[dict[int, int]] = [{}]
    links = [-1]
    lengths = [0]
    first_positions = [-1]
    last = 0
    for position, token in enumerate(reversed(draft)):
        current = len(transitions)
        transitions.append({})
        lengths.append(lengths[last] + 1)
        links.append(0)
        first_positions.append(position)
        parent = last
        while parent >= 0 and token not in transitions[parent]:
            transitions[parent][token] = current
            parent = links[parent]
        if parent < 0:
            links[current] = 0
        else:
            successor = transitions[parent][token]
            if lengths[parent] + 1 == lengths[successor]:
                links[current] = successor
            else:
                clone = len(transitions)
                transitions.append(dict(transitions[successor]))
                lengths.append(lengths[parent] + 1)
                links.append(links[successor])
                first_positions.append(first_positions[successor])
                while parent >= 0 and transitions[parent].get(token) == successor:
                    transitions[parent][token] = clone
                    parent = links[parent]
                links[successor] = clone
                links[current] = clone
        last = current

    result = [(0, 0)] * len(target)
    state = 0
    matched = 0
    draft_length = len(draft)
    for reverse_index, token in enumerate(reversed(target)):
        while state and token not in transitions[state]:
            state = links[state]
            matched = min(matched, lengths[state])
        if token in transitions[state]:
            state = transitions[state][token]
            matched += 1
        else:
            state = 0
            matched = 0
        target_start = len(target) - reverse_index - 1
        draft_start = (
            draft_length - first_positions[state] - 1
            if matched
            else 0
        )
        result[target_start] = (matched, draft_start)
    return result


def simulate(
    target: list[int],
    draft: list[int],
    matcher: str,
    block_size: int,
    block_cost: float,
    maximum_anchor: int,
    backtrack_tokens: int,
    continuation_index: dict[int, dict[tuple[int, ...], list[int]]],
) -> dict[str, Any]:
    if not target:
        raise ValueError("target must not be empty")
    state = MatcherState()
    position = 1  # The full-table prefill produces token zero.
    calls = 0
    speculative_calls = 0
    fallback_calls = 0
    accepted_tokens = 0
    proposed_tokens = 0
    weighted_forward_equivalents = 0.0
    anchors: list[int] = []
    accept_lengths: list[int] = []
    trace: list[dict[str, Any]] = []

    matcher_parts = matcher.split("_")
    is_oracle = matcher_parts[0] == "oracle"
    monotonic = "monotonic" in matcher_parts
    minimum_anchor = int(matcher_parts[-1][1:]) if matcher_parts[-1].startswith("a") else 1
    oracle_matches = oracle_start_matches(target, draft) if is_oracle else []

    while position < len(target):
        prefix = target[:position]
        if is_oracle:
            match_length, draft_start = oracle_matches[position]
            candidate_length = min(match_length, block_size)
            candidate = (
                Candidate(
                    draft_start,
                    tuple(draft[draft_start : draft_start + candidate_length]),
                    0,
                )
                if candidate_length
                else None
            )
        else:
            candidate = suffix_candidate(
                prefix,
                draft,
                continuation_index,
                state,
                block_size,
                minimum_anchor,
                maximum_anchor,
                backtrack_tokens,
                monotonic,
            )

        calls += 1
        if candidate is None:
            fallback_calls += 1
            weighted_forward_equivalents += 1.0
            position += 1
            continue

        speculative_calls += 1
        weighted_forward_equivalents += block_cost
        proposed_tokens += len(candidate.tokens)
        accepted = lcp(candidate.tokens, target[position:])
        accepted_tokens += accepted
        accept_lengths.append(accepted)
        anchors.append(candidate.anchor_tokens)
        correction_matches_draft = (
            position + accepted < len(target)
            and candidate.start + accepted < len(draft)
            and target[position + accepted] == draft[candidate.start + accepted]
        )
        if accepted:
            state.cursor = max(state.cursor, candidate.start + accepted)
        if correction_matches_draft:
            state.cursor = max(state.cursor, candidate.start + accepted + 1)
        emitted = min(len(target) - position, accepted + 1)
        if len(trace) < 16:
            trace.append(
                {
                    "target_position": position,
                    "draft_start": candidate.start,
                    "anchor_tokens": candidate.anchor_tokens,
                    "proposed_tokens": len(candidate.tokens),
                    "accepted_tokens": accepted,
                    "emitted_tokens": emitted,
                }
            )
        position += emitted

    baseline_iterations = max(0, len(target) - 1)
    return {
        "target_tokens_including_eos": len(target),
        "draft_tokens_without_row_eos": len(draft),
        "baseline_decode_iterations": baseline_iterations,
        "target_calls": calls,
        "speculative_calls": speculative_calls,
        "fallback_calls": fallback_calls,
        "accepted_draft_tokens": accepted_tokens,
        "proposed_draft_tokens": proposed_tokens,
        "accepted_fraction_of_proposed": (
            accepted_tokens / proposed_tokens if proposed_tokens else 0.0
        ),
        "accepted_coverage_of_target_decode": (
            accepted_tokens / baseline_iterations if baseline_iterations else 0.0
        ),
        "accepted_tokens_per_speculative_call": (
            accepted_tokens / speculative_calls if speculative_calls else 0.0
        ),
        "target_tokens_per_call": (
            baseline_iterations / calls if calls else None
        ),
        "weighted_target_forward_equivalents": weighted_forward_equivalents,
        "ideal_target_decode_speedup": (
            baseline_iterations / weighted_forward_equivalents
            if weighted_forward_equivalents
            else None
        ),
        "anchor_tokens": distribution([float(value) for value in anchors]),
        "accept_length": distribution([float(value) for value in accept_lengths]),
        "trace_head": trace,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    target_tokens = sum(row["simulation"]["target_tokens_including_eos"] for row in rows)
    baseline_iterations = sum(row["simulation"]["baseline_decode_iterations"] for row in rows)
    accepted = sum(row["simulation"]["accepted_draft_tokens"] for row in rows)
    proposed = sum(row["simulation"]["proposed_draft_tokens"] for row in rows)
    speculative_calls = sum(row["simulation"]["speculative_calls"] for row in rows)
    target_calls = sum(row["simulation"]["target_calls"] for row in rows)
    weighted = sum(row["simulation"]["weighted_target_forward_equivalents"] for row in rows)
    baseline_s = sum(row["cost_model_s"]["baseline_total"] for row in rows)
    speculative_s = sum(row["cost_model_s"]["speculative_total"] for row in rows)
    speedups = [row["cost_model_s"]["total_speedup"] for row in rows]
    return {
        "tables": len(rows),
        "target_tokens_including_eos": target_tokens,
        "baseline_decode_iterations": baseline_iterations,
        "target_calls": target_calls,
        "speculative_calls": speculative_calls,
        "accepted_draft_tokens": accepted,
        "proposed_draft_tokens": proposed,
        "accepted_fraction_of_proposed": accepted / proposed if proposed else 0.0,
        "accepted_coverage_of_target_decode": (
            accepted / baseline_iterations if baseline_iterations else 0.0
        ),
        "accepted_tokens_per_speculative_call": (
            accepted / speculative_calls if speculative_calls else 0.0
        ),
        "target_tokens_per_call": baseline_iterations / target_calls if target_calls else None,
        "weighted_target_forward_equivalents": weighted,
        "ideal_target_decode_speedup": baseline_iterations / weighted if weighted else None,
        "projected_baseline_s": baseline_s,
        "projected_speculative_s": speculative_s,
        "projected_total_speedup": baseline_s / speculative_s if speculative_s else None,
        "tables_faster_than_baseline": sum(value > 1.0 for value in speedups),
        "table_speedup": distribution(speedups),
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Offline table speculative-decoding simulation",
        "",
        "The suffix matchers use only the generated target prefix. Oracle lanes inspect future target tokens and are upper bounds only.",
        "",
        "## Aggregate configurations",
        "",
        "| strategy | matcher | block | accept/proposed | target coverage | accepted/spec call | target decode speedup | projected total speedup |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    configs = sorted(
        report["aggregate"],
        key=lambda row: row["metrics"]["projected_total_speedup"],
        reverse=True,
    )
    for row in configs:
        metrics = row["metrics"]
        lines.append(
            "| {strategy} | {matcher} | {block} | {accept:.3f} | {coverage:.3f} | "
            "{accepted:.2f} | {decode:.2f}x | {total:.2f}x |".format(
                strategy=row["strategy"],
                matcher=row["matcher"],
                block=row["block_size"],
                accept=metrics["accepted_fraction_of_proposed"],
                coverage=metrics["accepted_coverage_of_target_decode"],
                accepted=metrics["accepted_tokens_per_speculative_call"],
                decode=metrics["ideal_target_decode_speedup"] or 0.0,
                total=metrics["projected_total_speedup"] or 0.0,
            )
        )
    lines.extend(["", "## Cost model", ""])
    for key, value in report["cost_model"].items():
        lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    block_sizes = tuple(int(value) for value in args.block_sizes.split(",") if value.strip())
    unknown_blocks = set(block_sizes) - set(DEFAULT_BLOCK_COSTS)
    if unknown_blocks:
        raise ValueError(f"missing cost factors for block sizes: {sorted(unknown_blocks)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    targets = {row["request_id"]: row for row in read_jsonl(args.targets)}
    drafts = read_jsonl(args.drafts)
    baseline = (
        {row["request_id"]: row for row in read_jsonl(args.baseline_records)}
        if args.baseline_records
        else {}
    )
    matchers = tuple(
        value.strip()
        for value in args.matchers.split(",")
        if value.strip()
    )
    detailed: list[dict[str, Any]] = []

    for draft_record in drafts:
        request_id = draft_record["request_id"]
        if request_id not in targets:
            continue
        target_record = targets[request_id]
        tokens = target_tokens(target_record)
        flat_draft, _row_for_token = flatten_drafts(draft_record, tokens[-1])
        continuation_index = build_continuation_index(flat_draft, args.max_anchor_tokens)
        target_real_vision = int(target_record["metrics"]["real_vision_tokens"])
        draft_real_vision = int(draft_record["metrics"]["real_vision_tokens"])
        draft_output_tokens = int(draft_record["metrics"]["output_tokens_including_eos"])
        measured_baseline_latency = baseline.get(request_id, {}).get("worker_wall_s")

        for matcher in matchers:
            for block_size in block_sizes:
                simulation = simulate(
                    tokens,
                    flat_draft,
                    matcher,
                    block_size,
                    DEFAULT_BLOCK_COSTS[block_size],
                    args.max_anchor_tokens,
                    args.backtrack_tokens,
                    continuation_index,
                )
                baseline_target_vision_s = target_real_vision / args.vision_tok_per_s
                baseline_target_decode_s = len(tokens) / args.target_decode_tok_per_s
                draft_vision_s = draft_real_vision / args.vision_tok_per_s
                draft_decode_s = draft_output_tokens / args.draft_decode_tok_per_s
                speculative_target_decode_s = (
                    1.0 + simulation["weighted_target_forward_equivalents"]
                ) / args.target_decode_tok_per_s
                baseline_total_s = baseline_target_vision_s + baseline_target_decode_s
                speculative_total_s = (
                    draft_vision_s
                    + draft_decode_s
                    + baseline_target_vision_s
                    + speculative_target_decode_s
                )
                detailed.append(
                    {
                        "request_id": request_id,
                        "page_name": draft_record.get("page_name"),
                        "strategy": draft_record["strategy"],
                        "matcher": matcher,
                        "block_size": block_size,
                        "draft_rows": len(draft_record.get("rows") or []),
                        "measured_b1_worker_wall_s": measured_baseline_latency,
                        "simulation": simulation,
                        "cost_model_s": {
                            "target_vision": baseline_target_vision_s,
                            "baseline_target_decode": baseline_target_decode_s,
                            "draft_vision": draft_vision_s,
                            "draft_decode": draft_decode_s,
                            "speculative_target_decode": speculative_target_decode_s,
                            "baseline_total": baseline_total_s,
                            "speculative_total": speculative_total_s,
                            "total_speedup": (
                                baseline_total_s / speculative_total_s
                                if speculative_total_s
                                else None
                            ),
                        },
                    }
                )

    grouped: defaultdict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in detailed:
        grouped[(row["strategy"], row["matcher"], row["block_size"])].append(row)
    aggregate_rows = [
        {
            "strategy": strategy,
            "matcher": matcher,
            "block_size": block_size,
            "metrics": aggregate(rows),
        }
        for (strategy, matcher, block_size), rows in sorted(grouped.items())
    ]

    baseline_latencies = sorted(
        float(row["measured_b1_worker_wall_s"])
        for row in detailed
        if row["measured_b1_worker_wall_s"] is not None
    )
    baseline_p75 = quantile(baseline_latencies, 0.75)
    p75_request_ids = {
        row["request_id"]
        for row in detailed
        if baseline_p75 is not None
        and row["measured_b1_worker_wall_s"] is not None
        and float(row["measured_b1_worker_wall_s"]) >= baseline_p75
    }
    p75_aggregate = []
    for key, rows in sorted(grouped.items()):
        selected = [row for row in rows if row["request_id"] in p75_request_ids]
        if selected:
            p75_aggregate.append(
                {
                    "strategy": key[0],
                    "matcher": key[1],
                    "block_size": key[2],
                    "metrics": aggregate(selected),
                }
            )

    report = {
        "inputs": {
            "targets": str(args.targets),
            "drafts": str(args.drafts),
            "baseline_records": str(args.baseline_records) if args.baseline_records else None,
            "matched_target_tables": len({row["request_id"] for row in detailed}),
            "draft_records": len(drafts),
        },
        "cost_model": {
            "vision_tok_per_s": args.vision_tok_per_s,
            "draft_decode_tok_per_s": args.draft_decode_tok_per_s,
            "target_decode_tok_per_s": args.target_decode_tok_per_s,
            "block_forward_costs": DEFAULT_BLOCK_COSTS,
            "included": "row draft vision + row draft decode + target vision + weighted target decode",
            "excluded": "CPU, HTTP, text prefill, and runtime implementation overhead",
        },
        "baseline_latency_p75_s": baseline_p75,
        "baseline_p75_request_ids": sorted(p75_request_ids),
        "aggregate": aggregate_rows,
        "p75_plus_aggregate": p75_aggregate,
        "detailed": detailed,
    }
    (args.output_dir / "simulation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(args.output_dir / "report.md", report)
    print(
        f"tables={report['inputs']['matched_target_tables']} "
        f"draft_records={len(drafts)} configs={len(aggregate_rows)} "
        f"output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
