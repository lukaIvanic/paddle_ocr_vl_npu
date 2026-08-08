#!/usr/bin/env python3
"""Compare legal CPU-only matchers for precomputed table-row drafts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Iterable
import unicodedata

from table_speculative_simulator import (
    DEFAULT_BLOCK_COSTS,
    MatcherState,
    build_continuation_index,
    lcp,
    oracle_start_matches,
    preceding_match,
    simulate as baseline_simulate,
    suffix_candidate,
    target_tokens,
)


@dataclass(frozen=True)
class PositionMeta:
    band: int
    logical_row: int
    column: int
    row_width: int


@dataclass(frozen=True)
class Proposal:
    start: int
    tokens: tuple[int, ...]
    score: float
    anchor_tokens: int = 0


@dataclass
class TargetStructure:
    row: int = 0
    column: int = -1
    width_counts: Counter[int] = field(default_factory=Counter)
    modal_width: int | None = None
    modal_width_count: int = 0

    @property
    def width(self) -> int | None:
        return self.modal_width


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--drafts", type=Path, required=True)
    parser.add_argument("--baseline-records", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--beam-widths", default="16,32,64")
    parser.add_argument("--column-weights", default="1,2,4")
    parser.add_argument("--hybrid-thresholds", default="2,4,8")
    parser.add_argument("--maximum-anchor", type=int, default=64)
    parser.add_argument("--backtrack", type=int, default=8)
    parser.add_argument("--minimum-latency-s", type=float, default=0.0)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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
        return {key: None for key in ("min", "mean", "p50", "p75", "p90", "p99", "max")}
    return {
        "min": min(values),
        "mean": statistics.mean(values),
        "p50": quantile(values, 0.50),
        "p75": quantile(values, 0.75),
        "p90": quantile(values, 0.90),
        "p99": quantile(values, 0.99),
        "max": max(values),
    }


def normalize_piece(piece: str) -> str:
    return unicodedata.normalize("NFKC", piece.replace("▁", " ")).strip().lower()


def rows_with_metadata(
    record: dict[str, Any],
    eos_token: int,
    cell_tokens: set[int],
    newline_token: int,
) -> tuple[list[int], list[PositionMeta], list[int]]:
    flat: list[int] = []
    metadata: list[PositionMeta] = []
    band_starts: list[int] = []
    rows = sorted(record.get("rows") or [], key=lambda item: item["row_index"])
    for band_index, row_record in enumerate(rows):
        band_starts.append(len(flat))
        tokens = [int(value) for value in row_record.get("token_ids") or ()]
        if tokens and tokens[-1] == eos_token:
            tokens.pop()
        logical_rows: list[list[int]] = [[]]
        for token in tokens:
            logical_rows[-1].append(token)
            if token == newline_token:
                logical_rows.append([])
        logical_rows = [row for row in logical_rows if row]
        for logical_index, logical_tokens in enumerate(logical_rows):
            width = sum(token in cell_tokens for token in logical_tokens)
            column = -1
            for token in logical_tokens:
                if token in cell_tokens:
                    column += 1
                flat.append(token)
                metadata.append(PositionMeta(band_index, logical_index, column, width))
    return flat, metadata, band_starts


def observe_structure(
    state: TargetStructure,
    token: int,
    cell_tokens: set[int],
    newline_token: int,
) -> None:
    if token in cell_tokens:
        state.column += 1
    if token == newline_token:
        completed_width = state.column + 1
        state.width_counts[completed_width] += 1
        count = state.width_counts[completed_width]
        if count > state.modal_width_count:
            state.modal_width = completed_width
            state.modal_width_count = count
        state.row += 1
        state.column = -1


def expected_column(state: TargetStructure, next_token: int, cell_tokens: set[int]) -> int:
    return state.column + 1 if next_token in cell_tokens else state.column


def column_score(
    state: TargetStructure,
    next_token: int,
    meta: PositionMeta,
    cell_tokens: set[int],
    weight: float,
    patch: bool,
) -> float:
    expected = expected_column(state, next_token, cell_tokens)
    score = weight if expected == meta.column else -weight
    target_width = state.width
    if target_width is None:
        return score
    width_delta = meta.row_width - target_width
    if width_delta == 0:
        return score + weight
    if not patch or abs(width_delta) > 2:
        return score - weight
    # A bounded insertion/deletion patch can shift every later column by at
    # most the row-width difference. This is matcher metadata only.
    lower = max(-1, meta.column - max(0, width_delta))
    upper = meta.column + max(0, -width_delta)
    if lower <= expected <= upper:
        return score + weight * (0.75 - 0.25 * abs(width_delta))
    return score - weight * 0.5


def exact_pool_candidate(
    prefix: list[int],
    draft: list[int],
    metadata: list[PositionMeta],
    continuation_index: dict[int, dict[tuple[int, ...], list[int]]],
    cursor: int,
    block_size: int,
    maximum_anchor: int,
    target_structure: TargetStructure,
    cell_tokens: set[int],
    column_weight: float,
    patch: bool,
) -> Proposal | None:
    usable_lengths = [length for length in continuation_index if length <= len(prefix)]
    for indexed_anchor in sorted(usable_lengths, reverse=True):
        positions = continuation_index[indexed_anchor].get(
            tuple(prefix[-indexed_anchor:]), ()
        )
        best: tuple[tuple[float, float, int, int], Proposal] | None = None
        for continuation in positions:
            if continuation >= len(draft):
                continue
            tokens = tuple(draft[continuation : continuation + block_size])
            if not tokens:
                continue
            anchor = preceding_match(prefix, draft, continuation, maximum_anchor)
            structure_score = column_score(
                target_structure,
                tokens[0],
                metadata[continuation],
                cell_tokens,
                column_weight,
                patch,
            )
            forward = int(continuation >= cursor)
            distance = abs(continuation - cursor)
            # Exact-prefix length remains authoritative. Structural information
            # only breaks ties between candidates with the same exact anchor.
            score = (float(anchor), structure_score, forward, -distance)
            proposal = Proposal(continuation, tokens, structure_score, anchor)
            if best is None or score > best[0]:
                best = (score, proposal)
        if best is not None:
            return best[1]
    return None


class BeamMatcher:
    def __init__(
        self,
        draft: list[int],
        metadata: list[PositionMeta],
        token_keys: list[str],
        token_frequency: Counter[int],
        normalized_index: dict[str, list[int]],
        beam_width: int,
        block_size: int,
        target_structure: TargetStructure,
        cell_tokens: set[int],
        column_weight: float,
        patch: bool,
    ) -> None:
        self.draft = draft
        self.metadata = metadata
        self.token_keys = token_keys
        self.token_frequency = token_frequency
        self.normalized_index = normalized_index
        self.beam_width = beam_width
        self.block_size = block_size
        self.target_structure = target_structure
        self.cell_tokens = cell_tokens
        self.column_weight = column_weight
        self.patch = patch
        self.states: dict[int, float] = {}

    def token_weight(self, token: int) -> float:
        frequency = self.token_frequency[token]
        return min(5.0, 1.0 + math.log((len(self.draft) + 1) / (frequency + 1)))

    def observe(self, token: int, key: str) -> None:
        candidates: dict[int, float] = {}

        def keep(position: int, score: float) -> None:
            if 0 <= position <= len(self.draft) and score > candidates.get(position, -math.inf):
                candidates[position] = score

        for position, old_score in self.states.items():
            decayed = old_score * 0.92
            keep(position, decayed - 1.25)  # target insertion
            for skipped in range(4):
                draft_position = position + skipped
                if draft_position >= len(self.draft):
                    break
                draft_token = self.draft[draft_position]
                if draft_token == token:
                    delta = self.token_weight(token)
                elif self.token_keys[draft_position] == key and key:
                    delta = 0.65 * self.token_weight(token)
                else:
                    delta = -1.0
                keep(draft_position + 1, decayed + delta - 0.65 * skipped)

        for draft_position in self.normalized_index.get(key, ()):
            draft_token = self.draft[draft_position]
            delta = self.token_weight(draft_token) * (1.0 if draft_token == token else 0.65)
            keep(draft_position + 1, delta)

        ranked = sorted(candidates.items(), key=lambda item: item[1], reverse=True)
        # Preserve some positional diversity instead of allowing one repeated
        # numeric region to occupy the full beam.
        per_band: Counter[int] = Counter()
        selected: dict[int, float] = {}
        band_limit = max(2, self.beam_width // 4)
        for position, score in ranked:
            meta_position = min(position, len(self.metadata) - 1)
            band = self.metadata[meta_position].band if self.metadata else 0
            if per_band[band] >= band_limit:
                continue
            selected[position] = score
            per_band[band] += 1
            if len(selected) >= self.beam_width:
                break
        self.states = selected

    def propose(self) -> Proposal | None:
        best: Proposal | None = None
        for position, alignment_score in self.states.items():
            if position >= len(self.draft):
                continue
            tokens = tuple(self.draft[position : position + self.block_size])
            if not tokens:
                continue
            score = alignment_score + column_score(
                self.target_structure,
                tokens[0],
                self.metadata[position],
                self.cell_tokens,
                self.column_weight,
                self.patch,
            )
            if best is None or score > best.score:
                best = Proposal(position, tokens, score, 0)
        return best


def simulate_custom(
    target: list[int],
    draft: list[int],
    metadata: list[PositionMeta],
    tokenizer: Any,
    matcher: str,
    block_size: int,
    block_cost: float,
    maximum_anchor: int,
    backtrack: int,
    cell_tokens: set[int],
    newline_token: int,
    beam_width: int = 32,
    column_weight: float = 0.0,
    patch: bool = False,
    continuation_index: dict[int, dict[tuple[int, ...], list[int]]] | None = None,
) -> dict[str, Any]:
    index = continuation_index or build_continuation_index(draft, maximum_anchor)
    structure = TargetStructure()
    cursor = 0
    position = 1
    calls = 0
    speculative_calls = 0
    fallback_calls = 0
    accepted_tokens = 0
    proposed_tokens = 0
    weighted = 0.0
    accept_lengths: list[float] = []
    uses_beam = matcher.startswith("beam") or matcher.startswith("hybrid")
    keys: list[str] = []
    target_keys: list[str] = []
    normalized_index: defaultdict[str, list[int]] = defaultdict(list)
    frequency: Counter[int] = Counter()
    if uses_beam:
        keys = [normalize_piece(piece) for piece in tokenizer.convert_ids_to_tokens(draft)]
        for draft_position, key in enumerate(keys):
            if key:
                normalized_index[key].append(draft_position)
        target_keys = [
            normalize_piece(piece) for piece in tokenizer.convert_ids_to_tokens(target)
        ]
        frequency = Counter(draft)
    beam = (
        BeamMatcher(
            draft,
            metadata,
            keys,
            frequency,
            dict(normalized_index),
            beam_width,
            block_size,
            structure,
            cell_tokens,
            column_weight,
            patch,
        )
        if uses_beam
        else None
    )
    exact_state = MatcherState()
    observe_structure(structure, target[0], cell_tokens, newline_token)
    if beam is not None:
        beam.observe(target[0], target_keys[0])

    while position < len(target):
        if matcher.startswith("beam"):
            proposal = beam.propose() if beam is not None else None
        elif matcher.startswith("hybrid"):
            threshold = int(matcher.rsplit("a", 1)[1])
            exact = suffix_candidate(
                target[:position],
                draft,
                index,
                exact_state,
                block_size,
                1,
                maximum_anchor,
                backtrack,
                False,
            )
            exact_proposal = (
                Proposal(exact.start, exact.tokens, float(exact.anchor_tokens), exact.anchor_tokens)
                if exact is not None
                else None
            )
            proposal = (
                exact_proposal
                if exact_proposal is not None and exact_proposal.anchor_tokens >= threshold
                else (beam.propose() if beam is not None else exact_proposal)
            )
        else:
            proposal = exact_pool_candidate(
                target[:position],
                draft,
                metadata,
                index,
                cursor,
                block_size,
                maximum_anchor,
                structure,
                cell_tokens,
                column_weight,
                patch,
            )
        calls += 1
        if proposal is None:
            accepted = 0
            fallback_calls += 1
            weighted += 1.0
        else:
            accepted = lcp(proposal.tokens, target[position:])
            speculative_calls += 1
            proposed_tokens += len(proposal.tokens)
            accepted_tokens += accepted
            accept_lengths.append(float(accepted))
            weighted += block_cost
            if accepted:
                cursor = proposal.start + accepted
                exact_state.cursor = proposal.start + accepted
        correction_matches_draft = (
            proposal is not None
            and position + accepted < len(target)
            and proposal.start + accepted < len(draft)
            and target[position + accepted] == draft[proposal.start + accepted]
        )
        if correction_matches_draft:
            cursor = proposal.start + accepted + 1
            exact_state.cursor = cursor
        emitted = min(len(target) - position, accepted + 1)
        for target_position in range(position, position + emitted):
            token = target[target_position]
            observe_structure(structure, token, cell_tokens, newline_token)
            if beam is not None:
                beam.observe(token, target_keys[target_position])
        position += emitted

    baseline_iterations = len(target) - 1
    return {
        "target_tokens_including_eos": len(target),
        "draft_tokens_without_row_eos": len(draft),
        "baseline_decode_iterations": baseline_iterations,
        "target_calls": calls,
        "speculative_calls": speculative_calls,
        "fallback_calls": fallback_calls,
        "accepted_draft_tokens": accepted_tokens,
        "proposed_draft_tokens": proposed_tokens,
        "accepted_fraction_of_proposed": accepted_tokens / proposed_tokens if proposed_tokens else 0.0,
        "accepted_coverage_of_target_decode": accepted_tokens / baseline_iterations if baseline_iterations else 0.0,
        "accepted_tokens_per_speculative_call": accepted_tokens / speculative_calls if speculative_calls else 0.0,
        "target_tokens_per_call": baseline_iterations / calls if calls else None,
        "weighted_target_forward_equivalents": weighted,
        "ideal_target_decode_speedup": baseline_iterations / weighted if weighted else None,
        "accept_length": distribution(accept_lengths),
    }


def aggregate(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    simulations = [row["simulation"] for row in materialized]
    baseline = sum(row["baseline_decode_iterations"] for row in simulations)
    calls = sum(row["target_calls"] for row in simulations)
    speculative = sum(row["speculative_calls"] for row in simulations)
    accepted = sum(row["accepted_draft_tokens"] for row in simulations)
    proposed = sum(row["proposed_draft_tokens"] for row in simulations)
    weighted = sum(row["weighted_target_forward_equivalents"] for row in simulations)
    matcher_cpu_s = sum(row["matcher_cpu_s"] for row in materialized)
    return {
        "tables": len(simulations),
        "baseline_decode_iterations": baseline,
        "target_calls": calls,
        "speculative_calls": speculative,
        "fallback_calls": sum(row["fallback_calls"] for row in simulations),
        "accepted_draft_tokens": accepted,
        "proposed_draft_tokens": proposed,
        "accepted_fraction_of_proposed": accepted / proposed if proposed else 0.0,
        "accepted_coverage_of_target_decode": accepted / baseline if baseline else 0.0,
        "accepted_tokens_per_speculative_call": accepted / speculative if speculative else 0.0,
        "target_tokens_per_call": baseline / calls if calls else None,
        "weighted_target_forward_equivalents": weighted,
        "ideal_target_decode_speedup": baseline / weighted if weighted else None,
        "matcher_cpu_s": matcher_cpu_s,
        "matcher_cpu_ms_per_table": 1_000.0 * matcher_cpu_s / len(materialized) if materialized else None,
        "matcher_cpu_us_per_target_call": 1_000_000.0 * matcher_cpu_s / calls if calls else None,
        "zero_accept_calls": sum(
            round(row["speculative_calls"] * (row["accept_length"]["p50"] == 0))
            for row in simulations
            if row.get("accept_length", {}).get("p50") is not None
        ),
    }


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer

    if args.block_size not in DEFAULT_BLOCK_COSTS:
        raise ValueError(f"no block cost for K={args.block_size}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    cell_tokens = {
        tokenizer.convert_tokens_to_ids(token)
        for token in ("<fcel>", "<ecel>", "<lcel>", "<ucel>", "<xcel>")
    }
    newline_token = tokenizer.convert_tokens_to_ids("<nl>")
    targets = {row["request_id"]: row for row in read_jsonl(args.targets)}
    drafts = {row["request_id"]: row for row in read_jsonl(args.drafts)}
    latencies = {
        row["request_id"]: float(row["worker_wall_s"])
        for row in read_jsonl(args.baseline_records)
    }
    beam_widths = [int(value) for value in args.beam_widths.split(",") if value]
    column_weights = [float(value) for value in args.column_weights.split(",") if value]
    hybrid_thresholds = [int(value) for value in args.hybrid_thresholds.split(",") if value]
    configurations: list[dict[str, Any]] = [
        {"name": "current", "kind": "baseline", "baseline_matcher": "suffix_global_a1"},
        {"name": "reversible", "kind": "baseline", "baseline_matcher": "suffix_global_reversible_a1"},
        {"name": "oracle", "kind": "baseline", "baseline_matcher": "oracle_global"},
    ]
    for weight in column_weights:
        configurations.extend(
            [
                {"name": f"column_w{weight:g}", "kind": "column", "column_weight": weight, "patch": False},
                {"name": f"column_patch_w{weight:g}", "kind": "column", "column_weight": weight, "patch": True},
            ]
        )
    for width in beam_widths:
        configurations.append(
            {"name": f"beam_b{width}", "kind": "beam", "beam_width": width, "column_weight": 0.0, "patch": False}
        )
        for weight in column_weights:
            configurations.extend(
                [
                    {"name": f"beam_column_b{width}_w{weight:g}", "kind": "beam", "beam_width": width, "column_weight": weight, "patch": False},
                    {"name": f"beam_patch_b{width}_w{weight:g}", "kind": "beam", "beam_width": width, "column_weight": weight, "patch": True},
                ]
            )
        for threshold in hybrid_thresholds:
            configurations.extend(
                [
                    {
                        "name": f"hybrid_b{width}_a{threshold}",
                        "kind": f"hybrid_a{threshold}",
                        "beam_width": width,
                        "column_weight": 0.0,
                        "patch": False,
                    },
                    {
                        "name": f"hybrid_patch_b{width}_a{threshold}",
                        "kind": f"hybrid_a{threshold}",
                        "beam_width": width,
                        "column_weight": min(column_weights, default=0.0),
                        "patch": True,
                    },
                ]
            )

    detailed: list[dict[str, Any]] = []
    request_ids = sorted(
        request_id
        for request_id in set(targets) & set(drafts)
        if latencies.get(request_id, 0.0) >= args.minimum_latency_s
    )
    if args.limit is not None:
        request_ids = request_ids[: args.limit]
    for table_index, request_id in enumerate(request_ids, start=1):
        target = target_tokens(targets[request_id])
        draft, metadata, _band_starts = rows_with_metadata(
            drafts[request_id], target[-1], cell_tokens, newline_token
        )
        baseline_index = build_continuation_index(draft, args.maximum_anchor)
        oracle_matches = oracle_start_matches(target, draft)
        for config in configurations:
            matcher_started = time.perf_counter()
            if config["kind"] == "baseline":
                simulation = baseline_simulate(
                    target,
                    draft,
                    config["baseline_matcher"],
                    args.block_size,
                    DEFAULT_BLOCK_COSTS[args.block_size],
                    args.maximum_anchor,
                    args.backtrack,
                    baseline_index,
                    oracle_matches,
                )
            else:
                simulation = simulate_custom(
                    target,
                    draft,
                    metadata,
                    tokenizer,
                    config["kind"],
                    args.block_size,
                    DEFAULT_BLOCK_COSTS[args.block_size],
                    args.maximum_anchor,
                    args.backtrack,
                    cell_tokens,
                    newline_token,
                    int(config.get("beam_width", 32)),
                    float(config.get("column_weight", 0.0)),
                    bool(config.get("patch", False)),
                    baseline_index,
                )
            matcher_cpu_s = time.perf_counter() - matcher_started
            detailed.append(
                {
                    "request_id": request_id,
                    "page_name": drafts[request_id].get("page_name"),
                    "matcher": config["name"],
                    "measured_b1_worker_wall_s": latencies.get(request_id),
                    "matcher_cpu_s": matcher_cpu_s,
                    "simulation": simulation,
                }
            )
        if table_index % 50 == 0 or table_index == len(request_ids):
            print(f"progress tables={table_index}/{len(request_ids)}", flush=True)

    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in detailed:
        grouped[row["matcher"]].append(row)
    latency_values = sorted(latencies.values())
    p75 = quantile(latency_values, 0.75)
    cohorts = {
        "all": set(request_ids),
        "latency_gt_1s": {request_id for request_id in request_ids if latencies.get(request_id, 0.0) > 1.0},
        "latency_p75_plus": {request_id for request_id in request_ids if p75 is not None and latencies.get(request_id, 0.0) >= p75},
    }
    results: dict[str, dict[str, Any]] = {}
    for cohort_name, cohort_ids in cohorts.items():
        results[cohort_name] = {
            matcher: aggregate(row for row in rows if row["request_id"] in cohort_ids)
            for matcher, rows in grouped.items()
        }
    report = {
        "inputs": {
            "targets": str(args.targets),
            "drafts": str(args.drafts),
            "baseline_records": str(args.baseline_records),
            "tables": len(request_ids),
            "block_size": args.block_size,
            "beam_widths": beam_widths,
            "column_weights": column_weights,
            "hybrid_thresholds": hybrid_thresholds,
        },
        "cohorts": {name: len(ids) for name, ids in cohorts.items()},
        "results": results,
        "detailed": detailed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    lines = [
        "# Table matcher strategy lab",
        "",
        "All matchers use target-prefix tokens only, except the marked oracle upper bound.",
    ]
    for cohort, matchers in results.items():
        lines.extend(
            [
                "",
                f"## {cohort} ({report['cohorts'][cohort]} tables)",
                "",
                "| matcher | accepted/call | coverage | target calls | decode speedup | CPU ms/table | CPU us/call |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for matcher, metrics in sorted(
            matchers.items(), key=lambda item: item[1]["target_calls"]
        ):
            lines.append(
                f"| {matcher} | {metrics['accepted_tokens_per_speculative_call']:.3f} | "
                f"{metrics['accepted_coverage_of_target_decode']:.4f} | {metrics['target_calls']:,} | "
                f"{metrics['ideal_target_decode_speedup']:.3f}x | "
                f"{metrics['matcher_cpu_ms_per_table']:.3f} | "
                f"{metrics['matcher_cpu_us_per_target_call']:.3f} |"
            )
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(f"complete output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
