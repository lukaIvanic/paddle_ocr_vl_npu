#!/usr/bin/env python3
"""Explain the gap between the practical and oracle table-draft matchers."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from typing import Any

from table_speculative_simulator import (
    MatcherState,
    build_continuation_index,
    flatten_drafts,
    lcp,
    oracle_start_matches,
    preceding_match,
    suffix_candidate,
    target_tokens,
)


STRUCTURE_TOKENS = {"<ecel>", "<fcel>", "<xcel>", "<lcel>", "<ucel>", "<nl>"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--drafts", type=Path, required=True)
    parser.add_argument("--baseline-records", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--latency-threshold-s", type=float, default=1.0)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-anchor-tokens", type=int, default=64)
    parser.add_argument("--backtrack-tokens", type=int, default=8)
    parser.add_argument("--manual-tables", type=int, default=10)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def token_kind(piece: str) -> str:
    if piece in STRUCTURE_TOKENS:
        return "table_structure"
    text = piece.replace("▁", " ")
    if not text.strip():
        return "whitespace"
    if re.fullmatch(r"[\d\s.,:+\-/%]+", text):
        return "numeric"
    if any(ch in text for ch in "\\{}_^$=±×÷∑∫√"):
        return "math_or_latex"
    if any("\u4e00" <= ch <= "\u9fff" for ch in text):
        return "cjk"
    if any(ch.isalpha() for ch in text):
        return "text"
    return "punctuation"


def row_index(row_for_token: list[int], position: int) -> int | None:
    return row_for_token[position] if 0 <= position < len(row_for_token) else None


def indexed_anchor_details(
    prefix: list[int],
    index: dict[int, dict[tuple[int, ...], list[int]]],
) -> tuple[int, int]:
    for length in sorted((value for value in index if value <= len(prefix)), reverse=True):
        positions = index[length].get(tuple(prefix[-length:]), ())
        if positions:
            return length, len(positions)
    return 0, 0


def compact_decode(tokenizer: Any, tokens: list[int] | tuple[int, ...], limit: int = 180) -> str:
    text = tokenizer.decode(list(tokens), skip_special_tokens=False)
    text = text.replace("\n", "\\n")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def edit_distance(left: list[int] | tuple[int, ...], right: list[int] | tuple[int, ...]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_token in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_token in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + int(left_token != right_token),
                )
            )
        previous = current
    return previous[-1]


def magnitude_bucket(value: int) -> str:
    if value == 0:
        return "0"
    if value == 1:
        return "1"
    if value <= 3:
        return "2-3"
    if value <= 7:
        return "4-7"
    if value <= 15:
        return "8-15"
    if value <= 31:
        return "16-31"
    if value <= 63:
        return "32-63"
    return "64+"


def signed_progress_bucket(value: float) -> str:
    if value < -0.25:
        return "cursor_>25%_behind"
    if value < -0.10:
        return "cursor_10-25%_behind"
    if value <= 0.10:
        return "cursor_within_10%"
    if value <= 0.25:
        return "cursor_10-25%_ahead"
    return "cursor_>25%_ahead"


def analyze_table(
    target_record: dict[str, Any],
    draft_record: dict[str, Any],
    tokenizer: Any,
    block_size: int,
    max_anchor: int,
    backtrack: int,
) -> dict[str, Any]:
    target = target_tokens(target_record)
    draft, row_for_token = flatten_drafts(draft_record, target[-1])
    index = build_continuation_index(draft, max_anchor)
    oracle = oracle_start_matches(target, draft)
    pieces = tokenizer.convert_ids_to_tokens(target)
    state = MatcherState()
    position = 1
    events: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    category_events: Counter[str] = Counter()
    category_lost: Counter[str] = Counter()
    selected_rows: Counter[int] = Counter()
    oracle_rows: Counter[int] = Counter()
    position_deciles: Counter[str] = Counter()
    gap_kinds: Counter[str] = Counter()
    ambiguity_histogram: Counter[str] = Counter()
    anchor_relation: Counter[str] = Counter()
    edit_histogram: Counter[str] = Counter()
    accept_histogram: Counter[str] = Counter()
    post_mismatch_recovery: Counter[str] = Counter()
    cursor_progress_lost: Counter[str] = Counter()
    row_direction: Counter[str] = Counter()

    while position < len(target):
        prefix = target[:position]
        candidate = suffix_candidate(
            prefix,
            draft,
            index,
            state,
            block_size,
            1,
            max_anchor,
            backtrack,
            False,
        )
        oracle_length, oracle_start = oracle[position]
        oracle_accept = min(oracle_length, block_size)
        indexed_anchor, ambiguity = indexed_anchor_details(prefix, index)
        if candidate is None:
            practical_accept = 0
            proposed = 0
            selected_start = None
            selected_anchor = 0
        else:
            practical_accept = lcp(candidate.tokens, target[position:])
            proposed = len(candidate.tokens)
            selected_start = candidate.start
            selected_anchor = candidate.anchor_tokens
            selected = row_index(row_for_token, candidate.start)
            if selected is not None:
                selected_rows[selected] += 1
        oracle_anchor = preceding_match(prefix, draft, oracle_start, max_anchor) if oracle_accept else 0
        oracle_row = row_index(row_for_token, oracle_start) if oracle_accept else None
        if oracle_row is not None:
            oracle_rows[oracle_row] += 1
        lost = max(0, oracle_accept - practical_accept)
        totals["calls"] += 1
        totals["practical_accepted"] += practical_accept
        totals["oracle_local_accepted"] += oracle_accept
        totals["proposed"] += proposed
        accept_histogram[magnitude_bucket(practical_accept)] += 1
        totals["oracle_opportunity_calls"] += int(oracle_accept > 0)
        totals["optimal_calls"] += int(practical_accept == oracle_accept)
        totals["gap_calls"] += int(lost > 0)
        totals["lost_tokens"] += lost
        totals["unanchored_gap_calls"] += int(lost > 0 and oracle_anchor == 0)
        totals["unanchored_lost_tokens"] += lost if oracle_anchor == 0 else 0
        totals["anchored_gap_calls"] += int(lost > 0 and oracle_anchor > 0)
        totals["anchored_lost_tokens"] += lost if oracle_anchor > 0 else 0
        totals["different_row_gap_calls"] += int(
            lost > 0 and selected_start is not None and row_index(row_for_token, selected_start) != oracle_row
        )
        totals["ambiguous_gap_calls"] += int(lost > 0 and ambiguity > 1)
        if lost > 0:
            cursor_progress_lost[
                signed_progress_bucket(
                    state.cursor / max(1, len(draft))
                    - position / max(1, len(target))
                )
            ] += lost
            position_deciles[f"{min(9, int(10 * position / max(1, len(target)))) * 10:02d}-{min(100, (min(9, int(10 * position / max(1, len(target)))) + 1) * 10):02d}%"] += lost
            ambiguity_histogram[magnitude_bucket(ambiguity)] += 1
            if candidate is None:
                gap_kinds["no_suffix_candidate"] += 1
            elif oracle_anchor == 0:
                gap_kinds["oracle_has_no_prefix_anchor"] += 1
            elif row_index(row_for_token, candidate.start) != oracle_row:
                gap_kinds["wrong_band"] += 1
            else:
                gap_kinds["wrong_location_within_band"] += 1
            if oracle_anchor > 0:
                if selected_anchor > oracle_anchor:
                    anchor_relation["selected_anchor_longer"] += 1
                elif selected_anchor == oracle_anchor:
                    anchor_relation["anchors_equal"] += 1
                else:
                    anchor_relation["oracle_anchor_longer"] += 1
            if candidate is not None and oracle_row is not None:
                selected_row = row_index(row_for_token, candidate.start)
                if selected_row is not None:
                    if selected_row < oracle_row:
                        row_direction["selected_earlier_band"] += lost
                    elif selected_row > oracle_row:
                        row_direction["selected_later_band"] += lost
                    else:
                        row_direction["selected_same_band"] += lost
            failure_position = min(position + practical_accept, len(target) - 1)
            category = token_kind(pieces[failure_position])
            category_events[category] += 1
            category_lost[category] += lost
            if candidate is not None:
                target_window = target[position : position + len(candidate.tokens)]
                distance = edit_distance(candidate.tokens, target_window)
                edit_histogram[magnitude_bucket(distance)] += 1
                recovery = lcp(
                    candidate.tokens[practical_accept + 1 :],
                    target[position + practical_accept + 1 :],
                )
                post_mismatch_recovery[magnitude_bucket(recovery)] += 1
                totals["one_edit_gap_calls"] += int(distance == 1)
                totals["post_mismatch_recovery_ge4_calls"] += int(recovery >= 4)
            selected_context = (
                compact_decode(tokenizer, draft[selected_start : selected_start + block_size])
                if selected_start is not None
                else "<no candidate>"
            )
            event = {
                "target_position": position,
                "target_fraction": position / max(1, len(target) - 1),
                "practical_accept": practical_accept,
                "oracle_accept": oracle_accept,
                "lost_tokens": lost,
                "failure_token": pieces[failure_position],
                "failure_category": category,
                "indexed_anchor": indexed_anchor,
                "selected_anchor": selected_anchor,
                "oracle_anchor": oracle_anchor,
                "anchor_ambiguity": ambiguity,
                "selected_start": selected_start,
                "oracle_start": oracle_start,
                "selected_row": row_index(row_for_token, selected_start) if selected_start is not None else None,
                "oracle_row": oracle_row,
                "target_context": compact_decode(
                    tokenizer,
                    target[max(0, position - 12) : min(len(target), position + block_size + 8)],
                ),
                "selected_context": selected_context,
                "oracle_context": compact_decode(tokenizer, draft[oracle_start : oracle_start + block_size]),
            }
            events.append(event)
        if candidate is not None:
            correction_matches_draft = (
                position + practical_accept < len(target)
                and candidate.start + practical_accept < len(draft)
                and target[position + practical_accept] == draft[candidate.start + practical_accept]
            )
            if practical_accept:
                state.cursor = max(state.cursor, candidate.start + practical_accept)
            if correction_matches_draft:
                state.cursor = max(state.cursor, candidate.start + practical_accept + 1)
        position += min(len(target) - position, practical_accept + 1)

    return {
        "request_id": target_record["request_id"],
        "page_name": target_record.get("page_name"),
        "target_tokens": len(target),
        "draft_tokens": len(draft),
        "rows": len(draft_record.get("rows") or []),
        "totals": dict(totals),
        "failure_event_categories": dict(category_events),
        "lost_token_categories": dict(category_lost),
        "selected_row_histogram": dict(selected_rows),
        "oracle_row_histogram": dict(oracle_rows),
        "lost_tokens_by_target_decile": dict(position_deciles),
        "gap_kinds": dict(gap_kinds),
        "gap_anchor_ambiguity": dict(ambiguity_histogram),
        "anchored_gap_anchor_relation": dict(anchor_relation),
        "gap_candidate_edit_distance": dict(edit_histogram),
        "practical_accept_length": dict(accept_histogram),
        "post_mismatch_recovery": dict(post_mismatch_recovery),
        "lost_tokens_by_cursor_progress": dict(cursor_progress_lost),
        "lost_tokens_by_selected_vs_oracle_band": dict(row_direction),
        "worst_events": sorted(events, key=lambda row: (row["lost_tokens"], row["oracle_accept"]), reverse=True)[:12],
        "target_text": target_record["rows"][0].get("raw_text") or target_record["rows"][0].get("text"),
        "draft_row_texts": [
            {
                "row_index": row.get("row_index"),
                "row_y": row.get("row_y"),
                "stop_reason": row.get("stop_reason"),
                "text": row.get("raw_text") or row.get("text"),
            }
            for row in draft_record.get("rows") or []
        ],
    }


def pct(numerator: float, denominator: float) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    targets = {row["request_id"]: row for row in read_jsonl(args.targets)}
    drafts = {row["request_id"]: row for row in read_jsonl(args.drafts)}
    latencies = {
        row["request_id"]: float(row["worker_wall_s"])
        for row in read_jsonl(args.baseline_records)
    }
    selected_ids = sorted(
        request_id
        for request_id, latency in latencies.items()
        if latency > args.latency_threshold_s and request_id in targets and request_id in drafts
    )
    tables = [
        analyze_table(
            targets[request_id], drafts[request_id], tokenizer,
            args.block_size, args.max_anchor_tokens, args.backtrack_tokens,
        )
        for request_id in selected_ids
    ]
    total: Counter[str] = Counter()
    event_categories: Counter[str] = Counter()
    lost_categories: Counter[str] = Counter()
    aggregate_histograms = {
        key: Counter()
        for key in (
            "lost_tokens_by_target_decile",
            "gap_kinds",
            "gap_anchor_ambiguity",
            "anchored_gap_anchor_relation",
            "gap_candidate_edit_distance",
            "practical_accept_length",
            "post_mismatch_recovery",
            "lost_tokens_by_cursor_progress",
            "lost_tokens_by_selected_vs_oracle_band",
        )
    }
    for table in tables:
        total.update(table["totals"])
        event_categories.update(table["failure_event_categories"])
        lost_categories.update(table["lost_token_categories"])
        for key, counter in aggregate_histograms.items():
            counter.update(table[key])
        table["measured_b1_worker_wall_s"] = latencies[table["request_id"]]
    ranked = sorted(
        tables,
        key=lambda row: (row["totals"].get("lost_tokens", 0), row["target_tokens"]),
        reverse=True,
    )
    summary = {
        "cohort": {
            "latency_threshold_s": args.latency_threshold_s,
            "tables": len(tables),
            "block_size": args.block_size,
        },
        "totals": dict(total),
        "rates_percent": {
            "calls_locally_optimal": pct(total["optimal_calls"], total["calls"]),
            "gap_calls": pct(total["gap_calls"], total["calls"]),
            "lost_tokens_unanchored": pct(total["unanchored_lost_tokens"], total["lost_tokens"]),
            "lost_tokens_anchored": pct(total["anchored_lost_tokens"], total["lost_tokens"]),
            "gap_calls_with_ambiguous_anchor": pct(total["ambiguous_gap_calls"], total["gap_calls"]),
            "gap_calls_different_row": pct(total["different_row_gap_calls"], total["gap_calls"]),
        },
        "failure_event_categories": dict(event_categories.most_common()),
        "lost_token_categories": dict(lost_categories.most_common()),
        "histograms": {
            key: dict(counter)
            for key, counter in aggregate_histograms.items()
        },
        "worst_tables": [
            {
                "request_id": row["request_id"],
                "page_name": row["page_name"],
                "latency_s": row["measured_b1_worker_wall_s"],
                "target_tokens": row["target_tokens"],
                "draft_tokens": row["draft_tokens"],
                "totals": row["totals"],
                "worst_events": row["worst_events"][:4],
            }
            for row in ranked[: args.manual_tables]
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    with (args.output_dir / "tables.jsonl").open("w", encoding="utf-8") as handle:
        for table in tables:
            handle.write(json.dumps(table, ensure_ascii=False) + "\n")

    lines = [
        "# Table matcher failure analysis",
        "",
        f"Cohort: {len(tables)} tables with measured B1 latency > {args.latency_threshold_s:.3f} s. K={args.block_size}.",
        "",
        "## Headline",
        "",
        f"- Practical calls analyzed: {total['calls']:,}",
        f"- Calls where practical equals the local oracle: {summary['rates_percent']['calls_locally_optimal']:.1f}%",
        f"- Calls with an oracle gap: {summary['rates_percent']['gap_calls']:.1f}%",
        f"- Lost opportunity tokens with no matching prefix anchor: {summary['rates_percent']['lost_tokens_unanchored']:.1f}%",
        f"- Lost opportunity tokens despite a legal prefix anchor: {summary['rates_percent']['lost_tokens_anchored']:.1f}%",
        f"- Gap calls with ambiguous anchors: {summary['rates_percent']['gap_calls_with_ambiguous_anchor']:.1f}%",
        f"- Gap calls selecting a different draft row than the oracle: {summary['rates_percent']['gap_calls_different_row']:.1f}%",
        "",
        "## First divergence category",
        "",
        "| category | gap events | oracle-opportunity tokens lost |",
        "|---|---:|---:|",
    ]
    for category in sorted(set(event_categories) | set(lost_categories), key=lambda key: lost_categories[key], reverse=True):
        lines.append(f"| {category} | {event_categories[category]:,} | {lost_categories[category]:,} |")
    lines.extend(["", "## Worst tables", "", "| request | latency s | target | draft | gap calls | lost opportunity tokens |", "|---|---:|---:|---:|---:|---:|"])
    for row in ranked[: args.manual_tables]:
        lines.append(
            f"| `{row['request_id']}` | {row['measured_b1_worker_wall_s']:.3f} | {row['target_tokens']:,} | "
            f"{row['draft_tokens']:,} | {row['totals'].get('gap_calls', 0):,} | {row['totals'].get('lost_tokens', 0):,} |"
        )
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(f"tables={len(tables)} calls={total['calls']} output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
