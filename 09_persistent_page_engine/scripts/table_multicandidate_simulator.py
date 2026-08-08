#!/usr/bin/env python3
"""Simulate legal top-K table-draft verification from saved target tokens.

The CPU matcher ranks candidates using only the generated target prefix and
draft metadata.  The simulation then models a single batched target call which
verifies all ranked candidates and commits the candidate with the longest
target-confirmed prefix.  Looking at the future target is restricted to that
verification step; candidate generation never sees future tokens.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

from paddleocr_vl.serving.table_speculative import (
    DraftProposal,
    TableDraftMatcher,
    _preceding_match,
)


@dataclass(frozen=True)
class RankedProposal:
    proposal: DraftProposal
    rank: int
    score: tuple[float, float, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--drafts", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-counts", default="1,2,4,8,16")
    parser.add_argument("--draft-length", type=int, default=16)
    parser.add_argument("--maximum-anchor", type=int, default=64)
    parser.add_argument("--column-weight", type=float, default=0.25)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def target_tokens(record: dict[str, Any]) -> list[int]:
    rows = record.get("rows") or []
    if len(rows) != 1:
        raise ValueError(f"{record.get('request_id')}: expected one whole-table row")
    tokens = [int(token) for token in rows[0].get("token_ids") or ()]
    if not tokens:
        raise ValueError(f"{record.get('request_id')}: target tokens are empty")
    return tokens


def lcp(left: Iterable[int], right: Iterable[int]) -> int:
    matched = 0
    for lhs, rhs in zip(left, right):
        if int(lhs) != int(rhs):
            break
        matched += 1
    return matched


def ranked_candidates(
    matcher: TableDraftMatcher,
    prefix: list[int],
    limit: int,
) -> list[RankedProposal]:
    if not prefix or not matcher.draft or limit <= 0:
        return []

    candidates: list[tuple[tuple[float, float, int, int], DraftProposal]] = []
    if (
        len(prefix) == 1
        and prefix[0] == matcher.ecel_token
        and matcher.draft[0] == matcher.fcel_token
    ):
        proposal = DraftProposal(
            0,
            tuple(matcher.draft[: matcher.block_size]),
            0,
        )
        candidates.append(((0.0, 0.0, 1, 0), proposal))

    usable = [length for length in matcher.index if length <= len(prefix)]
    for indexed_anchor in sorted(usable, reverse=True):
        continuations = matcher.index[indexed_anchor].get(
            tuple(prefix[-indexed_anchor:]),
            (),
        )
        for continuation in continuations:
            if continuation >= len(matcher.draft):
                continue
            tokens = tuple(
                matcher.draft[continuation : continuation + matcher.block_size]
            )
            if not tokens:
                continue
            anchor = _preceding_match(
                prefix,
                matcher.draft,
                continuation,
                matcher.maximum_anchor,
            )
            score = (
                float(anchor),
                matcher._column_score(tokens[0], matcher.metadata[continuation]),
                int(continuation >= matcher.cursor),
                -abs(continuation - matcher.cursor),
            )
            candidates.append(
                (score, DraftProposal(continuation, tokens, anchor))
            )

    # Different draft locations can contain the same token continuation.  They
    # are one target-verification branch, so retain only the best-ranked copy.
    unique: dict[tuple[int, ...], tuple[tuple[float, float, int, int], DraftProposal]] = {}
    for score, proposal in candidates:
        previous = unique.get(proposal.tokens)
        if previous is None or score > previous[0]:
            unique[proposal.tokens] = (score, proposal)
    ordered = sorted(unique.values(), key=lambda item: item[0], reverse=True)
    return [
        RankedProposal(proposal, rank, score)
        for rank, (score, proposal) in enumerate(ordered[:limit])
    ]


def simulate_one(
    target: list[int],
    draft_record: dict[str, Any],
    tokenizer: Any,
    *,
    candidate_count: int,
    draft_length: int,
    maximum_anchor: int,
    column_weight: float,
) -> dict[str, Any]:
    matcher = TableDraftMatcher(
        draft_record,
        tokenizer,
        eos_token_id=target[-1],
        block_size=draft_length,
        maximum_anchor=maximum_anchor,
        column_weight=column_weight,
    )
    position = 1
    matcher.start(target[0])
    calls = 0
    speculative_calls = 0
    fallback_calls = 0
    accepted = 0
    proposed = 0
    candidate_branches = 0
    winning_ranks: Counter[int] = Counter()
    accept_lengths: list[int] = []
    candidate_counts: list[int] = []

    while position < len(target):
        proposals = ranked_candidates(
            matcher,
            target[:position],
            candidate_count,
        )
        calls += 1
        if not proposals:
            fallback_calls += 1
            emitted = [target[position]]
            matcher.commit(None, accepted_draft_tokens=0, emitted_tokens=emitted)
            position += 1
            continue

        speculative_calls += 1
        candidate_counts.append(len(proposals))
        candidate_branches += len(proposals)
        proposed += sum(len(item.proposal.tokens) for item in proposals)
        scored = [
            (lcp(item.proposal.tokens, target[position:]), -item.rank, item)
            for item in proposals
        ]
        accepted_here, _negative_rank, winner = max(scored, key=lambda item: item[:2])
        accepted += accepted_here
        accept_lengths.append(accepted_here)
        winning_ranks[winner.rank] += 1
        emitted = list(winner.proposal.tokens[:accepted_here])
        if position + accepted_here < len(target):
            emitted.append(target[position + accepted_here])
        matcher.commit(
            winner.proposal,
            accepted_draft_tokens=accepted_here,
            emitted_tokens=emitted,
        )
        position += len(emitted)

    baseline = len(target) - 1
    return {
        "baseline_decode_iterations": baseline,
        "target_calls": calls,
        "speculative_calls": speculative_calls,
        "fallback_calls": fallback_calls,
        "accepted_draft_tokens": accepted,
        "accepted_tokens_per_speculative_call": (
            accepted / speculative_calls if speculative_calls else 0.0
        ),
        "target_tokens_per_call": baseline / calls if calls else None,
        "candidate_branches": candidate_branches,
        "mean_candidates_per_speculative_call": (
            candidate_branches / speculative_calls if speculative_calls else 0.0
        ),
        "mean_accept_length": statistics.mean(accept_lengths) if accept_lengths else 0.0,
        "winning_ranks": dict(sorted(winning_ranks.items())),
        "mean_proposed_tokens_per_call": proposed / calls if calls else 0.0,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = sum(row["simulation"]["baseline_decode_iterations"] for row in rows)
    calls = sum(row["simulation"]["target_calls"] for row in rows)
    speculative = sum(row["simulation"]["speculative_calls"] for row in rows)
    accepted = sum(row["simulation"]["accepted_draft_tokens"] for row in rows)
    branches = sum(row["simulation"]["candidate_branches"] for row in rows)
    return {
        "tables": len(rows),
        "baseline_decode_iterations": baseline,
        "target_calls": calls,
        "speculative_calls": speculative,
        "fallback_calls": sum(row["simulation"]["fallback_calls"] for row in rows),
        "accepted_draft_tokens": accepted,
        "accepted_tokens_per_speculative_call": accepted / speculative if speculative else 0.0,
        "target_tokens_per_call": baseline / calls if calls else None,
        "target_call_reduction": baseline / calls if calls else None,
        "candidate_branches": branches,
        "mean_candidates_per_speculative_call": branches / speculative if speculative else 0.0,
    }


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        local_files_only=True,
    )
    targets = {
        record["request_id"]: record for record in read_jsonl(args.targets)
    }
    drafts = {
        record["request_id"]: record for record in read_jsonl(args.drafts)
    }
    counts = [int(value) for value in args.candidate_counts.split(",") if value]
    detailed: list[dict[str, Any]] = []
    for request_id in sorted(set(targets) & set(drafts)):
        target = target_tokens(targets[request_id])
        for count in counts:
            detailed.append(
                {
                    "request_id": request_id,
                    "candidate_count": count,
                    "simulation": simulate_one(
                        target,
                        drafts[request_id],
                        tokenizer,
                        candidate_count=count,
                        draft_length=args.draft_length,
                        maximum_anchor=args.maximum_anchor,
                        column_weight=args.column_weight,
                    ),
                }
            )

    result = {
        "configuration": {
            "targets": str(args.targets),
            "drafts": str(args.drafts),
            "draft_length": args.draft_length,
            "maximum_anchor": args.maximum_anchor,
            "column_weight": args.column_weight,
            "candidate_counts": counts,
        },
        "aggregate": {
            str(count): aggregate(
                [row for row in detailed if row["candidate_count"] == count]
            )
            for count in counts
        },
        "detailed": detailed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for count in counts:
        metrics = result["aggregate"][str(count)]
        print(
            f"K={count:2d} calls={metrics['target_calls']:6d} "
            f"reduction={metrics['target_call_reduction']:.3f}x "
            f"accepted/spec={metrics['accepted_tokens_per_speculative_call']:.3f} "
            f"mean_candidates={metrics['mean_candidates_per_speculative_call']:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
