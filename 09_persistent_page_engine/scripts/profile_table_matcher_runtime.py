#!/usr/bin/env python3
"""Profile the production table matcher with saved native token-ID streams.

This replays the ordinary adaptive-K verifier policy entirely on CPU. It never
decodes or re-encodes generated text. The saved target and draft token IDs are
the only generation data used by the replay.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
from pathlib import Path
import pstats
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paddleocr_vl.serving.table_speculative import TableDraftMatcher  # noqa: E402


DEFAULT_REQUEST_IDS = (
    "page_000263_table_box_id_7",
    "page_000271_table_box_id_1",
    "page_000277_table_box_id_1",
    "page_000279_table_box_id_0",
    "page_000279_table_box-fy04hrwa",
    "page_000288_table_box_id_1",
    "page_000290_table_box_id_1",
    "page_001595_table_box_id_1",
)
K_VALUES = (7, 15, 31, 63)


class PaddleTableTokenIds:
    """Minimal tokenizer interface for the fixed PaddleOCR-VL table tokens."""

    TOKEN_IDS = {
        "<ecel>": 101308,
        "<fcel>": 101309,
        "<xcel>": 101310,
        "<lcel>": 101311,
        "<ucel>": 101312,
        "<nl>": 101313,
    }

    def token_to_id(self, token: str) -> int | None:
        return self.TOKEN_IDS.get(token)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--drafts", type=Path, required=True)
    parser.add_argument("--request-id", action="append", default=[])
    parser.add_argument("--eos-token-id", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--profile", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def target_tokens(record: dict[str, Any]) -> list[int]:
    speculative = record.get("speculative") or {}
    values = speculative.get("token_ids") or record.get("token_ids")
    if not values:
        raise ValueError(f"target record has no native token IDs: {record.get('request_id')}")
    return [int(value) for value in values]


def next_k(current: int, fully_accepted: bool) -> int:
    index = K_VALUES.index(int(current))
    if fully_accepted:
        return K_VALUES[min(len(K_VALUES) - 1, index + 1)]
    return K_VALUES[max(0, index - 1)]


def replay_one(
    target_record: dict[str, Any],
    draft_record: dict[str, Any],
    *,
    eos_token_id: int,
) -> dict[str, int]:
    target = target_tokens(target_record)
    matcher = TableDraftMatcher(
        draft_record,
        PaddleTableTokenIds(),
        eos_token_id=eos_token_id,
        block_size=15,
    )
    prefix = [target[0]]
    matcher.start(prefix[0])
    policy_k = 15
    cache_length = 4096
    cache_position = int(target_record["input_tokens"])
    calls = 0
    fallback_calls = 0
    speculative_calls = 0
    proposed = 0
    accepted = 0

    while len(prefix) < len(target):
        usable = [
            value
            for value in K_VALUES
            if value <= policy_k and cache_position + value + 1 <= cache_length
        ]
        effective_k = max(usable) if usable else None
        if effective_k is not None:
            matcher.block_size = effective_k
        proposal = matcher.propose(prefix)
        calls += 1
        if effective_k is None or proposal is None or not proposal.tokens:
            emitted = [target[len(prefix)]]
            matcher.commit(None, accepted_draft_tokens=0, emitted_tokens=emitted)
            prefix.extend(emitted)
            cache_position += 1
            fallback_calls += 1
            continue

        speculative_calls += 1
        proposed += len(proposal.tokens)
        target_suffix = target[len(prefix) :]
        accepted_here = 0
        for draft_token, target_token in zip(proposal.tokens, target_suffix):
            if int(draft_token) != int(target_token):
                break
            accepted_here += 1
        accepted += accepted_here
        fully_accepted = accepted_here == len(proposal.tokens)
        emitted = list(proposal.tokens[:accepted_here])
        correction_index = len(prefix) + accepted_here
        if correction_index < len(target):
            emitted.append(target[correction_index])
        matcher.commit(
            proposal,
            accepted_draft_tokens=accepted_here,
            emitted_tokens=emitted,
        )
        prefix.extend(emitted)
        cache_position += len(emitted)
        policy_k = next_k(effective_k, fully_accepted)

    if prefix != target:
        raise AssertionError(f"native-ID replay diverged for {target_record['request_id']}")
    return {
        "target_calls": calls,
        "fallback_calls": fallback_calls,
        "speculative_calls": speculative_calls,
        "proposed_draft_tokens": proposed,
        "accepted_draft_tokens": accepted,
    }


def run_replay(
    selected: list[dict[str, Any]],
    drafts: dict[str, dict[str, Any]],
    *,
    eos_token_id: int,
) -> list[dict[str, int]]:
    return [
        replay_one(row, drafts[str(row["request_id"])], eos_token_id=eos_token_id)
        for row in selected
    ]


def main() -> None:
    args = parse_args()
    targets = read_jsonl(args.targets)
    drafts = {str(row["request_id"]): row for row in read_jsonl(args.drafts)}
    wanted = tuple(args.request_id) if args.request_id else DEFAULT_REQUEST_IDS
    by_id = {str(row["request_id"]): row for row in targets}
    selected = [by_id[request_id] for request_id in wanted]

    reference = run_replay(selected, drafts, eos_token_id=args.eos_token_id)
    for row, measured in zip(selected, reference):
        saved = row.get("speculative") or {}
        comparable = {
            key: int(saved[key])
            for key in measured
            if saved.get(key) is not None
        }
        if comparable and any(measured[key] != value for key, value in comparable.items()):
            raise AssertionError(
                f"saved-call replay mismatch for {row['request_id']}: "
                f"measured={measured} saved={comparable}"
            )

    durations: list[float] = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        run_replay(selected, drafts, eos_token_id=args.eos_token_id)
        durations.append(time.perf_counter() - started)
    calls = sum(row["target_calls"] for row in reference)
    print(
        json.dumps(
            {
                "tables": len(selected),
                "calls": calls,
                "repeats": args.repeats,
                "replay_ms": [round(value * 1e3, 3) for value in durations],
                "best_us_per_call": min(durations) * 1e6 / calls,
                "native_id_exact": True,
                "saved_counts_exact": True,
            },
            indent=2,
        )
    )

    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        run_replay(selected, drafts, eos_token_id=args.eos_token_id)
        profiler.disable()
        buffer = io.StringIO()
        pstats.Stats(profiler, stream=buffer).strip_dirs().sort_stats("cumtime").print_stats(30)
        print(buffer.getvalue())


if __name__ == "__main__":
    main()
