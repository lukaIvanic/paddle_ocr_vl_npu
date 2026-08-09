#!/usr/bin/env python3
"""Replay saved B1 table histories and inspect U2 syntax alternatives in top-k."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
REPO_ROOT = EXPERIMENT_ROOT.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))
sys.path.insert(0, str(HERE))

from paddleocr_vl.serving.table_speculative import TableDraftMatcher
from table_spec_decode_lab import (
    DEFAULT_TEXT_BUCKETS,
    DEFAULT_VISION_BUCKETS,
    build_recognizer,
    exact_target_crop,
    read_jsonl,
    request_for,
    target_tokens,
)


DEFAULT_TARGETS = Path(
    "tmp/09_persistent_page_engine/table_spec_full_d1e6d00/"
    "whole/row_ocr_records.jsonl"
)
DEFAULT_DRAFTS = Path(
    "tmp/09_persistent_page_engine/"
    "table_row_full_uniform2_default_178605c/row_ocr_records.jsonl"
)
DEFAULT_REQUEST_IDS = (
    "page_000271_table_box_id_1",
    "page_000500_table_5",
    "page_000106_table_box_id_2",
)

SYNTAX_TEXTS = (
    r"\(",
    r" \(",
    r"\)",
    r"\[",
    r" \[",
    r"\]",
    "$",
    " $",
    "$$",
    " $$",
    " ",
    "  ",
    "\\",
    " \\",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--drafts", type=Path, default=DEFAULT_DRAFTS)
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("/workspace/datasets/OmniDocBench/images"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/workspace/models/PaddleOCR-VL-1.6"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request-id", action="append", default=[])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--draft-length", type=int, default=16)
    parser.add_argument("--cache-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--min-pixels", type=int, default=28224)
    parser.add_argument("--max-pixels", type=int, default=802816)
    parser.add_argument("--vision-buckets", default=DEFAULT_VISION_BUCKETS)
    parser.add_argument("--text-buckets", default=DEFAULT_TEXT_BUCKETS)
    parser.add_argument(
        "--decode-cache-dir",
        type=Path,
        default=REPO_ROOT / ".runtime_cache/09_persistent_page_engine_torchair",
    )
    parser.add_argument(
        "--vision-cache-dir",
        type=Path,
        default=REPO_ROOT
        / ".runtime_cache/09_persistent_page_engine_vision_torchair",
    )
    parser.add_argument(
        "--text-cache-dir",
        type=Path,
        default=REPO_ROOT
        / ".runtime_cache/09_persistent_page_engine_text_torchair",
    )
    return parser.parse_args()


def token_text(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode([int(token_id)], skip_special_tokens=False)


def token_piece(tokenizer: Any, token_id: int) -> str:
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        value = tokenizer.convert_ids_to_tokens(int(token_id))
        return str(value)
    if hasattr(tokenizer, "id_to_token"):
        value = tokenizer.id_to_token(int(token_id))
        return str(value)
    return token_text(tokenizer, token_id)


def token_payload(tokenizer: Any, token_id: int) -> dict[str, Any]:
    return {
        "id": int(token_id),
        "piece": token_piece(tokenizer, token_id),
        "decoded": token_text(tokenizer, token_id),
    }


def encode_one(tokenizer: Any, text: str) -> int:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    token_ids = encoded.ids if hasattr(encoded, "ids") else encoded
    if len(token_ids) != 1:
        raise ValueError(f"expected one token for {text!r}, got {token_ids}")
    return int(token_ids[0])


def syntax_tokens(tokenizer: Any) -> dict[int, list[str]]:
    result: dict[int, list[str]] = {}
    for text in SYNTAX_TEXTS:
        token_id = encode_one(tokenizer, text)
        result.setdefault(token_id, []).append(text)
    return result


def is_syntax_token(
    tokenizer: Any,
    token_id: int | None,
    selected: dict[int, list[str]],
) -> bool:
    if token_id is None:
        return False
    if int(token_id) in selected:
        return True
    decoded = token_text(tokenizer, int(token_id))
    return (
        "\\(" in decoded
        or "\\)" in decoded
        or "\\[" in decoded
        or "\\]" in decoded
        or "$" in decoded
        or decoded != decoded.strip()
    )


def topk_payload(
    vector: torch.Tensor,
    tokenizer: Any,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    values, indices = torch.topk(vector, k=min(int(limit), int(vector.numel())))
    cpu_values = values.detach().float().cpu().tolist()
    cpu_indices = indices.detach().cpu().tolist()
    return [
        {
            "rank": rank,
            **token_payload(tokenizer, int(token_id)),
            "logit": float(logit),
        }
        for rank, (token_id, logit) in enumerate(
            zip(cpu_indices, cpu_values),
            start=1,
        )
    ]


def token_rank_and_logit(vector: torch.Tensor, token_id: int) -> tuple[int, float]:
    candidate = vector[int(token_id)]
    rank = int(torch.count_nonzero(vector > candidate).detach().cpu().item()) + 1
    return rank, float(candidate.detach().float().cpu().item())


def first_difference(left: list[int], right: list[int]) -> int | None:
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if int(left_token) != int(right_token):
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def summarize_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    disagreements = [event for event in events if event["draft_disagrees"]]
    syntax_disagreements = [
        event for event in disagreements if event["syntax_related"]
    ]
    proposed_ranks = [
        int(event["draft_token_rank"])
        for event in disagreements
        if event.get("draft_token_rank") is not None
    ]
    pair_counts = Counter(
        (
            event["target_token"]["piece"],
            event["draft_token"]["piece"] if event.get("draft_token") else "<none>",
        )
        for event in disagreements
    )
    return {
        "captured_events": len(events),
        "draft_disagreements": len(disagreements),
        "syntax_related_draft_disagreements": len(syntax_disagreements),
        "draft_alternative_rank_le_2": sum(rank <= 2 for rank in proposed_ranks),
        "draft_alternative_rank_le_5": sum(rank <= 5 for rank in proposed_ranks),
        "draft_alternative_rank_le_20": sum(rank <= 20 for rank in proposed_ranks),
        "draft_alternative_rank_gt_20": sum(rank > 20 for rank in proposed_ranks),
        "draft_alternative_rank_missing": sum(
            event.get("draft_token_rank") is None for event in disagreements
        ),
        "most_common_target_vs_draft_tokens": [
            {"target_piece": target, "draft_piece": draft, "events": count}
            for (target, draft), count in pair_counts.most_common(20)
        ],
    }


@torch.inference_mode()
def probe_table(
    recognizer: Any,
    target: dict[str, Any],
    draft: dict[str, Any],
    args: argparse.Namespace,
    selected_syntax_tokens: dict[int, list[str]],
) -> dict[str, Any]:
    request_id = str(target["request_id"])
    reference = target_tokens(target)
    crop = exact_target_crop(target, args.images_dir)
    prefilled = recognizer.prefill_one(request_for(target, crop, args))
    (
        cache,
        rope_deltas,
        cache_position,
        _first_token_tensor,
        cache_release,
    ) = prefilled.take_device_state()
    matcher = TableDraftMatcher(
        draft,
        recognizer.tokenizer,
        eos_token_id=int(recognizer.model.config.eos_token_id),
        block_size=args.draft_length,
    )
    prefix = [int(reference[0])]
    matcher.start(prefix[0])
    live_tokens = [int(prefilled.first_token)]
    position = int(cache_position.detach().cpu().item())
    flat_cache = cache.flat_tensors()
    device_input = torch.empty((1, 1), device=recognizer.device, dtype=torch.int64)
    events: list[dict[str, Any]] = []
    started = time.perf_counter()

    try:
        for target_index in range(1, min(len(reference), args.max_new_tokens)):
            proposal = matcher.propose(prefix)
            draft_token = (
                int(proposal.tokens[0])
                if proposal is not None and proposal.tokens
                else None
            )
            target_token = int(reference[target_index])
            cache_position.fill_(position)
            device_input.fill_(int(prefix[-1]))
            logits = recognizer.decode_fn(
                device_input,
                cache_position,
                rope_deltas,
                *flat_cache,
            )
            vector = logits[0, -1, :].float()
            live_token = int(torch.argmax(vector).detach().cpu().item())
            live_tokens.append(live_token)
            draft_disagrees = draft_token is not None and draft_token != target_token
            syntax_related = is_syntax_token(
                recognizer.tokenizer,
                target_token,
                selected_syntax_tokens,
            ) or is_syntax_token(
                recognizer.tokenizer,
                draft_token,
                selected_syntax_tokens,
            )
            capture = draft_disagrees or syntax_related or live_token != target_token
            if capture:
                topk = topk_payload(
                    vector,
                    recognizer.tokenizer,
                    limit=args.top_k,
                )
                target_rank, target_logit = token_rank_and_logit(vector, target_token)
                draft_rank: int | None = None
                draft_logit: float | None = None
                if draft_token is not None:
                    draft_rank, draft_logit = token_rank_and_logit(vector, draft_token)
                events.append(
                    {
                        "target_index": target_index,
                        "cache_position": position,
                        "prefix_tail": recognizer.tokenizer.decode(
                            prefix[-24:],
                            skip_special_tokens=False,
                        ),
                        "proposal_start": proposal.start if proposal is not None else None,
                        "proposal_anchor_tokens": (
                            proposal.anchor_tokens if proposal is not None else None
                        ),
                        "draft_disagrees": draft_disagrees,
                        "syntax_related": syntax_related,
                        "live_matches_saved_target": live_token == target_token,
                        "target_token": token_payload(
                            recognizer.tokenizer,
                            target_token,
                        ),
                        "target_token_rank": target_rank,
                        "target_token_logit": target_logit,
                        "draft_token": (
                            token_payload(recognizer.tokenizer, draft_token)
                            if draft_token is not None
                            else None
                        ),
                        "draft_token_rank": draft_rank,
                        "draft_token_logit": draft_logit,
                        "target_minus_draft_logit": (
                            target_logit - draft_logit
                            if draft_logit is not None
                            else None
                        ),
                        "live_token": token_payload(
                            recognizer.tokenizer,
                            live_token,
                        ),
                        "topk": topk,
                    }
                )

            accepted = int(draft_token == target_token) if draft_token is not None else 0
            matcher.commit(
                proposal,
                accepted_draft_tokens=accepted,
                emitted_tokens=(target_token,),
            )
            prefix.append(target_token)
            position += 1
            if target_token == int(recognizer.model.config.eos_token_id):
                break
    finally:
        if cache_release is not None:
            cache_release()

    return {
        "request_id": request_id,
        "crop_size": list(crop.size),
        "input_tokens": int(prefilled.input_tokens),
        "projected_image_tokens": int(prefilled.projected_image_tokens),
        "saved_tokens": len(reference),
        "replayed_tokens": len(prefix),
        "wall_s": time.perf_counter() - started,
        "live_first_token": int(prefilled.first_token),
        "saved_first_token": int(reference[0]),
        "live_first_matches_saved": int(prefilled.first_token) == int(reference[0]),
        "live_argmax_matches_saved_count": sum(
            int(left) == int(right)
            for left, right in zip(live_tokens, reference[: len(live_tokens)])
        ),
        "live_first_difference": first_difference(
            live_tokens,
            reference[: len(live_tokens)],
        ),
        "summary": summarize_events(events),
        "events": events,
    }


def main() -> None:
    args = parse_args()
    import torch_npu  # noqa: F401

    torch.npu.config.allow_internal_format = True
    torch.npu.set_compile_mode(jit_compile=False)
    targets = {record["request_id"]: record for record in read_jsonl(args.targets)}
    drafts = {record["request_id"]: record for record in read_jsonl(args.drafts)}
    request_ids = tuple(args.request_id or DEFAULT_REQUEST_IDS)
    missing_targets = set(request_ids) - set(targets)
    missing_drafts = set(request_ids) - set(drafts)
    if missing_targets or missing_drafts:
        raise KeyError(
            f"missing targets={sorted(missing_targets)} drafts={sorted(missing_drafts)}"
        )

    recognizer = build_recognizer(args)
    selected = syntax_tokens(recognizer.tokenizer)
    tables = []
    for table_index, request_id in enumerate(request_ids, start=1):
        print(
            f"TABLE_SYNTAX_TOPK_PROGRESS table={table_index}/{len(request_ids)} "
            f"id={request_id}",
            flush=True,
        )
        table = probe_table(
            recognizer,
            targets[request_id],
            drafts[request_id],
            args,
            selected,
        )
        tables.append(table)
        print(
            f"TABLE_SYNTAX_TOPK_RESULT id={request_id} "
            f"tokens={table['replayed_tokens']} events={len(table['events'])} "
            f"disagreements={table['summary']['draft_disagreements']} "
            f"draft_top5={table['summary']['draft_alternative_rank_le_5']} "
            f"exact={table['live_first_difference'] is None} "
            f"wall_s={table['wall_s']:.3f}",
            flush=True,
        )

    all_events = [event for table in tables for event in table["events"]]
    payload = {
        "configuration": {
            "targets": str(args.targets),
            "drafts": str(args.drafts),
            "request_ids": list(request_ids),
            "top_k": args.top_k,
            "cache_length": args.cache_length,
            "max_new_tokens": args.max_new_tokens,
            "mode": (
                "teacher-forced saved B1 history; U2 next-token candidate from "
                "TableDraftMatcher; ordinary B1 decode logits"
            ),
            "recognizer": recognizer.configuration(),
            "syntax_tokens": [
                {**token_payload(recognizer.tokenizer, token_id), "forms": forms}
                for token_id, forms in sorted(selected.items())
            ],
        },
        "summary": summarize_events(all_events),
        "tables": tables,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("TABLE_SYNTAX_TOPK_SUMMARY " + json.dumps(payload["summary"]), flush=True)
    print(f"OUTPUT={args.output}", flush=True)


if __name__ == "__main__":
    main()
