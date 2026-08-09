#!/usr/bin/env python3
"""Replay saved B1/U2 IDs and inspect every cell-start math-open candidate."""

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

from table_draft_syntax_topk_probe import (
    generated_ids,
    reconstruct_row_crops,
    row_request,
    tokenizer_token_id,
)
from table_spec_decode_lab import (
    DEFAULT_TEXT_BUCKETS,
    DEFAULT_VISION_BUCKETS,
    exact_target_crop,
    request_for,
    target_tokens,
)
from table_syntax_topk_probe import (
    build_recognizer,
    first_difference,
    read_jsonl,
    token_payload,
    token_rank_and_logit,
    topk_payload,
)


DEFAULT_TARGETS = Path(
    "tmp/09_persistent_page_engine/table_spec_full_d1e6d00/"
    "whole/row_ocr_records.jsonl"
)
DEFAULT_DRAFTS = Path(
    "tmp/09_persistent_page_engine/"
    "table_row_full_uniform2_default_178605c/row_ocr_records.jsonl"
)
DEFAULT_CASES = (
    EXPERIMENT_ROOT / "accuracy_lab/table_wrapper_first_divergences.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--drafts", type=Path, default=DEFAULT_DRAFTS)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
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
        default=(
            REPO_ROOT / ".runtime_cache/09_persistent_page_engine_vision_torchair"
        ),
    )
    parser.add_argument(
        "--text-cache-dir",
        type=Path,
        default=(
            REPO_ROOT / ".runtime_cache/09_persistent_page_engine_text_torchair"
        ),
    )
    return parser.parse_args()


@torch.inference_mode()
def probe_sequence(
    recognizer: Any,
    *,
    request: Any,
    reference: list[int],
    source: str,
    row_index: int,
    cell_token_ids: tuple[int, ...],
    math_open_id: int,
    math_close_id: int,
    top_k: int,
    inspect_all_positions: bool = True,
) -> dict[str, Any]:
    prefilled = recognizer.prefill_one(request)
    (
        cache,
        rope_deltas,
        cache_position,
        _first_token,
        cache_release,
    ) = prefilled.take_device_state()
    flat_cache = cache.flat_tensors()
    device_input = torch.empty((1, 1), device=recognizer.device, dtype=torch.int64)
    prefix = [int(reference[0])]
    live = [int(prefilled.first_token)]
    position = int(cache_position.detach().cpu().item())
    candidates: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for index in range(1, len(reference)):
            current_token = int(prefix[-1])
            cache_position.fill_(position)
            device_input.fill_(current_token)
            logits = recognizer.decode_fn(
                device_input,
                cache_position,
                rope_deltas,
                *flat_cache,
            )
            vector = logits[0, -1, :].float()
            live_token = int(torch.argmax(vector).detach().cpu().item())
            live.append(live_token)
            if inspect_all_positions or current_token in cell_token_ids:
                math_rank, math_logit = token_rank_and_logit(vector, math_open_id)
                if math_rank <= 2:
                    topk = topk_payload(
                        vector,
                        recognizer.tokenizer,
                        limit=top_k,
                    )
                    probabilities = torch.softmax(vector, dim=-1)
                    for item in topk:
                        item["probability"] = float(
                            probabilities[int(item["id"])].detach().cpu().item()
                        )
                    top1 = topk[0]
                    math_probability = float(
                        probabilities[math_open_id].detach().cpu().item()
                    )
                    top1_probability = float(top1["probability"])
                    candidates.append(
                        {
                            "candidate_index": len(candidates),
                            "is_first_rank2_override_candidate": (
                                math_rank == 2
                                and not any(
                                    item["math_open_rank"] == 2
                                    for item in candidates
                                )
                            ),
                            "token_index": index,
                            "cache_position": position,
                            "after_cell_marker": current_token in cell_token_ids,
                            "prefix_tail": recognizer.tokenizer.decode(
                                prefix[-24:], skip_special_tokens=False
                            ),
                            "saved_token": token_payload(
                                recognizer.tokenizer,
                                int(reference[index]),
                            ),
                            "live_greedy_token": token_payload(
                                recognizer.tokenizer,
                                live_token,
                            ),
                            "math_open_rank": math_rank,
                            "math_open_logit": math_logit,
                            "top1_logit": float(top1["logit"]),
                            "top1_minus_math_open_logit": (
                                float(top1["logit"]) - math_logit
                            ),
                            "softmax": {
                                "vocabulary_size": int(vector.numel()),
                                "top1_probability": top1_probability,
                                "math_open_probability": math_probability,
                                "top1_minus_math_open_probability": (
                                    top1_probability - math_probability
                                ),
                                "top1_over_math_open_probability": (
                                    top1_probability / math_probability
                                    if math_probability > 0.0
                                    else None
                                ),
                                "top2_probability_mass": sum(
                                    float(item["probability"])
                                    for item in topk[:2]
                                ),
                            },
                            "inside_unclosed_math_region": (
                                prefix.count(math_open_id)
                                > prefix.count(math_close_id)
                            ),
                            "math_open_count_before": prefix.count(math_open_id),
                            "math_close_count_before": prefix.count(math_close_id),
                            "topk": topk,
                        }
                    )
            prefix.append(int(reference[index]))
            position += 1
            if int(reference[index]) == int(recognizer.model.config.eos_token_id):
                break
    finally:
        if cache_release is not None:
            cache_release()
    return {
        "source": source,
        "request_id": request.request_id.split(":cell-start-logits")[0],
        "row_index": row_index,
        "input_tokens": int(prefilled.input_tokens),
        "saved_tokens": len(reference),
        "live_first_difference": first_difference(live, reference[: len(live)]),
        "candidate_count": len(candidates),
        "rank2_candidate_count": sum(
            item["math_open_rank"] == 2 for item in candidates
        ),
        "wall_s": time.perf_counter() - started,
        "candidates": candidates,
    }


def main() -> None:
    args = parse_args()
    import torch_npu  # noqa: F401

    torch.npu.config.allow_internal_format = True
    torch.npu.set_compile_mode(jit_compile=False)
    targets = {
        record["request_id"]: record for record in read_jsonl(args.targets)
    }
    drafts = {
        record["request_id"]: record for record in read_jsonl(args.drafts)
    }
    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    request_ids = [str(case["request_id"]) for case in cases]
    recognizer = build_recognizer(args)
    eos_token_id = int(recognizer.model.config.eos_token_id)
    cell_token_ids = tuple(
        tokenizer_token_id(recognizer.tokenizer, piece)
        for piece in ("<fcel>", "<ecel>", "<lcel>", "<ucel>", "<xcel>")
    )
    math_open_id = tokenizer_token_id(recognizer.tokenizer, r"\(")
    math_close_id = tokenizer_token_id(recognizer.tokenizer, r"\)")
    sequences: list[dict[str, Any]] = []
    total = len(request_ids) * 3
    progress = 0
    for request_id in request_ids:
        target = targets[request_id]
        target_crop = exact_target_crop(target, args.images_dir)
        target_request = request_for(target, target_crop, args)
        target_request = type(target_request)(
            **{
                **target_request.__dict__,
                "request_id": f"{request_id}:cell-start-logits:b1",
            }
        )
        progress += 1
        print(
            f"TABLE_MATH_OPEN_PROGRESS sequence={progress}/{total} "
            f"id={request_id} source=B1",
            flush=True,
        )
        sequences.append(
            probe_sequence(
                recognizer,
                request=target_request,
                reference=target_tokens(target),
                source="B1",
                row_index=0,
                cell_token_ids=cell_token_ids,
                math_open_id=math_open_id,
                math_close_id=math_close_id,
                top_k=args.top_k,
            )
        )

        draft = drafts[request_id]
        row_images = reconstruct_row_crops(draft, args.images_dir)
        for row_record in draft["rows"]:
            row_index = int(row_record["row_index"])
            progress += 1
            print(
                f"TABLE_MATH_OPEN_PROGRESS sequence={progress}/{total} "
                f"id={request_id} source=U2 lane={row_index}",
                flush=True,
            )
            sequences.append(
                probe_sequence(
                    recognizer,
                    request=row_request(
                        f"{request_id}:cell-start-logits:u2:{row_index}",
                        row_images[row_index],
                    ),
                    reference=generated_ids(row_record, eos_token_id),
                    source="U2",
                    row_index=row_index,
                    cell_token_ids=cell_token_ids,
                    math_open_id=math_open_id,
                    math_close_id=math_close_id,
                    top_k=args.top_k,
                )
            )

    all_candidates = [
        candidate
        for sequence in sequences
        for candidate in sequence["candidates"]
    ]
    payload = {
        "configuration": {
            "targets": str(args.targets),
            "drafts": str(args.drafts),
            "request_ids": request_ids,
            "top_k": args.top_k,
            "mode": (
                "teacher-forced saved generated IDs; capture every generation "
                "position where exact \\( is rank 1 or 2; full-vocabulary fp32 "
                "softmax; no text encoding"
            ),
            "math_open_token_id": math_open_id,
            "math_close_token_id": math_close_id,
            "cell_token_ids": list(cell_token_ids),
            "recognizer": recognizer.configuration(),
        },
        "summary": {
            "sequences": len(sequences),
            "exact_replays": sum(
                sequence["live_first_difference"] is None
                for sequence in sequences
            ),
            "candidates": len(all_candidates),
            "rank1_candidates": sum(
                candidate["math_open_rank"] == 1
                for candidate in all_candidates
            ),
            "rank2_candidates": sum(
                candidate["math_open_rank"] == 2
                for candidate in all_candidates
            ),
            "inside_unclosed_math_region": sum(
                candidate["inside_unclosed_math_region"]
                for candidate in all_candidates
            ),
            "after_cell_marker": sum(
                candidate["after_cell_marker"]
                for candidate in all_candidates
            ),
            "top1_piece_counts": dict(
                Counter(
                    candidate["topk"][0]["piece"]
                    for candidate in all_candidates
                )
            ),
        },
        "sequences": sequences,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("TABLE_MATH_OPEN_SUMMARY " + json.dumps(payload["summary"]), flush=True)
    print(f"OUTPUT={args.output}", flush=True)


if __name__ == "__main__":
    main()
