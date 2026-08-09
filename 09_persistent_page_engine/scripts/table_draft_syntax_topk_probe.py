#!/usr/bin/env python3
"""Inspect real U2-lane logits at manually reviewed wrapper divergences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch
from PIL import Image


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
REPO_ROOT = EXPERIMENT_ROOT.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))
sys.path.insert(0, str(HERE))

from paddleocr_vl.serving.types import RecognitionRequest
from table_row_split_lab import load_crop
from table_spec_decode_lab import DEFAULT_TEXT_BUCKETS, DEFAULT_VISION_BUCKETS
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


def tokenizer_token_id(tokenizer: Any, token: str) -> int:
    if hasattr(tokenizer, "token_to_id"):
        value = tokenizer.token_to_id(token)
    else:
        value = tokenizer.convert_tokens_to_ids(token)
    if value is None:
        raise ValueError(f"tokenizer does not contain {token!r}")
    return int(value)


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
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--window", type=int, default=2)
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


def rotate_cw(image: Image.Image, degrees: int) -> Image.Image:
    transforms = {
        0: None,
        90: Image.Transpose.ROTATE_270,
        180: Image.Transpose.ROTATE_180,
        270: Image.Transpose.ROTATE_90,
    }
    if degrees not in transforms:
        raise ValueError(f"unsupported clockwise rotation: {degrees}")
    transform = transforms[degrees]
    return image if transform is None else image.transpose(transform)


def reconstruct_row_crops(
    record: dict[str, Any],
    images_dir: Path,
) -> dict[int, Image.Image]:
    image = load_crop(record, images_dir)
    expected_raw = tuple(int(value) for value in record["raw_crop_size"])
    if image.size != expected_raw:
        raise ValueError(
            f"{record['request_id']}: raw crop {image.size} != {expected_raw}"
        )
    image = rotate_cw(image, int(record.get("row_draft_rotation_cw", 0)))
    expected_source = tuple(int(value) for value in record["row_draft_source_size"])
    if image.size != expected_source:
        image = image.resize(expected_source, resample=Image.Resampling.BICUBIC)
    trim_box = tuple(int(value) for value in record["trim_box_in_raw_crop"])
    image = image.crop(trim_box)
    expected_crop = tuple(int(value) for value in record["crop_size"])
    if image.size != expected_crop:
        raise ValueError(
            f"{record['request_id']}: prepared crop {image.size} != {expected_crop}"
        )

    result: dict[int, Image.Image] = {}
    for row_record in record.get("rows") or ():
        row_index = int(row_record["row_index"])
        top, bottom = (int(value) for value in row_record["row_y"])
        row = image.crop((0, top, image.width, bottom))
        expected = tuple(int(value) for value in row_record["crop_size"])
        if row.size != expected:
            if row.width > expected[0] or row.height > expected[1]:
                raise ValueError(
                    f"{record['request_id']} lane {row_index}: "
                    f"row crop {row.size} exceeds {expected}"
                )
            padded = Image.new("RGB", expected, "white")
            padded.paste(row, (0, 0))
            row = padded
        result[row_index] = row
    return result


def generated_ids(row_record: dict[str, Any], eos_token_id: int) -> list[int]:
    tokens = [int(value) for value in row_record.get("token_ids") or ()]
    if not tokens:
        raise ValueError(f"{row_record.get('request_id')}: empty generated IDs")
    return tokens


def cell_content_position(
    tokens: list[int],
    *,
    logical_row: int,
    column: int,
    cell_tokens: set[int],
    newline_token: int,
    eos_token_id: int,
) -> tuple[int, list[int]]:
    row = 0
    col = -1
    cell_start: int | None = None
    for index, token in enumerate(tokens):
        if token == eos_token_id:
            break
        if token == newline_token:
            if row == logical_row and col == column and cell_start is not None:
                return cell_start + 1, tokens[cell_start:index]
            row += 1
            col = -1
            cell_start = None
            continue
        if token in cell_tokens:
            if row == logical_row and col == column and cell_start is not None:
                return cell_start + 1, tokens[cell_start:index]
            col += 1
            cell_start = index
    if row == logical_row and col == column and cell_start is not None:
        end = len(tokens) - int(bool(tokens and tokens[-1] == eos_token_id))
        return cell_start + 1, tokens[cell_start:end]
    raise KeyError(f"cell row={logical_row} column={column} not found")


def row_request(
    request_id: str,
    row: Image.Image,
) -> RecognitionRequest:
    pixels = row.width * row.height
    return RecognitionRequest(
        request_id=request_id,
        crop=row,
        prompt="Table Recognition:",
        min_pixels=pixels,
        max_pixels=pixels,
        source_crop_size=row.size,
    )


@torch.inference_mode()
def probe_lane(
    recognizer: Any,
    *,
    request_id: str,
    row_record: dict[str, Any],
    row_image: Image.Image,
    target_index: int,
    base_alternative_id: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    reference = generated_ids(
        row_record,
        int(recognizer.model.config.eos_token_id),
    )
    prefilled = recognizer.prefill_one(
        row_request(f"{request_id}:draft-logits", row_image)
    )
    (
        cache,
        rope_deltas,
        cache_position,
        _first_token,
        cache_release,
    ) = prefilled.take_device_state()
    flat_cache = cache.flat_tensors()
    device_input = torch.empty((1, 1), device=recognizer.device, dtype=torch.int64)
    position = int(cache_position.detach().cpu().item())
    live = [int(prefilled.first_token)]
    prefix = [int(reference[0])]
    captures = []
    started = time.perf_counter()
    try:
        for index in range(1, len(reference)):
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
            live.append(live_token)
            if abs(index - target_index) <= int(args.window):
                target_token = int(reference[index])
                target_rank, target_logit = token_rank_and_logit(
                    vector, target_token
                )
                alternative_rank = None
                alternative_logit = None
                if index == target_index:
                    alternative_rank, alternative_logit = token_rank_and_logit(
                        vector, base_alternative_id
                    )
                captures.append(
                    {
                        "relative_position": index - target_index,
                        "token_index": index,
                        "prefix_tail": recognizer.tokenizer.decode(
                            prefix[-24:], skip_special_tokens=False
                        ),
                        "saved_token": token_payload(
                            recognizer.tokenizer, target_token
                        ),
                        "live_token": token_payload(
                            recognizer.tokenizer, live_token
                        ),
                        "saved_rank": target_rank,
                        "saved_logit": target_logit,
                        "base_alternative": (
                            token_payload(
                                recognizer.tokenizer, base_alternative_id
                            )
                            if index == target_index
                            else None
                        ),
                        "base_alternative_rank": alternative_rank,
                        "base_alternative_logit": alternative_logit,
                        "saved_minus_base_alternative_logit": (
                            target_logit - alternative_logit
                            if alternative_logit is not None
                            else None
                        ),
                        "topk": topk_payload(
                            vector,
                            recognizer.tokenizer,
                            limit=args.top_k,
                        ),
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
        "input_tokens": int(prefilled.input_tokens),
        "projected_image_tokens": int(prefilled.projected_image_tokens),
        "saved_tokens": len(reference),
        "live_first_difference": first_difference(live, reference[: len(live)]),
        "wall_s": time.perf_counter() - started,
        "captures": captures,
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
    recognizer = build_recognizer(args)
    eos_token_id = int(recognizer.model.config.eos_token_id)
    cell_tokens = {
        tokenizer_token_id(recognizer.tokenizer, token)
        for token in ("<fcel>", "<ecel>", "<lcel>", "<ucel>", "<xcel>")
    }
    newline_token = tokenizer_token_id(recognizer.tokenizer, "<nl>")
    results = []
    for case_index, case in enumerate(cases, start=1):
        request_id = str(case["request_id"])
        target = targets[request_id]
        draft = drafts[request_id]
        base_row, base_column = (int(value) for value in case["base_cell"])
        lane, draft_row, draft_column = (
            int(value) for value in case["draft_cell"]
        )
        target_ids = generated_ids(target["rows"][0], eos_token_id)
        base_position, base_cell_ids = cell_content_position(
            target_ids,
            logical_row=base_row,
            column=base_column,
            cell_tokens=cell_tokens,
            newline_token=newline_token,
            eos_token_id=eos_token_id,
        )
        row_record = next(
            row for row in draft["rows"] if int(row["row_index"]) == lane
        )
        draft_ids = generated_ids(row_record, eos_token_id)
        draft_position, draft_cell_ids = cell_content_position(
            draft_ids,
            logical_row=draft_row,
            column=draft_column,
            cell_tokens=cell_tokens,
            newline_token=newline_token,
            eos_token_id=eos_token_id,
        )
        base_token_id = int(target_ids[base_position])
        draft_token_id = int(draft_ids[draft_position])
        row_images = reconstruct_row_crops(draft, args.images_dir)
        print(
            f"TABLE_DRAFT_SYNTAX_TOPK_PROGRESS case={case_index}/{len(cases)} "
            f"id={request_id} lane={lane}",
            flush=True,
        )
        probe = probe_lane(
            recognizer,
            request_id=request_id,
            row_record=row_record,
            row_image=row_images[lane],
            target_index=draft_position,
            base_alternative_id=base_token_id,
            args=args,
        )
        payload = {
            **case,
            "base_cell_ids": base_cell_ids,
            "base_cell_text": recognizer.tokenizer.decode(
                base_cell_ids, skip_special_tokens=False
            ),
            "base_first_content_token": token_payload(
                recognizer.tokenizer, base_token_id
            ),
            "draft_cell_ids": draft_cell_ids,
            "draft_cell_text": recognizer.tokenizer.decode(
                draft_cell_ids, skip_special_tokens=False
            ),
            "draft_first_content_token": token_payload(
                recognizer.tokenizer, draft_token_id
            ),
            "draft_first_content_index": draft_position,
            "probe": probe,
        }
        results.append(payload)
        central = next(
            item
            for item in probe["captures"]
            if item["relative_position"] == 0
        )
        print(
            f"TABLE_DRAFT_SYNTAX_TOPK_RESULT id={request_id} lane={lane} "
            f"draft={central['saved_token']['piece']} "
            f"base_alt={central['base_alternative']['piece']} "
            f"base_alt_rank={central['base_alternative_rank']} "
            f"margin={central['saved_minus_base_alternative_logit']:.6f} "
            f"exact={probe['live_first_difference'] is None}",
            flush=True,
        )

    output = {
        "configuration": {
            "targets": str(args.targets),
            "drafts": str(args.drafts),
            "cases": str(args.cases),
            "top_k": args.top_k,
            "window": args.window,
            "mode": (
                "teacher-forced saved U2 lane histories with reconstructed exact "
                "row crops; actual generated IDs only; no text encoding"
            ),
            "recognizer": recognizer.configuration(),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"OUTPUT={args.output}", flush=True)


if __name__ == "__main__":
    main()
