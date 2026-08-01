#!/usr/bin/env python3
"""Mine and render possible OCR degenerations for manual review.

The scores in this script are deliberately high-recall triage signals. They do
not decide which device is more accurate: the generated HTML places the crop,
an IoU-matched OmniDocBench annotation, and both outputs side by side so a
person can make that decision.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import html
import json
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable

from PIL import Image


EOS_TOKEN_ID = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-output", type=Path, required=True)
    parser.add_argument("--candidate-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--review-limit", type=int, default=100)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def visible_tokens(row: dict[str, Any]) -> list[int]:
    tokens = [int(value) for value in row.get("token_ids") or ()]
    if tokens and tokens[-1] == EOS_TOKEN_ID:
        tokens.pop()
    return tokens


def shared_prefix(left: list[int], right: list[int]) -> int:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right))


def ngram_dominance(tokens: list[int]) -> float:
    if len(tokens) < 8:
        return 0.0
    best = 0.0
    for width in range(1, min(8, len(tokens)) + 1):
        counts = Counter(
            tuple(tokens[index : index + width])
            for index in range(len(tokens) - width + 1)
        )
        repetitions = counts.most_common(1)[0][1]
        best = max(best, min(1.0, repetitions * width / len(tokens)))
    return best


def tail_periodicity(tokens: list[int]) -> tuple[float, int | None]:
    tail = tokens[-min(256, len(tokens)) :]
    if len(tail) < 16:
        return 0.0, None
    best_ratio = 0.0
    best_period = None
    for period in range(1, min(32, len(tail) // 2) + 1):
        total = len(tail) - period
        matches = sum(
            tail[index] == tail[index - period]
            for index in range(period, len(tail))
        )
        ratio = matches / total
        if ratio > best_ratio:
            best_ratio = ratio
            best_period = period
    return best_ratio, best_period


SCRIPT_PATTERNS = {
    "arabic": ("ARABIC",),
    "cyrillic": ("CYRILLIC",),
    "devanagari": ("DEVANAGARI",),
    "greek": ("GREEK",),
    "hangul": ("HANGUL",),
    "hiragana": ("HIRAGANA",),
    "katakana": ("KATAKANA",),
    "thai": ("THAI",),
    "cjk": ("CJK", "IDEOGRAPH"),
    "latin": ("LATIN",),
}


def script_counts(text: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for character in text:
        if not character.isalpha():
            continue
        name = unicodedata.name(character, "")
        for script, patterns in SCRIPT_PATTERNS.items():
            if any(pattern in name for pattern in patterns):
                counts[script] += 1
                break
    return dict(counts)


def fingerprint_status(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> str:
    left = reference.get("input_fingerprints") or {}
    right = candidate.get("input_fingerprints") or {}
    left_crop = (left.get("crop") or {}).get("sha256")
    right_crop = (right.get("crop") or {}).get("sha256")
    left_prepared = left.get("prepared_inputs_sha256")
    right_prepared = right.get("prepared_inputs_sha256")
    if not all((left_crop, right_crop, left_prepared, right_prepared)):
        return "unavailable"
    if left_crop == right_crop and left_prepared == right_prepared:
        return "exact"
    return "different"


def candidate_flags(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    ref_tokens = visible_tokens(reference)
    cand_tokens = visible_tokens(candidate)
    ref_count = len(ref_tokens)
    cand_count = len(cand_tokens)
    ref_text = str(reference.get("text") or "")
    cand_text = str(candidate.get("text") or "")
    text_ratio = SequenceMatcher(None, ref_text, cand_text, autojunk=False).ratio()
    cand_dominance = ngram_dominance(cand_tokens)
    ref_dominance = ngram_dominance(ref_tokens)
    cand_periodicity, cand_period = tail_periodicity(cand_tokens)
    ref_periodicity, ref_period = tail_periodicity(ref_tokens)
    cand_scripts = script_counts(cand_text)
    ref_scripts = script_counts(ref_text)
    added_scripts = {
        name: count
        for name, count in cand_scripts.items()
        if count >= 4 and ref_scripts.get(name, 0) == 0
    }

    flags: list[str] = []
    if cand_count >= 128 and cand_count >= max(ref_count * 3, ref_count + 128):
        flags.append("candidate_runaway_length")
    if ref_count >= 128 and ref_count >= max(cand_count * 3, cand_count + 128):
        flags.append("candidate_possible_early_eos")
    if cand_count >= 64 and (
        cand_dominance >= 0.45 or cand_periodicity >= 0.88
    ):
        flags.append("candidate_repetition")
    if ref_count >= 64 and (
        ref_dominance >= 0.45 or ref_periodicity >= 0.88
    ):
        flags.append("reference_repetition")
    if added_scripts:
        flags.append("candidate_added_script")
    if abs(cand_count - ref_count) >= 128:
        flags.append("large_length_delta")
    if text_ratio < 0.30 and max(ref_count, cand_count) >= 16:
        flags.append("low_text_similarity")

    metrics = {
        "reference_tokens": ref_count,
        "candidate_tokens": cand_count,
        "candidate_minus_reference_tokens": cand_count - ref_count,
        "shared_prefix_tokens": shared_prefix(ref_tokens, cand_tokens),
        "text_similarity": text_ratio,
        "reference_ngram_dominance": ref_dominance,
        "candidate_ngram_dominance": cand_dominance,
        "reference_tail_periodicity": ref_periodicity,
        "candidate_tail_periodicity": cand_periodicity,
        "reference_tail_period": ref_period,
        "candidate_tail_period": cand_period,
        "reference_scripts": ref_scripts,
        "candidate_scripts": cand_scripts,
        "candidate_added_scripts": added_scripts,
    }
    return flags, metrics


FLAG_WEIGHT = {
    "candidate_runaway_length": 100,
    "candidate_repetition": 90,
    "candidate_possible_early_eos": 80,
    "candidate_added_script": 70,
    "large_length_delta": 50,
    "low_text_similarity": 30,
    "reference_repetition": 10,
}


def bbox_from_poly(poly: Iterable[float]) -> list[float]:
    values = [float(value) for value in poly]
    xs = values[0::2]
    ys = values[1::2]
    return [min(xs), min(ys), max(xs), max(ys)]


def bbox_iou(left: Iterable[float], right: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in left]
    bx1, by1, bx2, by2 = [float(value) for value in right]
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(
        0.0, bx2 - bx1
    ) * max(0.0, by2 - by1) - intersection
    return intersection / union if union else 0.0


def gt_content(item: dict[str, Any]) -> str:
    for key in ("latex", "text", "html"):
        value = item.get(key)
        if value:
            return str(value)
    return ""


def label_matches_gt(label: str, category: str) -> bool:
    category = category.lower()
    if label == "formula":
        return "equation" in category or "formula" in category
    if label == "table":
        return "table" in category
    return not any(token in category for token in ("equation", "formula", "table", "image"))


def best_gt_match(
    gt_page: dict[str, Any] | None,
    bbox: list[float] | None,
    label: str,
) -> dict[str, Any]:
    if gt_page is None or bbox is None:
        return {"iou": None, "category": None, "content": ""}
    candidates = []
    for item in gt_page.get("layout_dets") or ():
        poly = item.get("poly")
        if not poly or not label_matches_gt(label, str(item.get("category_type", ""))):
            continue
        candidates.append((bbox_iou(bbox, bbox_from_poly(poly)), item))
    if not candidates:
        return {"iou": None, "category": None, "content": ""}
    iou, item = max(candidates, key=lambda pair: pair[0])
    return {
        "iou": iou,
        "category": item.get("category_type"),
        "content": gt_content(item),
    }


def page_block(
    page_rows: list[dict[str, Any]], page_index: int, block_index: int
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if page_index < 0 or page_index >= len(page_rows):
        return None, None
    page = page_rows[page_index]
    blocks = page.get("parsing_res_list") or ()
    block = next(
        (item for item in blocks if int(item.get("block_id", -1)) == block_index),
        None,
    )
    if block is None and 0 <= block_index < len(blocks):
        block = blocks[block_index]
    return page, block


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def save_crop(
    output_dir: Path,
    request_id: str,
    page: dict[str, Any] | None,
    block: dict[str, Any] | None,
) -> tuple[str | None, list[float] | None]:
    if page is None or block is None:
        return None, None
    bbox = [float(value) for value in block.get("block_bbox") or ()]
    if len(bbox) != 4:
        return None, None
    image_path = Path(str(page.get("input_path") or ""))
    if not image_path.is_file():
        return None, bbox
    crops = output_dir / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    name = f"{safe_name(request_id)}.png"
    destination = crops / name
    with Image.open(image_path) as image:
        image.convert("RGB").crop(tuple(int(round(value)) for value in bbox)).save(
            destination
        )
    return f"crops/{name}", bbox


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts = Counter(str(row.get(key)) for row in rows)
    return dict(sorted(counts.items(), key=lambda pair: (-pair[1], pair[0])))


def markdown_table(counts: dict[str, int]) -> list[str]:
    lines = ["| value | candidates |", "|---|---:|"]
    lines.extend(f"| {key} | {value} |" for key, value in counts.items())
    return lines


def main() -> None:
    args = parse_args()
    if args.review_limit <= 0:
        raise ValueError("--review-limit must be positive")
    reference_root = args.reference_output.resolve()
    candidate_root = args.candidate_output.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    reference_summary = load_json(reference_root / "run_summary.json")
    candidate_summary = load_json(candidate_root / "run_summary.json")
    reference_rows = load_jsonl(reference_root / "recognition_trace.jsonl")
    candidate_rows = load_jsonl(candidate_root / "recognition_trace.jsonl")
    reference_by_id = {str(row["request_id"]): row for row in reference_rows}
    candidate_by_id = {str(row["request_id"]): row for row in candidate_rows}
    if set(reference_by_id) != set(candidate_by_id):
        missing_reference = sorted(set(candidate_by_id) - set(reference_by_id))
        missing_candidate = sorted(set(reference_by_id) - set(candidate_by_id))
        raise ValueError(
            "request sets differ: "
            f"missing_reference={missing_reference[:10]} "
            f"missing_candidate={missing_candidate[:10]}"
        )

    page_rows_path = candidate_root / "page_regions.jsonl"
    page_rows = load_jsonl(page_rows_path) if page_rows_path.is_file() else []
    gt_path = candidate_root / "OmniDocBench_subset.json"
    gt_pages = load_json(gt_path) if gt_path.is_file() else []

    all_rows: list[dict[str, Any]] = []
    for request_id in reference_by_id:
        reference = reference_by_id[request_id]
        candidate = candidate_by_id[request_id]
        if visible_tokens(reference) == visible_tokens(candidate):
            continue
        flags, metrics = candidate_flags(reference, candidate)
        generation = (candidate.get("text_prefill") or {}).get(
            "private_cache_generation"
        )
        slot_index = (candidate.get("text_prefill") or {}).get(
            "private_cache_slot_index"
        )
        row = {
            "request_id": request_id,
            "page_input_index": int(candidate.get("page_input_index", -1)),
            "block_index": int(candidate.get("block_index", -1)),
            "global_request_index": int(candidate.get("global_request_index", -1)),
            "label": str(candidate.get("label") or "unknown"),
            "input_fingerprint_status": fingerprint_status(reference, candidate),
            "flags": flags,
            "priority": sum(FLAG_WEIGHT.get(flag, 0) for flag in flags),
            "reference_text": str(reference.get("text") or ""),
            "candidate_text": str(candidate.get("text") or ""),
            "reference_stop_reason": reference.get("stop_reason"),
            "candidate_stop_reason": candidate.get("stop_reason"),
            "vision_bucket": (candidate.get("vision") or {}).get("bucket"),
            "vision_pack_crops": (candidate.get("vision") or {}).get("pack_crops"),
            "text_bucket": (candidate.get("text_prefill") or {}).get("bucket"),
            "text_pack_members": (candidate.get("text_prefill") or {}).get(
                "pack_members"
            ),
            "private_cache_slot_index": slot_index,
            "private_cache_generation": generation,
            "private_cache_reused": generation is not None and int(generation) > 1,
            "decode_slot_index": candidate.get("decode_slot_index"),
            "decode_slot_epoch": candidate.get("decode_slot_epoch"),
            **metrics,
        }
        all_rows.append(row)

    all_rows.sort(
        key=lambda row: (
            -int(row["priority"]),
            -abs(int(row["candidate_minus_reference_tokens"])),
            float(row["text_similarity"]),
            str(row["request_id"]),
        )
    )
    candidates = [row for row in all_rows if row["flags"]]
    review_rows = candidates[: args.review_limit]

    for row in review_rows:
        page_index = int(row["page_input_index"])
        page, block = page_block(page_rows, page_index, int(row["block_index"]))
        crop_path, bbox = save_crop(
            output_dir, str(row["request_id"]), page, block
        )
        gt_page = gt_pages[page_index] if 0 <= page_index < len(gt_pages) else None
        row["source_image_name"] = page.get("image_name") if page else None
        row["crop_path"] = crop_path
        row["crop_bbox"] = bbox
        row["ground_truth_match"] = best_gt_match(
            gt_page, bbox, str(row["label"])
        )

    flag_counts = Counter(flag for row in candidates for flag in row["flags"])
    summary = {
        "kind": "experiment09_manual_degeneration_survey",
        "reference_output": str(reference_root),
        "candidate_output": str(candidate_root),
        "reference_project_configuration": reference_summary.get("configuration"),
        "candidate_project_configuration": candidate_summary.get("configuration"),
        "shared_requests": len(reference_by_id),
        "token_exact_requests": len(reference_by_id) - len(all_rows),
        "token_different_requests": len(all_rows),
        "triage_candidate_requests": len(candidates),
        "review_rows_rendered": len(review_rows),
        "candidate_minus_reference_tokens_all": sum(
            int(row["candidate_minus_reference_tokens"]) for row in all_rows
        ),
        "candidate_minus_reference_tokens_triage": sum(
            int(row["candidate_minus_reference_tokens"]) for row in candidates
        ),
        "flag_counts": dict(sorted(flag_counts.items())),
        "triage_by_label": count_by(candidates, "label"),
        "triage_by_input_fingerprint": count_by(
            candidates, "input_fingerprint_status"
        ),
        "triage_by_private_cache_reused": count_by(
            candidates, "private_cache_reused"
        ),
        "triage_by_vision_bucket": count_by(candidates, "vision_bucket"),
        "triage_by_text_bucket": count_by(candidates, "text_bucket"),
        "manual_review_required": True,
        "warning": (
            "Triage flags are high-recall heuristics, not accuracy verdicts. "
            "Inspect the crop and GT candidate before calling a case degenerate."
        ),
    }
    (output_dir / "survey.json").write_text(
        json.dumps({"summary": summary, "rows": all_rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    report = [
        "# Experiment 09 manual degeneration survey",
        "",
        "> Triage flags are not correctness judgments. Review the crop, matched GT,",
        "> 910B output, and 310P output before assigning a manual disposition.",
        "",
        f"- shared requests: {summary['shared_requests']}",
        f"- token-exact: {summary['token_exact_requests']}",
        f"- token-different: {summary['token_different_requests']}",
        f"- triage candidates: {summary['triage_candidate_requests']}",
        f"- rendered for review: {summary['review_rows_rendered']}",
        f"- total candidate-minus-reference tokens: {summary['candidate_minus_reference_tokens_all']:+d}",
        "",
        "## Triage flags",
        "",
        *markdown_table(dict(sorted(flag_counts.items()))),
        "",
        "## Candidates by label",
        "",
        *markdown_table(summary["triage_by_label"]),
        "",
        "## Candidates by private-cache reuse",
        "",
        *markdown_table(summary["triage_by_private_cache_reused"]),
        "",
        "## Review queue",
        "",
        "| rank | request | label | input | flags | ref/cand tokens | similarity | private cache | decode slot |",
        "|---:|---|---|---|---|---:|---:|---|---|",
    ]
    for rank, row in enumerate(review_rows, 1):
        report.append(
            f"| {rank} | {row['request_id']} | {row['label']} | "
            f"{row['input_fingerprint_status']} | {', '.join(row['flags'])} | "
            f"{row['reference_tokens']}/{row['candidate_tokens']} | "
            f"{row['text_similarity']:.4f} | "
            f"{row['private_cache_slot_index']} gen={row['private_cache_generation']} | "
            f"{row['decode_slot_index']} epoch={row['decode_slot_epoch']} |"
        )
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    cards = []
    for rank, row in enumerate(review_rows, 1):
        gt = row.get("ground_truth_match") or {}
        image = (
            f'<img src="{html.escape(str(row["crop_path"]))}" alt="crop">'
            if row.get("crop_path")
            else "<p>Crop unavailable</p>"
        )
        cards.append(
            f"""
<section class="case">
  <h2>{rank}. {html.escape(str(row['request_id']))}</h2>
  <p><b>{html.escape(str(row['label']))}</b> · input {row['input_fingerprint_status']} ·
     flags: {html.escape(', '.join(row['flags']))}</p>
  <p>tokens {row['reference_tokens']} → {row['candidate_tokens']} ·
     similarity {row['text_similarity']:.4f} · shared prefix {row['shared_prefix_tokens']} ·
     private cache slot {row['private_cache_slot_index']} generation {row['private_cache_generation']} ·
     decode slot {row['decode_slot_index']} epoch {row['decode_slot_epoch']}</p>
  {image}
  <div class="columns">
    <div><h3>Matched GT candidate (IoU {gt.get('iou')})</h3><pre>{html.escape(str(gt.get('content') or ''))}</pre></div>
    <div><h3>910B</h3><pre>{html.escape(str(row['reference_text']))}</pre></div>
    <div><h3>310P</h3><pre>{html.escape(str(row['candidate_text']))}</pre></div>
  </div>
  <p class="manual">Manual disposition: ☐ equivalent syntax ☐ 310P better ☐ 910B better
     ☐ both wrong ☐ candidate degeneration ☐ reference degeneration ☐ input mismatch</p>
</section>"""
        )
    html_text = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Degeneration survey</title>
<style>
body {{ font: 15px system-ui, sans-serif; margin: 24px; background: #f5f6f8; color: #17191c; }}
.case {{ background: white; border: 1px solid #ccd1d8; border-radius: 10px; padding: 18px; margin: 18px 0; }}
.case > img {{ max-width: 100%; max-height: 360px; border: 1px solid #bbb; image-rendering: auto; }}
.columns {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }}
pre {{ white-space: pre-wrap; word-break: break-word; background: #f1f3f5; padding: 10px; border-radius: 6px; max-height: 360px; overflow: auto; }}
.manual {{ padding: 10px; background: #fff8db; }}
@media (max-width: 1000px) {{ .columns {{ grid-template-columns: 1fr; }} }}
</style></head><body>
<h1>Experiment 09 manual degeneration survey</h1>
<p>Automated flags are triage only. {summary['triage_candidate_requests']} candidates from {summary['shared_requests']} shared requests; showing {len(review_rows)}.</p>
{''.join(cards)}
</body></html>"""
    (output_dir / "review.html").write_text(html_text, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
