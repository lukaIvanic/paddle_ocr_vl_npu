#!/usr/bin/env python3
"""Attribute paired Experiment-09 generation and OmniDocBench differences.

This is a frozen-artifact analysis.  It never imports model code, changes a
prediction, or reruns a metric.  It keeps three distinct questions separate:

1. How do the raw recognizer generations differ?
2. Which paired evaluator samples actually move an OmniDocBench metric?
3. Which exact-input table cases are suitable for a later logit replay?

Edit-distance contributions reproduce the evaluator's page-weighted metric:
within each page, sample edit counts are divided by the page's total upper
length, then pages are averaged.  TEDS is reported both sample-weighted and
page-weighted because the recovery tool records both views.
"""

from __future__ import annotations

import argparse
import collections
import difflib
import hashlib
import html
import json
import math
import re
import unicodedata
import zipfile
from pathlib import Path
from typing import Any, Iterable, Sequence


EOS_TOKEN_ID = 2
EVAL_KINDS = ("text_block", "display_formula", "table", "reading_order")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    reference = parser.add_mutually_exclusive_group(required=True)
    reference.add_argument("--reference-bundle", type=Path)
    reference.add_argument("--reference-output", type=Path)
    candidate = parser.add_mutually_exclusive_group(required=True)
    candidate.add_argument("--candidate-bundle", type=Path)
    candidate.add_argument("--candidate-output", type=Path)
    parser.add_argument("--reference-eval-dir", type=Path)
    parser.add_argument("--candidate-eval-dir", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reference-label", default="910B")
    parser.add_argument("--candidate-label", default="310P")
    parser.add_argument("--reference-table-scores", type=Path)
    parser.add_argument("--candidate-table-scores", type=Path)
    parser.add_argument("--reference-teds-summary", type=Path)
    parser.add_argument("--candidate-teds-summary", type=Path)
    parser.add_argument("--review-limit", type=int, default=100)
    parser.add_argument("--expected-pages", type=int)
    parser.add_argument("--expected-shared-requests", type=int)
    parser.add_argument("--expected-reference-table-requests", type=int)
    parser.add_argument("--expected-candidate-table-requests", type=int)
    args = parser.parse_args()
    if args.review_limit <= 0:
        parser.error("--review-limit must be positive")
    for side in ("reference", "candidate"):
        output = getattr(args, f"{side}_output")
        eval_dir = getattr(args, f"{side}_eval_dir")
        bundle = getattr(args, f"{side}_bundle")
        scores = getattr(args, f"{side}_table_scores")
        teds_summary = getattr(args, f"{side}_teds_summary")
        if output is not None and eval_dir is None:
            parser.error(f"--{side}-eval-dir is required with --{side}-output")
        if bundle is not None and any(
            value is not None for value in (eval_dir, scores, teds_summary)
        ):
            parser.error(
                f"--{side}-bundle already contains evaluator/TEDS evidence"
            )
        if bool(scores) != bool(teds_summary):
            parser.error(
                f"--{side}-table-scores and --{side}-teds-summary must be supplied together"
            )
    return args


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _require_output(root: Path) -> Path:
    root = root.expanduser().resolve()
    for name in ("recognition_trace.jsonl", "run_summary.json"):
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)
    return root


def _zip_json(archive: zipfile.ZipFile, member: str) -> Any:
    with archive.open(member) as handle:
        return json.load(handle)


def _zip_jsonl(archive: zipfile.ZipFile, member: str) -> list[dict[str, Any]]:
    with archive.open(member) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_side(
    *,
    bundle: Path | None,
    output: Path | None,
    eval_dir: Path | None,
    table_scores: Path | None,
    teds_summary: Path | None,
    label: str,
) -> dict[str, Any]:
    if bundle is not None:
        bundle = bundle.expanduser().resolve()
        print(f"[atlas] reading {label} bundle: {bundle}", flush=True)
        with zipfile.ZipFile(bundle) as archive:
            names = set(archive.namelist())
            required = {
                "manifest.json",
                "recognition_trace.jsonl",
                "run_summary.json",
                "metric_result.json",
                *(f"evaluator/{kind}.json" for kind in EVAL_KINDS),
            }
            missing = required - names
            if missing:
                raise ValueError(f"bundle {bundle} is missing members: {sorted(missing)}")
            manifest = _zip_json(archive, "manifest.json")
            if (
                manifest.get("schema_version") != 1
                or manifest.get("kind") != "experiment09_generation_difference_bundle"
            ):
                raise ValueError(f"unsupported bundle schema: {manifest}")
            embedded_scores = (
                _zip_json(archive, "corrected_table_scores.json")
                if "corrected_table_scores.json" in names
                else {}
            )
            embedded_teds = (
                _zip_json(archive, "teds_summary.json")
                if "teds_summary.json" in names
                else None
            )
            if bool(embedded_scores) != bool(embedded_teds):
                raise ValueError("bundle has incomplete corrected TEDS evidence")
            loaded = {
                "source": str(bundle),
                "manifest": manifest,
                "trace": _zip_jsonl(archive, "recognition_trace.jsonl"),
                "run_summary": _zip_json(archive, "run_summary.json"),
                "eval": {
                    kind: _zip_json(archive, f"evaluator/{kind}.json")
                    for kind in EVAL_KINDS
                },
                "official": _zip_json(archive, "metric_result.json"),
                "table_scores": embedded_scores,
                "teds_summary": embedded_teds,
            }
        counts = manifest.get("counts") or {}
        run_images = loaded["run_summary"].get("images")
        run_page_count = len(run_images) if isinstance(run_images, list) else int(loaded["run_summary"].get("count", 0))
        actual = {
            "pages_in_run_summary": run_page_count,
            "recognition_requests": len(loaded["trace"]),
            "table_recognition_requests": sum(row.get("label") == "table" for row in loaded["trace"]),
            "table_pages": len({
                _page_name(str(row.get("img_id") or row.get("image_name") or ""))
                for row in loaded["eval"]["table"]
            }),
        }
        for name, value in actual.items():
            if int(counts.get(name, -1)) != value:
                raise ValueError(
                    f"bundle manifest count mismatch for {name}: {counts.get(name)} vs {value}"
                )
        expected_rows = counts.get("evaluator_rows") or {}
        for kind in EVAL_KINDS:
            if int(expected_rows.get(kind, -1)) != len(loaded["eval"][kind]):
                raise ValueError(f"bundle manifest evaluator-row mismatch for {kind}")
        return loaded

    assert output is not None and eval_dir is not None
    output = _require_output(output)
    eval_dir = eval_dir.expanduser().resolve()
    print(f"[atlas] reading {label} frozen directories", flush=True)
    return {
        "source": str(output),
        "manifest": None,
        "trace": _read_jsonl(output / "recognition_trace.jsonl"),
        "run_summary": _read_json(output / "run_summary.json"),
        "eval": {
            kind: _read_json(_find_eval_result(eval_dir, kind))
            for kind in EVAL_KINDS
        },
        "official": _read_json(_find_metric_result(eval_dir)),
        "table_scores": _load_table_scores(table_scores),
        "teds_summary": _read_json(teds_summary.resolve()) if teds_summary else None,
    }


def _find_eval_result(root: Path, kind: str) -> Path:
    root = root.expanduser().resolve()
    matches = sorted(root.glob(f"*_{kind}_result.json"))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one *_{kind}_result.json under {root}, got {matches}"
        )
    return matches[0]


def _find_metric_result(root: Path) -> Path:
    root = root.expanduser().resolve()
    matches = sorted(root.glob("*_metric_result.json"))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one *_metric_result.json under {root}, got {matches}"
        )
    return matches[0]


def _counter(values: Iterable[Any]) -> dict[str, int]:
    counter = collections.Counter(str(value) for value in values)
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def _page_name(img_id: str) -> str:
    if img_id.endswith((".jpg", ".jpeg", ".png")):
        return img_id
    return "_".join(img_id.split("_")[:-1])


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _excerpt(text: Any, limit: int = 600) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    head = limit * 2 // 3
    tail = limit - head
    return value[:head] + f"\n…[{len(value) - limit} chars omitted]…\n" + value[-tail:]


def _visible_tokens(row: dict[str, Any]) -> list[int]:
    tokens = [int(value) for value in row.get("token_ids") or ()]
    if tokens and tokens[-1] == EOS_TOKEN_ID:
        tokens.pop()
    return tokens


def _common_prefix(left: Sequence[Any], right: Sequence[Any]) -> int:
    for index, (left_value, right_value) in enumerate(zip(left, right)):
        if left_value != right_value:
            return index
    return min(len(left), len(right))


def _sequence_ratio(left: Sequence[Any], right: Sequence[Any]) -> float | None:
    # SequenceMatcher with autojunk disabled can become effectively quadratic.
    # The frozen full run contains a multi-million-character table prediction;
    # exact/normalization checks remain valid, but a ratio is not worth hanging
    # the analysis for.
    if max(len(left), len(right)) > 100_000 or len(left) * len(right) > 50_000_000:
        return None
    return difflib.SequenceMatcher(None, left, right, autojunk=False).ratio()


def _compact_whitespace(text: str) -> str:
    return "".join(text.split())


def _outer_wrapper_normalize(text: str, kind: str | None) -> str:
    value = unicodedata.normalize("NFKC", text).strip()
    if kind not in {"display_formula", "formula"}:
        return value
    pairs = (("```latex", "```"), ("```math", "```"), ("$$", "$$"), ("\\[", "\\]"), ("$", "$"))
    changed = True
    while changed:
        changed = False
        for prefix, suffix in pairs:
            if value.startswith(prefix) and value.endswith(suffix) and len(value) >= len(prefix) + len(suffix):
                value = value[len(prefix) : len(value) - len(suffix)].strip()
                changed = True
    return _compact_whitespace(value)


def _alphanumeric_skeleton(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).casefold()
    return "".join(character for character in value if character.isalnum())


def _format_family(text: Any) -> str:
    value = str(text or "").strip().lower()
    if not value:
        return "empty"
    if "<fcel" in value:
        return "fcel"
    if "<table" in value:
        return "html"
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    pipe_lines = sum(line.count("|") >= 2 for line in lines)
    if pipe_lines >= 2 or (lines and pipe_lines / len(lines) >= 0.5):
        return "pipe"
    return "plain"


def _difference_class(reference: str, candidate: str, kind: str | None = None) -> str:
    if reference == candidate:
        return "exact"
    if unicodedata.normalize("NFKC", reference) == unicodedata.normalize("NFKC", candidate):
        return "unicode_normalization_only"
    if _compact_whitespace(reference) == _compact_whitespace(candidate):
        return "equal_after_all_whitespace_removal"
    if _compact_whitespace(unicodedata.normalize("NFKC", reference)) == _compact_whitespace(unicodedata.normalize("NFKC", candidate)):
        return "equal_after_nfkc_and_all_whitespace_removal"
    if _outer_wrapper_normalize(reference, kind) == _outer_wrapper_normalize(candidate, kind):
        return "approved_formula_wrapper_only"
    ratio = _sequence_ratio(reference, candidate)
    if ratio is None:
        return "oversized_content_difference"
    if ratio >= 0.95:
        return "content_changed_similarity_ge_0_95"
    if ratio >= 0.75:
        return "content_changed_similarity_ge_0_75"
    if ratio >= 0.40:
        return "content_changed_similarity_ge_0_40"
    return "content_changed_similarity_lt_0_40"


def _ngram_dominance(tokens: list[int]) -> float:
    if len(tokens) < 8:
        return 0.0
    best = 0.0
    for width in range(1, min(8, len(tokens)) + 1):
        counts = collections.Counter(
            tuple(tokens[index : index + width])
            for index in range(len(tokens) - width + 1)
        )
        repetitions = counts.most_common(1)[0][1]
        best = max(best, min(1.0, repetitions * width / len(tokens)))
    return best


def _tail_periodicity(tokens: list[int]) -> float:
    tail = tokens[-min(256, len(tokens)) :]
    if len(tail) < 16:
        return 0.0
    best = 0.0
    for period in range(1, min(32, len(tail) // 2) + 1):
        matches = sum(
            tail[index] == tail[index - period]
            for index in range(period, len(tail))
        )
        best = max(best, matches / (len(tail) - period))
    return best


def _script_counts(text: str) -> dict[str, int]:
    patterns = {
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
    counts: collections.Counter[str] = collections.Counter()
    for character in text:
        if not character.isalpha():
            continue
        name = unicodedata.name(character, "")
        for script, needles in patterns.items():
            if any(needle in name for needle in needles):
                counts[script] += 1
                break
    return dict(counts)


def _degeneration_flags(reference: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    ref_tokens = _visible_tokens(reference)
    cand_tokens = _visible_tokens(candidate)
    ref_count = len(ref_tokens)
    cand_count = len(cand_tokens)
    ref_text = str(reference.get("text") or "")
    cand_text = str(candidate.get("text") or "")
    flags = []
    if cand_count >= 128 and cand_count >= max(ref_count * 3, ref_count + 128):
        flags.append("candidate_runaway_length")
    ref_dominance = _ngram_dominance(ref_tokens)
    cand_dominance = _ngram_dominance(cand_tokens)
    ref_periodicity = _tail_periodicity(ref_tokens)
    cand_periodicity = _tail_periodicity(cand_tokens)
    if cand_count >= 64 and (
        (cand_dominance >= 0.45 and cand_dominance > ref_dominance + 0.10)
        or (cand_periodicity >= 0.88 and cand_periodicity > ref_periodicity + 0.05)
    ):
        flags.append("candidate_repetition")
    ref_scripts = _script_counts(ref_text)
    cand_scripts = _script_counts(cand_text)
    added = {
        name: count
        for name, count in cand_scripts.items()
        if count >= 4 and ref_scripts.get(name, 0) == 0
    }
    if added:
        flags.append("candidate_added_script")
    if abs(cand_count - ref_count) >= 128:
        flags.append("large_length_delta")
    text_ratio = _sequence_ratio(ref_text, cand_text)
    if text_ratio is not None and text_ratio < 0.30 and max(ref_count, cand_count) >= 16:
        flags.append("low_text_similarity")
    return flags


def _input_proof(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    left = reference.get("input_fingerprints") or {}
    right = candidate.get("input_fingerprints") or {}
    left_crop = (left.get("crop") or {}).get("sha256")
    right_crop = (right.get("crop") or {}).get("sha256")
    left_prepared = left.get("prepared_inputs_sha256")
    right_prepared = right.get("prepared_inputs_sha256")
    tensor_names = sorted(set((left.get("tensors") or {})) | set((right.get("tensors") or {})))
    tensors = {}
    for name in tensor_names:
        left_hash = ((left.get("tensors") or {}).get(name) or {}).get("sha256")
        right_hash = ((right.get("tensors") or {}).get(name) or {}).get("sha256")
        tensors[name] = (
            "unavailable"
            if not left_hash or not right_hash
            else ("exact" if left_hash == right_hash else "different")
        )
    statuses = {
        "crop": (
            "unavailable"
            if not left_crop or not right_crop
            else ("exact" if left_crop == right_crop else "different")
        ),
        "prepared": (
            "unavailable"
            if not left_prepared or not right_prepared
            else ("exact" if left_prepared == right_prepared else "different")
        ),
        "tensors": tensors,
    }
    flat = [statuses["crop"], statuses["prepared"], *tensors.values()]
    statuses["overall"] = (
        "different"
        if "different" in flat
        else ("exact" if flat and all(value == "exact" for value in flat) else "unavailable")
    )
    return statuses


def _input_status(reference: dict[str, Any], candidate: dict[str, Any]) -> str:
    return str(_input_proof(reference, candidate)["overall"])


def _trace_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["source_image_name"]), int(row["block_index"])


def _route_summary(row: dict[str, Any]) -> dict[str, Any]:
    vision = row.get("vision") or {}
    text = row.get("text_prefill") or {}
    return {
        "vision": {
            key: vision.get(key)
            for key in ("execution", "bucket", "packing", "pack_crops", "pack_sequence_length", "pack_row_sizes")
        },
        "text_prefill": {
            key: text.get(key)
            for key in ("execution", "bucket", "packing", "pack_members", "segment_lengths", "private_cache_generation")
        },
        "decode_slot_epoch": row.get("decode_slot_epoch"),
    }


def _analyze_generations(reference_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    reference = {_trace_key(row): row for row in reference_rows}
    candidate = {_trace_key(row): row for row in candidate_rows}
    if len(reference) != len(reference_rows) or len(candidate) != len(candidate_rows):
        raise ValueError("duplicate source-image/block keys in recognition trace")
    all_keys = sorted(reference.keys() | candidate.keys())
    records = []
    for key in all_keys:
        left = reference.get(key)
        right = candidate.get(key)
        pair_status = (
            "paired"
            if left is not None and right is not None
            else ("reference_only" if left is not None else "candidate_only")
        )
        left = left or {}
        right = right or {}
        left_text = str(left.get("text") or "")
        right_text = str(right.get("text") or "")
        left_raw_tokens = [int(value) for value in left.get("token_ids") or ()]
        right_raw_tokens = [int(value) for value in right.get("token_ids") or ()]
        left_tokens = _visible_tokens(left)
        right_tokens = _visible_tokens(right)
        flags = (
            _degeneration_flags(left, right)
            if pair_status == "paired"
            and (left_text != right_text or left_raw_tokens != right_raw_tokens)
            else []
        )
        label = str(right.get("label") or left.get("label") or "")
        input_proof = (
            _input_proof(left, right)
            if pair_status == "paired"
            else {"overall": "missing_side", "crop": "missing_side", "prepared": "missing_side", "tensors": {}}
        )
        text_ratio = (
            (1.0 if left_text == right_text else _sequence_ratio(left_text, right_text))
            if pair_status == "paired"
            else None
        )
        token_ratio = (
            (1.0 if left_tokens == right_tokens else _sequence_ratio(left_tokens, right_tokens))
            if pair_status == "paired"
            else None
        )
        record = {
            "source_image_name": key[0],
            "block_index": key[1],
            "pair_status": pair_status,
            "reference_request_id": left.get("request_id"),
            "candidate_request_id": right.get("request_id"),
            "reference_label": left.get("label"),
            "candidate_label": right.get("label"),
            "reference_prompt": left.get("prompt"),
            "candidate_prompt": right.get("prompt"),
            "input_status": input_proof["overall"],
            "input_proof": input_proof,
            "difference_class": (
                _difference_class(left_text, right_text, label)
                if pair_status == "paired"
                else None
            ),
            "same_alphanumeric_skeleton": bool(
                pair_status == "paired"
                and _alphanumeric_skeleton(left_text)
                and _alphanumeric_skeleton(left_text) == _alphanumeric_skeleton(right_text)
            ),
            "reference_format": _format_family(left_text),
            "candidate_format": _format_family(right_text),
            "reference_tokens": len(left_tokens),
            "candidate_tokens": len(right_tokens),
            "reference_token_ids_including_eos": left_raw_tokens,
            "candidate_token_ids_including_eos": right_raw_tokens,
            "token_ids_exact_including_eos": (
                left_raw_tokens == right_raw_tokens if pair_status == "paired" else None
            ),
            "visible_token_ids_exact": (
                left_tokens == right_tokens if pair_status == "paired" else None
            ),
            "first_divergence_token": (
                _common_prefix(left_tokens, right_tokens) if pair_status == "paired" else None
            ),
            "token_sequence_ratio": token_ratio,
            "text_sequence_ratio": text_ratio,
            "triage_flags": flags,
            "reference_stop_reason": left.get("stop_reason"),
            "candidate_stop_reason": right.get("stop_reason"),
            "reference_text_sha256": _sha256_text(left_text),
            "candidate_text_sha256": _sha256_text(right_text),
            "reference_text": left_text,
            "candidate_text": right_text,
            "reference_route": _route_summary(left),
            "candidate_route": _route_summary(right),
        }
        records.append(record)

    table_left = [row for row in reference_rows if row.get("label") == "table"]
    table_right = [row for row in candidate_rows if row.get("label") == "table"]
    table_transitions = collections.Counter(
        (record["reference_format"], record["candidate_format"])
        for record in records
        if record["pair_status"] == "paired"
        and record["reference_label"] == "table"
        and record["candidate_label"] == "table"
    )
    verified_table_transitions = collections.Counter(
        (record["reference_format"], record["candidate_format"])
        for record in records
        if record["pair_status"] == "paired"
        and record["reference_label"] == "table"
        and record["candidate_label"] == "table"
        and record["input_status"] == "exact"
    )
    logit_candidates = [
        record
        for record in records
        if record["reference_label"] == "table"
        and record["candidate_label"] == "table"
        and record["input_status"] == "exact"
        and record["reference_format"] == "fcel"
        and record["candidate_format"] in {"pipe", "plain"}
    ]
    logit_candidates.sort(
        key=lambda record: (
            record["first_divergence_token"] != 0,
            record["candidate_format"],
            record["first_divergence_token"],
            max(record["reference_tokens"], record["candidate_tokens"]),
            record["source_image_name"],
            record["block_index"],
        )
    )
    summary = {
        "reference_requests": len(reference_rows),
        "candidate_requests": len(candidate_rows),
        "shared_requests": len(reference.keys() & candidate.keys()),
        "reference_only": len(reference.keys() - candidate.keys()),
        "candidate_only": len(candidate.keys() - reference.keys()),
        "pair_status": _counter(record["pair_status"] for record in records),
        "ambiguous_duplicate_identity_rows": {
            "reference": sum("duplicate_lane" in key for key in reference),
            "candidate": sum("duplicate_lane" in key for key in candidate),
            "note": "lane-only localization rows; not interpreted as cross-device harms",
        },
        "difference_class": _counter(record["difference_class"] for record in records),
        "input_status": _counter(record["input_status"] for record in records),
        "triage_flag": _counter(flag for record in records for flag in record["triage_flags"]),
        "requests_with_any_triage_flag": sum(bool(record["triage_flags"]) for record in records),
        "by_label_and_difference_class": _counter(
            f"{record['candidate_label']}::{record['difference_class']}" for record in records
        ),
        "reference_table_requests": len(table_left),
        "candidate_table_requests": len(table_right),
        "reference_table_formats": _counter(_format_family(row.get("text")) for row in table_left),
        "candidate_table_formats": _counter(_format_family(row.get("text")) for row in table_right),
        "stable_key_table_format_transitions": {
            f"{left}->{right}": count
            for (left, right), count in sorted(table_transitions.items(), key=lambda item: (-item[1], item[0]))
        },
        "verified_exact_input_table_format_transitions": {
            f"{left}->{right}": count
            for (left, right), count in sorted(verified_table_transitions.items(), key=lambda item: (-item[1], item[0]))
        },
        "stable_key_table_transitions_without_exact_input_proof": sum(table_transitions.values()) - sum(verified_table_transitions.values()),
        "exact_target_input_table_candidates": len(logit_candidates),
        "exact_target_input_table_candidates_token0": sum(
            record["first_divergence_token"] == 0 for record in logit_candidates
        ),
    }
    return summary, records, logit_candidates


def _gt_base_key(sample: dict[str, Any], lane: str) -> tuple[str, ...]:
    indices = sample.get("gt_idx")
    if indices is None:
        indices = []
    normalized_indices = indices if isinstance(indices, list) else [indices]
    has_gt_index = any(value not in (None, "") for value in normalized_indices)
    page = _page_name(str(sample.get("img_id") or sample.get("image_name") or ""))
    raw_gt = _raw_sample_text(sample, "gt")
    normalized_gt = _normalized_sample_text(sample, "gt")
    if not has_gt_index and not raw_gt and not normalized_gt:
        run_specific = {
            key: sample.get(key)
            for key in (
                "pred_idx",
                "pred",
                "norm_pred",
                "pred_position",
                "pred_category_type",
            )
        }
        return (
            "pred_only",
            lane,
            page,
            _sha256_text(json.dumps(run_specific, ensure_ascii=False, sort_keys=True)),
        )
    identity = {
        "indices": indices,
        "category": sample.get("gt_category_type"),
        "position": sample.get("gt_position"),
        "attribute": sample.get("gt_attribute"),
        "raw_gt": raw_gt,
        "normalized_gt": normalized_gt,
    }
    return (
        "gt",
        page,
        _sha256_text(json.dumps(identity, ensure_ascii=False, sort_keys=True)),
    )


def _indexed_samples(samples: list[dict[str, Any]], lane: str) -> dict[tuple[str, ...], dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    for sample in samples:
        grouped[_gt_base_key(sample, lane)].append(sample)
    indexed = {}
    for key, rows in grouped.items():
        if len(rows) == 1:
            indexed[key] = rows[0]
            continue
        # Duplicate GT identities are retained as lane-only localization rows.
        # Their page totals remain exact, but we refuse to create a cross-device
        # element pairing from incidental array order.
        for index, sample in enumerate(rows):
            indexed[key + ("duplicate_lane", lane, str(index))] = sample
    return indexed


def _indexed_reading_order(samples: list[dict[str, Any]]) -> dict[tuple[str, ...], dict[str, Any]]:
    indexed = {}
    for sample in samples:
        page = _page_name(str(sample.get("img_id") or sample.get("image_name") or ""))
        key = ("reading_order_page", page)
        if key in indexed:
            raise ValueError(f"duplicate reading-order result for page {page}")
        indexed[key] = sample
    return indexed


def _raw_sample_text(sample: dict[str, Any] | None, name: str) -> str:
    if sample is None:
        return ""
    return str(sample.get(name) or "")


def _normalized_sample_text(sample: dict[str, Any] | None, name: str) -> str:
    if sample is None:
        return ""
    normalized_key = f"norm_{name}"
    if normalized_key in sample and sample[normalized_key] not in (None, ""):
        return str(sample[normalized_key])
    return str(sample.get(name) or "")


def _edit_components(samples: dict[Any, dict[str, Any]]) -> tuple[dict[Any, float], float, dict[str, dict[str, float]]]:
    page_upper: collections.Counter[str] = collections.Counter()
    values = {}
    for key, sample in samples.items():
        page = _page_name(str(sample.get("img_id") or sample.get("image_name") or ""))
        if "upper_len" not in sample or "Edit_num" not in sample:
            raise ValueError(
                f"evaluator Edit result is missing upper_len/Edit_num: {key}"
            )
        if "Edit_dist" not in (sample.get("metric") or {}):
            raise ValueError(
                f"evaluator Edit result is missing metric.Edit_dist: {key}"
            )
        upper = int(sample["upper_len"])
        edit_num = sample.get("Edit_num")
        values[key] = (page, float(edit_num), upper)
        page_upper[page] += upper
    pages = sorted(page_upper)
    if not pages:
        return {}, math.nan, {}
    contributions = {
        key: (edit_num / page_upper[page] / len(pages) if page_upper[page] else 0.0)
        for key, (page, edit_num, _upper) in values.items()
    }
    page_edit: collections.Counter[str] = collections.Counter()
    page_samples: collections.Counter[str] = collections.Counter()
    for page, edit_num, _upper in values.values():
        page_edit[page] += edit_num
        page_samples[page] += 1
    page_scores = {
        page: {
            "edit_num": page_edit[page],
            "upper_len": float(page_upper[page]),
            "sample_count": float(page_samples[page]),
            "score": page_edit[page] / page_upper[page] if page_upper[page] else 0.0,
            "aggregate_contribution": (
                page_edit[page] / page_upper[page] / len(pages)
                if page_upper[page]
                else 0.0
            ),
        }
        for page in pages
    }
    return contributions, sum(contributions.values()), page_scores


def _load_table_scores(path: Path | None) -> dict[str, dict[str, float]]:
    if path is None:
        return {}
    payload = _read_json(path.resolve())
    if not isinstance(payload, dict):
        raise TypeError(f"expected table-score object: {path}")
    return payload


def _official_metric(
    payload: dict[str, Any],
    kind: str,
    metric: str,
    weighting: str = "page",
) -> float:
    section = payload.get(kind) or {}
    if metric == "Edit_dist":
        value = ((section.get("page") or {}).get(metric) or {}).get("ALL")
        if value is None:
            value = ((section.get("all") or {}).get(metric) or {}).get("ALL_page_avg")
    elif weighting == "page":
        value = ((section.get("page") or {}).get(metric) or {}).get("ALL")
    else:
        value = ((section.get("all") or {}).get(metric) or {}).get("all")
    if value in (None, "NaN"):
        raise ValueError(f"official metric is missing: {kind}.{metric}.{weighting}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"official metric is non-finite: {kind}.{metric}.{weighting}")
    return result


def _reconcile_edit_summary(
    summary: dict[str, Any],
    reference_official: dict[str, Any],
    candidate_official: dict[str, Any],
    kind: str,
) -> None:
    reference = _official_metric(reference_official, kind, "Edit_dist")
    candidate = _official_metric(candidate_official, kind, "Edit_dist")
    summary["official_reference"] = reference
    summary["official_candidate"] = candidate
    summary["official_candidate_minus_reference"] = candidate - reference
    tolerance = 1e-9
    if abs(summary["reference_page_average"] - reference) > tolerance:
        raise ValueError(
            f"{kind} reference Edit_dist recompute mismatch: "
            f"{summary['reference_page_average']} vs official {reference}"
        )
    if abs(summary["candidate_page_average"] - candidate) > tolerance:
        raise ValueError(
            f"{kind} candidate Edit_dist recompute mismatch: "
            f"{summary['candidate_page_average']} vs official {candidate}"
        )


def _reconcile_teds_summary(
    summary: dict[str, Any],
    reference_official: dict[str, Any],
    candidate_official: dict[str, Any],
    reference_teds_summary: dict[str, Any] | None,
    candidate_teds_summary: dict[str, Any] | None,
    metric: str,
    page_weighted: bool,
) -> None:
    weighting = "page" if page_weighted else "sample"

    def authority(
        official: dict[str, Any],
        corrected: dict[str, Any] | None,
    ) -> tuple[float, str]:
        if corrected is not None:
            if page_weighted:
                return float(corrected["page_aggregate"][metric]["ALL"]), "corrected_process_isolated"
            return float(corrected["sample_aggregate"][metric]["all"]), "corrected_process_isolated"
        return _official_metric(official, "table", metric, weighting), "frozen_evaluator"

    reference, reference_authority = authority(reference_official, reference_teds_summary)
    candidate, candidate_authority = authority(candidate_official, candidate_teds_summary)
    summary.update(
        {
            "official_reference": reference,
            "official_candidate": candidate,
            "reference_authority": reference_authority,
            "candidate_authority": candidate_authority,
            "official_candidate_minus_reference": candidate - reference,
        }
    )
    tolerance = 1e-9
    if abs(summary["reference"] - reference) > tolerance:
        raise ValueError(
            f"reference {metric} {weighting} recompute mismatch: "
            f"{summary['reference']} vs authority {reference}"
        )
    if abs(summary["candidate"] - candidate) > tolerance:
        raise ValueError(
            f"candidate {metric} {weighting} recompute mismatch: "
            f"{summary['candidate']} vs authority {candidate}"
        )


def _table_score_key(sample: dict[str, Any]) -> str:
    return str(sample["img_id"]) + "_" + str(sample.get("gt_idx"))


def _score(sample: dict[str, Any], metric: str, overrides: dict[str, dict[str, float]]) -> float:
    key = _table_score_key(sample)
    if overrides:
        if key not in overrides or metric not in overrides[key]:
            raise ValueError(f"corrected TEDS map is missing {key}.{metric}")
        value = float(overrides[key][metric])
    else:
        metrics = sample.get("metric") or {}
        if metric not in metrics:
            raise ValueError(f"frozen table row is missing metric.{metric}: {key}")
        value = float(metrics[metric])
    if not math.isfinite(value):
        raise ValueError(f"non-finite table score {key}.{metric}: {value}")
    return value


def _score_components(samples: dict[Any, dict[str, Any]], metric: str, overrides: dict[str, dict[str, float]], page_weighted: bool) -> tuple[dict[Any, float], float, dict[str, dict[str, float]]]:
    if not samples:
        return {}, math.nan, {}
    if not page_weighted:
        contributions = {key: _score(sample, metric, overrides) / len(samples) for key, sample in samples.items()}
        by_page_values: dict[str, list[float]] = collections.defaultdict(list)
        for sample in samples.values():
            by_page_values[_page_name(str(sample.get("img_id") or ""))].append(
                _score(sample, metric, overrides)
            )
        page_scores = {
            page: {
                "score": sum(values) / len(values),
                "sample_count": float(len(values)),
                "aggregate_contribution": sum(values) / len(samples),
            }
            for page, values in by_page_values.items()
        }
        return contributions, sum(contributions.values()), page_scores
    by_page: collections.Counter[str] = collections.Counter(
        _page_name(str(sample.get("img_id") or "")) for sample in samples.values()
    )
    page_count = len(by_page)
    contributions = {
        key: _score(sample, metric, overrides) / by_page[_page_name(str(sample.get("img_id") or ""))] / page_count
        for key, sample in samples.items()
    }
    page_values: collections.Counter[str] = collections.Counter()
    for sample in samples.values():
        page = _page_name(str(sample.get("img_id") or ""))
        page_values[page] += _score(sample, metric, overrides)
    page_scores = {
        page: {
            "score": page_values[page] / by_page[page],
            "sample_count": float(by_page[page]),
            "aggregate_contribution": page_values[page] / by_page[page] / page_count,
        }
        for page in sorted(by_page)
    }
    return contributions, sum(contributions.values()), page_scores


def _validate_teds_evidence(
    samples: list[dict[str, Any]],
    scores: dict[str, dict[str, float]],
    summary: dict[str, Any] | None,
) -> None:
    if bool(scores) != bool(summary):
        raise ValueError("corrected TEDS scores require their teds_only_summary")
    if not scores:
        return
    expected = {_table_score_key(sample) for sample in samples}
    actual = set(scores)
    if expected != actual:
        raise ValueError(
            "corrected TEDS key coverage mismatch: "
            f"missing={len(expected - actual)} extra={len(actual - expected)}"
        )
    pages = {_page_name(str(sample.get("img_id") or "")) for sample in samples}
    assert summary is not None
    if int(summary.get("sample_count", -1)) != len(samples):
        raise ValueError("corrected TEDS summary sample_count mismatch")
    if int(summary.get("page_count", -1)) != len(pages):
        raise ValueError("corrected TEDS summary page_count mismatch")
    execution = summary.get("execution") or {}
    if execution.get("scheduler") != "process_isolated_parent_timeout":
        raise ValueError("corrected TEDS summary did not use the process-isolated scheduler")
    if int(execution.get("sample_count", -1)) != len(samples):
        raise ValueError("corrected TEDS execution sample_count mismatch")
    if int(execution.get("error_case_count", -1)) != 0:
        raise ValueError("corrected TEDS summary still contains metric execution errors")
    for metric in ("TEDS", "TEDS_structure_only"):
        for value in (
            summary["sample_aggregate"][metric]["all"],
            summary["page_aggregate"][metric]["ALL"],
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"corrected TEDS summary has non-finite {metric}")
    for key, payload in scores.items():
        for metric in ("TEDS", "TEDS_structure_only"):
            if metric not in payload or not math.isfinite(float(payload[metric])):
                raise ValueError(f"invalid corrected score {key}.{metric}")


def _validate_frozen_teds_authority(
    official: dict[str, Any],
    corrected_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    if corrected_summary is not None:
        execution = corrected_summary.get("execution") or {}
        return {
            "authority": "corrected_process_isolated",
            "timeout_case_count": int(execution.get("timeout_case_count", 0)),
            "error_case_count": int(execution.get("error_case_count", 0)),
            "timeout_cases": execution.get("timeout_cases") or [],
        }
    debug = (((official.get("table") or {}).get("metric_debug") or {}).get("TEDS") or {})
    errors = int(debug.get("error_case_count", 0))
    if errors:
        raise ValueError(
            "frozen evaluator TEDS contains execution errors; corrected sidecars are required"
        )
    return {
        "authority": "frozen_evaluator",
        "timeout_case_count": int(debug.get("timeout_case_count", 0)),
        "error_case_count": errors,
        "timeout_cases": debug.get("timeout_cases") or [],
    }


def _concentration(losses: list[float]) -> dict[str, Any]:
    positive = sorted((value for value in losses if value > 0), reverse=True)
    gross_worse = sum(positive)
    improvements = -sum(value for value in losses if value < 0)
    net = sum(losses)
    result: dict[str, Any] = {
        "net_candidate_loss": net,
        "gross_candidate_worse": gross_worse,
        "gross_candidate_better": improvements,
        "positive_loss_records": len(positive),
    }
    for fraction in (0.01, 0.05, 0.10, 0.25):
        count = max(1, math.ceil(len(losses) * fraction)) if losses else 0
        top = sum(positive[:count])
        result[f"top_{int(fraction * 100)}pct"] = {
            "record_count": count,
            "loss": top,
            "fraction_of_gross_worse": top / gross_worse if gross_worse else 0.0,
            "fraction_of_net_loss": top / net if net > 0 else None,
        }
    return result


def _metric_record(key: Any, reference: dict[str, Any] | None, candidate: dict[str, Any] | None, reference_contribution: float, candidate_contribution: float, higher_is_better: bool, metric: str, kind: str) -> dict[str, Any]:
    reference_gt_raw = _raw_sample_text(reference, "gt")
    candidate_gt_raw = _raw_sample_text(candidate, "gt")
    gt_raw = candidate_gt_raw if candidate is not None else reference_gt_raw
    ref_pred_raw = _raw_sample_text(reference, "pred")
    cand_pred_raw = _raw_sample_text(candidate, "pred")
    reference_gt_normalized = _normalized_sample_text(reference, "gt")
    candidate_gt_normalized = _normalized_sample_text(candidate, "gt")
    gt_normalized = candidate_gt_normalized if candidate is not None else reference_gt_normalized
    ref_pred_normalized = _normalized_sample_text(reference, "pred")
    cand_pred_normalized = _normalized_sample_text(candidate, "pred")
    loss = (reference_contribution - candidate_contribution) if higher_is_better else (candidate_contribution - reference_contribution)
    sample = candidate or reference or {}
    paired = reference is not None and candidate is not None
    textual = kind != "reading_order"
    raw_class = (
        _difference_class(ref_pred_raw, cand_pred_raw, kind)
        if paired and textual
        else None
    )
    normalized_class = (
        _difference_class(ref_pred_normalized, cand_pred_normalized, kind)
        if paired and textual
        else None
    )
    return {
        "metric": metric,
        "image_name": _page_name(str(sample.get("img_id") or sample.get("image_name") or "")),
        "gt_idx": sample.get("gt_idx"),
        "gt_category_type": sample.get("gt_category_type"),
        "pair_status": "paired" if paired else ("reference_only" if reference is not None else "candidate_only"),
        "gt_exact_between_runs": bool(
            reference is not None
            and candidate is not None
            and _raw_sample_text(reference, "gt")
            == _raw_sample_text(candidate, "gt")
            and _normalized_sample_text(reference, "gt")
            == _normalized_sample_text(candidate, "gt")
        ),
        "reference_contribution": reference_contribution,
        "candidate_contribution": candidate_contribution,
        "candidate_loss_contribution": loss,
        "raw_difference_class": raw_class,
        "normalized_difference_class": normalized_class,
        "evaluator_normalized_exact": (
            ref_pred_normalized == cand_pred_normalized if paired and textual else None
        ),
        "same_alphanumeric_skeleton": bool(
            paired
            and textual
            and _alphanumeric_skeleton(ref_pred_raw)
            and _alphanumeric_skeleton(ref_pred_raw) == _alphanumeric_skeleton(cand_pred_raw)
        ),
        "reference_format": _format_family(ref_pred_raw),
        "candidate_format": _format_family(cand_pred_raw),
        "gt_length": len(gt_normalized),
        "reference_pred_length": len(ref_pred_normalized),
        "candidate_pred_length": len(cand_pred_normalized),
        "reference_gt_sha256": _sha256_text(reference_gt_raw),
        "candidate_gt_sha256": _sha256_text(candidate_gt_raw),
        "reference_pred_sha256": _sha256_text(ref_pred_raw),
        "candidate_pred_sha256": _sha256_text(cand_pred_raw),
        "reference_gt": reference_gt_raw,
        "candidate_gt": candidate_gt_raw,
        "gt": gt_raw,
        "reference_pred": ref_pred_raw,
        "candidate_pred": cand_pred_raw,
        "normalized_reference_gt": reference_gt_normalized,
        "normalized_candidate_gt": candidate_gt_normalized,
        "normalized_gt": gt_normalized,
        "normalized_reference_pred": ref_pred_normalized,
        "normalized_candidate_pred": cand_pred_normalized,
        "stored_reference_norm_gt": reference.get("norm_gt") if reference else None,
        "stored_candidate_norm_gt": candidate.get("norm_gt") if candidate else None,
        "stored_reference_norm_pred": reference.get("norm_pred") if reference else None,
        "stored_candidate_norm_pred": candidate.get("norm_pred") if candidate else None,
        "reference_edit_num": reference.get("Edit_num") if reference else None,
        "candidate_edit_num": candidate.get("Edit_num") if candidate else None,
        "reference_upper_len": reference.get("upper_len") if reference else None,
        "candidate_upper_len": candidate.get("upper_len") if candidate else None,
        "reference_sample_metric": (reference.get("metric") or {}) if reference else None,
        "candidate_sample_metric": (candidate.get("metric") or {}) if candidate else None,
        "reference_gt_sequence": reference.get("gt") if reference and kind == "reading_order" else None,
        "reference_pred_sequence": reference.get("pred") if reference and kind == "reading_order" else None,
        "candidate_gt_sequence": candidate.get("gt") if candidate and kind == "reading_order" else None,
        "candidate_pred_sequence": candidate.get("pred") if candidate and kind == "reading_order" else None,
    }


def _page_delta_records(
    metric: str,
    reference_pages: dict[str, dict[str, float]],
    candidate_pages: dict[str, dict[str, float]],
    higher_is_better: bool,
) -> list[dict[str, Any]]:
    if set(reference_pages) != set(candidate_pages):
        raise ValueError(
            f"{metric} page universes differ: "
            f"reference_only={len(set(reference_pages) - set(candidate_pages))} "
            f"candidate_only={len(set(candidate_pages) - set(reference_pages))}"
        )
    records = []
    for page in sorted(reference_pages):
        reference = reference_pages[page]
        candidate = candidate_pages[page]
        reference_contribution = float(reference["aggregate_contribution"])
        candidate_contribution = float(candidate["aggregate_contribution"])
        loss = (
            reference_contribution - candidate_contribution
            if higher_is_better
            else candidate_contribution - reference_contribution
        )
        records.append(
            {
                "record_type": "page_metric_contribution",
                "metric": metric,
                "image_name": page,
                "reference_score": reference["score"],
                "candidate_score": candidate["score"],
                "reference_contribution": reference_contribution,
                "candidate_contribution": candidate_contribution,
                "candidate_loss_contribution": loss,
                "reference_operands": reference,
                "candidate_operands": candidate,
            }
        )
    return records


def _analyze_edit_metric(kind: str, reference_samples: list[dict[str, Any]], candidate_samples: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if kind == "reading_order":
        reference = _indexed_reading_order(reference_samples)
        candidate = _indexed_reading_order(candidate_samples)
    else:
        reference = _indexed_samples(reference_samples, "reference")
        candidate = _indexed_samples(candidate_samples, "candidate")
    ref_contrib, ref_aggregate, ref_pages = _edit_components(reference)
    cand_contrib, cand_aggregate, cand_pages = _edit_components(candidate)
    records = [
        _metric_record(
            key,
            reference.get(key),
            candidate.get(key),
            ref_contrib.get(key, 0.0),
            cand_contrib.get(key, 0.0),
            False,
            f"{kind}.Edit_dist",
            kind,
        )
        for key in sorted(reference.keys() | candidate.keys())
    ]
    page_records = _page_delta_records(
        f"{kind}.Edit_dist", ref_pages, cand_pages, False
    )
    summary = {
        "reference_samples": len(reference),
        "candidate_samples": len(candidate),
        "paired_samples": sum(record["pair_status"] == "paired" for record in records),
        "reference_page_average": ref_aggregate,
        "candidate_page_average": cand_aggregate,
        "candidate_minus_reference": cand_aggregate - ref_aggregate,
        "pair_status": _counter(record["pair_status"] for record in records),
        "raw_difference_class": _counter(record["raw_difference_class"] for record in records),
        "normalized_difference_class": _counter(record["normalized_difference_class"] for record in records),
        "page_concentration": _concentration([record["candidate_loss_contribution"] for record in page_records]),
        "sample_localization_concentration": _concentration([
            record["candidate_loss_contribution"]
            for record in records
            if record["pair_status"] == "paired"
        ]),
        "attribution_note": "page records exactly recompose the official metric; sample rows are denominator-coupled localization evidence, not causal attribution",
    }
    return summary, records, page_records


def _analyze_teds(reference_samples: list[dict[str, Any]], candidate_samples: list[dict[str, Any]], reference_scores: dict[str, dict[str, float]], candidate_scores: dict[str, dict[str, float]], metric: str, page_weighted: bool) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    reference = _indexed_samples(reference_samples, "reference")
    candidate = _indexed_samples(candidate_samples, "candidate")
    ref_contrib, ref_aggregate, ref_pages = _score_components(reference, metric, reference_scores, page_weighted)
    cand_contrib, cand_aggregate, cand_pages = _score_components(candidate, metric, candidate_scores, page_weighted)
    records = [
        _metric_record(
            key,
            reference.get(key),
            candidate.get(key),
            ref_contrib.get(key, 0.0),
            cand_contrib.get(key, 0.0),
            True,
            f"table.{metric}.{'page' if page_weighted else 'sample'}",
            "table",
        )
        for key in sorted(reference.keys() | candidate.keys())
    ]
    page_records = _page_delta_records(
        f"table.{metric}.{'page' if page_weighted else 'sample'}",
        ref_pages,
        cand_pages,
        True,
    )
    summary = {
        "weighting": "page_then_mean" if page_weighted else "sample_mean",
        "reference": ref_aggregate,
        "candidate": cand_aggregate,
        "candidate_minus_reference": cand_aggregate - ref_aggregate,
        "candidate_loss": ref_aggregate - cand_aggregate,
        "page_concentration": _concentration([record["candidate_loss_contribution"] for record in page_records]),
        "sample_localization_concentration": _concentration([
            record["candidate_loss_contribution"]
            for record in records
            if record["pair_status"] == "paired"
        ]),
    }
    return summary, records, page_records


def _order_shape(sample: dict[str, Any] | None) -> str:
    if sample is None:
        return "missing_sample"
    gt = list(sample.get("gt") or ())
    pred = list(sample.get("pred") or ())
    if gt == pred:
        return "exact"
    gt_count = collections.Counter(gt)
    pred_count = collections.Counter(pred)
    if gt_count == pred_count:
        return "same_members_reordered"
    if not (pred_count - gt_count):
        return "missing_members"
    if not (gt_count - pred_count):
        return "extra_members"
    return "mixed_membership_and_order"


def _inversions(sample: dict[str, Any] | None) -> int:
    if sample is None:
        return 0
    gt = list(sample.get("gt") or ())
    pred = list(sample.get("pred") or ())
    def occurrences(values: list[Any]) -> list[tuple[Any, int]]:
        seen: collections.Counter[Any] = collections.Counter()
        tagged = []
        for value in values:
            tagged.append((value, seen[value]))
            seen[value] += 1
        return tagged

    positions = {value: index for index, value in enumerate(occurrences(gt))}
    sequence = [positions[value] for value in occurrences(pred) if value in positions]
    return sum(sequence[i] > sequence[j] for i in range(len(sequence)) for j in range(i + 1, len(sequence)))


def _augment_reading_order(records: list[dict[str, Any]], reference_samples: list[dict[str, Any]], candidate_samples: list[dict[str, Any]]) -> dict[str, Any]:
    reference = _indexed_reading_order(reference_samples)
    candidate = _indexed_reading_order(candidate_samples)
    shapes = collections.Counter()
    for record, key in zip(records, sorted(reference.keys() | candidate.keys())):
        left = reference.get(key)
        right = candidate.get(key)
        record["reference_order_shape"] = _order_shape(left)
        record["candidate_order_shape"] = _order_shape(right)
        record["reference_inversions"] = _inversions(left)
        record["candidate_inversions"] = _inversions(right)
        record["reference_missing_order_members"] = dict(
            collections.Counter((left or {}).get("gt") or ())
            - collections.Counter((left or {}).get("pred") or ())
        )
        record["candidate_missing_order_members"] = dict(
            collections.Counter((right or {}).get("gt") or ())
            - collections.Counter((right or {}).get("pred") or ())
        )
        record["reference_extra_order_members"] = dict(
            collections.Counter((left or {}).get("pred") or ())
            - collections.Counter((left or {}).get("gt") or ())
        )
        record["candidate_extra_order_members"] = dict(
            collections.Counter((right or {}).get("pred") or ())
            - collections.Counter((right or {}).get("gt") or ())
        )
        shapes[(record["reference_order_shape"], record["candidate_order_shape"])] += 1
    return {f"{left}->{right}": count for (left, right), count in sorted(shapes.items(), key=lambda item: (-item[1], item[0]))}


def _audit_pred_index_zero(eval_side: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    cases = []
    for kind in ("text_block", "display_formula", "table"):
        for sample in eval_side[kind]:
            pred_idx = sample.get("pred_idx")
            indices = pred_idx if isinstance(pred_idx, list) else [pred_idx]
            if (
                0 in indices
                and bool(sample.get("pred"))
                and sample.get("pred_position") in (None, "", [])
            ):
                cases.append(
                    {
                        "kind": kind,
                        "image_name": _page_name(str(sample.get("img_id") or sample.get("image_name") or "")),
                        "gt_idx": sample.get("gt_idx"),
                        "pred_idx": pred_idx,
                    }
                )
    return {
        "case_count": len(cases),
        "page_count": len({case["image_name"] for case in cases}),
        "pages": sorted({case["image_name"] for case in cases}),
        "cases": cases,
        "note": "Pinned evaluator truthiness-check defect: a valid pred_idx=0 can receive an empty pred_position and then look missing in reading order",
    }


def _audit_evaluator_gt_universes(
    reference: dict[str, list[dict[str, Any]]],
    candidate: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    by_kind: dict[str, dict[str, Any]] = {}
    difference_records: list[dict[str, Any]] = []
    for kind in EVAL_KINDS:
        if kind == "reading_order":
            left = {
                _page_name(str(row.get("img_id") or row.get("image_name") or "")): list(row.get("gt") or ())
                for row in reference[kind]
            }
            right = {
                _page_name(str(row.get("img_id") or row.get("image_name") or "")): list(row.get("gt") or ())
                for row in candidate[kind]
            }
            differing_pages = []
            for page in sorted(set(left) | set(right)):
                if left.get(page) == right.get(page):
                    continue
                differing_pages.append(page)
                difference_records.append(
                    {
                        "kind": kind,
                        "image_name": page,
                        "difference_type": "reading_order_gt_sequence",
                        "reference_gt_sequence": left.get(page),
                        "candidate_gt_sequence": right.get(page),
                    }
                )
            by_kind[kind] = {
                "exact": not differing_pages,
                "reference_pages": len(left),
                "candidate_pages": len(right),
                "reference_only_pages": sorted(set(left) - set(right)),
                "candidate_only_pages": sorted(set(right) - set(left)),
                "pages_with_different_gt_sequences": differing_pages,
                "difference_page_count": len(differing_pages),
            }
            continue

        def atoms(rows: list[dict[str, Any]]) -> dict[str, collections.Counter[str]]:
            result: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
            for row in rows:
                page = _page_name(str(row.get("img_id") or row.get("image_name") or ""))
                indices = row.get("gt_idx")
                indices = indices if isinstance(indices, list) else [indices]
                for value in indices:
                    if value not in (None, ""):
                        result[page][json.dumps(value, ensure_ascii=False, sort_keys=True)] += 1
            return dict(result)

        left = atoms(reference[kind])
        right = atoms(candidate[kind])
        differing_pages = []
        reference_only_atoms = 0
        candidate_only_atoms = 0
        for page in sorted(set(left) | set(right)):
            left_counter = left.get(page, collections.Counter())
            right_counter = right.get(page, collections.Counter())
            if left_counter == right_counter:
                continue
            differing_pages.append(page)
            left_only = left_counter - right_counter
            right_only = right_counter - left_counter
            reference_only_atoms += sum(left_only.values())
            candidate_only_atoms += sum(right_only.values())
            difference_records.append(
                {
                    "kind": kind,
                    "image_name": page,
                    "difference_type": "concrete_gt_index_atom_membership",
                    "reference_only_atoms": [
                        {"gt_idx": json.loads(value), "count": count}
                        for value, count in sorted(left_only.items())
                    ],
                    "candidate_only_atoms": [
                        {"gt_idx": json.loads(value), "count": count}
                        for value, count in sorted(right_only.items())
                    ],
                }
            )
        by_kind[kind] = {
            "exact": not differing_pages,
            "reference_pages_with_concrete_atoms": len(left),
            "candidate_pages_with_concrete_atoms": len(right),
            "reference_concrete_atom_count": sum(sum(counter.values()) for counter in left.values()),
            "candidate_concrete_atom_count": sum(sum(counter.values()) for counter in right.values()),
            "difference_page_count": len(differing_pages),
            "difference_pages": differing_pages,
            "reference_only_atom_count": reference_only_atoms,
            "candidate_only_atom_count": candidate_only_atoms,
        }
    return {
        "exact_all_kinds": not difference_records,
        "difference_record_count": len(difference_records),
        "by_kind": by_kind,
        "differences": difference_records,
        "interpretation": (
            "This audits prediction-dependent evaluator result membership, not source-dataset identity. "
            "Differences are preserved as evidence and excluded from forced element pairing; exact "
            "page-level metric recomposition remains authoritative."
        ),
    }


def _compact_for_review(record: dict[str, Any]) -> dict[str, Any]:
    compact = dict(record)
    for field in (
        "gt",
        "reference_gt",
        "candidate_gt",
        "reference_pred",
        "candidate_pred",
        "normalized_gt",
        "normalized_reference_gt",
        "normalized_candidate_gt",
        "normalized_reference_pred",
        "normalized_candidate_pred",
        "reference_text",
        "candidate_text",
    ):
        if field in compact:
            value = str(compact.pop(field) or "")
            compact[f"{field}_excerpt"] = _excerpt(value)
            compact[f"{field}_length"] = len(value)
            compact[f"{field}_truncated"] = len(value) > 600
    for field in (
        "reference_token_ids_including_eos",
        "candidate_token_ids_including_eos",
    ):
        if field in compact:
            values = list(compact.pop(field) or ())
            compact[f"{field}_excerpt"] = values[:40] + (["…"] if len(values) > 50 else []) + values[-10:]
            compact[f"{field}_length"] = len(values)
    return compact


def _top_records(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    selected = sorted(
        (
            record
            for record in records
            if record["candidate_loss_contribution"] > 0
            and record.get("pair_status", "paired") == "paired"
        ),
        key=lambda record: (
            -record["candidate_loss_contribution"],
            record["image_name"],
            str(record.get("gt_idx")),
        ),
    )[:limit]
    return [_compact_for_review(record) for record in selected]


def _table_relevance_bridge(
    generation_records: list[dict[str, Any]],
    candidate_table_samples: list[dict[str, Any]],
    metric_page_records: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    transitions_by_page: dict[str, list[str]] = collections.defaultdict(list)
    for record in generation_records:
        if record["pair_status"] != "paired":
            continue
        if record["reference_label"] == "table" and record["candidate_label"] == "table":
            proof = "verified" if record["input_status"] == "exact" else "unverified_stable_key"
            transitions_by_page[record["source_image_name"]].append(
                f"{proof}:{record['reference_format']}->{record['candidate_format']}"
            )
    candidate_table_by_page: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for sample in candidate_table_samples:
        page = _page_name(str(sample.get("img_id") or sample.get("image_name") or ""))
        candidate_table_by_page[page].append(sample)
    loss_maps = {
        metric: {record["image_name"]: record["candidate_loss_contribution"] for record in records}
        for metric, records in metric_page_records.items()
        if metric in {"table.Edit_dist", "table.TEDS.page", "table.TEDS_structure_only.page"}
    }
    grouped: dict[str, dict[str, Any]] = {}
    pages = []
    for page in sorted(set(transitions_by_page) | set(candidate_table_by_page)):
        signature = ",".join(sorted(transitions_by_page.get(page) or ["no_paired_raw_table_trace"]))
        samples = candidate_table_by_page.get(page, [])
        empty_predictions = sum(not bool(sample.get("pred")) for sample in samples)
        row = {
            "image_name": page,
            "transition_signature": signature,
            "candidate_evaluator_table_rows": len(samples),
            "candidate_empty_table_predictions": empty_predictions,
            "candidate_nonempty_table_predictions": len(samples) - empty_predictions,
            "table_edit_loss_contribution": loss_maps.get("table.Edit_dist", {}).get(page, 0.0),
            "table_teds_loss_contribution": loss_maps.get("table.TEDS.page", {}).get(page, 0.0),
            "table_teds_structure_loss_contribution": loss_maps.get("table.TEDS_structure_only.page", {}).get(page, 0.0),
        }
        pages.append(row)
        bucket = grouped.setdefault(
            signature,
            {
                "pages": 0,
                "candidate_evaluator_table_rows": 0,
                "candidate_empty_table_predictions": 0,
                "table_edit_loss_contribution": 0.0,
                "table_teds_loss_contribution": 0.0,
                "table_teds_structure_loss_contribution": 0.0,
            },
        )
        bucket["pages"] += 1
        bucket["candidate_evaluator_table_rows"] += len(samples)
        bucket["candidate_empty_table_predictions"] += empty_predictions
        for field in (
            "table_edit_loss_contribution",
            "table_teds_loss_contribution",
            "table_teds_structure_loss_contribution",
        ):
            bucket[field] += row[field]
    return {"by_transition_signature": grouped, "pages": pages}


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def _render_html(report: dict[str, Any], top_by_metric: dict[str, list[dict[str, Any]]], generation_records: list[dict[str, Any]], output: Path) -> None:
    def e(value: Any) -> str:
        return html.escape(str(value))

    sections = []
    for metric, records in top_by_metric.items():
        rows = []
        for record in records:
            rows.append(
                "<tr>"
                f"<td>{e(record['image_name'])}</td>"
                f"<td>{e(record.get('gt_idx'))}</td>"
                f"<td>{record['candidate_loss_contribution']:.6g}</td>"
                f"<td>{e(record.get('raw_difference_class'))}<br><small>normalized: {e(record.get('normalized_difference_class'))}</small></td>"
                f"<td><pre>{e(record.get('gt_excerpt', record.get('candidate_gt_excerpt', '')))}</pre></td>"
                f"<td><pre>{e(record.get('reference_pred_excerpt', ''))}</pre></td>"
                f"<td><pre>{e(record.get('candidate_pred_excerpt', ''))}</pre></td>"
                "</tr>"
            )
        sections.append(
            f"<h2>{e(metric)}</h2><table><thead><tr><th>page</th><th>GT idx</th><th>loss contribution</th><th>difference</th><th>GT</th><th>reference</th><th>candidate</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
        )
    flagged = [record for record in generation_records if record["triage_flags"]]
    flagged.sort(key=lambda record: (-record["candidate_tokens"] + record["reference_tokens"], record["source_image_name"]))
    flag_rows = "".join(
        "<tr>"
        f"<td>{e(record['source_image_name'])}</td><td>{e(record['block_index'])}</td>"
        f"<td>{e(record['candidate_label'])}</td><td>{e(record['input_status'])}</td>"
        f"<td>{e(', '.join(record['triage_flags']))}</td>"
        f"<td>{record['reference_tokens']} / {record['candidate_tokens']}</td>"
        f"<td><pre>{e(_excerpt(record['reference_text']))}</pre></td><td><pre>{e(_excerpt(record['candidate_text']))}</pre></td></tr>"
        for record in flagged[:100]
    )
    document = f"""<!doctype html>
<meta charset="utf-8"><title>Experiment 09 generation difference atlas</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;background:#f7f7f8;color:#171717}}table{{border-collapse:collapse;width:100%;background:white;margin-bottom:32px}}th,td{{border:1px solid #ccc;padding:6px;vertical-align:top}}th{{position:sticky;top:0;background:#eee}}pre{{white-space:pre-wrap;word-break:break-word;max-width:420px;max-height:260px;overflow:auto;margin:0}}code{{background:#eee;padding:2px 4px}}.summary{{white-space:pre-wrap;background:white;border:1px solid #ccc;padding:12px;max-height:520px;overflow:auto}}
</style>
<h1>Experiment 09 generation difference atlas</h1>
<p>This page is diagnostic. Raw recognizer generations are unchanged; evaluator panes show raw and evaluator-normalized evidence separately.</p>
<div class="summary">{e(json.dumps(report, indent=2, ensure_ascii=False))}</div>
<h2>Heuristic generation triage candidates</h2>
<table><thead><tr><th>page</th><th>block</th><th>label</th><th>input</th><th>flags</th><th>tokens ref/cand</th><th>reference</th><th>candidate</th></tr></thead><tbody>{flag_rows}</tbody></table>
{''.join(sections)}
"""
    output.write_text(document, encoding="utf-8")


def main() -> None:
    args = _parse_args()
    reference_side = _load_side(
        bundle=args.reference_bundle,
        output=args.reference_output,
        eval_dir=args.reference_eval_dir,
        table_scores=args.reference_table_scores,
        teds_summary=args.reference_teds_summary,
        label=args.reference_label,
    )
    candidate_side = _load_side(
        bundle=args.candidate_bundle,
        output=args.candidate_output,
        eval_dir=args.candidate_eval_dir,
        table_scores=args.candidate_table_scores,
        teds_summary=args.candidate_teds_summary,
        label=args.candidate_label,
    )
    reference_trace = reference_side["trace"]
    candidate_trace = candidate_side["trace"]
    reference_eval = reference_side["eval"]
    candidate_eval = candidate_side["eval"]
    reference_official = reference_side["official"]
    candidate_official = candidate_side["official"]
    for kind in EVAL_KINDS:
        if not isinstance(reference_eval[kind], list) or not isinstance(candidate_eval[kind], list):
            raise TypeError(f"{kind} evaluator results must be JSON lists")
    print("[atlas] auditing evaluator GT-result universes", flush=True)
    evaluator_gt_universe_audit = _audit_evaluator_gt_universes(
        reference_eval, candidate_eval
    )
    if not evaluator_gt_universe_audit["exact_all_kinds"]:
        print(
            "[atlas] preserving evaluator GT-result membership differences: "
            f"records={evaluator_gt_universe_audit['difference_record_count']}",
            flush=True,
        )
    if reference_side["manifest"] and candidate_side["manifest"]:
        if reference_side["manifest"].get("evaluator_commit") != candidate_side["manifest"].get("evaluator_commit"):
            raise ValueError("reference and candidate bundles use different evaluator commits")

    print("[atlas] pairing raw recognizer generations", flush=True)
    generation_summary, generation_records, logit_candidates = _analyze_generations(reference_trace, candidate_trace)

    if args.expected_shared_requests is not None and generation_summary["shared_requests"] != args.expected_shared_requests:
        raise ValueError(f"expected {args.expected_shared_requests} shared requests, got {generation_summary['shared_requests']}")
    if args.expected_reference_table_requests is not None and generation_summary["reference_table_requests"] != args.expected_reference_table_requests:
        raise ValueError(f"expected {args.expected_reference_table_requests} reference table requests, got {generation_summary['reference_table_requests']}")
    if args.expected_candidate_table_requests is not None and generation_summary["candidate_table_requests"] != args.expected_candidate_table_requests:
        raise ValueError(f"expected {args.expected_candidate_table_requests} candidate table requests, got {generation_summary['candidate_table_requests']}")

    def run_pages(side: dict[str, Any]) -> int:
        images = side["run_summary"].get("images")
        return len(images) if isinstance(images, list) else int(side["run_summary"].get("count", 0))

    reference_pages = run_pages(reference_side)
    candidate_pages = run_pages(candidate_side)
    reference_images = reference_side["run_summary"].get("images")
    candidate_images = candidate_side["run_summary"].get("images")
    if isinstance(reference_images, list) and isinstance(candidate_images, list):
        if reference_images != candidate_images:
            raise ValueError("reference and candidate run_summary image lists/order differ")
        generation_summary["ordered_page_set_sha256"] = _sha256_text(
            json.dumps(reference_images, ensure_ascii=False, separators=(",", ":"))
        )
    if args.expected_pages is not None:
        if reference_pages != args.expected_pages or candidate_pages != args.expected_pages:
            raise ValueError(
                f"expected {args.expected_pages} pages on both sides, got "
                f"reference={reference_pages} candidate={candidate_pages}"
            )
    generation_summary["reference_run_pages"] = reference_pages
    generation_summary["candidate_run_pages"] = candidate_pages
    generation_summary["reference_pages_with_requests"] = len({row["source_image_name"] for row in reference_trace})
    generation_summary["candidate_pages_with_requests"] = len({row["source_image_name"] for row in candidate_trace})

    metric_summaries = {}
    metric_records: dict[str, list[dict[str, Any]]] = {}
    metric_page_records: dict[str, list[dict[str, Any]]] = {}
    for kind in ("text_block", "display_formula", "table", "reading_order"):
        print(f"[atlas] attributing {kind} Edit_dist", flush=True)
        summary, records, page_records = _analyze_edit_metric(kind, reference_eval[kind], candidate_eval[kind])
        _reconcile_edit_summary(
            summary,
            reference_official,
            candidate_official,
            kind,
        )
        if kind == "reading_order":
            summary["order_transition"] = _augment_reading_order(records, reference_eval[kind], candidate_eval[kind])
            summary["interpretation"] = "content-match-coupled ordering score; not a pure layout metric, and unmatched extra predictions are not directly penalized"
        metric_summaries[f"{kind}.Edit_dist"] = summary
        metric_records[f"{kind}.Edit_dist"] = records
        metric_page_records[f"{kind}.Edit_dist"] = page_records

    reference_scores = reference_side["table_scores"]
    candidate_scores = candidate_side["table_scores"]
    reference_teds_summary = reference_side["teds_summary"]
    candidate_teds_summary = candidate_side["teds_summary"]
    _validate_teds_evidence(reference_eval["table"], reference_scores, reference_teds_summary)
    _validate_teds_evidence(candidate_eval["table"], candidate_scores, candidate_teds_summary)
    teds_authority_audit = {
        "reference": _validate_frozen_teds_authority(
            reference_official, reference_teds_summary
        ),
        "candidate": _validate_frozen_teds_authority(
            candidate_official, candidate_teds_summary
        ),
    }
    for metric in ("TEDS", "TEDS_structure_only"):
        for page_weighted in (False, True):
            name = f"table.{metric}.{'page' if page_weighted else 'sample'}"
            print(f"[atlas] attributing {name}", flush=True)
            summary, records, page_records = _analyze_teds(reference_eval["table"], candidate_eval["table"], reference_scores, candidate_scores, metric, page_weighted)
            _reconcile_teds_summary(
                summary,
                reference_official,
                candidate_official,
                reference_teds_summary,
                candidate_teds_summary,
                metric,
                page_weighted,
            )
            metric_summaries[name] = summary
            metric_records[name] = records
            metric_page_records[name] = page_records

    top_by_metric = {name: _top_records(records, args.review_limit) for name, records in metric_records.items()}
    top_pages_by_metric = {name: _top_records(records, args.review_limit) for name, records in metric_page_records.items()}
    table_relevance = _table_relevance_bridge(
        generation_records,
        candidate_eval["table"],
        metric_page_records,
    )
    report = {
        "schema_version": 1,
        "inputs": {
            "reference_label": args.reference_label,
            "candidate_label": args.candidate_label,
            "reference_source": reference_side["source"],
            "candidate_source": candidate_side["source"],
            "reference_manifest": reference_side["manifest"],
            "candidate_manifest": candidate_side["manifest"],
            "run_configuration": {
                "equal": reference_side["run_summary"].get("configuration")
                == candidate_side["run_summary"].get("configuration"),
                "reference": reference_side["run_summary"].get("configuration"),
                "candidate": candidate_side["run_summary"].get("configuration"),
            },
        },
        "generation": generation_summary,
        "metrics": metric_summaries,
        "evaluator_gt_universe_audit": evaluator_gt_universe_audit,
        "teds_authority_audit": teds_authority_audit,
        "reading_order_evaluator_pred_idx_zero_audit": {
            "reference": _audit_pred_index_zero(reference_eval),
            "candidate": _audit_pred_index_zero(candidate_eval),
        },
        "table_format_to_omnidocbench": table_relevance["by_transition_signature"],
        "table_logit_candidates": {
            "count": len(logit_candidates),
            "evidence_boundary": "target crop/prepared tensors are exact; full vision/text pack companion membership, order, and offsets must be reconstructed before a replay is considered faithful",
            "first_25": [
                _compact_for_review(record) for record in logit_candidates[:25]
            ],
        },
        "top_harmful_pages": top_pages_by_metric,
        "top_harmful_samples": top_by_metric,
    }
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    print("[atlas] writing reviewed artifacts", flush=True)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_jsonl(output_dir / "generation_records.jsonl", generation_records)
    _write_jsonl(output_dir / "metric_records.jsonl", (record for records in metric_records.values() for record in records))
    _write_jsonl(output_dir / "page_metric_records.jsonl", (record for records in metric_page_records.values() for record in records))
    _write_jsonl(
        output_dir / "evaluator_gt_universe_differences.jsonl",
        evaluator_gt_universe_audit["differences"],
    )
    _write_jsonl(output_dir / "table_relevance_pages.jsonl", table_relevance["pages"])
    (output_dir / "table_logit_candidates.json").write_text(json.dumps(logit_candidates, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _render_html(report, top_by_metric, generation_records, output_dir / "review.html")
    print(json.dumps({"generation": generation_summary, "metrics": metric_summaries, "table_logit_candidates": len(logit_candidates)}, indent=2, ensure_ascii=False))
    print(f"[generation-difference-atlas] saved to {output_dir}")


if __name__ == "__main__":
    main()
