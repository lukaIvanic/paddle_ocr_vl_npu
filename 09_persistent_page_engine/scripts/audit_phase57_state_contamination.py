#!/usr/bin/env python3
"""Localize Phase-57 runaways to prefill, decode, EOS, and cache reuse.

This reads existing JSONL traces only.  It performs no NPU work, inference, or
evaluation.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import zipfile
from pathlib import Path
from typing import Any


REFERENCE = Path(
    "tmp/09_persistent_page_engine/910b_phase57_authority_898ced7/"
    "phase57_910b_authority.gdatlas.zip"
)
INPUT_FIELDS = (
    "prompt",
    "input_tokens",
    "projected_image_tokens",
    "crop_size",
    "min_pixels",
    "max_pixels",
)
RUNAWAY_STOPS = {"kv_cache_full", "repetition"}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-bundle", type=Path, default=REFERENCE)
    parser.add_argument("--candidate-trace", type=Path)
    parser.add_argument("--fresh-report", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict-extra-tokens", type=int, default=128)
    return parser.parse_args()


def _key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["source_image_name"]), int(row["block_index"])


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _zip_jsonl(path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as archive, archive.open(
        "recognition_trace.jsonl"
    ) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _find_candidate() -> Path:
    traces = []
    for root in Path("tmp/09_persistent_page_engine").glob(
        "310p_phase57_cap4096_b64_pse_*/full"
    ):
        trace = root / "output/recognition_trace.jsonl"
        summary = root / "output/run_summary.json"
        if not trace.is_file() or not summary.is_file():
            continue
        data = json.loads(summary.read_text(encoding="utf-8"))
        if data.get("result_count") == data.get("prediction_count") == 1651:
            traces.append(trace)
    if not traces:
        raise FileNotFoundError("no completed Phase-57 candidate trace")
    return max(traces, key=lambda path: path.stat().st_mtime)


def _first_divergence(left: list[int], right: list[int]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if int(a) != int(b):
            return index
    return min(len(left), len(right)) if len(left) != len(right) else None


def _divergence_category(left: list[int], right: list[int]) -> str:
    first = _first_divergence(left, right)
    if first is None:
        return "exact"
    if first == 0:
        return "prefill_token_zero"
    if first == len(left) - 1:
        return "expected_eos_replaced"
    if first == len(left):
        return "reference_is_exact_prefix_then_candidate_extra"
    return "interior_decode"


def _private_predecessors(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, int], dict[str, Any] | None]:
    previous_by_slot: dict[int, dict[str, Any]] = {}
    maximum_prompt_by_slot: dict[int, int] = {}
    maximum_prompt_row_by_slot: dict[int, dict[str, Any]] = {}
    result: dict[tuple[str, int], dict[str, Any] | None] = {}
    for row in sorted(rows, key=lambda item: int(item["global_request_index"])):
        route = row.get("text_prefill") or {}
        slot = route.get("private_cache_slot_index")
        if slot is None:
            result[_key(row)] = None
            continue
        slot = int(slot)
        result[_key(row)] = {
            "previous_row": previous_by_slot.get(slot),
            "previous_maximum_prompt_length": maximum_prompt_by_slot.get(slot),
            "previous_maximum_prompt_row": maximum_prompt_row_by_slot.get(slot),
        }
        previous_by_slot[slot] = row
        prompt_length = int(row["input_tokens"])
        if prompt_length > maximum_prompt_by_slot.get(slot, 0):
            maximum_prompt_by_slot[slot] = prompt_length
            maximum_prompt_row_by_slot[slot] = row
    return result


def _percentiles(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "p90": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(values),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
        "max": ordered[-1],
    }


def main() -> None:
    args = _args()
    reference_rows = _zip_jsonl(args.reference_bundle.resolve())
    candidate_path = (args.candidate_trace or _find_candidate()).resolve()
    candidate_rows = _jsonl(candidate_path)
    reference = {_key(row): row for row in reference_rows}
    candidate = {_key(row): row for row in candidate_rows}
    predecessors = _private_predecessors(candidate_rows)

    records = []
    for stable in sorted(reference.keys() & candidate.keys()):
        left, right = reference[stable], candidate[stable]
        left_tokens = [int(value) for value in left.get("token_ids") or ()]
        right_tokens = [int(value) for value in right.get("token_ids") or ()]
        if str(left.get("stop_reason")) != "eos":
            continue
        extra = len(right_tokens) - len(left_tokens)
        broad = str(right.get("stop_reason")) in RUNAWAY_STOPS and extra > 0
        strict = broad and extra >= int(args.strict_extra_tokens)
        route = right.get("text_prefill") or {}
        history = predecessors.get(stable)
        predecessor = None if history is None else history["previous_row"]
        maximum_predecessor = (
            None if history is None else history["previous_maximum_prompt_row"]
        )
        prior_prompt = (
            int(predecessor["input_tokens"]) if predecessor is not None else None
        )
        prior_maximum_prompt = (
            None
            if history is None or history["previous_maximum_prompt_length"] is None
            else int(history["previous_maximum_prompt_length"])
        )
        current_prompt = int(right["input_tokens"])
        stale_tail = (
            max(0, prior_maximum_prompt - current_prompt)
            if prior_maximum_prompt is not None
            else 0
        )
        left_text_route = left.get("text_prefill") or {}
        right_text_route = right.get("text_prefill") or {}
        left_vision_route = left.get("vision") or {}
        right_vision_route = right.get("vision") or {}
        text_pack_fields = (
            "bucket",
            "pack_members",
            "segment_lengths",
            "text_pack_index",
            "physical_text_tokens",
        )
        vision_pack_fields = (
            "bucket",
            "pack_crops",
            "pack_row_sizes",
            "pack_sequence_length",
            "physical_vision_tokens",
        )
        records.append(
            {
                "source_image_name": stable[0],
                "block_index": stable[1],
                "global_request_index": int(right["global_request_index"]),
                "label": right.get("label"),
                "reference_tokens": len(left_tokens),
                "candidate_tokens": len(right_tokens),
                "extra_tokens": extra,
                "candidate_stop": right.get("stop_reason"),
                "broad_runaway": broad,
                "strict_runaway": strict,
                "first_divergence": _first_divergence(left_tokens, right_tokens),
                "divergence_category": _divergence_category(left_tokens, right_tokens),
                "common_prefix_fraction_of_reference": (
                    (_first_divergence(left_tokens, right_tokens) or 0)
                    / len(left_tokens)
                    if left_tokens
                    else 0.0
                ),
                "metadata_exact": all(
                    left.get(field) == right.get(field) for field in INPUT_FIELDS
                ),
                "private_cache_slot": route.get("private_cache_slot_index"),
                "private_cache_generation": route.get("private_cache_generation"),
                "previous_private_request": (
                    None if predecessor is None else predecessor.get("request_id")
                ),
                "previous_prompt_length": prior_prompt,
                "previous_maximum_prompt_length": prior_maximum_prompt,
                "previous_maximum_prompt_request": (
                    None
                    if maximum_predecessor is None
                    else maximum_predecessor.get("request_id")
                ),
                "previous_maximum_prompt_source_image_name": (
                    None
                    if maximum_predecessor is None
                    else maximum_predecessor.get("source_image_name")
                ),
                "previous_maximum_prompt_block_index": (
                    None
                    if maximum_predecessor is None
                    else int(maximum_predecessor["block_index"])
                ),
                "previous_maximum_prompt_global_request_index": (
                    None
                    if maximum_predecessor is None
                    else int(maximum_predecessor["global_request_index"])
                ),
                "current_prompt_length": current_prompt,
                "stale_private_tail_tokens": stale_tail,
                "text_pack_contract_exact": all(
                    left_text_route.get(field) == right_text_route.get(field)
                    for field in text_pack_fields
                ),
                "vision_pack_contract_exact": all(
                    left_vision_route.get(field) == right_vision_route.get(field)
                    for field in vision_pack_fields
                ),
                "decode_slot_index": right.get("decode_slot_index"),
                "decode_slot_epoch": right.get("decode_slot_epoch"),
                "text_pack_members": route.get("pack_members"),
                "text_segment_lengths": route.get("segment_lengths"),
                "vision_pack_crops": (right.get("vision") or {}).get("pack_crops"),
            }
        )

    def summarize(selected: list[dict[str, Any]]) -> dict[str, Any]:
        categories = collections.Counter(
            str(row["divergence_category"]) for row in selected
        )
        generations = [
            int(row["private_cache_generation"])
            for row in selected
            if row["private_cache_generation"] is not None
        ]
        stale = [int(row["stale_private_tail_tokens"]) for row in selected]
        return {
            "count": len(selected),
            "metadata_exact": sum(bool(row["metadata_exact"]) for row in selected),
            "stop_reasons": dict(
                collections.Counter(str(row["candidate_stop"]) for row in selected)
            ),
            "divergence_categories": dict(categories),
            "first_use_private_cache": sum(value == 1 for value in generations),
            "reused_private_cache": sum(value > 1 for value in generations),
            "positive_stale_private_tail": sum(value > 0 for value in stale),
            "text_pack_contract_exact": sum(
                bool(row["text_pack_contract_exact"]) for row in selected
            ),
            "vision_pack_contract_exact": sum(
                bool(row["vision_pack_contract_exact"]) for row in selected
            ),
            "stale_private_tail_tokens": _percentiles(stale),
            "first_divergence": _percentiles(
                [
                    int(row["first_divergence"])
                    for row in selected
                    if row["first_divergence"] is not None
                ]
            ),
        }

    broad = [row for row in records if row["broad_runaway"]]
    strict = [row for row in records if row["strict_runaway"]]
    controls = [row for row in records if not row["broad_runaway"]]
    control_by_stale = collections.Counter()
    runaway_by_stale = collections.Counter()
    for row in records:
        bucket = "positive_stale_tail" if row["stale_private_tail_tokens"] > 0 else "no_positive_stale_tail"
        control_by_stale[bucket] += 1
        if row["broad_runaway"]:
            runaway_by_stale[bucket] += 1

    report: dict[str, Any] = {
        "classification": "PHASE57_STATE_CONTAMINATION_AUDIT",
        "candidate_trace": str(candidate_path),
        "strict_extra_tokens": int(args.strict_extra_tokens),
        "broad_runaways": summarize(broad),
        "strict_runaways": summarize(strict),
        "non_runaway_reference_eos_controls": summarize(controls),
        "runaway_rate_by_private_tail": {
            bucket: {
                "reference_eos_rows": control_by_stale[bucket],
                "broad_runaways": runaway_by_stale[bucket],
                "rate": (
                    runaway_by_stale[bucket] / control_by_stale[bucket]
                    if control_by_stale[bucket]
                    else None
                ),
            }
            for bucket in ("positive_stale_tail", "no_positive_stale_tail")
        },
        "strict_cases": sorted(
            strict,
            key=lambda row: (row["global_request_index"], row["block_index"]),
        ),
    }

    fresh_report = args.fresh_report
    if fresh_report is None:
        reports = list(candidate_path.parents[2].glob("fresh_runaway_replay_*/report.json"))
        fresh_report = max(reports, key=lambda path: path.stat().st_mtime) if reports else None
    if fresh_report is not None and fresh_report.is_file():
        fresh = json.loads(fresh_report.read_text(encoding="utf-8"))
        strict_map = {(_key(row)): row for row in strict}
        report["fresh_cases"] = [
            {
                **case,
                "production_state": strict_map.get(
                    (str(case["source_image_name"]), int(case["block_index"]))
                ),
            }
            for case in fresh.get("cases", [])
        ]

    output = (
        args.output.resolve()
        if args.output is not None
        else candidate_path.parents[1] / "state_contamination_audit.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    broad_summary = report["broad_runaways"]
    strict_summary = report["strict_runaways"]
    print(
        "PHASE57_STATE_CONTAMINATION_AUDIT PASS "
        f"broad={broad_summary['count']} strict={strict_summary['count']} "
        f"strict_divergence={json.dumps(strict_summary['divergence_categories'], separators=(',', ':'))} "
        f"strict_reused={strict_summary['reused_private_cache']} "
        f"strict_positive_stale={strict_summary['positive_stale_private_tail']} "
        f"report={output}"
    )


if __name__ == "__main__":
    main()
