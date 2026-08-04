#!/usr/bin/env python3
"""Audit whether Phase-57 runaways cluster at PSE-sentinel EOS boundaries.

This is a CPU-only trace join.  It performs no inference or evaluation.
"""

from __future__ import annotations

import argparse
import collections
import json
import zipfile
from pathlib import Path
from typing import Any


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
    parser.add_argument(
        "--reference-bundle",
        type=Path,
        default=Path(
            "tmp/09_persistent_page_engine/910b_phase57_authority_898ced7/"
            "phase57_910b_authority.gdatlas.zip"
        ),
    )
    parser.add_argument("--candidate-trace", type=Path)
    parser.add_argument("--fresh-report", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--runaway-min-extra-tokens", type=int, default=128)
    return parser.parse_args()


def _key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["source_image_name"]), int(row["block_index"])


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _zip_jsonl(path: Path, member: str) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as archive, archive.open(member) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _find_candidate_trace() -> Path:
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
        raise FileNotFoundError("no completed Phase-57 310P candidate trace found")
    return max(traces, key=lambda path: path.stat().st_mtime)


def _find_fresh_report(candidate_trace: Path) -> Path | None:
    run_root = candidate_trace.parents[2]
    reports = list(run_root.glob("fresh_runaway_replay_*/report.json"))
    return max(reports, key=lambda path: path.stat().st_mtime) if reports else None


def _eos_coordinates(row: dict[str, Any]) -> dict[str, int | None]:
    """Map trace token counts onto the scheduler's zero-based KV positions.

    Token zero is produced by prefill.  For N >= 2, EOS is sampled by the
    decode graph at P + N - 2 and would itself occupy P + N - 1 if fed into
    the following speculative decode iteration.
    """

    prompt = int(row["input_tokens"])
    generated = len(row.get("token_ids") or ())
    return {
        "prompt_length": prompt,
        "generated_tokens_including_eos": generated,
        "eos_sampling_cache_position": (
            prompt + generated - 2 if generated >= 2 else None
        ),
        "eos_token_position": prompt + generated - 1,
    }


def _nearest_boundary(position: int | None, boundaries: tuple[int, ...]) -> int | None:
    if position is None:
        return None
    return min(boundaries, key=lambda boundary: abs(position - boundary))


def _boundary_summary(
    rows: list[dict[str, Any]], boundaries: tuple[int, ...]
) -> dict[str, Any]:
    sampling_counts: collections.Counter[int] = collections.Counter()
    token_counts: collections.Counter[int] = collections.Counter()
    exact_sampling = collections.Counter()
    exact_token = collections.Counter()
    near_sampling = collections.Counter()
    exact_cases = []
    for row in rows:
        coords = _eos_coordinates(row)
        sampling = coords["eos_sampling_cache_position"]
        token_position = coords["eos_token_position"]
        if sampling is not None:
            sampling_counts[int(sampling)] += 1
            nearest = _nearest_boundary(int(sampling), boundaries)
            assert nearest is not None
            delta = int(sampling) - nearest
            if abs(delta) <= 8:
                near_sampling[delta] += 1
        token_counts[int(token_position)] += 1
        if sampling in boundaries:
            exact_sampling[int(sampling)] += 1
        if token_position in boundaries:
            exact_token[int(token_position)] += 1
        if sampling in boundaries or token_position in boundaries:
            exact_cases.append(
                {
                    "source_image_name": row["source_image_name"],
                    "block_index": int(row["block_index"]),
                    "label": row.get("label"),
                    **coords,
                }
            )
    return {
        "row_count": len(rows),
        "eos_sampled_by_pse_boundary_graph": sum(exact_sampling.values()),
        "eos_sampled_by_pse_boundary_graph_by_position": dict(exact_sampling),
        "eos_already_sampled_before_speculative_pse_boundary_graph": sum(
            exact_token.values()
        ),
        "eos_already_sampled_before_speculative_pse_boundary_graph_by_position": dict(
            exact_token
        ),
        "exact_boundary_cases": exact_cases,
        "eos_sampling_delta_from_nearest_boundary_within_8": dict(
            sorted(near_sampling.items())
        ),
        "top_eos_sampling_positions": sampling_counts.most_common(20),
        "top_eos_token_positions": token_counts.most_common(20),
    }


def main() -> None:
    args = _args()
    reference_bundle = args.reference_bundle.resolve()
    reference_rows = _zip_jsonl(reference_bundle, "recognition_trace.jsonl")
    reference = {_key(row): row for row in reference_rows}
    cache_length = 4096
    boundaries = tuple(
        position
        for position in range(1279, cache_length, 1280)
        if position + 1 < cache_length
    )
    reference_eos = [
        row for row in reference_rows if str(row.get("stop_reason")) == "eos"
    ]

    report: dict[str, Any] = {
        "classification": "PHASE57_EOS_BOUNDARY_AUDIT",
        "position_semantics": {
            "pse_boundary_cache_positions": boundaries,
            "eos_sampling_cache_position_formula": (
                "input_tokens + len(token_ids_including_eos) - 2"
            ),
            "eos_token_position_formula": (
                "input_tokens + len(token_ids_including_eos) - 1"
            ),
        },
        "reference_910b_all_eos": _boundary_summary(reference_eos, boundaries),
    }

    candidate_trace = args.candidate_trace
    if candidate_trace is None:
        try:
            candidate_trace = _find_candidate_trace()
        except FileNotFoundError:
            candidate_trace = None
    if candidate_trace is not None:
        candidate_trace = candidate_trace.resolve()
        candidate = {_key(row): row for row in _jsonl(candidate_trace)}
        runaways = []
        for stable in sorted(reference.keys() & candidate.keys()):
            left, right = reference[stable], candidate[stable]
            left_tokens = left.get("token_ids") or []
            right_tokens = right.get("token_ids") or []
            if str(left.get("stop_reason")) != "eos":
                continue
            if str(right.get("stop_reason")) not in RUNAWAY_STOPS:
                continue
            if len(right_tokens) - len(left_tokens) < args.runaway_min_extra_tokens:
                continue
            coordinates = _eos_coordinates(left)
            runaways.append(
                {
                    **left,
                    "_stable_key": stable,
                    "_candidate_stop": right.get("stop_reason"),
                    "_candidate_tokens": len(right_tokens),
                    "_metadata_exact": all(
                        left.get(field) == right.get(field) for field in INPUT_FIELDS
                    ),
                    "_coordinates": coordinates,
                }
            )
        exact = [row for row in runaways if row["_metadata_exact"]]
        report["candidate_310p_runaways"] = {
            "candidate_trace": str(candidate_trace),
            "definition": (
                f"reference EOS, candidate stop in {sorted(RUNAWAY_STOPS)}, "
                f"candidate has at least {args.runaway_min_extra_tokens} extra tokens"
            ),
            "all": _boundary_summary(runaways, boundaries),
            "metadata_exact": _boundary_summary(exact, boundaries),
            "metadata_exact_count": len(exact),
            "cases_at_or_near_1279": [
                {
                    "source_image_name": row["_stable_key"][0],
                    "block_index": row["_stable_key"][1],
                    "candidate_stop": row["_candidate_stop"],
                    "candidate_tokens": row["_candidate_tokens"],
                    "metadata_exact": row["_metadata_exact"],
                    **row["_coordinates"],
                }
                for row in runaways
                if row["_coordinates"]["eos_sampling_cache_position"] is not None
                and abs(
                    int(row["_coordinates"]["eos_sampling_cache_position"]) - 1279
                ) <= 8
            ],
        }

        fresh_report = args.fresh_report or _find_fresh_report(candidate_trace)
        if fresh_report is not None and fresh_report.is_file():
            fresh = json.loads(fresh_report.read_text(encoding="utf-8"))
            selected = []
            for case in fresh.get("cases", []):
                stable = (
                    str(case["source_image_name"]),
                    int(case["block_index"]),
                )
                left = reference.get(stable)
                if left is None:
                    continue
                selected.append(
                    {
                        "source_image_name": stable[0],
                        "block_index": stable[1],
                        "classification": case.get("classification"),
                        "isolated_stop": case.get("isolated_stop"),
                        "isolated_equals_reference_tokens": case.get(
                            "isolated_equals_reference_tokens"
                        ),
                        **_eos_coordinates(left),
                    }
                )
            report["fresh_replay_cases"] = {
                "fresh_report": str(fresh_report.resolve()),
                "cases": selected,
            }

    output = args.output
    if output is None:
        if candidate_trace is not None:
            output = candidate_trace.parents[1] / "eos_boundary_audit.json"
        else:
            output = reference_bundle.with_suffix(".eos_boundary_audit.json")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    authority = report["reference_910b_all_eos"]
    candidate_report = report.get("candidate_310p_runaways")
    if candidate_report is None:
        candidate_sentence = "candidate=unavailable"
    else:
        all_runaways = candidate_report["all"]
        exact_runaways = candidate_report["metadata_exact"]
        candidate_sentence = (
            f"runaways={all_runaways['row_count']} "
            f"metadata_exact={exact_runaways['row_count']} "
            f"eos_sampled_at_pse_boundary={all_runaways['eos_sampled_by_pse_boundary_graph']} "
            "eos_before_speculative_pse_boundary="
            f"{all_runaways['eos_already_sampled_before_speculative_pse_boundary_graph']}"
        )
    print(
        "PHASE57_EOS_BOUNDARY_AUDIT PASS "
        f"authority_eos={authority['row_count']} "
        "authority_eos_sampled_at_pse_boundary="
        f"{authority['eos_sampled_by_pse_boundary_graph']} "
        "authority_eos_before_speculative_pse_boundary="
        f"{authority['eos_already_sampled_before_speculative_pse_boundary_graph']} "
        f"{candidate_sentence} report={output}"
    )


if __name__ == "__main__":
    main()
