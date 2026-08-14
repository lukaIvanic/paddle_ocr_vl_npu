#!/usr/bin/env python3
"""Analyze original cross-host and fresh same-host UniRec CDM results."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-original", type=Path, required=True)
    parser.add_argument("--candidate-original", type=Path, required=True)
    parser.add_argument("--reference-recheck", type=Path, required=True)
    parser.add_argument("--candidate-recheck", type=Path, required=True)
    parser.add_argument("--reference-fingerprint", type=Path)
    parser.add_argument("--candidate-fingerprint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["img_id"]), json.dumps(row["gt_idx"], separators=(",", ":"))


def index(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    result = {}
    for row in load(path.resolve()):
        item_key = key(row)
        if item_key in result:
            raise RuntimeError(f"duplicate formula key in {path}: {item_key}")
        result[item_key] = row
    return result


def score(row: dict[str, Any]) -> float:
    return float(row["metric"]["CDM"])


def inputs(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("gt_cdm", "")), str(row.get("pred_cdm", ""))


def aggregate(rows: dict[tuple[str, str], dict[str, Any]]) -> dict[str, float | int]:
    by_page: dict[str, list[float]] = defaultdict(list)
    for (page, _), row in rows.items():
        by_page[page].append(score(row))
    return {
        "formula_samples": len(rows),
        "formula_pages": len(by_page),
        "sample_cdm": sum(score(row) for row in rows.values()) / len(rows),
        "page_cdm": sum(sum(values) / len(values) for values in by_page.values()) / len(by_page),
    }


def pair_summary(
    left: dict[tuple[str, str], dict[str, Any]],
    right: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    shared = sorted(set(left) & set(right))
    exact_inputs = [item for item in shared if inputs(left[item]) == inputs(right[item])]
    score_changed = [item for item in shared if abs(score(left[item]) - score(right[item])) > 1e-12]
    exact_input_score_changed = [
        item for item in exact_inputs if abs(score(left[item]) - score(right[item])) > 1e-12
    ]
    return {
        "left_count": len(left),
        "right_count": len(right),
        "shared_count": len(shared),
        "left_only_count": len(set(left) - set(right)),
        "right_only_count": len(set(right) - set(left)),
        "exact_input_count": len(exact_inputs),
        "changed_input_count": len(shared) - len(exact_inputs),
        "score_changed_count": len(score_changed),
        "score_changed_with_exact_input_count": len(exact_input_score_changed),
        "mean_score_delta_right_minus_left": sum(
            score(right[item]) - score(left[item]) for item in shared
        ) / len(shared),
    }


def page_scores(rows: dict[tuple[str, str], dict[str, Any]]) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for (page, _), row in rows.items():
        values[page].append(score(row))
    return {page: sum(scores) / len(scores) for page, scores in values.items()}


def fingerprint_differences(left: Any, right: Any, prefix: str = "") -> list[dict[str, Any]]:
    if isinstance(left, dict) and isinstance(right, dict):
        rows = []
        for name in sorted(set(left) | set(right)):
            child = f"{prefix}.{name}" if prefix else name
            rows.extend(fingerprint_differences(left.get(name), right.get(name), child))
        return rows
    if left == right:
        return []
    return [{"path": prefix, "reference": left, "candidate": right}]


def map_differences(left: dict[str, Any], right: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"name": name, "reference": left.get(name), "candidate": right.get(name)}
        for name in sorted(set(left) | set(right))
        if left.get(name) != right.get(name)
    ]


def runtime_comparison(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    reference_sources = {
        name: None if value is None else value.get("sha256")
        for name, value in reference.get("evaluator_files", {}).items()
    }
    candidate_sources = {
        name: None if value is None else value.get("sha256")
        for name, value in candidate.get("evaluator_files", {}).items()
    }
    reference_resources = {
        name: value.get("sha256") for name, value in reference.get("tex_resources", {}).items()
    }
    candidate_resources = {
        name: value.get("sha256") for name, value in candidate.get("tex_resources", {}).items()
    }
    version_names = ("pdflatex", "kpsewhich", "imagemagick", "ghostscript")
    reference_versions = {
        name: reference.get("runtime_versions", {}).get(name, {}).get("stdout")
        for name in version_names
    }
    candidate_versions = {
        name: candidate.get("runtime_versions", {}).get(name, {}).get("stdout")
        for name in version_names
    }
    reference_binaries = {
        name: value.get("sha256") for name, value in reference.get("runtime_tools", {}).items()
    }
    candidate_binaries = {
        name: value.get("sha256") for name, value in candidate.get("runtime_tools", {}).items()
    }
    reference_commit = reference.get("evaluator_git", {}).get("commit", {}).get("stdout", "").strip()
    candidate_commit = candidate.get("evaluator_git", {}).get("commit", {}).get("stdout", "").strip()
    return {
        "platform_machine": {
            "reference": reference.get("platform", {}).get("machine"),
            "candidate": candidate.get("platform", {}).get("machine"),
        },
        "python_version": {
            "reference": reference.get("platform", {}).get("python_version"),
            "candidate": candidate.get("platform", {}).get("python_version"),
        },
        "evaluator_commit": {
            "reference": reference_commit,
            "candidate": candidate_commit,
            "exact": reference_commit == candidate_commit,
        },
        "evaluator_source_hash_differences": map_differences(reference_sources, candidate_sources),
        "python_package_differences": map_differences(
            reference.get("python_packages", {}), candidate.get("python_packages", {})
        ),
        "tex_resource_hash_differences": map_differences(reference_resources, candidate_resources),
        "runtime_version_differences": map_differences(reference_versions, candidate_versions),
        "runtime_binary_hash_differences": map_differences(reference_binaries, candidate_binaries),
        "environment_differences": map_differences(
            reference.get("environment", {}), candidate.get("environment", {})
        ),
    }


def main() -> None:
    parsed = args()
    output = parsed.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    ref_original = index(parsed.reference_original)
    cand_original = index(parsed.candidate_original)
    ref_recheck = index(parsed.reference_recheck)
    cand_recheck = index(parsed.candidate_recheck)

    scores = {
        "reference_original_910b": aggregate(ref_original),
        "candidate_original_310p": aggregate(cand_original),
        "reference_recheck_310p": aggregate(ref_recheck),
        "candidate_recheck_310p": aggregate(cand_recheck),
    }
    pairs = {
        "original_cross_host": pair_summary(ref_original, cand_original),
        "reference_environment_shift": pair_summary(ref_original, ref_recheck),
        "candidate_same_host_repeatability": pair_summary(cand_original, cand_recheck),
        "fresh_same_host_reference_vs_candidate": pair_summary(ref_recheck, cand_recheck),
    }
    same_host = pairs["fresh_same_host_reference_vs_candidate"]
    candidate_repeat = pairs["candidate_same_host_repeatability"]
    environment_shift = pairs["reference_environment_shift"]
    if same_host["score_changed_with_exact_input_count"]:
        classification = "SAME_HOST_CDM_NONDETERMINISM_OR_CONCURRENCY"
    elif candidate_repeat["score_changed_with_exact_input_count"]:
        classification = "CANDIDATE_CDM_NOT_REPEATABLE"
    elif environment_shift["score_changed_with_exact_input_count"]:
        classification = "CROSS_ENVIRONMENT_CDM_DRIFT_CONFIRMED"
    elif same_host["changed_input_count"]:
        classification = "CDM_INPUT_DIFFERENCE_ONLY"
    else:
        classification = "CDM_EXACT_REPRODUCTION"

    ref_pages = page_scores(ref_recheck)
    cand_pages = page_scores(cand_recheck)
    page_rows = []
    for page in sorted(set(ref_pages) & set(cand_pages)):
        page_rows.append(
            {
                "page": page,
                "reference_recheck_310p_page_cdm": ref_pages[page],
                "candidate_recheck_310p_page_cdm": cand_pages[page],
                "delta_candidate_minus_reference": cand_pages[page] - ref_pages[page],
            }
        )
    page_rows.sort(key=lambda row: (row["delta_candidate_minus_reference"], row["page"]))

    candidate_fingerprint = load(parsed.candidate_fingerprint)
    reference_fingerprint = load(parsed.reference_fingerprint) if parsed.reference_fingerprint else None
    environment_differences = (
        []
        if reference_fingerprint is None
        else fingerprint_differences(reference_fingerprint, candidate_fingerprint)
    )
    focused_runtime_comparison = (
        None
        if reference_fingerprint is None
        else runtime_comparison(reference_fingerprint, candidate_fingerprint)
    )
    report = {
        "status": "pass",
        "classification": classification,
        "scores": scores,
        "comparisons": pairs,
        "same_host_page_cdm_delta": (
            float(scores["candidate_recheck_310p"]["page_cdm"])
            - float(scores["reference_recheck_310p"]["page_cdm"])
        ),
        "same_host_overall_percentage_point_contribution": 100.0
        * (
            float(scores["candidate_recheck_310p"]["page_cdm"])
            - float(scores["reference_recheck_310p"]["page_cdm"])
        )
        / 3.0,
        "runtime_fingerprints": {
            "reference_available": reference_fingerprint is not None,
            "candidate": str(parsed.candidate_fingerprint.resolve()),
            "reference": (
                None if parsed.reference_fingerprint is None else str(parsed.reference_fingerprint.resolve())
            ),
            "difference_count": len(environment_differences),
            "differences": environment_differences,
            "focused_comparison": focused_runtime_comparison,
        },
        "worst_30_same_host_pages": page_rows[:30],
    }
    (output / "same_host_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output / "same_host_page_comparison.jsonl").open("w", encoding="utf-8") as handle:
        for row in page_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    lines = [
        "# UniRec CDM same-host audit",
        "",
        f"- Classification: `{classification}`",
        f"- Original Page CDM, 910B / 310P: `{scores['reference_original_910b']['page_cdm']:.9f}` / `{scores['candidate_original_310p']['page_cdm']:.9f}`",
        f"- Fresh 310P Page CDM, 910B output / 310P output: `{scores['reference_recheck_310p']['page_cdm']:.9f}` / `{scores['candidate_recheck_310p']['page_cdm']:.9f}`",
        f"- Fresh same-host Page-CDM delta: `{report['same_host_page_cdm_delta']:+.9f}`",
        f"- Overall contribution: `{report['same_host_overall_percentage_point_contribution']:+.4f}` points",
        f"- Candidate repeat exact-input score changes: `{candidate_repeat['score_changed_with_exact_input_count']}`",
        f"- Reference environment-shift exact-input score changes: `{environment_shift['score_changed_with_exact_input_count']}`",
        f"- Fresh same-host exact-input score changes: `{same_host['score_changed_with_exact_input_count']}`",
        "",
        "Full details are in `same_host_audit.json`.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        "UNIREC_CDM_SAME_HOST_AUDIT PASS "
        f"classification={classification} "
        f"ref_910b_original={scores['reference_original_910b']['page_cdm']:.9f} "
        f"cand_310p_original={scores['candidate_original_310p']['page_cdm']:.9f} "
        f"ref_310p_recheck={scores['reference_recheck_310p']['page_cdm']:.9f} "
        f"cand_310p_recheck={scores['candidate_recheck_310p']['page_cdm']:.9f} "
        f"same_host_delta={report['same_host_page_cdm_delta']:+.9f} "
        f"output={output}"
    )


if __name__ == "__main__":
    main()
