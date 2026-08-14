#!/usr/bin/env python3
"""Compare a completed UniRec CDM evaluation with the full 910B2 reference.

This is deliberately CPU-only.  The reference is a gzip-compressed tar archive
containing every textual artifact from the completed 1,651-page 910B2 run.
The candidate is an already-completed run root with the same evaluation layout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


REFERENCE_CDM_RESULT = (
    "evaluation_image_tags_stripped/cdm/result/"
    "predictions_quick_match_cdm_result.json"
)
REFERENCE_EVAL_SUMMARY = "evaluation_image_tags_stripped/full_eval_summary.json"
REFERENCE_PREDICTIONS_PREFIX = "evaluation_image_tags_stripped/predictions/"
REFERENCE_RAW_OUTPUT_PREFIX = "output/"


def parse_args() -> argparse.Namespace:
    default_archive = (
        Path(__file__).resolve().parent
        / "references/unirec_full1651_910b_470d8a6_text_outputs.tar.gz"
    )
    parser = argparse.ArgumentParser(
        description="Compare completed 310P UniRec CDM output with 910B2."
    )
    parser.add_argument("--candidate-run-root", type=Path, required=True)
    parser.add_argument("--reference-archive", type=Path, default=default_archive)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-pages", type=int, default=30)
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(json_bytes(value))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


class ReferenceArchive:
    def __init__(self, path: Path):
        self.path = path.resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"reference archive does not exist: {self.path}")
        self.tar = tarfile.open(self.path, mode="r:gz")
        self.members = {
            member.name: member
            for member in self.tar.getmembers()
            if member.isfile()
        }

    def close(self) -> None:
        self.tar.close()

    def read(self, name: str) -> bytes:
        member = self.members.get(name)
        if member is None:
            raise FileNotFoundError(f"reference member is absent: {name}")
        handle = self.tar.extractfile(member)
        if handle is None:
            raise RuntimeError(f"could not read reference member: {name}")
        return handle.read()

    def read_json(self, name: str) -> Any:
        return json.loads(self.read(name).decode("utf-8"))

    def files_under(self, prefix: str, suffix: str) -> dict[str, bytes]:
        result: dict[str, bytes] = {}
        for name in sorted(self.members):
            if not name.startswith(prefix) or not name.endswith(suffix):
                continue
            key = Path(name).name
            if key in result:
                raise RuntimeError(f"duplicate reference basename under {prefix}: {key}")
            result[key] = self.read(name)
        return result


def load_candidate(candidate_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    evaluation = candidate_root / "evaluation_image_tags_stripped"
    cdm_path = evaluation / "cdm/result/predictions_quick_match_cdm_result.json"
    summary_path = evaluation / "full_eval_summary.json"
    if not cdm_path.is_file():
        raise FileNotFoundError(f"candidate CDM result does not exist: {cdm_path}")
    if not summary_path.is_file():
        raise FileNotFoundError(f"candidate summary does not exist: {summary_path}")
    return (
        json.loads(cdm_path.read_text(encoding="utf-8")),
        json.loads(summary_path.read_text(encoding="utf-8")),
    )


def sample_score(sample: dict[str, Any]) -> float:
    return float(sample["metric"]["CDM"])


def sample_key(sample: dict[str, Any]) -> tuple[str, str]:
    image = str(sample.get("img_id") or sample["image_name"])
    gt_idx = json.dumps(sample["gt_idx"], separators=(",", ":"), ensure_ascii=False)
    return image, gt_idx


def index_samples(samples: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for sample in samples:
        key = sample_key(sample)
        if key in result:
            raise RuntimeError(f"duplicate CDM sample key: {key}")
        result[key] = sample
    return result


def page_scores(samples: list[dict[str, Any]]) -> dict[str, float]:
    by_page: dict[str, list[float]] = defaultdict(list)
    for sample in samples:
        by_page[str(sample.get("img_id") or sample["image_name"])].append(
            sample_score(sample)
        )
    return {page: sum(scores) / len(scores) for page, scores in by_page.items()}


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("cannot compute mean of empty values")
    return sum(values) / len(values)


def candidate_files(root: Path, directory: str, suffix: str) -> dict[str, bytes]:
    base = root / directory
    if not base.is_dir():
        raise FileNotFoundError(f"candidate directory does not exist: {base}")
    result: dict[str, bytes] = {}
    for path in sorted(base.rglob(f"*{suffix}")):
        key = path.name
        if key in result:
            raise RuntimeError(f"duplicate candidate basename under {base}: {key}")
        result[key] = path.read_bytes()
    return result


def compare_file_sets(
    reference: dict[str, bytes], candidate: dict[str, bytes]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    exact = 0
    different = 0
    for name in sorted(set(reference) | set(candidate)):
        ref = reference.get(name)
        cand = candidate.get(name)
        same = ref is not None and cand is not None and ref == cand
        if same:
            exact += 1
        elif ref is not None and cand is not None:
            different += 1
        rows.append(
            {
                "name": name,
                "reference_present": ref is not None,
                "candidate_present": cand is not None,
                "exact": same,
                "reference_bytes": None if ref is None else len(ref),
                "candidate_bytes": None if cand is None else len(cand),
                "reference_sha256": None if ref is None else sha256_bytes(ref),
                "candidate_sha256": None if cand is None else sha256_bytes(cand),
            }
        )
    summary = {
        "reference_count": len(reference),
        "candidate_count": len(candidate),
        "union_count": len(rows),
        "exact_count": exact,
        "different_count": different,
        "missing_from_candidate_count": len(set(reference) - set(candidate)),
        "extra_in_candidate_count": len(set(candidate) - set(reference)),
    }
    return summary, rows


def formula_row(
    key: tuple[str, str],
    reference: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
) -> dict[str, Any]:
    ref_score = None if reference is None else sample_score(reference)
    cand_score = None if candidate is None else sample_score(candidate)
    return {
        "image_name": key[0],
        "gt_idx": json.loads(key[1]),
        "reference_present": reference is not None,
        "candidate_present": candidate is not None,
        "reference_cdm": ref_score,
        "candidate_cdm": cand_score,
        "cdm_delta_candidate_minus_reference": (
            None if ref_score is None or cand_score is None else cand_score - ref_score
        ),
        "match_topology_exact": (
            reference is not None
            and candidate is not None
            and reference.get("pred_idx") == candidate.get("pred_idx")
            and reference.get("gt_idx") == candidate.get("gt_idx")
        ),
        "normalized_prediction_exact": (
            reference is not None
            and candidate is not None
            and reference.get("norm_pred") == candidate.get("norm_pred")
        ),
        "reference_pred_idx": None if reference is None else reference.get("pred_idx"),
        "candidate_pred_idx": None if candidate is None else candidate.get("pred_idx"),
        "reference_prediction": None if reference is None else reference.get("pred"),
        "candidate_prediction": None if candidate is None else candidate.get("pred"),
        "reference_normalized_prediction": (
            None if reference is None else reference.get("norm_pred")
        ),
        "candidate_normalized_prediction": (
            None if candidate is None else candidate.get("norm_pred")
        ),
        "ground_truth": (
            reference.get("gt") if reference is not None else candidate.get("gt")
        ),
    }


def markdown_report(summary: dict[str, Any], page_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# UniRec 310P versus 910B2 page CDM comparison",
        "",
        "No inference or evaluation was rerun. Both sides use completed CDM outputs.",
        "",
        "## Headline",
        "",
        f"- 910B2 Page CDM: `{summary['reference']['page_cdm']:.9f}`",
        f"- Candidate Page CDM: `{summary['candidate']['page_cdm']:.9f}`",
        f"- Candidate minus 910B2: `{summary['delta']['page_cdm']:+.9f}`",
        f"- Overall-score contribution: `{summary['delta']['overall_percentage_points_from_page_cdm']:+.4f}` points",
        f"- Formula pages: `{summary['reference']['formula_pages']}` / `{summary['candidate']['formula_pages']}`",
        f"- Formula samples: `{summary['reference']['formula_samples']}` / `{summary['candidate']['formula_samples']}`",
        f"- Page outcomes better / equal / worse: `{summary['page_outcomes']['better']}` / `{summary['page_outcomes']['equal']}` / `{summary['page_outcomes']['worse']}`",
        f"- Exact stripped page Markdown: `{summary['stripped_prediction_markdown']['exact_count']}` / `{summary['stripped_prediction_markdown']['reference_count']}`",
        "",
        "## Largest page-CDM regressions",
        "",
        "| Rank | Page | Formulas ref/candidate | 910B2 | Candidate | Delta | Overall points | Formula text exact | Match topology exact | Markdown exact |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for rank, row in enumerate(page_rows, start=1):
        lines.append(
            "| "
            f"{rank} | `{row['image_name']}` | "
            f"{row['reference_formula_count']}/{row['candidate_formula_count']} | "
            f"{row['reference_page_cdm']:.6f} | {row['candidate_page_cdm']:.6f} | "
            f"{row['cdm_delta_candidate_minus_reference']:+.6f} | "
            f"{row['overall_percentage_points_delta']:+.4f} | "
            f"{row['normalized_prediction_exact_count']}/{row['aligned_formula_count']} | "
            f"{row['match_topology_exact_count']}/{row['aligned_formula_count']} | "
            f"{'yes' if row['stripped_markdown_exact'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "The complete page and formula rows are in `page_cdm_comparison.jsonl` and ",
            "`formula_cdm_comparison.jsonl` in this output directory.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    candidate_root = args.candidate_run_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reference = ReferenceArchive(args.reference_archive)
    try:
        reference_samples = reference.read_json(REFERENCE_CDM_RESULT)
        reference_summary = reference.read_json(REFERENCE_EVAL_SUMMARY)
        reference_prediction_md = reference.files_under(
            REFERENCE_PREDICTIONS_PREFIX, ".md"
        )
        reference_raw_md = reference.files_under(REFERENCE_RAW_OUTPUT_PREFIX, ".md")
        archive_manifest = [
            {
                "name": member.name,
                "size": member.size,
                "sha256": sha256_bytes(reference.read(member.name)),
            }
            for member in sorted(reference.members.values(), key=lambda item: item.name)
        ]
    finally:
        reference.close()

    candidate_samples, candidate_summary = load_candidate(candidate_root)
    reference_index = index_samples(reference_samples)
    candidate_index = index_samples(candidate_samples)
    ref_page_scores = page_scores(reference_samples)
    cand_page_scores = page_scores(candidate_samples)

    reference_page_cdm = mean(ref_page_scores.values())
    candidate_page_cdm = mean(cand_page_scores.values())
    reference_reported = float(reference_summary["display_formula_page_cdm"])
    candidate_reported = float(candidate_summary["display_formula_page_cdm"])
    if not math.isclose(reference_page_cdm, reference_reported, abs_tol=1e-12):
        raise RuntimeError(
            "recomputed reference Page CDM disagrees with its summary: "
            f"{reference_page_cdm} versus {reference_reported}"
        )
    if not math.isclose(candidate_page_cdm, candidate_reported, abs_tol=1e-12):
        raise RuntimeError(
            "recomputed candidate Page CDM disagrees with its summary: "
            f"{candidate_page_cdm} versus {candidate_reported}"
        )

    candidate_prediction_md = candidate_files(
        candidate_root, "evaluation_image_tags_stripped/predictions", ".md"
    )
    candidate_raw_md = candidate_files(candidate_root, "output", ".md")
    prediction_md_summary, prediction_md_rows = compare_file_sets(
        reference_prediction_md, candidate_prediction_md
    )
    raw_md_summary, raw_md_rows = compare_file_sets(reference_raw_md, candidate_raw_md)
    stripped_exact_by_page = {
        Path(row["name"]).stem: bool(row["exact"]) for row in prediction_md_rows
    }
    raw_exact_by_page = {Path(row["name"]).stem: bool(row["exact"]) for row in raw_md_rows}

    formula_rows = [
        formula_row(key, reference_index.get(key), candidate_index.get(key))
        for key in sorted(set(reference_index) | set(candidate_index))
    ]
    formula_rows_by_page: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in formula_rows:
        formula_rows_by_page[row["image_name"]].append(row)
    ref_count_by_page: dict[str, int] = defaultdict(int)
    cand_count_by_page: dict[str, int] = defaultdict(int)
    for sample in reference_samples:
        ref_count_by_page[str(sample["img_id"])] += 1
    for sample in candidate_samples:
        cand_count_by_page[str(sample["img_id"])] += 1

    all_formula_pages = sorted(set(ref_page_scores) | set(cand_page_scores))
    if set(ref_page_scores) != set(cand_page_scores):
        missing = sorted(set(ref_page_scores) - set(cand_page_scores))
        extra = sorted(set(cand_page_scores) - set(ref_page_scores))
        raise RuntimeError(
            "formula page sets differ; page-weighted delta is not directly comparable: "
            f"missing={missing[:10]} extra={extra[:10]}"
        )
    formula_page_count = len(all_formula_pages)
    page_rows: list[dict[str, Any]] = []
    for page in all_formula_pages:
        delta = cand_page_scores[page] - ref_page_scores[page]
        aligned_page_formulas = [
            row
            for row in formula_rows_by_page[page]
            if row["reference_present"] and row["candidate_present"]
        ]
        page_rows.append(
            {
                "image_name": page,
                "reference_page_cdm": ref_page_scores[page],
                "candidate_page_cdm": cand_page_scores[page],
                "cdm_delta_candidate_minus_reference": delta,
                "overall_percentage_points_delta": 100.0 * delta / (3 * formula_page_count),
                "reference_formula_count": ref_count_by_page[page],
                "candidate_formula_count": cand_count_by_page[page],
                "aligned_formula_count": len(aligned_page_formulas),
                "normalized_prediction_exact_count": sum(
                    row["normalized_prediction_exact"] for row in aligned_page_formulas
                ),
                "match_topology_exact_count": sum(
                    row["match_topology_exact"] for row in aligned_page_formulas
                ),
                "stripped_markdown_exact": stripped_exact_by_page.get(Path(page).stem, False),
                "raw_markdown_exact": raw_exact_by_page.get(Path(page).stem, False),
            }
        )
    page_rows.sort(key=lambda row: (row["cdm_delta_candidate_minus_reference"], row["image_name"]))

    epsilon = 1e-12
    page_outcomes = {
        "better": sum(row["cdm_delta_candidate_minus_reference"] > epsilon for row in page_rows),
        "equal": sum(abs(row["cdm_delta_candidate_minus_reference"]) <= epsilon for row in page_rows),
        "worse": sum(row["cdm_delta_candidate_minus_reference"] < -epsilon for row in page_rows),
    }
    aligned_formula_rows = [
        row for row in formula_rows if row["reference_present"] and row["candidate_present"]
    ]
    formula_alignment = {
        "reference_keys": len(reference_index),
        "candidate_keys": len(candidate_index),
        "union_keys": len(formula_rows),
        "aligned_keys": len(aligned_formula_rows),
        "missing_candidate_keys": len(set(reference_index) - set(candidate_index)),
        "extra_candidate_keys": len(set(candidate_index) - set(reference_index)),
        "match_topology_exact": sum(row["match_topology_exact"] for row in aligned_formula_rows),
        "match_topology_changed": sum(
            not row["match_topology_exact"] for row in aligned_formula_rows
        ),
        "normalized_prediction_exact": sum(
            row["normalized_prediction_exact"] for row in aligned_formula_rows
        ),
        "normalized_prediction_changed": sum(
            not row["normalized_prediction_exact"] for row in aligned_formula_rows
        ),
        "cdm_exact": sum(
            abs(row["cdm_delta_candidate_minus_reference"]) <= epsilon
            for row in aligned_formula_rows
        ),
        "cdm_changed": sum(
            abs(row["cdm_delta_candidate_minus_reference"]) > epsilon
            for row in aligned_formula_rows
        ),
    }

    trace_path = candidate_root / "output/recognition_trace.jsonl"
    summary = {
        "status": "pass",
        "reference": {
            "chip": "Ascend 910B2",
            "source_commit": "470d8a6",
            "archive": str(args.reference_archive.resolve()),
            "archive_sha256": sha256_file(args.reference_archive),
            "archive_file_count": len(archive_manifest),
            "formula_pages": len(ref_page_scores),
            "formula_samples": len(reference_samples),
            "page_cdm": reference_page_cdm,
            "official_overall": float(reference_summary["official_overall"]),
        },
        "candidate": {
            "run_root": str(candidate_root),
            "formula_pages": len(cand_page_scores),
            "formula_samples": len(candidate_samples),
            "page_cdm": candidate_page_cdm,
            "official_overall": float(candidate_summary["official_overall"]),
            "recognition_trace": (
                None
                if not trace_path.is_file()
                else {
                    "path": str(trace_path),
                    "bytes": trace_path.stat().st_size,
                    "sha256": sha256_file(trace_path),
                }
            ),
        },
        "delta": {
            "page_cdm": candidate_page_cdm - reference_page_cdm,
            "overall_percentage_points_from_page_cdm": (
                100.0 * (candidate_page_cdm - reference_page_cdm) / 3.0
            ),
            "reported_overall_percentage_points": 100.0
            * (
                float(candidate_summary["official_overall"])
                - float(reference_summary["official_overall"])
            ),
        },
        "page_outcomes": page_outcomes,
        "formula_alignment": formula_alignment,
        "stripped_prediction_markdown": prediction_md_summary,
        "raw_output_markdown": raw_md_summary,
        "output_files": {
            "summary": "summary.json",
            "report": "report.md",
            "page_rows": "page_cdm_comparison.jsonl",
            "formula_rows": "formula_cdm_comparison.jsonl",
            "stripped_prediction_digests": "stripped_prediction_markdown_digests.jsonl",
            "raw_output_digests": "raw_output_markdown_digests.jsonl",
            "reference_archive_manifest": "reference_archive_manifest.json",
            "worst_page_formula_details": "worst_page_formula_details.json",
        },
    }

    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "page_cdm_comparison.jsonl", page_rows)
    write_jsonl(output_dir / "formula_cdm_comparison.jsonl", formula_rows)
    write_jsonl(
        output_dir / "stripped_prediction_markdown_digests.jsonl", prediction_md_rows
    )
    write_jsonl(output_dir / "raw_output_markdown_digests.jsonl", raw_md_rows)
    write_json(output_dir / "reference_archive_manifest.json", archive_manifest)
    top_pages = page_rows[: max(args.top_pages, 0)]
    write_json(
        output_dir / "worst_page_formula_details.json",
        [
            {
                "page": page_row,
                "formulas": [
                    formula for formula in formula_rows_by_page[page_row["image_name"]]
                ],
            }
            for page_row in top_pages
        ],
    )
    (output_dir / "report.md").write_text(
        markdown_report(summary, top_pages), encoding="utf-8"
    )

    print(
        "UNIREC_CDM_PAGE_COMPARE PASS "
        f"reference_page_cdm={reference_page_cdm:.9f} "
        f"candidate_page_cdm={candidate_page_cdm:.9f} "
        f"delta={candidate_page_cdm - reference_page_cdm:+.9f} "
        f"formula_pages={len(ref_page_scores)}/{len(cand_page_scores)} "
        f"formula_samples={len(reference_samples)}/{len(candidate_samples)} "
        f"md_exact={prediction_md_summary['exact_count']}/"
        f"{prediction_md_summary['reference_count']} "
        f"output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()
