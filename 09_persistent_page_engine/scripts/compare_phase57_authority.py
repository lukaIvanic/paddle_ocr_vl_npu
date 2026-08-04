#!/usr/bin/env python3
"""Mechanically compare a Phase-57 run with the committed 910B authority.

This is deliberately inference-free.  It compares the production contract,
per-crop inputs and generations, final page Markdown, the saved official
metrics, and a same-host reevaluation of the 910B Markdown predictions.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
import unicodedata
import zipfile
from pathlib import Path
from typing import Any


CONTRACT_PATHS = (
    "pipeline",
    "page_preprocessing_mode",
    "configuration.dtype",
    "configuration.batch_size",
    "configuration.cache_length",
    "configuration.max_new_tokens",
    "configuration.decode_backend",
    "configuration.decode_optimization",
    "configuration.preprocessor_min_pixels",
    "configuration.preprocessor_max_pixels",
    "configuration.text_crop_scale",
    "configuration.effective_global_min_pixels",
    "configuration.effective_global_max_pixels",
    "configuration.vision_backend",
    "configuration.vision_attention",
    "configuration.vision_mlp.source_intermediate_size",
    "configuration.vision_mlp.target_intermediate_size",
    "configuration.vision_mlp.zero_extended",
    "configuration.vision_linear_weight_format.effective_mode",
    "configuration.vision_linear_weight_format.all_after_are_nz",
    "configuration.vision_padding",
    "configuration.vision_packing",
    "configuration.vision_pack_target",
    "configuration.vision_router_lookahead",
    "configuration.vision_buckets",
    "configuration.text_buckets",
    "configuration.text_packing",
    "configuration.text_pack_buckets",
    "configuration.text_pack_max_members",
    "configuration.page_preprocessing_mode",
    "layout_frontend.implementation",
    "layout_frontend.graph_capture",
    "layout_frontend.worker_pipeline.strategy",
    "layout_frontend.worker_pipeline.input_workers",
    "layout_frontend.worker_pipeline.page_prepare_workers",
    "layout_frontend.worker_pipeline.max_inflight_pages",
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-bundle", required=True, type=Path)
    parser.add_argument("--candidate-output", required=True, type=Path)
    parser.add_argument("--candidate-metric", required=True, type=Path)
    parser.add_argument("--candidate-cdm-summary", required=True, type=Path)
    parser.add_argument("--reference-recheck-metric", required=True, type=Path)
    parser.add_argument("--reference-recheck-cdm-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _zip_json(archive: zipfile.ZipFile, name: str) -> Any:
    with archive.open(name) as handle:
        return json.load(handle)


def _zip_jsonl(archive: zipfile.ZipFile, name: str) -> list[dict[str, Any]]:
    with archive.open(name) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _get(root: dict[str, Any], dotted: str) -> Any:
    value: Any = root
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return {"missing": True}
        value = value[part]
    return value


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def _trace_key(row: dict[str, Any]) -> tuple[str, int, str]:
    return (
        str(row.get("source_image_name")),
        int(row.get("block_index", -1)),
        str(row.get("label")),
    )


def _trace_index(rows: list[dict[str, Any]], side: str) -> dict[tuple[str, int, str], dict[str, Any]]:
    result: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in rows:
        key = _trace_key(row)
        if key in result:
            raise ValueError(f"duplicate {side} trace key: {key}")
        result[key] = row
    return result


def _metric_summary(metric: dict[str, Any], cdm_summary: dict[str, Any]) -> dict[str, float]:
    evaluated_path = Path(str(cdm_summary["evaluated_samples"]))
    if not evaluated_path.is_file():
        raise FileNotFoundError(evaluated_path)
    evaluated = _json(evaluated_path)
    by_page: dict[str, list[float]] = collections.defaultdict(list)
    for sample in evaluated:
        by_page[str(sample["img_id"])].append(float(sample["metric"]["CDM"]))
    page_cdm = sum(sum(values) / len(values) for values in by_page.values()) / len(by_page)
    sample_cdm = sum(value for values in by_page.values() for value in values) / sum(
        len(values) for values in by_page.values()
    )
    text_edit = float(metric["text_block"]["all"]["Edit_dist"]["ALL_page_avg"])
    page_teds = float(metric["table"]["page"]["TEDS"]["ALL"])
    return {
        "text_edit": text_edit,
        "formula_edit": float(metric["display_formula"]["all"]["Edit_dist"]["ALL_page_avg"]),
        "sample_cdm": sample_cdm,
        "page_cdm": page_cdm,
        "sample_teds": float(metric["table"]["all"]["TEDS"]["all"]),
        "page_teds": page_teds,
        "page_teds_structure_only": float(metric["table"]["page"]["TEDS_structure_only"]["ALL"]),
        "reading_order_edit": float(metric["reading_order"]["all"]["Edit_dist"]["ALL_page_avg"]),
        "official_overall": ((1.0 - text_edit) + page_cdm + page_teds) / 3.0,
    }


def _delta(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    return {key: left[key] - right[key] for key in left.keys() & right.keys()}


def main() -> None:
    args = _args()
    candidate_output = args.candidate_output.resolve()
    candidate_trace = _jsonl(candidate_output / "recognition_trace.jsonl")
    candidate_run = _json(candidate_output / "run_summary.json")
    candidate_metric = _json(args.candidate_metric.resolve())
    candidate_cdm = _json(args.candidate_cdm_summary.resolve())
    recheck_metric = _json(args.reference_recheck_metric.resolve())
    recheck_cdm = _json(args.reference_recheck_cdm_summary.resolve())

    print("[authority-audit] reading committed 910B authority", flush=True)
    with zipfile.ZipFile(args.reference_bundle.resolve()) as archive:
        names = set(archive.namelist())
        required = {
            "manifest.json",
            "recognition_trace.jsonl",
            "run_summary.json",
            "metric_result.json",
            "official_score_summary.json",
        }
        missing = required - names
        if missing:
            raise ValueError(f"reference bundle is missing {sorted(missing)}")
        reference_manifest = _zip_json(archive, "manifest.json")
        reference_trace = _zip_jsonl(archive, "recognition_trace.jsonl")
        reference_run = _zip_json(archive, "run_summary.json")
        reference_metric = _zip_json(archive, "metric_result.json")
        reference_score = _zip_json(archive, "official_score_summary.json")
        reference_markdown = {
            Path(name).name: archive.read(name).decode("utf-8")
            for name in names
            if name.startswith("predictions/") and name.endswith(".md")
        }
    if len(reference_markdown) != 1651:
        raise ValueError(f"expected 1651 reference Markdown files, got {len(reference_markdown)}")

    print("[authority-audit] comparing run contract", flush=True)
    contract_differences = []
    for path in CONTRACT_PATHS:
        reference_value = _get(reference_run, path)
        candidate_value = _get(candidate_run, path)
        if reference_value != candidate_value:
            contract_differences.append(
                {"path": path, "reference": reference_value, "candidate": candidate_value}
            )

    print("[authority-audit] pairing crop generations", flush=True)
    reference_by_key = _trace_index(reference_trace, "reference")
    candidate_by_key = _trace_index(candidate_trace, "candidate")
    shared_keys = sorted(reference_by_key.keys() & candidate_by_key.keys())
    input_fields = (
        "prompt",
        "input_tokens",
        "projected_image_tokens",
        "crop_size",
        "min_pixels",
        "max_pixels",
    )
    generation_counts: collections.Counter[str] = collections.Counter()
    generation_by_label: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    largest_divergences = []
    input_mismatches = []
    for key in shared_keys:
        reference = reference_by_key[key]
        candidate = candidate_by_key[key]
        mismatched = [field for field in input_fields if reference.get(field) != candidate.get(field)]
        if mismatched:
            input_mismatches.append({"key": key, "fields": mismatched})
        reference_text = str(reference.get("text", ""))
        candidate_text = str(candidate.get("text", ""))
        reference_tokens = reference.get("token_ids") or []
        candidate_tokens = candidate.get("token_ids") or []
        if reference_tokens == candidate_tokens and reference_text == candidate_text:
            category = "exact"
        elif _normalized(reference_text) == _normalized(candidate_text):
            category = "whitespace_nfkc_only"
        else:
            category = "content_difference"
        generation_counts[category] += 1
        generation_by_label[key[2]][category] += 1
        if category == "content_difference":
            ref_len = len(reference_tokens)
            cand_len = len(candidate_tokens)
            ratio = (max(ref_len, cand_len) + 1) / (min(ref_len, cand_len) + 1)
            largest_divergences.append(
                {
                    "source_image_name": key[0],
                    "block_index": key[1],
                    "label": key[2],
                    "reference_tokens": ref_len,
                    "candidate_tokens": cand_len,
                    "length_ratio": ratio,
                    "reference_stop": reference.get("stop_reason"),
                    "candidate_stop": candidate.get("stop_reason"),
                    "reference_preview": reference_text[:240],
                    "candidate_preview": candidate_text[:240],
                }
            )
    largest_divergences.sort(key=lambda row: (row["length_ratio"], abs(row["candidate_tokens"] - row["reference_tokens"])), reverse=True)

    print("[authority-audit] comparing final page Markdown", flush=True)
    candidate_prediction_root = candidate_output / "predictions"
    candidate_markdown = {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted(candidate_prediction_root.glob("*.md"))
    }
    shared_pages = sorted(reference_markdown.keys() & candidate_markdown.keys())
    markdown_counts: collections.Counter[str] = collections.Counter()
    markdown_length_divergences = []
    for name in shared_pages:
        reference = reference_markdown[name]
        candidate = candidate_markdown[name]
        if reference == candidate:
            category = "exact"
        elif _normalized(reference) == _normalized(candidate):
            category = "whitespace_nfkc_only"
        else:
            category = "content_difference"
        markdown_counts[category] += 1
        if category == "content_difference":
            ratio = (max(len(reference), len(candidate)) + 1) / (min(len(reference), len(candidate)) + 1)
            markdown_length_divergences.append(
                {
                    "page": name,
                    "reference_chars": len(reference),
                    "candidate_chars": len(candidate),
                    "length_ratio": ratio,
                }
            )
    markdown_length_divergences.sort(key=lambda row: row["length_ratio"], reverse=True)

    print("[authority-audit] comparing metric authorities", flush=True)
    candidate_scores = _metric_summary(candidate_metric, candidate_cdm)
    recheck_scores = _metric_summary(recheck_metric, recheck_cdm)
    reference_scores = {
        "text_edit": float(reference_score["text_block"]["edit_distance"]),
        "formula_edit": float(reference_metric["display_formula"]["all"]["Edit_dist"]["ALL_page_avg"]),
        "sample_cdm": float(reference_score["display_formula"]["sample_cdm"]),
        "page_cdm": float(reference_score["display_formula"]["page_cdm"]),
        "sample_teds": float(reference_score["table"]["sample_teds"]),
        "page_teds": float(reference_score["table"]["page_teds"]),
        "page_teds_structure_only": float(reference_score["table"]["page_teds_structure_only"]),
        "reading_order_edit": float(reference_score["reading_order"]["edit_distance"]),
        "official_overall": float(reference_score["official_overall"]),
    }
    recheck_delta = _delta(recheck_scores, reference_scores)
    evaluator_reproduction_pass = all(
        math.isclose(delta, 0.0, abs_tol=1e-12) for delta in recheck_delta.values()
    )

    report = {
        "classification": (
            "RUN_CONTRACT_MISMATCH"
            if contract_differences or input_mismatches
            else "GENERATION_DIFFERENCE"
            if generation_counts["content_difference"]
            else "EVALUATOR_DIFFERENCE"
            if not evaluator_reproduction_pass
            else "EXACT_REPRODUCTION"
        ),
        "reference_bundle": str(args.reference_bundle.resolve()),
        "reference_manifest": reference_manifest,
        "run_contract": {
            "pass": not contract_differences,
            "differences": contract_differences,
        },
        "crop_inputs": {
            "reference": len(reference_by_key),
            "candidate": len(candidate_by_key),
            "shared": len(shared_keys),
            "reference_only": len(reference_by_key.keys() - candidate_by_key.keys()),
            "candidate_only": len(candidate_by_key.keys() - reference_by_key.keys()),
            "mismatch_count": len(input_mismatches),
            "first_20_mismatches": input_mismatches[:20],
        },
        "crop_generations": {
            "counts": dict(generation_counts),
            "by_label": {label: dict(counts) for label, counts in sorted(generation_by_label.items())},
            "largest_30_length_divergences": largest_divergences[:30],
        },
        "page_markdown": {
            "reference": len(reference_markdown),
            "candidate": len(candidate_markdown),
            "shared": len(shared_pages),
            "reference_only": sorted(reference_markdown.keys() - candidate_markdown.keys()),
            "candidate_only": sorted(candidate_markdown.keys() - reference_markdown.keys()),
            "counts": dict(markdown_counts),
            "largest_30_length_divergences": markdown_length_divergences[:30],
        },
        "scores": {
            "reference_authority_910b": reference_scores,
            "reference_recheck_on_work_server": recheck_scores,
            "candidate_310p": candidate_scores,
            "reference_recheck_minus_authority": recheck_delta,
            "candidate_minus_reference_authority": _delta(candidate_scores, reference_scores),
            "evaluator_reproduction_exact": evaluator_reproduction_pass,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        "PHASE57_AUTHORITY_AUDIT "
        f"classification={report['classification']} "
        f"contract_diffs={len(contract_differences)} "
        f"input_mismatches={len(input_mismatches)} "
        f"crop_exact={generation_counts['exact']} "
        f"crop_ws_only={generation_counts['whitespace_nfkc_only']} "
        f"crop_content_diff={generation_counts['content_difference']} "
        f"page_exact={markdown_counts['exact']} "
        f"page_ws_only={markdown_counts['whitespace_nfkc_only']} "
        f"page_content_diff={markdown_counts['content_difference']} "
        f"reference_recheck_exact={evaluator_reproduction_pass} "
        f"candidate_overall={100*candidate_scores['official_overall']:.4f} "
        f"reference_overall={100*reference_scores['official_overall']:.4f} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
