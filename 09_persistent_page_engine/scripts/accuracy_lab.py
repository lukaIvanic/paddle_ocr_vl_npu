#!/usr/bin/env python3
"""Compare the fixed Experiment-09 accuracy corpus across two NPU runs.

Unlike the broad E2E comparator, this lab keys requests by original image name
and layout block.  It therefore compares the same crop contract even when a
run uses a different ``--offset`` and renumbers its request IDs.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from compare_e2e_outputs import (
    compare_configurations,
    compare_layout,
    compare_requests,
    read_json,
    read_jsonl,
    require_output,
)


HERE = Path(__file__).resolve().parent
DEFAULT_CASES = HERE.parent / "accuracy_lab/cases.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--reference-output", type=Path, required=True)
    parser.add_argument("--candidate-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worst-limit", type=int, default=20)
    parser.add_argument(
        "--allow-missing-fingerprints",
        action="store_true",
        help="Permit historical traces without accuracy input fingerprints.",
    )
    args = parser.parse_args(argv)
    if args.worst_limit <= 0:
        parser.error("--worst-limit must be positive")
    return args


def load_cases(path: Path) -> dict[str, Any]:
    payload = read_json(path.expanduser().resolve())
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("accuracy case manifest must use schema_version=1")
    page_corpus = payload.get("page_corpus") or {}
    images = page_corpus.get("source_images")
    indices = page_corpus.get("source_page_indices")
    cases = payload.get("cases")
    if not isinstance(images, list) or not images:
        raise ValueError("accuracy case manifest has no source_images")
    if not isinstance(indices, list) or len(indices) != len(images):
        raise ValueError("source_page_indices and source_images must align")
    if not isinstance(cases, list) or not cases:
        raise ValueError("accuracy case manifest has no cases")
    expected_images = set(str(image) for image in images)
    seen_ids: set[str] = set()
    seen_keys: set[tuple[str, int]] = set()
    for case in cases:
        case_id = str(case["case_id"])
        key = (str(case["source_image_name"]), int(case["block_index"]))
        if case_id in seen_ids or key in seen_keys:
            raise ValueError(f"duplicate accuracy case: id={case_id} key={key}")
        if key[0] not in expected_images:
            raise ValueError(f"case image is outside page corpus: {key[0]}")
        seen_ids.add(case_id)
        seen_keys.add(key)
    return payload


def token_sha256(tokens: list[int]) -> str:
    return hashlib.sha256(
        json.dumps(tokens, separators=(",", ":")).encode()
    ).hexdigest()


def normalized_trace(
    root: Path,
    *,
    source_index_by_image: dict[str, int],
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    summary = read_json(root / "run_summary.json")
    images = [str(value) for value in summary.get("images", ())]
    normalized: list[dict[str, Any]] = []
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for original in read_jsonl(root / "recognition_trace.jsonl"):
        page_input_index = int(original["page_input_index"])
        image_name = str(
            original.get("source_image_name")
            or images[page_input_index]
        )
        if image_name not in source_index_by_image:
            continue
        block_index = int(original["block_index"])
        key = (image_name, block_index)
        if key in by_key:
            raise ValueError(f"duplicate stable request key in {root}: {key}")
        row = dict(original)
        row["original_request_id"] = str(original["request_id"])
        row["source_image_name"] = image_name
        row["source_page_index"] = source_index_by_image[image_name]
        row["request_id"] = f"{image_name}#block_{block_index:06d}"
        row["page_input_index"] = source_index_by_image[image_name]
        normalized.append(row)
        by_key[key] = row
    return normalized, by_key


def filtered_layout(root: Path, images: set[str]) -> list[dict[str, Any]]:
    return [
        row
        for row in read_jsonl(root / "page_regions.jsonl")
        if str(row.get("image_name")) in images
    ]


def filtered_page_outputs(
    reference_root: Path,
    candidate_root: Path,
    images: list[str],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for image in images:
        name = f"{Path(image).stem}.md"
        left_path = reference_root / "predictions" / name
        right_path = candidate_root / "predictions" / name
        if not left_path.is_file() or not right_path.is_file():
            raise FileNotFoundError(f"missing fixed-corpus prediction: {name}")
        left = left_path.read_text(encoding="utf-8")
        right = right_path.read_text(encoding="utf-8")
        rows.append(
            {
                "image_name": image,
                "exact": left == right,
                "sequence_ratio": difflib.SequenceMatcher(
                    None, left, right, autojunk=False
                ).ratio(),
                "reference_characters": len(left),
                "candidate_characters": len(right),
            }
        )
    return {
        "pages": len(rows),
        "exact_pages": sum(row["exact"] for row in rows),
        "minimum_sequence_ratio": min(
            (float(row["sequence_ratio"]) for row in rows),
            default=None,
        ),
        "per_page": rows,
    }


def fingerprint_missing(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "crop": sum(
            not ((row.get("input_fingerprints") or {}).get("crop") or {}).get(
                "sha256"
            )
            for row in rows
        ),
        "prepared_inputs": sum(
            not (row.get("input_fingerprints") or {}).get(
                "prepared_inputs_sha256"
            )
            for row in rows
        ),
    }


def selected_cases(
    manifest: dict[str, Any],
    reference_by_key: dict[tuple[str, int], dict[str, Any]],
    candidate_by_key: dict[tuple[str, int], dict[str, Any]],
    comparisons: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    contract_warnings: list[str] = []
    for case in manifest["cases"]:
        image = str(case["source_image_name"])
        block = int(case["block_index"])
        key = (image, block)
        if key not in reference_by_key or key not in candidate_by_key:
            raise KeyError(f"fixed accuracy case is missing from a trace: {key}")
        reference = reference_by_key[key]
        comparison_id = f"{image}#block_{block:06d}"
        comparison = comparisons[comparison_id]
        structural_contract = {
            "reference_label": reference.get("label"),
            "reference_prompt": reference.get("prompt"),
            "reference_input_tokens": reference.get("input_tokens"),
            "reference_projected_image_tokens": reference.get(
                "projected_image_tokens"
            ),
        }
        for field, actual in structural_contract.items():
            expected = case.get(field)
            if expected != actual:
                contract_warnings.append(
                    f"{case['case_id']}: {field} expected={expected!r} actual={actual!r}"
                )
        historical_tokens_exact = (
            int(case["reference_output_tokens"])
            == len(reference.get("token_ids") or ())
            and str(case["reference_token_sha256"])
            == token_sha256([int(value) for value in reference.get("token_ids", ())])
        )
        rows.append(
            {
                "case_id": str(case["case_id"]),
                "role": str(case["role"]),
                "phase38_observation": str(case["phase38_observation"]),
                "source_page_index": int(case["source_page_index"]),
                "source_image_name": image,
                "block_index": block,
                "reference_request_id": reference["original_request_id"],
                "candidate_request_id": candidate_by_key[key][
                    "original_request_id"
                ],
                "historical_910b_token_contract_exact": historical_tokens_exact,
                **comparison,
            }
        )
    return rows, contract_warnings


def decisive_classification(rows: list[dict[str, Any]]) -> dict[str, Any]:
    divergent = [row for row in rows if not row["token_ids_exact"]]
    exact_input_and_route = [
        row
        for row in divergent
        if row["prepared_input_fingerprint_status"] == "exact"
        and row["vision_route_status"] == "exact"
        and row["text_prefill_route_status"] == "exact"
    ]
    different_prepared = [
        row
        for row in divergent
        if row["prepared_input_fingerprint_status"] == "different"
    ]
    unavailable = [
        row
        for row in divergent
        if row["prepared_input_fingerprint_status"] == "unavailable"
    ]
    if exact_input_and_route:
        label = "MODEL_EXECUTION_DIFFERENCE_PROVEN"
    elif divergent and len(different_prepared) == len(divergent):
        label = "ALL_DIVERGENCES_HAVE_DIFFERENT_PREPARED_INPUTS"
    elif unavailable:
        label = "INPUT_IDENTITY_UNRESOLVED"
    elif not divergent:
        label = "FIXED_CORPUS_TOKEN_EXACT"
    else:
        label = "MIXED_INPUT_AND_ROUTE_EVIDENCE"
    return {
        "label": label,
        "divergent_crops": len(divergent),
        "exact_prepared_input_and_route_divergences": len(
            exact_input_and_route
        ),
        "different_prepared_input_divergences": len(different_prepared),
        "unavailable_prepared_input_divergences": len(unavailable),
        "exact_input_and_route_case_ids": [
            row.get("case_id", row["request_id"])
            for row in exact_input_and_route
        ],
    }


def status_total(cross_tab: dict[str, int], status: str) -> int:
    return sum(
        int(count)
        for key, count in cross_tab.items()
        if key.startswith(f"{status} -> ")
    )


def render_markdown(report: dict[str, Any]) -> str:
    recognition = report["fixed_page_corpus"]["recognition"]
    layout = report["fixed_page_corpus"]["layout"]
    pages = report["fixed_page_corpus"]["page_outputs"]
    cross = recognition["evidence_cross_tabs"]
    lines = [
        "# Experiment 09 fixed-corpus accuracy lab",
        "",
        f"- Classification: **{report['classification']['label']}**",
        f"- Fixed pages: **{len(report['page_corpus']['source_images'])}**",
        f"- Shared crops: **{recognition['shared_requests']}**",
        f"- Token-exact crops: **{recognition['token_exact_requests']}/{recognition['shared_requests']}**",
        f"- First-token divergences: **{recognition['first_generated_token_differences']}**",
        f"- Later divergences: **{recognition['after_shared_prefix_differences']}**",
        f"- Exact crop hashes: **{status_total(cross['crop_fingerprint_status'], 'exact')}/{recognition['shared_requests']}**",
        f"- Exact prepared-input hashes: **{status_total(cross['prepared_input_fingerprint_status'], 'exact')}/{recognition['shared_requests']}**",
        f"- Layout geometry exact: **{layout['geometry_exact_pages']}/{layout['shared_pages']} pages**",
        f"- Assembled Markdown exact: **{pages['exact_pages']}/{pages['pages']} pages**",
        "",
        "## Selected diagnostic crops",
        "",
        "| Case | Role | Label | Input | Vision route | Text route | Divergence | First index | Ref/Candidate tokens | Text ratio |",
        "|---|---|---|---|---|---|---|---:|---:|---:|",
    ]
    for row in report["selected_cases"]:
        lines.append(
            f"| `{row['case_id']}` | {row['role']} | {row['label']} | "
            f"crop={row['crop_fingerprint_status']}, prepared={row['prepared_input_fingerprint_status']} | "
            f"{row['vision_route_status']} | {row['text_prefill_route_status']} | "
            f"{row['divergence_kind']} | {row['first_divergence_index']} | "
            f"{row['reference_tokens']}/{row['candidate_tokens']} | "
            f"{row['text_sequence_ratio']:.4f} |"
        )
    lines.extend(["", "## Evidence cross-tabs", ""])
    for name, table in cross.items():
        lines.append(f"### {name}")
        lines.append("")
        for key, count in table.items():
            lines.append(f"- `{key}`: {count}")
        lines.append("")
    if report["contract_warnings"]:
        lines.extend(["## Contract warnings", ""])
        lines.extend(f"- {warning}" for warning in report["contract_warnings"])
        lines.append("")
    lines.extend(
        [
            "Full token excerpts, routes, hashes, per-page counts, and all fixed-corpus crop comparisons are in `report.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    cases_path = args.cases.expanduser().resolve()
    manifest = load_cases(cases_path)
    reference_root = require_output(args.reference_output)
    candidate_root = require_output(args.candidate_output)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    page_corpus = manifest["page_corpus"]
    images = [str(image) for image in page_corpus["source_images"]]
    image_set = set(images)
    source_index_by_image = {
        str(image): int(index)
        for image, index in zip(
            images,
            page_corpus["source_page_indices"],
            strict=True,
        )
    }
    reference_rows, reference_by_key = normalized_trace(
        reference_root,
        source_index_by_image=source_index_by_image,
    )
    candidate_rows, candidate_by_key = normalized_trace(
        candidate_root,
        source_index_by_image=source_index_by_image,
    )
    recognition = compare_requests(
        reference_rows,
        candidate_rows,
        worst_limit=args.worst_limit,
    )
    per_request = {
        str(row["request_id"]): row
        for row in recognition["per_request"]
    }
    selected, contract_warnings = selected_cases(
        manifest,
        reference_by_key,
        candidate_by_key,
        per_request,
    )
    reference_missing = fingerprint_missing(reference_rows)
    candidate_missing = fingerprint_missing(candidate_rows)
    fingerprint_contract_exact = not any(
        (*reference_missing.values(), *candidate_missing.values())
    )
    if not fingerprint_contract_exact:
        contract_warnings.append(
            "input fingerprints are missing: "
            f"reference={reference_missing} candidate={candidate_missing}"
        )

    report = {
        "schema_version": 1,
        "kind": "experiment09_fixed_corpus_accuracy_lab",
        "cases_path": str(cases_path),
        "reference_output": str(reference_root),
        "candidate_output": str(candidate_root),
        "page_corpus": page_corpus,
        "fingerprint_contract": {
            "exact": fingerprint_contract_exact,
            "reference_missing": reference_missing,
            "candidate_missing": candidate_missing,
        },
        "configuration": compare_configurations(
            read_json(reference_root / "run_summary.json"),
            read_json(candidate_root / "run_summary.json"),
        ),
        "fixed_page_corpus": {
            "layout": compare_layout(
                filtered_layout(reference_root, image_set),
                filtered_layout(candidate_root, image_set),
            ),
            "recognition": recognition,
            "page_outputs": filtered_page_outputs(
                reference_root,
                candidate_root,
                images,
            ),
        },
        "selected_cases": selected,
        "contract_warnings": contract_warnings,
    }
    report["classification"] = decisive_classification(
        recognition["per_request"]
    )
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown = render_markdown(report)
    markdown_path = output_dir / "report.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    print(markdown, flush=True)
    print(f"json={report_path}", flush=True)
    print(f"markdown={markdown_path}", flush=True)
    if not fingerprint_contract_exact and not args.allow_missing_fingerprints:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
