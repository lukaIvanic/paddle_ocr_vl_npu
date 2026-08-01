#!/usr/bin/env python3
"""Compare two Experiment 09 E2E outputs crop by crop and page by page.

The comparison deliberately separates four boundaries:

1. run configuration;
2. layout geometry and recognition-request construction;
3. generated token streams, including the first prefill-produced token;
4. assembled page Markdown.

It is dependency-free so the same command runs in both the 910B and 310P
environments without importing torch, torch_npu, PaddleX, or the evaluator.
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


TRACE_NAME = "recognition_trace.jsonl"
REGIONS_NAME = "page_regions.jsonl"
SUMMARY_NAME = "run_summary.json"

REQUEST_INPUT_FIELDS = (
    "global_request_index",
    "page_input_index",
    "block_index",
    "label",
    "prompt",
    "crop_size",
    "min_pixels",
    "max_pixels",
    "input_tokens",
    "projected_image_tokens",
    "spotting_group_size",
)

CONFIG_FIELDS = (
    "dtype",
    "batch_size",
    "cache_length",
    "max_new_tokens",
    "decode_backend",
    "decode_optimization",
    "preprocessor_min_pixels",
    "effective_global_min_pixels",
    "vision_backend",
    "vision_attention",
    "vision_padding",
    "vision_packing",
    "vision_pack_target",
    "vision_router_lookahead",
    "vision_buckets",
    "text_buckets",
    "text_packing",
    "text_pack_buckets",
    "text_pack_max_members",
    "layout_graph_capture",
    "page_preprocessing_mode",
    "recognition_input_fingerprints",
)

LAYOUT_BLOCK_FIELDS = (
    "block_label",
    "block_bbox",
    "block_id",
    "block_order",
    "group_id",
    "block_polygon_points",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-output", type=Path, required=True)
    parser.add_argument("--candidate-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worst-limit", type=int, default=30)
    args = parser.parse_args(argv)
    if args.worst_limit <= 0:
        parser.error("--worst-limit must be positive")
    return args


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def require_output(root: Path) -> Path:
    root = root.expanduser().resolve()
    for name in (TRACE_NAME, REGIONS_NAME, SUMMARY_NAME):
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)
    if not (root / "predictions").is_dir():
        raise FileNotFoundError(root / "predictions")
    return root


def percentile(values: Iterable[float], quantile: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    materialized = [float(value) for value in values]
    if not materialized:
        return {"count": 0, "min": None, "p50": None, "p95": None, "max": None}
    return {
        "count": len(materialized),
        "min": min(materialized),
        "p50": percentile(materialized, 0.50),
        "p95": percentile(materialized, 0.95),
        "max": max(materialized),
    }


def common_prefix(left: list[int], right: list[int]) -> int:
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return index
    return min(len(left), len(right))


def common_suffix(left: list[int], right: list[int], prefix: int) -> int:
    maximum = min(len(left), len(right)) - prefix
    count = 0
    while count < maximum and left[-count - 1] == right[-count - 1]:
        count += 1
    return count


def sequence_ratio(left: Sequence[Any], right: Sequence[Any]) -> float:
    return difflib.SequenceMatcher(
        None,
        left,
        right,
        autojunk=False,
    ).ratio()


def compact_text(text: str) -> str:
    return "".join(text.split())


def token_excerpt(tokens: list[int], divergence: int, radius: int = 4) -> list[int]:
    start = max(0, divergence - radius)
    end = min(len(tokens), divergence + radius + 1)
    return tokens[start:end]


def exact_status(left: Any, right: Any) -> str:
    if left is None or right is None:
        return "unavailable"
    return "exact" if left == right else "different"


def input_fingerprint(row: dict[str, Any], name: str) -> str | None:
    fingerprints = row.get("input_fingerprints") or {}
    if name == "crop":
        return (fingerprints.get("crop") or {}).get("sha256")
    if name == "prepared_inputs":
        return fingerprints.get("prepared_inputs_sha256")
    return (
        (fingerprints.get("tensors") or {}).get(name) or {}
    ).get("sha256")


VISION_ROUTE_FIELDS = (
    "execution",
    "real_vision_tokens",
    "physical_vision_tokens",
    "bucket",
    "packing",
    "pack_crops",
    "pack_real_vision_tokens",
    "pack_physical_vision_tokens",
    "pack_batch_size",
    "pack_sequence_length",
    "pack_row_sizes",
)

TEXT_ROUTE_FIELDS = (
    "execution",
    "real_text_tokens",
    "physical_text_tokens",
    "bucket",
    "packing",
    "pack_members",
    "segment_lengths",
    "pack_real_text_tokens",
    "pack_physical_text_tokens",
)


def route_signature(row: dict[str, Any], section: str) -> dict[str, Any]:
    route = dict(row.get(section) or {})
    fields = VISION_ROUTE_FIELDS if section == "vision" else TEXT_ROUTE_FIELDS
    return {field: route.get(field) for field in fields}


def compare_configurations(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    left = dict(reference.get("configuration") or {})
    right = dict(candidate.get("configuration") or {})
    rows = {
        field: {
            "reference": left.get(field),
            "candidate": right.get(field),
            "exact": left.get(field) == right.get(field),
        }
        for field in CONFIG_FIELDS
    }
    return {
        "exact_fields": sum(row["exact"] for row in rows.values()),
        "different_fields": [field for field, row in rows.items() if not row["exact"]],
        "fields": rows,
    }


def layout_signature(page: dict[str, Any]) -> dict[str, Any]:
    return {
        "image_name": page.get("image_name"),
        "width": page.get("width"),
        "height": page.get("height"),
        "blocks": [
            {field: block.get(field) for field in LAYOUT_BLOCK_FIELDS}
            for block in page.get("parsing_res_list", ())
        ],
    }


def compare_layout(
    reference_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    reference = {str(row["image_name"]): row for row in reference_rows}
    candidate = {str(row["image_name"]): row for row in candidate_rows}
    shared = sorted(reference.keys() & candidate.keys())
    differing: list[dict[str, Any]] = []
    for image_name in shared:
        left = layout_signature(reference[image_name])
        right = layout_signature(candidate[image_name])
        if left == right:
            continue
        left_blocks = left["blocks"]
        right_blocks = right["blocks"]
        first_block = next(
            (
                index
                for index, (left_block, right_block) in enumerate(
                    zip(left_blocks, right_blocks)
                )
                if left_block != right_block
            ),
            min(len(left_blocks), len(right_blocks)),
        )
        differing.append(
            {
                "image_name": image_name,
                "reference_blocks": len(left_blocks),
                "candidate_blocks": len(right_blocks),
                "first_different_block": first_block,
                "reference_block": (
                    left_blocks[first_block] if first_block < len(left_blocks) else None
                ),
                "candidate_block": (
                    right_blocks[first_block] if first_block < len(right_blocks) else None
                ),
            }
        )
    return {
        "reference_pages": len(reference),
        "candidate_pages": len(candidate),
        "shared_pages": len(shared),
        "missing_from_candidate": sorted(reference.keys() - candidate.keys()),
        "extra_in_candidate": sorted(candidate.keys() - reference.keys()),
        "geometry_exact_pages": len(shared) - len(differing),
        "geometry_different_pages": len(differing),
        "geometry_differences": differing,
    }


def request_comparison(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    reference_tokens = [int(value) for value in reference.get("token_ids", ())]
    candidate_tokens = [int(value) for value in candidate.get("token_ids", ())]
    prefix = common_prefix(reference_tokens, candidate_tokens)
    suffix = common_suffix(reference_tokens, candidate_tokens, prefix)
    minimum_length = min(len(reference_tokens), len(candidate_tokens))
    denominator = max(1, max(len(reference_tokens), len(candidate_tokens)))
    token_exact = reference_tokens == candidate_tokens
    reference_text = str(reference.get("text", ""))
    candidate_text = str(candidate.get("text", ""))
    tensor_names = (
        "attention_mask",
        "image_grid_thw",
        "input_ids",
        "pixel_values",
        "position_ids",
        "rope_deltas",
    )
    tensor_fingerprint_status = {
        name: exact_status(
            input_fingerprint(reference, name),
            input_fingerprint(candidate, name),
        )
        for name in tensor_names
    }
    input_differences = {
        field: {
            "reference": reference.get(field),
            "candidate": candidate.get(field),
        }
        for field in REQUEST_INPUT_FIELDS
        if reference.get(field) != candidate.get(field)
    }
    if token_exact:
        divergence_kind = "exact"
    elif prefix == 0:
        divergence_kind = "first_generated_token"
    elif prefix == minimum_length:
        divergence_kind = "shared_stream_then_length_difference"
    else:
        divergence_kind = "after_shared_prefix"
    return {
        "request_id": str(reference["request_id"]),
        "page_input_index": reference.get("page_input_index"),
        "block_index": reference.get("block_index"),
        "label": reference.get("label"),
        "input_exact": not input_differences,
        "input_differences": input_differences,
        "crop_fingerprint_status": exact_status(
            input_fingerprint(reference, "crop"),
            input_fingerprint(candidate, "crop"),
        ),
        "prepared_input_fingerprint_status": exact_status(
            input_fingerprint(reference, "prepared_inputs"),
            input_fingerprint(candidate, "prepared_inputs"),
        ),
        "tensor_fingerprint_status": tensor_fingerprint_status,
        "reference_crop_sha256": input_fingerprint(reference, "crop"),
        "candidate_crop_sha256": input_fingerprint(candidate, "crop"),
        "reference_prepared_inputs_sha256": input_fingerprint(
            reference, "prepared_inputs"
        ),
        "candidate_prepared_inputs_sha256": input_fingerprint(
            candidate, "prepared_inputs"
        ),
        "vision_route_status": exact_status(
            route_signature(reference, "vision"),
            route_signature(candidate, "vision"),
        ),
        "text_prefill_route_status": exact_status(
            route_signature(reference, "text_prefill"),
            route_signature(candidate, "text_prefill"),
        ),
        "reference_vision_route": route_signature(reference, "vision"),
        "candidate_vision_route": route_signature(candidate, "vision"),
        "reference_text_prefill_route": route_signature(
            reference, "text_prefill"
        ),
        "candidate_text_prefill_route": route_signature(
            candidate, "text_prefill"
        ),
        "token_ids_exact": token_exact,
        "text_exact": reference_text == candidate_text,
        "compact_text_exact": compact_text(reference_text) == compact_text(candidate_text),
        "stop_reason_exact": reference.get("stop_reason") == candidate.get("stop_reason"),
        "reference_stop_reason": reference.get("stop_reason"),
        "candidate_stop_reason": candidate.get("stop_reason"),
        "reference_tokens": len(reference_tokens),
        "candidate_tokens": len(candidate_tokens),
        "token_count_delta": len(candidate_tokens) - len(reference_tokens),
        "common_prefix_tokens": prefix,
        "common_prefix_fraction": prefix / denominator,
        "common_suffix_tokens": suffix,
        "token_sequence_ratio": sequence_ratio(reference_tokens, candidate_tokens),
        "text_sequence_ratio": sequence_ratio(reference_text, candidate_text),
        "divergence_kind": divergence_kind,
        "first_divergence_index": None if token_exact else prefix,
        "reference_token_excerpt": (
            [] if token_exact else token_excerpt(reference_tokens, prefix)
        ),
        "candidate_token_excerpt": (
            [] if token_exact else token_excerpt(candidate_tokens, prefix)
        ),
        "reference_text_excerpt": reference_text[:400],
        "candidate_text_excerpt": candidate_text[:400],
    }


def compare_requests(
    reference_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    worst_limit: int,
) -> dict[str, Any]:
    reference = {str(row["request_id"]): row for row in reference_rows}
    candidate = {str(row["request_id"]): row for row in candidate_rows}
    reference_order = [str(row["request_id"]) for row in reference_rows]
    candidate_order = [str(row["request_id"]) for row in candidate_rows]
    shared = [request_id for request_id in reference_order if request_id in candidate]
    comparisons = [
        request_comparison(reference[request_id], candidate[request_id])
        for request_id in shared
    ]
    divergent = [row for row in comparisons if not row["token_ids_exact"]]
    by_label: dict[str, dict[str, Any]] = {}
    for label in sorted({str(row["label"]) for row in comparisons}):
        rows = [row for row in comparisons if str(row["label"]) == label]
        different = [row for row in rows if not row["token_ids_exact"]]
        by_label[label] = {
            "requests": len(rows),
            "token_exact": len(rows) - len(different),
            "token_different": len(different),
            "token_exact_fraction": (
                (len(rows) - len(different)) / len(rows) if rows else None
            ),
            "candidate_minus_reference_tokens": sum(
                int(row["token_count_delta"]) for row in rows
            ),
            "first_generated_token_differences": sum(
                row["divergence_kind"] == "first_generated_token" for row in rows
            ),
        }
    page_counts: dict[int, Counter[str]] = defaultdict(Counter)
    for row in comparisons:
        page = int(row["page_input_index"])
        page_counts[page]["requests"] += 1
        page_counts[page]["token_different"] += int(not row["token_ids_exact"])
        page_counts[page]["first_token_different"] += int(
            row["divergence_kind"] == "first_generated_token"
        )
        page_counts[page]["token_count_delta"] += int(row["token_count_delta"])
    worst = sorted(
        divergent,
        key=lambda row: (
            float(row["token_sequence_ratio"]),
            float(row["text_sequence_ratio"]),
            int(row["common_prefix_tokens"]),
            str(row["request_id"]),
        ),
    )[:worst_limit]
    stop_pairs = Counter(
        f"{row['reference_stop_reason']} -> {row['candidate_stop_reason']}"
        for row in comparisons
    )
    evidence_cross_tabs = {}
    for field in (
        "crop_fingerprint_status",
        "prepared_input_fingerprint_status",
        "vision_route_status",
        "text_prefill_route_status",
    ):
        counts = Counter(
            f"{row[field]} -> {row['divergence_kind']}"
            for row in comparisons
        )
        evidence_cross_tabs[field] = dict(sorted(counts.items()))
    input_counts = Counter(
        f"{'exact' if row['input_exact'] else 'different'} -> "
        f"{row['divergence_kind']}"
        for row in comparisons
    )
    evidence_cross_tabs["recorded_request_metadata"] = dict(
        sorted(input_counts.items())
    )
    return {
        "reference_requests": len(reference),
        "candidate_requests": len(candidate),
        "shared_requests": len(comparisons),
        "request_order_exact": reference_order == candidate_order,
        "missing_from_candidate": sorted(reference.keys() - candidate.keys()),
        "extra_in_candidate": sorted(candidate.keys() - reference.keys()),
        "input_exact_requests": sum(row["input_exact"] for row in comparisons),
        "input_different_requests": sum(not row["input_exact"] for row in comparisons),
        "token_exact_requests": len(comparisons) - len(divergent),
        "token_different_requests": len(divergent),
        "token_exact_fraction": (
            (len(comparisons) - len(divergent)) / len(comparisons)
            if comparisons
            else None
        ),
        "first_generated_token_differences": sum(
            row["divergence_kind"] == "first_generated_token"
            for row in comparisons
        ),
        "after_shared_prefix_differences": sum(
            row["divergence_kind"] == "after_shared_prefix"
            for row in comparisons
        ),
        "length_only_differences": sum(
            row["divergence_kind"] == "shared_stream_then_length_difference"
            for row in comparisons
        ),
        "text_exact_requests": sum(row["text_exact"] for row in comparisons),
        "compact_text_exact_requests": sum(
            row["compact_text_exact"] for row in comparisons
        ),
        "candidate_minus_reference_tokens": sum(
            int(row["token_count_delta"]) for row in comparisons
        ),
        "common_prefix_tokens": distribution(
            row["common_prefix_tokens"] for row in divergent
        ),
        "common_prefix_fraction": distribution(
            row["common_prefix_fraction"] for row in divergent
        ),
        "token_sequence_ratio": distribution(
            row["token_sequence_ratio"] for row in divergent
        ),
        "text_sequence_ratio": distribution(
            row["text_sequence_ratio"] for row in divergent
        ),
        "divergence_kind_counts": dict(
            sorted(Counter(row["divergence_kind"] for row in comparisons).items())
        ),
        "stop_reason_pairs": dict(sorted(stop_pairs.items())),
        "evidence_cross_tabs": evidence_cross_tabs,
        "by_label": by_label,
        "by_page": {
            str(page): dict(counts)
            for page, counts in sorted(page_counts.items())
        },
        "worst_divergences": worst,
        "per_request": comparisons,
    }


def cross_tab_status_total(table: dict[str, int], status: str) -> int:
    return sum(
        int(count)
        for key, count in table.items()
        if key.startswith(f"{status} -> ")
    )


def read_predictions(root: Path) -> dict[str, str]:
    return {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted((root / "predictions").glob("*.md"))
    }


def compare_predictions(reference_root: Path, candidate_root: Path) -> dict[str, Any]:
    reference = read_predictions(reference_root)
    candidate = read_predictions(candidate_root)
    shared = sorted(reference.keys() & candidate.keys())
    rows = []
    for name in shared:
        left = reference[name]
        right = candidate[name]
        rows.append(
            {
                "name": name,
                "exact": left == right,
                "compact_exact": compact_text(left) == compact_text(right),
                "sequence_ratio": sequence_ratio(left, right),
                "reference_characters": len(left),
                "candidate_characters": len(right),
            }
        )
    return {
        "reference_pages": len(reference),
        "candidate_pages": len(candidate),
        "shared_pages": len(shared),
        "missing_from_candidate": sorted(reference.keys() - candidate.keys()),
        "extra_in_candidate": sorted(candidate.keys() - reference.keys()),
        "exact_pages": sum(row["exact"] for row in rows),
        "compact_exact_pages": sum(row["compact_exact"] for row in rows),
        "sequence_ratio": distribution(row["sequence_ratio"] for row in rows),
        "per_page": rows,
    }


def render_markdown(report: dict[str, Any]) -> str:
    requests = report["recognition"]
    layout = report["layout"]
    pages = report["page_outputs"]
    cross_tabs = requests["evidence_cross_tabs"]
    exact_crop_fingerprints = cross_tab_status_total(
        cross_tabs["crop_fingerprint_status"], "exact"
    )
    exact_prepared_fingerprints = cross_tab_status_total(
        cross_tabs["prepared_input_fingerprint_status"], "exact"
    )
    lines = [
        "# Experiment 09 E2E output comparison",
        "",
        f"- Reference: `{report['reference_output']}`",
        f"- Candidate: `{report['candidate_output']}`",
        "",
        "## Boundary summary",
        "",
        f"- Layout geometry exact: **{layout['geometry_exact_pages']}/{layout['shared_pages']} pages**",
        f"- Recorded request metadata exact: **{requests['input_exact_requests']}/{requests['shared_requests']} crops**",
        f"- Crop-pixel fingerprints exact: **{exact_crop_fingerprints}/{requests['shared_requests']} crops** (missing fingerprints remain unavailable)",
        f"- Prepared-input fingerprints exact: **{exact_prepared_fingerprints}/{requests['shared_requests']} crops** (missing fingerprints remain unavailable)",
        f"- Generated token streams exact: **{requests['token_exact_requests']}/{requests['shared_requests']} crops**",
        f"- First generated token differs: **{requests['first_generated_token_differences']} crops**",
        f"- Diverges after a shared prefix: **{requests['after_shared_prefix_differences']} crops**",
        f"- Only stream length differs: **{requests['length_only_differences']} crops**",
        f"- Candidate minus reference output tokens: **{requests['candidate_minus_reference_tokens']:+d}**",
        f"- Assembled Markdown exact: **{pages['exact_pages']}/{pages['shared_pages']} pages**",
        "",
        "The first generated token is produced by multimodal prefill. A mismatch at token zero therefore proves a prefill-output difference. A later mismatch only proves divergence after a shared prefix; it does not by itself distinguish prefill-KV drift from decode drift.",
        "When both traces include accuracy fingerprints, the crop hash covers exact RGB crop bytes and the prepared-input hash covers pixel values, token IDs, masks, image grid, MRoPE positions, and rope deltas before H2D. Older traces report these fields as unavailable.",
        "",
        "## Configuration differences",
        "",
    ]
    differences = report["configuration"]["different_fields"]
    if differences:
        lines.extend(
            f"- `{field}`: `{report['configuration']['fields'][field]['reference']}` -> `{report['configuration']['fields'][field]['candidate']}`"
            for field in differences
        )
    else:
        lines.append("- None across the compared fields.")
    lines.extend(["", "## Recognition differences by label", ""])
    lines.append("| Label | Requests | Exact | Different | First-token differences | Token delta |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for label, row in requests["by_label"].items():
        lines.append(
            f"| {label} | {row['requests']} | {row['token_exact']} | "
            f"{row['token_different']} | {row['first_generated_token_differences']} | "
            f"{row['candidate_minus_reference_tokens']:+d} |"
        )
    lines.extend(["", "## Worst crop divergences", ""])
    lines.append("| Request | Label | First divergence | Ref/Candidate tokens | Token ratio | Text ratio |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in requests["worst_divergences"]:
        lines.append(
            f"| `{row['request_id']}` | {row['label']} | "
            f"{row['first_divergence_index']} | "
            f"{row['reference_tokens']}/{row['candidate_tokens']} | "
            f"{row['token_sequence_ratio']:.4f} | {row['text_sequence_ratio']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Stop reasons",
            "",
        ]
    )
    lines.extend(
        f"- `{pair}`: {count}"
        for pair, count in requests["stop_reason_pairs"].items()
    )
    lines.extend(
        [
            "",
            "Full per-request evidence, token excerpts, per-page counts, and layout differences are in `comparison.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    reference_root = require_output(args.reference_output)
    candidate_root = require_output(args.candidate_output)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    reference_summary = read_json(reference_root / SUMMARY_NAME)
    candidate_summary = read_json(candidate_root / SUMMARY_NAME)
    report = {
        "schema_version": 1,
        "kind": "experiment09_e2e_output_comparison",
        "reference_output": str(reference_root),
        "candidate_output": str(candidate_root),
        "configuration": compare_configurations(
            reference_summary,
            candidate_summary,
        ),
        "layout": compare_layout(
            read_jsonl(reference_root / REGIONS_NAME),
            read_jsonl(candidate_root / REGIONS_NAME),
        ),
        "recognition": compare_requests(
            read_jsonl(reference_root / TRACE_NAME),
            read_jsonl(candidate_root / TRACE_NAME),
            worst_limit=args.worst_limit,
        ),
        "page_outputs": compare_predictions(reference_root, candidate_root),
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown = render_markdown(report)
    (output_dir / "comparison.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"json={output_dir / 'comparison.json'}")
    print(f"markdown={output_dir / 'comparison.md'}")


if __name__ == "__main__":
    main()
