#!/usr/bin/env python3
"""Compare complete representative-128 UniRec prefill traces across chips."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


TRACE_SCHEMA = "unirec_production_prefill_trace_v1"
RUN_CONTRACT_KEYS = (
    "execution",
    "page_count",
    "workers",
    "recognition_preprocess_threads",
    "layout_batch_size",
    "layout_execution",
    "layout_dtype",
    "layout_reading_order_dtype",
    "layout_threshold",
    "vision_page_lookahead",
    "vision_focal_depthwise_rewrite",
    "vision_weight_format",
    "cross_cache_length",
    "self_cache_length",
)
RETAINED_BANK_KEYS = (
    "crop_count",
    "cross_kv_bytes",
    "rejected_crop_count",
    "real_source_tokens",
    "physical_source_tokens",
    "shared_payload_bytes",
    "storage",
    "retained_images",
    "disk_bytes",
)
TIMING_FIELDS = (
    "sum_s",
    "mean_ms",
    "min_ms",
    "p50_ms",
    "p75_ms",
    "p90_ms",
    "p95_ms",
    "p99_ms",
    "max_ms",
)
EVENT_TIMING_KEYS = {
    "wall_s",
    "stage_s",
    "host_stage_s",
    "device_stage_s",
    "initial_sync_s",
    "lookahead_collect_s",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-trace-summary", type=Path, required=True)
    parser.add_argument("--reference-trace-events", type=Path, required=True)
    parser.add_argument("--reference-trace-pages", type=Path, required=True)
    parser.add_argument("--reference-trace-run", type=Path, required=True)
    parser.add_argument("--reference-clean-run", type=Path, required=True)
    parser.add_argument("--candidate-trace-summary", type=Path, required=True)
    parser.add_argument("--candidate-trace-events", type=Path, required=True)
    parser.add_argument("--candidate-trace-pages", type=Path, required=True)
    parser.add_argument("--candidate-trace-run", type=Path, required=True)
    parser.add_argument("--candidate-clean-run", type=Path, required=True)
    parser.add_argument("--reference-chip", default="910B2")
    parser.add_argument("--candidate-chip", default="310P")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _ratio(candidate: float, reference: float) -> float | None:
    if reference == 0.0:
        return None
    return candidate / reference


def _strip_event_timing(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_event_timing(item)
            for key, item in sorted(value.items())
            if key not in EVENT_TIMING_KEYS
            and key not in {"schema", "trace_index"}
        }
    if isinstance(value, list):
        return [_strip_event_timing(item) for item in value]
    return value


def _canonical_digest(rows: list[Any]) -> str:
    canonical = sorted(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in rows
    )
    digest = hashlib.sha256()
    for row in canonical:
        digest.update(row.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _event_workload_digest(events: list[dict[str, Any]]) -> str:
    return _canonical_digest([_strip_event_timing(event) for event in events])


def _page_workload_digest(pages: list[dict[str, Any]]) -> str:
    rows = []
    for page in pages:
        rows.append(
            {
                "page_index": int(page["page_index"]),
                "page_image": Path(page["image_path"]).name,
                "width": int(page["width"]),
                "height": int(page["height"]),
                "crop_count": int(page["crop_count"]),
                "rejected_crop_count": int(page["rejected_crop_count"]),
            }
        )
    return _canonical_digest(rows)


def _contract(run: dict[str, Any]) -> dict[str, Any]:
    return {key: run[key] for key in RUN_CONTRACT_KEYS}


def _retained_bank(run: dict[str, Any]) -> dict[str, Any]:
    bank = run["retained_bank"]
    return {key: bank[key] for key in RETAINED_BANK_KEYS}


def _clean_metrics(run: dict[str, Any]) -> dict[str, Any]:
    wall = float(run["timing_s"]["prefill_phase"])
    return {
        "wall_s": wall,
        "pages_per_s": float(run["throughput"]["prefill_pages_per_s"]),
        "setup_s": float(run["timing_s"]["prefill_worker_setup"]),
        "warmup_s": float(run["timing_s"]["prefill_warmup"]),
        "shutdown_s": float(run["timing_s"]["prefill_worker_shutdown"]),
        "lifecycle_s": float(run["timing_s"]["lifecycle"]),
        "retained_bank": _retained_bank(run),
    }


def _compare_stage_distributions(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    output: dict[str, Any] = {}
    lines = []
    names = sorted(set(reference) | set(candidate))
    for name in names:
        reference_row = reference.get(name)
        candidate_row = candidate.get(name)
        present = reference_row is not None and candidate_row is not None
        count_match = bool(
            present and reference_row.get("count") == candidate_row.get("count")
        )
        row: dict[str, Any] = {
            "present_on_both": present,
            "count_match": count_match,
            "reference": reference_row,
            "candidate": candidate_row,
            "candidate_over_reference": {},
        }
        if present:
            row["candidate_over_reference"] = {
                field: _ratio(
                    float(candidate_row[field]),
                    float(reference_row[field]),
                )
                for field in TIMING_FIELDS
                if field in reference_row and field in candidate_row
            }
        output[name] = row
        if not present:
            lines.append(
                "UNIREC_PREFILL_CROSSCHIP_STAGE "
                f"name={name} present_on_both=false"
            )
            continue
        ratios = row["candidate_over_reference"]
        lines.append(
            "UNIREC_PREFILL_CROSSCHIP_STAGE "
            f"name={name} count={candidate_row['count']} "
            f"count_match={str(count_match).lower()} "
            f"reference_sum_s={reference_row.get('sum_s', 0.0):.9f} "
            f"candidate_sum_s={candidate_row.get('sum_s', 0.0):.9f} "
            f"sum_ratio={_render_ratio(ratios.get('sum_s'))} "
            f"mean_ratio={_render_ratio(ratios.get('mean_ms'))} "
            f"p50_ratio={_render_ratio(ratios.get('p50_ms'))} "
            f"p90_ratio={_render_ratio(ratios.get('p90_ms'))} "
            f"p95_ratio={_render_ratio(ratios.get('p95_ms'))} "
            f"p99_ratio={_render_ratio(ratios.get('p99_ms'))} "
            f"max_ratio={_render_ratio(ratios.get('max_ms'))}"
        )
    return output, lines


def _render_ratio(value: float | None) -> str:
    return "na" if value is None else f"{value:.6f}x"


def main() -> None:
    args = parse_args()
    reference_trace = _load_json(args.reference_trace_summary)
    candidate_trace = _load_json(args.candidate_trace_summary)
    reference_events = _load_jsonl(args.reference_trace_events)
    candidate_events = _load_jsonl(args.candidate_trace_events)
    reference_pages = _load_jsonl(args.reference_trace_pages)
    candidate_pages = _load_jsonl(args.candidate_trace_pages)
    reference_trace_run = _load_json(args.reference_trace_run)
    candidate_trace_run = _load_json(args.candidate_trace_run)
    reference_clean_run = _load_json(args.reference_clean_run)
    candidate_clean_run = _load_json(args.candidate_clean_run)

    for trace in (reference_trace, candidate_trace):
        if trace.get("schema") != TRACE_SCHEMA:
            raise ValueError(f"unsupported trace schema: {trace.get('schema')!r}")
    for run, traced in (
        (reference_trace_run, True),
        (candidate_trace_run, True),
        (reference_clean_run, False),
        (candidate_clean_run, False),
    ):
        if run.get("status") != "ok":
            raise ValueError("cannot compare a failed run")
        if bool(run.get("prefill_trace_enabled")) is not traced:
            raise ValueError("trace-enabled state does not match input role")

    reference_contract = _contract(reference_clean_run)
    candidate_contract = _contract(candidate_clean_run)
    contract_exact = reference_contract == candidate_contract
    reference_trace_contract_match = (
        _contract(reference_trace_run) == reference_contract
    )
    candidate_trace_contract_match = (
        _contract(candidate_trace_run) == candidate_contract
    )
    reference_bank = _retained_bank(reference_clean_run)
    candidate_bank = _retained_bank(candidate_clean_run)
    retained_bank_exact = reference_bank == candidate_bank
    reference_trace_bank_match = (
        _retained_bank(reference_trace_run) == reference_bank
    )
    candidate_trace_bank_match = (
        _retained_bank(candidate_trace_run) == candidate_bank
    )
    event_counts_exact = (
        reference_trace["event_counts"] == candidate_trace["event_counts"]
    )
    shape_histograms_exact = (
        reference_trace["shape_histograms"]
        == candidate_trace["shape_histograms"]
    )
    reference_event_digest = _event_workload_digest(reference_events)
    candidate_event_digest = _event_workload_digest(candidate_events)
    event_workload_exact = reference_event_digest == candidate_event_digest
    reference_page_digest = _page_workload_digest(reference_pages)
    candidate_page_digest = _page_workload_digest(candidate_pages)
    page_workload_exact = reference_page_digest == candidate_page_digest

    stage_comparison, stage_lines = _compare_stage_distributions(
        reference_trace["stage_distributions"],
        candidate_trace["stage_distributions"],
    )
    stage_keys_exact = all(
        row["present_on_both"] and row["count_match"]
        for row in stage_comparison.values()
    )

    shape_comparison = {}
    shape_lines = []
    shape_names = sorted(
        set(reference_trace["shape_histograms"])
        | set(candidate_trace["shape_histograms"])
    )
    for name in shape_names:
        reference_histogram = reference_trace["shape_histograms"].get(name)
        candidate_histogram = candidate_trace["shape_histograms"].get(name)
        exact = reference_histogram == candidate_histogram
        shape_comparison[name] = {
            "exact": exact,
            "reference": reference_histogram,
            "candidate": candidate_histogram,
        }
        shape_lines.append(
            "UNIREC_PREFILL_CROSSCHIP_SHAPE "
            f"name={name} exact={str(exact).lower()} "
            f"reference_distinct={len(reference_histogram or {})} "
            f"candidate_distinct={len(candidate_histogram or {})}"
        )

    reference_clean = _clean_metrics(reference_clean_run)
    candidate_clean = _clean_metrics(candidate_clean_run)
    reference_trace_wall = float(reference_trace_run["timing_s"]["prefill_phase"])
    candidate_trace_wall = float(candidate_trace_run["timing_s"]["prefill_phase"])
    clean_wall_ratio = candidate_clean["wall_s"] / reference_clean["wall_s"]
    clean_pages_ratio = (
        candidate_clean["pages_per_s"] / reference_clean["pages_per_s"]
    )
    reference_overhead = reference_trace_wall / reference_clean["wall_s"] - 1.0
    candidate_overhead = candidate_trace_wall / candidate_clean["wall_s"] - 1.0

    all_required_checks_passed = all(
        (
            contract_exact,
            reference_trace_contract_match,
            candidate_trace_contract_match,
            retained_bank_exact,
            reference_trace_bank_match,
            candidate_trace_bank_match,
            event_counts_exact,
            shape_histograms_exact,
            event_workload_exact,
            page_workload_exact,
            stage_keys_exact,
        )
    )
    report = {
        "schema": "unirec_representative128_prefill_crosschip_compare_v1",
        "reference_chip": args.reference_chip,
        "candidate_chip": args.candidate_chip,
        "all_required_checks_passed": all_required_checks_passed,
        "contract": {
            "exact": contract_exact,
            "reference_trace_matches_clean": reference_trace_contract_match,
            "candidate_trace_matches_clean": candidate_trace_contract_match,
            "reference": reference_contract,
            "candidate": candidate_contract,
        },
        "workload": {
            "retained_bank_exact": retained_bank_exact,
            "reference_trace_bank_matches_clean": reference_trace_bank_match,
            "candidate_trace_bank_matches_clean": candidate_trace_bank_match,
            "reference_retained_bank": reference_bank,
            "candidate_retained_bank": candidate_bank,
            "event_counts_exact": event_counts_exact,
            "shape_histograms_exact": shape_histograms_exact,
            "event_workload_exact": event_workload_exact,
            "page_workload_exact": page_workload_exact,
            "reference_event_digest": reference_event_digest,
            "candidate_event_digest": candidate_event_digest,
            "reference_page_digest": reference_page_digest,
            "candidate_page_digest": candidate_page_digest,
        },
        "clean": {
            "reference": reference_clean,
            "candidate": candidate_clean,
            "candidate_over_reference_wall": clean_wall_ratio,
            "candidate_over_reference_pages_per_s": clean_pages_ratio,
        },
        "trace": {
            "reference_wall_s": reference_trace_wall,
            "candidate_wall_s": candidate_trace_wall,
            "reference_overhead_fraction": reference_overhead,
            "candidate_overhead_fraction": candidate_overhead,
        },
        "stage_distributions": stage_comparison,
        "shape_histograms": shape_comparison,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    status = "PASS" if all_required_checks_passed else "FAIL"
    print(
        f"UNIREC_PREFILL_CROSSCHIP_CONTRACT {status} "
        f"contract_exact={str(contract_exact).lower()} "
        "reference_trace_contract_match="
        f"{str(reference_trace_contract_match).lower()} "
        "candidate_trace_contract_match="
        f"{str(candidate_trace_contract_match).lower()} "
        f"retained_bank_exact={str(retained_bank_exact).lower()} "
        "reference_trace_bank_match="
        f"{str(reference_trace_bank_match).lower()} "
        "candidate_trace_bank_match="
        f"{str(candidate_trace_bank_match).lower()} "
        f"event_counts_exact={str(event_counts_exact).lower()} "
        f"shape_histograms_exact={str(shape_histograms_exact).lower()} "
        f"event_workload_exact={str(event_workload_exact).lower()} "
        f"page_workload_exact={str(page_workload_exact).lower()} "
        f"stage_keys_exact={str(stage_keys_exact).lower()}"
    )
    print(
        "UNIREC_PREFILL_CROSSCHIP_CLEAN "
        f"reference_chip={args.reference_chip} "
        f"reference_wall_s={reference_clean['wall_s']:.9f} "
        f"reference_pages_s={reference_clean['pages_per_s']:.9f} "
        f"candidate_chip={args.candidate_chip} "
        f"candidate_wall_s={candidate_clean['wall_s']:.9f} "
        f"candidate_pages_s={candidate_clean['pages_per_s']:.9f} "
        f"candidate_over_reference_wall={clean_wall_ratio:.6f}x "
        f"candidate_over_reference_pages_s={clean_pages_ratio:.6f}x"
    )
    print(
        "UNIREC_PREFILL_CROSSCHIP_TRACE "
        f"reference_wall_s={reference_trace_wall:.9f} "
        f"candidate_wall_s={candidate_trace_wall:.9f} "
        f"reference_overhead={reference_overhead * 100.0:.3f}% "
        f"candidate_overhead={candidate_overhead * 100.0:.3f}%"
    )
    for line in stage_lines:
        print(line)
    for line in shape_lines:
        print(line)
    print(f"UNIREC_PREFILL_CROSSCHIP_OUTPUT {args.output.resolve()}")
    if not all_required_checks_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
