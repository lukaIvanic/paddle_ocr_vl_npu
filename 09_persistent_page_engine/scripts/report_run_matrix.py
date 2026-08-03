#!/usr/bin/env python3
"""Compare Experiment 09 production runs in one compact evidence matrix.

Each input may be either a run root (containing ``output/run_summary.json``)
or the output directory itself.  Evaluation artifacts are discovered below the
run root when present; missing evaluations are recorded rather than treated as
an error, so the report can be refreshed while a benchmark matrix is running.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


SUMMARY_NAME = "run_summary.json"
TRACE_NAME = "recognition_trace.jsonl"
METRIC_SUFFIX = "_metric_result.json"
STAGE_SUFFIX = "_stage_execution.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument(
        "--candidate-run",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Candidate name and run/output path; repeat for every lane.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.candidate_run:
        parser.error("at least one --candidate-run NAME=PATH is required")
    return args


def parse_named_path(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise ValueError(
            f"invalid --candidate-run {value!r}; expected NAME=PATH"
        )
    return name.strip(), Path(path.strip())


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def resolve_run(path: Path) -> tuple[Path, Path]:
    path = path.expanduser().resolve()
    direct = path / SUMMARY_NAME
    nested = path / "output" / SUMMARY_NAME
    if direct.is_file():
        return path.parent if path.name == "output" else path, path
    if nested.is_file():
        return path, path / "output"
    raise FileNotFoundError(
        f"could not find {SUMMARY_NAME} under {path} or {path / 'output'}"
    )


def get(payload: Any, *keys: str, default: Any = None) -> Any:
    value = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def safe_div(numerator: Any, denominator: Any) -> float | None:
    numerator = number(numerator)
    denominator = number(denominator)
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def first_artifact(run_root: Path, suffix: str) -> Path | None:
    matches = sorted(
        path
        for path in run_root.rglob(f"*{suffix}")
        if path.is_file()
    )
    if not matches:
        return None
    # A normal run has one result.  Prefer the shallowest deterministic match
    # if a run keeps an old evaluator retry alongside the final result.
    return min(
        matches,
        key=lambda path: (len(path.relative_to(run_root).parts), str(path)),
    )


def broad_label(value: Any) -> str:
    label = str(value or "unknown").strip().lower()
    if label in {"formula", "display_formula", "equation", "equation_interline"}:
        return "formula"
    if label == "table" or "table" in label:
        return "table"
    if label == "text":
        return "text"
    return f"other:{label}"


def trace_label_totals(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {"available": False, "path": str(path) if path else None, "labels": {}}
    totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "requests": 0,
            "real_vision_tokens": 0,
            "physical_vision_tokens": 0,
            "input_text_tokens": 0,
            "generated_tokens_including_eos": 0,
        }
    )
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"expected object at {path}:{line_number}")
            target = totals[broad_label(row.get("label"))]
            vision = row.get("vision") or {}
            target["requests"] += 1
            target["real_vision_tokens"] += int(
                vision.get("real_vision_tokens") or 0
            )
            target["physical_vision_tokens"] += int(
                vision.get("physical_vision_tokens") or 0
            )
            target["input_text_tokens"] += int(row.get("input_tokens") or 0)
            target["generated_tokens_including_eos"] += int(
                row.get("generated_tokens_including_eos") or 0
            )
    labels = {name: totals[name] for name in sorted(totals)}
    return {"available": True, "path": str(path), "labels": labels}


def evaluation_metrics(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"available": False, "path": None, "metrics": {}}
    payload = read_json(path)
    metrics = {
        "text_block_edit_distance": number(
            get(payload, "text_block", "all", "Edit_dist", "ALL_page_avg")
        ),
        "display_formula_edit_distance": number(
            get(payload, "display_formula", "all", "Edit_dist", "ALL_page_avg")
        ),
        "table_edit_distance": number(
            get(payload, "table", "all", "Edit_dist", "ALL_page_avg")
        ),
        # The official notebook summary uses page.TEDS.ALL, not table.all.TEDS.all.
        "table_teds": number(get(payload, "table", "page", "TEDS", "ALL")),
        "table_teds_structure_only": number(
            get(payload, "table", "page", "TEDS_structure_only", "ALL")
        ),
        "reading_order_edit_distance": number(
            get(payload, "reading_order", "all", "Edit_dist", "ALL_page_avg")
        ),
    }
    return {"available": True, "path": str(path), "metrics": metrics}


def evaluation_stage(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"available": False, "path": None}
    payload = read_json(path)
    page_match = get(payload, "page_match", default={}) or {}
    teds = get(payload, "metrics", "table", "TEDS", default={}) or {}
    fallbacks = page_match.get("fallbacks") or {}
    return {
        "available": True,
        "path": str(path),
        "page_count": number(page_match.get("page_count")),
        "quick_match_timeout_count": number(
            get(fallbacks, "quick_match_timeout", "count")
        ),
        "page_timeout_count": number(get(fallbacks, "page_timeout", "count")),
        "teds_sample_count": number(teds.get("sample_count")),
        "teds_timeout_count": number(teds.get("timeout_case_count")),
        "teds_error_count": number(teds.get("error_case_count")),
        "teds_exception_count": number(teds.get("exception_case_count")),
    }


def selected_configuration(summary: dict[str, Any]) -> dict[str, Any]:
    configuration = summary.get("configuration") or {}
    fields = (
        "batch_size",
        "cache_length",
        "dtype",
        "decode_backend",
        "decode_optimization",
        "preprocessor_min_pixels",
        "preprocessor_max_pixels",
        "effective_global_min_pixels",
        "effective_global_max_pixels",
        "text_preprocessor_max_pixels",
        "text_crop_scale",
        "vision_backend",
        "vision_attention",
        "vision_packing",
        "vision_pack_target",
        "vision_router_lookahead",
        "vision_buckets",
        "text_packing",
        "text_pack_buckets",
        "page_preprocessing_mode",
    )
    return {field: configuration.get(field) for field in fields if field in configuration}


def collect_run(name: str, supplied_path: Path) -> dict[str, Any]:
    run_root, output_root = resolve_run(supplied_path)
    summary_path = output_root / SUMMARY_NAME
    summary = read_json(summary_path)
    recognition = summary.get("recognition") or {}
    device = recognition.get("device_stage_s") or {}
    layout_stage = get(summary, "layout_frontend", "stage_s", default={}) or {}
    vision_packing = recognition.get("vision_packing") or {}
    text_packing = recognition.get("text_packing") or {}
    trace_path = output_root / TRACE_NAME
    metric_path = first_artifact(run_root, METRIC_SUFFIX)
    stage_path = first_artifact(run_root, STAGE_SUFFIX)
    cache_manifest_path = run_root / "cache_manifest_exact.txt"
    cache_manifest_exact = (
        cache_manifest_path.read_text(encoding="utf-8").strip() == "YES"
        if cache_manifest_path.is_file()
        else None
    )
    pipeline = {
        "pages": number(summary.get("count")),
        "results": number(summary.get("result_count")),
        "pipeline_e2e_s": number(summary.get("pipeline_e2e_s")),
        "pages_per_s": number(summary.get("pages_per_s")),
        "s_per_page": number(summary.get("s_per_page")),
        "setup_s": number(summary.get("setup_s")),
        "layout_page_total_s": number(layout_stage.get("page_total_s")),
        "recognition_wall_s": number(recognition.get("wall_s")),
        "vision_prefill_s": number(device.get("vision_prefill")),
        "text_prefill_s": number(device.get("text_prefill")),
        "decode_wall_s": number(recognition.get("decode_wall_s")),
    }
    tokens = {
        key: number(recognition.get(key))
        for key in (
            "requests",
            "input_tokens",
            "real_vision_tokens",
            "physical_vision_tokens",
            "real_text_tokens",
            "physical_text_tokens",
            "generated_tokens_including_eos",
            "effective_decode_tokens",
            "raw_decode_token_slots",
            "active_decode_token_slots",
            "idle_decode_token_slots",
        )
    }
    throughput = {
        "real_vision_tokens_per_s": safe_div(
            tokens["real_vision_tokens"], pipeline["vision_prefill_s"]
        ),
        "physical_vision_tokens_per_s": safe_div(
            tokens["physical_vision_tokens"], pipeline["vision_prefill_s"]
        ),
        "real_text_tokens_per_s": safe_div(
            tokens["real_text_tokens"], pipeline["text_prefill_s"]
        ),
        "physical_text_tokens_per_s": safe_div(
            tokens["physical_text_tokens"], pipeline["text_prefill_s"]
        ),
        "effective_decode_tokens_per_s": safe_div(
            tokens["effective_decode_tokens"], pipeline["decode_wall_s"]
        ),
        "raw_decode_tokens_per_s": safe_div(
            tokens["raw_decode_token_slots"], pipeline["decode_wall_s"]
        ),
    }
    label_tokens = trace_label_totals(trace_path)
    trace_totals = {
        key: sum(label.get(key, 0) for label in label_tokens["labels"].values())
        for key in ("requests", "real_vision_tokens", "physical_vision_tokens")
    }
    label_tokens["summary_cross_check"] = {
        "trace_totals": trace_totals,
        "matches_summary": {
            "requests": trace_totals["requests"] == tokens["requests"],
            "real_vision_tokens": (
                trace_totals["real_vision_tokens"] == tokens["real_vision_tokens"]
            ),
            "physical_vision_tokens": (
                trace_totals["physical_vision_tokens"]
                == tokens["physical_vision_tokens"]
            ),
        },
    }
    return {
        "name": name,
        "run_root": str(run_root),
        "output_root": str(output_root),
        "artifacts": {
            "run_summary": str(summary_path),
            "recognition_trace": str(trace_path) if trace_path.is_file() else None,
            "evaluation_metrics": str(metric_path) if metric_path else None,
            "evaluation_stage": str(stage_path) if stage_path else None,
            "cache_manifest_exact": (
                str(cache_manifest_path)
                if cache_manifest_path.is_file()
                else None
            ),
        },
        "cache_manifest_exact": cache_manifest_exact,
        "configuration": selected_configuration(summary),
        "pipeline": pipeline,
        "stages": {
            "layout_frontend_s": {
                key: number(value) for key, value in sorted(layout_stage.items())
            },
            "recognition_device_s": {
                key: number(value) for key, value in sorted(device.items())
            },
        },
        "tokens": tokens,
        "throughput": throughput,
        "packing": {
            "vision": {
                key: vision_packing.get(key)
                for key in (
                    "mode",
                    "target",
                    "lookahead",
                    "groups",
                    "crops",
                    "packed_groups",
                    "singleton_groups",
                    "eager_overflow_groups",
                    "crops_per_group",
                    "packed_fill_fraction",
                    "graph_shape_histogram",
                )
            },
            "text": {
                key: text_packing.get(key)
                for key in (
                    "mode",
                    "groups",
                    "packs",
                    "packed_crops",
                    "fallback_crops",
                    "packed_fill_fraction",
                    "bucket_histogram",
                )
            },
        },
        "label_tokens": label_tokens,
        "evaluation": evaluation_metrics(metric_path),
        "evaluation_stage": evaluation_stage(stage_path),
    }


def delta(candidate: Any, reference: Any) -> float | int | None:
    candidate = number(candidate)
    reference = number(reference)
    if candidate is None or reference is None:
        return None
    return candidate - reference


def scalar_deltas(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    sections = ("pipeline", "tokens", "throughput")
    result: dict[str, Any] = {}
    for section in sections:
        result[section] = {
            key: delta(value, reference[section].get(key))
            for key, value in candidate[section].items()
        }
    result["evaluation"] = {
        key: delta(value, reference["evaluation"]["metrics"].get(key))
        for key, value in candidate["evaluation"]["metrics"].items()
    }
    result["stages"] = {}
    for stage_family in ("layout_frontend_s", "recognition_device_s"):
        left = reference["stages"][stage_family]
        right = candidate["stages"][stage_family]
        result["stages"][stage_family] = {
            key: delta(right.get(key), left.get(key))
            for key in sorted(set(left) | set(right))
        }
    labels = set(reference["label_tokens"]["labels"]) | set(
        candidate["label_tokens"]["labels"]
    )
    result["label_tokens"] = {}
    for label in sorted(labels):
        left = reference["label_tokens"]["labels"].get(label, {})
        right = candidate["label_tokens"]["labels"].get(label, {})
        result["label_tokens"][label] = {
            key: delta(right.get(key), left.get(key))
            for key in ("requests", "real_vision_tokens", "physical_vision_tokens")
        }
    return result


def comparability(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, bool]:
    return {
        "same_page_count": (
            candidate["pipeline"]["pages"] == reference["pipeline"]["pages"]
        ),
        "same_result_count": (
            candidate["pipeline"]["results"] == reference["pipeline"]["results"]
        ),
        "same_request_count": (
            candidate["tokens"]["requests"] == reference["tokens"]["requests"]
        ),
    }


def format_number(value: Any, *, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    return str(value)


def value_delta(value: Any, change: Any, *, digits: int = 3) -> str:
    rendered = format_number(value, digits=digits)
    if change is None:
        return rendered
    sign = "+" if change >= 0 else ""
    return f"{rendered} ({sign}{format_number(change, digits=digits)})"


def markdown_table(headers: list[str], rows: Iterable[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def render_markdown(report: dict[str, Any]) -> str:
    runs = report["runs"]
    reference_name = report["reference"]
    reference = runs[reference_name]

    def show(run: dict[str, Any], section: str, key: str, digits: int = 3) -> str:
        change = None if run["name"] == reference_name else run["deltas"][section][key]
        return value_delta(run[section].get(key), change, digits=digits)

    lines = [
        "# Experiment 09 run matrix",
        "",
        f"Reference: `{reference_name}`. Parenthesized values are signed candidate minus reference deltas.",
        "",
        "## Configuration",
        "",
    ]
    lines += markdown_table(
        [
            "run",
            "global max pixels",
            "text max pixels",
            "text crop scale",
            "vision pack target",
            "page mode",
        ],
        [
            [
                name,
                format_number(
                    run["configuration"].get(
                        "effective_global_max_pixels",
                        run["configuration"].get("preprocessor_max_pixels"),
                    ),
                    digits=0,
                ),
                format_number(
                    run["configuration"].get("text_preprocessor_max_pixels"),
                    digits=0,
                ),
                format_number(run["configuration"].get("text_crop_scale")),
                format_number(
                    run["configuration"].get("vision_pack_target"), digits=0
                ),
                str(run["configuration"].get("page_preprocessing_mode", "n/a")),
            ]
            for name, run in runs.items()
        ],
    )
    lines += [
        "",
        "## Pipeline and stage timing",
        "",
    ]
    lines += markdown_table(
        ["run", "pages/s", "e2e s", "layout s", "vision s", "text s", "decode s"],
        [
            [
                name,
                show(run, "pipeline", "pages_per_s"),
                show(run, "pipeline", "pipeline_e2e_s", 2),
                show(run, "pipeline", "layout_page_total_s", 2),
                show(run, "pipeline", "vision_prefill_s", 2),
                show(run, "pipeline", "text_prefill_s", 2),
                show(run, "pipeline", "decode_wall_s", 2),
            ]
            for name, run in runs.items()
        ],
    )
    lines += ["", "## Tokens and measured throughput", ""]
    lines += markdown_table(
        [
            "run",
            "real vision",
            "physical vision",
            "vision physical tok/s",
            "physical text",
            "text physical tok/s",
            "decode effective tok/s",
        ],
        [
            [
                name,
                show(run, "tokens", "real_vision_tokens", 0),
                show(run, "tokens", "physical_vision_tokens", 0),
                show(run, "throughput", "physical_vision_tokens_per_s", 0),
                show(run, "tokens", "physical_text_tokens", 0),
                show(run, "throughput", "physical_text_tokens_per_s", 0),
                show(run, "throughput", "effective_decode_tokens_per_s", 0),
            ]
            for name, run in runs.items()
        ],
    )
    lines += ["", "## Vision tokens by recognition label", ""]
    label_rows: list[list[str]] = []
    all_labels = sorted(
        {label for run in runs.values() for label in run["label_tokens"]["labels"]}
    )
    for name, run in runs.items():
        for label in all_labels:
            values = run["label_tokens"]["labels"].get(label, {})
            changes = (run.get("deltas") or {}).get("label_tokens", {}).get(label, {})
            label_rows.append(
                [
                    name,
                    label,
                    value_delta(values.get("requests"), changes.get("requests"), digits=0),
                    value_delta(
                        values.get("real_vision_tokens"),
                        changes.get("real_vision_tokens"),
                        digits=0,
                    ),
                    value_delta(
                        values.get("physical_vision_tokens"),
                        changes.get("physical_vision_tokens"),
                        digits=0,
                    ),
                ]
            )
    lines += markdown_table(
        ["run", "label", "requests", "real vision", "physical vision"],
        label_rows,
    )
    lines += ["", "## Packing", ""]
    lines += markdown_table(
        [
            "run",
            "vision groups",
            "crops/group",
            "vision fill",
            "eager overflow",
            "text packs",
            "text fill",
        ],
        [
            [
                name,
                format_number(get(run, "packing", "vision", "groups"), digits=0),
                format_number(get(run, "packing", "vision", "crops_per_group")),
                format_number(get(run, "packing", "vision", "packed_fill_fraction")),
                format_number(
                    get(run, "packing", "vision", "eager_overflow_groups"),
                    digits=0,
                ),
                format_number(get(run, "packing", "text", "packs"), digits=0),
                format_number(get(run, "packing", "text", "packed_fill_fraction")),
            ]
            for name, run in runs.items()
        ],
    )
    lines += ["", "## Official OmniDocBench metrics", ""]
    metric_keys = (
        ("text_block_edit_distance", "text Edit"),
        ("display_formula_edit_distance", "formula Edit"),
        ("table_edit_distance", "table Edit"),
        ("table_teds", "table TEDS"),
        ("table_teds_structure_only", "table TEDS-structure"),
        ("reading_order_edit_distance", "reading-order Edit"),
    )
    lines += markdown_table(
        ["run", *(title for _, title in metric_keys)],
        [
            [
                name,
                *[
                    value_delta(
                        run["evaluation"]["metrics"].get(key),
                        None
                        if name == reference_name
                        else run["deltas"]["evaluation"].get(key),
                        digits=6,
                    )
                    for key, _ in metric_keys
                ],
            ]
            for name, run in runs.items()
        ],
    )
    lines += ["", "## Artifact status", ""]
    lines += markdown_table(
        [
            "run",
            "same corpus shape",
            "trace",
            "trace totals match summary",
            "evaluation",
            "evaluation stages",
            "cache unchanged",
        ],
        [
            [
                name,
                (
                    "yes"
                    if all(run["comparability"].values())
                    else "no"
                ),
                "yes" if run["label_tokens"]["available"] else "no",
                (
                    "yes"
                    if all(run["label_tokens"]["summary_cross_check"]["matches_summary"].values())
                    else "no"
                ),
                "yes" if run["evaluation"]["available"] else "not present",
                "yes" if run["evaluation_stage"]["available"] else "not present",
                (
                    "yes"
                    if run["cache_manifest_exact"] is True
                    else "no"
                    if run["cache_manifest_exact"] is False
                    else "not recorded"
                ),
            ]
            for name, run in runs.items()
        ],
    )
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    named_candidates = [parse_named_path(value) for value in args.candidate_run]
    names = [name for name, _ in named_candidates]
    if len(names) != len(set(names)) or "reference" in names:
        raise ValueError("candidate names must be unique and may not be 'reference'")
    reference = collect_run("reference", args.reference_run)
    runs = {"reference": reference}
    for name, path in named_candidates:
        runs[name] = collect_run(name, path)
    for name, run in runs.items():
        run["deltas"] = (
            {
                "pipeline": {key: None for key in run["pipeline"]},
                "tokens": {key: None for key in run["tokens"]},
                "throughput": {key: None for key in run["throughput"]},
                "evaluation": {
                    key: None for key in run["evaluation"]["metrics"]
                },
                "stages": {
                    stage_family: {key: None for key in values}
                    for stage_family, values in run["stages"].items()
                },
                "label_tokens": {
                    label: {
                        key: None
                        for key in ("requests", "real_vision_tokens", "physical_vision_tokens")
                    }
                    for label in run["label_tokens"]["labels"]
                },
            }
            if name == "reference"
            else scalar_deltas(run, reference)
        )
        run["comparability"] = comparability(run, reference)
    report = {
        "schema_version": 1,
        "reference": "reference",
        "delta_convention": "candidate_minus_reference",
        "metric_direction": {
            "pages_per_s": "higher_is_better",
            "pipeline_and_stage_seconds": "lower_is_better",
            "edit_distance": "lower_is_better",
            "teds": "higher_is_better",
        },
        "runs": runs,
    }
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "run_matrix.json"
    markdown_path = output_dir / "run_matrix.md"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
