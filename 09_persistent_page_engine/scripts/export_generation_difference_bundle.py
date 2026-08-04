#!/usr/bin/env python3
"""Export frozen Experiment-09 and OmniDocBench evidence into one compact bundle.

The bundle is deliberately inference-free.  It preserves the raw recognizer
text/token IDs, the evaluator's raw and normalized sample records, official
aggregate output, and (when supplied) the corrected process-isolated TEDS
scores.  ``generation_difference_atlas.py`` reads the ZIP directly; no
extraction or server-specific path layout is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
EVAL_KINDS = ("text_block", "display_formula", "table", "reading_order")
TRACE_FIELDS = (
    "request_id",
    "source_image_name",
    "page_input_index",
    "block_index",
    "global_request_index",
    "label",
    "prompt",
    "input_tokens",
    "projected_image_tokens",
    "crop_size",
    "min_pixels",
    "max_pixels",
    "text",
    "token_ids",
    "generated_tokens_including_eos",
    "decode_tokens_after_prefill_including_eos",
    "stop_reason",
    "input_fingerprints",
    "vision",
    "text_prefill",
    "decode_slot_index",
    "decode_slot_epoch",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e2e-output", required=True, type=Path)
    parser.add_argument("--eval-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--project-commit", required=True)
    parser.add_argument("--evaluator-commit", required=True)
    parser.add_argument("--table-scores", type=Path)
    parser.add_argument("--teds-summary", type=Path)
    parser.add_argument(
        "--official-score-summary",
        type=Path,
        help="Optional page-CDM/page-TEDS Overall authority to embed.",
    )
    parser.add_argument("--require-corrected-teds", action="store_true")
    parser.add_argument(
        "--include-predictions",
        action="store_true",
        help="Embed the final top-level Markdown prediction set in the bundle.",
    )
    parser.add_argument("--expected-pages", type=int)
    parser.add_argument("--expected-requests", type=int)
    parser.add_argument("--expected-table-requests", type=int)
    parser.add_argument("--expected-text-rows", type=int)
    parser.add_argument("--expected-formula-rows", type=int)
    parser.add_argument("--expected-table-rows", type=int)
    parser.add_argument("--expected-reading-order-rows", type=int)
    parser.add_argument("--expected-table-pages", type=int)
    args = parser.parse_args()
    if bool(args.table_scores) != bool(args.teds_summary):
        parser.error("--table-scores and --teds-summary must be supplied together")
    if args.require_corrected_teds and not args.table_scores:
        parser.error("--require-corrected-teds requires both corrected TEDS sidecars")
    return args


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {pattern} under {root}, got {matches}")
    return matches[0]


def _trace_rows(path: Path) -> list[dict[str, Any]]:
    return [
        {field: row[field] for field in TRACE_FIELDS if field in row}
        for row in _read_jsonl(path)
    ]


def _page_name(row: dict[str, Any]) -> str:
    return str(row["source_image_name"])


def _table_score_key(sample: dict[str, Any]) -> str:
    return str(sample["img_id"]) + "_" + str(sample.get("gt_idx"))


def _finite_number(value: Any, context: str) -> float:
    number = float(value)
    if not (-float("inf") < number < float("inf")):
        raise ValueError(f"non-finite value for {context}: {value!r}")
    return number


def _validate_corrected_teds(
    table_rows: list[dict[str, Any]],
    scores: dict[str, dict[str, Any]],
    summary: dict[str, Any],
    table_result_path: Path,
) -> None:
    source_table_result = summary.get("source_table_result")
    if not source_table_result:
        raise ValueError("TEDS summary does not identify its source_table_result")
    source_table_path = Path(str(source_table_result)).expanduser().resolve()
    if not source_table_path.is_file():
        raise FileNotFoundError(
            f"TEDS summary source_table_result does not exist: {source_table_path}"
        )
    if _sha256(source_table_path) != _sha256(table_result_path):
        raise ValueError(
            "corrected TEDS sidecars are not bound to the selected frozen table result"
        )
    expected_keys = {_table_score_key(row) for row in table_rows}
    actual_keys = set(scores)
    if actual_keys != expected_keys:
        raise ValueError(
            "corrected TEDS key coverage mismatch: "
            f"missing={len(expected_keys - actual_keys)} extra={len(actual_keys - expected_keys)}"
        )
    for key, payload in scores.items():
        for metric in ("TEDS", "TEDS_structure_only"):
            if metric not in payload:
                raise ValueError(f"corrected score {key} is missing {metric}")
            _finite_number(payload[metric], f"{key}.{metric}")
    pages = {str(row.get("img_id") or row.get("image_name")) for row in table_rows}
    if int(summary.get("sample_count", -1)) != len(table_rows):
        raise ValueError("TEDS summary sample_count does not match table rows")
    if int(summary.get("page_count", -1)) != len(pages):
        raise ValueError("TEDS summary page_count does not match table page universe")
    execution = summary.get("execution") or {}
    if execution.get("scheduler") != "process_isolated_parent_timeout":
        raise ValueError("corrected TEDS summary did not use the process-isolated scheduler")
    if int(execution.get("sample_count", -1)) != len(table_rows):
        raise ValueError("corrected TEDS execution sample_count mismatch")
    if int(execution.get("error_case_count", -1)) != 0:
        raise ValueError("corrected TEDS summary still contains metric execution errors")
    for metric in ("TEDS", "TEDS_structure_only"):
        _finite_number(
            summary["sample_aggregate"][metric]["all"],
            f"summary.sample_aggregate.{metric}",
        )
        _finite_number(
            summary["page_aggregate"][metric]["ALL"],
            f"summary.page_aggregate.{metric}",
        )


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(
        _json_bytes(row)
        for row in rows
    )


def _write_member(archive: zipfile.ZipFile, name: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    archive.writestr(info, payload, compresslevel=9)


def _prediction_markdown_files(output_root: Path) -> list[Path]:
    prediction_root = output_root / "predictions"
    if not prediction_root.is_dir():
        raise FileNotFoundError(prediction_root)
    files = sorted(prediction_root.glob("*.md"), key=lambda path: path.name)
    if not files:
        raise ValueError(f"no top-level Markdown predictions under {prediction_root}")
    return files


def _assert_expected(name: str, actual: int, expected: int | None) -> None:
    if expected is not None and actual != expected:
        raise ValueError(f"expected {expected} {name}, got {actual}")


def main() -> None:
    args = _parse_args()
    output_root = args.e2e_output.expanduser().resolve()
    eval_root = args.eval_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)

    trace_path = output_root / "recognition_trace.jsonl"
    summary_path = output_root / "run_summary.json"
    if not trace_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"missing frozen E2E artifacts under {output_root}")
    eval_paths = {
        kind: _find_one(eval_root, f"*_{kind}_result.json")
        for kind in EVAL_KINDS
    }
    metric_path = _find_one(eval_root, "*_metric_result.json")

    print("[bundle] reading frozen recognition trace", flush=True)
    trace = _trace_rows(trace_path)
    run_summary = _read_json(summary_path)
    evaluator = {kind: _read_json(path) for kind, path in eval_paths.items()}
    metric_result = _read_json(metric_path)
    if any(not isinstance(rows, list) for rows in evaluator.values()):
        raise TypeError("all evaluator result artifacts must be JSON arrays")

    prediction_files = (
        _prediction_markdown_files(output_root)
        if args.include_predictions
        else []
    )

    trace_pages = set(map(_page_name, trace))
    page_count = len(trace_pages)
    summary_images = run_summary.get("images")
    summary_page_count = (
        len(summary_images)
        if isinstance(summary_images, list)
        else int(run_summary.get("count", page_count))
    )
    if page_count > summary_page_count:
        raise ValueError("recognition trace contains more pages than run_summary")
    if isinstance(summary_images, list):
        run_page_set = {str(value) for value in summary_images}
        if not trace_pages <= run_page_set:
            raise ValueError("recognition trace contains pages outside run_summary.images")
        for kind, rows in evaluator.items():
            evaluator_pages = {
                str(row.get("img_id") or row.get("image_name") or "")
                for row in rows
            }
            if not evaluator_pages <= run_page_set:
                raise ValueError(f"{kind} evaluator contains pages outside run_summary.images")
    _assert_expected("pages", summary_page_count, args.expected_pages)
    if prediction_files and len(prediction_files) != summary_page_count:
        raise ValueError(
            "Markdown prediction count does not match run_summary: "
            f"{len(prediction_files)} vs {summary_page_count}"
        )
    _assert_expected("requests", len(trace), args.expected_requests)
    table_requests = sum(row.get("label") == "table" for row in trace)
    _assert_expected("table requests", table_requests, args.expected_table_requests)
    row_expectations = {
        "text_block": args.expected_text_rows,
        "display_formula": args.expected_formula_rows,
        "table": args.expected_table_rows,
        "reading_order": args.expected_reading_order_rows,
    }
    for kind, expected in row_expectations.items():
        _assert_expected(f"{kind} rows", len(evaluator[kind]), expected)
    table_pages = {
        str(row.get("img_id") or row.get("image_name"))
        for row in evaluator["table"]
    }
    _assert_expected("table pages", len(table_pages), args.expected_table_pages)

    table_scores = None
    teds_summary = None
    if args.table_scores:
        print("[bundle] validating corrected TEDS sidecars", flush=True)
        table_scores = _read_json(args.table_scores.expanduser().resolve())
        teds_summary = _read_json(args.teds_summary.expanduser().resolve())
        _validate_corrected_teds(
            evaluator["table"],
            table_scores,
            teds_summary,
            eval_paths["table"],
        )
    else:
        frozen_debug = (((metric_result.get("table") or {}).get("metric_debug") or {}).get("TEDS") or {})
        if int(frozen_debug.get("error_case_count", 0)):
            raise ValueError(
                "frozen evaluator TEDS contains metric execution errors; corrected sidecars are required"
            )

    source_paths = {
        "recognition_trace": trace_path,
        "run_summary": summary_path,
        "metric_result": metric_path,
        **{f"evaluator_{kind}": path for kind, path in eval_paths.items()},
    }
    if args.table_scores:
        source_paths["corrected_table_scores"] = args.table_scores.expanduser().resolve()
        source_paths["teds_summary"] = args.teds_summary.expanduser().resolve()
    official_score_summary = None
    if args.official_score_summary:
        score_path = args.official_score_summary.expanduser().resolve()
        official_score_summary = _read_json(score_path)
        source_paths["official_score_summary"] = score_path
    print("[bundle] hashing source artifacts", flush=True)
    hashes = {name: _sha256(path) for name, path in source_paths.items()}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "experiment09_generation_difference_bundle",
        "label": args.label,
        "project_commit": args.project_commit,
        "evaluator_commit": args.evaluator_commit,
        "source_paths": {name: str(path) for name, path in source_paths.items()},
        "source_sha256": hashes,
        "counts": {
            "pages_in_run_summary": summary_page_count,
            "pages_with_recognition_requests": page_count,
            "recognition_requests": len(trace),
            "table_recognition_requests": table_requests,
            "evaluator_rows": {kind: len(rows) for kind, rows in evaluator.items()},
            "table_pages": len(table_pages),
            "prediction_markdown_files": len(prediction_files),
        },
        "prediction_markdown_sha256": {
            path.name: _sha256(path) for path in prediction_files
        },
        "teds_authority": "corrected_process_isolated" if table_scores else "frozen_evaluator",
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=output.name + ".",
        suffix=".tmp",
        dir=output.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        print("[bundle] writing compact deterministic ZIP", flush=True)
        with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
            _write_member(archive, "manifest.json", _json_bytes(manifest))
            _write_member(archive, "recognition_trace.jsonl", _jsonl_bytes(trace))
            _write_member(archive, "run_summary.json", _json_bytes(run_summary))
            _write_member(archive, "metric_result.json", _json_bytes(metric_result))
            for kind, rows in evaluator.items():
                _write_member(archive, f"evaluator/{kind}.json", _json_bytes(rows))
            if table_scores is not None:
                _write_member(archive, "corrected_table_scores.json", _json_bytes(table_scores))
                _write_member(archive, "teds_summary.json", _json_bytes(teds_summary))
            if official_score_summary is not None:
                _write_member(
                    archive,
                    "official_score_summary.json",
                    _json_bytes(official_score_summary),
                )
            for path in prediction_files:
                _write_member(archive, f"predictions/{path.name}", path.read_bytes())
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()

    result = {
        "bundle": str(output),
        "sha256": _sha256(output),
        "bytes": output.stat().st_size,
        "manifest": manifest,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print("[bundle] PASS", flush=True)


if __name__ == "__main__":
    main()
