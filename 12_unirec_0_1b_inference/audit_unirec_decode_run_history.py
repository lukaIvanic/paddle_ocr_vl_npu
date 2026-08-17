#!/usr/bin/env python3
"""Audit saved UniRec production runs without using an NPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--min-pages",
        type=int,
        default=512,
        help="Only inspect recognition traces for runs with at least this many pages.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return {"_read_error": f"{type(error).__name__}: {error}"}
    return value if isinstance(value, dict) else {"_read_error": "not an object"}


def percentile(values: list[int], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return float(ordered[lower] * (1 - fraction) + ordered[upper] * fraction)


def distribution(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "sum": int(sum(values)),
        "min": int(min(values)),
        "mean": float(sum(values) / len(values)),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": int(max(values)),
    }


def max_token_run(tokens: list[int]) -> int:
    if not tokens:
        return 0
    maximum = current = 1
    for previous, value in zip(tokens, tokens[1:]):
        current = current + 1 if value == previous else 1
        maximum = max(maximum, current)
    return maximum


def trace_stats(path: Path, max_length: int) -> dict[str, Any]:
    lengths: list[int] = []
    max_runs: list[int] = []
    cap_ids: list[str] = []
    repeated_cap_ids: list[str] = []
    malformed = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            tokens_raw = row.get("token_ids")
            if isinstance(tokens_raw, list):
                tokens = [int(value) for value in tokens_raw]
                length = int(row.get("decode_token_count", max(0, len(tokens) - 1)))
                run = max_token_run(tokens)
            else:
                tokens = []
                length = int(
                    row.get(
                        "decode_token_count",
                        row.get("generated_token_count", row.get("token_count", 0)),
                    )
                )
                run = 0
            lengths.append(length)
            max_runs.append(run)
            request_id = str(row.get("request_id", ""))
            if length >= max_length - 1:
                cap_ids.append(request_id)
                if run >= max_length - 2:
                    repeated_cap_ids.append(request_id)
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "malformed_rows": malformed,
        "generated_length": distribution(lengths),
        "max_same_token_run": distribution(max_runs),
        "length_cap_count": len(cap_ids),
        "single_token_repeated_to_cap_count": len(repeated_cap_ids),
        "first_length_cap_request_ids": cap_ids[:20],
        "first_repeated_cap_request_ids": repeated_cap_ids[:20],
    }


def find_run_root(summary: Path, search_root: Path) -> Path:
    current = summary.parent
    best = current
    while current == search_root or search_root in current.parents:
        if any(
            (current / name).exists()
            for name in (
                "preflight.log",
                "command.sh",
                "run.log",
                "exit_code.txt",
                "final_report.txt",
                "inference_process_wall_s.txt",
            )
        ):
            best = current
        if current == search_root:
            break
        current = current.parent
    return best


def first_existing(paths: Iterable[Path]) -> Path | None:
    return next((path for path in paths if path.is_file()), None)


def relevant_preflight_lines(path: Path | None) -> list[str]:
    if path is None:
        return []
    pattern = re.compile(
        r"commit|revision|torch|cann|ascend|npu|python|cpu|affinity|taskset|shm|host",
        re.IGNORECASE,
    )
    try:
        return [line for line in path.read_text(errors="replace").splitlines() if pattern.search(line)][:100]
    except OSError:
        return []


def commit_from_text(*texts: str) -> str | None:
    for text in texts:
        match = re.search(r"\b[0-9a-f]{40}\b", text)
        if match:
            return match.group(0)
    return None


def command_evidence(run_root: Path) -> dict[str, Any]:
    path = first_existing(
        (
            run_root / "command.sh",
            run_root / "command.txt",
            run_root / "run_command.sh",
        )
    )
    if path is None:
        return {"path": None, "text": None}
    text = path.read_text(errors="replace")
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "text": text,
        "taskset_mentions": re.findall(r"taskset\s+-c\s+([^\s\\]+)", text),
    }


def locate_trace(summary: Path, run_root: Path) -> Path | None:
    direct = first_existing(
        (
            summary.parent / "recognition_trace.jsonl",
            run_root / "output" / "recognition_trace.jsonl",
            run_root / "recognition_trace.jsonl",
        )
    )
    if direct is not None:
        return direct
    matches = list(run_root.glob("**/recognition_trace.jsonl"))
    return matches[0] if len(matches) == 1 else None


def identify_chip(preflight_lines: list[str], run_root: Path) -> str:
    text = "\n".join(preflight_lines) + "\n" + str(run_root)
    if re.search(r"310P", text, re.IGNORECASE):
        return "Ascend310P"
    if re.search(r"910B", text, re.IGNORECASE):
        return "Ascend910B"
    return "unknown"


def safe_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def audit_summary(path: Path, search_root: Path, min_pages: int) -> dict[str, Any]:
    payload = read_json(path)
    run_root = find_run_root(path, search_root)
    preflight = first_existing((run_root / "preflight.log", run_root / "environment.txt"))
    preflight_lines = relevant_preflight_lines(preflight)
    command = command_evidence(run_root)
    decode = payload.get("decode") or {}
    timing = payload.get("timing_s") or {}
    throughput = payload.get("throughput") or {}
    config = payload.get("config") or payload
    pages = int(payload.get("page_count", payload.get("workload", {}).get("page_count", 0)) or 0)
    crops = int(payload.get("crop_count", payload.get("workload", {}).get("selected_crops", decode.get("submitted", 0))) or 0)
    batch = int(config.get("decode_batch_size", config.get("batch_size", decode.get("batch_size", 0))) or 0)
    self_kv = int(config.get("self_cache_length", 0) or 0)
    cross_kv = int(config.get("cross_cache_length", 0) or 0)
    max_length = int(config.get("max_length", self_kv) or self_kv or 0)
    decode_s = safe_float(decode.get("decode_s", timing.get("decode_graph")))
    iterations = int(decode.get("decode_iterations", 0) or 0)
    raw_slots = int(decode.get("raw_decode_token_slots", 0) or 0)
    effective_tokens = int(decode.get("effective_decode_tokens", 0) or 0)
    raw_tok_s = safe_float(
        decode.get("raw_decode_tokens_per_s", throughput.get("decode_raw_token_slots_per_s"))
    )
    effective_tok_s = safe_float(
        decode.get("effective_decode_tokens_per_s", throughput.get("decode_effective_tokens_per_s"))
    )
    trace = locate_trace(path, run_root) if pages >= min_pages else None
    trace_summary = (
        trace_stats(trace, max_length)
        if trace is not None and max_length > 0
        else None
    )
    preflight_text = "\n".join(preflight_lines)
    command_text = str(command.get("text") or "")
    result = {
        "summary_path": str(path.resolve()),
        "run_root": str(run_root.resolve()),
        "mtime": path.stat().st_mtime,
        "mtime_iso_utc": datetime.fromtimestamp(
            path.stat().st_mtime, tz=timezone.utc
        ).isoformat(),
        "status": payload.get("status"),
        "kind": payload.get("kind", "full_run_summary"),
        "chip": identify_chip(preflight_lines, run_root),
        "project_commit": payload.get("project_commit") or commit_from_text(preflight_text, command_text),
        "preflight_path": str(preflight.resolve()) if preflight is not None else None,
        "preflight_relevant_lines": preflight_lines,
        "command": command,
        "config": {
            "pages": pages,
            "crops": crops,
            "decode_batch_size": batch,
            "self_cache_length": self_kv,
            "cross_cache_length": cross_kv,
            "max_length": max_length,
            "workers": config.get("workers"),
            "recognition_preprocess_threads": config.get("recognition_preprocess_threads"),
            "layout_cpu_threads": config.get("layout_cpu_threads"),
            "layout_batch_size": config.get("layout_batch_size"),
            "decode_admission_prefetch_depth": config.get("decode_admission_prefetch_depth"),
            "decode_graph_warmup": config.get("decode_graph_warmup", payload.get("decode_graph_warmup")),
            "decode_mode": config.get("decode_mode", payload.get("decode_mode")),
        },
        "decode": {
            "iterations": iterations,
            "decode_graph_s": decode_s,
            "mean_step_ms": (1000.0 * decode_s / iterations) if decode_s and iterations else None,
            "raw_token_slots": raw_slots,
            "effective_tokens": effective_tokens,
            "raw_tok_s": raw_tok_s,
            "effective_tok_s": effective_tok_s,
            "slot_efficiency": (effective_tokens / raw_slots) if raw_slots else None,
            "decode_wall_including_ingress_s": timing.get("decode_inference_including_ingress", payload.get("decode_wall_excluding_warmup_s")),
            "timing_detail": decode.get("timing_detail"),
        },
        "throughput": throughput,
        "recognition_trace": trace_summary,
        "evaluation_summaries": [
            str(candidate.resolve())
            for candidate in run_root.glob("**/full_eval_summary.json")
        ],
    }
    return result


def compact_row(row: dict[str, Any]) -> str:
    config = row["config"]
    decode = row["decode"]
    trace = row.get("recognition_trace") or {}
    lengths = trace.get("generated_length") or {}
    return (
        "UNIREC_DECODE_HISTORY_ROW "
        f"chip={row['chip']} pages={config['pages']} crops={config['crops']} "
        f"b={config['decode_batch_size']} self={config['self_cache_length']} "
        f"cross={config['cross_cache_length']} workers={config['workers']} "
        f"threads={config['recognition_preprocess_threads']} "
        f"iterations={decode['iterations']} graph_s={decode['decode_graph_s']} "
        f"step_ms={decode['mean_step_ms']} raw_tok_s={decode['raw_tok_s']} "
        f"effective_tok_s={decode['effective_tok_s']} slot_eff={decode['slot_efficiency']} "
        f"prefill_pg_s={row['throughput'].get('prefill_pages_per_s')} "
        f"pipeline_pg_s={row['throughput'].get('sequential_core_pages_per_s')} "
        f"generated_sum={lengths.get('sum')} generated_mean={lengths.get('mean')} "
        f"caps={trace.get('length_cap_count')} repeated_caps={trace.get('single_token_repeated_to_cap_count')} "
        f"commit={row['project_commit']} root={row['run_root']}"
    )


def main() -> None:
    args = parse_args()
    search_root = args.search_root.expanduser().resolve()
    candidates = sorted(
        {
            *search_root.glob("**/run_summary.json"),
            *search_root.glob("**/clean.json"),
        }
    )
    rows = [audit_summary(path, search_root, args.min_pages) for path in candidates]
    rows = [
        row
        for row in rows
        if row["config"]["pages"] > 0 or row["config"]["crops"] > 0
    ]
    rows.sort(key=lambda row: row["mtime"])
    report = {
        "schema": "unirec_decode_run_history_v1",
        "search_root": str(search_root),
        "summary_files_scanned": len(candidates),
        "qualifying_runs": len(rows),
        "runs": rows,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        "UNIREC_DECODE_HISTORY_AUDIT: PASS "
        f"scanned={len(candidates)} qualifying={len(rows)} output={output}",
        flush=True,
    )
    for row in rows:
        print(compact_row(row), flush=True)


if __name__ == "__main__":
    main()
