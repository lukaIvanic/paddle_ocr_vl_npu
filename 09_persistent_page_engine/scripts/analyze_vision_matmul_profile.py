#!/usr/bin/env python3
"""Normalize and analyze detailed Ascend vision-prefill profiler captures.

The script is deliberately offline and stdlib-only.  ``kernel_details.csv`` is
the canonical execution-level source because it is the exported table that
associates each kernel launch with its task/stream identity, tensor contract,
and the PMU family selected for that capture.  Profiler databases are
inventoried for discoverability, but are not silently joined to kernel rows.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import datetime as dt
import hashlib
import json
import math
import re
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = 1
ROLES = ("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2")
OUTPUT_FILES = (
    "profile_manifest.json",
    "profile_analysis.json",
    "kernel_executions.csv",
    "vision_linear_executions.csv",
    "vision_layer_summary.csv",
    "profile.sqlite",
    "profile_report.md",
)
CANONICAL_ALIASES = {
    "step_id": ("Step Id", "Step ID", "Step"),
    "device_id": ("Device_id", "Device ID", "Device Id"),
    "model_id": ("Model ID", "Model Id"),
    "task_id": ("Task ID", "Task Id"),
    "stream_id": ("Stream ID", "Stream Id"),
    "name": ("Name", "Kernel Name"),
    "type": ("Type", "Op Type", "OP Type"),
    "op_state": ("OP State", "Op State"),
    "accelerator_core": ("Accelerator Core", "Core Type"),
    "start_us": ("Start Time(us)", "Start Time (us)", "Start(us)"),
    "duration_us": ("Duration(us)", "Duration (us)", "Task Duration(us)"),
    "wait_us": ("Wait Time(us)", "Wait Time (us)", "Wait(us)"),
    "block_dim": ("Block Num", "Block Dim", "Block Number"),
    "mix_block_dim": ("Mix Block Num", "Mix Block Dim", "Mix Block Number"),
    "input_shapes": ("Input Shapes", "Input Shape"),
    "input_dtypes": ("Input Data Types", "Input Dtypes", "Input Data Type"),
    "input_formats": ("Input Formats", "Input Format"),
    "output_shapes": ("Output Shapes", "Output Shape"),
    "output_dtypes": ("Output Data Types", "Output Dtypes", "Output Data Type"),
    "output_formats": ("Output Formats", "Output Format"),
    "context_id": ("Context ID", "Context Id"),
}
TIME_FALLBACKS = {
    "start_us": ("Start Time(ns)", "Start Time (ns)", "Start Time(ms)"),
    "duration_us": ("Duration(ns)", "Duration (ns)", "Duration(ms)"),
    "wait_us": ("Wait Time(ns)", "Wait Time (ns)", "Wait Time(ms)"),
}
PMU_ALIASES = {
    "aicore_time_us": ("aicore_time(us)", "aic_time(us)", "AI Core Time(us)"),
    "mac_time_us": ("aic_mac_time(us)", "mac_time(us)"),
    "mac_ratio": ("aic_mac_ratio", "mac_ratio"),
    "mte1_time_us": ("aic_mte1_time(us)", "mte1_time(us)"),
    "mte1_ratio": ("aic_mte1_ratio", "mte1_ratio"),
    "mte2_time_us": ("aic_mte2_time(us)", "mte2_time(us)"),
    "mte2_ratio": ("aic_mte2_ratio", "mte2_ratio"),
    "mte3_time_us": ("aic_mte3_time(us)", "mte3_time(us)"),
    "mte3_ratio": ("aic_mte3_ratio", "mte3_ratio"),
    "fixpipe_time_us": ("aic_fixpipe_time(us)", "fixpipe_time(us)"),
    "fixpipe_ratio": ("aic_fixpipe_ratio", "fixpipe_ratio"),
    "cube_utilization_pct": ("cube_utilization(%)", "cube_utilization"),
}


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _finite_number(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().rstrip("\t").replace(",", "")
    if not text or text.upper() in {"N/A", "NA", "NULL", "NONE", "-"}:
        return None
    if text.endswith("%"):
        text = text[:-1]
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _integer(value: Any) -> int | None:
    number = _finite_number(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _find_header(headers: Sequence[str], aliases: Sequence[str]) -> str | None:
    by_norm = {_norm(header): header for header in headers}
    for alias in aliases:
        if _norm(alias) in by_norm:
            return by_norm[_norm(alias)]
    return None


def _time_to_us(value: Any, header: str | None) -> float | None:
    number = _finite_number(value)
    if number is None or header is None:
        return None
    lowered = header.lower().replace(" ", "")
    if "(ns)" in lowered:
        return number / 1_000.0
    if "(ms)" in lowered:
        return number * 1_000.0
    if "(s)" in lowered and "(us)" not in lowered:
        return number * 1_000_000.0
    return number


def _shape_groups(value: str | None) -> list[list[int]]:
    if not value:
        return []
    text = value.strip().strip('"').strip("'")
    groups: list[list[int]] = []
    for group in text.split(";"):
        dims = [int(item) for item in re.findall(r"-?\d+", group)]
        if dims:
            groups.append(dims)
    return groups


def _matmul_flops(row: dict[str, Any]) -> int | None:
    inputs = _shape_groups(row.get("input_shapes"))
    outputs = _shape_groups(row.get("output_shapes"))
    if not inputs or not outputs or len(inputs[0]) < 2 or len(outputs[0]) < 2:
        return None
    activation, output = inputs[0], outputs[0]
    m_in = math.prod(activation[:-1])
    m_out = math.prod(output[:-1])
    k, n = activation[-1], output[-1]
    if min(m_in, m_out, k, n) <= 0 or m_in != m_out:
        return None
    return 2 * m_in * k * n


def _is_matmul(row: dict[str, Any]) -> bool:
    return "matmul" in f"{row.get('name', '')} {row.get('type', '')}".lower()


def _stats(values: Iterable[float | None]) -> dict[str, Any]:
    found = [value for value in values if value is not None]
    if not found:
        return {"available": False, "count": 0, "sum": None, "mean": None,
                "min": None, "max": None}
    return {
        "available": True,
        "count": len(found),
        "sum": sum(found),
        "mean": statistics.fmean(found),
        "min": min(found),
        "max": max(found),
    }


def _weighted(values: Iterable[tuple[float | None, float | None]]) -> float | None:
    pairs = [(value, weight) for value, weight in values
             if value is not None and weight is not None and weight > 0]
    total = sum(weight for _, weight in pairs)
    return sum(value * weight for value, weight in pairs) / total if total else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_contract(argument: str | None) -> dict[str, Any] | None:
    if argument is None:
        return None
    candidate = argument[1:] if argument.startswith("@") else argument
    path = Path(candidate)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    parsed = json.loads(argument)
    if not isinstance(parsed, dict):
        raise ValueError("--contract must decode to a JSON object")
    return parsed


def _contract_dims(contract: dict[str, Any] | None) -> dict[str, Any] | None:
    if contract is None:
        return None
    shape = contract.get("shape", contract)
    attention = contract.get("attention", {})
    head_padding = attention.get("head_padding", {})
    dims = {
        "batch_size": shape.get("batch_size"),
        "sequence_length": shape.get("sequence_length"),
        "hidden_size": shape.get("hidden_size"),
        "intermediate_size": shape.get(
            "candidate_intermediate_size", shape.get("intermediate_size")
        ),
        "layers": shape.get("layers", shape.get("num_layers")),
        "linear_calls_per_layer": shape.get("linear_calls_per_layer", 6),
        "linear_calls_per_full_stack": shape.get("linear_calls_per_full_stack"),
        "head_padding_mode": head_padding.get("mode"),
        "real_head_dim": head_padding.get(
            "real_head_dim", attention.get("model_head_dim")
        ),
        "padded_head_dim": head_padding.get(
            "padded_head_dim", attention.get("promptfa_call_head_dim")
        ),
    }
    for key in ("batch_size", "sequence_length", "hidden_size",
                "intermediate_size", "layers", "linear_calls_per_layer"):
        dims[key] = _integer(dims[key])
    if dims["linear_calls_per_full_stack"] is None and dims["layers"]:
        dims["linear_calls_per_full_stack"] = (
            dims["layers"] * dims["linear_calls_per_layer"]
        )
    else:
        dims["linear_calls_per_full_stack"] = _integer(
            dims["linear_calls_per_full_stack"]
        )
    required = ("hidden_size", "intermediate_size", "layers",
                "linear_calls_per_full_stack")
    return dims if all(dims.get(key) for key in required) else None


def _database_inventory(profile_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    inventory: list[dict[str, Any]] = []
    warnings: list[str] = []
    for path in sorted({
        *profile_dir.rglob("*.db"),
        *profile_dir.rglob("*.sqlite"),
        *profile_dir.rglob("*.sqlite3"),
    }):
        entry: dict[str, Any] = {"path": str(path.resolve()), "tables": []}
        try:
            connection = sqlite3.connect(
                f"file:{path.resolve()}?mode=ro", uri=True
            )
            for name, sql in connection.execute(
                "SELECT name, sql FROM sqlite_master "
                "WHERE type='table' ORDER BY name"
            ):
                quoted = '"' + str(name).replace('"', '""') + '"'
                columns = [
                    {
                        "ordinal": column[0],
                        "name": column[1],
                        "type": column[2],
                        "not_null": bool(column[3]),
                        "default": column[4],
                        "primary_key": bool(column[5]),
                    }
                    for column in connection.execute(
                        f"PRAGMA table_info({quoted})"
                    )
                ]
                entry["tables"].append(
                    {"name": name, "create_sql": sql, "columns": columns}
                )
            connection.close()
        except sqlite3.Error as exc:
            entry["error"] = str(exc)
            warnings.append(f"Could not inventory database {path}: {exc}")
        inventory.append(entry)
    return inventory, warnings


def _read_lane(metric: str, profile_dir: Path) -> tuple[
    list[dict[str, Any]], dict[str, Any], list[str]
]:
    if profile_dir.is_file() and profile_dir.name == "kernel_details.csv":
        csv_paths = [profile_dir]
    elif (profile_dir / "kernel_details.csv").is_file():
        csv_paths = [profile_dir / "kernel_details.csv"]
    else:
        csv_paths = sorted(profile_dir.rglob("ASCEND_PROFILER_OUTPUT/kernel_details.csv"))
    if not csv_paths:
        raise FileNotFoundError(
            f"{metric}: no ASCEND_PROFILER_OUTPUT/kernel_details.csv under "
            f"{profile_dir}"
        )
    rows: list[dict[str, Any]] = []
    csv_inventory: list[dict[str, Any]] = []
    warnings: list[str] = []
    execution_id = 0
    for source_index, path in enumerate(csv_paths):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            headers = list(reader.fieldnames or [])
            if not headers:
                warnings.append(f"Empty header in {path}")
                continue
            resolved: dict[str, str | None] = {}
            for canonical, aliases in CANONICAL_ALIASES.items():
                resolved[canonical] = _find_header(headers, aliases)
            for canonical, fallbacks in TIME_FALLBACKS.items():
                if resolved[canonical] is None:
                    resolved[canonical] = _find_header(headers, fallbacks)
            canonical_headers = {header for header in resolved.values() if header}
            dynamic_headers = [h for h in headers if h not in canonical_headers]
            csv_inventory.append(
                {
                    "path": str(path.resolve()),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                    "columns": headers,
                    "canonical_column_resolution": resolved,
                    "dynamic_columns": dynamic_headers,
                }
            )
            for row_index, raw in enumerate(reader):
                normalized: dict[str, Any] = {
                    "execution_id": execution_id,
                    "lane": metric,
                    "metric_family": metric,
                    "source_index": source_index,
                    "source_csv": str(path.resolve()),
                    "source_row": row_index + 2,
                    "replay_id": None,
                    "matmul_ordinal": None,
                    "layer": None,
                    "role": None,
                    "mapping_status": None,
                }
                for field in CANONICAL_ALIASES:
                    header = resolved[field]
                    value = raw.get(header) if header else None
                    if field in {"start_us", "duration_us", "wait_us"}:
                        normalized[field] = _time_to_us(value, header)
                    elif field in {
                        "step_id", "device_id", "model_id", "task_id",
                        "stream_id", "block_dim", "mix_block_dim", "context_id",
                    }:
                        normalized[field] = _integer(value)
                    else:
                        normalized[field] = (
                            str(value).strip().rstrip("\t")
                            if value not in (None, "") else None
                        )
                metrics = {header: raw.get(header) for header in dynamic_headers}
                normalized["metrics"] = metrics
                normalized["metrics_json"] = _json(metrics)
                normalized["raw_json"] = _json(raw)
                normalized["is_matmul"] = _is_matmul(normalized)
                normalized["matmul_flops"] = (
                    _matmul_flops(normalized)
                    if normalized["is_matmul"] else None
                )
                duration = normalized["duration_us"]
                flops = normalized["matmul_flops"]
                normalized["matmul_tflops"] = (
                    flops / (duration * 1e6)
                    if flops is not None and duration and duration > 0 else None
                )
                rows.append(normalized)
                execution_id += 1
    databases, database_warnings = _database_inventory(profile_dir)
    warnings.extend(database_warnings)
    return rows, {
        "metric_family": metric,
        "profile_dir": str(profile_dir.resolve()),
        "kernel_csvs": csv_inventory,
        "profiler_databases": databases,
    }, warnings


def _pmu(row: dict[str, Any], aliases: Sequence[str]) -> float | None:
    metrics = row["metrics"]
    by_norm = {_norm(header): value for header, value in metrics.items()}
    for alias in aliases:
        if _norm(alias) in by_norm:
            return _finite_number(by_norm[_norm(alias)])
    return None


def _linear_shape(row: dict[str, Any]) -> tuple[int, int] | None:
    inputs = _shape_groups(row.get("input_shapes"))
    outputs = _shape_groups(row.get("output_shapes"))
    if not inputs or not outputs:
        return None
    return inputs[0][-1], outputs[0][-1]


def _map_vision_linears(
    rows: list[dict[str, Any]], dims: dict[str, Any] | None
) -> dict[str, Any]:
    if dims is None:
        return {
            "status": "unavailable",
            "method": None,
            "reason": "No complete vision contract was supplied.",
            "replays": [],
            "mismatches": [],
        }
    expected = dims["linear_calls_per_full_stack"]
    layers = dims["layers"]
    if expected != layers * len(ROLES):
        return {
            "status": "failed",
            "method": None,
            "reason": (
                f"Contract requires {expected} linears, but {layers} layers x "
                f"{len(ROLES)} roles is {layers * len(ROLES)}."
            ),
            "replays": [],
            "mismatches": [],
        }
    replays: list[tuple[str, list[dict[str, Any]]]] = []
    method = "step_id"
    by_source_step: dict[tuple[int, int | None], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["is_matmul"]:
            by_source_step[(row["source_index"], row["step_id"])].append(row)
    if by_source_step and all(len(group) == expected for group in by_source_step.values()):
        for key, group in sorted(by_source_step.items()):
            replays.append((f"source{key[0]}:step{key[1]}", sorted(
                group, key=lambda item: (
                    item["start_us"] if item["start_us"] is not None else math.inf,
                    item["source_row"],
                )
            )))
    else:
        method = "chronological_chunks"
        replays = []
        for source_index in sorted({row["source_index"] for row in rows}):
            mats = sorted(
                (row for row in rows
                 if row["source_index"] == source_index and row["is_matmul"]),
                key=lambda item: (
                    item["start_us"] if item["start_us"] is not None else math.inf,
                    item["source_row"],
                ),
            )
            if not mats or len(mats) % expected:
                return {
                    "status": "failed",
                    "method": method,
                    "reason": (
                        f"Source {source_index} has {len(mats)} MatMul kernels; "
                        f"expected a positive multiple of {expected}."
                    ),
                    "replays": [],
                    "mismatches": [],
                }
            for offset in range(0, len(mats), expected):
                replays.append((
                    f"source{source_index}:chunk{offset // expected}",
                    mats[offset:offset + expected],
                ))
    hidden = dims["hidden_size"]
    intermediate = dims["intermediate_size"]
    attention_width = hidden
    if (
        dims.get("head_padding_mode") == "weights"
        and dims.get("real_head_dim")
        and dims.get("padded_head_dim")
        and hidden % dims["real_head_dim"] == 0
    ):
        attention_width = (
            hidden // dims["real_head_dim"] * dims["padded_head_dim"]
        )
    expected_shapes = (
        (hidden, attention_width),
        (hidden, attention_width),
        (hidden, attention_width),
        (attention_width, hidden),
        (hidden, intermediate),
        (intermediate, hidden),
    )
    mismatches: list[dict[str, Any]] = []
    for replay_name, group in replays:
        if len(group) != expected:
            mismatches.append({
                "replay": replay_name, "observed": len(group), "expected": expected
            })
            continue
        for ordinal, row in enumerate(group):
            role_index = ordinal % len(ROLES)
            observed = _linear_shape(row)
            wanted = expected_shapes[role_index]
            if observed != wanted:
                mismatches.append({
                    "replay": replay_name,
                    "ordinal": ordinal,
                    "layer": ordinal // len(ROLES),
                    "role": ROLES[role_index],
                    "observed_k_n": observed,
                    "expected_k_n": wanted,
                    "kernel": row["name"],
                })
    if mismatches:
        return {
            "status": "failed",
            "method": method,
            "reason": (
                "The 162-kernel replay count or the six-role shape motif did "
                "not match the supplied contract; no ordinal labels were applied."
            ),
            "replays": [name for name, _ in replays],
            "mismatches": mismatches[:30],
        }
    for replay_id, (replay_name, group) in enumerate(replays):
        for ordinal, row in enumerate(group):
            row["replay_id"] = replay_id
            row["matmul_ordinal"] = ordinal
            row["layer"] = ordinal // len(ROLES)
            row["role"] = ROLES[ordinal % len(ROLES)]
            row["mapping_status"] = "validated"
    _assign_replays_to_all_rows(rows, replays)
    return {
        "status": "validated",
        "method": method,
        "reason": None,
        "expected_matmuls_per_replay": expected,
        "observed_replay_count": len(replays),
        "replays": [name for name, _ in replays],
        "shape_contract": {
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "attention_projection_width": attention_width,
            "roles": list(ROLES),
        },
        "mismatches": [],
    }


def _assign_replays_to_all_rows(
    rows: list[dict[str, Any]],
    replays: list[tuple[str, list[dict[str, Any]]]],
) -> None:
    by_source: dict[int, list[list[dict[str, Any]]]] = defaultdict(list)
    for _, group in replays:
        by_source[group[0]["source_index"]].append(group)
    replay_offset = 0
    for source_index, groups in sorted(by_source.items()):
        groups.sort(key=lambda group: group[0]["start_us"] or -math.inf)
        starts = [group[0]["start_us"] for group in groups]
        if any(start is None for start in starts):
            replay_offset += len(groups)
            continue
        source_rows = [row for row in rows if row["source_index"] == source_index]
        step_ids = [group[0]["step_id"] for group in groups]
        step_usable = (
            all(step_id is not None for step_id in step_ids)
            and len(set(step_ids)) == len(groups)
        )
        step_map = {group[0]["step_id"]: replay_offset + index
                    for index, group in enumerate(groups)} if step_usable else {}
        for row in source_rows:
            if step_usable and row["step_id"] in step_map:
                row["replay_id"] = step_map[row["step_id"]]
            elif row["start_us"] is not None:
                row["replay_id"] = replay_offset + max(
                    0, bisect.bisect_right(starts, row["start_us"]) - 1
                )
        replay_offset += len(groups)


def _metric_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    headers = sorted({header for row in rows for header in row["metrics"]})
    result: dict[str, Any] = {}
    for header in headers:
        values = [_finite_number(row["metrics"].get(header)) for row in rows]
        summary = _stats(values)
        if summary["available"]:
            summary["duration_weighted_mean"] = _weighted(
                (value, row["duration_us"]) for value, row in zip(values, rows)
            )
            result[header] = summary
    return result


def _interval_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    intervals = sorted(
        (row["start_us"], row["start_us"] + row["duration_us"])
        for row in rows
        if row["start_us"] is not None
        and row["duration_us"] is not None
        and row["duration_us"] >= 0
    )
    if not intervals:
        return {
            "available": False, "span_us": None, "union_us": None,
            "gap_us": None, "overlap_us": None,
        }
    union = 0.0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            union += current_end - current_start
            current_start, current_end = start, end
    union += current_end - current_start
    span = max(end for _, end in intervals) - min(start for start, _ in intervals)
    summed = sum(end - start for start, end in intervals)
    return {
        "available": True,
        "start_us": min(start for start, _ in intervals),
        "end_us": max(end for _, end in intervals),
        "span_us": span,
        "summed_kernel_duration_us": summed,
        "union_us": union,
        "gap_us": max(0.0, span - union),
        "overlap_us": max(0.0, summed - union),
    }


def _aggregate_group(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total_flops = sum(row["matmul_flops"] for row in rows
                      if row["matmul_flops"] is not None)
    duration = sum(row["duration_us"] for row in rows
                   if row["duration_us"] is not None)
    result = {
        "count": len(rows),
        "duration_us": _stats(row["duration_us"] for row in rows),
        "wait_us": _stats(row["wait_us"] for row in rows),
        "matmul_flops": total_flops or None,
        "matmul_tflops": (
            total_flops / (duration * 1e6)
            if total_flops and duration > 0 else None
        ),
    }
    for name, aliases in PMU_ALIASES.items():
        values = [_pmu(row, aliases) for row in rows]
        result[name] = _stats(values)
        result[name]["duration_weighted_mean"] = _weighted(
            (value, row["duration_us"]) for value, row in zip(values, rows)
        )
    return result


def _lane_analysis(
    metric: str, rows: list[dict[str, Any]], mapping: dict[str, Any]
) -> dict[str, Any]:
    kernel_types: dict[str, Any] = {}
    for kernel_type, group in sorted(
        _group(rows, lambda row: row.get("type") or row.get("name") or "<unknown>").items()
    ):
        kernel_types[str(kernel_type)] = _aggregate_group(group)
        kernel_types[str(kernel_type)]["numeric_pmu"] = _metric_summary(group)
    replay_rows = _group(
        [row for row in rows if row["replay_id"] is not None],
        lambda row: row["replay_id"],
    )
    replays = []
    for replay_id, group in sorted(replay_rows.items()):
        interval = _interval_summary(group)
        interval.update({
            "replay_id": replay_id,
            "kernel_count": len(group),
            "matmul_count": sum(row["is_matmul"] for row in group),
            "wait_us": _stats(row["wait_us"] for row in group),
        })
        replays.append(interval)
    mapped = [row for row in rows if row["mapping_status"] == "validated"]
    role_summary = {
        role: _aggregate_group(group)
        for role, group in sorted(_group(mapped, lambda row: row["role"]).items())
    }
    layer_summary = {
        str(layer): _aggregate_group(group)
        for layer, group in sorted(_group(mapped, lambda row: row["layer"]).items())
    }
    blocks = Counter(
        (row["block_dim"], row["mix_block_dim"], row["accelerator_core"])
        for row in rows
    )
    matmuls = [row for row in rows if row["is_matmul"]]
    return {
        "metric_family": metric,
        "kernel_count": len(rows),
        "matmul_count": len(matmuls),
        "mapping": mapping,
        "total_interval": _interval_summary(rows),
        "replays": replays,
        "kernel_type_totals": kernel_types,
        "block_distribution": [
            {
                "block_dim": block,
                "mix_block_dim": mix,
                "accelerator_core": core,
                "count": count,
            }
            for (block, mix, core), count in sorted(
                blocks.items(), key=lambda item: (-item[1], str(item[0]))
            )
        ],
        "matmul": _aggregate_group(matmuls),
        "vision_role_summary": role_summary,
        "vision_layer_summary": layer_summary,
        "numeric_pmu_summary": _metric_summary(rows),
    }


def _group(items: Iterable[Any], key: Any) -> dict[Any, list[Any]]:
    result: dict[Any, list[Any]] = defaultdict(list)
    for item in items:
        result[key(item)].append(item)
    return result


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    return value


def _write_csv(path: Path, fields: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _flat_kernel_rows(rows: Sequence[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    dynamic = sorted({header for row in rows for header in row["metrics"]})
    fields = [
        "execution_id", "lane", "metric_family", "source_index", "source_csv",
        "source_row", "replay_id", "step_id", "device_id", "model_id",
        "task_id", "stream_id", "name", "type", "op_state",
        "accelerator_core", "start_us", "duration_us", "wait_us", "block_dim",
        "mix_block_dim", "input_shapes", "input_dtypes", "input_formats",
        "output_shapes", "output_dtypes", "output_formats", "context_id",
        "is_matmul", "matmul_flops", "matmul_tflops", "matmul_ordinal",
        "layer", "role", "mapping_status", "metrics_json", "raw_json",
    ] + [f"metric:{header}" for header in dynamic]
    flattened = []
    for row in rows:
        item = {key: value for key, value in row.items() if key != "metrics"}
        item.update({
            f"metric:{header}": row["metrics"].get(header)
            for header in dynamic
        })
        flattened.append(item)
    return fields, flattened


def _linear_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        if row["mapping_status"] != "validated":
            continue
        item = {key: row.get(key) for key in (
            "execution_id", "lane", "source_csv", "source_row", "replay_id",
            "step_id", "task_id", "stream_id", "start_us", "duration_us",
            "wait_us", "block_dim", "mix_block_dim", "accelerator_core",
            "name", "type", "input_shapes", "input_formats", "output_shapes",
            "output_formats", "matmul_ordinal", "layer", "role",
            "matmul_flops", "matmul_tflops", "metrics_json", "raw_json",
        )}
        for name, aliases in PMU_ALIASES.items():
            item[name] = _pmu(row, aliases)
        result.append(item)
    return result


def _layer_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    mapped = [row for row in rows if row["mapping_status"] == "validated"]
    for (lane, layer), group in sorted(
        _group(mapped, lambda row: (row["lane"], row["layer"])).items()
    ):
        aggregate = _aggregate_group(group)
        item = {
            "lane": lane,
            "layer": layer,
            "replay_count": len({row["replay_id"] for row in group}),
            "linear_count": len(group),
            "duration_sum_us": aggregate["duration_us"]["sum"],
            "duration_mean_us": aggregate["duration_us"]["mean"],
            "matmul_flops": aggregate["matmul_flops"],
            "matmul_tflops": aggregate["matmul_tflops"],
        }
        for role in ROLES:
            role_rows = [row for row in group if row["role"] == role]
            item[f"{role}_duration_sum_us"] = _stats(
                row["duration_us"] for row in role_rows
            )["sum"]
        for name in PMU_ALIASES:
            item[f"{name}_mean"] = aggregate[name]["mean"]
            item[f"{name}_duration_weighted_mean"] = aggregate[name][
                "duration_weighted_mean"
            ]
        output.append(item)
    return output


def _create_database(
    path: Path,
    kernel_fields: Sequence[str],
    kernels: Sequence[dict[str, Any]],
    linears: Sequence[dict[str, Any]],
    layers: Sequence[dict[str, Any]],
    inventories: Sequence[dict[str, Any]],
) -> None:
    if path.exists():
        path.unlink()
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=DELETE")
    def create_and_insert(name: str, fields: Sequence[str], data: Sequence[dict[str, Any]]) -> None:
        quoted = [f'"{field.replace(chr(34), chr(34) * 2)}"' for field in fields]
        types = []
        for field in fields:
            values = [row.get(field) for row in data if row.get(field) is not None]
            if values and all(isinstance(value, (bool, int)) for value in values):
                types.append("INTEGER")
            elif values and all(
                isinstance(value, (bool, int, float)) for value in values
            ):
                types.append("REAL")
            else:
                types.append("TEXT")
        connection.execute(
            f'CREATE TABLE "{name}" ('
            + ", ".join(f"{column} {kind}" for column, kind in zip(quoted, types))
            + ")"
        )
        if data:
            placeholders = ", ".join("?" for _ in fields)
            connection.executemany(
                f'INSERT INTO "{name}" VALUES ({placeholders})',
                [[row.get(field) for field in fields] for row in data],
            )
    create_and_insert("kernel_executions", kernel_fields, kernels)
    linear_fields = list(linears[0]) if linears else [
        "execution_id", "lane", "replay_id", "layer", "role"
    ]
    create_and_insert("vision_linear_executions", linear_fields, linears)
    layer_fields = list(layers[0]) if layers else ["lane", "layer"]
    create_and_insert("vision_layer_summary", layer_fields, layers)
    schema_rows = []
    db_rows = []
    for inventory in inventories:
        for source in inventory["kernel_csvs"]:
            for ordinal, column in enumerate(source["columns"]):
                schema_rows.append({
                    "lane": inventory["metric_family"],
                    "source": source["path"],
                    "ordinal": ordinal,
                    "column_name": column,
                })
        for database in inventory["profiler_databases"]:
            for table in database.get("tables", []):
                for column in table["columns"]:
                    db_rows.append({
                        "lane": inventory["metric_family"],
                        "database": database["path"],
                        "table_name": table["name"],
                        **column,
                    })
    create_and_insert(
        "raw_csv_schema",
        ["lane", "source", "ordinal", "column_name"],
        schema_rows,
    )
    create_and_insert(
        "source_db_schema",
        ["lane", "database", "table_name", "ordinal", "name", "type",
         "not_null", "default", "primary_key"],
        db_rows,
    )
    connection.execute("CREATE INDEX kernel_lane_replay ON kernel_executions(lane, replay_id)")
    connection.execute("CREATE INDEX linear_lane_layer ON vision_linear_executions(lane, layer)")
    connection.commit()
    connection.close()


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _report(analysis: dict[str, Any], manifest: dict[str, Any]) -> str:
    lines = [
        "# Vision MatMul profiler analysis",
        "",
        f"Generated: `{manifest['generated_at_utc']}`",
        "",
        "This report treats `kernel_details.csv` as the canonical "
        "execution-to-PMU association. Separate metric lanes are separate "
        "captures and are never interpreted as simultaneous samples.",
        "",
        "## Lane summary",
        "",
        "| metric lane | kernels | MatMuls | mapping | span / replay | "
        "MatMul duration / replay | dense MatMul TFLOP/s |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for lane in analysis["lanes"]:
        replay_count = max(1, len(lane["replays"]))
        span = _stats(item["span_us"] for item in lane["replays"])["mean"]
        duration = lane["matmul"]["duration_us"]["sum"]
        per_replay = duration / replay_count if duration is not None else None
        lines.append(
            f"| `{lane['metric_family']}` | {lane['kernel_count']} | "
            f"{lane['matmul_count']} | {lane['mapping']['status']} | "
            f"{_fmt(span)} us | {_fmt(per_replay)} us | "
            f"{_fmt(lane['matmul']['matmul_tflops'])} |"
        )
    lines.extend([
        "",
        "## Validated vision linear mapping",
        "",
        "Ordinal layer/role names are emitted only when every replay contains "
        "exactly 162 MatMul kernels and its repeating six-shape motif matches "
        "the supplied hidden/intermediate/attention-width contract.",
        "",
    ])
    for lane in analysis["lanes"]:
        mapping = lane["mapping"]
        lines.append(
            f"- `{lane['metric_family']}`: **{mapping['status']}**, "
            f"method `{mapping.get('method') or 'unavailable'}`"
            + (f" — {mapping['reason']}" if mapping.get("reason") else "")
        )
        if mapping.get("mismatches"):
            lines.append(
                f"  First mismatch: `{_json(mapping['mismatches'][0])}`"
            )
    lines.extend(["", "## Per-role summary", ""])
    for lane in analysis["lanes"]:
        if not lane["vision_role_summary"]:
            continue
        lines.extend([
            f"### `{lane['metric_family']}`",
            "",
            "| role | count | duration (us mean) | duration / replay (us) | "
            "TFLOP/s | AICore (us mean) | "
            "MAC (us mean) | MTE1 | MTE2 | FixPipe | "
            "AI-core active occupancy |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        replay_count = max(
            1,
            int(
                lane["mapping"].get("observed_replay_count")
                or len(lane["replays"])
            ),
        )
        for role in ROLES:
            item = lane["vision_role_summary"].get(role)
            if not item:
                continue
            duration_sum = item["duration_us"]["sum"]
            duration_per_replay = (
                duration_sum / replay_count
                if duration_sum is not None
                else None
            )
            lines.append(
                f"| {role} | {item['count']} | "
                f"{_fmt(item['duration_us']['mean'])} | "
                f"{_fmt(duration_per_replay)} | "
                f"{_fmt(item['matmul_tflops'])} | "
                f"{_fmt(item['aicore_time_us']['mean'])} | "
                f"{_fmt(item['mac_time_us']['mean'])} | "
                f"{_fmt(item['mte1_time_us']['mean'])} | "
                f"{_fmt(item['mte2_time_us']['mean'])} | "
                f"{_fmt(item['fixpipe_time_us']['mean'])} | "
                f"{_fmt(item['cube_utilization_pct']['mean'])} |"
            )
        lines.append("")
    lines.extend([
        "## Interpretation warnings",
        "",
    ])
    lines.extend(f"- {warning}" for warning in analysis["warnings"])
    lines.extend([
        "",
        "## Durable outputs",
        "",
    ])
    lines.extend(f"- `{name}`" for name in OUTPUT_FILES)
    lines.append("")
    return "\n".join(lines)


def _parse_lane(argument: str) -> tuple[str, Path]:
    if "=" not in argument:
        raise argparse.ArgumentTypeError("--lane must be METRIC=PROFILE_DIR")
    metric, directory = argument.split("=", 1)
    if not metric.strip() or not directory.strip():
        raise argparse.ArgumentTypeError("--lane must be METRIC=PROFILE_DIR")
    return metric.strip(), Path(directory).expanduser()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lane", action="append", required=True, type=_parse_lane,
        metavar="METRIC=PROFILE_DIR",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--contract",
        help="JSON object, @path, or path to a run_summary/contract JSON file.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = _load_contract(args.contract)
    dims = _contract_dims(contract)
    all_rows: list[dict[str, Any]] = []
    inventories: list[dict[str, Any]] = []
    warnings = [
        "Pipe-utilization ratios overlap in time. Do not add MAC, MTE, "
        "Scalar, FixPipe, or Cube ratios as if they partition kernel duration.",
        "Block Num/Block Dim is a configured logical block count, not proof "
        "that the same number of physical AI Cores were active. Per-core "
        "records or hardware counters are required for that claim.",
        "On the observed CANN 9 whole-graph export, cube_utilization(%) is "
        "100 * aicore_time / kernel Duration (within export rounding). It is "
        "AI-core active-time occupancy, not achieved MAC utilization and not "
        "a percentage of peak FLOP/s.",
        "Profiler README semantics define aicore_time as average task time on "
        "AI Core derived from total cycles / Block Num. Generated CANN "
        "documentation warns that this value is inaccurate on Atlas 300V and "
        "Atlas 300I Pro; keep it unavailable or diagnostic on those products.",
        "Profiler captures perturb execution. Use unprofiled device-event "
        "timings for throughput claims and profiler data for diagnosis.",
        "Dense MatMul FLOPs are inferred as 2*M*K*N from activation and output "
        "shapes. This is a physical-work estimate, not an algorithmic FLOP "
        "claim for fused or sparse kernels.",
        "Missing PMU fields are represented as null/unavailable, never zero.",
    ]
    lane_names: set[str] = set()
    lane_rows: list[tuple[str, list[dict[str, Any]]]] = []
    for metric, profile_dir in args.lane:
        if metric in lane_names:
            raise ValueError(f"duplicate metric lane: {metric}")
        lane_names.add(metric)
        rows, inventory, lane_warnings = _read_lane(
            metric, profile_dir.expanduser().resolve()
        )
        all_rows.extend(rows)
        lane_rows.append((metric, rows))
        inventories.append(inventory)
        warnings.extend(lane_warnings)
    lane_analyses = []
    for metric, rows in lane_rows:
        mapping = _map_vision_linears(rows, dims)
        lane_analyses.append(_lane_analysis(metric, rows, mapping))
        if mapping["status"] != "validated":
            warnings.append(
                f"{metric}: vision layer/role mapping is {mapping['status']}; "
                "no unvalidated ordinal labels were emitted."
            )
    kernel_fields, flat_kernels = _flat_kernel_rows(all_rows)
    linears = _linear_rows(all_rows)
    layers = _layer_rows(all_rows)
    _write_csv(output_dir / "kernel_executions.csv", kernel_fields, flat_kernels)
    linear_fields = list(linears[0]) if linears else [
        "execution_id", "lane", "replay_id", "layer", "role"
    ]
    _write_csv(output_dir / "vision_linear_executions.csv", linear_fields, linears)
    layer_fields = list(layers[0]) if layers else ["lane", "layer"]
    _write_csv(output_dir / "vision_layer_summary.csv", layer_fields, layers)
    _create_database(
        output_dir / "profile.sqlite",
        kernel_fields,
        flat_kernels,
        linears,
        layers,
        inventories,
    )
    warnings = list(dict.fromkeys(warnings))
    analysis = {
        "schema_version": SCHEMA_VERSION,
        "contract_dims": dims,
        "lanes": lane_analyses,
        "warnings": warnings,
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": sys.argv,
        "canonical_execution_source": "ASCEND_PROFILER_OUTPUT/kernel_details.csv",
        "contract": contract,
        "contract_dims": dims,
        "sources": inventories,
        "outputs": list(OUTPUT_FILES),
        "notes": [
            "Profiler databases are schema-inventoried but are not implicitly "
            "joined to kernel_details.csv.",
            "Metric lanes are separate captures and may expose different "
            "dynamic PMU columns.",
        ],
    }
    (output_dir / "profile_analysis.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "profile_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "profile_report.md").write_text(
        _report(analysis, manifest), encoding="utf-8"
    )
    print(json.dumps({
        "output_dir": str(output_dir),
        "lanes": [
            {
                "metric": lane["metric_family"],
                "kernels": lane["kernel_count"],
                "mapping": lane["mapping"]["status"],
            }
            for lane in lane_analyses
        ],
        "outputs": list(OUTPUT_FILES),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
