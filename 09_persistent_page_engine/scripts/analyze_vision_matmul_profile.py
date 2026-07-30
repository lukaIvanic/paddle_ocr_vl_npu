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


SCHEMA_VERSION = 2
ROLES = ("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2")
PREFIX_TYPES = ("Data", "LayerNormV3")
LAYER0_TYPES = (
    "MatMulV2", "MatMulV2", "MatMulV2", "ConcatV2D", "SplitVD", "Cast",
    "StridedSliceD", "StridedSliceD", "Neg", "ConcatV2D", "Cast",
    "StridedSliceD", "StridedSliceD", "Neg", "ConcatV2D", "Mul", "Mul",
    "Mul", "Add", "Mul", "Add", "Transpose", "Transpose", "Transpose",
    "memset", "PadV3", "memset", "PadV3", "memset", "PadV3",
    "PromptFlashAttention", "StridedSliceD", "Transpose", "MatMulV2",
    "AddLayerNorm", "MatMulV2", "Gelu", "MatMulV2", "AddLayerNorm",
)
LATER_LAYER_TYPES = (
    "MatMulV2", "MatMulV2", "MatMulV2", "ConcatV2D", "SplitVD",
    "Transpose", "memset", "PadV3", "Cast", "StridedSliceD",
    "StridedSliceD", "Neg", "ConcatV2D", "Mul", "Mul", "Add",
    "Transpose", "memset", "PadV3", "Cast", "StridedSliceD",
    "StridedSliceD", "Neg", "ConcatV2D", "Mul", "Mul", "Add",
    "Transpose", "memset", "PadV3", "PromptFlashAttention",
    "StridedSliceD", "Transpose", "MatMulV2", "AddLayerNorm", "MatMulV2",
    "Gelu", "MatMulV2", "AddLayerNorm",
)
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
                    "layer_kernel_ordinal": None,
                    "layer_component": None,
                    "full_mapping_status": None,
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


def _ordered_replay_rows(
    rows: Sequence[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    grouped = _group(
        [row for row in rows if row["replay_id"] is not None],
        lambda row: int(row["replay_id"]),
    )
    return {
        replay_id: sorted(
            group,
            key=lambda row: (
                row["start_us"] if row["start_us"] is not None else math.inf,
                row["source_row"],
            ),
        )
        for replay_id, group in sorted(grouped.items())
    }


def _kernel_signature(row: dict[str, Any]) -> list[Any]:
    return [
        row.get("type"),
        row.get("accelerator_core"),
        row.get("block_dim"),
        row.get("mix_block_dim"),
        row.get("input_shapes"),
        row.get("input_dtypes"),
        row.get("input_formats"),
        row.get("output_shapes"),
        row.get("output_dtypes"),
        row.get("output_formats"),
    ]


def _signature_hash(rows: Sequence[dict[str, Any]]) -> str:
    payload = _json([_kernel_signature(row) for row in rows]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _layer_component(ordinal: int) -> str:
    if ordinal == 0:
        return "q_proj"
    if ordinal == 1:
        return "k_proj"
    if ordinal == 2:
        return "v_proj"
    if 3 <= ordinal <= 29:
        return "attention_prepare_rope_pad"
    if ordinal == 30:
        return "prompt_flash_attention"
    if ordinal in (31, 32):
        return "attention_output_transform"
    if ordinal == 33:
        return "out_proj"
    if ordinal == 34:
        return "attention_residual_norm"
    if ordinal == 35:
        return "fc1"
    if ordinal == 36:
        return "gelu"
    if ordinal == 37:
        return "fc2"
    if ordinal == 38:
        return "mlp_residual_norm"
    raise AssertionError(f"unexpected layer kernel ordinal {ordinal}")


def _map_full_layers(
    rows: list[dict[str, Any]],
    linear_mapping: dict[str, Any],
    dims: dict[str, Any],
) -> dict[str, Any]:
    if linear_mapping["status"] != "validated":
        return {
            "status": "unavailable",
            "reason": "full-layer mapping requires validated linear anchors",
            "replays": [],
        }
    expected_layers = int(dims["layers"])
    if expected_layers != 27:
        return {
            "status": "unavailable",
            "reason": f"no validated full-layer fixture for {expected_layers} layers",
            "replays": [],
        }
    layer_width = len(LAYER0_TYPES)
    expected_total = len(PREFIX_TYPES) + expected_layers * layer_width
    replays = _ordered_replay_rows(rows)
    failures: list[dict[str, Any]] = []
    validated: list[dict[str, Any]] = []
    for replay_id, group in replays.items():
        if len(group) != expected_total:
            failures.append({
                "replay_id": replay_id,
                "reason": "kernel_count",
                "observed": len(group),
                "expected": expected_total,
            })
            continue
        prefix = group[:len(PREFIX_TYPES)]
        if tuple(row["type"] for row in prefix) != PREFIX_TYPES:
            failures.append({
                "replay_id": replay_id,
                "reason": "prefix_signature",
                "observed": [row["type"] for row in prefix],
            })
            continue
        layer_hashes: list[str] = []
        replay_failed = False
        for layer in range(expected_layers):
            start = len(PREFIX_TYPES) + layer * layer_width
            segment = group[start:start + layer_width]
            expected_types = LAYER0_TYPES if layer == 0 else LATER_LAYER_TYPES
            observed_types = tuple(row["type"] for row in segment)
            if observed_types != expected_types:
                failures.append({
                    "replay_id": replay_id,
                    "layer": layer,
                    "reason": "layer_signature",
                    "observed": list(observed_types),
                    "expected": list(expected_types),
                })
                replay_failed = True
                break
            linear_anchors = {
                0: "q_proj",
                1: "k_proj",
                2: "v_proj",
                33: "out_proj",
                35: "fc1",
                37: "fc2",
            }
            for ordinal, role in linear_anchors.items():
                anchor = segment[ordinal]
                if (
                    anchor["mapping_status"] != "validated"
                    or anchor["layer"] != layer
                    or anchor["role"] != role
                ):
                    failures.append({
                        "replay_id": replay_id,
                        "layer": layer,
                        "ordinal": ordinal,
                        "reason": "linear_anchor",
                        "observed_layer": anchor["layer"],
                        "observed_role": anchor["role"],
                        "expected_role": role,
                    })
                    replay_failed = True
                    break
            if replay_failed:
                break
            layer_hashes.append(_signature_hash(segment))
        if replay_failed:
            continue
        if len(set(layer_hashes[1:])) != 1:
            failures.append({
                "replay_id": replay_id,
                "reason": "later_layer_hashes_differ",
                "hashes": layer_hashes[1:],
            })
            continue
        validated.append({
            "replay_id": replay_id,
            "kernel_count": len(group),
            "signature_sha256": _signature_hash(group),
            "prefix_sha256": _signature_hash(prefix),
            "layer0_sha256": layer_hashes[0],
            "later_layer_sha256": layer_hashes[1],
        })
    if failures or len(validated) != len(replays):
        return {
            "status": "failed",
            "reason": "full replay does not match the validated 27x39 fixture",
            "expected_kernel_count": expected_total,
            "replays": validated,
            "failures": failures[:20],
        }
    for replay_id, group in replays.items():
        for row in group[:len(PREFIX_TYPES)]:
            row["full_mapping_status"] = "validated"
            row["layer_component"] = "graph_prefix"
        for layer in range(expected_layers):
            start = len(PREFIX_TYPES) + layer * layer_width
            for ordinal, row in enumerate(group[start:start + layer_width]):
                row["layer"] = layer
                row["layer_kernel_ordinal"] = ordinal
                row["layer_component"] = _layer_component(ordinal)
                row["full_mapping_status"] = "validated"
    return {
        "status": "validated",
        "method": "paddleocr_vl_27x39_fixture_v1",
        "reason": None,
        "prefix_kernel_count": len(PREFIX_TYPES),
        "layer_kernel_count": layer_width,
        "layer_count": expected_layers,
        "suffix_kernel_count": 0,
        "replays": validated,
        "failures": [],
    }


def _cross_lane_validation(
    lane_analyses: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    signatures: list[dict[str, Any]] = []
    for lane in lane_analyses:
        mapping = lane["full_layer_mapping"]
        for replay in mapping.get("replays", []):
            signatures.append({
                "lane": lane["metric_family"],
                "replay_id": replay["replay_id"],
                "kernel_count": replay["kernel_count"],
                "signature_sha256": replay["signature_sha256"],
            })
    if len(lane_analyses) == 1:
        return {
            "status": "single_lane",
            "reason": "cross-lane comparison requires at least two lanes",
            "replays": signatures,
        }
    if any(
        lane["full_layer_mapping"]["status"] != "validated"
        for lane in lane_analyses
    ):
        return {
            "status": "failed",
            "reason": "at least one lane lacks a validated full-layer mapping",
            "replays": signatures,
        }
    hashes = {item["signature_sha256"] for item in signatures}
    counts = {item["kernel_count"] for item in signatures}
    return {
        "status": "validated" if len(hashes) == 1 and len(counts) == 1 else "failed",
        "reason": (
            None
            if len(hashes) == 1 and len(counts) == 1
            else "full replay signatures differ across lanes"
        ),
        "canonical_signature_sha256": next(iter(hashes)) if len(hashes) == 1 else None,
        "kernel_count": next(iter(counts)) if len(counts) == 1 else None,
        "replays": signatures,
    }


def _metric_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    headers = sorted({header for row in rows for header in row["metrics"]})
    result: dict[str, Any] = {}
    for header in headers:
        values = [_finite_number(row["metrics"].get(header)) for row in rows]
        summary = _stats(values)
        if summary["available"]:
            lowered = header.lower()
            if any(
                token in lowered
                for token in ("ratio", "rate", "bw(", "utilization", "(%)")
            ):
                summary["duration_weighted_mean"] = _weighted(
                    (value, row["duration_us"])
                    for value, row in zip(values, rows)
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
        if name.endswith("_ratio") or name.endswith("_pct"):
            result[name]["duration_weighted_mean"] = _weighted(
                (value, row["duration_us"])
                for value, row in zip(values, rows)
            )
    result["numeric_pmu"] = _metric_summary(rows)
    return result


def _lane_analysis(
    metric: str,
    rows: list[dict[str, Any]],
    mapping: dict[str, Any],
    full_mapping: dict[str, Any],
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
    full_mapped = [
        row
        for row in rows
        if row["full_mapping_status"] == "validated"
        and row["layer"] is not None
    ]
    layer_summary = {
        str(layer): {
            "full": _aggregate_group(group),
            "linears": _aggregate_group(
                [
                    row
                    for row in group
                    if row["mapping_status"] == "validated"
                ]
            ),
        }
        for layer, group in sorted(
            _group(full_mapped, lambda row: row["layer"]).items()
        )
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
        "full_layer_mapping": full_mapping,
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
        "layer", "role", "mapping_status", "layer_kernel_ordinal",
        "layer_component", "full_mapping_status", "metrics_json", "raw_json",
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
    mapped = [
        row
        for row in rows
        if row["full_mapping_status"] == "validated"
        and row["layer"] is not None
    ]
    for (lane, layer), group in sorted(
        _group(mapped, lambda row: (row["lane"], row["layer"])).items()
    ):
        linear_group = [
            row for row in group if row["mapping_status"] == "validated"
        ]
        full_aggregate = _aggregate_group(group)
        linear_aggregate = _aggregate_group(linear_group)
        replay_count = len({row["replay_id"] for row in group})
        item = {
            "lane": lane,
            "layer": layer,
            "replay_count": replay_count,
            "kernel_count": len(group),
            "linear_count": len(linear_group),
            "full_duration_sum_us": full_aggregate["duration_us"]["sum"],
            "full_duration_per_replay_us": (
                full_aggregate["duration_us"]["sum"] / replay_count
                if replay_count else None
            ),
            "linear_duration_sum_us": linear_aggregate["duration_us"]["sum"],
            "linear_duration_per_replay_us": (
                linear_aggregate["duration_us"]["sum"] / replay_count
                if replay_count else None
            ),
            # Compatibility aliases for the original linear-only table.
            "duration_sum_us": linear_aggregate["duration_us"]["sum"],
            "duration_mean_us": linear_aggregate["duration_us"]["mean"],
            "matmul_flops": linear_aggregate["matmul_flops"],
            "matmul_tflops": linear_aggregate["matmul_tflops"],
        }
        for role in ROLES:
            role_rows = [row for row in linear_group if row["role"] == role]
            item[f"{role}_duration_sum_us"] = _stats(
                row["duration_us"] for row in role_rows
            )["sum"]
        for component, component_rows in sorted(
            _group(group, lambda row: row["layer_component"]).items()
        ):
            component_sum = _stats(
                row["duration_us"] for row in component_rows
            )["sum"]
            item[f"component:{component}:duration_per_replay_us"] = (
                component_sum / replay_count if replay_count else None
            )
        for name in PMU_ALIASES:
            item[f"{name}_mean"] = linear_aggregate[name]["mean"]
            item[f"{name}_duration_weighted_mean"] = linear_aggregate[name].get(
                "duration_weighted_mean"
            )
        output.append(item)
    return output


def _metric_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        for metric, raw_value in sorted(row["metrics"].items()):
            value = _finite_number(raw_value)
            if value is None:
                continue
            unit_match = re.search(r"\(([^()]*)\)\s*$", metric)
            output.append({
                "execution_id": row["execution_id"],
                "lane": row["lane"],
                "replay_id": row["replay_id"],
                "layer": row["layer"],
                "role": row["role"],
                "kernel_type": row["type"],
                "metric": metric,
                "value": value,
                "unit": unit_match.group(1) if unit_match else None,
            })
    return output


def _create_database(
    path: Path,
    kernel_fields: Sequence[str],
    kernels: Sequence[dict[str, Any]],
    linears: Sequence[dict[str, Any]],
    layers: Sequence[dict[str, Any]],
    metrics: Sequence[dict[str, Any]],
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
    metric_fields = [
        "execution_id", "lane", "replay_id", "layer", "role",
        "kernel_type", "metric", "value", "unit",
    ]
    create_and_insert("kernel_metrics", metric_fields, metrics)
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
    connection.execute(
        "CREATE INDEX metric_lane_name ON kernel_metrics(lane, metric)"
    )
    connection.execute(
        "CREATE INDEX metric_execution ON kernel_metrics(lane, execution_id)"
    )
    connection.commit()
    connection.close()


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _pmu_value(
    aggregate: dict[str, Any],
    metric: str,
    statistic: str = "mean",
) -> float | None:
    summary = aggregate.get("numeric_pmu", {}).get(metric, {})
    value = summary.get(statistic)
    return float(value) if isinstance(value, (int, float)) else None


def _l2_hit_rate(
    aggregate: dict[str, Any],
    *,
    hit_metric: str,
    miss_metric: str,
) -> float | None:
    hits = _pmu_value(aggregate, hit_metric, "sum")
    misses = _pmu_value(aggregate, miss_metric, "sum")
    if hits is None or misses is None or hits + misses <= 0:
        return None
    return 100.0 * hits / (hits + misses)


def _unique_linear_io_kib(
    role: str,
    dims: dict[str, Any],
) -> tuple[float, float]:
    batch = int(dims["batch_size"])
    sequence = int(dims["sequence_length"])
    hidden = int(dims["hidden_size"])
    intermediate = int(dims["intermediate_size"])
    elements_per_token = batch * sequence
    bytes_per_element = 2  # Exact profiler contract is FLOAT16.
    if role in {"q_proj", "k_proj", "v_proj", "out_proj"}:
        input_elements = (
            elements_per_token * hidden + hidden * hidden + hidden
        )
        output_elements = elements_per_token * hidden
    elif role == "fc1":
        input_elements = (
            elements_per_token * hidden
            + hidden * intermediate
            + intermediate
        )
        output_elements = elements_per_token * intermediate
    elif role == "fc2":
        input_elements = (
            elements_per_token * intermediate
            + intermediate * hidden
            + hidden
        )
        output_elements = elements_per_token * hidden
    else:
        raise ValueError(f"unknown linear role {role}")
    return (
        input_elements * bytes_per_element / 1024.0,
        output_elements * bytes_per_element / 1024.0,
    )


def _coefficient_of_variation(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    if len(finite) < 2 or statistics.mean(finite) == 0:
        return None
    return 100.0 * statistics.pstdev(finite) / statistics.mean(finite)


def _report(analysis: dict[str, Any], manifest: dict[str, Any]) -> str:
    contract = manifest.get("contract") or {}
    environment = contract.get("environment") or {}
    shape = contract.get("shape") or {}
    requested = contract.get("requested") or {}
    compile_info = contract.get("compile") or {}
    weight_format = contract.get("weight_format") or {}
    unprofiled = contract.get("unprofiled_measurements") or {}
    unprofiled_median = (
        (unprofiled.get("device_event_per_call_ms") or {}).get("median")
    )
    unprofiled_tokens = unprofiled.get(
        "physical_tokens_per_s_device_median"
    )
    lines = [
        "# Vision MatMul profiler analysis",
        "",
        f"Generated: `{manifest['generated_at_utc']}`",
        "",
        "This report treats `kernel_details.csv` as the canonical "
        "execution-to-PMU association. Separate metric lanes are separate "
        "captures and are never interpreted as simultaneous samples.",
        "",
        "## Capture provenance",
        "",
        f"- Commit: `{environment.get('commit', 'unavailable')}`",
        f"- Host/device: `{environment.get('hostname', 'unavailable')}` / "
        f"`{environment.get('device_name', 'unavailable')}` / physical "
        f"`{environment.get('ascend_rt_visible_devices', 'unavailable')}`",
        f"- Torch / torch_npu / CANN: `{environment.get('torch', 'unavailable')}` / "
        f"`{environment.get('torch_npu', 'unavailable')}` / "
        f"`{environment.get('ascend_home_path', 'unavailable')}`",
        f"- Shape: `B{shape.get('batch_size', '?')} x "
        f"S{shape.get('sequence_length', '?')}`, H"
        f"`{shape.get('hidden_size', '?')}`, I"
        f"`{shape.get('candidate_intermediate_size', '?')}`, "
        f"{shape.get('layers', '?')} layers",
        f"- Execution: `{requested.get('execution', 'unavailable')}`, "
        f"attention padding `{requested.get('attention_head_padding', 'unavailable')}`, "
        f"RoPE `{requested.get('rotary_implementation', 'unavailable')}`",
        f"- Weight format request/status: "
        f"`{requested.get('weight_format', 'unavailable')}` / "
        f"`{weight_format.get('status', 'unavailable')}`",
        f"- Compile API/cache: `{compile_info.get('api', 'unavailable')}` / "
        f"`{compile_info.get('cache_dir', 'unavailable')}`",
        f"- Unprofiled device-event baseline: "
        f"`{_fmt(unprofiled_median)} ms`, "
        f"`{_fmt(unprofiled_tokens)} physical tok/s`",
        "",
        "## Lane summary",
        "",
        "| metric lane | kernels | MatMuls | mapping | span / replay | "
        "MatMul duration / replay | MatMul share | kernel-local TFLOP/s | "
        "stage-effective TFLOP/s |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for lane in analysis["lanes"]:
        replay_count = max(1, len(lane["replays"]))
        span = _stats(item["span_us"] for item in lane["replays"])["mean"]
        duration = lane["matmul"]["duration_us"]["sum"]
        per_replay = duration / replay_count if duration is not None else None
        matmul_share = (
            100.0 * per_replay / span
            if per_replay is not None and span else None
        )
        flops_per_replay = (
            lane["matmul"]["matmul_flops"] / replay_count
            if lane["matmul"]["matmul_flops"] is not None else None
        )
        stage_effective_tflops = (
            flops_per_replay / (span * 1e6)
            if flops_per_replay is not None and span else None
        )
        lines.append(
            f"| `{lane['metric_family']}` | {lane['kernel_count']} | "
            f"{lane['matmul_count']} | {lane['mapping']['status']} | "
            f"{_fmt(span)} us | {_fmt(per_replay)} us | "
            f"{_fmt(matmul_share)}% | "
            f"{_fmt(lane['matmul']['matmul_tflops'])} | "
            f"{_fmt(stage_effective_tflops)} |"
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
    lines.extend([
        "",
        "## Full-layer and cross-lane validation",
        "",
    ])
    for lane in analysis["lanes"]:
        mapping = lane["full_layer_mapping"]
        lines.append(
            f"- `{lane['metric_family']}` full graph: "
            f"**{mapping['status']}**, method "
            f"`{mapping.get('method') or 'unavailable'}`"
            + (f" — {mapping['reason']}" if mapping.get("reason") else "")
        )
    cross_lane = analysis["cross_lane_validation"]
    lines.append(
        f"- Cross-lane complete-kernel signature: "
        f"**{cross_lane['status']}**"
        + (
            f", SHA-256 `{cross_lane['canonical_signature_sha256']}`"
            if cross_lane.get("canonical_signature_sha256")
            else ""
        )
        + (f" — {cross_lane['reason']}" if cross_lane.get("reason") else "")
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
            "exported AI-core-time ratio (%) |",
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

    primary_lane = next(
        (
            lane
            for lane in analysis["lanes"]
            if lane["metric_family"] == "pipe"
        ),
        analysis["lanes"][0],
    )
    primary_replays = max(1, len(primary_lane["replays"]))
    primary_span = _stats(
        item["span_us"] for item in primary_lane["replays"]
    )["mean"]
    lines.extend([
        "## Whole-stage kernel-time composition",
        "",
        f"Timing composition uses the `{primary_lane['metric_family']}` lane. "
        "Counts and durations below are normalized to one full-stack replay.",
        "",
        "| kernel type | count / replay | duration / replay (us) | "
        "share of replay span |",
        "|---|---:|---:|---:|",
    ])
    top_kernel_types = sorted(
        primary_lane["kernel_type_totals"].items(),
        key=lambda item: item[1]["duration_us"]["sum"] or 0.0,
        reverse=True,
    )[:16]
    for kernel_type, item in top_kernel_types:
        duration_per_replay = (
            item["duration_us"]["sum"] / primary_replays
            if item["duration_us"]["sum"] is not None else None
        )
        share = (
            100.0 * duration_per_replay / primary_span
            if duration_per_replay is not None and primary_span else None
        )
        lines.append(
            f"| {kernel_type} | {_fmt(item['count'] / primary_replays)} | "
            f"{_fmt(duration_per_replay)} | {_fmt(share)}% |"
        )

    if primary_lane["full_layer_mapping"]["status"] == "validated":
        lines.extend([
            "",
            "## Per-layer full-stage timing",
            "",
            "Every row covers the complete 39-kernel layer region, not only "
            "its six Linear kernels.",
            "",
            "| layer | full layer / replay (us) | MatMul / replay (us) | "
            "MatMul share |",
            "|---:|---:|---:|---:|",
        ])
        full_values: list[float | None] = []
        linear_values: list[float | None] = []
        for layer in range(int(analysis["contract_dims"]["layers"])):
            item = primary_lane["vision_layer_summary"].get(str(layer))
            if not item:
                continue
            full_sum = item["full"]["duration_us"]["sum"]
            linear_sum = item["linears"]["duration_us"]["sum"]
            full_per_replay = (
                full_sum / primary_replays if full_sum is not None else None
            )
            linear_per_replay = (
                linear_sum / primary_replays
                if linear_sum is not None else None
            )
            full_values.append(full_per_replay)
            linear_values.append(linear_per_replay)
            share = (
                100.0 * linear_per_replay / full_per_replay
                if linear_per_replay is not None and full_per_replay else None
            )
            lines.append(
                f"| {layer} | {_fmt(full_per_replay)} | "
                f"{_fmt(linear_per_replay)} | {_fmt(share)}% |"
            )
        lines.extend([
            "",
            f"Full-layer duration CV: "
            f"`{_fmt(_coefficient_of_variation(full_values))}%`; "
            f"MatMul-duration CV: "
            f"`{_fmt(_coefficient_of_variation(linear_values))}%`.",
        ])

    memory_lane = next(
        (
            lane
            for lane in analysis["lanes"]
            if lane["metric_family"] == "memory"
        ),
        None,
    )
    if memory_lane is not None:
        lines.extend([
            "",
            "## Memory bandwidth-rate diagnostics",
            "",
            "These are PMU rates normalized to AI-core cycles. They are not "
            "whole-card HBM bandwidth. In this capture the L2 bandwidth "
            "columns are zero even though the separate L2 lane records cache "
            "events, so zero L2 bandwidth is treated as unpopulated.",
            "",
            "| role | L1 read | L1 write | main-memory read | "
            "main-memory write |",
            "|---|---:|---:|---:|---:|",
        ])
        for role in ROLES:
            item = memory_lane["vision_role_summary"].get(role)
            if not item:
                continue
            lines.append(
                f"| {role} | "
                f"{_fmt(_pmu_value(item, 'aic_l1_read_bw(GB/s)'))} GB/s | "
                f"{_fmt(_pmu_value(item, 'aic_l1_write_bw(GB/s)'))} GB/s | "
                f"{_fmt(_pmu_value(item, 'aic_main_mem_read_bw(GB/s)'))} GB/s | "
                f"{_fmt(_pmu_value(item, 'aic_main_mem_write_bw(GB/s)'))} GB/s |"
            )

    l2_lane = next(
        (
            lane
            for lane in analysis["lanes"]
            if lane["metric_family"] == "l2"
        ),
        None,
    )
    if l2_lane is not None:
        lines.extend([
            "",
            "## L2 event diagnostics",
            "",
            "Hit rates are `hit / (hit + miss_allocate)` over raw event "
            "counts. R0/R1 are hardware read channels, not semantic labels "
            "for activations and weights, and event counts are not bytes.",
            "",
            "| role | R0 read hit | R1 read hit | write hit |",
            "|---|---:|---:|---:|",
        ])
        for role in ROLES:
            item = l2_lane["vision_role_summary"].get(role)
            if not item:
                continue
            r0 = _l2_hit_rate(
                item,
                hit_metric="aic_r0_read_cache_hit",
                miss_metric="aic_r0_read_cache_miss_allocate",
            )
            r1 = _l2_hit_rate(
                item,
                hit_metric="aic_r1_read_cache_hit",
                miss_metric="aic_r1_read_cache_miss_allocate",
            )
            write = _l2_hit_rate(
                item,
                hit_metric="aic_write_cache_hit",
                miss_metric="aic_write_cache_miss_allocate",
            )
            lines.append(
                f"| {role} | {_fmt(r0)}% | {_fmt(r1)}% | {_fmt(write)}% |"
            )

    memory_access_lane = next(
        (
            lane
            for lane in analysis["lanes"]
            if lane["metric_family"] == "memory_access"
        ),
        None,
    )
    if memory_access_lane is not None:
        lines.extend([
            "",
            "## MemoryAccess traffic diagnostics",
            "",
            "The unique-read baseline is the FP16 activation + physical "
            "FRACTAL_NZ weight + bias tensor size for one Linear call. "
            "Reported access volume is task-level logical memory traffic; it "
            "must not be treated as off-chip HBM bytes without accounting for "
            "cache service and the msprof metric semantics.",
            "",
            "| role | reported read / op | reported write / op | GM->L1 | "
            "L0C->GM | unique read | read amplification | FLOP / reported byte |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        total_read_kib = 0.0
        total_write_kib = 0.0
        total_flops = 0.0
        layers = int(analysis["contract_dims"]["layers"])
        for role in ROLES:
            item = memory_access_lane["vision_role_summary"].get(role)
            if not item:
                continue
            read_kib = _pmu_value(
                item, "aic_read_main_memory_datas(KB)"
            )
            write_kib = _pmu_value(
                item, "aic_write_main_memory_datas(KB)"
            )
            gm_l1_kib = _pmu_value(item, "aic_GM_to_L1_datas(KB)")
            l0c_gm_kib = _pmu_value(item, "aic_L0C_to_GM_datas(KB)")
            unique_read_kib, _ = _unique_linear_io_kib(
                role, analysis["contract_dims"]
            )
            amplification = (
                read_kib / unique_read_kib
                if read_kib is not None and unique_read_kib else None
            )
            flops_per_op = (
                item["matmul_flops"] / item["count"]
                if item["matmul_flops"] is not None and item["count"] else None
            )
            intensity = (
                flops_per_op / ((read_kib + write_kib) * 1024.0)
                if flops_per_op is not None
                and read_kib is not None
                and write_kib is not None
                and read_kib + write_kib > 0
                else None
            )
            if read_kib is not None:
                total_read_kib += read_kib * layers
            if write_kib is not None:
                total_write_kib += write_kib * layers
            if flops_per_op is not None:
                total_flops += flops_per_op * layers
            lines.append(
                f"| {role} | {_fmt(read_kib)} KiB | "
                f"{_fmt(write_kib)} KiB | {_fmt(gm_l1_kib)} KiB | "
                f"{_fmt(l0c_gm_kib)} KiB | {_fmt(unique_read_kib)} KiB | "
                f"{_fmt(amplification)}x | {_fmt(intensity)} |"
            )
        total_intensity = (
            total_flops / ((total_read_kib + total_write_kib) * 1024.0)
            if total_read_kib + total_write_kib > 0 else None
        )
        lines.extend([
            "",
            f"Across all 162 Linear calls in one replay: reported read "
            f"`{_fmt(total_read_kib / 1024.0)} MiB`, reported write "
            f"`{_fmt(total_write_kib / 1024.0)} MiB`, and "
            f"`{_fmt(total_intensity)} FLOP/reported-byte`.",
        ])

    lines.extend([
        "",
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
        "an exported AI-core-time ratio that can exceed 100%, not physical "
        "occupancy, achieved MAC utilization, or a percentage of peak FLOP/s.",
        "Profiler README semantics define aicore_time as average task time on "
        "AI Core derived from total cycles / Block Num. Generated CANN "
        "documentation warns that this value is inaccurate on Atlas 300V and "
        "Atlas 300I Pro; keep it unavailable or diagnostic on those products.",
        "Profiler captures perturb execution. Use unprofiled device-event "
        "timings for throughput claims and profiler data for diagnosis.",
        "Dense MatMul FLOPs are inferred as 2*M*K*N from activation and output "
        "shapes. This is a physical-work estimate, not an algorithmic FLOP "
        "claim for fused or sparse kernels.",
        "The full-layer fixture is intentionally fail-closed and applies only "
        "to the exact validated 2-prefix + 27x39-kernel PaddleOCR-VL graph. "
        "A changed graph emits no non-linear layer labels.",
        "The first kernel's Wait Time in a replay can include the gap since "
        "the prior replay; replay wait totals are not internal graph stall "
        "time without timestamp decomposition.",
        "Profiler databases are schema-inventoried only. TASK, CANN_API, "
        "PYTORCH_API, and compiled-node relationships are not silently joined.",
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
    for execution_id, row in enumerate(all_rows):
        row["execution_id"] = execution_id
    lane_analyses = []
    for metric, rows in lane_rows:
        mapping = _map_vision_linears(rows, dims)
        full_mapping = _map_full_layers(rows, mapping, dims)
        lane_analyses.append(
            _lane_analysis(metric, rows, mapping, full_mapping)
        )
        if mapping["status"] != "validated":
            warnings.append(
                f"{metric}: vision layer/role mapping is {mapping['status']}; "
                "no unvalidated ordinal labels were emitted."
            )
        if full_mapping["status"] != "validated":
            warnings.append(
                f"{metric}: full 27-layer mapping is "
                f"{full_mapping['status']}; non-linear kernels remain "
                "unlabeled."
            )
    kernel_fields, flat_kernels = _flat_kernel_rows(all_rows)
    linears = _linear_rows(all_rows)
    layers = _layer_rows(all_rows)
    metrics = _metric_rows(all_rows)
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
        metrics,
        inventories,
    )
    warnings = list(dict.fromkeys(warnings))
    analysis = {
        "schema_version": SCHEMA_VERSION,
        "contract_dims": dims,
        "lanes": lane_analyses,
        "cross_lane_validation": _cross_lane_validation(lane_analyses),
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
