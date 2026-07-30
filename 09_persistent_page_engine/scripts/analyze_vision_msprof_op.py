#!/usr/bin/env python3
"""Validate and normalize deep ``msprof op`` vision-Linear captures.

This analyzer joins three deliberately separate sources of truth:

* the direct-capture target summaries (logical Linear shape and formats);
* recursively discovered ``msprof op`` artifacts (actual dispatch and metrics);
* a normalized full-graph ``vision_linear_executions.csv`` reference.

It is intentionally standard-library-only and derives product, Block Dim,
operator names, formats, and paths from the inputs.  Kernel-replay duration is
reported only as a diagnostic: replay timing is not an end-to-end throughput
measurement and is not used as a representativeness gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import struct
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DIRECT_ROLE_MAP: dict[str, tuple[str, ...]] = {
    "square": ("q_proj", "k_proj", "v_proj", "out_proj"),
    "fc1": ("fc1",),
    "fc2": ("fc2",),
}
REFERENCE_ROLES = tuple(
    role for roles in DIRECT_ROLE_MAP.values() for role in roles
)
FRAME_HEADER = struct.Struct("<QBBBB")
OCCUPANCY_RECORD_TYPE = 0x0C
HUAWEI_26_SCORE_THRESHOLD = 0.6
HUAWEI_26_Z_THRESHOLD = math.log(
    HUAWEI_26_SCORE_THRESHOLD / (1.0 - HUAWEI_26_SCORE_THRESHOLD)
)
HUAWEI_26_SOURCE = (
    "Ascend/msopprof 26.0.0 "
    "csrc/op_profiling/profiling/device/data_visualize/occupancy.cpp"
)
HUAWEI_26_SOURCE_COMMIT = "4af9e75f0f75b2f7252b40aaf0feb7e721f5f7da"
HUAWEI_26_SOURCE_URL = (
    "https://gitcode.com/Ascend/msopprof/blob/26.0.0/"
    "csrc/op_profiling/profiling/device/data_visualize/occupancy.cpp"
)

METRIC_RECORD_FIELDS = (
    "role",
    "capture_metric",
    "source_kind",
    "source_path",
    "record_type",
    "record_index",
    "core_id",
    "subcore_id",
    "subcore_type",
    "category",
    "metric",
    "unit",
    "raw_value",
    "numeric_value",
    "is_missing",
    "context_json",
)

CORE_METRIC_FIELDS = (
    "role",
    "capture_metric",
    "source_path",
    "device",
    "operator_name",
    "operator_core_type",
    "block_dim",
    "core_id",
    "subcore_id",
    "subcore_type",
    "duration_us",
    "cycles",
    "cycles_mean",
    "cycles_population_stddev",
    "cycles_z_score",
    "cycles_sigmoid_score",
    "cycles_huawei_26_flag",
    "throughput",
    "throughput_mean",
    "throughput_population_stddev",
    "throughput_z_score",
    "throughput_sigmoid_score",
    "throughput_huawei_26_flag",
    "l2_cache_hit_rate",
    "l2_cache_hit_rate_mean",
    "l2_cache_hit_rate_population_stddev",
    "l2_cache_hit_rate_z_score",
    "l2_cache_hit_rate_sigmoid_score",
    "l2_cache_hit_rate_huawei_26_flag",
    "simt_instructions",
    "simt_instructions_mean",
    "simt_instructions_population_stddev",
    "simt_instructions_z_score",
    "simt_instructions_sigmoid_score",
    "simt_instructions_huawei_26_flag",
    "other_metrics_json",
)


class AnalysisError(RuntimeError):
    """A fail-closed input, schema, or validation error."""


@dataclass(frozen=True)
class JsonFrame:
    path: Path
    offset: int
    payload_length: int
    record_type: int
    declared_padding: int
    version: int
    reserved: int
    observed_padding: int
    value: Any


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture-dir",
        type=Path,
        required=True,
        help="Evidence run containing suite/capture manifests and target summaries.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Raw msprof-op run root; timestamped descendants are discovered.",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        required=True,
        help="Normalized full-graph reference containing vision_linear_executions.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New empty directory for normalized analysis artifacts.",
    )
    return parser.parse_args(argv)


def _resolve_input(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise AnalysisError(f"{label} is not a directory: {resolved}")
    return resolved


def _prepare_output(path: Path, inputs: Iterable[Path]) -> Path:
    resolved = path.expanduser().resolve()
    if resolved in inputs:
        raise AnalysisError(f"output directory aliases an input: {resolved}")
    if resolved.exists() and any(resolved.iterdir()):
        raise AnalysisError(f"output directory is not empty: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"failed to read JSON {path}: {exc}") from exc


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _normalize_name(value: Any) -> str:
    text = str(value).strip().lower()
    text = text.replace("μ", "u")
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def _clean_csv_row(row: Mapping[Any, Any]) -> dict[str, str]:
    cleaned: dict[str, str] = {}
    for key, value in row.items():
        if key is None:
            continue
        name = str(key).strip()
        if not name:
            continue
        cleaned[name] = "" if value is None else str(value).strip()
    return cleaned


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise AnalysisError(f"CSV has no header: {path}")
            headers = [str(value).strip() for value in reader.fieldnames if value]
            rows = [_clean_csv_row(row) for row in reader]
    except (OSError, UnicodeError, csv.Error) as exc:
        raise AnalysisError(f"failed to read CSV {path}: {exc}") from exc
    return headers, rows


def _to_number(value: Any) -> float | int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    text = str(value).strip()
    if not text or text.lower() in {"na", "n/a", "nan", "none", "null", "-"}:
        return None
    text = text.replace(",", "")
    if text.endswith("%"):
        text = text[:-1].strip()
    try:
        result = float(text)
    except ValueError:
        return None
    if not math.isfinite(result):
        return None
    if result.is_integer() and not any(char in text.lower() for char in ".e"):
        return int(result)
    return result


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {
            "",
            "na",
            "n/a",
            "nan",
            "none",
            "null",
            "-",
        }
    return False


def _json_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _json_scalar(row.get(field)) for field in fields})


def _schema_walk(value: Any) -> dict[str, Any]:
    types: dict[str, Counter[str]] = defaultdict(Counter)
    list_lengths: dict[str, list[int]] = defaultdict(list)

    def visit(current: Any, path: str) -> None:
        if current is None:
            kind = "null"
        elif isinstance(current, bool):
            kind = "boolean"
        elif isinstance(current, int):
            kind = "integer"
        elif isinstance(current, float):
            kind = "number"
        elif isinstance(current, str):
            kind = "string"
        elif isinstance(current, list):
            kind = "array"
        elif isinstance(current, dict):
            kind = "object"
        else:
            kind = type(current).__name__
        types[path][kind] += 1
        if isinstance(current, dict):
            for key, child in current.items():
                visit(child, f"{path}.{key}")
        elif isinstance(current, list):
            list_lengths[path].append(len(current))
            for child in current:
                visit(child, f"{path}[]")

    visit(value, "$")
    paths = []
    for path in sorted(types):
        item: dict[str, Any] = {
            "path": path,
            "types": dict(sorted(types[path].items())),
        }
        lengths = list_lengths.get(path)
        if lengths:
            item["list_count"] = len(lengths)
            item["list_items_total"] = sum(lengths)
            item["list_length_min"] = min(lengths)
            item["list_length_max"] = max(lengths)
        paths.append(item)
    return {"paths": paths}


def _decode_visualize_data(path: Path) -> list[JsonFrame]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise AnalysisError(f"failed to read binary {path}: {exc}") from exc
    frames: list[JsonFrame] = []
    offset = 0
    while offset < len(data):
        remaining = len(data) - offset
        if remaining < FRAME_HEADER.size:
            raise AnalysisError(
                f"truncated visualize_data header at offset {offset} in {path}: "
                f"{remaining} bytes remain"
            )
        payload_length, record_type, declared_padding, version, reserved = (
            FRAME_HEADER.unpack_from(data, offset)
        )
        payload_start = offset + FRAME_HEADER.size
        payload_end = payload_start + payload_length
        if payload_end > len(data):
            raise AnalysisError(
                f"truncated visualize_data payload at offset {offset} in {path}: "
                f"declared {payload_length}, available {len(data) - payload_start}"
            )
        raw_payload = data[payload_start:payload_end]
        if declared_padding > len(raw_payload):
            raise AnalysisError(
                f"invalid declared JSON padding {declared_padding} for "
                f"{len(raw_payload)}-byte record at offset {offset} in {path}"
            )
        observed_padding = len(raw_payload) - len(raw_payload.rstrip(b"\x00"))
        if observed_padding != declared_padding:
            raise AnalysisError(
                f"visualize_data padding mismatch at offset {offset} in {path}: "
                f"header declares {declared_padding}, observed "
                f"{observed_padding} trailing NUL bytes"
            )
        json_payload = (
            raw_payload[:-declared_padding]
            if declared_padding
            else raw_payload
        )
        try:
            value = json.loads(json_payload.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise AnalysisError(
                f"invalid JSON record type 0x{record_type:02x} at "
                f"offset {offset} in {path}: {exc}"
            ) from exc
        frames.append(
            JsonFrame(
                path=path,
                offset=offset,
                payload_length=payload_length,
                record_type=record_type,
                declared_padding=declared_padding,
                version=version,
                reserved=reserved,
                observed_padding=observed_padding,
                value=value,
            )
        )
        offset = payload_end
    if not frames:
        raise AnalysisError(f"visualize_data contains no records: {path}")
    return frames


def _find_single_manifest(root: Path) -> tuple[Path | None, dict[str, Any] | None]:
    paths = sorted(root.rglob("suite_manifest.json"))
    if len(paths) > 1:
        raise AnalysisError(
            "capture directory contains multiple suite_manifest.json files: "
            + ", ".join(str(path) for path in paths)
        )
    if not paths:
        return None, None
    value = _read_json(paths[0])
    if not isinstance(value, dict):
        raise AnalysisError(f"suite manifest is not an object: {paths[0]}")
    return paths[0], value


def _load_targets(capture_root: Path) -> dict[str, dict[str, Any]]:
    targets: dict[str, dict[str, Any]] = {}
    for path in sorted(capture_root.rglob("target_summary.json")):
        value = _read_json(path)
        if not isinstance(value, dict):
            raise AnalysisError(f"target summary is not an object: {path}")
        spec = value.get("spec")
        if not isinstance(spec, dict) or not spec.get("role"):
            raise AnalysisError(f"target summary lacks spec.role: {path}")
        role = str(spec["role"])
        if role not in DIRECT_ROLE_MAP:
            raise AnalysisError(f"unsupported direct target role {role!r}: {path}")
        if role in targets:
            raise AnalysisError(
                f"multiple target summaries for role {role}: "
                f"{targets[role]['path']}, {path}"
            )
        targets[role] = {"path": path, "value": value}
    if not targets:
        raise AnalysisError(
            f"no target_summary.json found recursively under {capture_root}"
        )
    return targets


def _capture_metric(
    suite_manifest: Mapping[str, Any] | None,
    capture_root: Path,
) -> str:
    values: list[str] = []
    if suite_manifest and suite_manifest.get("metric"):
        values.append(str(suite_manifest["metric"]))
    for path in sorted(capture_root.rglob("capture_manifest.json")):
        value = _read_json(path)
        if not isinstance(value, dict):
            continue
        selection = (
            value.get("command", {}).get("selection", {})
            if isinstance(value.get("command"), dict)
            else {}
        )
        if isinstance(selection, dict) and selection.get("metric"):
            values.append(str(selection["metric"]))
    normalized = {_normalize_name(value): value for value in values}
    if len(normalized) > 1:
        raise AnalysisError(
            "capture manifests disagree on requested metric: "
            + ", ".join(sorted(values))
        )
    return next(iter(normalized.values())) if normalized else "unknown"


def _infer_role(path: Path, raw_root: Path, roles: Sequence[str]) -> str | None:
    try:
        parts = path.relative_to(raw_root).parts
    except ValueError:
        parts = path.parts
    matches = [role for role in roles if role in parts]
    if len(matches) > 1:
        raise AnalysisError(
            f"raw artifact path ambiguously names multiple roles {matches}: {path}"
        )
    if matches:
        return matches[0]
    if len(roles) == 1:
        return roles[0]
    return None


def _group_raw_files(
    raw_root: Path,
    roles: Sequence[str],
) -> tuple[dict[str, list[Path]], list[Path]]:
    grouped: dict[str, list[Path]] = {role: [] for role in roles}
    unassigned: list[Path] = []
    for path in sorted(raw_root.rglob("*")):
        if not path.is_file():
            continue
        role = _infer_role(path, raw_root, roles)
        if role is None:
            unassigned.append(path)
        else:
            grouped[role].append(path)
    return grouped, unassigned


def _parse_shape(value: str) -> tuple[tuple[int, ...], ...]:
    text = value.strip()
    while len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
    if not text:
        return ()
    shapes: list[tuple[int, ...]] = []
    for segment in text.split(";"):
        segment = segment.strip().strip('"').strip("'")
        if not segment:
            shapes.append(())
            continue
        try:
            shapes.append(tuple(int(part.strip()) for part in segment.split(",")))
        except ValueError as exc:
            raise AnalysisError(f"cannot parse shape {value!r}") from exc
    return tuple(shapes)


def _normalize_dtype(value: Any) -> str:
    text = _normalize_name(value)
    aliases = {
        "torch_float16": "float16",
        "fp16": "float16",
        "float16": "float16",
        "torch_bfloat16": "bfloat16",
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "torch_float32": "float32",
        "fp32": "float32",
        "float": "float32",
        "float32": "float32",
    }
    return aliases.get(text, text)


def _load_reference(reference_root: Path) -> dict[str, Any]:
    paths = sorted(reference_root.rglob("vision_linear_executions.csv"))
    if not paths:
        raise AnalysisError(
            "reference directory contains no vision_linear_executions.csv: "
            f"{reference_root}"
        )
    all_rows: list[dict[str, str]] = []
    schemas = []
    for path in paths:
        headers, rows = _read_csv(path)
        required = {
            "role",
            "type",
            "block_dim",
            "input_shapes",
            "input_formats",
            "output_shapes",
            "output_formats",
            "matmul_flops",
            "duration_us",
            "raw_json",
        }
        missing = sorted(required - set(headers))
        if missing:
            raise AnalysisError(
                f"reference CSV {path} lacks required columns: {missing}"
            )
        all_rows.extend(rows)
        schemas.append(
            {
                "path": _relative(path, reference_root),
                "headers": headers,
                "row_count": len(rows),
            }
        )
    by_role: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in all_rows:
        role = row.get("role", "")
        if role in REFERENCE_ROLES:
            by_role[role].append(row)
    missing_roles = [role for role in REFERENCE_ROLES if not by_role.get(role)]
    if missing_roles:
        raise AnalysisError(
            "full-graph reference lacks required production roles: "
            + ", ".join(missing_roles)
        )

    contracts: dict[str, dict[str, Any]] = {}
    for role in REFERENCE_ROLES:
        rows = by_role[role]

        def unique(column: str) -> str:
            values = {row[column].strip() for row in rows if row[column].strip()}
            if len(values) != 1:
                raise AnalysisError(
                    f"reference role {role} has non-unique {column}: "
                    f"{sorted(values)}"
                )
            return next(iter(values))

        durations = [
            float(value)
            for row in rows
            if (value := _to_number(row.get("duration_us"))) is not None
        ]
        if not durations:
            raise AnalysisError(f"reference role {role} has no numeric duration")

        def unique_raw_json_field(field: str) -> str:
            values = set()
            for row in rows:
                try:
                    raw = json.loads(row["raw_json"])
                except (TypeError, json.JSONDecodeError) as exc:
                    raise AnalysisError(
                        f"reference role {role} has invalid raw_json"
                    ) from exc
                if not isinstance(raw, dict):
                    raise AnalysisError(
                        f"reference role {role} raw_json is not an object"
                    )
                value = raw.get(field)
                if value not in {None, ""}:
                    values.add(str(value).strip())
            if len(values) != 1:
                raise AnalysisError(
                    f"reference role {role} has non-unique raw_json[{field!r}]: "
                    f"{sorted(values)}"
                )
            return next(iter(values))

        contract = {
            "role": role,
            "row_count": len(rows),
            "operator_type": unique("type"),
            "accelerator_core": (
                unique("accelerator_core")
                if "accelerator_core" in rows[0]
                and any(row.get("accelerator_core") for row in rows)
                else None
            ),
            "block_dim": int(unique("block_dim")),
            "input_shapes_raw": unique("input_shapes"),
            "input_shapes": _parse_shape(unique("input_shapes")),
            "input_formats": tuple(
                part.strip() for part in unique("input_formats").split(";")
            ),
            "input_dtypes": tuple(
                _normalize_dtype(part)
                for part in unique_raw_json_field("Input Data Types").split(";")
            ),
            "output_shapes_raw": unique("output_shapes"),
            "output_shapes": _parse_shape(unique("output_shapes")),
            "output_formats": tuple(
                part.strip() for part in unique("output_formats").split(";")
            ),
            "output_dtypes": tuple(
                _normalize_dtype(part)
                for part in unique_raw_json_field("Output Data Types").split(";")
            ),
            "flops": int(unique("matmul_flops")),
            "duration_us_diagnostic": {
                "count": len(durations),
                "min": min(durations),
                "mean": statistics.fmean(durations),
                "median": statistics.median(durations),
                "max": max(durations),
            },
        }
        contracts[role] = contract
    return {
        "paths": paths,
        "schemas": schemas,
        "row_count": len(all_rows),
        "contracts": contracts,
    }


def _add_metric_record(
    records: list[dict[str, Any]],
    *,
    role: str,
    capture_metric: str,
    source_kind: str,
    source_path: str,
    record_type: str,
    record_index: int | None,
    core_id: Any,
    subcore_id: Any = None,
    subcore_type: Any = None,
    category: str,
    metric: str,
    value: Any,
    unit: Any = None,
    context: Mapping[str, Any] | None = None,
) -> None:
    numeric = _to_number(value)
    records.append(
        {
            "role": role,
            "capture_metric": capture_metric,
            "source_kind": source_kind,
            "source_path": source_path,
            "record_type": record_type,
            "record_index": record_index,
            "core_id": core_id,
            "subcore_id": subcore_id,
            "subcore_type": subcore_type,
            "category": category,
            "metric": metric,
            "unit": unit,
            "raw_value": value,
            "numeric_value": numeric,
            "is_missing": _is_missing(value),
            "context_json": (
                json.dumps(context, sort_keys=True, ensure_ascii=False)
                if context
                else ""
            ),
        }
    )


def _canonical_occupancy_metric(name: str) -> str:
    normalized = _normalize_name(name)
    aliases = {
        "l2cache_hit_rate": "l2_cache_hit_rate",
        "l2_cache_hit_rate": "l2_cache_hit_rate",
        "cache_hit_rate": "l2_cache_hit_rate",
        "simt_instruction": "simt_instructions",
        "simt_instructions": "simt_instructions",
    }
    return aliases.get(normalized, normalized)


def _extract_binary_metrics(
    frames: Sequence[JsonFrame],
    *,
    role: str,
    capture_metric: str,
    raw_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    occupancy_rows: list[dict[str, Any]] = []
    block_durations: dict[str, float] = {}
    op_metadata: dict[str, Any] = {}
    source_path = _relative(frames[0].path, raw_root)

    for record_index, frame in enumerate(frames):
        value = frame.value
        record_type = f"0x{frame.record_type:02x}"
        if not isinstance(value, dict):
            continue

        block_detail = value.get("block_detail")
        if isinstance(block_detail, dict):
            headers = block_detail.get("head_name")
            rows = block_detail.get("row")
            if isinstance(headers, list) and isinstance(rows, list):
                normalized_headers = [_normalize_name(item) for item in headers]
                for row in rows:
                    cells = row.get("value") if isinstance(row, dict) else None
                    if not isinstance(cells, list):
                        continue
                    mapping = dict(zip(normalized_headers, cells))
                    core_id = mapping.get("block_id")
                    duration_key = next(
                        (
                            key
                            for key in mapping
                            if key.startswith("duration")
                        ),
                        None,
                    )
                    if duration_key is not None:
                        duration = _to_number(mapping[duration_key])
                        if duration is not None:
                            block_durations[str(core_id)] = float(duration)
                            _add_metric_record(
                                records,
                                role=role,
                                capture_metric=capture_metric,
                                source_kind="visualize_json",
                                source_path=source_path,
                                record_type=record_type,
                                record_index=record_index,
                                core_id=core_id,
                                subcore_type=mapping.get("core_type"),
                                category="block_detail",
                                metric="duration_us",
                                value=mapping[duration_key],
                                unit="us",
                            )
            for key in (
                "name",
                "op_type",
                "block_dim",
                "mix_block_dim",
                "device_id",
                "soc",
                "duration",
                "cur_freq",
                "rated_freq",
            ):
                if key in value:
                    op_metadata.setdefault(key, value[key])

        subblock_detail = value.get("subblock_detail")
        if isinstance(subblock_detail, list):
            for item in subblock_detail:
                if not isinstance(item, dict):
                    continue
                core_id = item.get("block_id")
                name = str(item.get("name", "unnamed"))
                context = {
                    key: child
                    for key, child in item.items()
                    if key not in {"value", "origin_value"}
                }
                if "value" in item:
                    _add_metric_record(
                        records,
                        role=role,
                        capture_metric=capture_metric,
                        source_kind="visualize_json",
                        source_path=source_path,
                        record_type=record_type,
                        record_index=record_index,
                        core_id=core_id,
                        subcore_type=item.get("block_type"),
                        category="subblock_detail",
                        metric=name,
                        value=item.get("value"),
                        unit=item.get("unit"),
                        context=context,
                    )
                if "origin_value" in item:
                    _add_metric_record(
                        records,
                        role=role,
                        capture_metric=capture_metric,
                        source_kind="visualize_json",
                        source_path=source_path,
                        record_type=record_type,
                        record_index=record_index,
                        core_id=core_id,
                        subcore_type=item.get("block_type"),
                        category="subblock_detail_origin",
                        metric=name,
                        value=item.get("origin_value"),
                        unit=item.get("unit"),
                        context=context,
                    )

        core_memory_map = value.get("core_memory_map")
        if isinstance(core_memory_map, list):
            for item in core_memory_map:
                if not isinstance(item, dict):
                    continue
                core_id = item.get("core_no")
                for category, child in item.items():
                    if category in {"core_no", "advice", "memory_unit"}:
                        continue
                    if isinstance(child, dict):
                        for metric, metric_value in child.items():
                            _add_metric_record(
                                records,
                                role=role,
                                capture_metric=capture_metric,
                                source_kind="visualize_json",
                                source_path=source_path,
                                record_type=record_type,
                                record_index=record_index,
                                core_id=core_id,
                                category=str(category),
                                metric=str(metric),
                                value=metric_value,
                            )
                units = item.get("memory_unit")
                if isinstance(units, list):
                    for unit_index, unit in enumerate(units):
                        if not isinstance(unit, dict):
                            continue
                        memory_path = unit.get("memory_path", unit_index)
                        for metric, metric_value in unit.items():
                            if metric == "memory_path":
                                continue
                            _add_metric_record(
                                records,
                                role=role,
                                capture_metric=capture_metric,
                                source_kind="visualize_json",
                                source_path=source_path,
                                record_type=record_type,
                                record_index=record_index,
                                core_id=core_id,
                                category=f"memory_path_{memory_path}",
                                metric=str(metric),
                                value=metric_value,
                                context={"memory_path": memory_path},
                            )

        tables = value.get("table_per_block")
        if isinstance(tables, list):
            for block in tables:
                if not isinstance(block, dict):
                    continue
                core_id = block.get("block_id")
                details = block.get("table_detail")
                if not isinstance(details, list):
                    continue
                for table in details:
                    if not isinstance(table, dict):
                        continue
                    table_name = str(table.get("table_name", "table"))
                    headers = table.get("header_name")
                    table_rows = table.get("row")
                    if not isinstance(headers, list) or not isinstance(
                        table_rows, list
                    ):
                        continue
                    value_headers = [str(item) for item in headers[1:]]
                    for table_row in table_rows:
                        if not isinstance(table_row, dict):
                            continue
                        row_name = str(table_row.get("name", "row"))
                        cells = table_row.get("value")
                        if not isinstance(cells, list):
                            continue
                        for header, cell in zip(value_headers, cells):
                            _add_metric_record(
                                records,
                                role=role,
                                capture_metric=capture_metric,
                                source_kind="visualize_json",
                                source_path=source_path,
                                record_type=record_type,
                                record_index=record_index,
                                core_id=core_id,
                                category=table_name,
                                metric=f"{row_name}.{header}",
                                value=cell,
                                unit=header,
                                context={
                                    "table_name": table_name,
                                    "row_name": row_name,
                                    "column": header,
                                },
                            )

        op_detail = value.get("op_detail")
        if isinstance(op_detail, list):
            for core in op_detail:
                if not isinstance(core, dict):
                    continue
                core_id = core.get("core_id")
                details = core.get("core_detail")
                if not isinstance(details, list):
                    continue
                for detail in details:
                    if not isinstance(detail, dict):
                        continue
                    metrics: dict[str, Any] = {}
                    for key, metric_value in detail.items():
                        if key in {"subcore_id", "subcore_type"}:
                            continue
                        canonical = _canonical_occupancy_metric(key)
                        metrics[canonical] = _to_number(metric_value)
                        _add_metric_record(
                            records,
                            role=role,
                            capture_metric=capture_metric,
                            source_kind="visualize_json",
                            source_path=source_path,
                            record_type=record_type,
                            record_index=record_index,
                            core_id=core_id,
                            subcore_id=detail.get("subcore_id"),
                            subcore_type=detail.get("subcore_type"),
                            category="occupancy",
                            metric=canonical,
                            value=metric_value,
                        )
                    occupancy_rows.append(
                        {
                            "role": role,
                            "capture_metric": capture_metric,
                            "source_path": source_path,
                            "device": value.get("soc", op_metadata.get("soc")),
                            "operator_name": op_metadata.get("name"),
                            "operator_core_type": value.get(
                                "op_type", op_metadata.get("op_type")
                            ),
                            "block_dim": op_metadata.get("block_dim"),
                            "core_id": core_id,
                            "subcore_id": detail.get("subcore_id"),
                            "subcore_type": detail.get("subcore_type"),
                            "duration_us": block_durations.get(str(core_id)),
                            "metrics": metrics,
                        }
                    )

    # The block-detail record normally precedes Occupancy.  Join again after
    # decoding all frames so record order is not a hidden requirement.
    for row in occupancy_rows:
        row["duration_us"] = block_durations.get(str(row.get("core_id")))
        row["operator_name"] = row.get("operator_name") or op_metadata.get("name")
        row["operator_core_type"] = (
            row.get("operator_core_type") or op_metadata.get("op_type")
        )
        row["block_dim"] = row.get("block_dim") or op_metadata.get("block_dim")
        row["device"] = row.get("device") or op_metadata.get("soc")
    return records, occupancy_rows, op_metadata


def _extract_csv_metrics(
    path: Path,
    rows: Sequence[Mapping[str, str]],
    *,
    role: str,
    capture_metric: str,
    raw_root: Path,
    source_kind: str = "metric_csv",
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    aliases = {
        "core_id",
        "core_no",
        "block_id",
        "aicore_id",
        "ai_core_id",
    }
    subcore_aliases = {"subcore_id", "sub_core_id"}
    type_aliases = {"subcore_type", "sub_core_type", "block_type", "core_type"}
    label_aliases = {
        "name",
        "metric",
        "metric_name",
        "item",
        "item_name",
        "memory_path",
        "type",
        "op_name",
    }
    source_path = _relative(path, raw_root)
    for row_index, row in enumerate(rows):
        normalized = {_normalize_name(key): value for key, value in row.items()}
        core_id = next(
            (
                normalized[key]
                for key in aliases
                if key in normalized and normalized[key] != ""
            ),
            None,
        )
        subcore_id = next(
            (
                normalized[key]
                for key in subcore_aliases
                if key in normalized and normalized[key] != ""
            ),
            None,
        )
        subcore_type = next(
            (
                normalized[key]
                for key in type_aliases
                if key in normalized and normalized[key] != ""
            ),
            None,
        )
        labels = [
            str(normalized[key])
            for key in label_aliases
            if key in normalized and normalized[key] not in {None, ""}
        ]
        category = path.stem
        if labels:
            category = f"{category}:{'|'.join(labels)}"
        context = {
            key: value
            for key, value in row.items()
            if _normalize_name(key) in aliases
            | subcore_aliases
            | type_aliases
            | label_aliases
        }
        for key, value in row.items():
            normalized_key = _normalize_name(key)
            if normalized_key in aliases | subcore_aliases | type_aliases:
                continue
            _add_metric_record(
                records,
                role=role,
                capture_metric=capture_metric,
                source_kind=source_kind,
                source_path=source_path,
                record_type="csv",
                record_index=row_index,
                core_id=core_id,
                subcore_id=subcore_id,
                subcore_type=subcore_type,
                category=category,
                metric=key,
                value=value,
                context=context,
            )
    return records


def _population_stats(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "population_variance": None,
            "population_stddev": None,
            "min": None,
            "max": None,
            "range": None,
            "relative_spread": None,
            "coefficient_of_variation": None,
            "max_min_ratio": None,
        }
    mean = statistics.fmean(values)
    variance = statistics.fmean((value - mean) ** 2 for value in values)
    stddev = math.sqrt(variance)
    minimum = min(values)
    maximum = max(values)
    return {
        "count": len(values),
        "mean": mean,
        "population_variance": variance,
        "population_stddev": stddev,
        "min": minimum,
        "max": maximum,
        "range": maximum - minimum,
        "relative_spread": (
            (maximum - minimum) / abs(mean) if mean != 0.0 else None
        ),
        "coefficient_of_variation": (
            stddev / abs(mean) if mean != 0.0 else None
        ),
        "max_min_ratio": maximum / minimum if minimum != 0.0 else None,
    }


def _sigmoid(value: float) -> float:
    if value >= 0:
        denominator = 1.0 + math.exp(-value)
        return 1.0 / denominator
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _apply_huawei_occupancy_scores(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply the exact msopprof 26.0.0 Occupancy imbalance calculation.

    Huawei source uses population variance.  Cycles, throughput, and SIMT
    instruction counts use sigmoid(z); cache hit rate uses sigmoid(-z).
    Scores >= 0.6 are named in advice.  A zero-variance metric is not scored.
    """

    metric_directions = {
        "cycles": "high",
        "throughput": "high",
        "l2_cache_hit_rate": "low",
        "simt_instructions": "high",
    }
    summaries: dict[str, Any] = {}
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row.get("role", "")),
                str(row.get("subcore_type", "")),
                str(row.get("source_path", "")),
            )
        ].append(row)

    for (role, subcore_type, source_path), group_rows in sorted(groups.items()):
        group_key = f"{role}:{subcore_type or '<unknown>'}:{source_path}"
        group_summary: dict[str, Any] = {}
        for metric, direction in metric_directions.items():
            metric_rows = [
                row
                for row in group_rows
                if _to_number(row.get("metrics", {}).get(metric)) is not None
            ]
            values = [
                float(row["metrics"][metric])
                for row in metric_rows
            ]
            stats = _population_stats(values)
            stats["missing_count"] = len(group_rows) - len(metric_rows)
            stats["direction"] = direction
            stats["score_threshold"] = HUAWEI_26_SCORE_THRESHOLD
            stats["z_threshold_magnitude"] = HUAWEI_26_Z_THRESHOLD
            stats["zero_variance_no_flags"] = bool(
                values
                and math.isclose(
                    float(stats["population_variance"]),
                    0.0,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )
            flagged_core_ids: list[Any] = []
            for row in group_rows:
                value = _to_number(row.get("metrics", {}).get(metric))
                row[f"{metric}_mean"] = stats["mean"]
                row[f"{metric}_population_stddev"] = stats[
                    "population_stddev"
                ]
                row[f"{metric}_z_score"] = None
                row[f"{metric}_sigmoid_score"] = None
                row[f"{metric}_huawei_26_flag"] = False
                if value is None or not values or stats["zero_variance_no_flags"]:
                    continue
                z_score = (
                    (float(value) - float(stats["mean"]))
                    / float(stats["population_stddev"])
                )
                directed_z = -z_score if direction == "low" else z_score
                score = _sigmoid(directed_z)
                flag = score >= HUAWEI_26_SCORE_THRESHOLD
                row[f"{metric}_z_score"] = z_score
                row[f"{metric}_sigmoid_score"] = score
                row[f"{metric}_huawei_26_flag"] = flag
                if flag:
                    flagged_core_ids.append(row.get("core_id"))
            stats["flagged_count"] = len(flagged_core_ids)
            stats["flagged_core_ids"] = flagged_core_ids
            group_summary[metric] = stats
        summaries[group_key] = group_summary
    return {
        "algorithm": {
            "source": HUAWEI_26_SOURCE,
            "source_commit": HUAWEI_26_SOURCE_COMMIT,
            "source_url": HUAWEI_26_SOURCE_URL,
            "mean": "sum(x) / N",
            "variance": "sum((x - mean)^2) / N",
            "z_score": "(x - mean) / sqrt(population_variance)",
            "normal_score": "sigmoid(z)",
            "cache_hit_score": "sigmoid(-z)",
            "flag": f"score >= {HUAWEI_26_SCORE_THRESHOLD}",
            "zero_variance": "no normalization and no flags",
            "guide_relative_spread": (
                "reported separately; it is not an extra gate in this "
                "26.0.0 source algorithm"
            ),
        },
        "groups": summaries,
    }


def _wide_core_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    known = {
        "cycles",
        "throughput",
        "l2_cache_hit_rate",
        "simt_instructions",
    }
    for row in rows:
        metrics = row.get("metrics", {})
        wide = {field: row.get(field) for field in CORE_METRIC_FIELDS}
        for metric in known:
            wide[metric] = metrics.get(metric) if isinstance(metrics, dict) else None
        other = (
            {key: value for key, value in metrics.items() if key not in known}
            if isinstance(metrics, dict)
            else {}
        )
        wide["other_metrics_json"] = json.dumps(
            other, sort_keys=True, ensure_ascii=False
        )
        output.append(wide)
    return output


def _active_mte_records(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    selected = []
    for row in records:
        text = " ".join(
            _normalize_name(row.get(key, ""))
            for key in ("category", "metric", "unit")
        )
        if (
            "mte" in text
            and "active" in text
            and ("bw" in text or "bandwidth" in text)
        ):
            selected.append(dict(row))
    return selected


def _summarize_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        label = f"{row.get('category', '')}.{row.get('metric', '')}"
        groups[label].append(row)
    output = {}
    for label, group in sorted(groups.items()):
        values = [
            float(value)
            for row in group
            if (value := _to_number(row.get("numeric_value"))) is not None
        ]
        stats = _population_stats(values)
        stats["records"] = len(group)
        stats["missing_count"] = sum(
            bool(row.get("is_missing")) for row in group
        )
        stats["zero_count"] = sum(value == 0 for value in values)
        stats["source_paths"] = sorted(
            {str(row.get("source_path", "")) for row in group}
        )
        output[label] = stats
    return output


def _reference_direct_contract(
    reference: Mapping[str, Any],
    direct_role: str,
) -> dict[str, Any]:
    production_roles = DIRECT_ROLE_MAP[direct_role]
    contracts = [reference["contracts"][role] for role in production_roles]
    comparable_fields = (
        "operator_type",
        "accelerator_core",
        "block_dim",
        "input_shapes",
        "input_formats",
        "input_dtypes",
        "output_shapes",
        "output_formats",
        "output_dtypes",
        "flops",
    )
    for field in comparable_fields:
        values = {json.dumps(contract[field], sort_keys=True) for contract in contracts}
        if len(values) != 1:
            raise AnalysisError(
                f"mapped reference roles {production_roles} disagree on {field}"
            )
    return {
        "direct_role": direct_role,
        "production_roles": list(production_roles),
        **{field: contracts[0][field] for field in comparable_fields},
        "duration_us_diagnostic_by_role": {
            contract["role"]: contract["duration_us_diagnostic"]
            for contract in contracts
        },
    }


def _check(
    checks: list[dict[str, Any]],
    errors: list[str],
    *,
    name: str,
    actual: Any,
    expected: Any,
    diagnostic: bool = False,
    note: str | None = None,
) -> bool:
    passed = actual == expected
    item = {
        "name": name,
        "actual": actual,
        "expected": expected,
        "passed": passed,
        "diagnostic_only": diagnostic,
    }
    if note:
        item["note"] = note
    checks.append(item)
    if not passed and not diagnostic:
        errors.append(f"{name}: expected {expected!r}, got {actual!r}")
    return passed


def _check_when_available(
    checks: list[dict[str, Any]],
    errors: list[str],
    *,
    name: str,
    actual: Any,
    expected: Any,
    note: str | None = None,
) -> bool | None:
    if actual is None:
        item: dict[str, Any] = {
            "name": name,
            "actual": None,
            "expected": expected,
            "passed": None,
            "available": False,
            "diagnostic_only": False,
            "note": (
                (note + "; " if note else "")
                + "field is unavailable in this capture; no value was imputed"
            ),
        }
        checks.append(item)
        return None
    return _check(
        checks,
        errors,
        name=name,
        actual=actual,
        expected=expected,
        note=note,
    )


def _validate_role(
    *,
    role: str,
    target: Mapping[str, Any],
    op_basic_rows: Sequence[Mapping[str, str]],
    op_metadata: Mapping[str, Any],
    occupancy_rows: Sequence[Mapping[str, Any]],
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    value = target["value"]
    spec = value.get("spec", {})
    observed = value.get("observed", {})
    ref = _reference_direct_contract(reference, role)
    if len(op_basic_rows) != 1:
        errors.append(
            f"expected exactly one OpBasicInfo row for {role}, "
            f"found {len(op_basic_rows)}"
        )
        op_basic = op_basic_rows[0] if op_basic_rows else {}
    else:
        op_basic = op_basic_rows[0]

    _check(
        checks,
        errors,
        name="target.status",
        actual=value.get("status"),
        expected="completed",
    )
    _check(
        checks,
        errors,
        name="target.spec.role",
        actual=spec.get("role"),
        expected=role,
    )
    _check(
        checks,
        errors,
        name="target.spec.production_roles",
        actual=list(spec.get("production_roles", [])),
        expected=list(DIRECT_ROLE_MAP[role]),
    )
    m, k, n = (spec.get("m"), spec.get("k"), spec.get("n"))
    expected_flops = 2 * int(m) * int(k) * int(n) if all(
        isinstance(item, int) for item in (m, k, n)
    ) else None
    _check(
        checks,
        errors,
        name="target.spec.flops_formula",
        actual=spec.get("flops"),
        expected=expected_flops,
    )
    _check(
        checks,
        errors,
        name="target.output_finite",
        actual=observed.get("output_finite"),
        expected=True,
    )
    expected_dtype = _normalize_dtype(spec.get("dtype"))
    _check(
        checks,
        errors,
        name="target.output_dtype",
        actual=_normalize_dtype(observed.get("output_dtype")),
        expected=expected_dtype,
    )
    _check(
        checks,
        errors,
        name="target.weight_format_code",
        actual=observed.get("weight_format_code"),
        expected=spec.get("weight_format_code_expected"),
    )
    _check(
        checks,
        errors,
        name="target.output_shape",
        actual=observed.get("output_shape"),
        expected=[m, n],
    )

    input_shapes = ref["input_shapes"]
    output_shapes = ref["output_shapes"]
    _check(
        checks,
        errors,
        name="reference.activation_shape",
        actual=list(input_shapes[0]) if input_shapes else None,
        expected=[m, k],
    )
    _check(
        checks,
        errors,
        name="reference.bias_shape",
        actual=list(input_shapes[-1]) if input_shapes else None,
        expected=[n],
    )
    _check(
        checks,
        errors,
        name="reference.output_shape",
        actual=list(output_shapes[0]) if output_shapes else None,
        expected=[m, n],
    )
    _check(
        checks,
        errors,
        name="reference.flops",
        actual=ref["flops"],
        expected=spec.get("flops"),
    )
    input_formats = list(ref["input_formats"])
    output_formats = list(ref["output_formats"])
    input_dtypes = list(ref["input_dtypes"])
    output_dtypes = list(ref["output_dtypes"])
    _check(
        checks,
        errors,
        name="reference.activation_format",
        actual=input_formats[0] if input_formats else None,
        expected=spec.get("activation_format_expected"),
    )
    _check(
        checks,
        errors,
        name="reference.weight_format",
        actual=input_formats[1] if len(input_formats) > 1 else None,
        expected=spec.get("weight_format_expected"),
    )
    _check(
        checks,
        errors,
        name="reference.output_format",
        actual=output_formats[0] if output_formats else None,
        expected=spec.get("activation_format_expected"),
    )
    _check(
        checks,
        errors,
        name="reference.input_dtypes",
        actual=input_dtypes,
        expected=[expected_dtype] * len(input_dtypes),
    )
    _check(
        checks,
        errors,
        name="reference.output_dtypes",
        actual=output_dtypes,
        expected=[expected_dtype] * len(output_dtypes),
    )

    op_name = op_basic.get("Op Name")
    operator_type = str(ref["operator_type"])
    _check(
        checks,
        errors,
        name="dispatch.operator_name_prefix",
        actual=bool(op_name and str(op_name).startswith(operator_type)),
        expected=True,
        note=f"reference operator type is {operator_type}",
    )
    op_block = _to_number(op_basic.get("Block Dim"))
    _check_when_available(
        checks,
        errors,
        name="dispatch.block_dim",
        actual=int(op_block) if op_block is not None else None,
        expected=ref["block_dim"],
    )
    if op_metadata.get("block_dim") not in {None, ""}:
        binary_block = _to_number(op_metadata.get("block_dim"))
        _check_when_available(
            checks,
            errors,
            name="visualize.block_dim",
            actual=int(binary_block) if binary_block is not None else None,
            expected=ref["block_dim"],
        )
    if op_metadata.get("name") not in {None, ""}:
        _check(
            checks,
            errors,
            name="visualize.operator_name",
            actual=op_metadata.get("name"),
            expected=op_name,
        )
    if op_metadata.get("op_type") not in {None, ""} and op_basic.get("Op Type"):
        _check(
            checks,
            errors,
            name="visualize.operator_core_type",
            actual=op_metadata.get("op_type"),
            expected=op_basic.get("Op Type"),
        )
    if occupancy_rows:
        distinct_cores = {
            str(row.get("core_id"))
            for row in occupancy_rows
            if row.get("core_id") is not None
        }
        _check_when_available(
            checks,
            errors,
            name="occupancy.distinct_core_count",
            actual=len(distinct_cores) if distinct_cores else None,
            expected=ref["block_dim"],
            note=(
                "checked only because this capture contains per-core "
                "Occupancy rows"
            ),
        )

    duration_diagnostics = {
        "msprof_op_basic_task_duration_us": _to_number(
            op_basic.get("Task Duration(us)")
        ),
        "msprof_visualize_duration_us": _to_number(op_metadata.get("duration")),
        "target_device_event_us": (
            float(observed["device_event_ms"]) * 1000.0
            if _to_number(observed.get("device_event_ms")) is not None
            else None
        ),
        "target_host_wall_us": (
            float(observed["host_wall_ms"]) * 1000.0
            if _to_number(observed.get("host_wall_ms")) is not None
            else None
        ),
        "full_graph_reference_us_by_production_role": ref[
            "duration_us_diagnostic_by_role"
        ],
        "interpretation": (
            "diagnostic only; msprof kernel replay, direct target event timing, "
            "and compiled full-graph timing have different execution contexts"
        ),
    }
    return {
        "status": "passed" if not errors else "failed",
        "role": role,
        "target_summary": str(target["path"]),
        "mapped_production_roles": list(DIRECT_ROLE_MAP[role]),
        "reference_contract": ref,
        "observed_dispatch": {
            "operator_name": op_name,
            "operator_core_type": op_basic.get("Op Type"),
            "block_dim": (
                int(op_block) if op_block is not None else None
            ),
            "mix_block_dim": op_basic.get("Mix Block Dim"),
            "device_id": op_basic.get("Device Id"),
            "pid": op_basic.get("Pid"),
            "current_freq": op_basic.get("Current Freq"),
            "rated_freq": op_basic.get("Rated Freq"),
        },
        "checks": checks,
        "errors": errors,
        "duration_diagnostics": duration_diagnostics,
    }


def _binary_schema_entry(
    path: Path,
    frames: Sequence[JsonFrame],
    raw_root: Path,
) -> dict[str, Any]:
    return {
        "path": _relative(path, raw_root),
        "size_bytes": path.stat().st_size,
        "framing": {
            "header_struct": "<QBBBB",
            "header_bytes": FRAME_HEADER.size,
            "header_fields": (
                "content_size, record_type, padding, version, reserved"
            ),
            "length_semantics": (
                "UTF-8 JSON bytes plus the declared trailing NUL padding"
            ),
            "record_boundary": "header_bytes + payload_length",
        },
        "record_count": len(frames),
        "records": [
            {
                "index": index,
                "offset": frame.offset,
                "payload_length": frame.payload_length,
                "record_type": frame.record_type,
                "record_type_hex": f"0x{frame.record_type:02x}",
                "declared_padding_bytes": frame.declared_padding,
                "version": frame.version,
                "reserved": frame.reserved,
                "observed_padding_bytes": frame.observed_padding,
                "top_level_type": type(frame.value).__name__,
                "top_level_keys": (
                    sorted(str(key) for key in frame.value)
                    if isinstance(frame.value, dict)
                    else []
                ),
                "json_schema": _schema_walk(frame.value),
            }
            for index, frame in enumerate(frames)
        ],
    }


def _role_raw_analysis(
    *,
    role: str,
    files: Sequence[Path],
    raw_root: Path,
    capture_metric: str,
) -> dict[str, Any]:
    op_paths = [path for path in files if path.name == "OpBasicInfo.csv"]
    visualize_paths = [path for path in files if path.name == "visualize_data.bin"]
    if len(op_paths) != 1:
        raise AnalysisError(
            f"{role}: expected exactly one recursive OpBasicInfo.csv, "
            f"found {len(op_paths)}: {op_paths}"
        )
    if len(visualize_paths) > 1:
        raise AnalysisError(
            f"{role}: multiple visualize_data.bin files are ambiguous: "
            f"{visualize_paths}"
        )

    op_headers, op_rows = _read_csv(op_paths[0])
    if not op_rows:
        raise AnalysisError(f"{role}: OpBasicInfo.csv has no rows: {op_paths[0]}")
    csv_schemas = [
        {
            "path": _relative(op_paths[0], raw_root),
            "headers": op_headers,
            "row_count": len(op_rows),
            "kind": "op_basic",
        }
    ]
    metric_records = _extract_csv_metrics(
        op_paths[0],
        op_rows,
        role=role,
        capture_metric=capture_metric,
        raw_root=raw_root,
        source_kind="op_basic_csv",
    )
    frames: list[JsonFrame] = []
    occupancy_rows: list[dict[str, Any]] = []
    op_metadata: dict[str, Any] = {}
    binary_schema = []
    if visualize_paths:
        frames = _decode_visualize_data(visualize_paths[0])
        binary_records, occupancy_rows, op_metadata = _extract_binary_metrics(
            frames,
            role=role,
            capture_metric=capture_metric,
            raw_root=raw_root,
        )
        metric_records.extend(binary_records)
        binary_schema.append(
            _binary_schema_entry(visualize_paths[0], frames, raw_root)
        )

    other_csv_paths = [
        path
        for path in files
        if path.suffix.lower() == ".csv" and path not in op_paths
    ]
    for path in other_csv_paths:
        headers, rows = _read_csv(path)
        csv_schemas.append(
            {
                "path": _relative(path, raw_root),
                "headers": headers,
                "row_count": len(rows),
                "kind": "metric",
            }
        )
        metric_records.extend(
            _extract_csv_metrics(
                path,
                rows,
                role=role,
                capture_metric=capture_metric,
                raw_root=raw_root,
            )
        )

    normalized_metric = _normalize_name(capture_metric)
    record_types = Counter(frame.record_type for frame in frames)
    metric_source_count = len(other_csv_paths) + len(frames)
    if normalized_metric == "occupancy":
        if not visualize_paths:
            raise AnalysisError(
                f"{role}: Occupancy was requested but visualize_data.bin is "
                "missing. This metric is product-limited; check msprof logs and "
                "the product support table instead of treating absence as zero."
            )
        if record_types[OCCUPANCY_RECORD_TYPE] == 0 or not occupancy_rows:
            raise AnalysisError(
                f"{role}: Occupancy was requested but record type "
                f"0x{OCCUPANCY_RECORD_TYPE:02x} with per-core op_detail is "
                "missing. The metric may be unsupported on the captured product."
            )
        for required_metric in (
            "cycles",
            "throughput",
            "l2_cache_hit_rate",
        ):
            missing_rows = [
                row.get("core_id")
                for row in occupancy_rows
                if _to_number(row.get("metrics", {}).get(required_metric))
                is None
            ]
            if missing_rows:
                raise AnalysisError(
                    f"{role}: Occupancy per-core metric {required_metric!r} "
                    f"is missing for cores {missing_rows}. Refusing a partial "
                    "metric result rather than imputing values."
                )
    elif normalized_metric == "memorydetail":
        if not other_csv_paths:
            raise AnalysisError(
                f"{role}: MemoryDetail was requested but no metric CSV exists "
                "beside OpBasicInfo.csv. This metric is product-limited; check "
                "msprof logs and product support rather than interpreting "
                "missing fields as zero."
            )
        active_mte = _active_mte_records(metric_records)
        if not active_mte:
            discovered = [
                {"path": item["path"], "headers": item["headers"]}
                for item in csv_schemas
                if item["kind"] == "metric"
            ]
            raise AnalysisError(
                f"{role}: MemoryDetail CSVs contain no recognizable active MTE "
                f"bandwidth fields; discovered schemas: {discovered}"
            )
        if not any(
            _to_number(row.get("numeric_value")) is not None
            for row in active_mte
        ):
            raise AnalysisError(
                f"{role}: MemoryDetail active MTE bandwidth fields are all "
                "missing/NA. This is an unavailable metric result, not zero; "
                "the feature may be unsupported or collection may have failed "
                "on the captured product."
            )
    elif normalized_metric != "unknown" and metric_source_count == 0:
        raise AnalysisError(
            f"{role}: requested metric {capture_metric!r} produced no metric "
            "CSV or decodable visualize_data.bin. It may be unsupported on "
            "the captured product."
        )

    return {
        "op_basic_path": op_paths[0],
        "op_basic_rows": op_rows,
        "visualize_paths": visualize_paths,
        "frames": frames,
        "occupancy_rows": occupancy_rows,
        "op_metadata": op_metadata,
        "metric_records": metric_records,
        "csv_schemas": csv_schemas,
        "binary_schema": binary_schema,
        "record_type_counts": {
            f"0x{key:02x}": count for key, count in sorted(record_types.items())
        },
        "other_files": [
            {
                "path": _relative(path, raw_root),
                "size_bytes": path.stat().st_size,
                "suffix": path.suffix.lower(),
            }
            for path in files
            if path not in op_paths
            and path not in visualize_paths
            and path not in other_csv_paths
        ],
    }


def _format_number(value: Any, digits: int = 3) -> str:
    number = _to_number(value)
    if number is None:
        return "missing"
    if isinstance(number, int):
        return f"{number:,}"
    return f"{float(number):,.{digits}f}"


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(str(value).replace("|", "\\|") for value in row)
            + " |"
        )
    return "\n".join(lines)


def _build_report(analysis: Mapping[str, Any]) -> str:
    lines = [
        "# Vision msprof-op analysis",
        "",
        f"**Status:** `{analysis['status']}`",
        "",
        (
            "This report validates the direct replay target against the "
            "normalized compiled full-graph dispatch. Replay duration is "
            "diagnostic only and is never used as a throughput gate."
        ),
        "",
        "## Inputs",
        "",
        f"- Capture metric: `{analysis['capture_metric']}`",
        f"- Capture directory: `{analysis['paths']['capture_dir']}`",
        f"- Raw directory: `{analysis['paths']['raw_dir']}`",
        f"- Reference directory: `{analysis['paths']['reference_dir']}`",
        "",
        "## Dispatch and reference validation",
        "",
    ]
    validation_rows = []
    for role, item in analysis["roles"].items():
        dispatch = item["validation"]["observed_dispatch"]
        reference = item["validation"]["reference_contract"]
        validation_rows.append(
            (
                role,
                ", ".join(item["validation"]["mapped_production_roles"]),
                dispatch.get("operator_name"),
                dispatch.get("operator_core_type"),
                dispatch.get("block_dim"),
                ";".join(reference["input_formats"]),
                ";".join(reference["output_formats"]),
                f"{reference['flops']:,}",
                item["validation"]["status"],
            )
        )
    lines.extend(
        [
            _markdown_table(
                (
                    "direct",
                    "production roles",
                    "captured op",
                    "core type",
                    "Block Dim",
                    "input formats",
                    "output formats",
                    "FLOPs",
                    "verdict",
                ),
                validation_rows,
            ),
            "",
            "## Duration diagnostics",
            "",
            (
                "These numbers come from different execution contexts. They "
                "are retained for investigation, not compared as a correctness "
                "or representativeness threshold."
            ),
            "",
        ]
    )
    duration_rows = []
    for role, item in analysis["roles"].items():
        diagnostic = item["validation"]["duration_diagnostics"]
        ref_means = {
            name: values["mean"]
            for name, values in diagnostic[
                "full_graph_reference_us_by_production_role"
            ].items()
        }
        duration_rows.append(
            (
                role,
                _format_number(
                    diagnostic["msprof_op_basic_task_duration_us"]
                ),
                _format_number(diagnostic["msprof_visualize_duration_us"]),
                _format_number(diagnostic["target_device_event_us"]),
                ", ".join(
                    f"{name}={_format_number(value)}"
                    for name, value in ref_means.items()
                ),
            )
        )
    lines.extend(
        [
            _markdown_table(
                (
                    "role",
                    "OpBasic us",
                    "visualize us",
                    "direct event us",
                    "compiled graph reference mean us",
                ),
                duration_rows,
            ),
            "",
        ]
    )

    occupancy = analysis.get("occupancy")
    if occupancy and occupancy.get("row_count"):
        lines.extend(
            [
                "## Per-core Occupancy",
                "",
                (
                    "Flags reproduce Huawei msopprof 26.0.0: population "
                    "z-score, sigmoid, score >= 0.6; cache hit rate reflects "
                    "the z-score so low hit rate is adverse. The separate "
                    "max/min relative spread is descriptive only. Occupancy "
                    "`throughput` is Huawei's per-core workload quantity; it "
                    "is not FLOP/s or GB/s."
                ),
                "",
            ]
        )
        occupancy_rows = []
        for group, metrics in occupancy["scores"]["groups"].items():
            for metric, stats in metrics.items():
                occupancy_rows.append(
                    (
                        group,
                        metric,
                        stats["count"],
                        _format_number(stats["mean"]),
                        _format_number(stats["population_stddev"]),
                        _format_number(stats["relative_spread"]),
                        _format_number(stats["coefficient_of_variation"]),
                        _format_number(stats["max_min_ratio"]),
                        ",".join(str(value) for value in stats["flagged_core_ids"])
                        or "none",
                    )
                )
        lines.extend(
            [
                _markdown_table(
                    (
                        "group",
                        "metric",
                        "N",
                        "mean",
                        "pop std",
                        "relative spread",
                        "CV",
                        "max/min",
                        "flagged cores",
                    ),
                    occupancy_rows,
                ),
                "",
            ]
        )

    active_mte = analysis.get("active_mte_bandwidth", {})
    if active_mte.get("record_count"):
        lines.extend(
            [
                "## Active MTE bandwidth",
                "",
                (
                    "These are active-cycle bandwidths, not whole-kernel or "
                    "whole-card averages. Missing values and numeric zero are "
                    "counted separately; a missing counter is never silently "
                    "converted to zero."
                ),
                "",
                _markdown_table(
                    (
                        "metric",
                        "records",
                        "missing",
                        "zeros",
                        "mean",
                        "min",
                        "max",
                    ),
                    (
                        (
                            name,
                            stats["records"],
                            stats["missing_count"],
                            stats["zero_count"],
                            _format_number(stats["mean"]),
                            _format_number(stats["min"]),
                            _format_number(stats["max"]),
                        )
                        for name, stats in active_mte["summary"].items()
                    ),
                ),
                "",
            ]
        )

    if analysis["errors"]:
        lines.extend(
            [
                "## Errors",
                "",
                *[f"- {error}" for error in analysis["errors"]],
                "",
            ]
        )
    if analysis["warnings"]:
        lines.extend(
            [
                "## Warnings",
                "",
                *[f"- {warning}" for warning in analysis["warnings"]],
                "",
            ]
        )
    lines.extend(
        [
            "## Machine-readable outputs",
            "",
            "- `analysis.json` — validation, statistics, and diagnostics",
            "- `core_metrics.csv` — flattened per-core Occupancy values and flags",
            "- `metric_records.csv` — normalized long-form metrics",
            "- `schema_manifest.json` — recursive CSV and binary schema inventory",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    capture_root = _resolve_input(args.capture_dir, "capture directory")
    raw_root = _resolve_input(args.raw_dir, "raw directory")
    reference_root = _resolve_input(args.reference_dir, "reference directory")
    output_root = _prepare_output(
        args.output_dir, (capture_root, raw_root, reference_root)
    )

    suite_path, suite_manifest = _find_single_manifest(capture_root)
    targets = _load_targets(capture_root)
    roles = sorted(targets)
    capture_metric = _capture_metric(suite_manifest, capture_root)
    reference = _load_reference(reference_root)
    grouped_files, unassigned_files = _group_raw_files(raw_root, roles)
    if unassigned_files:
        # Top-level ancillary files are retained in the schema inventory.  A
        # role-bearing metric file may not be silently ignored in a multi-role
        # capture, though.
        unassigned_metric_files = [
            path
            for path in unassigned_files
            if path.name in {"OpBasicInfo.csv", "visualize_data.bin"}
            or path.suffix.lower() == ".csv"
        ]
        if len(roles) > 1 and unassigned_metric_files:
            raise AnalysisError(
                "multi-role raw capture has metric artifacts outside a role "
                "directory: " + ", ".join(str(path) for path in unassigned_metric_files)
            )

    role_results: dict[str, Any] = {}
    all_metric_records: list[dict[str, Any]] = []
    all_occupancy_rows: list[dict[str, Any]] = []
    csv_schemas = []
    binary_schemas = []
    errors: list[str] = []
    warnings: list[str] = []

    for role in roles:
        raw = _role_raw_analysis(
            role=role,
            files=grouped_files[role],
            raw_root=raw_root,
            capture_metric=capture_metric,
        )
        validation = _validate_role(
            role=role,
            target=targets[role],
            op_basic_rows=raw["op_basic_rows"],
            op_metadata=raw["op_metadata"],
            occupancy_rows=raw["occupancy_rows"],
            reference=reference,
        )
        errors.extend(f"{role}: {message}" for message in validation["errors"])
        all_metric_records.extend(raw["metric_records"])
        all_occupancy_rows.extend(raw["occupancy_rows"])
        csv_schemas.extend(
            [{"role": role, **item} for item in raw["csv_schemas"]]
        )
        binary_schemas.extend(
            [{"role": role, **item} for item in raw["binary_schema"]]
        )
        role_results[role] = {
            "validation": validation,
            "raw": {
                "op_basic_path": _relative(raw["op_basic_path"], raw_root),
                "visualize_paths": [
                    _relative(path, raw_root) for path in raw["visualize_paths"]
                ],
                "record_type_counts": raw["record_type_counts"],
                "metric_record_count": len(raw["metric_records"]),
                "occupancy_core_row_count": len(raw["occupancy_rows"]),
                "other_files": raw["other_files"],
            },
        }

    occupancy_scores = _apply_huawei_occupancy_scores(all_occupancy_rows)
    core_rows = _wide_core_rows(all_occupancy_rows)
    active_mte = _active_mte_records(all_metric_records)
    active_mte_summary = _summarize_records(active_mte)

    normalized_metric = _normalize_name(capture_metric)
    if normalized_metric == "occupancy" and not all_occupancy_rows:
        errors.append(
            "Occupancy was requested but no per-core rows survived normalization"
        )
    if normalized_metric == "memorydetail" and not active_mte:
        errors.append(
            "MemoryDetail was requested but no active MTE bandwidth records "
            "survived normalization"
        )
    if normalized_metric == "memorydetail" and active_mte and not any(
        _to_number(row.get("numeric_value")) is not None for row in active_mte
    ):
        errors.append(
            "MemoryDetail active MTE bandwidth is entirely unavailable/NA; "
            "missing values were not converted to zero"
        )
    if normalized_metric in {"occupancy", "memorydetail"}:
        warnings.append(
            f"{capture_metric} is product-limited in Huawei documentation; "
            "absence on an unsupported product is reported as unavailable, "
            "never interpreted as zero."
        )
    if suite_manifest and suite_manifest.get("status") == "failed":
        errors.append("suite_manifest status is failed")
    if suite_manifest and suite_manifest.get("status") not in {
        None,
        "captured_unvalidated",
        "completed",
        "analyzed",
    }:
        warnings.append(
            f"suite_manifest status is {suite_manifest.get('status')!r}"
        )

    analysis = {
        "schema_version": 1,
        "status": "passed" if not errors else "failed",
        "purpose": (
            "portable, fail-closed validation and normalization of direct "
            "vision Linear msprof-op captures against a compiled full graph"
        ),
        "capture_metric": capture_metric,
        "paths": {
            "capture_dir": str(capture_root),
            "raw_dir": str(raw_root),
            "reference_dir": str(reference_root),
            "output_dir": str(output_root),
            "suite_manifest": str(suite_path) if suite_path else None,
        },
        "capture": {
            "suite_status": (
                suite_manifest.get("status") if suite_manifest else None
            ),
            "suite_roles": (
                suite_manifest.get("roles") if suite_manifest else None
            ),
            "target_roles": roles,
        },
        "reference": {
            "csv_paths": [
                _relative(path, reference_root) for path in reference["paths"]
            ],
            "row_count": reference["row_count"],
            "required_production_roles": list(REFERENCE_ROLES),
        },
        "roles": role_results,
        "occupancy": {
            "row_count": len(core_rows),
            "scores": occupancy_scores,
            "metric_semantics": {
                "throughput": (
                    "source-defined per-core workload quantity; do not "
                    "interpret as FLOP/s or bandwidth"
                ),
                "cycles": "source per-core cycle counter",
                "l2_cache_hit_rate": "source per-core percentage",
            },
        },
        "active_mte_bandwidth": {
            "record_count": len(active_mte),
            "missing_count": sum(
                bool(row.get("is_missing")) for row in active_mte
            ),
            "zero_count": sum(
                _to_number(row.get("numeric_value")) == 0
                for row in active_mte
                if _to_number(row.get("numeric_value")) is not None
            ),
            "summary": active_mte_summary,
            "interpretation": (
                "active-cycle bandwidth for the named MTE path; not a "
                "whole-kernel average or whole-card HBM bandwidth"
            ),
        },
        "metric_records": {
            "count": len(all_metric_records),
            "source_kind_counts": dict(
                sorted(
                    Counter(
                        str(row.get("source_kind")) for row in all_metric_records
                    ).items()
                )
            ),
            "missing_count": sum(
                bool(row.get("is_missing")) for row in all_metric_records
            ),
            "numeric_zero_count": sum(
                _to_number(row.get("numeric_value")) == 0
                for row in all_metric_records
                if _to_number(row.get("numeric_value")) is not None
            ),
        },
        "duration_policy": {
            "diagnostic_only": True,
            "reason": (
                "kernel replay, direct target event timing, and compiled "
                "full-graph execution are different contexts"
            ),
            "used_as_validation_gate": False,
        },
        "warnings": warnings,
        "errors": errors,
    }
    schema_manifest = {
        "schema_version": 1,
        "parser": {
            "implementation": str(Path(__file__).resolve()),
            "stdlib_only": True,
            "device_hardcoded": False,
            "cann_path_hardcoded": False,
            "block_dim_hardcoded": False,
            "binary_framing": "<QBBBB followed by payload_length bytes",
            "all_json_blocks_decoded": True,
        },
        "capture_json": [
            {
                "path": _relative(path, capture_root),
                "size_bytes": path.stat().st_size,
                "top_level_keys": (
                    sorted(_read_json(path))
                    if isinstance(_read_json(path), dict)
                    else []
                ),
            }
            for path in sorted(capture_root.rglob("*.json"))
        ],
        "raw_inventory": [
            {
                "path": _relative(path, raw_root),
                "size_bytes": path.stat().st_size,
                "suffix": path.suffix.lower(),
            }
            for path in sorted(raw_root.rglob("*"))
            if path.is_file()
        ],
        "csv_files": csv_schemas,
        "visualize_data_files": binary_schemas,
        "reference_csv_files": reference["schemas"],
        "unassigned_raw_files": [
            {
                "path": _relative(path, raw_root),
                "size_bytes": path.stat().st_size,
            }
            for path in unassigned_files
        ],
        "output_schemas": {
            "core_metrics.csv": list(CORE_METRIC_FIELDS),
            "metric_records.csv": list(METRIC_RECORD_FIELDS),
        },
    }
    return analysis, schema_manifest, core_rows, all_metric_records


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    try:
        analysis, schema_manifest, core_rows, metric_records = analyze(args)
    except AnalysisError as exc:
        # If the output directory can safely be created, leave a compact,
        # machine-readable failure instead of forcing the operator to recover
        # the exception only from terminal scrollback.
        try:
            if not output.exists():
                output.mkdir(parents=True, exist_ok=True)
            if output.is_dir() and not any(output.iterdir()):
                failure = {
                    "schema_version": 1,
                    "status": "failed",
                    "errors": [str(exc)],
                    "paths": {
                        "capture_dir": str(args.capture_dir.expanduser().resolve()),
                        "raw_dir": str(args.raw_dir.expanduser().resolve()),
                        "reference_dir": str(
                            args.reference_dir.expanduser().resolve()
                        ),
                        "output_dir": str(output),
                    },
                }
                _write_json(output / "analysis.json", failure)
                _write_json(
                    output / "schema_manifest.json",
                    {
                        "schema_version": 1,
                        "status": "failed_before_schema_completion",
                        "error": str(exc),
                        "parser": {
                            "implementation": str(Path(__file__).resolve()),
                            "stdlib_only": True,
                            "binary_framing": (
                                "<QBBBB followed by payload_length bytes"
                            ),
                        },
                    },
                )
                _write_csv(
                    output / "core_metrics.csv", CORE_METRIC_FIELDS, ()
                )
                _write_csv(
                    output / "metric_records.csv", METRIC_RECORD_FIELDS, ()
                )
                (output / "report.md").write_text(
                    "# Vision msprof-op analysis\n\n"
                    "**Status:** `failed`\n\n"
                    f"- {exc}\n",
                    encoding="utf-8",
                )
        except OSError:
            pass
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    _write_json(output / "analysis.json", analysis)
    _write_json(output / "schema_manifest.json", schema_manifest)
    _write_csv(output / "core_metrics.csv", CORE_METRIC_FIELDS, core_rows)
    _write_csv(
        output / "metric_records.csv", METRIC_RECORD_FIELDS, metric_records
    )
    (output / "report.md").write_text(
        _build_report(analysis), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": analysis["status"],
                "capture_metric": analysis["capture_metric"],
                "roles": sorted(analysis["roles"]),
                "core_metric_rows": len(core_rows),
                "metric_record_rows": len(metric_records),
                "output_dir": str(output),
                "errors": analysis["errors"],
            },
            indent=2,
        )
    )
    return 0 if analysis["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
