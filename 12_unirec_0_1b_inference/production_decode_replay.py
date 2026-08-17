#!/usr/bin/env python3
"""Replay real UniRec cross-KV through the unchanged production decoder.

The prefill artifact supplies real, variable-length CPU cross-KV rows.  This
runner excludes layout, crop preprocessing, vision, and text prefill, but keeps
the production fixed arena, row admission, input copies, sampled-token D2H,
EOS/length completion, slot refill, and compiled decoder graph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np


@dataclass
class ArtifactCrop:
    row: dict[str, Any]
    packed_cross_kv: np.ndarray
    source_length: int


@dataclass
class LoadedArtifact:
    directory: Path
    summary: dict[str, Any]
    storage: np.memmap
    crops: list[ArtifactCrop]
    skipped_rows: list[dict[str, Any]]
    prefault_s: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/workspace/models/unirec-0.1b"),
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", choices=("float16",), default="float16")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--self-cache-length", type=int, default=2048)
    parser.add_argument("--cross-cache-length", type=int, default=1320)
    parser.add_argument(
        "--max-length",
        type=int,
        help="Generation cap; defaults to --self-cache-length.",
    )
    parser.add_argument("--offset-crops", type=int, default=0)
    parser.add_argument(
        "--limit-crops",
        type=int,
        default=0,
        help="Zero replays every crop after --offset-crops.",
    )
    parser.add_argument(
        "--over-capacity",
        choices=("error", "skip"),
        default="error",
        help="How to handle artifact rows longer than --cross-cache-length.",
    )
    parser.add_argument(
        "--prefault-artifact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fault the mmap into host page cache before measured decode.",
    )
    parser.add_argument(
        "--verify-crc",
        action="store_true",
        help="Verify every selected row before decode; excluded from timing.",
    )
    parser.add_argument("--decode-warmup-passes", type=int, default=2)
    parser.add_argument("--decode-admission-prefetch-depth", type=int, default=0)
    parser.add_argument(
        "--compile-cache-dir",
        type=Path,
        default=Path(
            ".runtime_cache/12_unirec_0_1b_inference/opendoc_batched_decode"
        ),
    )
    parser.add_argument(
        "--reference-trace",
        type=Path,
        help="Optional production recognition_trace.jsonl for token parity.",
    )
    parser.add_argument(
        "--reference-run-summary",
        type=Path,
        help="Optional production run_summary.json for throughput comparison.",
    )
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument(
        "--step-trace-jsonl",
        type=Path,
        help=(
            "Optional per-iteration production decode trace. This adds only "
            "host timestamp and buffered JSONL overhead; use a separate clean "
            "run for headline throughput."
        ),
    )
    parser.add_argument(
        "--completion-trace-jsonl",
        type=Path,
        help=(
            "Optional compact per-crop token trace written after measured "
            "decode. This is outside the decode timing window."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "tmp/12_unirec_0_1b_inference/production_decode_replay/result.json"
        ),
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.self_cache_length < 1 or args.cross_cache_length < 1:
        parser.error("cache lengths must be positive")
    if args.max_length is None:
        args.max_length = args.self_cache_length
    if not 1 <= args.max_length <= args.self_cache_length:
        parser.error("--max-length must be in [1, --self-cache-length]")
    if args.offset_crops < 0 or args.limit_crops < 0:
        parser.error("crop offset and limit must be non-negative")
    if args.decode_warmup_passes < 1:
        parser.error("--decode-warmup-passes must be positive")
    if args.decode_admission_prefetch_depth < 0:
        parser.error("--decode-admission-prefetch-depth must be non-negative")
    if args.progress_every < 1:
        parser.error("--progress-every must be positive")
    return args


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def write_completion_trace(
    path: Path,
    completed: list[Any],
    *,
    eos_token_id: int,
    max_length: int,
) -> None:
    """Write compact token evidence without changing measured decode timing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("w", encoding="utf-8") as handle:
        for item in completed:
            token_ids = [int(token) for token in item.result["generated_ids"]]
            if token_ids and token_ids[-1] == eos_token_id:
                termination = "eos"
            elif len(token_ids) >= max_length:
                termination = "length_cap"
            else:
                termination = "other"
            payload = dict(item.payload or {})
            row = {
                "request_id": str(item.request_id),
                "page_index": payload.get("page_index"),
                "crop_index": payload.get("crop_index"),
                "label": payload.get("label"),
                "token_ids": token_ids,
                "generated_token_count": int(
                    item.result["generated_token_count"]
                ),
                "decode_generated_token_count": int(
                    item.result["decode_generated_token_count"]
                ),
                "termination": termination,
            }
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
    os.replace(partial, path)


def _percentile(values: list[int], quantile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def distribution(values: Iterable[int]) -> dict[str, Any]:
    resolved = [int(value) for value in values]
    if not resolved:
        return {"count": 0}
    return {
        "count": len(resolved),
        "sum": int(sum(resolved)),
        "min": int(min(resolved)),
        "mean": float(sum(resolved) / len(resolved)),
        "p50": _percentile(resolved, 50),
        "p90": _percentile(resolved, 90),
        "p95": _percentile(resolved, 95),
        "p99": _percentile(resolved, 99),
        "max": int(max(resolved)),
    }


def float_distribution(values: Iterable[float]) -> dict[str, Any]:
    resolved = np.asarray([float(value) for value in values], dtype=np.float64)
    if resolved.size == 0:
        return {"count": 0}
    return {
        "count": int(resolved.size),
        "sum": float(resolved.sum()),
        "min": float(resolved.min()),
        "mean": float(resolved.mean()),
        "p50": float(np.percentile(resolved, 50)),
        "p90": float(np.percentile(resolved, 90)),
        "p95": float(np.percentile(resolved, 95)),
        "p99": float(np.percentile(resolved, 99)),
        "max": float(resolved.max()),
    }


def _position_bucket(position: int | None) -> str:
    value = int(position or 0)
    lower = 0
    for upper in (32, 64, 128, 256, 512, 1024, 1536, 2048):
        if value <= upper:
            return f"{lower:04d}-{upper:04d}"
        lower = upper + 1
    return "2049+"


def _cross_length_bucket(length: int | None) -> str:
    value = int(length or 0)
    lower = 0
    for upper in (64, 128, 256, 512, 768, 1024, 1320):
        if value <= upper:
            return f"{lower:04d}-{upper:04d}"
        lower = upper + 1
    return "1321+"


def summarize_step_trace(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"enabled": False, "count": 0}
    timing_fields = (
        "input_build_s",
        "graph_submit_s",
        "token_select_d2h_wait_s",
        "decode_step_s",
        "scheduler_s",
        "iteration_wall_s",
    )

    def summarize_group(group: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "count": len(group),
            "active_count": float_distribution(
                float(row["active_count"]) for row in group
            ),
            **{
                field: float_distribution(float(row[field]) for row in group)
                for field in timing_fields
            },
        }

    by_position: dict[str, list[dict[str, Any]]] = {}
    by_cross_length: dict[str, list[dict[str, Any]]] = {}
    by_active: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_position.setdefault(
            _position_bucket(row.get("cache_position_max")), []
        ).append(row)
        by_cross_length.setdefault(
            _cross_length_bucket(row.get("cross_length_max")), []
        ).append(row)
        active_count = int(row["active_count"])
        active_bucket = (
            "001-032"
            if active_count <= 32
            else "033-064"
            if active_count <= 64
            else "065-096"
            if active_count <= 96
            else "097-128"
        )
        by_active.setdefault(active_bucket, []).append(row)
    return {
        "enabled": True,
        "count": len(rows),
        "overall": summarize_group(rows),
        "by_cache_position_max": {
            key: summarize_group(value)
            for key, value in sorted(by_position.items())
        },
        "by_active_count": {
            key: summarize_group(value)
            for key, value in sorted(by_active.items())
        },
        "by_cross_length_max": {
            key: summarize_group(value)
            for key, value in sorted(by_cross_length.items())
        },
        "slowest_decode_steps": sorted(
            rows,
            key=lambda row: float(row["decode_step_s"]),
            reverse=True,
        )[:20],
        "slowest_scheduler_steps": sorted(
            rows,
            key=lambda row: float(row["scheduler_s"]),
            reverse=True,
        )[:20],
        "interpretation": (
            "graph_submit_s is asynchronous host submission. "
            "token_select_d2h_wait_s contains the queued graph execution, "
            "argmax, token D2H, and the mandatory host synchronization."
        ),
    }


def load_artifact(
    artifact_dir: Path,
    *,
    cross_cache_length: int,
    offset_crops: int = 0,
    limit_crops: int = 0,
    over_capacity: str = "error",
    verify_crc: bool = False,
    prefault: bool = True,
) -> LoadedArtifact:
    directory = artifact_dir.expanduser().resolve()
    summary = json.loads((directory / "summary.json").read_text())
    if summary.get("format") != "unirec_cross_kv_v1":
        raise ValueError(f"unsupported prefill artifact: {summary.get('format')!r}")
    if summary.get("status") != "ok":
        raise ValueError(f"prefill artifact status is not ok: {summary.get('status')!r}")
    rows = read_jsonl(directory / "crops.jsonl")
    stop = None if limit_crops == 0 else offset_crops + limit_crops
    rows = rows[offset_crops:stop]
    if not rows:
        raise ValueError("selected artifact crop range is empty")
    data_names = {str(row["cross_kv"]["file"]) for row in rows}
    if len(data_names) != 1:
        raise ValueError(f"artifact crop rows use multiple data files: {data_names}")
    data_path = directory / next(iter(data_names))
    # Copy-on-write keeps the artifact immutable while exposing a writable
    # NumPy view. torch.from_numpy then avoids the non-writable-array warning
    # and the hidden fallback behavior that warning can accompany.
    storage = np.memmap(data_path, dtype=np.uint8, mode="c")
    prefault_s = 0.0
    if prefault:
        started = time.perf_counter()
        _ = int(storage[::4096].sum(dtype=np.uint64))
        prefault_s = time.perf_counter() - started

    selected: list[ArtifactCrop] = []
    skipped: list[dict[str, Any]] = []
    for row in rows:
        spec = row["cross_kv"]
        shape = tuple(int(value) for value in spec["shape"])
        dtype = np.dtype(spec["dtype"])
        source_length = int(spec["source_length"])
        offset = int(spec["offset"])
        nbytes = int(spec["nbytes"])
        expected_nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if len(shape) != 5 or dtype != np.dtype(np.float16):
            raise ValueError(
                f"unexpected cross-KV row {row.get('request_id')}: "
                f"shape={shape} dtype={dtype}"
            )
        if shape[1] != 1 or shape[-2] != source_length:
            raise ValueError(
                f"invalid cross-KV row shape for {row.get('request_id')}: {shape}"
            )
        if nbytes != expected_nbytes or offset < 0 or offset + nbytes > storage.nbytes:
            raise ValueError(
                f"invalid cross-KV byte range for {row.get('request_id')}"
            )
        metadata_length = int(
            row["prefill"]["actual_cross_attention_length"]
        )
        if metadata_length != source_length:
            raise ValueError(
                f"cross-KV metadata mismatch for {row.get('request_id')}: "
                f"{source_length} != {metadata_length}"
            )
        if source_length > cross_cache_length:
            skipped.append(row)
            continue
        array = np.ndarray(
            shape=shape,
            dtype=dtype,
            buffer=storage,
            offset=offset,
            order="C",
        )
        if not array.flags.c_contiguous or not array.flags.writeable:
            raise RuntimeError(
                f"cross-KV row is not a writable contiguous view: "
                f"{row.get('request_id')}"
            )
        if verify_crc:
            checksum = zlib.crc32(memoryview(array).cast("B")) & 0xFFFFFFFF
            if f"{checksum:08x}" != str(spec["crc32"]):
                raise RuntimeError(
                    f"cross-KV checksum mismatch for {row.get('request_id')}"
                )
        selected.append(
            ArtifactCrop(
                row=row,
                packed_cross_kv=array,
                source_length=source_length,
            )
        )
    if skipped and over_capacity == "error":
        maximum = max(int(row["cross_kv"]["source_length"]) for row in skipped)
        raise ValueError(
            f"{len(skipped)} selected crops exceed cross-KV capacity "
            f"{cross_cache_length}; maximum={maximum}. Use --over-capacity skip "
            "only for a deliberate capacity experiment."
        )
    if not selected:
        raise ValueError("no artifact crops fit the selected cross-KV capacity")
    return LoadedArtifact(
        directory=directory,
        summary=summary,
        storage=storage,
        crops=selected,
        skipped_rows=skipped,
        prefault_s=prefault_s,
    )


def load_reference_trace(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    rows = read_jsonl(path.expanduser().resolve())
    result = {str(row["request_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("reference trace contains duplicate request IDs")
    return result


def compare_completions(
    completed: list[Any],
    reference: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not reference:
        return {"enabled": False}
    compared = 0
    token_exact = 0
    length_exact = 0
    missing_reference: list[str] = []
    mismatches: list[dict[str, Any]] = []
    for item in completed:
        request_id = str(item.request_id)
        expected = reference.get(request_id)
        if expected is None:
            if len(missing_reference) < 20:
                missing_reference.append(request_id)
            continue
        actual_tokens = [int(token) for token in item.result["generated_ids"]]
        expected_tokens_raw = expected.get("token_ids")
        expected_tokens = (
            [int(token) for token in expected_tokens_raw]
            if expected_tokens_raw is not None
            else None
        )
        expected_length = (
            len(expected_tokens)
            if expected_tokens is not None
            else int(expected["generated_token_count"]) + 1
        )
        actual_digest = hashlib.sha256(
            json.dumps(actual_tokens, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        expected_digest = (
            hashlib.sha256(
                json.dumps(expected_tokens, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest()
            if expected_tokens is not None
            else str(expected["token_sha256"])
        )
        compared += 1
        if len(actual_tokens) == expected_length:
            length_exact += 1
        if actual_digest == expected_digest:
            token_exact += 1
        elif len(mismatches) < 20:
            mismatch = {
                "request_id": request_id,
                "actual_length": len(actual_tokens),
                "expected_length": expected_length,
                "actual_token_sha256": actual_digest,
                "expected_token_sha256": expected_digest,
                "first_actual_tokens": actual_tokens[:16],
            }
            if expected_tokens is not None:
                mismatch["first_expected_tokens"] = expected_tokens[:16]
            mismatches.append(mismatch)
    return {
        "enabled": True,
        "reference_rows": len(reference),
        "compared_rows": compared,
        "missing_reference_count": len(completed) - compared,
        "first_missing_reference_ids": missing_reference,
        "length_exact_count": length_exact,
        "length_exact_fraction": length_exact / compared if compared else None,
        "token_exact_count": token_exact,
        "token_exact_fraction": token_exact / compared if compared else None,
        "first_mismatches": mismatches,
    }


def _physical_devices() -> list[int]:
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    return [int(value) for value in visible.split(",") if value.strip().isdigit()]


def _project_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def warm_production_decode_graph(
    *,
    runner: Any,
    decoder: Any,
    batch_size: int,
    cross_cache_length: int,
    passes: int,
) -> dict[str, Any]:
    import torch

    from modeling_optimized_unirec import synchronize_device

    module, compile_meta = runner._compile_decode_module(
        backend="torchair",
        self_attention_backend="increfa_all",
        compile_dynamic=False,
        cross_cache_len=cross_cache_length,
        batch_size=batch_size,
    )
    arena = decoder._allocate_empty_arena()
    if arena.cross_attention_mask is None:
        raise RuntimeError("decode replay warmup has no cross-attention mask")
    decoder_input_ids, cache_position = decoder._allocate_decode_device_inputs(
        batch_size,
        runner.device,
    )
    with torch.inference_mode():
        arena.cross_attention_mask.zero_()
        decoder_input_ids.fill_(int(runner.config.decoder_start_token_id))
        cache_position.fill_(1)
    inputs = (
        decoder_input_ids,
        cache_position,
        0,
        arena.key_cache,
        arena.value_cache,
        arena.cross_key_cache,
        arena.cross_value_cache,
        arena.cross_attention_mask,
    )
    pass_s: list[float] = []
    with torch.inference_mode():
        for pass_index in range(passes):
            started = time.perf_counter()
            _ = module(*inputs)
            synchronize_device(runner.device)
            elapsed = time.perf_counter() - started
            pass_s.append(elapsed)
            print(
                "UNIREC_PRODUCTION_DECODE_REPLAY_WARMUP "
                f"pass={pass_index + 1}/{passes} wall_s={elapsed:.6f}",
                flush=True,
            )
    return {
        "passes": passes,
        "pass_wall_s": pass_s,
        "compile": compile_meta,
    }


def main() -> None:
    args = parse_args()
    physical_devices = _physical_devices()
    if 5 in physical_devices or 6 in physical_devices:
        raise RuntimeError("physical NPU 5 and NPU 6 are excluded")
    os.environ["UNIREC_STATIC_CACHE_LEN"] = str(args.self_cache_length)
    os.environ["UNIREC_STATIC_CROSS_CACHE_LEN"] = str(args.cross_cache_length)

    import torch
    import torch_npu  # noqa: F401

    from continuous_unirec import (
        ContinuousReadyItem,
        ContinuousUniRecDecoder,
        ContinuousWorkerPrefilledItem,
        production_decode_cache_parent,
    )
    from modeling_optimized_unirec import OptimizedUniRecRunner

    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    artifact_load_started = time.perf_counter()
    artifact = load_artifact(
        args.artifact_dir,
        cross_cache_length=args.cross_cache_length,
        offset_crops=args.offset_crops,
        limit_crops=args.limit_crops,
        over_capacity=args.over_capacity,
        verify_crc=args.verify_crc,
        prefault=args.prefault_artifact,
    )
    artifact_load_s = time.perf_counter() - artifact_load_started
    print(
        "UNIREC_PRODUCTION_DECODE_REPLAY_ARTIFACT "
        f"selected={len(artifact.crops)} skipped={len(artifact.skipped_rows)} "
        f"load_s={artifact_load_s:.3f} prefault_s={artifact.prefault_s:.3f}",
        flush=True,
    )

    model_started = time.perf_counter()
    decode_cache_parent = production_decode_cache_parent(
        args.compile_cache_dir
    )
    runner = OptimizedUniRecRunner(
        model_path=args.model_path.expanduser().resolve(),
        device=args.device,
        dtype=args.dtype,
        compile_cache_dir=decode_cache_parent,
    )
    processor_shape = tuple(int(value) for value in runner.processor.max_side)
    runner._static_cross_cache_len_by_processor_max_side[processor_shape] = (
        args.cross_cache_length
    )
    model_load_s = time.perf_counter() - model_started
    decoder = ContinuousUniRecDecoder(
        runner=runner,
        batch_size=args.batch_size,
        max_length=args.max_length,
        decode_mode="compiled_ifa",
        compile_backend="torchair",
        admission_prefetch_depth=args.decode_admission_prefetch_depth,
    )
    def source() -> Iterable[Any]:
        for crop in artifact.crops:
            metadata = crop.row["prefill"]
            prep = dict(metadata.get("prep") or {})
            prep.setdefault("image", crop.row.get("request_id", "artifact_crop"))
            prep.setdefault("prepare_total_s", 0.0)
            yield ContinuousReadyItem(
                request_id=str(crop.row["request_id"]),
                payload={
                    "label": crop.row.get("label"),
                    "page_index": crop.row.get("page_index"),
                    "crop_index": crop.row.get("crop_index"),
                },
                prefilled=ContinuousWorkerPrefilledItem(
                    packed_cross_kv=crop.packed_cross_kv,
                    prep=prep,
                    prefill_s=float(metadata.get("prefill_s", 0.0)),
                    actual_cross_attention_length=crop.source_length,
                    prefill_device_stage_s=metadata.get("prefill_device_stage_s"),
                    text_prefill_execution=str(
                        metadata.get("text_prefill_execution", "artifact_replay")
                    ),
                    text_prefill_real_source_tokens=int(
                        metadata.get("text_prefill_real_source_tokens", crop.source_length)
                    ),
                    text_prefill_physical_source_tokens=int(
                        metadata.get("text_prefill_physical_source_tokens", crop.source_length)
                    ),
                ),
            )

    completed: list[Any] = []
    step_trace_rows: list[dict[str, Any]] = []
    step_trace_handle = None
    step_trace_path = (
        args.step_trace_jsonl.expanduser().resolve()
        if args.step_trace_jsonl is not None
        else None
    )
    if step_trace_path is not None:
        step_trace_path.parent.mkdir(parents=True, exist_ok=True)
        step_trace_handle = step_trace_path.open("w", encoding="utf-8")

    def on_step(row: dict[str, Any]) -> None:
        step_trace_rows.append(row)
        if step_trace_handle is not None:
            step_trace_handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
            if int(row["iteration"]) % 100 == 0:
                step_trace_handle.flush()

    progress_started = time.perf_counter()

    def on_complete(item: Any) -> None:
        completed.append(item)
        count = len(completed)
        if count % args.progress_every == 0 or count == len(artifact.crops):
            print(
                "UNIREC_PRODUCTION_DECODE_REPLAY_PROGRESS "
                f"completed={count}/{len(artifact.crops)} "
                f"elapsed_s={time.perf_counter() - progress_started:.3f}",
                flush=True,
            )

    decode_wall_started = time.perf_counter()
    try:
        decode = decoder.run(
            source(),
            on_complete=on_complete,
            graph_warmup_passes=args.decode_warmup_passes,
            on_step=on_step if step_trace_path is not None else None,
        )
    finally:
        if step_trace_handle is not None:
            step_trace_handle.close()
    decode_wall_s = time.perf_counter() - decode_wall_started
    warmup = decode["production_graph_warmup"]
    decode_wall_excluding_warmup_s = decode_wall_s - float(warmup["wall_s"])
    completion_trace_path = (
        args.completion_trace_jsonl.expanduser().resolve()
        if args.completion_trace_jsonl is not None
        else None
    )
    if completion_trace_path is not None:
        write_completion_trace(
            completion_trace_path,
            completed,
            eos_token_id=int(runner.config.eos_token_id),
            max_length=int(args.max_length),
        )
    reference = load_reference_trace(args.reference_trace)
    validation = compare_completions(completed, reference)
    generated_lengths = [
        int(item.result["generated_token_count"]) for item in completed
    ]
    reference_comparison: dict[str, Any] | None = None
    if args.reference_run_summary is not None:
        reference_run = json.loads(
            args.reference_run_summary.expanduser().resolve().read_text()
        )
        reference_decode = reference_run["decode"]
        reference_comparison = {
            "path": str(args.reference_run_summary.expanduser().resolve()),
            "raw_tok_s": float(reference_decode["raw_decode_tokens_per_s"]),
            "effective_tok_s": float(
                reference_decode["effective_decode_tokens_per_s"]
            ),
            "raw_tok_s_ratio": (
                float(decode["raw_decode_tokens_per_s"])
                / float(reference_decode["raw_decode_tokens_per_s"])
            ),
            "effective_tok_s_ratio": (
                float(decode["effective_decode_tokens_per_s"])
                / float(reference_decode["effective_decode_tokens_per_s"])
            ),
        }

    source_lengths = [crop.source_length for crop in artifact.crops]
    skipped_lengths = [
        int(row["cross_kv"]["source_length"])
        for row in artifact.skipped_rows
    ]
    result = {
        "schema_version": 1,
        "kind": "unirec_production_decode_replay",
        "status": "ok",
        "scope": (
            "real cross-KV plus unchanged ContinuousUniRecDecoder; "
            "layout, recognition preprocessing, vision, and text prefill excluded"
        ),
        "project_commit": _project_commit(),
        "argv": [str(value) for value in sys.argv],
        "runtime": {
            "torch": str(torch.__version__),
            "torch_npu": str(torch_npu.__version__),
            "ascend_home_path": os.environ.get("ASCEND_HOME_PATH"),
        },
        "physical_devices": physical_devices,
        "config": {
            "model_path": str(args.model_path.expanduser().resolve()),
            "artifact_dir": str(artifact.directory),
            "device": args.device,
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "self_cache_length": args.self_cache_length,
            "cross_cache_length": args.cross_cache_length,
            "max_length": args.max_length,
            "decode_mode": "compiled_ifa",
            "self_attention_backend": "increfa_all",
            "compile_backend": "torchair",
            "graph_mode": "ge",
            "mask_mode": "per_step",
            "qkv_fused": False,
            "weights_nz": False,
            "prefetch_mode": "none",
            "ge_tuning": [],
            "decode_admission_prefetch_depth": args.decode_admission_prefetch_depth,
            "over_capacity": args.over_capacity,
            "step_trace_jsonl": (
                str(step_trace_path) if step_trace_path is not None else None
            ),
            "completion_trace_jsonl": (
                str(completion_trace_path)
                if completion_trace_path is not None
                else None
            ),
        },
        "setup_s": {
            "artifact_load_including_optional_prefault": artifact_load_s,
            "artifact_prefault": artifact.prefault_s,
            "model_load": model_load_s,
            "graph_warmup": float(warmup["wall_s"]),
        },
        "warmup": warmup,
        "workload": {
            "selected_crops": len(artifact.crops),
            "skipped_over_capacity_crops": len(artifact.skipped_rows),
            "source_length": distribution(source_lengths),
            "skipped_source_length": distribution(skipped_lengths),
            "source_length_histogram": dict(
                sorted(Counter(source_lengths).items())
            ),
            "generated_length": distribution(generated_lengths),
        },
        "decode_wall_s": decode_wall_s,
        "decode_wall_excluding_warmup_s": decode_wall_excluding_warmup_s,
        "decode": decode,
        "step_trace": summarize_step_trace(step_trace_rows),
        "slot_efficiency": (
            decode["effective_decode_tokens"] / decode["raw_decode_token_slots"]
            if decode["raw_decode_token_slots"]
            else None
        ),
        "validation": validation,
        "reference_throughput": reference_comparison,
    }
    output = args.output.expanduser().resolve()
    atomic_write_json(output, result)
    print(
        "UNIREC_PRODUCTION_DECODE_REPLAY_END "
        f"status=ok crops={len(artifact.crops)} "
        f"raw_tok_s={decode['raw_decode_tokens_per_s']:.3f} "
        f"effective_tok_s={decode['effective_decode_tokens_per_s']:.3f} "
        f"slot_efficiency={result['slot_efficiency']:.6f} "
        f"decode_s={decode['decode_s']:.6f} wall_s={decode_wall_s:.6f} "
        f"output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
