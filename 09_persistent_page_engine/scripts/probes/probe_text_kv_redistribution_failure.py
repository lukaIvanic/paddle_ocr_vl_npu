#!/usr/bin/env python3
"""Isolate packed-text graph versus KV-redistribution failures on NPU.

Integrated mode patches only the two packed-text runtime boundaries before
executing the real ``run_omnidocbench.py`` entrypoint with its original
arguments. Every diagnostic record is appended and fsynced so the last
pre-operation record survives an AICore process failure.

Replay mode reconstructs one recorded copy with fresh tensors. Run each replay
lane in a fresh process because an AICore exception can poison the NPU runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent.parent
PRODUCTION_RUNNER = EXPERIMENT_ROOT / "scripts" / "run_omnidocbench.py"
sys.path.insert(0, str(EXPERIMENT_ROOT))


class CrashSafeRecorder:
    """Append one complete JSON object and fsync it before returning."""

    def __init__(self, path: Path):
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(
            self.path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o644,
        )
        self._lock = threading.Lock()
        self._sequence = 0

    def record(self, event: str, **payload: Any) -> None:
        with self._lock:
            self._sequence += 1
            record = {
                "sequence": self._sequence,
                "timestamp_ns": time.time_ns(),
                "monotonic_ns": time.perf_counter_ns(),
                "pid": os.getpid(),
                "thread": threading.current_thread().name,
                "event": event,
                **payload,
            }
            encoded = (
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=str,
                )
                + "\n"
            ).encode("utf-8")
            written = 0
            while written < len(encoded):
                written += os.write(self._fd, encoded[written:])
            os.fsync(self._fd)

    def close(self) -> None:
        with self._lock:
            if self._fd >= 0:
                os.fsync(self._fd)
                os.close(self._fd)
                self._fd = -1


def _exception_record(exception: BaseException) -> dict[str, str]:
    return {
        "exception_type": type(exception).__name__,
        "exception": repr(exception),
        "traceback": traceback.format_exc(),
    }


def _storage_descriptor(tensor: Any) -> dict[str, Any]:
    element_size = int(tensor.element_size())
    shape = [int(value) for value in tensor.shape]
    strides = [int(value) for value in tensor.stride()]
    storage_offset = int(tensor.storage_offset())
    minimum_index = storage_offset
    maximum_index = storage_offset
    for size, stride in zip(shape, strides):
        extent = max(0, size - 1) * stride
        minimum_index += min(0, extent)
        maximum_index += max(0, extent)

    descriptor: dict[str, Any] = {
        "shape": shape,
        "strides": strides,
        "storage_offset": storage_offset,
        "minimum_storage_element": minimum_index,
        "maximum_storage_element": maximum_index,
        "numel": int(tensor.numel()),
        "element_size": element_size,
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "is_contiguous": bool(tensor.is_contiguous()),
        "data_ptr": int(tensor.data_ptr()),
    }
    try:
        storage = tensor.untyped_storage()
        storage_nbytes = int(storage.nbytes())
        storage_elements = storage_nbytes // element_size
        descriptor.update(
            {
                "storage_data_ptr": int(storage.data_ptr()),
                "storage_nbytes": storage_nbytes,
                "storage_elements": storage_elements,
                "view_within_storage": (
                    minimum_index >= 0
                    and (
                        tensor.numel() == 0
                        or maximum_index < storage_elements
                    )
                ),
            }
        )
    except BaseException as exception:
        descriptor["storage_error"] = repr(exception)
        descriptor["view_within_storage"] = None
    try:
        import torch_npu

        descriptor["npu_format"] = int(torch_npu.get_npu_format(tensor))
    except BaseException as exception:
        descriptor["npu_format"] = None
        descriptor["npu_format_error"] = repr(exception)
    return descriptor


def _cache_descriptors(cache: Any) -> list[dict[str, Any]]:
    layer_count = len(cache.key_caches)
    descriptors = []
    for flat_index, tensor in enumerate(cache.flat_tensors()):
        descriptors.append(
            {
                "flat_index": flat_index,
                "kind": "key" if flat_index < layer_count else "value",
                "layer": (
                    flat_index
                    if flat_index < layer_count
                    else flat_index - layer_count
                ),
                "tensor": _storage_descriptor(tensor),
            }
        )
    return descriptors


def _storage_key(descriptor: dict[str, Any]) -> int | None:
    pointer = descriptor.get("storage_data_ptr")
    return int(pointer) if pointer is not None else None


def _validate_storage_sets(
    scratch_descriptors: list[dict[str, Any]],
    destination_descriptors: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    scratch_keys = [
        _storage_key(item["tensor"]) for item in scratch_descriptors
    ]
    destination_keys = [
        _storage_key(item["tensor"])
        for cache in destination_descriptors
        for item in cache
    ]
    scratch_non_null = [key for key in scratch_keys if key is not None]
    destination_non_null = [
        key for key in destination_keys if key is not None
    ]
    duplicate_destinations = sorted(
        {
            key
            for key in destination_non_null
            if destination_non_null.count(key) > 1
        }
    )
    scratch_destination_aliases = sorted(
        set(scratch_non_null) & set(destination_non_null)
    )
    return {
        "scratch_storage_count": len(scratch_non_null),
        "scratch_storage_unique": (
            len(set(scratch_non_null)) == len(scratch_non_null)
        ),
        "destination_storage_count": len(destination_non_null),
        "destination_storage_unique": not duplicate_destinations,
        "duplicate_destination_storage_ptrs": duplicate_destinations,
        "scratch_destination_aliases": scratch_destination_aliases,
        "valid": (
            len(set(scratch_non_null)) == len(scratch_non_null)
            and not duplicate_destinations
            and not scratch_destination_aliases
        ),
    }


def _prepared_metadata(prepared: Any) -> dict[str, Any]:
    return {
        "prepared_id": id(prepared),
        "physical_seq_len": int(prepared.physical_seq_len),
        "real_seq_len": int(prepared.real_seq_len),
        "segment_lengths": [
            int(value) for value in prepared.segment_lengths
        ],
        "segment_offsets": [
            int(value) for value in prepared.segment_offsets
        ],
    }


def install_integrated_probe(
    recorder: CrashSafeRecorder,
    *,
    barriers: str,
) -> None:
    import torch

    from paddleocr_vl.model.text_packed_prefill import (
        PackedTextPrefillRuntime,
    )

    original_run_prepared = PackedTextPrefillRuntime.run_prepared
    graph_calls = 0
    prepared_to_graph: dict[int, int] = {}

    def run_prepared_with_barrier(
        self: Any,
        prepared: Any,
    ) -> Any:
        nonlocal graph_calls
        graph_calls += 1
        graph_call = graph_calls
        prepared_to_graph[id(prepared)] = graph_call
        scratch = self.scratch_caches[prepared.physical_seq_len]
        recorder.record(
            "packed_graph_before",
            graph_call=graph_call,
            prepared=_prepared_metadata(prepared),
            scratch=_cache_descriptors(scratch),
        )
        try:
            output = original_run_prepared(self, prepared)
            recorder.record(
                "packed_graph_enqueued",
                graph_call=graph_call,
                output=_storage_descriptor(output),
            )
        except BaseException as exception:
            recorder.record(
                "packed_graph_enqueue_failed",
                graph_call=graph_call,
                **_exception_record(exception),
            )
            raise
        try:
            torch.npu.synchronize()
            recorder.record(
                "packed_graph_sync_passed",
                graph_call=graph_call,
                scratch_after=_cache_descriptors(scratch),
            )
        except BaseException as exception:
            recorder.record(
                "packed_graph_sync_failed",
                graph_call=graph_call,
                **_exception_record(exception),
            )
            raise
        return output

    def redistribute_with_per_copy_barriers(
        self: Any,
        prepared: Any,
        destinations: list[Any],
    ) -> int:
        graph_call = prepared_to_graph.get(id(prepared))
        scratch = self.scratch_caches[prepared.physical_seq_len]
        scratch_descriptors = _cache_descriptors(scratch)
        destination_descriptors = [
            _cache_descriptors(destination) for destination in destinations
        ]
        storage_validation = _validate_storage_sets(
            scratch_descriptors,
            destination_descriptors,
        )
        recorder.record(
            "redistribution_before",
            graph_call=graph_call,
            prepared=_prepared_metadata(prepared),
            destination_count=len(destinations),
            destination_cache_lengths=[
                int(destination.cache_length)
                for destination in destinations
            ],
            storage_validation=storage_validation,
        )
        if len(destinations) != len(prepared.segment_lengths):
            raise ValueError("packed text cache destinations do not align")
        if not storage_validation["valid"]:
            recorder.record(
                "redistribution_storage_validation_failed",
                graph_call=graph_call,
                storage_validation=storage_validation,
            )
            raise RuntimeError(
                "packed text redistribution cache storage alias detected"
            )

        layer_count = len(scratch.key_caches)
        copied_bytes = 0
        copy_ordinal = 0
        for member_index, (destination, offset, length) in enumerate(
            zip(
                destinations,
                prepared.segment_offsets,
                prepared.segment_lengths,
            )
        ):
            offset = int(offset)
            length = int(length)
            member_invariants = {
                "offset_non_negative": offset >= 0,
                "length_positive": length > 0,
                "source_range_valid": (
                    offset + length <= int(scratch.cache_length)
                ),
                "destination_range_valid": (
                    length <= int(destination.cache_length)
                ),
            }
            if not all(member_invariants.values()):
                recorder.record(
                    "member_validation_failed",
                    graph_call=graph_call,
                    member_index=member_index,
                    offset=offset,
                    length=length,
                    invariants=member_invariants,
                )
                raise RuntimeError(
                    "packed text redistribution member bounds invalid"
                )

            for flat_index, (source_tensor, destination_tensor) in enumerate(
                zip(
                    scratch.flat_tensors(),
                    destination.flat_tensors(),
                )
            ):
                copy_ordinal += 1
                kind = "key" if flat_index < layer_count else "value"
                layer = (
                    flat_index
                    if flat_index < layer_count
                    else flat_index - layer_count
                )
                source = source_tensor[
                    :, :, offset : offset + length, :
                ]
                destination_view = destination_tensor[:, :, :length, :]
                source_descriptor = _storage_descriptor(source)
                destination_descriptor = _storage_descriptor(destination_view)
                shape_equal = tuple(source.shape) == tuple(
                    destination_view.shape
                )
                source_in_bounds = (
                    source_descriptor.get("view_within_storage") is True
                )
                destination_in_bounds = (
                    destination_descriptor.get("view_within_storage") is True
                )
                copy_id = (
                    f"graph{graph_call}:member{member_index}:"
                    f"{kind}:layer{layer}"
                )
                copy_metadata = {
                    "copy_id": copy_id,
                    "copy_ordinal": copy_ordinal,
                    "graph_call": graph_call,
                    "member_index": member_index,
                    "flat_index": flat_index,
                    "kind": kind,
                    "layer": layer,
                    "offset": offset,
                    "length": length,
                    "scratch_cache_length": int(scratch.cache_length),
                    "destination_cache_length": int(
                        destination.cache_length
                    ),
                    "source_base": _storage_descriptor(source_tensor),
                    "destination_base": _storage_descriptor(
                        destination_tensor
                    ),
                    "source_view": source_descriptor,
                    "destination_view": destination_descriptor,
                    "shape_equal": shape_equal,
                    "source_in_bounds": source_in_bounds,
                    "destination_in_bounds": destination_in_bounds,
                    "base_storage_alias": (
                        _storage_key(source_descriptor)
                        == _storage_key(destination_descriptor)
                    ),
                }
                recorder.record("copy_before", copy=copy_metadata)
                if not (
                    shape_equal
                    and source_in_bounds
                    and destination_in_bounds
                    and not copy_metadata["base_storage_alias"]
                ):
                    recorder.record(
                        "copy_validation_failed",
                        copy=copy_metadata,
                    )
                    raise RuntimeError(
                        f"packed KV copy metadata invalid: {copy_id}"
                    )
                try:
                    destination_view.copy_(source)
                    recorder.record("copy_enqueued", copy=copy_metadata)
                except BaseException as exception:
                    recorder.record(
                        "copy_enqueue_failed",
                        copy=copy_metadata,
                        **_exception_record(exception),
                    )
                    raise
                try:
                    torch.npu.synchronize()
                    recorder.record("copy_sync_passed", copy=copy_metadata)
                except BaseException as exception:
                    recorder.record(
                        "copy_sync_failed",
                        copy=copy_metadata,
                        **_exception_record(exception),
                    )
                    raise
                copied_bytes += int(source.numel()) * int(
                    source.element_size()
                )

        recorder.record(
            "redistribution_sync_passed",
            graph_call=graph_call,
            copy_count=copy_ordinal,
            copied_bytes=copied_bytes,
        )
        return copied_bytes

    PackedTextPrefillRuntime.run_prepared = run_prepared_with_barrier
    if barriers == "full":
        PackedTextPrefillRuntime.redistribute_cache = (
            redistribute_with_per_copy_barriers
        )
    elif barriers != "graph_only":
        raise ValueError(f"unknown integrated barrier strategy: {barriers}")


def run_integrated(
    args: argparse.Namespace,
    production_argv: Sequence[str],
) -> None:
    if not production_argv:
        raise ValueError(
            "integrated mode requires production runner arguments after --"
        )
    diagnostic_dir = args.diagnostic_dir.expanduser().resolve()
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    recorder = CrashSafeRecorder(diagnostic_dir / "events.jsonl")
    recorder.record(
        "probe_started",
        mode="integrated",
        integrated_barriers=args.integrated_barriers,
        probe_commit=os.environ.get("PROBE_GIT_COMMIT"),
        production_runner=str(PRODUCTION_RUNNER),
        production_argv=list(production_argv),
        python=sys.version,
    )
    install_integrated_probe(
        recorder,
        barriers=args.integrated_barriers,
    )
    original_argv = sys.argv[:]
    sys.argv = [str(PRODUCTION_RUNNER), *production_argv]
    try:
        runpy.run_path(str(PRODUCTION_RUNNER), run_name="__main__")
        recorder.record("production_run_passed")
    except BaseException as exception:
        recorder.record(
            "production_run_failed",
            **_exception_record(exception),
        )
        raise
    finally:
        sys.argv = original_argv
        recorder.close()


def _load_events(path: Path) -> list[dict[str, Any]]:
    events = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                events.append(json.loads(stripped))
            except json.JSONDecodeError as exception:
                raise RuntimeError(
                    f"invalid JSONL record at {path}:{line_number}"
                ) from exception
    return events


def _select_copy_records(
    events: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    before_by_id = {
        event["copy"]["copy_id"]: event
        for event in events
        if event.get("event") == "copy_before"
    }
    passed_ids = [
        event["copy"]["copy_id"]
        for event in events
        if event.get("event") == "copy_sync_passed"
    ]
    explicit_failures = [
        event
        for event in events
        if event.get("event") in {
            "copy_enqueue_failed",
            "copy_sync_failed",
            "copy_validation_failed",
        }
    ]
    if explicit_failures:
        candidate = explicit_failures[-1]
    else:
        unpassed = [
            event
            for event in events
            if event.get("event") == "copy_before"
            and event["copy"]["copy_id"] not in set(passed_ids)
        ]
        if not unpassed:
            graph_failure = next(
                (
                    event
                    for event in reversed(events)
                    if event.get("event")
                    in {
                        "packed_graph_enqueue_failed",
                        "packed_graph_sync_failed",
                    }
                ),
                None,
            )
            if graph_failure is not None:
                raise RuntimeError(
                    "the packed graph failed before redistribution; "
                    "there is no copy to replay"
                )
            raise RuntimeError("no failed or incomplete copy record found")
        candidate = unpassed[-1]
    candidate_before = before_by_id[candidate["copy"]["copy_id"]]
    candidate_sequence = int(candidate_before["sequence"])
    previous = [
        before_by_id[copy_id]
        for copy_id in passed_ids
        if copy_id in before_by_id
        and int(before_by_id[copy_id]["sequence"]) < candidate_sequence
    ]
    previous.sort(key=lambda event: int(event["sequence"]))
    return candidate_before, previous[-1] if previous else None


def _torch_dtype(label: str) -> Any:
    import torch

    mapping = {
        "torch.float16": torch.float16,
        "torch.bfloat16": torch.bfloat16,
        "torch.float32": torch.float32,
    }
    try:
        return mapping[label]
    except KeyError as exception:
        raise ValueError(f"unsupported replay dtype: {label}") from exception


def _allocate_like_descriptor(
    descriptor: dict[str, Any],
    *,
    device: Any,
    fill: float,
) -> Any:
    import torch
    import torch_npu

    tensor = torch.full(
        tuple(int(value) for value in descriptor["shape"]),
        fill,
        device=device,
        dtype=_torch_dtype(str(descriptor["dtype"])),
    )
    expected_format = descriptor.get("npu_format")
    if expected_format is not None:
        current_format = int(torch_npu.get_npu_format(tensor))
        if current_format != int(expected_format):
            tensor = torch_npu.npu_format_cast(
                tensor,
                int(expected_format),
            )
    return tensor


def _assert_replay_view(
    label: str,
    actual: Any,
    expected: dict[str, Any],
) -> None:
    actual_descriptor = _storage_descriptor(actual)
    keys = ["shape", "strides", "storage_offset", "dtype"]
    if expected.get("npu_format") is not None:
        keys.append("npu_format")
    for key in keys:
        if actual_descriptor.get(key) != expected.get(key):
            raise RuntimeError(
                f"{label} replay metadata differs for {key}: "
                f"actual={actual_descriptor.get(key)!r} "
                f"expected={expected.get(key)!r}"
            )


def _run_replay_lane(
    copy_record: dict[str, Any],
    lane: str,
    device: Any,
) -> dict[str, Any]:
    import torch

    source = _allocate_like_descriptor(
        copy_record["source_base"],
        device=device,
        fill=0.25,
    )
    destination = _allocate_like_descriptor(
        copy_record["destination_base"],
        device=device,
        fill=0.0,
    )
    offset = int(copy_record["offset"])
    length = int(copy_record["length"])
    source_view = source[:, :, offset : offset + length, :]
    destination_view = destination[:, :, :length, :]
    _assert_replay_view(
        "source",
        source_view,
        copy_record["source_view"],
    )
    _assert_replay_view(
        "destination",
        destination_view,
        copy_record["destination_view"],
    )
    torch.npu.synchronize()
    started_ns = time.perf_counter_ns()
    if lane in {"candidate_current", "neighbor_current"}:
        destination_view.copy_(source_view)
    elif lane == "candidate_per_head":
        for head in range(int(source.shape[1])):
            destination[
                :, head : head + 1, :length, :
            ].copy_(
                source[
                    :, head : head + 1, offset : offset + length, :
                ]
            )
    else:
        raise ValueError(f"unknown replay lane: {lane}")
    torch.npu.synchronize()
    elapsed_s = (time.perf_counter_ns() - started_ns) / 1_000_000_000
    actual = destination_view.cpu()
    expected = source_view.cpu()
    exact = bool(torch.equal(actual, expected))
    if not exact:
        raise RuntimeError(f"replay lane {lane} produced incorrect values")
    return {
        "lane": lane,
        "copy_id": copy_record["copy_id"],
        "elapsed_s": elapsed_s,
        "exact": exact,
        "source_view": _storage_descriptor(source_view),
        "destination_view": _storage_descriptor(destination_view),
    }


def run_replay(args: argparse.Namespace) -> None:
    import torch
    import torch_npu  # noqa: F401

    diagnostic_dir = args.diagnostic_dir.expanduser().resolve()
    trace_path = (
        args.trace.expanduser().resolve()
        if args.trace is not None
        else diagnostic_dir / "events.jsonl"
    )
    events = _load_events(trace_path)
    candidate, neighbor = _select_copy_records(events)
    selected = candidate
    if args.replay_lane == "neighbor_current":
        if neighbor is None:
            raise RuntimeError(
                "no passing copy precedes the failing copy in the trace"
            )
        selected = neighbor
    copy_record = selected["copy"]
    device = torch.device(args.device)
    torch.npu.set_device(device)
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else diagnostic_dir / f"replay_{args.replay_lane}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "mode": "replay",
        "trace_path": str(trace_path),
        "lane": args.replay_lane,
        "candidate_copy_id": candidate["copy"]["copy_id"],
        "selected_copy_id": copy_record["copy_id"],
        "status": "started",
    }
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    try:
        summary.update(
            {
                "status": "passed",
                "result": _run_replay_lane(
                    copy_record,
                    args.replay_lane,
                    device,
                ),
            }
        )
    except BaseException as exception:
        summary.update(
            {
                "status": "failed",
                **_exception_record(exception),
            }
        )
        output_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        raise
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def _split_arguments(
    argv: Sequence[str],
) -> tuple[list[str], list[str]]:
    values = list(argv)
    if "--" not in values:
        return values, []
    delimiter = values.index("--")
    return values[:delimiter], values[delimiter + 1 :]


def parse_args(
    argv: Sequence[str] | None = None,
) -> tuple[argparse.Namespace, list[str]]:
    probe_argv, production_argv = _split_arguments(
        sys.argv[1:] if argv is None else argv
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("integrated", "replay"),
    )
    parser.add_argument("--diagnostic-dir", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--integrated-barriers",
        choices=("graph_only", "full"),
        default="graph_only",
        help=(
            "graph_only proves the packed graph completed before leaving the "
            "production redistribution unchanged; full additionally logs and "
            "synchronizes every individual KV copy"
        ),
    )
    parser.add_argument(
        "--replay-lane",
        choices=(
            "candidate_current",
            "neighbor_current",
            "candidate_per_head",
        ),
        default="candidate_current",
    )
    args = parser.parse_args(probe_argv)
    if args.mode == "integrated" and not production_argv:
        parser.error("integrated mode requires production arguments after --")
    if args.mode == "replay" and production_argv:
        parser.error("replay mode does not accept arguments after --")
    return args, production_argv


def main() -> None:
    args, production_argv = parse_args()
    if args.mode == "integrated":
        run_integrated(args, production_argv)
    else:
        run_replay(args)


if __name__ == "__main__":
    main()
