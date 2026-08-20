#!/usr/bin/env python3
"""Compare one-shot B1 table decoding with the persistent decode scheduler."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import threading
from types import SimpleNamespace
from typing import Any
import sys
import time


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
REPO_ROOT = EXPERIMENT_ROOT.parent
DEFAULT_TARGETS = (
    REPO_ROOT
    / "tmp/09_persistent_page_engine/table_spec_full_d1e6d00/"
    "whole/whole/tables.jsonl"
)
DEFAULT_COMPACT_VOCAB = (
    EXPERIMENT_ROOT
    / "presets/table_compact_vocab/b1_verifier_topfreq_16384.json"
)
sys.path.insert(0, str(HERE))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("/workspace/datasets/OmniDocBench/images"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/workspace/models/PaddleOCR-VL-1.6"),
    )
    parser.add_argument(
        "--decode-vocab-token-ids",
        type=Path,
        default=DEFAULT_COMPACT_VOCAB,
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sample-count", type=int, default=32)
    parser.add_argument("--request-id", action="append", default=[])
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        help="Shuffle the selected request order with a fixed seed.",
    )
    parser.add_argument(
        "--arrival-mode",
        choices=("sequential", "queued"),
        default="sequential",
        help=(
            "Submit the next request after the prior response, or make the "
            "whole sample available immediately."
        ),
    )
    parser.add_argument(
        "--compare-one-shot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the current recognizer.run([request]) path before serve().",
    )
    parser.add_argument(
        "--reference-json",
        type=Path,
        help="Optional prior lab JSON whose native token IDs must match.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument(
        "--decode-optimization",
        default="combined_apply_complete_layer_prefetch1_rope_lut",
    )
    parser.add_argument("--vision-buckets", default="4096")
    parser.add_argument("--text-buckets", default="1152")
    parser.add_argument("--min-pixels", type=int, default=28224)
    parser.add_argument("--max-pixels", type=int, default=802816)
    parser.add_argument(
        "--decode-cache-dir",
        type=Path,
        default=REPO_ROOT / ".runtime_cache/09_persistent_page_engine_torchair",
    )
    parser.add_argument(
        "--vision-cache-dir",
        type=Path,
        default=(
            REPO_ROOT
            / ".runtime_cache/09_persistent_page_engine_vision_torchair"
        ),
    )
    parser.add_argument(
        "--text-cache-dir",
        type=Path,
        default=(
            REPO_ROOT
            / ".runtime_cache/09_persistent_page_engine_text_torchair"
        ),
    )
    return parser.parse_args()


def _distribution(values: list[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {}

    def percentile(fraction: float) -> float:
        index = min(
            len(ordered) - 1,
            max(0, int(fraction * len(ordered) - 1e-12)),
        )
        return ordered[index]

    return {
        "mean": sum(ordered) / len(ordered),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
    }


def _select_records(
    records: list[dict[str, Any]],
    request_ids: list[str],
    sample_count: int,
    *,
    target_tokens: Any,
) -> list[dict[str, Any]]:
    by_id = {str(record["request_id"]): record for record in records}
    if request_ids:
        missing = [request_id for request_id in request_ids if request_id not in by_id]
        if missing:
            raise KeyError(f"unknown request IDs: {missing}")
        return [by_id[request_id] for request_id in request_ids]
    if sample_count <= 0 or sample_count > len(records):
        raise ValueError("sample-count must be in [1, number of target tables]")
    ranked = sorted(
        records,
        key=lambda record: (
            len(target_tokens(record)),
            str(record["request_id"]),
        ),
    )
    if sample_count == 1:
        return [ranked[len(ranked) // 2]]
    indices = [
        round(index * (len(ranked) - 1) / (sample_count - 1))
        for index in range(sample_count)
    ]
    return [ranked[index] for index in indices]


class _ImmediateSource:
    def __init__(self, requests: list[Any]) -> None:
        self._requests = iter(requests)
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def pull(self, *, block: bool) -> Any | None:
        del block
        if self._closed:
            return None
        try:
            return next(self._requests)
        except StopIteration:
            self._closed = True
            return None


class _SequentialSource:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._pending: list[Any] = []
        self._closed = False

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed

    def submit(self, request: Any) -> None:
        with self._condition:
            if self._closed:
                raise RuntimeError("cannot submit to a closed source")
            self._pending.append(request)
            self._condition.notify()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def pull(self, *, block: bool) -> Any | None:
        with self._condition:
            while block and not self._pending and not self._closed:
                self._condition.wait()
            if self._pending:
                return self._pending.pop(0)
            return None


def _recognition_record(recognition: Any, completed_at: float) -> dict[str, Any]:
    return {
        "request_id": str(recognition.request_id),
        "token_ids": [int(value) for value in recognition.token_ids],
        "text": str(recognition.text),
        "stop_reason": str(recognition.stop_reason),
        "output_tokens": int(recognition.generated_tokens_including_eos),
        "input_tokens": int(recognition.input_tokens),
        "request_total_s": float(recognition.timing_s["request_total"]),
        "decode_calls": int(recognition.decode_calls_executed),
        "decode_slot_index": recognition.decode_slot_index,
        "completed_at_s": float(completed_at),
    }


def _reference_tokens(path: Path | None) -> dict[str, list[int]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    phase = payload.get("one_shot") or payload.get("persistent") or {}
    return {
        str(record["request_id"]): [int(value) for value in record["token_ids"]]
        for record in phase.get("records") or []
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.compare_one_shot and args.batch_size != 1:
        raise ValueError("--compare-one-shot requires --batch-size 1")
    if args.arrival_mode == "sequential" and args.batch_size != 1:
        raise ValueError("sequential arrival validation requires --batch-size 1")

    import torch
    import torch_npu  # noqa: F401
    import table_spec_decode_lab as fixed_lab

    torch.npu.config.allow_internal_format = True
    torch.npu.set_compile_mode(jit_compile=False)

    records = fixed_lab.read_jsonl(args.targets)
    selected = _select_records(
        records,
        list(args.request_id),
        args.sample_count,
        target_tokens=fixed_lab.target_tokens,
    )
    if args.shuffle_seed is not None:
        random.Random(args.shuffle_seed).shuffle(selected)
    selected_ids = [str(record["request_id"]) for record in selected]
    selected_set = set(selected_ids)
    warm_record = next(
        record for record in records if str(record["request_id"]) not in selected_set
    )

    recognizer_args = SimpleNamespace(
        model=args.model,
        decode_optimization=args.decode_optimization,
        decode_vocab_token_ids=args.decode_vocab_token_ids,
        cache_length=args.cache_length,
        max_new_tokens=args.max_new_tokens,
        token_selection="greedy",
        decode_cache_dir=args.decode_cache_dir,
        vision_cache_dir=args.vision_cache_dir,
        text_cache_dir=args.text_cache_dir,
        vision_buckets=args.vision_buckets,
        text_buckets=args.text_buckets,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        compact_uint8_preprocess=False,
        image_resize_backend="pillow",
    )
    recognizer = fixed_lab.build_recognizer(
        recognizer_args,
        batch_size=args.batch_size,
    )
    configuration = recognizer.configuration()
    print(
        f"READY batch_size={args.batch_size} samples={len(selected)} "
        f"decode={configuration['decode_optimization']}",
        flush=True,
    )

    def make_request(record: dict[str, Any]) -> Any:
        crop = fixed_lab.exact_target_crop(record, args.images_dir)
        return fixed_lab.request_for(record, crop, recognizer_args)

    selected_requests = [make_request(record) for record in selected]

    warm_requests = [make_request(warm_record) for _ in range(args.batch_size)]
    for index, request in enumerate(warm_requests):
        request_id = f"warm-{index}-{request.request_id}"
        warm_requests[index] = type(request)(
            request_id=request_id,
            crop=request.crop,
            prompt=request.prompt,
            skip_special_tokens=request.skip_special_tokens,
            min_pixels=request.min_pixels,
            max_pixels=request.max_pixels,
            source_crop_size=request.source_crop_size,
        )
    warm_emitted: list[Any] = []
    recognizer.run(
        warm_requests,
        schedule_id=f"warm:b{args.batch_size}",
        emit_result=warm_emitted.append,
    )
    if len(warm_emitted) != args.batch_size:
        raise RuntimeError(
            f"warmup emitted {len(warm_emitted)} of {args.batch_size} requests"
        )
    print(f"WARM batch_size={args.batch_size}", flush=True)

    one_shot_payload: dict[str, Any] | None = None
    one_shot_by_id: dict[str, list[int]] = {}
    if args.compare_one_shot:
        one_shot_records: list[dict[str, Any]] = []
        phase_started = time.perf_counter()
        for index, (record, request) in enumerate(
            zip(selected, selected_requests),
            start=1,
        ):
            emitted: list[Any] = []
            call_started = time.perf_counter()
            recognizer.run(
                [request],
                schedule_id=f"one-shot:{record['request_id']}",
                emit_result=emitted.append,
            )
            call_finished = time.perf_counter()
            if len(emitted) != 1:
                raise RuntimeError(
                    f"{record['request_id']}: one-shot emitted {len(emitted)} results"
                )
            item = _recognition_record(emitted[0], call_finished - phase_started)
            item["call_wall_s"] = call_finished - call_started
            one_shot_records.append(item)
            one_shot_by_id[item["request_id"]] = item["token_ids"]
            print(
                f"ONE_SHOT {index}/{len(selected)} id={item['request_id']} "
                f"wall={item['call_wall_s']:.4f}s tokens={item['output_tokens']}",
                flush=True,
            )
        one_shot_wall_s = time.perf_counter() - phase_started
        one_shot_payload = {
            "wall_s": one_shot_wall_s,
            "tables_per_s": len(one_shot_records) / one_shot_wall_s,
            "request_total_s": _distribution(
                [record["request_total_s"] for record in one_shot_records]
            ),
            "call_wall_s": _distribution(
                [record["call_wall_s"] for record in one_shot_records]
            ),
            "records": one_shot_records,
        }

    external_reference = _reference_tokens(args.reference_json)
    reference = one_shot_by_id or external_reference
    persistent_records: list[dict[str, Any]] = []
    submitted_at: dict[str, float] = {}
    phase_started = time.perf_counter()

    def emit_persistent(recognition: Any) -> None:
        completed = time.perf_counter() - phase_started
        item = _recognition_record(recognition, completed)
        item["service_wall_s"] = (
            time.perf_counter() - submitted_at[item["request_id"]]
        )
        persistent_records.append(item)
        print(
            f"PERSISTENT {len(persistent_records)}/{len(selected)} "
            f"id={item['request_id']} service={item['service_wall_s']:.4f}s "
            f"queued={completed:.4f}s tokens={item['output_tokens']}",
            flush=True,
        )

    errors: list[dict[str, str]] = []
    producer: threading.Thread | None = None
    if args.arrival_mode == "queued":
        for request in selected_requests:
            submitted_at[str(request.request_id)] = phase_started
        request_source: Any = _ImmediateSource(selected_requests)
    else:
        request_source = _SequentialSource()
        completions = {
            str(request.request_id): threading.Event()
            for request in selected_requests
        }

        def produce_sequentially() -> None:
            for request in selected_requests:
                request_id = str(request.request_id)
                submitted_at[request_id] = time.perf_counter()
                request_source.submit(request)
                completions[request_id].wait()
            request_source.close()

        producer = threading.Thread(
            target=produce_sequentially,
            name="table-b1-sequential-source",
        )
        producer.start()

    def emit_persistent_and_release(recognition: Any) -> None:
        emit_persistent(recognition)
        if args.arrival_mode == "sequential":
            completions[str(recognition.request_id)].set()

    def emit_error(request_id: str, exc: BaseException) -> None:
        errors.append(
            {
                "request_id": str(request_id),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        if args.arrival_mode == "sequential":
            completions[str(request_id)].set()

    schedule = recognizer.serve(
        request_source,
        schedule_id=f"persistent:b{args.batch_size}",
        emit_result=emit_persistent_and_release,
        on_request_error=emit_error,
    )
    if producer is not None:
        producer.join()
    persistent_wall_s = time.perf_counter() - phase_started
    if errors:
        raise RuntimeError(f"persistent source errors: {errors}")
    if len(persistent_records) != len(selected):
        raise RuntimeError(
            f"persistent emitted {len(persistent_records)} of {len(selected)} requests"
        )
    persistent_by_id = {
        record["request_id"]: record["token_ids"] for record in persistent_records
    }
    missing_reference = sorted(set(selected_ids) - set(reference)) if reference else []
    mismatched_ids = (
        sorted(
            request_id
            for request_id in selected_ids
            if request_id in reference
            and persistent_by_id[request_id] != reference[request_id]
        )
        if reference
        else []
    )
    payload = {
        "format": "table_b1_persistent_scheduler_lab_v1",
        "configuration": configuration,
        "batch_size": args.batch_size,
        "arrival_mode": args.arrival_mode,
        "shuffle_seed": args.shuffle_seed,
        "selected_request_ids": selected_ids,
        "warm_request_id": str(warm_record["request_id"]),
        "one_shot": one_shot_payload,
        "persistent": {
            "wall_s": persistent_wall_s,
            "tables_per_s": len(persistent_records) / persistent_wall_s,
            "request_total_s": _distribution(
                [record["request_total_s"] for record in persistent_records]
            ),
            "service_wall_s": _distribution(
                [record["service_wall_s"] for record in persistent_records]
            ),
            "queued_completion_s": _distribution(
                [record["completed_at_s"] for record in persistent_records]
            ),
            "records": persistent_records,
            "schedule": asdict(schedule),
        },
        "parity": {
            "reference": (
                "same_process_one_shot"
                if one_shot_by_id
                else (str(args.reference_json) if external_reference else None)
            ),
            "compared": len(selected_ids) - len(missing_reference),
            "token_identical": (
                len(selected_ids) - len(missing_reference) - len(mismatched_ids)
            ),
            "mismatched_request_ids": mismatched_ids,
            "missing_reference_request_ids": missing_reference,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"DONE batch_size={args.batch_size} wall={persistent_wall_s:.4f}s "
        f"tables_per_s={payload['persistent']['tables_per_s']:.3f} "
        f"token_identical={payload['parity']['token_identical']}/"
        f"{payload['parity']['compared']} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
