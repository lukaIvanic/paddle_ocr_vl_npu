#!/usr/bin/env python3
"""Build an exact text-decode workload corpus from a recognition trace."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
REPO_ROOT = EXPERIMENT_ROOT.parent

DEFAULT_RUN = (
    REPO_ROOT
    / "tmp/09_persistent_page_engine"
    / "repeat_async_writer_b32_kv4096_256p_7899d40"
)
DEFAULT_TRACE = DEFAULT_RUN / "recognition_trace.jsonl"
DEFAULT_SUMMARY = DEFAULT_RUN / "run_summary.json"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "tmp/09_persistent_page_engine/text_decode_lab"
    / "corpus_256p_b32_kv4096_7899d40.json"
)
DEFAULT_CACHE_THRESHOLDS = (1024, 1536, 2048, 3072, 4096, 5120, 8192)


def _csv_ints(value: str) -> tuple[int, ...]:
    values = tuple(sorted({int(piece) for piece in value.split(",") if piece.strip()}))
    if not values or values[0] <= 0:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--run-summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--cache-thresholds",
        type=_csv_ints,
        default=DEFAULT_CACHE_THRESHOLDS,
        help="Cache capacities reported in the corpus coverage table.",
    )
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"expected an object at {path}:{line_number}")
            records.append(record)
    if not records:
        raise ValueError(f"trace contains no records: {path}")
    return records


def _percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _length_summary(values: list[int]) -> dict[str, float | int]:
    return {
        "min": min(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values),
    }


def build_corpus(
    trace_path: Path,
    summary_path: Path,
    *,
    cache_thresholds: tuple[int, ...],
) -> dict[str, Any]:
    trace_path = trace_path.expanduser().resolve()
    summary_path = summary_path.expanduser().resolve()
    records = _read_jsonl(trace_path)
    summary = _read_json(summary_path)
    try:
        recognition = dict(summary["recognition"])
        configuration = dict(summary["configuration"])
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "run summary lacks configuration/recognition sections"
        ) from exc

    items: list[dict[str, Any]] = []
    request_ids: set[str] = set()
    source_indices: set[int] = set()
    for source_line, record in enumerate(records, 1):
        try:
            source_index = int(record["global_request_index"])
            request_id = str(record["request_id"])
            token_ids = [int(value) for value in record["token_ids"]]
            prompt_tokens = int(record["input_tokens"])
            stop_reason = str(record["stop_reason"])
            generated_tokens = int(record["generated_tokens_including_eos"])
            effective_tokens = int(
                record["decode_tokens_after_prefill_including_eos"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"trace record {source_line} lacks required decode fields"
            ) from exc
        if source_index < 0 or source_index in source_indices:
            raise ValueError(f"duplicate/invalid global_request_index={source_index}")
        if not request_id or request_id in request_ids:
            raise ValueError(f"duplicate/empty request_id={request_id!r}")
        if prompt_tokens <= 0:
            raise ValueError(f"{request_id}: input_tokens must be positive")
        if not token_ids:
            raise ValueError(f"{request_id}: token_ids must not be empty")
        if generated_tokens != len(token_ids):
            raise ValueError(
                f"{request_id}: generated_tokens={generated_tokens} "
                f"but token_ids has {len(token_ids)} entries"
            )
        if effective_tokens != max(0, len(token_ids) - 1):
            raise ValueError(
                f"{request_id}: effective decode token accounting mismatch"
            )
        if stop_reason not in {"eos", "length"}:
            raise ValueError(f"{request_id}: unexpected stop_reason={stop_reason!r}")
        if (
            generated_tokens == 1
            and stop_reason == "length"
            and int(configuration["max_new_tokens"]) != 1
        ):
            raise ValueError(
                f"{request_id}: a one-token length stop requires max_new_tokens=1"
            )

        source_indices.add(source_index)
        request_ids.add(request_id)
        active_iterations = effective_tokens + (1 if effective_tokens > 0 else 0)
        replay_required_cache_tokens = prompt_tokens + active_iterations
        production_guard_min_cache_tokens = prompt_tokens + max(
            0,
            int(configuration["max_new_tokens"]) - 1,
        )
        if production_guard_min_cache_tokens > int(configuration["cache_length"]):
            raise ValueError(
                f"{request_id}: source run violates its production cache guard: "
                f"required={production_guard_min_cache_tokens} "
                f"configured={configuration['cache_length']}"
            )
        timing = dict(record.get("timing_s") or {})
        items.append(
            {
                "source_index": source_index,
                "page_input_index": int(record.get("page_input_index", -1)),
                "block_index": int(record.get("block_index", -1)),
                "request_id": request_id,
                "label": str(record.get("label", "unknown")),
                "prompt_tokens": prompt_tokens,
                "first_token": token_ids[0],
                "token_ids": token_ids,
                "generated_tokens": generated_tokens,
                "effective_decode_tokens": effective_tokens,
                "active_decode_iterations": active_iterations,
                "stop_reason": stop_reason,
                "replay_required_cache_tokens": replay_required_cache_tokens,
                "production_guard_min_cache_tokens": (
                    production_guard_min_cache_tokens
                ),
                "recorded_timing_s": {
                    name: float(timing[name])
                    for name in (
                        "decode_ready_queue_wait",
                        "decode_slot_residency",
                    )
                    if name in timing
                },
            }
        )

    items.sort(key=lambda item: int(item["source_index"]))
    expected_indices = list(range(len(items)))
    actual_indices = [int(item["source_index"]) for item in items]
    if actual_indices != expected_indices:
        raise ValueError(
            "global_request_index must be contiguous request-source order"
        )

    generated_total = sum(int(item["generated_tokens"]) for item in items)
    effective_total = sum(int(item["effective_decode_tokens"]) for item in items)
    for field, actual in (
        ("requests", len(items)),
        ("generated_tokens_including_eos", generated_total),
        ("decode_tokens_after_prefill_including_eos", effective_total),
    ):
        expected = int(recognition[field])
        if actual != expected:
            raise ValueError(
                f"run-summary mismatch for {field}: trace={actual} summary={expected}"
            )

    generated_lengths = [int(item["generated_tokens"]) for item in items]
    prompt_lengths = [int(item["prompt_tokens"]) for item in items]
    replay_cache_lengths = [
        int(item["replay_required_cache_tokens"]) for item in items
    ]
    production_guard_lengths = [
        int(item["production_guard_min_cache_tokens"]) for item in items
    ]
    stop_reasons = Counter(str(item["stop_reason"]) for item in items)
    cache_coverage = {
        str(threshold): {
            "requests": sum(length <= threshold for length in replay_cache_lengths),
            "fraction": sum(
                length <= threshold for length in replay_cache_lengths
            )
            / len(replay_cache_lengths),
            "overflow_requests": sum(
                length > threshold for length in replay_cache_lengths
            ),
        }
        for threshold in cache_thresholds
    }
    return {
        "schema_version": 1,
        "kind": "text_decode_trace_replay",
        "source": {
            "trace_path": str(trace_path),
            "trace_sha256": _sha256(trace_path),
            "run_summary_path": str(summary_path),
            "run_summary_sha256": _sha256(summary_path),
        },
        "contract": {
            "ordering": "items are stored in exact production request-source order",
            "tokens": (
                "token_ids include the prefill-produced first token and the final "
                "EOS/length token exactly as recorded"
            ),
            "scheduler_lifetime": (
                "active_decode_iterations includes production's one queue-depth-one "
                "completion look-ahead iteration"
            ),
            "cache_capacity": (
                "replay_required_cache_tokens includes every recorded graph write, "
                "including the queue-depth-one look-ahead; "
                "production_guard_min_cache_tokens records the source runner's "
                "prompt + configured max_new_tokens - 1 admission guard"
            ),
            "device_replay": (
                "recorded request lifetimes drive completion while the real decode "
                "arena, graph, D2H token path, admission, and refill machinery run"
            ),
        },
        "production_configuration": {
            "batch_size": int(configuration["batch_size"]),
            "cache_length": int(configuration["cache_length"]),
            "max_new_tokens": int(configuration["max_new_tokens"]),
            "dtype": str(configuration["dtype"]),
        },
        "production_reference": {
            name: recognition[name]
            for name in (
                "decode_graph_calls",
                "raw_decode_token_slots",
                "active_decode_token_slots",
                "effective_decode_tokens",
                "idle_decode_token_slots",
                "lookahead_decode_token_slots",
                "decode_wall_s",
                "run_scoped_scheduler_wall_s",
                "kv_prefix_bytes_copied",
            )
            if name in recognition
        },
        "items": items,
        "distribution": {
            "requests": len(items),
            "pages": len(
                {
                    int(item["page_input_index"])
                    for item in items
                    if int(item["page_input_index"]) >= 0
                }
            ),
            "generated_tokens": generated_total,
            "effective_decode_tokens": effective_total,
            "prefill_only_requests": sum(
                int(item["active_decode_iterations"]) == 0 for item in items
            ),
            "stop_reason_counts": dict(sorted(stop_reasons.items())),
            "prompt_tokens": _length_summary(prompt_lengths),
            "generated_tokens_per_request": _length_summary(generated_lengths),
            "replay_required_cache_tokens": _length_summary(
                replay_cache_lengths
            ),
            "production_guard_min_cache_tokens": _length_summary(
                production_guard_lengths
            ),
            "replay_cache_coverage": cache_coverage,
        },
        "self_check": {
            "passed": True,
            "records_checked": len(items),
            "request_indices_contiguous": True,
            "summary_totals_match": True,
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = build_corpus(
        args.trace,
        args.run_summary,
        cache_thresholds=tuple(args.cache_thresholds),
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    distribution = payload["distribution"]
    print(
        "DECODE_CORPUS "
        f"requests={distribution['requests']} "
        f"pages={distribution['pages']} "
        f"effective_tokens={distribution['effective_decode_tokens']} "
        f"max_replay_cache={distribution['replay_required_cache_tokens']['max']}"
    )
    print(f"output={output}")


if __name__ == "__main__":
    main()
