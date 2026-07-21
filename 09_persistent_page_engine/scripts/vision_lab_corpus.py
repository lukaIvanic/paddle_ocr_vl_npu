#!/usr/bin/env python3
"""Build a shape-only vision-prefill workload corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
REPO_ROOT = EXPERIMENT_ROOT.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

from paddleocr_vl.model.preprocessing import image_grid_thw_from_size
from paddleocr_vl.serving.runtime_defaults import OPTIMIZED_VISION_BUCKETS


DEFAULT_TRACE = (
    REPO_ROOT
    / "tmp/09_persistent_page_engine/prefill_pipeline_streaming_6b1642f"
    / "recognition_trace.jsonl"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "tmp/09_persistent_page_engine/vision_lab"
PATCH_SIZE = 14
MERGE_SIZE = 2
TEMPORAL_PATCH_SIZE = 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--trace",
        type=Path,
        nargs="?",
        const=DEFAULT_TRACE,
        help="Archived recognition_trace.jsonl (default: optimized 32-page run).",
    )
    source.add_argument(
        "--lengths",
        help="Comma-separated target tower-token counts for a synthetic corpus.",
    )
    parser.add_argument("--count-per-length", type=int, default=1)
    parser.add_argument(
        "--min-pixels-variant",
        type=int,
        action="append",
        default=[],
        help=(
            "Repeat to re-grid every trace crop with an alternate min_pixels. "
            "The default trace profile remains the hard regression anchor."
        ),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
    if not records:
        raise ValueError(f"trace contains no records: {path}")
    return records


def _percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return float(ordered[0])
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bucket_for_tokens(tokens: int) -> int | None:
    return next(
        (int(bucket) for bucket in OPTIMIZED_VISION_BUCKETS if tokens <= bucket),
        None,
    )


def _distribution(items: list[dict[str, Any]]) -> dict[str, Any]:
    tokens = [int(item["real_vision_tokens"]) for item in items]
    bucket_histogram: dict[str, int] = {}
    for value in tokens:
        bucket = _bucket_for_tokens(value)
        label = str(bucket) if bucket is not None else "eager_overflow"
        bucket_histogram[label] = bucket_histogram.get(label, 0) + 1
    return {
        "items": len(tokens),
        "total_real_vision_tokens": sum(tokens),
        "tokens_per_crop": {
            "min": min(tokens),
            "p50": _percentile(tokens, 0.50),
            "p95": _percentile(tokens, 0.95),
            "max": max(tokens),
        },
        "bucket_histogram": bucket_histogram,
        "eager_overflow_crops": bucket_histogram.get("eager_overflow", 0),
    }


def _variant_items(
    default_items: list[dict[str, Any]],
    *,
    min_pixels: int,
) -> list[dict[str, Any]]:
    if min_pixels <= 0:
        raise ValueError("min_pixels variants must be positive")
    variants: list[dict[str, Any]] = []
    for item in default_items:
        width, height = (int(value) for value in item["crop_size"])
        max_pixels = int(item["max_pixels"])
        if min_pixels > max_pixels:
            raise ValueError(
                f"min_pixels variant {min_pixels} exceeds max_pixels {max_pixels}"
            )
        grid = image_grid_thw_from_size(
            width,
            height,
            patch_size=PATCH_SIZE,
            merge_size=MERGE_SIZE,
            temporal_patch_size=TEMPORAL_PATCH_SIZE,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        real_tokens = math.prod(grid)
        variant = dict(item)
        variant.update(
            {
                "grid_thw": list(grid),
                "real_vision_tokens": real_tokens,
                "projected_image_tokens": real_tokens // (MERGE_SIZE * MERGE_SIZE),
                "min_pixels": min_pixels,
                "source_default_real_vision_tokens": int(item["real_vision_tokens"]),
            }
        )
        variants.append(variant)
    return variants


def build_trace_corpus(
    path: Path,
    *,
    min_pixels_variants: Sequence[int] = (),
) -> dict[str, Any]:
    path = path.expanduser().resolve()
    records = _read_jsonl(path)
    items: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        try:
            width, height = (int(value) for value in record["crop_size"])
            min_pixels = int(record["min_pixels"])
            max_pixels = int(record["max_pixels"])
            projected = int(record["projected_image_tokens"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"trace record {index} lacks corpus shape fields") from exc
        grid = image_grid_thw_from_size(
            width,
            height,
            patch_size=PATCH_SIZE,
            merge_size=MERGE_SIZE,
            temporal_patch_size=TEMPORAL_PATCH_SIZE,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        real_tokens = math.prod(grid)
        expected_tokens = projected * MERGE_SIZE * MERGE_SIZE
        recorded_real = int(record.get("vision", {}).get("real_vision_tokens", real_tokens))
        if real_tokens != expected_tokens or real_tokens != recorded_real:
            mismatches.append(
                {
                    "index": index,
                    "name": record.get("request_id"),
                    "crop_size": [width, height],
                    "grid_thw": list(grid),
                    "recomputed_patch_count": real_tokens,
                    "projected_times_merge_squared": expected_tokens,
                    "recorded_real_vision_tokens": recorded_real,
                }
            )
        items.append(
            {
                "name": str(record.get("request_id", f"crop_{index:06d}")),
                "grid_thw": list(grid),
                "real_vision_tokens": real_tokens,
                "projected_image_tokens": projected,
                "crop_size": [width, height],
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
                "source_index": index,
            }
        )
    if mismatches:
        preview = json.dumps(mismatches[:20], indent=2)
        raise RuntimeError(
            f"corpus self-check failed for {len(mismatches)}/{len(records)} crops; "
            f"first mismatches:\n{preview}"
        )
    variant_payload: dict[str, Any] = {}
    for value in dict.fromkeys(int(item) for item in min_pixels_variants):
        variant_items = _variant_items(items, min_pixels=value)
        variant_payload[f"min_pixels_{value}"] = {
            "min_pixels": value,
            "max_pixels_policy": "preserve each source crop max_pixels",
            "items": variant_items,
            "distribution": _distribution(variant_items),
        }
    return {
        "schema_version": 1,
        "kind": "recognition_trace",
        "source": {
            "path": str(path),
            "sha256": _sha256(path),
            "records": len(records),
        },
        "shape_policy": {
            "implementation": "paddleocr_vl.model.preprocessing.image_grid_thw_from_size",
            "patch_size": PATCH_SIZE,
            "merge_size": MERGE_SIZE,
            "temporal_patch_size": TEMPORAL_PATCH_SIZE,
        },
        "self_check": {
            "passed": True,
            "checked": len(records),
            "mismatches": 0,
            "identity": "patch_count == projected_image_tokens * merge_size^2",
        },
        "default_distribution": _distribution(items),
        "variants": variant_payload,
        "items": items,
    }


def _near_square_merge_aligned_grid(target: int) -> tuple[int, int, int, int]:
    """Choose the nearest >= target merge-aligned grid with aspect <= 2.

    Both grid axes must be divisible by the spatial merge size for the real
    projector. We scan upward by merge^2 tokens and choose the first token
    count with a factorization no wider than 2:1, then the closest-to-square
    factor pair at that count. The actual count is recorded when an arbitrary
    requested length must be rounded (for example, 1337 -> 1344).
    """
    target = int(target)
    if target <= 0:
        raise ValueError(f"synthetic lengths must be positive, got {target}")
    quantum = MERGE_SIZE * MERGE_SIZE
    first = ((target + quantum - 1) // quantum) * quantum
    upper = max(first + 4096, int(math.ceil(target * 1.25)))
    fallback: tuple[float, int, int, int] | None = None
    for actual in range(first, upper + 1, quantum):
        units = actual // quantum
        pairs = [
            (factor, units // factor)
            for factor in range(1, math.isqrt(units) + 1)
            if units % factor == 0
        ]
        factor_h, factor_w = min(
            pairs,
            key=lambda pair: pair[1] / pair[0],
        )
        ratio = factor_w / factor_h
        candidate = (ratio, actual, factor_h * MERGE_SIZE, factor_w * MERGE_SIZE)
        if fallback is None or candidate < fallback:
            fallback = candidate
        if ratio <= 2.0:
            return 1, candidate[2], candidate[3], actual
    assert fallback is not None
    return 1, fallback[2], fallback[3], fallback[1]


def build_synthetic_corpus(lengths: list[int], count_per_length: int) -> dict[str, Any]:
    if count_per_length <= 0:
        raise ValueError("count-per-length must be positive")
    items: list[dict[str, Any]] = []
    for requested in lengths:
        t, h, w, actual = _near_square_merge_aligned_grid(requested)
        for ordinal in range(count_per_length):
            items.append(
                {
                    "name": f"synthetic_{requested:06d}_{ordinal:04d}",
                    "grid_thw": [t, h, w],
                    "real_vision_tokens": actual,
                    "requested_vision_tokens": requested,
                    "source_index": len(items),
                }
            )
    return {
        "schema_version": 1,
        "kind": "synthetic",
        "source": {
            "requested_lengths": lengths,
            "count_per_length": count_per_length,
        },
        "shape_policy": {
            "name": "near_square_merge_aligned",
            "description": (
                "smallest >= target merge^2-aligned token count admitting a "
                "near-square (aspect <= 2) integer grid; actual count is explicit"
            ),
            "patch_size": PATCH_SIZE,
            "merge_size": MERGE_SIZE,
            "temporal_patch_size": TEMPORAL_PATCH_SIZE,
        },
        "self_check": {"passed": True, "checked": len(items), "mismatches": 0},
        "items": items,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.trace is not None:
        corpus = build_trace_corpus(
            args.trace,
            min_pixels_variants=args.min_pixels_variant,
        )
        default_name = f"corpus_{args.trace.stem}.json"
    else:
        lengths = [int(piece) for piece in args.lengths.split(",") if piece.strip()]
        if not lengths:
            raise ValueError("--lengths must contain at least one integer")
        corpus = build_synthetic_corpus(lengths, args.count_per_length)
        default_name = "corpus_synthetic_" + "_".join(map(str, lengths)) + ".json"
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else (DEFAULT_OUTPUT_ROOT / default_name).resolve()
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(corpus, indent=2) + "\n", encoding="utf-8")
    total_tokens = sum(int(item["real_vision_tokens"]) for item in corpus["items"])
    print(
        json.dumps(
            {
                "output": str(output),
                "items": len(corpus["items"]),
                "real_vision_tokens": total_tokens,
                "self_check": corpus["self_check"],
                "variants": {
                    name: payload["distribution"]
                    for name, payload in corpus.get("variants", {}).items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
