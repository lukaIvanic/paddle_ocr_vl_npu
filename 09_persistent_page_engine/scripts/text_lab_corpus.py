#!/usr/bin/env python3
"""Build an exact, shape-faithful text-prefill workload corpus from a run trace."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import torch
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
REPO_ROOT = EXPERIMENT_ROOT.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

from paddleocr_vl.model.preprocessing import (
    build_inputs,
    image_grid_thw_from_size,
    load_preprocessor_config,
)
from paddleocr_vl.model.text_prefill import select_text_bucket
from paddleocr_vl.serving.runtime_defaults import OPTIMIZED_TEXT_BUCKETS


DEFAULT_MODEL = Path("/workspace/models/PaddleOCR-VL-1.6")
DEFAULT_TRACE = (
    REPO_ROOT
    / "tmp/09_persistent_page_engine/live_profile_router_h2d_concurrent_256p_5a37baf"
    / "recognition_trace.jsonl"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "tmp/09_persistent_page_engine/text_lab"
    / "corpus_256p_minpixels_div4_5a37baf.json"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--buckets",
        default=",".join(str(value) for value in OPTIMIZED_TEXT_BUCKETS),
        help="Compiled B=1 text buckets used to verify each recorded route.",
    )
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


def _parse_buckets(value: str) -> tuple[int, ...]:
    buckets = tuple(sorted({int(piece) for piece in value.split(",") if piece.strip()}))
    if not buckets or buckets[0] <= 0:
        raise ValueError("--buckets must contain positive integers")
    return buckets


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


def build_corpus(
    trace_path: Path,
    model_dir: Path,
    *,
    buckets: tuple[int, ...],
) -> dict[str, Any]:
    trace_path = trace_path.expanduser().resolve()
    model_dir = model_dir.expanduser().resolve()
    records = _read_jsonl(trace_path)
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    config = load_preprocessor_config(model_dir)
    merge_size = int(config["merge_size"])
    items: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []

    for source_line, record in enumerate(records):
        try:
            width, height = (int(value) for value in record["crop_size"])
            min_pixels = int(record["min_pixels"])
            max_pixels = int(record["max_pixels"])
            prompt = str(record["prompt"])
            recorded_input_tokens = int(record["input_tokens"])
            recorded_projected_tokens = int(record["projected_image_tokens"])
            recorded_route = dict(record["text_prefill"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"trace record {source_line} lacks required text-lab fields"
            ) from exc

        grid = image_grid_thw_from_size(
            width,
            height,
            patch_size=int(config["patch_size"]),
            merge_size=merge_size,
            temporal_patch_size=int(config["temporal_patch_size"]),
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            do_resize=bool(config["do_resize"]),
        )
        grid_tensor = torch.tensor([grid], dtype=torch.long)
        input_ids, _attention_mask = build_inputs(
            tokenizer,
            grid_tensor,
            prompt,
            merge_size=merge_size,
        )
        input_id_list = [int(value) for value in input_ids[0].tolist()]
        input_tokens = len(input_id_list)
        projected_tokens = math.prod(grid) // (merge_size * merge_size)
        selected_bucket = select_text_bucket(input_tokens, buckets)
        expected_execution = "compiled" if selected_bucket is not None else "eager_overflow"
        expected_physical = selected_bucket if selected_bucket is not None else input_tokens

        errors: list[str] = []
        if input_tokens != recorded_input_tokens:
            errors.append(
                f"input_tokens recomputed={input_tokens} recorded={recorded_input_tokens}"
            )
        if projected_tokens != recorded_projected_tokens:
            errors.append(
                "projected_image_tokens "
                f"recomputed={projected_tokens} recorded={recorded_projected_tokens}"
            )
        for field, expected in (
            ("execution", expected_execution),
            ("real_text_tokens", input_tokens),
            ("physical_text_tokens", expected_physical),
            ("bucket", selected_bucket),
        ):
            if recorded_route.get(field) != expected:
                errors.append(
                    f"route.{field} expected={expected!r} recorded={recorded_route.get(field)!r}"
                )
        if errors:
            mismatches.append(
                {
                    "source_line": source_line,
                    "request_id": record.get("request_id"),
                    "errors": errors,
                }
            )

        vision = dict(record.get("vision") or {})
        items.append(
            {
                "source_line": source_line,
                "source_index": int(record.get("global_request_index", source_line)),
                "page_input_index": int(record.get("page_input_index", -1)),
                "block_index": int(record.get("block_index", -1)),
                "request_id": str(record.get("request_id", f"crop_{source_line:06d}")),
                "label": str(record.get("label", "unknown")),
                "prompt": prompt,
                "crop_size": [width, height],
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
                "grid_thw": list(grid),
                "input_ids": input_id_list,
                "input_tokens": input_tokens,
                "projected_image_tokens": projected_tokens,
                "route": {
                    "execution": expected_execution,
                    "real_text_tokens": input_tokens,
                    "physical_text_tokens": expected_physical,
                    "padding_text_tokens": expected_physical - input_tokens,
                    "bucket": selected_bucket,
                },
                "production_group_id": (
                    int(vision["pack_group_id"])
                    if vision.get("pack_group_id") is not None
                    else None
                ),
            }
        )

    if mismatches:
        raise RuntimeError(
            f"text corpus self-check failed for {len(mismatches)}/{len(records)} "
            f"records; first mismatches:\n{json.dumps(mismatches[:20], indent=2)}"
        )

    items.sort(key=lambda item: (int(item["source_index"]), int(item["source_line"])))
    real_lengths = [int(item["input_tokens"]) for item in items]
    physical_lengths = [int(item["route"]["physical_text_tokens"]) for item in items]
    bucket_histogram = Counter(
        "eager_overflow" if item["route"]["bucket"] is None else str(item["route"]["bucket"])
        for item in items
    )
    return {
        "schema_version": 1,
        "kind": "text_prefill_trace_replay",
        "source": {
            "trace_path": str(trace_path),
            "trace_sha256": _sha256(trace_path),
            "model_dir": str(model_dir),
            "model_config_sha256": _sha256(model_dir / "config.json"),
            "tokenizer_sha256": _sha256(model_dir / "tokenizer.json"),
        },
        "contract": {
            "ordering": "global_request_index_then_trace_line",
            "batch_size": 1,
            "compiled_buckets": list(buckets),
            "embeddings": (
                "exact token ids and MRoPE layout; text lab substitutes deterministic "
                "image-token values after the real token embedding lookup"
            ),
            "measured_boundary": "text transformer plus in-place prefill KV writes",
        },
        "items": items,
        "distribution": {
            "items": len(items),
            "pages": len({int(item["page_input_index"]) for item in items}),
            "real_text_tokens": sum(real_lengths),
            "physical_text_tokens": sum(physical_lengths),
            "padding_text_tokens": sum(physical_lengths) - sum(real_lengths),
            "useful_token_fraction": sum(real_lengths) / sum(physical_lengths),
            "tokens_per_crop": {
                "min": min(real_lengths),
                "p50": _percentile(real_lengths, 0.50),
                "p95": _percentile(real_lengths, 0.95),
                "max": max(real_lengths),
            },
            "bucket_histogram": dict(sorted(bucket_histogram.items())),
            "eager_overflow_items": bucket_histogram.get("eager_overflow", 0),
        },
        "self_check": {
            "passed": True,
            "records_checked": len(items),
            "mismatches": 0,
            "checks": [
                "shape-only image grid",
                "projected image-token count",
                "tokenizer-expanded prompt length",
                "compiled bucket route",
            ],
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    corpus = build_corpus(
        args.trace,
        args.model,
        buckets=_parse_buckets(args.buckets),
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(corpus, indent=2) + "\n", encoding="utf-8")
    distribution = corpus["distribution"]
    print(f"Wrote {output}")
    print(
        "TEXT_CORPUS "
        f"items={distribution['items']} pages={distribution['pages']} "
        f"real={distribution['real_text_tokens']} "
        f"physical={distribution['physical_text_tokens']} "
        f"useful={distribution['useful_token_fraction']:.4f}"
    )
    print("TEXT_BUCKET_HISTOGRAM " + json.dumps(distribution["bucket_histogram"]))


if __name__ == "__main__":
    main()
