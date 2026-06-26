#!/usr/bin/env python3
"""Audit fixed-S static visual batching prospects without resizing crops.

Fixed physical visual-token length makes the transformer sequence static, but true
vision batching still needs compatible patch input shapes unless we add a later
padding policy for images/tokens. This audit reports same-grid/same-pixel-shape
groups from the real baseline crop bundle.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from vision_prefill_bench import (
    DEFAULT_MODEL,
    build_inputs_from_manifest,
    clean_json,
    json_default,
    load_baseline_manifest,
    load_preprocessor_config,
    projected_tokens,
    resolve_dataset_dir,
    stats,
    vision_tokens,
)


def parse_int_list(raw: str) -> list[int]:
    values = []
    for chunk in str(raw).replace(";", ",").split(","):
        chunk = chunk.strip()
        if chunk:
            values.append(int(chunk))
    if not values:
        raise ValueError(f"expected at least one integer in {raw!r}")
    return values


def summarize_batch_size(groups: list[dict[str, Any]], batch_size: int) -> dict[str, Any]:
    full_batches = 0
    partial_batches = 0
    usable_items = 0
    raw_slots = 0
    singletons = 0
    for group in groups:
        count = int(group["count"])
        if count == 1:
            singletons += 1
        full_batches += count // int(batch_size)
        if count % int(batch_size):
            partial_batches += 1
        usable_items += count
        raw_slots += int(batch_size) * ((count + int(batch_size) - 1) // int(batch_size))
    return {
        "batch_size": int(batch_size),
        "group_count": int(len(groups)),
        "singleton_group_count": int(singletons),
        "eligible_item_count": int(usable_items),
        "full_batch_count": int(full_batches),
        "partial_batch_count": int(partial_batches),
        "raw_slot_count_if_padded_within_group": int(raw_slots),
        "slot_utilization_if_group_padded": float(usable_items / raw_slots) if raw_slots else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Local model directory. HF download is disabled.")
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--baseline", default=str(Path(__file__).resolve().parent / "baselines" / "promptfa_fp16_eager_64"))
    parser.add_argument("--fixed-physical-seq-len", type=int, default=1024)
    parser.add_argument("--batch-sizes", default="2,4,8")
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent / "outputs" / "static_visual_batching_audit.json"))
    args = parser.parse_args()

    baseline_path = Path(args.baseline).expanduser().resolve()
    manifest = load_baseline_manifest(baseline_path)
    model_dir = Path(args.model).expanduser().resolve()
    if not (model_dir / "tokenizer.json").is_file():
        raise FileNotFoundError(f"tokenizer.json not found in local model directory: {model_dir}")
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    dataset_dir = resolve_dataset_dir(args.dataset_dir or manifest["build_summary"]["page"]["dataset_dir"])
    inputs = build_inputs_from_manifest(manifest=manifest, model_dir=model_dir, tokenizer=tokenizer, dataset_dir=dataset_dir)
    merge_size = int(load_preprocessor_config(model_dir)["merge_size"])
    fixed_s = int(args.fixed_physical_seq_len)

    eligible = []
    excluded = []
    for manifest_index, item in enumerate(inputs):
        real_tokens = int(vision_tokens(item))
        row = {
            "manifest_index": int(manifest_index),
            "id": str(item.entry.get("id")),
            "layout_label": str(item.entry.get("layout_label", "")),
            "image_grid_thw": [int(value) for value in item.image_grid_thw.flatten().tolist()],
            "pixel_values_shape": [int(dim) for dim in item.pixel_values.shape],
            "vision_tokens": int(real_tokens),
            "projected_image_tokens": int(projected_tokens(item, merge_size=merge_size)),
        }
        if fixed_s and real_tokens > fixed_s:
            row["reason"] = "real_visual_tokens_exceed_fixed_physical_seq_len"
            excluded.append(row)
        else:
            eligible.append(row)

    groups_by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        key = (
            tuple(row["image_grid_thw"]),
            tuple(row["pixel_values_shape"]),
            str(row["layout_label"]),
        )
        groups_by_key[key].append(row)
    groups = []
    for key, rows in groups_by_key.items():
        rows_sorted = sorted(rows, key=lambda row: (int(row["vision_tokens"]), str(row["id"])))
        groups.append(
            {
                "image_grid_thw": list(key[0]),
                "pixel_values_shape": list(key[1]),
                "layout_label": str(key[2]),
                "count": int(len(rows_sorted)),
                "vision_tokens": int(rows_sorted[0]["vision_tokens"]),
                "projected_image_tokens": int(rows_sorted[0]["projected_image_tokens"]),
                "sample_ids": [str(row["id"]) for row in rows_sorted[:12]],
            }
        )
    groups.sort(key=lambda row: (-int(row["count"]), int(row["vision_tokens"]), str(row["layout_label"])))
    batch_sizes = parse_int_list(args.batch_sizes)
    output = {
        "schema_version": 1,
        "experiment": "07_vision_prefill_optimization",
        "kind": "static_visual_batching_audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": {
            "path": str(baseline_path),
            "item_count": int(manifest.get("item_count", len(manifest.get("items", [])))),
        },
        "fixed_physical_seq_len": int(fixed_s),
        "summary": {
            "manifest_item_count": int(len(inputs)),
            "eligible_count": int(len(eligible)),
            "excluded_count": int(len(excluded)),
            "excluded_reason_counts": dict(sorted(Counter(row["reason"] for row in excluded).items())),
            "unique_same_shape_groups": int(len(groups)),
            "group_size": stats([float(row["count"]) for row in groups]),
            "vision_tokens": stats([float(row["vision_tokens"]) for row in eligible]),
            "batch_size_summaries": [summarize_batch_size(groups, batch_size) for batch_size in batch_sizes],
            "top_groups": clean_json(groups[:24]),
            "first_excluded": clean_json(excluded[:16]),
            "interpretation": (
                "This is a no-resize batching audit. Groups require identical image_grid_thw, pixel_values shape, "
                "and label prompt. High singleton count means true batching needs a later padding policy or a larger corpus."
            ),
        },
        "groups": clean_json(groups),
        "excluded": clean_json(excluded),
    }
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    print(json.dumps({"batching_audit_output": str(output_path), "summary": output["summary"]}, indent=2, default=json_default))


if __name__ == "__main__":
    main()
