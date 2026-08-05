#!/usr/bin/env python3
"""Reconstruct MinerU layout/crop vision-token distributions from a saved run."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


DEFAULT_BOUNDS = (128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 5120)
LOG_PREFIX = re.compile(r"^\d{4}-\d{2}-\d{2} .*? \| [A-Z]+\s+\|", re.MULTILINE)
LAYOUT_MARKER = "Layout raw output:"


def percentile(values: list[int], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def summarize(values: list[int], bounds: tuple[int, ...]) -> dict[str, Any]:
    total = sum(values)
    exact_bins: list[dict[str, Any]] = []
    previous = 0
    for bound in bounds:
        selected = [value for value in values if previous < value <= bound]
        exact_bins.append(
            {
                "range": f"{previous + 1}-{bound}",
                "count": len(selected),
                "request_fraction": len(selected) / len(values) if values else 0.0,
                "tokens": sum(selected),
                "token_fraction": sum(selected) / total if total else 0.0,
            }
        )
        previous = bound
    selected = [value for value in values if value > previous]
    exact_bins.append(
        {
            "range": f">{previous}",
            "count": len(selected),
            "request_fraction": len(selected) / len(values) if values else 0.0,
            "tokens": sum(selected),
            "token_fraction": sum(selected) / total if total else 0.0,
        }
    )
    cumulative = []
    for bound in bounds:
        selected = [value for value in values if value <= bound]
        cumulative.append(
            {
                "at_most": bound,
                "count": len(selected),
                "request_fraction": len(selected) / len(values) if values else 0.0,
                "tokens": sum(selected),
                "token_fraction": sum(selected) / total if total else 0.0,
            }
        )
    return {
        "count": len(values),
        "total_tokens": total,
        "min": min(values) if values else 0,
        "p10": percentile(values, 0.10),
        "p25": percentile(values, 0.25),
        "median": statistics.median(values) if values else 0.0,
        "mean": statistics.mean(values) if values else 0.0,
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else 0,
        "bins": exact_bins,
        "cumulative": cumulative,
    }


def extract_layout_outputs(log_path: Path) -> list[str]:
    text = log_path.read_text(encoding="utf-8")
    outputs: list[str] = []
    cursor = 0
    while True:
        marker = text.find(LAYOUT_MARKER, cursor)
        if marker < 0:
            break
        start = text.find("\n", marker)
        if start < 0:
            break
        start += 1
        next_log = LOG_PREFIX.search(text, start)
        end = len(text) if next_log is None else next_log.start()
        outputs.append(text[start:end].strip())
        cursor = end
    return outputs


def image_tokens(processor: Any, image: Image.Image) -> int:
    prepared = processor.image_processor(images=[image], return_tensors="pt")
    pixel_values = prepared["pixel_values"]
    grid_thw = prepared["image_grid_thw"]
    tokens = int(pixel_values.shape[0])
    grid_tokens = sum(math.prod(int(value) for value in row) for row in grid_thw.tolist())
    if tokens != grid_tokens:
        raise RuntimeError(f"pixel/grid token mismatch: pixels={tokens} grid={grid_tokens}")
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("/workspace/models/MinerU2.5-Pro-2605-1.2B"))
    parser.add_argument("--dataset-json", type=Path, default=Path("/workspace/datasets/OmniDocBench/OmniDocBench.json"))
    parser.add_argument("--images-dir", type=Path, default=Path("/workspace/datasets/OmniDocBench/images"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bounds", default=",".join(str(value) for value in DEFAULT_BOUNDS))
    args = parser.parse_args()

    from transformers import AutoProcessor
    from mineru_vl_utils.mineru_client import MinerUClientHelper

    run_dir = args.run_dir.expanduser().resolve()
    summary_path = run_dir / "output" / "run_summary_shard_00.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    layout_outputs = extract_layout_outputs(run_dir / "run.log")
    expected_pages = int(summary["shard_pages"])
    if len(layout_outputs) != expected_pages:
        raise RuntimeError(
            f"expected {expected_pages} raw layout outputs, found {len(layout_outputs)}"
        )

    dataset = json.loads(args.dataset_json.read_text(encoding="utf-8"))
    offset = int(summary["offset"])
    limit = int(summary["limit"])
    selected = dataset[offset : offset + limit]
    if len(selected) != expected_pages:
        raise RuntimeError("dataset selection does not match saved run")

    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_fast=True,
    )
    replay_client = MinerUClientHelper(
        backend="transformers",
        prompts={"[default]": ""},
        sampling_params={},
        layout_image_size=(1036, 1036),
        min_image_edge=28,
        max_image_edge_ratio=50,
        simple_post_process=False,
        handle_equation_block=True,
        abandon_list=False,
        abandon_paratext=False,
        image_analysis=False,
        debug=False,
        enable_table_formula_eq_wrap=False,
        enable_cross_page_table_merge=False,
    )

    layout_tokens: list[int] = []
    crop_tokens: list[int] = []
    crop_tokens_by_type: dict[str, list[int]] = defaultdict(list)
    crop_type_counts: Counter[str] = Counter()

    for page_index, (sample, raw_layout) in enumerate(zip(selected, layout_outputs)):
        image_name = Path(sample["page_info"]["image_path"]).name
        with Image.open(args.images_dir / image_name) as opened:
            image = opened.convert("RGB").copy()
        layout_tokens.append(image_tokens(processor, replay_client.prepare_for_layout(image)))
        blocks = replay_client.parse_layout_output(raw_layout)
        block_images, _prompts, _params, indices = replay_client.prepare_for_extract(
            image,
            blocks,
            image_analysis=False,
        )
        if len(block_images) != len(indices):
            raise RuntimeError(f"page {page_index}: crop/index count mismatch")
        for block_image, block_index in zip(block_images, indices):
            token_count = image_tokens(processor, block_image)
            block_type = str(blocks[block_index].type)
            crop_tokens.append(token_count)
            crop_tokens_by_type[block_type].append(token_count)
            crop_type_counts[block_type] += 1

    bounds = tuple(int(value) for value in args.bounds.split(",") if value.strip())
    expected = summary["local_compiled_generation"]["prefill_metrics"]
    checks = {
        "layout_request_count": [len(layout_tokens), int(summary["local_compiled_generation"]["prefill_calls"][0]["request_count"])],
        "crop_request_count": [len(crop_tokens), int(summary["local_compiled_generation"]["prefill_calls"][1]["request_count"])],
        "layout_tokens": [sum(layout_tokens), int(summary["local_compiled_generation"]["prefill_calls"][0]["prefill_metrics"]["raw_vision_tokens"])],
        "crop_tokens": [sum(crop_tokens), int(summary["local_compiled_generation"]["prefill_calls"][1]["prefill_metrics"]["raw_vision_tokens"])],
        "all_tokens": [sum(layout_tokens) + sum(crop_tokens), int(expected["raw_vision_tokens"])],
    }
    if any(actual != expected_value for actual, expected_value in checks.values()):
        raise RuntimeError(f"replay did not match measured run totals: {checks}")

    result = {
        "run_dir": str(run_dir),
        "replay_checks": {name: {"actual": values[0], "expected": values[1], "exact": values[0] == values[1]} for name, values in checks.items()},
        "layout": summarize(layout_tokens, bounds),
        "recognition_crops": summarize(crop_tokens, bounds),
        "recognition_by_type": {
            block_type: summarize(values, bounds)
            for block_type, values in sorted(crop_tokens_by_type.items())
        },
        "recognition_type_counts": dict(sorted(crop_type_counts.items())),
        "layout_token_counts": layout_tokens,
        "recognition_crop_token_counts": crop_tokens,
    }
    output = args.output or (run_dir / "prefill_token_distribution.json")
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("replay_checks", "layout", "recognition_crops", "recognition_type_counts")}, indent=2, ensure_ascii=False))
    print(f"[output] {output}")


if __name__ == "__main__":
    main()
