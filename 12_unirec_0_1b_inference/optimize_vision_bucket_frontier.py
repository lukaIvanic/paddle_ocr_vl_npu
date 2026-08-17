#!/usr/bin/env python3
"""Search a one-page UniRec vision bucket frontier from measured 310P latency."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True, order=True)
class Variant:
    width: int
    height: int
    batch_size: int
    latency_ms: float

    @property
    def canvas(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def physical_pixels(self) -> int:
        return self.width * self.height * self.batch_size

    @property
    def key(self) -> str:
        return f"{self.width}x{self.height}_b{self.batch_size}"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iterations",
        type=Path,
        default=(
            root.parent
            / "tmp/12_unirec_0_1b_inference/"
            "representative128_w1t1_prefill_trace_4cf871c_20260814T184415/"
            "output/prefill_iterations.jsonl"
        ),
    )
    parser.add_argument(
        "--latency-reference",
        type=Path,
        default=(
            root
            / "references/"
            "unirec_vision_shape_batch_sweep_310p_20260817_transcribed.json"
        ),
    )
    parser.add_argument("--max-buckets", type=int, default=20)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--page-lookahead", type=int, default=1)
    parser.add_argument("--include-fallback", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_compiled_crops(
    path: Path, *, include_fallback: bool = False
) -> tuple[list[list[tuple[int, int]]], int, int]:
    by_page: dict[int, list[tuple[int, int]]] = defaultdict(list)
    fallback_count = 0
    fallback_pixels = 0
    pattern = re.compile(r"page_(\d+)_crop_")
    with path.expanduser().resolve().open(encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            name = event.get("event")
            if name not in {"vision_bucket_call", "vision_fallback_call"}:
                continue
            for member in event["members"]:
                width, height = map(int, member["processed_image_size"])
                match = pattern.search(str(member["request_id"]))
                if match is None:
                    raise ValueError(f"cannot recover page from {member['request_id']}")
                if name == "vision_fallback_call":
                    fallback_count += 1
                    fallback_pixels += width * height
                    if not include_fallback:
                        continue
                by_page[int(match.group(1))].append((width, height))
    if not by_page:
        raise ValueError("no compiled vision crops found")
    page_count = max(by_page) + 1
    return [by_page[index] for index in range(page_count)], fallback_count, fallback_pixels


def latency_curve(path: Path) -> tuple[list[int], list[float], dict[str, Any]]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in payload["rows"]:
        pixels = int(row["width"]) * int(row["height"]) * int(row["batch_size"])
        grouped[pixels].append(float(row["median_ms"]))
    xs = sorted(grouped)
    ys = [statistics.median(grouped[value]) for value in xs]
    if len(xs) < 2:
        raise ValueError("latency reference needs at least two physical-pixel points")
    return xs, ys, payload


def interpolate_latency(pixels: int, xs: list[int], ys: list[float]) -> float:
    if pixels <= xs[0]:
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        return max(0.0, ys[0] + slope * (pixels - xs[0]))
    if pixels >= xs[-1]:
        slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        return ys[-1] + slope * (pixels - xs[-1])
    lo = 0
    hi = len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= pixels:
            lo = mid
        else:
            hi = mid
    fraction = (pixels - xs[lo]) / (xs[hi] - xs[lo])
    return ys[lo] + fraction * (ys[hi] - ys[lo])


def make_variants(
    pages: list[list[tuple[int, int]]], xs: list[int], ys: list[float]
) -> list[Variant]:
    widths = sorted({width for page in pages for width, _height in page})
    heights = sorted({height for page in pages for _width, height in page})
    variants = []
    for width in widths:
        for height in heights:
            for batch_size in (1, 2, 4):
                pixels = width * height * batch_size
                variants.append(
                    Variant(
                        width=width,
                        height=height,
                        batch_size=batch_size,
                        latency_ms=interpolate_latency(pixels, xs, ys),
                    )
                )
    return variants


def minimum_call_plan(
    count: int, variants: Iterable[Variant]
) -> tuple[float, int, int, tuple[tuple[int, int], ...]]:
    options = tuple(sorted(variants, key=lambda item: item.batch_size))
    max_batch = max(item.batch_size for item in options)
    # Each row is (time_ms, physical_pixels, call_count, batch histogram).
    inf = (float("inf"), 2**63 - 1, 2**31 - 1, ())
    dp: list[tuple[float, int, int, tuple[tuple[int, int], ...]]] = [inf] * (
        count + max_batch + 1
    )
    dp[0] = (0.0, 0, 0, ())
    for done in range(count + 1):
        if not math.isfinite(dp[done][0]):
            continue
        for variant in options:
            target = min(count, done + variant.batch_size)
            histogram = Counter(dict(dp[done][3]))
            histogram[variant.batch_size] += 1
            candidate = (
                dp[done][0] + variant.latency_ms,
                dp[done][1] + variant.physical_pixels,
                dp[done][2] + 1,
                tuple(sorted(histogram.items())),
            )
            if candidate[:3] < dp[target][:3]:
                dp[target] = candidate
    return dp[count]


def main() -> None:
    args = parse_args()
    if (
        args.max_buckets < 1
        or args.beam_width < 1
        or args.page_lookahead < 1
    ):
        raise ValueError(
            "max buckets, beam width, and page lookahead must be positive"
        )
    started = time.perf_counter()
    pages, fallback_count, fallback_pixels = load_compiled_crops(
        args.iterations, include_fallback=args.include_fallback
    )
    xs, ys, latency_payload = latency_curve(args.latency_reference)
    variants = make_variants(pages, xs, ys)
    variant_by_canvas_batch = {
        (variant.width, variant.height, variant.batch_size): variant
        for variant in variants
    }
    shapes = sorted({shape for page in pages for shape in page})
    shape_index = {shape: index for index, shape in enumerate(shapes)}
    page_shape_counts = [Counter(shape_index[shape] for shape in page) for page in pages]
    grouped_shape_counts = []
    for start in range(0, len(page_shape_counts), args.page_lookahead):
        counts: Counter[int] = Counter()
        for page_counts in page_shape_counts[
            start : start + args.page_lookahead
        ]:
            counts.update(page_counts)
        grouped_shape_counts.append(counts)
    effective_pixels = sum(width * height for page in pages for width, height in page)
    crop_count = sum(len(page) for page in pages)

    compatible: list[tuple[int, ...]] = []
    for width, height in shapes:
        compatible.append(
            tuple(
                index
                for index, variant in enumerate(variants)
                if width <= variant.width and height <= variant.height
            )
        )

    @lru_cache(maxsize=None)
    def cached_call_plan(
        width: int, height: int, batches: tuple[int, ...], count: int
    ) -> tuple[float, int, int, tuple[tuple[int, int], ...]]:
        return minimum_call_plan(
            count,
            [variant_by_canvas_batch[(width, height, batch)] for batch in batches],
        )

    @lru_cache(maxsize=None)
    def evaluate(state: tuple[int, ...]) -> tuple[float, int, int, float, tuple[int, ...]]:
        by_canvas: dict[tuple[int, int], list[Variant]] = defaultdict(list)
        state_set = set(state)
        for index in state:
            by_canvas[variants[index].canvas].append(variants[index])
        canvases = sorted(
            by_canvas,
            key=lambda item: (item[0] * item[1], item[1], item[0]),
        )
        route = []
        for shape_id, (width, height) in enumerate(shapes):
            selected_compatible = [
                canvas
                for canvas in canvases
                if width <= canvas[0] and height <= canvas[1]
            ]
            if not selected_compatible:
                return float("inf"), 2**63 - 1, 2**31 - 1, 0.0, ()
            route.append(selected_compatible[0])

        total_ms = 0.0
        physical_pixels = 0
        calls = 0
        used: set[int] = set()
        variant_indices_by_canvas = {
            canvas: tuple(
                index
                for index in state
                if variants[index].canvas == canvas
            )
            for canvas in canvases
        }
        for counts in grouped_shape_counts:
            canvas_counts: Counter[tuple[int, int]] = Counter()
            for shape_id, count in counts.items():
                canvas_counts[route[shape_id]] += count
            for canvas, count in canvas_counts.items():
                batches = tuple(
                    sorted(
                        variants[index].batch_size
                        for index in variant_indices_by_canvas[canvas]
                    )
                )
                plan = cached_call_plan(canvas[0], canvas[1], batches, count)
                total_ms += plan[0]
                physical_pixels += plan[1]
                calls += plan[2]
                used_batches = {batch for batch, batch_calls in plan[3] if batch_calls}
                for index in variant_indices_by_canvas[canvas]:
                    if variants[index].batch_size in used_batches:
                        used.add(index)
        return (
            total_ms,
            physical_pixels,
            calls,
            effective_pixels / physical_pixels,
            tuple(sorted(used & state_set)),
        )

    # A single graph must cover the largest compiled shape. Retain a small beam
    # at every K; this is deterministic best-found search, not a proof that the
    # general two-dimensional facility-location optimum was found.
    maximum_width = max(width for width, _height in shapes)
    maximum_height = max(height for _width, height in shapes)
    initial = []
    for index, variant in enumerate(variants):
        if variant.width >= maximum_width and variant.height >= maximum_height:
            result = evaluate((index,))
            if math.isfinite(result[0]):
                initial.append((result, (index,)))
    initial.sort(key=lambda item: item[0][:3])
    beam = initial[: args.beam_width]
    index_by_key = {variant.key: index for index, variant in enumerate(variants)}
    anchor_keys = (
        "960x64_b4",
        "512x256_b2",
        "960x256_b1",
        "512x512_b1",
        "960x512_b1",
    )
    anchor_state = tuple(sorted(index_by_key[key] for key in anchor_keys))
    frontier = []
    for bucket_count in range(1, args.max_buckets + 1):
        if bucket_count > 1:
            expanded: dict[tuple[int, ...], tuple[Any, ...]] = {}
            for _parent_result, state in beam:
                state_set = set(state)
                for index in range(len(variants)):
                    if index in state_set:
                        continue
                    candidate_state = tuple(sorted((*state, index)))
                    if candidate_state in expanded:
                        continue
                    result = evaluate(candidate_state)
                    if math.isfinite(result[0]):
                        expanded[candidate_state] = result
            ranked = sorted(
                ((result, state) for state, result in expanded.items()),
                key=lambda item: item[0][:3],
            )
            beam = ranked[: args.beam_width]
        if bucket_count == len(anchor_state):
            candidates = {state: result for result, state in beam}
            candidates[anchor_state] = evaluate(anchor_state)
            beam = sorted(
                ((result, state) for state, result in candidates.items()),
                key=lambda item: item[0][:3],
            )[: args.beam_width]
        best_result, best_state = beam[0]
        selected_rows = []
        used_set = set(best_result[4])
        for index in best_state:
            variant = variants[index]
            selected_rows.append(
                {
                    "key": variant.key,
                    "width": variant.width,
                    "height": variant.height,
                    "batch_size": variant.batch_size,
                    "estimated_latency_ms": variant.latency_ms,
                    "used": index in used_set,
                }
            )
        row = {
            "bucket_count": bucket_count,
            "estimated_graph_s": best_result[0] / 1000.0,
            "physical_pixels": best_result[1],
            "graph_calls": best_result[2],
            "pixel_efficiency": best_result[3],
            "used_variant_count": len(best_result[4]),
            "variants": selected_rows,
        }
        frontier.append(row)
        print(
            "UNIREC_VISION_BUCKET_FRONTIER "
            f"k={bucket_count} graph_s={row['estimated_graph_s']:.6f} "
            f"calls={row['graph_calls']} pixel_eff={row['pixel_efficiency']:.6f} "
            f"used={row['used_variant_count']}"
        )

    report = {
        "schema": "unirec_vision_bucket_frontier_v1",
        "status": "best_found_not_global_optimum_proof",
        "search": {
            "page_lookahead": args.page_lookahead,
            "page_group_count": len(grouped_shape_counts),
            "include_fallback": args.include_fallback,
            "max_buckets": args.max_buckets,
            "beam_width": args.beam_width,
            "candidate_canvas_count": len({variant.canvas for variant in variants}),
            "candidate_variant_count": len(variants),
            "batch_sizes": [1, 2, 4],
            "routing": "smallest_compatible_canvas_then_exact_batch_mix",
            "latency_model": "piecewise_linear_by_total_physical_pixels",
            "search_anchor": list(anchor_keys),
            "elapsed_s": time.perf_counter() - started,
        },
        "workload": {
            "page_count": len(pages),
            "compiled_crop_count": crop_count,
            "compiled_unique_shape_count": len(shapes),
            "compiled_effective_pixels": effective_pixels,
            "fixed_fallback_count": fallback_count,
            "fixed_fallback_effective_pixels": fallback_pixels,
        },
        "latency_reference": {
            "path": str(args.latency_reference.expanduser().resolve()),
            "device_name": latency_payload.get("device_name"),
            "physical_pixel_points": [
                {"physical_pixels": x, "median_ms": y} for x, y in zip(xs, ys)
            ],
        },
        "frontier": frontier,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        "UNIREC_VISION_BUCKET_FRONTIER_DONE "
        f"elapsed_s={report['search']['elapsed_s']:.6f} output={output}"
    )


if __name__ == "__main__":
    main()
