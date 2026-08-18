#!/usr/bin/env python3
"""Search a one-page UniRec vision bucket frontier from measured 310P latency."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from vision_bucket_presets import VisionBucketSpec, resolve_vision_bucket_specs


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
    parser.add_argument(
        "--refine-target",
        action="store_true",
        help="run exhaustive one-swap descent on the max-bucket beam result",
    )
    parser.add_argument(
        "--evolution-generations",
        type=int,
        default=0,
        help="evolutionary generations for the max-bucket set",
    )
    parser.add_argument("--evolution-population", type=int, default=32)
    parser.add_argument("--random-seed", type=int, default=20260818)
    parser.add_argument("--page-lookahead", type=int, default=1)
    parser.add_argument("--include-fallback", action="store_true")
    parser.add_argument(
        "--allow-unaligned",
        action="store_true",
        help=(
            "search the native processed-shape frontier without expanding "
            "canvases to 16-row final-stage tiles"
        ),
    )
    parser.add_argument(
        "--baseline-preset",
        default="310p_k10_l4_aligned",
        help="existing vision preset to score with the same workload and model",
    )
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
    pages: list[list[tuple[int, int]]],
    xs: list[int],
    ys: list[float],
    *,
    require_aligned_final_stage: bool = True,
) -> tuple[list[Variant], list[str], list[dict[str, Any]]]:
    widths = sorted({width for page in pages for width, _height in page})
    heights = sorted({height for page in pages for _width, height in page})
    variants_by_key: dict[tuple[int, int, int], Variant] = {}
    rejected_unaligned = []
    alignment_adjustments = []
    for width in widths:
        for height in heights:
            for batch_size in (1, 2, 4):
                spec = VisionBucketSpec(width, height, batch_size)
                aligned_canvases = [(width, height)]
                if (
                    require_aligned_final_stage
                    and not spec.has_aligned_final_stage_rows
                ):
                    rejected_unaligned.append(spec.key)
                    # An optimum under a monotone physical-pixel cost never
                    # needs more than the minimum-area aligned expansion of a
                    # candidate canvas. Search the complete 16-row tile period
                    # in both spatial dimensions and retain all minimum-area
                    # ties, because their coverage of other shapes can differ.
                    expanded = []
                    for width_steps in range(16):
                        for height_steps in range(16):
                            candidate_width = width + width_steps * 32
                            candidate_height = height + height_steps * 32
                            candidate = VisionBucketSpec(
                                candidate_width,
                                candidate_height,
                                batch_size,
                            )
                            if candidate.has_aligned_final_stage_rows:
                                expanded.append(
                                    (candidate_width * candidate_height, candidate)
                                )
                    minimum_area = min(area for area, _candidate in expanded)
                    aligned_canvases = [
                        (candidate.width, candidate.height)
                        for area, candidate in expanded
                        if area == minimum_area
                    ]
                    alignment_adjustments.append(
                        {
                            "source_key": spec.key,
                            "replacement_keys": [
                                f"{candidate_width}x{candidate_height}_b{batch_size}"
                                for candidate_width, candidate_height in aligned_canvases
                            ],
                        }
                    )
                for candidate_width, candidate_height in aligned_canvases:
                    pixels = candidate_width * candidate_height * batch_size
                    variant = Variant(
                        width=candidate_width,
                        height=candidate_height,
                        batch_size=batch_size,
                        latency_ms=interpolate_latency(pixels, xs, ys),
                    )
                    variants_by_key[
                        (candidate_width, candidate_height, batch_size)
                    ] = variant
    return sorted(variants_by_key.values()), rejected_unaligned, alignment_adjustments


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
        or args.evolution_generations < 0
        or args.evolution_population < 2
    ):
        raise ValueError(
            "max buckets, beam width, and page lookahead must be positive"
        )
    started = time.perf_counter()
    pages, fallback_count, fallback_pixels = load_compiled_crops(
        args.iterations, include_fallback=args.include_fallback
    )
    xs, ys, latency_payload = latency_curve(args.latency_reference)
    variants, rejected_unaligned, alignment_adjustments = make_variants(
        pages,
        xs,
        ys,
        require_aligned_final_stage=not args.allow_unaligned,
    )
    baseline_specs = resolve_vision_bucket_specs(args.baseline_preset)
    variants_by_key = {variant.key: variant for variant in variants}
    for spec in baseline_specs:
        if spec.key not in variants_by_key:
            physical_pixels = spec.width * spec.height * spec.batch_size
            variants_by_key[spec.key] = Variant(
                width=spec.width,
                height=spec.height,
                batch_size=spec.batch_size,
                latency_ms=interpolate_latency(physical_pixels, xs, ys),
            )
    variants = sorted(variants_by_key.values())
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
    baseline_state = tuple(
        sorted(index_by_key[spec.key] for spec in baseline_specs)
    )
    baseline_result = evaluate(baseline_state)
    anchor_keys = (
        "960x64_b4",
        "512x256_b2",
        "960x256_b1",
        "512x512_b1",
        "960x512_b1",
    )
    anchor_state = tuple(sorted(index_by_key[key] for key in anchor_keys))
    frontier = []
    target_beam: list[tuple[tuple[Any, ...], tuple[int, ...]]] = []
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
                    "final_stage_rows": (
                        variant.batch_size
                        * (variant.width // 32)
                        * (variant.height // 32)
                    ),
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
        if bucket_count == args.max_buckets:
            target_beam = list(beam)
        print(
            "UNIREC_VISION_BUCKET_FRONTIER "
            f"k={bucket_count} graph_s={row['estimated_graph_s']:.6f} "
            f"calls={row['graph_calls']} pixel_eff={row['pixel_efficiency']:.6f} "
            f"used={row['used_variant_count']}"
        )

    def one_swap_refine(
        state: tuple[int, ...],
        result: tuple[float, int, int, float, tuple[int, ...]],
    ) -> tuple[tuple[float, int, int, float, tuple[int, ...]], tuple[int, ...], int]:
        rounds = 0
        while True:
            state_set = set(state)
            best_result = result
            best_state = state
            for removed in state:
                retained = state_set - {removed}
                for added in range(len(variants)):
                    if added in state_set:
                        continue
                    candidate_state = tuple(sorted((*retained, added)))
                    candidate_result = evaluate(candidate_state)
                    if candidate_result[:3] < best_result[:3]:
                        best_result = candidate_result
                        best_state = candidate_state
            if best_state == state:
                return result, state, rounds
            result = best_result
            state = best_state
            rounds += 1
            print(
                "UNIREC_VISION_BUCKET_LOCAL_REFINE "
                f"round={rounds} graph_s={result[0] / 1000.0:.6f}",
                flush=True,
            )

    target_result, target_state = target_beam[0]
    local_rounds = 0
    if args.refine_target:
        target_result, target_state, local_rounds = one_swap_refine(
            target_state,
            target_result,
        )

    evolutionary_report = None
    if args.evolution_generations:
        rng = random.Random(args.random_seed)
        population: dict[tuple[int, ...], tuple[float, int, int, float, tuple[int, ...]]] = {
            state: result for result, state in target_beam
        }

        def mutate(state: tuple[int, ...]) -> tuple[int, ...]:
            values = set(state)
            swaps = 2 if rng.random() < 0.20 else 1
            for _ in range(swaps):
                values.remove(rng.choice(tuple(values)))
                while True:
                    candidate = rng.randrange(len(variants))
                    if candidate not in values:
                        values.add(candidate)
                        break
            return tuple(sorted(values))

        def crossover(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
            target_count = args.max_buckets
            common = set(left) & set(right)
            pool = list((set(left) | set(right)) - common)
            rng.shuffle(pool)
            values = set(common)
            values.update(pool[: max(0, target_count - len(values))])
            while len(values) < target_count:
                values.add(rng.randrange(len(variants)))
            while len(values) > target_count:
                values.remove(rng.choice(tuple(values)))
            return tuple(sorted(values))

        while len(population) < args.evolution_population:
            parent = rng.choice(tuple(population))
            child = mutate(parent)
            result = evaluate(child)
            if math.isfinite(result[0]):
                population[child] = result

        for generation in range(1, args.evolution_generations + 1):
            ranked = sorted(
                ((result, state) for state, result in population.items()),
                key=lambda item: item[0][:3],
            )
            elites = ranked[: max(4, args.evolution_population // 4)]
            candidates = {state: result for result, state in elites}
            while len(candidates) < args.evolution_population * 2:
                left = rng.choice(elites)[1]
                if rng.random() < 0.50:
                    right = rng.choice(ranked[: max(8, len(ranked) // 2)])[1]
                    child = crossover(left, right)
                else:
                    child = left
                child = mutate(child)
                result = evaluate(child)
                if math.isfinite(result[0]):
                    candidates[child] = result
            population = dict(
                sorted(
                    candidates.items(),
                    key=lambda item: item[1][:3],
                )[: args.evolution_population]
            )
            if generation == 1 or generation % 10 == 0:
                generation_best = min(population.values(), key=lambda item: item[:3])
                print(
                    "UNIREC_VISION_BUCKET_EVOLUTION "
                    f"generation={generation} "
                    f"graph_s={generation_best[0] / 1000.0:.6f}",
                    flush=True,
                )
        evolutionary_state, evolutionary_result = min(
            population.items(),
            key=lambda item: item[1][:3],
        )
        if args.refine_target:
            evolutionary_result, evolutionary_state, evolutionary_local_rounds = (
                one_swap_refine(evolutionary_state, evolutionary_result)
            )
        else:
            evolutionary_local_rounds = 0
        if evolutionary_result[:3] < target_result[:3]:
            target_result = evolutionary_result
            target_state = evolutionary_state
        evolutionary_report = {
            "generations": args.evolution_generations,
            "population": args.evolution_population,
            "random_seed": args.random_seed,
            "post_evolution_local_rounds": evolutionary_local_rounds,
            "best_estimated_graph_s": evolutionary_result[0] / 1000.0,
        }

    all_variants_state = tuple(range(len(variants)))
    all_variants_result = evaluate(all_variants_state)

    def state_rows(state: tuple[int, ...]) -> list[dict[str, Any]]:
        return [
            {
                "key": variants[index].key,
                "estimated_latency_ms": variants[index].latency_ms,
                "used": index in set(target_result[4]),
            }
            for index in state
        ]

    target_optimization = {
        "bucket_count": args.max_buckets,
        "estimated_graph_s": target_result[0] / 1000.0,
        "physical_pixels": target_result[1],
        "graph_calls": target_result[2],
        "pixel_efficiency": target_result[3],
        "used_variant_count": len(target_result[4]),
        "local_refinement_rounds": local_rounds,
        "evolution": evolutionary_report,
        "variants": state_rows(target_state),
    }
    all_variants_smallest_routing_reference = {
        "candidate_variant_count": len(variants),
        "estimated_graph_s": all_variants_result[0] / 1000.0,
        "physical_pixels": all_variants_result[1],
        "graph_calls": all_variants_result[2],
        "pixel_efficiency": all_variants_result[3],
        "used_variant_count": len(all_variants_result[4]),
        "interpretation": (
            "diagnostic only, not a lower bound: exposing every canvas forces "
            "smallest-canvas routing and can fragment batches"
        ),
    }

    report = {
        "schema": "unirec_vision_bucket_frontier_v2",
        "status": "best_found_not_global_optimum_proof",
        "search": {
            "page_lookahead": args.page_lookahead,
            "page_group_count": len(grouped_shape_counts),
            "include_fallback": args.include_fallback,
            "max_buckets": args.max_buckets,
            "beam_width": args.beam_width,
            "candidate_canvas_count": len({variant.canvas for variant in variants}),
            "candidate_variant_count": len(variants),
            "rejected_unaligned_variant_count": len(rejected_unaligned),
            "rejected_unaligned_variants": rejected_unaligned,
            "alignment_adjustment_count": len(alignment_adjustments),
            "alignment_adjustments": alignment_adjustments,
            "batch_sizes": [1, 2, 4],
            "compiled_shape_constraint": (
                "batch_size*(width/32)*(height/32) divisible by 16"
                if not args.allow_unaligned
                else "width and height divisible by 32; no final-stage tile constraint"
            ),
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
        "baseline": {
            "preset": args.baseline_preset,
            "bucket_count": len(baseline_specs),
            "estimated_graph_s": baseline_result[0] / 1000.0,
            "physical_pixels": baseline_result[1],
            "graph_calls": baseline_result[2],
            "pixel_efficiency": baseline_result[3],
            "used_variant_count": len(baseline_result[4]),
            "variants": [spec.key for spec in baseline_specs],
        },
        "target_optimization": target_optimization,
        "all_variants_smallest_routing_reference": (
            all_variants_smallest_routing_reference
        ),
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
