#!/usr/bin/env python3
"""Execute current FIFO-B1 and profile-guided routes on an exact page subset."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
REPO_ROOT = EXPERIMENT_ROOT.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXPERIMENT_ROOT))

from paddleocr_vl.model.modeling import (  # noqa: E402
    LocalPaddleOCRVLForConditionalGeneration,
)
from paddleocr_vl.model.vision_prefill import (  # noqa: E402
    PreparedVisionPrefill,
    VisionPrefillRuntime,
)
from paddleocr_vl.serving.runtime_defaults import (  # noqa: E402
    OPTIMIZED_VISION_BUCKETS,
)
from utils.timing import DeviceTimeline, synchronize  # noqa: E402
from vision_lab_batched_packed import (  # noqa: E402
    DEFAULT_CACHE_ROOT as DEFAULT_BATCHED_CACHE_ROOT,
    _compiled_shape,
    _compiled_shape_cache_dir,
)
from vision_lab_e2e_replay import (  # noqa: E402
    DEFAULT_CACHE_ROOT as DEFAULT_B1_CACHE_ROOT,
    DEFAULT_CORPUS,
    DEFAULT_MODEL,
    DEFAULT_PACKED_TRACE,
    _load_items,
    _load_packed_groups,
    _materialize,
    _warm_cache_preflight,
)
from vision_lab_router_sim import _best_candidate  # noqa: E402


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "tmp/09_persistent_page_engine/vision_lab"
    / "router_replay_64p_minpixels_div4.json"
)
PROFILE_BATCHED_SHAPES = ((2, 3072), (4, 1024))
PAGE_PATTERN = re.compile(r"^page_(\d+)_")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--variant", default="min_pixels_28224")
    parser.add_argument("--packed-trace", type=Path, default=DEFAULT_PACKED_TRACE)
    parser.add_argument("--pages", type=int, default=64)
    parser.add_argument("--lookahead", type=int, default=8)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--b1-cache-dir", type=Path, default=DEFAULT_B1_CACHE_ROOT)
    parser.add_argument(
        "--batched-cache-dir",
        type=Path,
        default=DEFAULT_BATCHED_CACHE_ROOT,
    )
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--profile-first",
        action="store_true",
        help="Run the profile-guided lane before the current-router baseline.",
    )
    parser.add_argument(
        "--allow-compile",
        action="store_true",
        help="Allow the two selected batched shapes to compile when absent.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.pages <= 0 or args.lookahead <= 0:
        parser.error("--pages and --lookahead must be positive")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")
    return args


def _page_index(item: dict[str, Any]) -> int:
    match = PAGE_PATTERN.match(str(item["name"]))
    if match is None:
        raise ValueError(f"cannot derive page index from crop {item['name']!r}")
    return int(match.group(1))


def _profile_plan(
    items: Sequence[dict[str, Any]],
    *,
    lookahead: int,
) -> list[dict[str, Any]]:
    pending: list[dict[str, Any]] = []
    cursor = 0
    decisions: list[dict[str, Any]] = []

    def refill() -> None:
        nonlocal cursor
        while len(pending) < lookahead and cursor < len(items):
            pending.append(items[cursor])
            cursor += 1

    refill()
    while pending:
        oldest = pending[0]
        candidate = _best_candidate(pending)
        if candidate is None:
            decisions.append(
                {
                    "kind": "eager_overflow",
                    "rows": [[oldest]],
                    "batch_size": 1,
                    "sequence_length": int(oldest["real_vision_tokens"]),
                    "real_tokens": int(oldest["real_vision_tokens"]),
                    "physical_tokens": int(oldest["real_vision_tokens"]),
                }
            )
            pending.pop(0)
        else:
            selected = {
                int(item["source_index"]) for item in candidate["selected"]
            }
            decisions.append(
                {
                    "kind": "compiled",
                    "rows": candidate["rows"],
                    "batch_size": int(candidate["batch_size"]),
                    "sequence_length": int(candidate["sequence_length"]),
                    "real_tokens": int(candidate["real_tokens"]),
                    "physical_tokens": int(candidate["physical_tokens"]),
                }
            )
            pending[:] = [
                item
                for item in pending
                if int(item["source_index"]) not in selected
            ]
        refill()
    routed = [
        str(item["name"])
        for decision in decisions
        for row in decision["rows"]
        for item in row
    ]
    expected = [str(item["name"]) for item in items]
    if sorted(routed) != sorted(expected) or len(routed) != len(set(routed)):
        raise AssertionError("profile-guided route did not cover every crop exactly once")
    return decisions


def _baseline_plan(
    all_groups: Sequence[Sequence[dict[str, Any]]],
    selected_names: set[str],
) -> tuple[list[dict[str, Any]], int]:
    decisions: list[dict[str, Any]] = []
    partial_groups = 0
    routed: set[str] = set()
    for source_group in all_groups:
        group = [item for item in source_group if str(item["name"]) in selected_names]
        if not group:
            continue
        partial_groups += int(len(group) != len(source_group))
        names = {str(item["name"]) for item in group}
        if routed & names:
            raise AssertionError("baseline trace group repeats a crop")
        routed.update(names)
        decisions.append({"kind": "runtime_b1", "rows": [group]})
    if routed != selected_names:
        raise AssertionError("baseline trace groups do not cover selected crops")
    return decisions, partial_groups


def _dummy_row(template: PreparedVisionPrefill) -> PreparedVisionPrefill:
    return PreparedVisionPrefill(
        prefix_hidden_states=torch.zeros_like(template.prefix_hidden_states),
        rope_cos=torch.ones_like(template.rope_cos),
        rope_sin=torch.zeros_like(template.rope_sin),
        attention_mask=torch.zeros_like(template.attention_mask),
        real_seq_len=0,
        physical_seq_len=template.physical_seq_len,
        execution="compiled_batched_packed_dummy_row",
    )


def _run_decision(
    decision: dict[str, Any],
    *,
    model: LocalPaddleOCRVLForConditionalGeneration,
    runtime: VisionPrefillRuntime,
    batched_graphs: dict[tuple[int, int], Callable[..., torch.Tensor]],
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, Any]:
    rows: list[list[dict[str, Any]]] = decision["rows"]
    flat = [item for row in rows for item in row]
    materialized = [
        _materialize(item, seed=seed, dtype=dtype, device=device) for item in flat
    ]
    pixels = [pair[0] for pair in materialized]
    grids = [pair[1] for pair in materialized]
    timeline = DeviceTimeline(device)
    hidden = timeline.measure(
        "vision_embeddings",
        lambda: [
            model.visual.vision_model.embeddings(
                pixel.unsqueeze(0),
                image_grid_thw=grid,
            )
            for pixel, grid in zip(pixels, grids)
        ],
    )
    lengths = [int(value.shape[0]) for value in hidden]
    real_tokens = sum(lengths)

    if decision["kind"] == "runtime_b1":
        route = runtime.route(real_tokens)
        if len(flat) == 1:
            prepared = timeline.measure(
                "vision_prefill_input_prep",
                lambda: runtime.prepare(hidden[0], grids[0], route=route),
            )
            output = timeline.measure(
                "vision_tower",
                lambda: runtime.run_prepared(prepared),
            )
            output_lengths = [int(output.shape[0])]
        else:
            prepared_packed = timeline.measure(
                "vision_prefill_input_prep",
                lambda: runtime.prepare_packed(hidden, grids, route=route),
            )
            output = timeline.measure(
                "vision_tower",
                lambda: runtime.run_prepared(prepared_packed.prepared),
            )
            output_lengths = [
                int(value.shape[0])
                for value in torch.split(
                    output,
                    prepared_packed.segment_lengths,
                    dim=0,
                )
            ]
        execution = str(route["execution"])
        batch_size = 1
        sequence_length = int(route["physical_vision_tokens"])
        physical_tokens = sequence_length
    elif decision["kind"] == "eager_overflow":
        route = runtime.route(real_tokens)
        if str(route["execution"]) != "eager_overflow":
            raise AssertionError("profile overflow unexpectedly has a compiled B1 route")
        prepared = timeline.measure(
            "vision_prefill_input_prep",
            lambda: runtime.prepare(hidden[0], grids[0], route=route),
        )
        output = timeline.measure(
            "vision_tower",
            lambda: runtime.run_prepared(prepared),
        )
        output_lengths = [int(output.shape[0])]
        execution = "eager_overflow"
        batch_size = 1
        sequence_length = real_tokens
        physical_tokens = real_tokens
    elif int(decision["batch_size"]) == 1:
        batch_size = 1
        sequence_length = int(decision["sequence_length"])
        route = {
            "execution": "compiled",
            "physical_vision_tokens": sequence_length,
        }
        prepared_packed = timeline.measure(
            "vision_prefill_input_prep",
            lambda: runtime.prepare_packed(hidden, grids, route=route),
        )
        output = timeline.measure(
            "vision_tower",
            lambda: runtime.run_prepared(prepared_packed.prepared),
        )
        output_lengths = [
            int(value.shape[0])
            for value in torch.split(
                output,
                prepared_packed.segment_lengths,
                dim=0,
            )
        ]
        execution = "compiled"
        physical_tokens = sequence_length
    else:
        batch_size = int(decision["batch_size"])
        sequence_length = int(decision["sequence_length"])
        route = {
            "execution": "compiled",
            "physical_vision_tokens": sequence_length,
        }
        row_prepared = []
        row_lengths: list[list[int]] = []
        offset = 0
        for row in rows:
            if not row:
                continue
            end = offset + len(row)
            prepared_row = timeline.measure(
                f"vision_prefill_input_prep_row_{len(row_prepared)}",
                lambda offset=offset, end=end: runtime.prepare_packed(
                    hidden[offset:end],
                    grids[offset:end],
                    route=route,
                ).prepared,
            )
            row_prepared.append(prepared_row)
            row_lengths.append(lengths[offset:end])
            offset = end
        while len(row_prepared) < batch_size:
            row_prepared.append(_dummy_row(row_prepared[0]))
            row_lengths.append([])
        prefix = torch.cat(
            [prepared.prefix_hidden_states for prepared in row_prepared], dim=0
        )
        rope_cos = torch.cat([prepared.rope_cos for prepared in row_prepared], dim=0)
        rope_sin = torch.cat([prepared.rope_sin for prepared in row_prepared], dim=0)
        attention_mask = torch.cat(
            [prepared.attention_mask for prepared in row_prepared], dim=0
        )
        run = batched_graphs[(batch_size, sequence_length)]
        output = timeline.measure(
            "vision_tower",
            lambda: run(prefix, rope_cos, rope_sin, attention_mask),
        )
        if tuple(output.shape[:2]) != (batch_size, sequence_length):
            raise RuntimeError(
                "batched graph returned the wrong shape: "
                f"expected={(batch_size, sequence_length)} got={tuple(output.shape)}"
            )
        output_lengths = []
        for row_index, expected_lengths in enumerate(row_lengths):
            row_real = sum(expected_lengths)
            if not expected_lengths:
                continue
            row_output = output[row_index, :row_real]
            output_lengths.extend(
                int(value.shape[0])
                for value in torch.split(row_output, expected_lengths, dim=0)
            )
        execution = "compiled"
        physical_tokens = batch_size * sequence_length

    if output_lengths != lengths:
        raise RuntimeError(
            f"vision output split mismatch: expected={lengths} got={output_lengths}"
        )
    spans = timeline.resolve_spans()
    stage_s = {
        name: float(span["seconds"])
        for name, span in spans.items()
    }
    del output, hidden, pixels, grids, materialized
    return {
        "crops": len(flat),
        "real_tokens": real_tokens,
        "physical_tokens": physical_tokens,
        "execution": execution,
        "shape": f"b{batch_size}_s{sequence_length}",
        "stage_s": stage_s,
    }


def _run_lane(
    name: str,
    decisions: Sequence[dict[str, Any]],
    *,
    model: LocalPaddleOCRVLForConditionalGeneration,
    runtime: VisionPrefillRuntime,
    batched_graphs: dict[tuple[int, int], Callable[..., torch.Tensor]],
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
    progress_every: int,
) -> dict[str, Any]:
    stage_s: Counter[str] = Counter()
    shapes: Counter[str] = Counter()
    executions: Counter[str] = Counter()
    crops = 0
    real_tokens = 0
    physical_tokens = 0
    wall_started = time.perf_counter()
    for index, decision in enumerate(decisions, 1):
        result = _run_decision(
            decision,
            model=model,
            runtime=runtime,
            batched_graphs=batched_graphs,
            seed=seed,
            dtype=dtype,
            device=device,
        )
        crops += int(result["crops"])
        real_tokens += int(result["real_tokens"])
        physical_tokens += int(result["physical_tokens"])
        shapes[str(result["shape"])] += 1
        executions[str(result["execution"])] += 1
        for stage, seconds in result["stage_s"].items():
            stage_s[stage] += float(seconds)
        if progress_every and (index % progress_every == 0 or index == len(decisions)):
            print(
                f"lane={name} calls={index}/{len(decisions)} "
                f"tower_s={stage_s['vision_tower']:.3f}",
                flush=True,
            )
    synchronize(device)
    tower_s = float(stage_s["vision_tower"])
    return {
        "calls": len(decisions),
        "crops": crops,
        "crops_per_call": crops / len(decisions),
        "real_vision_tokens": real_tokens,
        "physical_vision_tokens": physical_tokens,
        "padding_vision_tokens": physical_tokens - real_tokens,
        "real_token_fraction": real_tokens / physical_tokens,
        "shape_calls": dict(sorted(shapes.items())),
        "execution_calls": dict(sorted(executions.items())),
        "device_stage_s": dict(stage_s),
        "vision_tower_s": tower_s,
        "effective_real_tokens_per_s": real_tokens / tower_s,
        "raw_physical_tokens_per_s": physical_tokens / tower_s,
        "lab_wall_s": time.perf_counter() - wall_started,
    }


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    import torch_npu  # noqa: F401

    if not torch.npu.is_available():
        raise RuntimeError("vision router replay requires an NPU")
    device = torch.device("npu:0")
    dtype = torch.float16
    torch.npu.set_compile_mode(jit_compile=False)
    all_items = _load_items(args.corpus, args.variant)
    selected = [item for item in all_items if _page_index(item) < args.pages]
    if not selected:
        raise ValueError("page selection contains no crops")
    selected_names = {str(item["name"]) for item in selected}
    all_packed_groups, trace_summary = _load_packed_groups(
        args.packed_trace,
        all_items,
    )
    baseline_plan, partial_groups = _baseline_plan(
        all_packed_groups,
        selected_names,
    )
    profile_plan = _profile_plan(selected, lookahead=args.lookahead)

    model_dir = args.model.expanduser().resolve()
    b1_cache_root = args.b1_cache_dir.expanduser().resolve()
    batched_cache_root = args.batched_cache_dir.expanduser().resolve()
    setup_started = time.perf_counter()
    model = LocalPaddleOCRVLForConditionalGeneration.from_pretrained(
        model_dir,
        dtype=dtype,
        device=device,
    )
    b1_preflight = _warm_cache_preflight(
        model=model,
        model_dir=model_dir,
        cache_root=b1_cache_root,
        device=device,
        dtype=dtype,
        allow_compile=False,
    )
    missing_batched: list[str] = []
    for batch_size, sequence_length in PROFILE_BATCHED_SHAPES:
        cache_dir = _compiled_shape_cache_dir(
            model=model,
            batch_size=batch_size,
            sequence_length=sequence_length,
            cache_root=batched_cache_root,
            model_dir=model_dir,
            dtype=dtype,
            device=device,
        )
        if not cache_dir.is_dir() or not any(cache_dir.rglob("*")):
            missing_batched.append(f"b{batch_size}_s{sequence_length}")
    if missing_batched and not args.allow_compile:
        raise RuntimeError(f"missing compatible cached batched graphs: {missing_batched}")
    runtime = VisionPrefillRuntime(
        model,
        backend="torchair",
        buckets=OPTIMIZED_VISION_BUCKETS,
        cache_root=b1_cache_root,
        device=device,
        dtype=dtype,
        model_dir=model_dir,
        attention_impl="prompt_flash_attention",
        padding="bucket",
    )
    batched_graphs: dict[tuple[int, int], Callable[..., torch.Tensor]] = {}
    batched_metadata: dict[str, Any] = {}
    for batch_size, sequence_length in PROFILE_BATCHED_SHAPES:
        run, metadata = _compiled_shape(
            model=model,
            batch_size=batch_size,
            sequence_length=sequence_length,
            cache_root=batched_cache_root,
            model_dir=model_dir,
            dtype=dtype,
            device=device,
        )
        batched_graphs[(batch_size, sequence_length)] = run
        batched_metadata[f"b{batch_size}_s{sequence_length}"] = metadata
    synchronize(device)
    setup_s = time.perf_counter() - setup_started

    lane_specs = [
        ("current_fifo_b1", baseline_plan),
        (f"profile_guided_lookahead_{args.lookahead}", profile_plan),
    ]
    if args.profile_first:
        lane_specs.reverse()
    lane_results: dict[str, dict[str, Any]] = {}
    for lane_name, plan in lane_specs:
        lane_results[lane_name] = _run_lane(
            lane_name,
            plan,
            model=model,
            runtime=runtime,
            batched_graphs=batched_graphs,
            seed=args.seed,
            dtype=dtype,
            device=device,
            progress_every=args.progress_every,
        )
    baseline = lane_results["current_fifo_b1"]
    profiled = lane_results[f"profile_guided_lookahead_{args.lookahead}"]
    for key in ("crops", "real_vision_tokens"):
        if baseline[key] != profiled[key]:
            raise AssertionError(
                f"lane workload mismatch for {key}: {baseline[key]} != {profiled[key]}"
            )
    comparison = {
        "vision_tower_s_saved": baseline["vision_tower_s"] - profiled["vision_tower_s"],
        "vision_tower_speedup": baseline["vision_tower_s"] / profiled["vision_tower_s"],
        "vision_tower_reduction_fraction": 1.0 - profiled["vision_tower_s"] / baseline["vision_tower_s"],
    }
    payload = {
        "schema_version": 1,
        "purpose": "actual cached-graph replay of current and profile-guided vision routing",
        "inputs": {
            "corpus": str(args.corpus.expanduser().resolve()),
            "variant": args.variant,
            "packed_trace": str(args.packed_trace.expanduser().resolve()),
            "pages": args.pages,
            "crops": len(selected),
            "lookahead": args.lookahead,
            "lane_order": [name for name, _plan in lane_specs],
            "real_vision_tokens": sum(int(item["real_vision_tokens"]) for item in selected),
            "tensor_policy": (
                "deterministic shape-equivalent random patch tensors; crop IDs and grids are exact"
            ),
        },
        "setup_s": setup_s,
        "cache_preflight": {
            "b1": b1_preflight,
            "batched_missing_before_run": missing_batched,
            "batched_compile_allowed": bool(args.allow_compile),
            "batched": batched_metadata,
        },
        "plans": {
            "baseline_trace": trace_summary,
            "baseline_partial_groups_at_page_boundary": partial_groups,
            "baseline_calls": len(baseline_plan),
            "profiled_calls": len(profile_plan),
        },
        "results": {
            "current_fifo_b1": baseline,
            f"profile_guided_lookahead_{args.lookahead}": profiled,
        },
        "comparison": comparison,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "comparison": comparison, "results": payload["results"]}, indent=2))


if __name__ == "__main__":
    main()
