#!/usr/bin/env python3
"""Replay the exact E2E vision workload inside the standalone vision lab."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch

HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
REPO_ROOT = EXPERIMENT_ROOT.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

from paddleocr_vl.model.modeling import (  # noqa: E402
    LocalPaddleOCRVLForConditionalGeneration,
)
from paddleocr_vl.model.vision_prefill import (  # noqa: E402
    VisionPrefillRuntime,
    vision_cache_dir_for_bucket,
)
from paddleocr_vl.serving.runtime_defaults import (  # noqa: E402
    OPTIMIZED_VISION_BUCKETS,
)
from utils.timing import DeviceTimeline, synchronize  # noqa: E402


DEFAULT_MODEL = Path("/workspace/models/PaddleOCR-VL-1.6")
DEFAULT_CORPUS = (
    REPO_ROOT
    / "tmp/09_persistent_page_engine/vision_lab"
    / "corpus_256p_minpixels_div4_ee29c91.json"
)
DEFAULT_PACKED_TRACE = (
    REPO_ROOT
    / "tmp/09_persistent_page_engine"
    / "packed_integration_stageB_minpixels_div4_packed_256p_44c20b6"
    / "recognition_trace.jsonl"
)
DEFAULT_SINGLE_SUMMARY = (
    REPO_ROOT
    / "tmp/09_persistent_page_engine"
    / "packed_integration_stageB_minpixels_div4_off_256p_44c20b6"
    / "run_summary.json"
)
DEFAULT_PACKED_SUMMARY = (
    REPO_ROOT
    / "tmp/09_persistent_page_engine"
    / "packed_integration_stageB_minpixels_div4_packed_256p_44c20b6"
    / "run_summary.json"
)
DEFAULT_CACHE_ROOT = (
    REPO_ROOT / ".runtime_cache/09_persistent_page_engine_vision_torchair"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "tmp/09_persistent_page_engine/vision_lab"
    / "e2e_replay_256p_minpixels_div4.json"
)
LANES = ("single", "packed")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--variant", default="min_pixels_28224")
    parser.add_argument("--packed-trace", type=Path, default=DEFAULT_PACKED_TRACE)
    parser.add_argument("--single-summary", type=Path, default=DEFAULT_SINGLE_SUMMARY)
    parser.add_argument("--packed-summary", type=Path, default=DEFAULT_PACKED_SUMMARY)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--lane",
        action="append",
        choices=LANES,
        help="Repeat to select lanes; the default runs single then packed.",
    )
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--allow-compile",
        action="store_true",
        help="Allow missing cache entries to compile instead of failing preflight.",
    )
    args = parser.parse_args(argv)
    args.lane = tuple(dict.fromkeys(args.lane or LANES))
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")
    return args


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.expanduser().resolve().open(encoding="utf-8") as handle:
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


def _load_items(path: Path, variant: str) -> list[dict[str, Any]]:
    corpus = _read_json(path)
    if not corpus.get("self_check", {}).get("passed"):
        raise ValueError("vision corpus self-check is not marked passed")
    if variant == "default":
        items = list(corpus.get("items", []))
    else:
        try:
            items = list(corpus["variants"][variant]["items"])
        except KeyError as exc:
            raise KeyError(f"corpus does not contain variant {variant!r}") from exc
    if not items:
        raise ValueError("selected vision corpus is empty")
    names: set[str] = set()
    for index, item in enumerate(items):
        item["source_index"] = int(item.get("source_index", index))
        item["name"] = str(item["name"])
        item["grid_thw"] = [int(value) for value in item["grid_thw"]]
        item["real_vision_tokens"] = int(item["real_vision_tokens"])
        if item["name"] in names:
            raise ValueError(f"duplicate corpus item {item['name']!r}")
        if math.prod(item["grid_thw"]) != item["real_vision_tokens"]:
            raise ValueError(f"invalid grid/token count for {item['name']!r}")
        names.add(item["name"])
    return items


def _load_packed_groups(
    trace_path: Path,
    items: list[dict[str, Any]],
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    records = _read_jsonl(trace_path)
    item_by_name = {str(item["name"]): item for item in items}
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    metadata: dict[int, tuple[int, int, int]] = {}
    seen: set[str] = set()
    for record in records:
        name = str(record["request_id"])
        if name not in item_by_name:
            raise KeyError(f"trace crop {name!r} is missing from the lab corpus")
        if name in seen:
            raise ValueError(f"trace repeats crop {name!r}")
        seen.add(name)
        vision = record["vision"]
        group_id = int(vision["pack_group_id"])
        grouped[group_id].append(
            {
                "global_request_index": int(record["global_request_index"]),
                "item": item_by_name[name],
            }
        )
        current = (
            int(vision["pack_crops"]),
            int(vision["pack_real_vision_tokens"]),
            int(vision["pack_physical_vision_tokens"]),
        )
        if group_id in metadata and metadata[group_id] != current:
            raise ValueError(f"inconsistent trace metadata for pack group {group_id}")
        metadata[group_id] = current
        if int(vision["real_vision_tokens"]) != int(
            item_by_name[name]["real_vision_tokens"]
        ):
            raise ValueError(f"trace/corpus token mismatch for {name!r}")

    expected_names = set(item_by_name)
    if seen != expected_names:
        missing = sorted(expected_names - seen)
        extra = sorted(seen - expected_names)
        raise ValueError(
            f"trace/corpus identity mismatch: missing={missing[:10]} extra={extra[:10]}"
        )

    groups: list[list[dict[str, Any]]] = []
    group_sizes: Counter[int] = Counter()
    for group_id in sorted(grouped):
        rows = sorted(grouped[group_id], key=lambda row: row["global_request_index"])
        group = [row["item"] for row in rows]
        expected_crops, expected_real, _expected_physical = metadata[group_id]
        actual_real = sum(int(item["real_vision_tokens"]) for item in group)
        if len(group) != expected_crops or actual_real != expected_real:
            raise ValueError(
                f"pack group {group_id} mismatch: crops={len(group)}/{expected_crops} "
                f"real={actual_real}/{expected_real}"
            )
        groups.append(group)
        group_sizes[len(group)] += 1
    return groups, {
        "trace": str(trace_path.expanduser().resolve()),
        "records": len(records),
        "groups": len(groups),
        "group_size_histogram": dict(sorted(group_sizes.items())),
        "crops_per_group": len(records) / len(groups),
    }


def _materialize(
    item: dict[str, Any],
    *,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + int(item["source_index"]) * 104729)
    pixels = torch.randn(
        (int(item["real_vision_tokens"]), 3, 14, 14),
        generator=generator,
        dtype=dtype,
    ).to(device=device)
    grid = torch.tensor([item["grid_thw"]], dtype=torch.long)
    return pixels, grid


def _summary_reference(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    recognition = payload["recognition"]
    return {
        "path": str(path.expanduser().resolve()),
        "vision_tower_s": float(recognition["device_stage_s"]["vision_prefill"]),
        "real_vision_tokens": int(recognition["real_vision_tokens"]),
        "physical_vision_tokens": int(recognition["physical_vision_tokens"]),
        "groups": int(recognition["vision_packing"]["groups"]),
    }


def _warm_cache_preflight(
    *,
    model: LocalPaddleOCRVLForConditionalGeneration,
    model_dir: Path,
    cache_root: Path,
    device: torch.device,
    dtype: torch.dtype,
    allow_compile: bool,
) -> dict[str, Any]:
    hidden_size = int(model.config.vision_config.hidden_size)
    heads = int(model.config.vision_config.num_attention_heads)
    head_dim = hidden_size // heads
    missing: list[int] = []
    paths: dict[str, str] = {}
    for bucket in OPTIMIZED_VISION_BUCKETS:
        cache_dir = vision_cache_dir_for_bucket(
            cache_root,
            bucket=int(bucket),
            dtype=dtype,
            device=device,
            model_dir=model_dir,
            attention_impl="prompt_flash_attention",
            head_dim=head_dim,
        )
        paths[str(bucket)] = str(cache_dir)
        if not cache_dir.is_dir() or not any(cache_dir.rglob("*")):
            missing.append(int(bucket))
    if missing and not allow_compile:
        raise RuntimeError(
            "warm-cache preflight failed for buckets "
            f"{missing}; pass --allow-compile only when new compilation is intended"
        )
    return {"passed": not missing, "missing_buckets": missing, "paths": paths}


def _run_lane(
    *,
    name: str,
    groups: list[list[dict[str, Any]]],
    model: LocalPaddleOCRVLForConditionalGeneration,
    runtime: VisionPrefillRuntime,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
    progress_every: int,
) -> dict[str, Any]:
    if name not in LANES:
        raise ValueError(f"unknown replay lane {name!r}")
    stage_s: Counter[str] = Counter()
    route_histogram: Counter[str] = Counter()
    group_histogram: Counter[int] = Counter()
    real_tokens = 0
    physical_tokens = 0
    overflow_groups = 0
    wall_started = time.perf_counter()
    for group_index, group in enumerate(groups, 1):
        materialized = [
            _materialize(item, seed=seed, dtype=dtype, device=device)
            for item in group
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
        total_real = sum(lengths)
        route = runtime.route(total_real)
        if name == "packed" and len(group) > 1:
            prepared = timeline.measure(
                "vision_prefill_input_prep",
                lambda: runtime.prepare_packed(hidden, grids, route=route),
            )
            output = timeline.measure(
                "vision_tower",
                lambda: runtime.run_prepared(prepared.prepared),
            )
            segments = torch.split(output, prepared.segment_lengths, dim=0)
            if [int(value.shape[0]) for value in segments] != lengths:
                raise RuntimeError(f"packed split mismatch in group {group_index}")
        else:
            if len(group) != 1:
                raise ValueError("single replay lane received a multi-crop group")
            prepared_single = timeline.measure(
                "vision_prefill_input_prep",
                lambda: runtime.prepare(hidden[0], grids[0], route=route),
            )
            output = timeline.measure(
                "vision_tower",
                lambda: runtime.run_prepared(prepared_single),
            )
            if int(output.shape[0]) != lengths[0]:
                raise RuntimeError(f"single output mismatch in group {group_index}")
        spans = timeline.resolve_spans()
        for stage, span in spans.items():
            stage_s[stage] += float(span["seconds"])
        real_tokens += total_real
        physical_tokens += int(route["physical_vision_tokens"])
        route_key = str(route["bucket"] or route["execution"])
        route_histogram[route_key] += 1
        group_histogram[len(group)] += 1
        overflow_groups += int(route["execution"] == "eager_overflow")
        if progress_every and (
            group_index % progress_every == 0 or group_index == len(groups)
        ):
            print(
                f"lane={name} groups={group_index}/{len(groups)} "
                f"tower_s={stage_s['vision_tower']:.3f}",
                flush=True,
            )
        del output, hidden, pixels, grids, materialized
    synchronize(device)
    wall_s = time.perf_counter() - wall_started
    tower_s = float(stage_s["vision_tower"])
    return {
        "groups": len(groups),
        "crops": sum(len(group) for group in groups),
        "crops_per_group": sum(len(group) for group in groups) / len(groups),
        "group_size_histogram": dict(sorted(group_histogram.items())),
        "route_histogram": dict(sorted(route_histogram.items())),
        "eager_overflow_groups": overflow_groups,
        "real_vision_tokens": real_tokens,
        "physical_vision_tokens": physical_tokens,
        "padding_vision_tokens": physical_tokens - real_tokens,
        "real_token_fraction": real_tokens / physical_tokens,
        "device_stage_s": dict(stage_s),
        "vision_tower_s": tower_s,
        "effective_real_tokens_per_s": real_tokens / tower_s,
        "raw_physical_tokens_per_s": physical_tokens / tower_s,
        "lab_wall_s": wall_s,
    }


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    import torch_npu  # noqa: F401

    if not torch.npu.is_available():
        raise RuntimeError("E2E vision replay requires an NPU")
    device = torch.device("npu:0")
    dtype = torch.float16
    torch.npu.set_compile_mode(jit_compile=False)
    items = _load_items(args.corpus, args.variant)
    packed_groups, trace_summary = _load_packed_groups(args.packed_trace, items)
    single_groups = [[item] for item in sorted(items, key=lambda row: row["source_index"])]
    references = {
        "single": _summary_reference(args.single_summary),
        "packed": _summary_reference(args.packed_summary),
    }
    model_dir = args.model.expanduser().resolve()
    cache_root = args.cache_dir.expanduser().resolve()
    synchronize(device)
    setup_started = time.perf_counter()
    model = LocalPaddleOCRVLForConditionalGeneration.from_pretrained(
        model_dir,
        dtype=dtype,
        device=device,
    )
    cache_preflight = _warm_cache_preflight(
        model=model,
        model_dir=model_dir,
        cache_root=cache_root,
        device=device,
        dtype=dtype,
        allow_compile=bool(args.allow_compile),
    )
    runtime = VisionPrefillRuntime(
        model,
        backend="torchair",
        buckets=OPTIMIZED_VISION_BUCKETS,
        cache_root=cache_root,
        device=device,
        dtype=dtype,
        model_dir=model_dir,
        attention_impl="prompt_flash_attention",
        padding="bucket",
    )
    synchronize(device)
    setup_s = time.perf_counter() - setup_started

    results: dict[str, Any] = {}
    for lane in args.lane:
        groups = single_groups if lane == "single" else packed_groups
        result = _run_lane(
            name=lane,
            groups=groups,
            model=model,
            runtime=runtime,
            seed=args.seed,
            dtype=dtype,
            device=device,
            progress_every=args.progress_every,
        )
        reference = references[lane]
        for key in ("real_vision_tokens", "physical_vision_tokens", "groups"):
            if int(result[key]) != int(reference[key]):
                raise RuntimeError(
                    f"{lane} replay/reference {key} mismatch: "
                    f"{result[key]} != {reference[key]}"
                )
        reference_s = float(reference["vision_tower_s"])
        result["e2e_reference"] = reference
        result["vision_tower_delta_vs_e2e_s"] = result["vision_tower_s"] - reference_s
        result["vision_tower_ratio_vs_e2e"] = result["vision_tower_s"] / reference_s
        results[lane] = result

    comparison = None
    if set(LANES).issubset(results):
        single = results["single"]
        packed = results["packed"]
        comparison = {
            "vision_tower_s_saved": single["vision_tower_s"]
            - packed["vision_tower_s"],
            "vision_tower_speedup": single["vision_tower_s"]
            / packed["vision_tower_s"],
            "vision_tower_reduction_fraction": 1.0
            - packed["vision_tower_s"] / single["vision_tower_s"],
        }

    payload = {
        "schema_version": 1,
        "purpose": "exact E2E crop-shape and pack-group replay in the vision lab",
        "inputs": {
            "corpus": str(args.corpus.expanduser().resolve()),
            "variant": args.variant,
            "packed_trace": str(args.packed_trace.expanduser().resolve()),
            "model": str(model_dir),
            "cache_dir": str(cache_root),
            "seed": args.seed,
            "tensor_policy": (
                "deterministic shape-equivalent random patch tensors; crop IDs, "
                "grids, routes, and group membership are exact"
            ),
        },
        "setup_s": setup_s,
        "cache_preflight": cache_preflight,
        "trace_plan": trace_summary,
        "results": results,
        "comparison": comparison,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("lane | groups | tower_s | e2e_reference_s | real_tok/s | physical_tok/s")
    for lane in args.lane:
        result = results[lane]
        print(
            f"{lane} | {result['groups']} | {result['vision_tower_s']:.3f} | "
            f"{result['e2e_reference']['vision_tower_s']:.3f} | "
            f"{result['effective_real_tokens_per_s']:.1f} | "
            f"{result['raw_physical_tokens_per_s']:.1f}"
        )
    if comparison is not None:
        print(
            "replay speedup="
            f"{comparison['vision_tower_speedup']:.3f}x "
            f"saved_s={comparison['vision_tower_s_saved']:.3f}"
        )
    print(f"output={output}")


if __name__ == "__main__":
    main()
