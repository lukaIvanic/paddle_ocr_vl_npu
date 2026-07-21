#!/usr/bin/env python3
"""Standalone PaddleOCR-VL vision-prefill experimentation harness."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import torch

HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
REPO_ROOT = EXPERIMENT_ROOT.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

from paddleocr_vl.model.modeling import LocalPaddleOCRVLForConditionalGeneration
from paddleocr_vl.model.vision_prefill import (
    PreparedVisionPrefill,
    VisionPrefillRuntime,
    build_vision_rope,
    parse_vision_buckets,
    prepare_vision_prefill,
    select_vision_bucket,
)
from paddleocr_vl.serving.runtime_defaults import OPTIMIZED_VISION_BUCKETS
from utils.timing import DeviceTimeline, synchronize


DEFAULT_MODEL = Path("/workspace/models/PaddleOCR-VL-1.6")
DEFAULT_CORPUS = (
    REPO_ROOT
    / "tmp/09_persistent_page_engine/vision_lab"
    / "corpus_recognition_trace.json"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "tmp/09_persistent_page_engine/vision_lab"
DEFAULT_CACHE_ROOT = REPO_ROOT / ".runtime_cache/09_vision_lab"
STRATEGIES = ("single", "batched", "packed")
ATTENTIONS = ("manual", "prompt_flash_attention")
EXECUTIONS = ("eager", "torchair")
CALIBRATION_BUCKET_MS = {
    640: 17.9,
    704: 20.0,
    768: 19.4,
    1024: 22.1,
    1920: 29.9,
}


def _csv_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(piece) for piece in value.split(",") if piece.strip())
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def _strategies(values: list[str] | None) -> tuple[str, ...]:
    pieces = [
        piece.strip()
        for value in (values or ["single"])
        for piece in value.split(",")
        if piece.strip()
    ]
    unknown = sorted(set(pieces) - set(STRATEGIES))
    if unknown:
        raise ValueError(f"unknown strategies: {unknown}; expected {STRATEGIES}")
    return tuple(dict.fromkeys(pieces))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compare",
        type=Path,
        nargs="+",
        help="Print a delta table from existing result JSON files and exit.",
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--name")
    parser.add_argument(
        "--strategy",
        action="append",
        help="Repeat or pass a comma list: single,batched,packed.",
    )
    parser.add_argument("--attention", choices=ATTENTIONS, default="manual")
    parser.add_argument("--execution", choices=EXECUTIONS, default="eager")
    parser.add_argument(
        "--buckets",
        type=_csv_ints,
        default=tuple(OPTIMIZED_VISION_BUCKETS),
    )
    parser.add_argument(
        "--pack-lengths",
        type=_csv_ints,
        default=(1280, 1792, 1920),
        help="Greedy composed real-token targets for packed/batched cases.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=_csv_ints,
        default=(2, 4),
        help="Batched strategy group limits.",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--max-abs", type=float, default=2e-2)
    parser.add_argument("--max-rel", type=float, default=2e-2)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_ROOT)
    args = parser.parse_args(argv)
    if args.compare:
        return args
    if args.warmup < 0 or args.repeats <= 0:
        parser.error("--warmup must be non-negative and --repeats positive")
    if any(size <= 1 for size in args.batch_sizes):
        parser.error("--batch-sizes values must be greater than one")
    args.strategy = _strategies(args.strategy)
    args.buckets = parse_vision_buckets(args.buckets)
    return args


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_corpus(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = path.expanduser().resolve()
    corpus = json.loads(path.read_text(encoding="utf-8"))
    if not corpus.get("self_check", {}).get("passed"):
        raise ValueError(f"corpus self-check is not marked passed: {path}")
    items = list(corpus.get("items", []))
    if not items:
        raise ValueError(f"corpus has no items: {path}")
    for index, item in enumerate(items):
        grid = tuple(int(value) for value in item["grid_thw"])
        if len(grid) != 3 or grid[0] != 1 or math.prod(grid) != int(
            item["real_vision_tokens"]
        ):
            raise ValueError(f"invalid corpus grid at item {index}: {item}")
        item["source_index"] = int(item.get("source_index", index))
    return corpus, items


def _first_fit_groups(
    items: list[dict[str, Any]],
    *,
    target: int,
    max_items: int | None,
    seed: int,
) -> list[list[dict[str, Any]]]:
    order = list(items)
    random.Random(seed).shuffle(order)
    order.sort(key=lambda item: int(item["real_vision_tokens"]), reverse=True)
    groups: list[list[dict[str, Any]]] = []
    totals: list[int] = []
    for item in order:
        length = int(item["real_vision_tokens"])
        if length > target:
            groups.append([item])
            totals.append(length)
            continue
        candidates = [
            index
            for index, group in enumerate(groups)
            if totals[index] <= target
            and totals[index] + length <= target
            and (max_items is None or len(group) < max_items)
        ]
        if candidates:
            selected = max(candidates, key=lambda index: totals[index])
            groups[selected].append(item)
            totals[selected] += length
        else:
            groups.append([item])
            totals.append(length)
    if sum(len(group) for group in groups) != len(items):
        raise AssertionError("greedy composition lost corpus items")
    return groups


def _configuration_groups(
    strategies: Iterable[str],
    items: list[dict[str, Any]],
    pack_lengths: Iterable[int],
    batch_sizes: Iterable[int],
    seed: int,
) -> list[dict[str, Any]]:
    configurations: list[dict[str, Any]] = []
    for strategy in strategies:
        if strategy == "single":
            configurations.append(
                {
                    "name": "single",
                    "strategy": strategy,
                    "target_length": None,
                    "batch_size": 1,
                    "groups": [[item] for item in items],
                }
            )
        elif strategy == "packed":
            for target in pack_lengths:
                configurations.append(
                    {
                        "name": f"packed_target{target}",
                        "strategy": strategy,
                        "target_length": int(target),
                        "batch_size": None,
                        "groups": _first_fit_groups(
                            items,
                            target=int(target),
                            max_items=None,
                            seed=seed + int(target),
                        ),
                    }
                )
        else:
            for target in pack_lengths:
                for batch_size in batch_sizes:
                    configurations.append(
                        {
                            "name": f"batched_b{batch_size}_target{target}",
                            "strategy": strategy,
                            "target_length": int(target),
                            "batch_size": int(batch_size),
                            "groups": _first_fit_groups(
                                items,
                                target=int(target),
                                max_items=int(batch_size),
                                seed=seed + int(target) + int(batch_size),
                            ),
                        }
                    )
    return configurations


def _route(
    *,
    strategy: str,
    lengths: list[int],
    execution: str,
    buckets: tuple[int, ...],
) -> dict[str, Any]:
    real = max(lengths) if strategy == "batched" else sum(lengths)
    bucket = select_vision_bucket(real, buckets) if execution == "torchair" else None
    return {
        "execution": "compiled" if bucket is not None else "eager_overflow" if execution == "torchair" else "eager",
        "real_vision_tokens": real,
        "physical_vision_tokens": bucket if bucket is not None else real,
        "padding_vision_tokens": (bucket - real) if bucket is not None else 0,
        "bucket": bucket,
    }


def _needed_compiled_buckets(
    configurations: list[dict[str, Any]],
    execution: str,
    buckets: tuple[int, ...],
) -> tuple[int, ...]:
    if execution != "torchair":
        return (buckets[0],)
    needed: set[int] = set()
    for configuration in configurations:
        for group in configuration["groups"]:
            route = _route(
                strategy=configuration["strategy"],
                lengths=[int(item["real_vision_tokens"]) for item in group],
                execution=execution,
                buckets=buckets,
            )
            if route["bucket"] is not None:
                needed.add(int(route["bucket"]))
    return tuple(sorted(needed)) or (buckets[0],)


def _make_pixels(
    item: dict[str, Any],
    *,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + int(item["source_index"]) * 104729)
    tokens = int(item["real_vision_tokens"])
    pixels = torch.randn(
        (tokens, 3, 14, 14),
        generator=generator,
        dtype=dtype,
    )
    return pixels.to(device=device)


def _packed_prepared(
    model: LocalPaddleOCRVLForConditionalGeneration,
    hidden: list[torch.Tensor],
    grids: list[torch.Tensor],
    *,
    physical: int,
    execution: str,
) -> PreparedVisionPrefill:
    lengths = [int(value.shape[0]) for value in hidden]
    total = sum(lengths)
    ropes = [
        build_vision_rope(
            model,
            grid,
            real_seq_len=length,
            device=hidden[0].device,
        )
        for hidden_value, grid, length in zip(hidden, grids, lengths)
    ]
    prefix = torch.cat(hidden, dim=0)
    cos = torch.cat([pair[0] for pair in ropes], dim=0)
    sin = torch.cat([pair[1] for pair in ropes], dim=0)
    pad = physical - total
    if pad < 0:
        raise ValueError(f"packed real length {total} exceeds physical length {physical}")
    if pad:
        prefix = torch.nn.functional.pad(prefix, (0, 0, 0, pad))
        cos = torch.cat(
            [
                cos,
                torch.ones((pad, cos.shape[-1]), device=cos.device, dtype=cos.dtype),
            ]
        )
        sin = torch.cat([sin, torch.zeros_like(cos[-pad:])])
    segment_ids = torch.cat(
        [
            torch.full((length,), index, device=prefix.device, dtype=torch.int32)
            for index, length in enumerate(lengths)
        ]
        + ([torch.full((pad,), -1, device=prefix.device, dtype=torch.int32)] if pad else [])
    )
    mask = (segment_ids[:, None] != segment_ids[None, :]).view(
        1, 1, physical, physical
    )
    return PreparedVisionPrefill(
        prefix_hidden_states=prefix.unsqueeze(0).contiguous(),
        rope_cos=cos.unsqueeze(0).contiguous(),
        rope_sin=sin.unsqueeze(0).contiguous(),
        attention_mask=mask.contiguous(),
        real_seq_len=total,
        physical_seq_len=physical,
        execution=execution,
    )


def _batched_prepared(
    model: LocalPaddleOCRVLForConditionalGeneration,
    hidden: list[torch.Tensor],
    grids: list[torch.Tensor],
    *,
    physical: int,
    execution: str,
) -> PreparedVisionPrefill:
    rows = [
        prepare_vision_prefill(
            model,
            hidden_value,
            grid,
            physical_seq_len=physical,
            execution=execution,
        )
        for hidden_value, grid in zip(hidden, grids)
    ]
    return PreparedVisionPrefill(
        prefix_hidden_states=torch.cat(
            [row.prefix_hidden_states for row in rows], dim=0
        ),
        rope_cos=torch.cat([row.rope_cos for row in rows], dim=0),
        rope_sin=torch.cat([row.rope_sin for row in rows], dim=0),
        attention_mask=torch.cat([row.attention_mask for row in rows], dim=0),
        real_seq_len=max(row.real_seq_len for row in rows),
        physical_seq_len=physical,
        execution=execution,
    )


def _run_group(
    model: LocalPaddleOCRVLForConditionalGeneration,
    runtime: VisionPrefillRuntime,
    group: list[dict[str, Any]],
    *,
    strategy: str,
    route: dict[str, Any],
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, Any]:
    grids = [torch.tensor([item["grid_thw"]], dtype=torch.long) for item in group]
    pixels = [
        _make_pixels(item, seed=seed, dtype=dtype, device=device) for item in group
    ]
    timeline = DeviceTimeline(device)
    enqueue_started = time.perf_counter()
    hidden = timeline.measure(
        "patch_position_embeddings",
        lambda: [
            model.visual.vision_model.embeddings(
                pixel.unsqueeze(0),
                image_grid_thw=grid,
            )
            for pixel, grid in zip(pixels, grids)
        ],
    )
    physical = int(route["physical_vision_tokens"])
    execution = str(route["execution"])
    lengths = [int(value.shape[0]) for value in hidden]
    if strategy == "single":
        prepared = runtime.prepare(hidden[0], grids[0], route=route)
        tower_segments = timeline.measure(
            "vision_tower",
            lambda: [runtime.run_prepared(prepared)],
        )
    elif strategy == "packed":
        prepared = _packed_prepared(
            model,
            hidden,
            grids,
            physical=physical,
            execution=execution,
        )
        packed_output = timeline.measure(
            "vision_tower",
            lambda: runtime.run_prepared(prepared),
        )
        tower_segments = list(torch.split(packed_output, lengths, dim=0))
    else:
        prepared = _batched_prepared(
            model,
            hidden,
            grids,
            physical=physical,
            execution=execution,
        )
        run: Callable[..., torch.Tensor] = (
            runtime.compiled[physical]
            if execution == "compiled"
            else runtime.eager_stage
        )
        batched_output = timeline.measure(
            "vision_tower",
            lambda: run(
                prepared.prefix_hidden_states,
                prepared.rope_cos,
                prepared.rope_sin,
                prepared.attention_mask,
            ),
        )
        tower_segments = [
            batched_output[index, :length].contiguous()
            for index, length in enumerate(lengths)
        ]
    projector_segments = timeline.measure(
        "adaptive_mlp_projector",
        lambda: [
            model.mlp_AR(tower, grid)
            for tower, grid in zip(tower_segments, grids)
        ],
    )
    enqueue_wall_ms = (time.perf_counter() - enqueue_started) * 1000.0
    spans = timeline.resolve_spans()
    stage_ms = {
        name: float(span["seconds"]) * 1000.0 for name, span in spans.items()
    }
    stage_ms["total"] = sum(stage_ms.values())
    return {
        "tower": tower_segments,
        "projector": projector_segments,
        "stage_ms": stage_ms,
        "host_enqueue_ms": enqueue_wall_ms,
    }


def _cpu_outputs(run: dict[str, Any]) -> dict[str, list[torch.Tensor]]:
    return {
        key: [tensor.detach().cpu() for tensor in run[key]]
        for key in ("tower", "projector")
    }


def _error_accumulator() -> dict[str, dict[str, float | int]]:
    return {
        "tower": {"max_abs": 0.0, "max_rel": 0.0, "abs_sum": 0.0, "count": 0},
        "projector": {"max_abs": 0.0, "max_rel": 0.0, "abs_sum": 0.0, "count": 0},
    }


def _accumulate_errors(
    accumulator: dict[str, dict[str, float | int]],
    actual: dict[str, list[torch.Tensor]],
    reference: list[dict[str, torch.Tensor]],
) -> None:
    for output_name in ("tower", "projector"):
        for actual_tensor, reference_item in zip(actual[output_name], reference):
            reference_tensor = reference_item[output_name]
            delta = (actual_tensor.float() - reference_tensor.float()).abs()
            relative = delta / reference_tensor.float().abs().clamp_min(1e-6)
            values = accumulator[output_name]
            values["max_abs"] = max(float(values["max_abs"]), float(delta.max()))
            values["max_rel"] = max(float(values["max_rel"]), float(relative.max()))
            values["abs_sum"] = float(values["abs_sum"]) + float(delta.sum())
            values["count"] = int(values["count"]) + delta.numel()


def _finalize_errors(
    accumulator: dict[str, dict[str, float | int]],
) -> dict[str, dict[str, float]]:
    return {
        name: {
            "max_abs": float(values["max_abs"]),
            "max_rel": float(values["max_rel"]),
            "mean_abs": (
                float(values["abs_sum"]) / int(values["count"])
                if int(values["count"])
                else 0.0
            ),
        }
        for name, values in accumulator.items()
    }


def _memory_baseline(device: torch.device) -> int:
    try:
        torch.npu.reset_peak_memory_stats(device)
        return int(torch.npu.memory_allocated(device))
    except Exception:
        return 0


def _peak_memory_delta(device: torch.device, baseline: int) -> int | None:
    try:
        return max(0, int(torch.npu.max_memory_allocated(device)) - baseline)
    except Exception:
        return None


def _environment(device: torch.device) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except Exception:
        commit = None
    try:
        npu_smi = subprocess.run(
            ["npu-smi", "info"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout[:12000]
    except Exception as exc:
        npu_smi = f"unavailable: {exc!r}"
    return {
        "commit": commit,
        "device": str(device),
        "device_name": torch.npu.get_device_name(device),
        "torch": torch.__version__,
        "torch_npu": getattr(sys.modules.get("torch_npu"), "__version__", None),
        "npu_smi": npu_smi,
    }


def _compare_mode(paths: list[Path]) -> None:
    rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for configuration in payload.get("configurations", []):
            projection = configuration.get("corpus_projection", {})
            rows.append(
                {
                    "run": path.stem,
                    "configuration": configuration.get("name"),
                    "supported": configuration.get("supported"),
                    "valid": configuration.get("numerics", {}).get("valid"),
                    "tower_s": projection.get("vision_tower_s"),
                    "total_s": projection.get("total_stages_s"),
                    "ratio": projection.get("ratio_vs_single"),
                }
            )
            for case in configuration.get("cases", []):
                case_rows.append(
                    {
                        "run": path.stem,
                        "configuration": configuration.get("name"),
                        "case": case.get("case_index"),
                        "real": case.get("real_tokens"),
                        "physical": case.get("physical_tokens"),
                        "tower_ms": case.get("device_ms", {})
                        .get("vision_tower", {})
                        .get("mean"),
                        "total_ms": case.get("device_ms", {})
                        .get("total", {})
                        .get("mean"),
                    }
                )
    baseline = next(
        (
            float(row["tower_s"])
            for row in rows
            if row["configuration"] == "single"
            and row["supported"]
            and row["tower_s"] is not None
        ),
        None,
    )
    if baseline is not None:
        for row in rows:
            if row["tower_s"] is not None:
                row["ratio"] = float(row["tower_s"]) / baseline
    headers = ("run", "configuration", "supported", "valid", "tower_s", "total_s", "ratio")
    print(" | ".join(headers))
    print(" | ".join("---" for _ in headers))
    for row in rows:
        print(" | ".join(str(row.get(header)) for header in headers))
    print("\nPer-case device means")
    case_headers = (
        "run",
        "configuration",
        "case",
        "real",
        "physical",
        "tower_ms",
        "total_ms",
    )
    print(" | ".join(case_headers))
    print(" | ".join("---" for _ in case_headers))
    for row in case_rows:
        print(" | ".join(str(row.get(header)) for header in case_headers))


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.compare:
        _compare_mode(args.compare)
        return

    import torch_npu

    device = torch.device("npu:0")
    if not torch.npu.is_available():
        raise RuntimeError("vision lab requires an available NPU")
    torch.npu.set_compile_mode(jit_compile=False)
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    corpus, items = _load_corpus(args.corpus)
    configurations = _configuration_groups(
        args.strategy,
        items,
        args.pack_lengths,
        args.batch_sizes,
        args.seed,
    )
    compile_buckets = _needed_compiled_buckets(
        configurations,
        args.execution,
        args.buckets,
    )

    synchronize(device)
    setup_started = time.perf_counter()
    model = LocalPaddleOCRVLForConditionalGeneration.from_pretrained(
        args.model.expanduser().resolve(),
        dtype=dtype,
        device=device,
    )
    runtime = VisionPrefillRuntime(
        model,
        backend="torchair" if args.execution == "torchair" else "raw_eager",
        buckets=compile_buckets,
        cache_root=args.cache_dir,
        device=device,
        dtype=dtype,
        model_dir=args.model.expanduser().resolve(),
        attention_impl=args.attention,
        padding="bucket" if args.execution == "torchair" else "none",
    )
    reference_runtime = VisionPrefillRuntime(
        model,
        backend="raw_eager",
        buckets=(args.buckets[-1],),
        cache_root=args.cache_dir / "reference",
        device=device,
        dtype=dtype,
        model_dir=args.model.expanduser().resolve(),
        attention_impl="manual",
        padding="none",
    )
    synchronize(device)
    setup_s = time.perf_counter() - setup_started
    reference_cache: dict[int, dict[str, torch.Tensor]] = {}

    def reference_for(item: dict[str, Any]) -> dict[str, torch.Tensor]:
        index = int(item["source_index"])
        if index not in reference_cache:
            reference_route = _route(
                strategy="single",
                lengths=[int(item["real_vision_tokens"])],
                execution="eager",
                buckets=args.buckets,
            )
            run = _run_group(
                model,
                reference_runtime,
                [item],
                strategy="single",
                route=reference_route,
                seed=args.seed,
                dtype=dtype,
                device=device,
            )
            reference_cache[index] = {
                "tower": run["tower"][0].detach().cpu(),
                "projector": run["projector"][0].detach().cpu(),
            }
        return reference_cache[index]

    result_configurations: list[dict[str, Any]] = []
    single_projection: float | None = None
    for configuration in configurations:
        errors = _error_accumulator()
        cases: list[dict[str, Any]] = []
        deterministic = True
        supported = True
        failure: dict[str, Any] | None = None
        for case_index, group in enumerate(configuration["groups"]):
            lengths = [int(item["real_vision_tokens"]) for item in group]
            route = _route(
                strategy=configuration["strategy"],
                lengths=lengths,
                execution=args.execution,
                buckets=args.buckets,
            )
            try:
                for _ in range(args.warmup):
                    _run_group(
                        model,
                        runtime,
                        group,
                        strategy=configuration["strategy"],
                        route=route,
                        seed=args.seed,
                        dtype=dtype,
                        device=device,
                    )
                memory_baseline = _memory_baseline(device)
                run_records: list[dict[str, Any]] = []
                last_run: dict[str, Any] | None = None
                for _ in range(args.repeats):
                    current_run = _run_group(
                        model,
                        runtime,
                        group,
                        strategy=configuration["strategy"],
                        route=route,
                        seed=args.seed,
                        dtype=dtype,
                        device=device,
                    )
                    run_records.append(
                        {
                            "stage_ms": dict(current_run["stage_ms"]),
                            "host_enqueue_ms": float(current_run["host_enqueue_ms"]),
                        }
                    )
                    last_run = current_run
                assert last_run is not None
                peak_bytes = _peak_memory_delta(device, memory_baseline)
                determinism_run = _run_group(
                    model,
                    runtime,
                    group,
                    strategy=configuration["strategy"],
                    route=route,
                    seed=args.seed,
                    dtype=dtype,
                    device=device,
                )
                for output_name in ("tower", "projector"):
                    deterministic = deterministic and all(
                        torch.equal(left, right)
                        for left, right in zip(
                            last_run[output_name], determinism_run[output_name]
                        )
                    )
                actual = _cpu_outputs(last_run)
                references = [reference_for(item) for item in group]
                _accumulate_errors(errors, actual, references)
                stage_samples = {
                    stage: [float(run["stage_ms"][stage]) for run in run_records]
                    for stage in run_records[0]["stage_ms"]
                }
                total_real = sum(lengths)
                physical_per_call = (
                    int(route["physical_vision_tokens"]) * len(group)
                    if configuration["strategy"] == "batched"
                    else int(route["physical_vision_tokens"])
                )
                tower_mean = statistics.mean(stage_samples["vision_tower"])
                cases.append(
                    {
                        "case_index": case_index,
                        "items": [item["name"] for item in group],
                        "item_real_tokens": lengths,
                        "real_tokens": total_real,
                        "physical_tokens": physical_per_call,
                        "route": route,
                        "device_ms": {
                            stage: {
                                "mean": statistics.mean(samples),
                                "p50": _percentile(samples, 0.50),
                                "p95": _percentile(samples, 0.95),
                            }
                            for stage, samples in stage_samples.items()
                        },
                        "host_enqueue_ms": {
                            "mean": statistics.mean(
                                float(run["host_enqueue_ms"]) for run in run_records
                            ),
                            "p50": _percentile(
                                [float(run["host_enqueue_ms"]) for run in run_records], 0.50
                            ),
                            "p95": _percentile(
                                [float(run["host_enqueue_ms"]) for run in run_records], 0.95
                            ),
                        },
                        "real_tokens_per_s": total_real / (tower_mean / 1000.0),
                        "ms_per_1k_physical_tokens": (
                            tower_mean * 1000.0 / physical_per_call
                        ),
                        "peak_allocated_bytes_delta": peak_bytes,
                    }
                )
                del last_run, determinism_run
            except Exception as exc:
                supported = False
                failure = {
                    "case_index": case_index,
                    "items": [item["name"] for item in group],
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc().splitlines()[-20:],
                }
                break

        numerical_errors = _finalize_errors(errors)
        valid = (
            supported
            and numerical_errors["projector"]["max_abs"] <= args.max_abs
            and numerical_errors["projector"]["max_rel"] <= args.max_rel
        )
        tower_s = (
            sum(case["device_ms"]["vision_tower"]["mean"] for case in cases)
            / 1000.0
            if supported
            else None
        )
        total_s = (
            sum(case["device_ms"]["total"]["mean"] for case in cases) / 1000.0
            if supported
            else None
        )
        if configuration["strategy"] == "single" and tower_s is not None:
            single_projection = tower_s
        result_configurations.append(
            {
                "name": configuration["name"],
                "strategy": configuration["strategy"],
                "target_length": configuration["target_length"],
                "batch_size": configuration["batch_size"],
                "supported": supported,
                "failure": failure,
                "groups": len(configuration["groups"]),
                "completed_groups": len(cases),
                "numerics": {
                    "reference": "single_eager_manual_unpadded",
                    "thresholds": {
                        "projector_max_abs": args.max_abs,
                        "projector_max_rel": args.max_rel,
                    },
                    "errors": numerical_errors,
                    "valid": valid,
                },
                "determinism": {
                    "expected": "bitwise_equal",
                    "passed": deterministic if supported else False,
                },
                "corpus_projection": {
                    "vision_tower_s": tower_s,
                    "total_stages_s": total_s,
                    "ratio_vs_single": (
                        tower_s / single_projection
                        if tower_s is not None and single_projection is not None
                        else None
                    ),
                    "in_pipeline_reference_s": 10.7,
                },
                "cases": cases,
            }
        )

    if single_projection is not None:
        for configuration in result_configurations:
            projection = configuration["corpus_projection"]
            if projection["vision_tower_s"] is not None:
                projection["ratio_vs_single"] = (
                    projection["vision_tower_s"] / single_projection
                )

    calibration: dict[str, Any] | None = None
    for configuration in result_configurations:
        if (
            configuration["strategy"] == "single"
            and args.execution == "torchair"
            and args.attention == "prompt_flash_attention"
        ):
            by_bucket: dict[int, list[float]] = defaultdict(list)
            for case in configuration["cases"]:
                bucket = case["route"]["bucket"]
                if bucket is not None:
                    by_bucket[int(bucket)].append(
                        float(case["device_ms"]["vision_tower"]["mean"])
                    )
            comparisons = {}
            for bucket, reference_ms in CALIBRATION_BUCKET_MS.items():
                if bucket not in by_bucket:
                    continue
                measured_ms = statistics.mean(by_bucket[bucket])
                comparisons[str(bucket)] = {
                    "calls": len(by_bucket[bucket]),
                    "measured_mean_ms": measured_ms,
                    "pipeline_reference_ms": reference_ms,
                    "deviation_fraction": measured_ms / reference_ms - 1.0,
                    "within_15_percent": abs(measured_ms / reference_ms - 1.0) <= 0.15,
                }
            calibration = {
                "corpus_self_check": corpus["self_check"],
                "bucket_comparisons": comparisons,
                "numerics_valid": configuration["numerics"]["valid"],
                "determinism_passed": configuration["determinism"]["passed"],
                "passed": (
                    bool(comparisons)
                    and all(item["within_15_percent"] for item in comparisons.values())
                    and configuration["numerics"]["valid"]
                    and configuration["determinism"]["passed"]
                ),
            }
            break

    payload = {
        "schema_version": 1,
        "created_at_unix_s": time.time(),
        "corpus": {
            "path": str(args.corpus.expanduser().resolve()),
            "sha256": _sha256(args.corpus.expanduser().resolve()),
            "kind": corpus.get("kind"),
            "items": len(items),
            "real_vision_tokens": sum(
                int(item["real_vision_tokens"]) for item in items
            ),
            "self_check": corpus["self_check"],
        },
        "config": {
            "model": str(args.model.expanduser().resolve()),
            "dtype": args.dtype,
            "strategies": list(args.strategy),
            "attention": args.attention,
            "execution": args.execution,
            "configured_buckets": list(args.buckets),
            "compiled_buckets": list(compile_buckets),
            "pack_lengths": list(args.pack_lengths),
            "batch_sizes": list(args.batch_sizes),
            "warmup": args.warmup,
            "repeats": args.repeats,
            "seed": args.seed,
            "cache_dir": str(args.cache_dir.expanduser().resolve()),
        },
        "setup_s": setup_s,
        "runtime_metadata": runtime.metadata,
        "environment": _environment(device),
        "calibration": calibration,
        "configurations": result_configurations,
    }
    name = args.name or (
        f"vision_lab_{args.execution}_{args.attention}_"
        + "-".join(args.strategy)
        + f"_{int(time.time())}"
    )
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else (DEFAULT_OUTPUT_ROOT / f"{name}.json").resolve()
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"output={output}")
    print("configuration | supported | valid | deterministic | tower_s | ratio")
    print("--- | --- | --- | --- | --- | ---")
    for configuration in result_configurations:
        projection = configuration["corpus_projection"]
        print(
            " | ".join(
                str(value)
                for value in (
                    configuration["name"],
                    configuration["supported"],
                    configuration["numerics"]["valid"],
                    configuration["determinism"]["passed"],
                    projection["vision_tower_s"],
                    projection["ratio_vs_single"],
                )
            )
        )
    if calibration is not None:
        print("calibration=" + json.dumps(calibration, indent=2))


if __name__ == "__main__":
    main()
