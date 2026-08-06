#!/usr/bin/env python3
"""Corpus-driven MinerU vision-prefill throughput and parity lab.

The standard layout lane resizes real OmniDocBench pages to 1036x1036 before
the MinerU processor runs.  CPU preprocessing and H2D are measured separately.
The headline throughput measures only ``model.get_image_features`` on tensors
that are already resident on the NPU.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from PIL import Image

from local_modeling_mineru import LocalMinerU2_5ForConditionalGeneration
from prefill_timing import PrefillDeviceTimeline
from run_transformers_recognition_smoke import configure_npu, synchronize
from vision_prefill_compile import (
    DEFAULT_VISION_BUCKETS,
    MinerUVisionPrefillRuntime,
    parse_vision_buckets,
    select_vision_bucket,
)


DEFAULT_MODEL = Path("/workspace/models/MinerU2.5-Pro-2605-1.2B")
DEFAULT_DATASET_JSON = Path("/workspace/datasets/OmniDocBench/OmniDocBench.json")
DEFAULT_IMAGES_DIR = Path("/workspace/datasets/OmniDocBench/images")
DEFAULT_CACHE_DIR = Path(
    ".runtime_cache/11_mineru_2_5_pro_inference/vision_prefill_b1_fp16"
)
DEFAULT_OUTPUT = Path(
    "tmp/11_mineru_2_5_pro_inference/vision_prefill_lab/layout_1036/result.json"
)
EXECUTIONS = ("eager", "torchair")


def _csv_executions(values: list[str] | None) -> tuple[str, ...]:
    pieces = [
        piece.strip()
        for value in (values or ["eager,torchair"])
        for piece in value.split(",")
        if piece.strip()
    ]
    unknown = sorted(set(pieces) - set(EXECUTIONS))
    if unknown:
        raise ValueError(f"unknown executions {unknown}; expected {EXECUTIONS}")
    return tuple(dict.fromkeys(pieces))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset-json", type=Path, default=DEFAULT_DATASET_JSON)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument(
        "--image",
        type=Path,
        action="append",
        default=[],
        help="Use explicit real images instead of selecting OmniDocBench pages.",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument(
        "--layout-size",
        type=int,
        nargs=2,
        default=(1036, 1036),
        metavar=("W", "H"),
    )
    parser.add_argument(
        "--execution",
        action="append",
        help="Repeat or pass a comma list: eager,torchair (default: both).",
    )
    parser.add_argument(
        "--buckets",
        default=",".join(str(value) for value in DEFAULT_VISION_BUCKETS),
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--warmup-pages", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--parity-pages",
        type=int,
        default=2,
        help="Compare final eager and compiled image features for this many pages.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.execution = _csv_executions(args.execution)
    args.buckets = parse_vision_buckets(args.buckets)
    if args.offset < 0 or args.limit <= 0:
        parser.error("--offset must be non-negative and --limit must be positive")
    if args.warmup_pages < 0 or args.repeats <= 0 or args.parity_pages < 0:
        parser.error("warmup/parity counts must be non-negative; repeats must be positive")
    if any(value <= 0 for value in args.layout_size):
        parser.error("--layout-size values must be positive")
    return args


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


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


def _summary(values: Iterable[float]) -> dict[str, float]:
    samples = [float(value) for value in values]
    if not samples:
        return {}
    return {
        "count": len(samples),
        "total": sum(samples),
        "mean": statistics.mean(samples),
        "min": min(samples),
        "p50": _percentile(samples, 0.50),
        "p90": _percentile(samples, 0.90),
        "max": max(samples),
    }


def _dataset_images(args: argparse.Namespace) -> list[tuple[str, Path]]:
    if args.image:
        return [
            (path.expanduser().resolve().name, path.expanduser().resolve())
            for path in args.image
        ]
    dataset_path = args.dataset_json.expanduser().resolve()
    samples = json.loads(dataset_path.read_text(encoding="utf-8"))
    selected = samples[args.offset : args.offset + args.limit]
    if len(selected) != args.limit:
        raise ValueError(
            f"requested {args.limit} pages at offset {args.offset}; got {len(selected)}"
        )
    images_dir = args.images_dir.expanduser().resolve()
    result: list[tuple[str, Path]] = []
    for index, sample in enumerate(selected, args.offset):
        image_path = (sample.get("page_info") or {}).get("image_path")
        if not image_path:
            raise ValueError(f"dataset page {index} has no page_info.image_path")
        name = Path(image_path).name
        result.append((name, images_dir / name))
    return result


def _prepare_pages(
    processor: Any,
    pages: list[tuple[str, Path]],
    *,
    layout_size: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    cpu_times: list[float] = []
    h2d_times: list[float] = []
    for page_index, (name, path) in enumerate(pages, 1):
        cpu_started = time.perf_counter()
        with Image.open(path) as source:
            original_size = source.size
            image = source.convert("RGB").resize(layout_size, Image.Resampling.BICUBIC)
        inputs = processor.image_processor(images=[image], return_tensors="pt")
        pixel_values_cpu = inputs["pixel_values"]
        grid_cpu = inputs["image_grid_thw"]
        cpu_s = time.perf_counter() - cpu_started

        real_tokens = int(pixel_values_cpu.shape[0])
        grid_tokens = sum(
            math.prod(int(value) for value in row) for row in grid_cpu.tolist()
        )
        if real_tokens != grid_tokens:
            raise RuntimeError(
                f"{name}: pixel/grid token mismatch {real_tokens} != {grid_tokens}"
            )
        synchronize()
        h2d_started = time.perf_counter()
        pixel_values = pixel_values_cpu.to(device=device, dtype=dtype)
        grid_thw = grid_cpu.to(device=device)
        synchronize()
        h2d_s = time.perf_counter() - h2d_started
        cpu_times.append(cpu_s)
        h2d_times.append(h2d_s)
        prepared.append(
            {
                "page_index": page_index - 1,
                "name": name,
                "path": str(path),
                "original_size_wh": list(original_size),
                "processed_size_wh": list(layout_size),
                "grid_thw": grid_cpu.tolist(),
                "real_tokens": real_tokens,
                "pixel_values": pixel_values,
                "grid": grid_thw,
                "cpu_preprocess_s": cpu_s,
                "h2d_s": h2d_s,
            }
        )
        print(
            f"[prepare {page_index}/{len(pages)}] {name} "
            f"original={original_size} layout={layout_size} tokens={real_tokens} "
            f"cpu={cpu_s:.3f}s h2d={h2d_s:.3f}s",
            flush=True,
        )
    return prepared, {
        "cpu_preprocess_s": _summary(cpu_times),
        "h2d_s": _summary(h2d_times),
    }


def _run_one(
    model: LocalMinerU2_5ForConditionalGeneration,
    page: dict[str, Any],
) -> tuple[torch.Tensor, float, dict[str, float]]:
    timeline = PrefillDeviceTimeline(model.device)
    started = time.perf_counter()
    output = model.get_image_features(
        page["pixel_values"],
        page["grid"],
        device_timeline=timeline,
    )
    stages = timeline.resolve()
    wall_s = time.perf_counter() - started
    return output, wall_s, stages


def _physical_tokens(execution: str, real_tokens: int, buckets: tuple[int, ...]) -> int:
    if execution == "eager":
        return real_tokens
    return select_vision_bucket(real_tokens, buckets) or real_tokens


def _reset_runtime_counts(runtime: MinerUVisionPrefillRuntime | None) -> None:
    if runtime is None:
        return
    runtime.route_counts.clear()
    runtime.real_tokens = 0
    runtime.physical_tokens = 0


def _measure_execution(
    model: LocalMinerU2_5ForConditionalGeneration,
    pages: list[dict[str, Any]],
    *,
    execution: str,
    runtime: MinerUVisionPrefillRuntime | None,
    buckets: tuple[int, ...],
    warmup_pages: int,
    repeats: int,
) -> dict[str, Any]:
    model.set_vision_prefill_runtime(runtime if execution == "torchair" else None)
    warm_count = min(warmup_pages, len(pages))
    print(f"[{execution}] warming {warm_count} real pages", flush=True)
    for page in pages[:warm_count]:
        _run_one(model, page)
    _reset_runtime_counts(runtime)

    records: list[dict[str, Any]] = []
    stage_totals: dict[str, float] = defaultdict(float)
    total_real_tokens = 0
    total_physical_tokens = 0
    print(
        f"[{execution}] measuring {len(pages)} pages x {repeats} repeats",
        flush=True,
    )
    measured_started = time.perf_counter()
    for repeat in range(repeats):
        for page_number, page in enumerate(pages, 1):
            _output, wall_s, stages = _run_one(model, page)
            real_tokens = int(page["real_tokens"])
            physical_tokens = _physical_tokens(execution, real_tokens, buckets)
            total_real_tokens += real_tokens
            total_physical_tokens += physical_tokens
            for name, value in stages.items():
                stage_totals[name] += float(value)
            records.append(
                {
                    "repeat": repeat,
                    "page_index": int(page["page_index"]),
                    "name": page["name"],
                    "real_tokens": real_tokens,
                    "physical_tokens": physical_tokens,
                    "wall_s": wall_s,
                    "device_stages_s": stages,
                }
            )
            print(
                f"[{execution} {repeat + 1}/{repeats} {page_number}/{len(pages)}] "
                f"{wall_s * 1000:.2f}ms real={real_tokens} physical={physical_tokens}",
                flush=True,
            )
    measured_wall_s = time.perf_counter() - measured_started
    call_wall_s = sum(record["wall_s"] for record in records)
    device_total_s = sum(stage_totals.values())
    return {
        "execution": execution,
        "pages_per_repeat": len(pages),
        "repeats": repeats,
        "calls": len(records),
        "measured_loop_wall_s": measured_wall_s,
        "call_wall_s": call_wall_s,
        "pages_per_s": len(records) / call_wall_s,
        "real_tokens": total_real_tokens,
        "physical_tokens": total_physical_tokens,
        "useful_token_fraction": total_real_tokens / total_physical_tokens,
        "effective_tok_s": total_real_tokens / call_wall_s,
        "physical_tok_s": total_physical_tokens / call_wall_s,
        "call_wall_summary_s": _summary(record["wall_s"] for record in records),
        "device_stage_total_s": dict(sorted(stage_totals.items())),
        "device_stage_fraction": {
            name: value / device_total_s
            for name, value in sorted(stage_totals.items())
        },
        "records": records,
        "runtime": runtime.metadata() if execution == "torchair" and runtime else None,
    }


def _compare_features(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    left_float = left.float()
    right_float = right.float()
    delta = (right_float - left_float).abs()
    left_flat = left_float.flatten()
    right_flat = right_float.flatten()
    return {
        "shape": list(left.shape),
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
        "relative_l2": float(
            torch.linalg.vector_norm(right_flat - left_flat).item()
            / max(torch.linalg.vector_norm(left_flat).item(), 1e-12)
        ),
        "cosine": float(F.cosine_similarity(left_flat, right_flat, dim=0).item()),
        "nonfinite_eager": int((~torch.isfinite(left)).sum().item()),
        "nonfinite_torchair": int((~torch.isfinite(right)).sum().item()),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    print("[setup] configure NPU and load processor/model", flush=True)
    configure_npu()
    import torch_npu  # noqa: F401
    from transformers import AutoProcessor

    model_dir = args.model.expanduser().resolve()
    setup_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        model_dir,
        use_fast=True,
        local_files_only=True,
    )
    model = LocalMinerU2_5ForConditionalGeneration.from_pretrained(
        model_dir,
        dtype=torch.float16,
        device="npu:0",
    )
    model.set_vision_attention_impl("prompt_flash_attention")
    synchronize()
    setup_s = time.perf_counter() - setup_started
    print(f"[setup] complete in {setup_s:.3f}s", flush=True)

    page_paths = _dataset_images(args)
    pages, preparation = _prepare_pages(
        processor,
        page_paths,
        layout_size=tuple(args.layout_size),
        device=model.device,
        dtype=model.dtype,
    )
    unique_real_tokens = sorted({int(page["real_tokens"]) for page in pages})
    selected_buckets = {
        str(value): select_vision_bucket(value, args.buckets)
        for value in unique_real_tokens
    }
    print(
        f"[corpus] pages={len(pages)} token_shapes={unique_real_tokens} "
        f"routes={selected_buckets}",
        flush=True,
    )

    runtime = None
    if "torchair" in args.execution:
        runtime = MinerUVisionPrefillRuntime(
            model.visual,
            buckets=args.buckets,
            cache_root=args.cache_dir,
            model_dir=model_dir,
            device=model.device,
            dtype=model.dtype,
        )

    # Prime each requested lane before parity or measured work.  For TorchAir,
    # this isolates cache restore/compile and first-call cost from throughput.
    for execution in args.execution:
        model.set_vision_prefill_runtime(runtime if execution == "torchair" else None)
        for page in pages[: min(args.warmup_pages, len(pages))]:
            _run_one(model, page)
    _reset_runtime_counts(runtime)

    parity: list[dict[str, Any]] = []
    if runtime is not None and "eager" in args.execution and args.parity_pages:
        count = min(args.parity_pages, len(pages))
        print(f"[parity] comparing {count} real pages", flush=True)
        for page in pages[:count]:
            model.set_vision_prefill_runtime(None)
            eager_features, _wall, _stages = _run_one(model, page)
            model.set_vision_prefill_runtime(runtime)
            compiled_features, _wall, _stages = _run_one(model, page)
            comparison = _compare_features(eager_features, compiled_features)
            parity.append({"name": page["name"], **comparison})
            print(
                f"[parity] {page['name']} cosine={comparison['cosine']:.8f} "
                f"mean_abs={comparison['mean_abs']:.6f}",
                flush=True,
            )
    _reset_runtime_counts(runtime)

    results: dict[str, Any] = {}
    for execution in args.execution:
        results[execution] = _measure_execution(
            model,
            pages,
            execution=execution,
            runtime=runtime,
            buckets=args.buckets,
            warmup_pages=args.warmup_pages,
            repeats=args.repeats,
        )

    result = {
        "schema_version": 1,
        "kind": "mineru_vision_prefill_lab",
        "git_commit": _git_commit(),
        "device": "Ascend NPU",
        "dtype": "fp16",
        "attention": "prompt_flash_attention",
        "model": str(model_dir),
        "setup_s": setup_s,
        "scenario": {
            "name": "layout",
            "image_resize": "forced_bicubic",
            "layout_size_wh": list(args.layout_size),
            "offset": args.offset,
            "limit": len(pages),
            "buckets": list(args.buckets),
        },
        "corpus": [
            {key: value for key, value in page.items() if key not in {"pixel_values", "grid"}}
            for page in pages
        ],
        "corpus_summary": {
            "pages": len(pages),
            "unique_real_token_counts": unique_real_tokens,
            "real_tokens_per_pass": sum(int(page["real_tokens"]) for page in pages),
            "selected_buckets": selected_buckets,
        },
        "preparation": preparation,
        "parity": parity,
        "executions": results,
    }
    if "eager" in results and "torchair" in results:
        result["comparison"] = {
            "torchair_wall_speedup": (
                results["eager"]["call_wall_s"] / results["torchair"]["call_wall_s"]
            ),
            "effective_tok_s_speedup": (
                results["torchair"]["effective_tok_s"]
                / results["eager"]["effective_tok_s"]
            ),
        }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    headline = {
        execution: {
            key: results[execution][key]
            for key in (
                "pages_per_s",
                "effective_tok_s",
                "physical_tok_s",
                "useful_token_fraction",
                "call_wall_s",
            )
        }
        for execution in args.execution
    }
    print(json.dumps({"headline": headline, "comparison": result.get("comparison")}, indent=2))
    print(f"[output] {output}", flush=True)


if __name__ == "__main__":
    main()
