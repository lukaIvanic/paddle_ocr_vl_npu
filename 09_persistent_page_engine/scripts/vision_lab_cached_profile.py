#!/usr/bin/env python3
"""Profile every compatible cached PromptFA vision graph without compiling."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
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
    VisionPrefillRuntime,
    vision_cache_dir_for_bucket,
)
from paddleocr_vl.serving.runtime_defaults import (  # noqa: E402
    OPTIMIZED_VISION_BUCKETS,
)
from utils.timing import DeviceTimeline, synchronize  # noqa: E402
from vision_lab import DEFAULT_MODEL, _environment  # noqa: E402
from vision_lab_batched_packed import (  # noqa: E402
    _compiled_shape,
    _compiled_shape_cache_dir,
)


DEFAULT_B1_CACHE_ROOT = (
    REPO_ROOT / ".runtime_cache/09_persistent_page_engine_vision_torchair"
)
DEFAULT_BATCHED_CACHE_ROOT = Path("/dev/shm/vision_lab_cache_all")
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "tmp/09_persistent_page_engine/vision_lab"
    / "cached_promptfa_physical_profile.json"
)

# Include every batched-packed shape explored so far. Cache-key preflight keeps
# only graphs compatible with the current model, software stack, and source.
CANDIDATE_BATCHED_SHAPES = (
    (1, 3072),
    (2, 1536),
    (4, 768),
    (2, 3072),
    (4, 1024),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--b1-cache-dir", type=Path, default=DEFAULT_B1_CACHE_ROOT)
    parser.add_argument(
        "--batched-cache-dir", type=Path, default=DEFAULT_BATCHED_CACHE_ROOT
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.repeats <= 0:
        parser.error("--warmup must be non-negative and --repeats positive")
    return args


def _cache_populated(path: Path) -> bool:
    return path.is_dir() and any(item.is_file() for item in path.rglob("*"))


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _measure_shape(
    *,
    run: Callable[..., torch.Tensor],
    batch_size: int,
    sequence_length: int,
    hidden_size: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    prefix = torch.zeros(
        (batch_size, sequence_length, hidden_size),
        device=device,
        dtype=dtype,
    )
    rope_cos = torch.ones(
        (batch_size, sequence_length, head_dim),
        device=device,
        dtype=torch.float32,
    )
    rope_sin = torch.zeros_like(rope_cos)
    attention_mask = torch.zeros(
        (batch_size, 1, sequence_length, sequence_length),
        device=device,
        dtype=torch.bool,
    )
    for _ in range(warmup):
        output = run(prefix, rope_cos, rope_sin, attention_mask)
        if tuple(output.shape[:2]) != (batch_size, sequence_length):
            raise RuntimeError(
                "cached graph returned the wrong shape: "
                f"expected={(batch_size, sequence_length)} got={tuple(output.shape)}"
            )
        del output
    synchronize(device)

    samples_ms: list[float] = []
    for _ in range(repeats):
        timeline = DeviceTimeline(device)
        output = timeline.measure(
            "graph",
            lambda: run(prefix, rope_cos, rope_sin, attention_mask),
        )
        spans = timeline.resolve_spans()
        if tuple(output.shape[:2]) != (batch_size, sequence_length):
            raise RuntimeError(
                "cached graph returned the wrong shape: "
                f"expected={(batch_size, sequence_length)} got={tuple(output.shape)}"
            )
        samples_ms.append(float(spans["graph"]["seconds"]) * 1000.0)
        del output

    physical_tokens = batch_size * sequence_length
    mean_ms = statistics.mean(samples_ms)
    median_ms = statistics.median(samples_ms)
    result = {
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "physical_tokens": physical_tokens,
        "samples_ms": samples_ms,
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "p95_ms": _percentile(samples_ms, 0.95),
        "raw_physical_tokens_per_s_mean": physical_tokens / (mean_ms / 1000.0),
        "raw_physical_tokens_per_s_median": physical_tokens
        / (median_ms / 1000.0),
    }
    del prefix, rope_cos, rope_sin, attention_mask
    return result


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    import torch_npu  # noqa: F401

    device = torch.device("npu:0")
    if not torch.npu.is_available():
        raise RuntimeError("cached vision profiling requires an NPU")
    torch.npu.set_compile_mode(jit_compile=False)
    dtype = torch.float16
    model_dir = args.model.expanduser().resolve()
    synchronize(device)
    setup_started = time.perf_counter()
    model = LocalPaddleOCRVLForConditionalGeneration.from_pretrained(
        model_dir,
        dtype=dtype,
        device=device,
    )
    hidden_size = int(model.config.vision_config.hidden_size)
    head_dim = hidden_size // int(model.config.vision_config.num_attention_heads)

    cached_b1: list[int] = []
    skipped_b1: list[dict[str, Any]] = []
    for bucket in OPTIMIZED_VISION_BUCKETS:
        cache_dir = vision_cache_dir_for_bucket(
            args.b1_cache_dir,
            bucket=int(bucket),
            dtype=dtype,
            device=device,
            model_dir=model_dir,
            attention_impl="prompt_flash_attention",
            head_dim=head_dim,
        )
        if _cache_populated(cache_dir):
            cached_b1.append(int(bucket))
        else:
            skipped_b1.append({"bucket": int(bucket), "cache_dir": str(cache_dir)})
    if not cached_b1:
        raise RuntimeError("no compatible cached B1 PromptFA graphs were found")

    compatible_batched: list[tuple[int, int, Path]] = []
    skipped_batched: list[dict[str, Any]] = []
    for batch_size, sequence_length in CANDIDATE_BATCHED_SHAPES:
        cache_dir = _compiled_shape_cache_dir(
            model=model,
            batch_size=batch_size,
            sequence_length=sequence_length,
            cache_root=args.batched_cache_dir,
            model_dir=model_dir,
            dtype=dtype,
            device=device,
        )
        if _cache_populated(cache_dir):
            compatible_batched.append((batch_size, sequence_length, cache_dir))
        else:
            skipped_batched.append(
                {
                    "batch_size": batch_size,
                    "sequence_length": sequence_length,
                    "reason": "no cache matching the current version/source key",
                    "expected_cache_dir": str(cache_dir),
                }
            )

    print(
        f"cache_only_preflight b1={len(cached_b1)} "
        f"batched={len(compatible_batched)} "
        f"skipped_b1={len(skipped_b1)} "
        f"skipped_batched={len(skipped_batched)}",
        flush=True,
    )
    runtime = VisionPrefillRuntime(
        model,
        backend="torchair",
        buckets=tuple(cached_b1),
        cache_root=args.b1_cache_dir,
        device=device,
        dtype=dtype,
        model_dir=model_dir,
        attention_impl="prompt_flash_attention",
        padding="bucket",
    )
    setup_s = time.perf_counter() - setup_started

    rows: list[dict[str, Any]] = []
    for bucket in cached_b1:
        print(f"benchmark=b1_s{bucket}", flush=True)
        row = _measure_shape(
            run=runtime.compiled[bucket],
            batch_size=1,
            sequence_length=bucket,
            hidden_size=hidden_size,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        row.update(
            {
                "kind": "production_b1",
                "cache_dir": runtime.metadata["per_bucket"][str(bucket)][
                    "torchair_cache_dir"
                ],
            }
        )
        rows.append(row)

    for batch_size, sequence_length, cache_dir in compatible_batched:
        print(f"benchmark=b{batch_size}_s{sequence_length}", flush=True)
        run, metadata = _compiled_shape(
            model=model,
            batch_size=batch_size,
            sequence_length=sequence_length,
            cache_root=args.batched_cache_dir,
            model_dir=model_dir,
            dtype=dtype,
            device=device,
        )
        if not metadata["cache_existed_before_run"]:
            raise RuntimeError(
                "cache-only invariant violated for "
                f"B{batch_size}xS{sequence_length}: {cache_dir}"
            )
        row = _measure_shape(
            run=run,
            batch_size=batch_size,
            sequence_length=sequence_length,
            hidden_size=hidden_size,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        row.update(
            {
                "kind": "batched_packed",
                "cache_dir": str(cache_dir),
                "cache_load_first_call_s": metadata["compile_first_call_s"],
            }
        )
        rows.append(row)

    payload = {
        "schema_version": 1,
        "created_at_unix_s": time.time(),
        "purpose": "cache-only physical-throughput profile for vision routing",
        "cache_only": True,
        "attention": "prompt_flash_attention",
        "timing_basis": "NPU device events around graph execution only",
        "routing_cost_basis": "median_ms",
        "warmup": args.warmup,
        "repeats": args.repeats,
        "setup_s": setup_s,
        "environment": _environment(device),
        "runtime_metadata": runtime.metadata,
        "graphs": sorted(
            rows, key=lambda row: (row["batch_size"], row["sequence_length"])
        ),
        "skipped_b1": skipped_b1,
        "skipped_batched": skipped_batched,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"output={output}")
    print("shape | kind | median_ms | p95_ms | raw_physical_tok_s")
    for row in payload["graphs"]:
        print(
            " | ".join(
                str(value)
                for value in (
                    f"B{row['batch_size']}xS{row['sequence_length']}",
                    row["kind"],
                    row["median_ms"],
                    row["p95_ms"],
                    row["raw_physical_tokens_per_s_median"],
                )
            )
        )


if __name__ == "__main__":
    main()
