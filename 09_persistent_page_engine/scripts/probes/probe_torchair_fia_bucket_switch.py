#!/usr/bin/env python3
"""Verify TorchAir FIA graph switching without moving the physical KV cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import torch
import torch_npu


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = REPO_ROOT / "09_persistent_page_engine"
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmark_paged_fia_full_decoder as bench  # noqa: E402

from paddleocr_vl.model.text_decode import (  # noqa: E402
    TextDecodeRuntime,
    cast_decode_linear_weights_to_nz,
    prepare_decode_optimization_modules,
)
from utils.timing import synchronize  # noqa: E402


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "tmp/09_persistent_page_engine/text_decode_lab"
    / "torchair_fia_bucket_switch.json"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=bench.DEFAULT_MODEL)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=bench.DEFAULT_CACHE_ROOT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-length", type=int, default=4096)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--buckets", default="512,1024")
    parser.add_argument("--initial-positions", default="508,509,510,511")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    try:
        args.buckets = tuple(
            int(value.strip())
            for value in args.buckets.split(",")
            if value.strip()
        )
        args.initial_positions = tuple(
            int(value.strip())
            for value in args.initial_positions.split(",")
            if value.strip()
        )
    except ValueError as exc:
        parser.error(f"invalid integer list: {exc}")
    if args.batch_size <= 0 or args.steps <= 0:
        parser.error("--batch-size and --steps must be positive")
    if len(args.initial_positions) != args.batch_size:
        parser.error(
            "--initial-positions must contain exactly --batch-size values"
        )
    if not args.buckets or tuple(sorted(args.buckets)) != args.buckets:
        parser.error("--buckets must be a non-empty increasing list")
    if args.buckets[-1] > args.cache_length:
        parser.error("--buckets must fit in --cache-length")
    if max(args.initial_positions) + args.steps > args.buckets[-1]:
        parser.error("decode steps would exceed the largest bucket")
    return args


def _select_bucket(
    positions: torch.Tensor,
    buckets: Sequence[int],
) -> int:
    maximum_position = int(positions.max().cpu())
    for bucket in buckets:
        if maximum_position < bucket:
            return bucket
    raise RuntimeError("no compiled attention bucket can hold this step")


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not torch.npu.is_available():
        raise RuntimeError("probe requires an available Ascend NPU")
    device = torch.device("npu:0")
    dtype = torch.float16
    torch.npu.set_compile_mode(jit_compile=False)
    config = bench.PaddleOCRVLConfig.from_model_dir(args.model)
    model = bench._create_random_model(
        config,
        device=device,
        dtype=dtype,
        seed=args.seed,
    )
    optimization = prepare_decode_optimization_modules(
        model,
        bench.OPTIMIZATION,
    )
    weight_format = cast_decode_linear_weights_to_nz(model)
    synchronize(device)
    incre_runtime = TextDecodeRuntime(
        model,
        backend="torchair",
        device=device,
        cache_root=args.cache_dir,
        batch_size=args.batch_size,
        cache_length=args.cache_length,
        dtype=dtype,
        model_dir=args.model,
        linear_weight_format=str(weight_format["effective_mode"]),
        optimization=optimization,
    )

    paged_functions = {}
    compile_metadata = {}
    for bucket in args.buckets:
        stage = bench.PagedFIATextDecodeStage(
            model,
            block_size=args.block_size,
            cache_update_mode="scatter_nd",
            optimization=optimization,
            native_fia=True,
            fixed_actual_kv_lengths=[bucket] * args.batch_size,
        ).eval()
        fn, metadata = bench._compile_paged_stage(
            stage,
            cache_root=args.cache_dir,
            batch_size=args.batch_size,
            cache_length=args.cache_length,
            attention_bucket_length=bucket,
            block_size=args.block_size,
            cache_update_mode="scatter_nd",
            metadata_mode="fixed_bucket_mask",
            single_stream=False,
        )
        _dense_warm, paged_warm = bench._allocate_matching_caches(
            config.text_config,
            batch_size=args.batch_size,
            cache_length=args.cache_length,
            block_size=args.block_size,
            device=device,
            dtype=dtype,
            seed=args.seed + bucket,
        )
        warm_positions = torch.zeros(
            (args.batch_size,),
            device=device,
            dtype=torch.int64,
        )
        fn(
            torch.zeros(
                (args.batch_size, 1),
                device=device,
                dtype=torch.int64,
            ),
            warm_positions,
            torch.zeros(
                (args.batch_size, 1),
                device=device,
                dtype=torch.int64,
            ),
            paged_warm.block_table,
            *paged_warm.flat_tensors(),
        )
        synchronize(device)
        paged_functions[bucket] = fn
        compile_metadata[str(bucket)] = metadata

    dense_cache, paged_cache = bench._allocate_matching_caches(
        config.text_config,
        batch_size=args.batch_size,
        cache_length=args.cache_length,
        block_size=args.block_size,
        device=device,
        dtype=dtype,
        seed=args.seed + 50_000,
    )
    dense_input = (
        torch.arange(args.batch_size, device=device, dtype=torch.int64)
        .add_(17)
        .view(args.batch_size, 1)
    )
    paged_input = dense_input.clone()
    dense_positions = torch.tensor(
        args.initial_positions,
        device=device,
        dtype=torch.int64,
    )
    paged_positions = dense_positions.clone()
    rope_deltas = torch.zeros(
        (args.batch_size, 1),
        device=device,
        dtype=torch.int64,
    )
    rows = []
    passed = True
    previous_bucket = None
    for step in range(args.steps):
        bucket = _select_bucket(paged_positions, args.buckets)
        dense_logits = incre_runtime.fn(
            dense_input,
            dense_positions,
            rope_deltas,
            *dense_cache.flat_tensors(),
        )
        paged_logits = paged_functions[bucket](
            paged_input,
            paged_positions,
            rope_deltas,
            paged_cache.block_table,
            *paged_cache.flat_tensors(),
        )
        synchronize(device)
        dense_tokens = torch.argmax(
            dense_logits[:, -1, :].float(),
            dim=-1,
            keepdim=True,
        )
        paged_tokens = torch.argmax(
            paged_logits[:, -1, :].float(),
            dim=-1,
            keepdim=True,
        )
        logits_delta = bench._delta_stats(paged_logits, dense_logits)
        cache_delta = bench._cache_delta_stats(
            bench._dense_cache_written_values(
                dense_cache,
                dense_positions,
            ),
            bench._page_cache_written_values(
                paged_cache,
                paged_positions,
            ),
        )
        argmax_matches = int((dense_tokens == paged_tokens).sum().cpu())
        step_passed = (
            argmax_matches == args.batch_size
            and cache_delta["mean_abs"] < 1e-3
        )
        passed = passed and step_passed
        rows.append(
            {
                "step": step,
                "positions": [
                    int(position)
                    for position in dense_positions.cpu().tolist()
                ],
                "bucket": bucket,
                "bucket_changed": (
                    previous_bucket is not None
                    and bucket != previous_bucket
                ),
                "argmax_matches": argmax_matches,
                "argmax_total": args.batch_size,
                "logits": logits_delta,
                "written_kv": {
                    "max_abs": cache_delta["max_abs"],
                    "mean_abs": cache_delta["mean_abs"],
                },
                "passed": step_passed,
            }
        )
        previous_bucket = bucket
        dense_input.copy_(dense_tokens)
        paged_input.copy_(paged_tokens)
        dense_positions.add_(1)
        paged_positions.add_(1)

    result = {
        "schema_version": 1,
        "kind": "torchair_fixed_fia_bucket_switch",
        "passed": passed,
        "configuration": {
            "batch_size": args.batch_size,
            "cache_length": args.cache_length,
            "block_size": args.block_size,
            "buckets": list(args.buckets),
            "initial_positions": list(args.initial_positions),
            "steps": args.steps,
            "cache_transition": (
                "same physical PA_NZ cache tensors passed to every graph"
            ),
        },
        "compile": compile_metadata,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
