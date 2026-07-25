#!/usr/bin/env python3
"""Profile reusable multi-query paged FIA in the full Paddle decoder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import torch

HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXPERIMENT_ROOT))

import benchmark_paged_fia_full_decoder as base
import probe_torchair_fia_multi_query as multi
import profile_paged_fia_full_decoder as profile_base

from paddleocr_vl.model.text_decode import (
    cast_decode_linear_weights_to_nz,
    prepare_decode_optimization_modules,
)
from utils.timing import synchronize


DEFAULT_PROFILE_ROOT = (
    REPO_ROOT
    / ".runtime_cache/09_persistent_page_engine_profiles"
    / "paged_fia_multi_query"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "tmp/09_persistent_page_engine/text_decode_lab"
    / "torchair_fia_multi_query_profile.json"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=base.DEFAULT_MODEL)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=base.DEFAULT_CACHE_ROOT,
    )
    parser.add_argument(
        "--profile-root",
        type=Path,
        default=DEFAULT_PROFILE_ROOT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--query-bucket", type=int, default=8)
    parser.add_argument("--cache-length", type=int, default=4096)
    parser.add_argument("--attention-bucket-length", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--query-pattern", default="1,2,4,8")
    parser.add_argument("--initial-positions", default="63,127,254,508")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--profile-iters", type=int, default=5)
    parser.add_argument(
        "--profile-metric",
        choices=("pipe", "memory", "l2", "memory_access"),
        default="pipe",
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    try:
        args.query_pattern = multi._parse_patterns(
            args.query_pattern,
            batch_size=args.batch_size,
            query_bucket=args.query_bucket,
        )[0]
        args.initial_positions = multi._parse_int_tuple(
            args.initial_positions,
            expected=args.batch_size,
            label="--initial-positions",
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.batch_size <= 0 or args.query_bucket <= 0:
        parser.error("--batch-size and --query-bucket must be positive")
    if args.cache_length <= 0:
        parser.error("--cache-length must be positive")
    if args.block_size <= 0 or args.cache_length % args.block_size:
        parser.error("--block-size must evenly divide --cache-length")
    if (
        args.attention_bucket_length <= 0
        or args.attention_bucket_length > args.cache_length
    ):
        parser.error(
            "--attention-bucket-length must be positive and no larger than "
            "--cache-length"
        )
    if max(
        args.initial_positions[row] + args.query_pattern[row]
        for row in range(args.batch_size)
    ) > args.attention_bucket_length:
        parser.error("runtime queries exceed --attention-bucket-length")
    if args.warmup < 0 or args.profile_iters <= 0:
        parser.error("--warmup must be non-negative and profile-iters positive")
    return args


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not torch.npu.is_available():
        raise RuntimeError("profiling requires an available Ascend NPU")
    device = torch.device("npu:0")
    dtype = torch.float16
    torch.npu.set_compile_mode(jit_compile=False)
    config = base.PaddleOCRVLConfig.from_model_dir(args.model)
    model = base._create_random_model(
        config,
        device=device,
        dtype=dtype,
        seed=args.seed,
    )
    optimization = prepare_decode_optimization_modules(
        model,
        base.OPTIMIZATION,
    )
    weight_format = cast_decode_linear_weights_to_nz(model)
    synchronize(device)
    _dense_rows, paged_cache, dummy_slot_base = (
        multi._allocate_matching_multi_query_caches(
            config,
            batch_size=args.batch_size,
            query_bucket=args.query_bucket,
            cache_length=args.cache_length,
            block_size=args.block_size,
            device=device,
            dtype=dtype,
            seed=args.seed + 90_000,
        )
    )
    stage = multi.MultiQueryPagedFIAStage(
        model,
        batch_size=args.batch_size,
        query_bucket=args.query_bucket,
        cache_length=args.cache_length,
        attention_bucket_length=args.attention_bucket_length,
        block_size=args.block_size,
        dummy_slot_base=dummy_slot_base,
        optimization=optimization,
    ).eval()
    fn, compile_metadata = multi._compile_stage(
        stage,
        cache_root=args.cache_dir,
    )
    input_ids = multi._input_ids(
        batch_size=args.batch_size,
        query_bucket=args.query_bucket,
        vocab_size=config.text_config.vocab_size,
        step=100,
        device=device,
    )
    start_positions = torch.tensor(
        args.initial_positions,
        device=device,
        dtype=torch.int64,
    )
    query_lengths = torch.tensor(
        args.query_pattern,
        device=device,
        dtype=torch.int64,
    )
    rope_deltas = torch.zeros(
        (args.batch_size, 1),
        device=device,
        dtype=torch.int64,
    )
    call_args = (
        input_ids,
        start_positions,
        query_lengths,
        rope_deltas,
        paged_cache.block_table,
        *paged_cache.flat_tensors(),
    )
    profile_dir = (
        args.profile_root.expanduser().resolve()
        / (
            f"b{args.batch_size}_q{args.query_bucket}_"
            f"k{args.cache_length}_a{args.attention_bucket_length}_"
            f"pattern{'-'.join(str(value) for value in args.query_pattern)}"
        )
    )
    profile_result = profile_base._profile_lane(
        name="paged_fia_multi_query",
        fn=fn,
        call_args=call_args,
        device=device,
        profile_dir=profile_dir,
        metric=args.profile_metric,
        warmup=args.warmup,
        profile_iters=args.profile_iters,
    )
    result = {
        "schema_version": 1,
        "kind": "full_decoder_paged_fia_multi_query_profile",
        "configuration": {
            "batch_size": args.batch_size,
            "query_bucket": args.query_bucket,
            "query_pattern": list(args.query_pattern),
            "physical_query_tokens_per_call": (
                args.batch_size * args.query_bucket
            ),
            "effective_query_tokens_per_call": sum(args.query_pattern),
            "cache_length": args.cache_length,
            "attention_bucket_length": args.attention_bucket_length,
            "block_size": args.block_size,
            "profile_metric": args.profile_metric,
            "linear_weight_format": weight_format,
        },
        "compile": compile_metadata,
        "profile": profile_result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"OUTPUT_JSON={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
