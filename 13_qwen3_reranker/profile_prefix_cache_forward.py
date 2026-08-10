#!/usr/bin/env python3
"""Profile one warm static prefix-cached Qwen3 reranker forward graph."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import statistics
import time
from pathlib import Path

import torch

from benchmark_local_qwen3_reranker import KernelAttributor, summarize_profile
from benchmark_prefix_cache_throughput import (
    PREFIX_BLOCK,
    PrefixLastHiddenStage,
    build_prefix_cache,
    compile_stage,
    git_commit,
    make_continuation_inputs,
    make_prefix_inputs,
    set_attention_impl,
    source_hash,
    synchronize,
    timed_call,
)
from local_modeling_qwen3_reranker import build_left_padded_causal_bool_mask_chunk
from run_local_qwen3_reranker import LocalQwen3RerankerRunner
from transformers_rerank import DEFAULT_TASK


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--continuation-length", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--profile-iters", type=int, default=3)
    parser.add_argument("--topn", type=int, default=30)
    parser.add_argument(
        "--compile-cache-dir",
        type=Path,
        default=Path(".runtime_cache/13_qwen3_reranker/prefix_throughput"),
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=Path("tmp/13_qwen3_reranker/profile_prefix_b4_c128_910b2"),
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--task", default=DEFAULT_TASK)
    return parser.parse_args()


def profiler_experimental_config(npu_prof):
    return npu_prof._ExperimentalConfig(
        profiler_level=npu_prof.ProfilerLevel.Level1,
        aic_metrics=npu_prof.AiCMetrics.PipeUtilization,
        export_type=npu_prof.ExportType.Text,
    )


def run_profile(
    fn,
    *,
    device: torch.device,
    profile_dir: Path,
    iterations: int,
    npu_prof,
) -> tuple[list[float], float]:
    if profile_dir.exists():
        shutil.rmtree(profile_dir)
    profile_dir.mkdir(parents=True)
    synchronized_seconds: list[float] = []
    synchronize(device)
    context_started = time.perf_counter()
    with npu_prof.profile(
        activities=[npu_prof.ProfilerActivity.CPU, npu_prof.ProfilerActivity.NPU],
        on_trace_ready=npu_prof.tensorboard_trace_handler(
            str(profile_dir),
            analyse_flag=True,
        ),
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
        experimental_config=profiler_experimental_config(npu_prof),
    ) as profiler:
        for _ in range(iterations):
            synchronize(device)
            started = time.perf_counter()
            with torch.inference_mode(), torch.profiler.record_function(
                "qwen3_reranker.prefix_cached_compiled_forward"
            ):
                fn()
            synchronize(device)
            synchronized_seconds.append(time.perf_counter() - started)
            profiler.step()
    return synchronized_seconds, time.perf_counter() - context_started


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.continuation_length <= 0 or args.continuation_length % 128 != 0:
        raise ValueError("--continuation-length must be a positive multiple of 128")
    if min(args.warmups, args.repeats, args.profile_iters, args.topn) <= 0:
        raise ValueError("warmups, repeats, profile-iters, and topn must be positive")

    import torch_npu
    import torch_npu.profiler as npu_prof

    device = torch.device(args.device)
    torch.npu.set_device(device)
    torch.npu.set_compile_mode(jit_compile=False)
    args.compile_cache_dir = args.compile_cache_dir.expanduser().resolve()
    args.profile_dir = args.profile_dir.expanduser().resolve()
    json_out = (
        args.json_out.expanduser().resolve()
        if args.json_out is not None
        else args.profile_dir / "prefix_profile_summary.json"
    )
    print(
        f"ENV host={platform.node()} device={torch.npu.get_device_name(device)!r} "
        f"torch={torch.__version__} torch_npu={torch_npu.__version__} commit={git_commit()}",
        flush=True,
    )

    full_length = PREFIX_BLOCK + args.continuation_length
    runner = LocalQwen3RerankerRunner(
        args.model_dir,
        device=device,
        dtype=torch.float16,
        max_length=full_length,
        batch_size=args.batch_size,
        compile_forward=False,
        attention_impl="prompt_flash_attention",
        ffn_weight_mode="dense",
    )
    model = runner.model
    tokenizer = runner.tokenizer
    prefix_ids, prefix_attention, prefix_valid_tokens = make_prefix_inputs(
        tokenizer,
        task=args.task,
        device=device,
    )
    prefix_build_s, prefix_key_caches, prefix_value_caches = build_prefix_cache(
        model,
        prefix_ids,
        prefix_attention,
        device=device,
    )
    continuation_ids, continuation_attention = make_continuation_inputs(
        tokenizer,
        batch_size=args.batch_size,
        continuation_length=args.continuation_length,
        device=device,
    )
    full_attention = torch.cat(
        (
            prefix_attention.expand(args.batch_size, -1),
            continuation_attention,
        ),
        dim=1,
    )
    full_position_ids = (full_attention.cumsum(dim=-1) - 1).clamp(min=0)
    continuation_position_ids = full_position_ids[:, PREFIX_BLOCK:].contiguous()
    continuation_mask = build_left_padded_causal_bool_mask_chunk(
        full_attention,
        query_start=PREFIX_BLOCK,
        query_end=full_length,
    )
    flat_caches = tuple(
        cache.expand(args.batch_size, -1, -1, -1).contiguous()
        for cache in prefix_key_caches
    ) + tuple(
        cache.expand(args.batch_size, -1, -1, -1).contiguous()
        for cache in prefix_value_caches
    )

    set_attention_impl(model, "prompt_flash_attention")
    stage = PrefixLastHiddenStage(model).eval()
    cache_dir = args.compile_cache_dir / (
        f"prefix_promptfa_b{args.batch_size}_q{full_length}_kv{full_length}_"
        f"realq{args.continuation_length}_fp16_src{source_hash()}"
    )
    compiled, compile_wrapper_s, cache_was_warm = compile_stage(
        stage,
        entrypoint_name=(
            f"reranker_prefix_promptfa_b{args.batch_size}_"
            f"realq{args.continuation_length}_s{full_length}"
        ),
        cache_dir=cache_dir,
        device=device,
    )
    fn = lambda: compiled(
        continuation_ids,
        continuation_position_ids,
        continuation_mask,
        *flat_caches,
    )

    first_call_s, _ = timed_call(device, fn)
    warmup_seconds = [timed_call(device, fn)[0] for _ in range(args.warmups)]
    baseline_seconds = [timed_call(device, fn)[0] for _ in range(args.repeats)]
    baseline_median_s = statistics.median(baseline_seconds)
    served_tokens = args.batch_size * (prefix_valid_tokens + args.continuation_length)
    executed_tokens = args.batch_size * args.continuation_length
    print(
        "UNPROFILED "
        f"B={args.batch_size} C={args.continuation_length} S={full_length} "
        f"median_ms={baseline_median_s * 1000.0:.3f} "
        f"served_tok_s={served_tokens / baseline_median_s:.2f} "
        f"executed_tok_s={executed_tokens / baseline_median_s:.2f} "
        f"cache_was_warm={cache_was_warm}",
        flush=True,
    )

    profiled_seconds, profile_context_wall_s = run_profile(
        fn,
        device=device,
        profile_dir=args.profile_dir,
        iterations=args.profile_iters,
        npu_prof=npu_prof,
    )
    profiled_median_s = statistics.median(profiled_seconds)
    profile_summary = summarize_profile(
        args.profile_dir,
        attributor=KernelAttributor(runner.config),
        topn=args.topn,
    )
    print(
        "PROFILED "
        f"median_sync_ms={profiled_median_s * 1000.0:.3f} "
        f"overhead_vs_unprofiled={profiled_median_s / baseline_median_s:.3f}x "
        f"context_wall_s={profile_context_wall_s:.3f}",
        flush=True,
    )

    result = {
        "environment": {
            "host": platform.node(),
            "device": torch.npu.get_device_name(device),
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "git_commit": git_commit(),
        },
        "configuration": {
            "model_dir": str(args.model_dir),
            "batch_size": args.batch_size,
            "continuation_length": args.continuation_length,
            "full_physical_length": full_length,
            "prefix_block": PREFIX_BLOCK,
            "prefix_valid_tokens": prefix_valid_tokens,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "profile_iters": args.profile_iters,
            "compile_cache_dir": str(cache_dir),
            "cache_was_warm": cache_was_warm,
            "source_hash": source_hash(),
        },
        "setup": {
            "prefix_build_s": prefix_build_s,
            "compile_wrapper_s": compile_wrapper_s,
            "first_call_s": first_call_s,
            "warmup_seconds": warmup_seconds,
        },
        "unprofiled": {
            "seconds": baseline_seconds,
            "median_s": baseline_median_s,
            "served_input_tok_s": served_tokens / baseline_median_s,
            "executed_model_tok_s": executed_tokens / baseline_median_s,
        },
        "profiled": {
            "synchronized_seconds": profiled_seconds,
            "median_s": profiled_median_s,
            "overhead_vs_unprofiled": profiled_median_s / baseline_median_s,
            "profile_context_wall_s": profile_context_wall_s,
        },
        "profile": profile_summary,
    }
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"PROFILE_ROOT {profile_summary['profile_root']}")
    print(f"OUTPUT_JSON {json_out}")


if __name__ == "__main__":
    main()
