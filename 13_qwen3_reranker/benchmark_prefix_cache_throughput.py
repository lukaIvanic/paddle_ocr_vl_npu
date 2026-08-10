#!/usr/bin/env python3
"""Benchmark full and prefix-cached Qwen3 reranker prefill static shapes."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Callable

import torch
from torch import nn

from local_modeling_qwen3_reranker import (
    build_left_padded_causal_bool_mask,
    build_left_padded_causal_bool_mask_chunk,
    build_left_padded_causal_mask,
)
from run_local_qwen3_reranker import LocalQwen3RerankerRunner, _import_cache_compile
from transformers_rerank import DEFAULT_TASK, PREFIX, SUFFIX


PREFIX_BLOCK = 128
LANES = (
    "full_manual",
    "full_promptfa_compiled",
    "prefix_promptfa_compiled",
)


def parse_int_list(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or min(parsed) <= 0:
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-sizes", type=parse_int_list, default=(1, 2, 4, 8))
    parser.add_argument(
        "--continuation-lengths",
        type=parse_int_list,
        default=(128, 256, 512),
        help="Physical continuation token rows; every value must be a multiple of 128.",
    )
    parser.add_argument("--batch-sweep-continuation", type=int, default=128)
    parser.add_argument("--length-sweep-batch", type=int, default=1)
    parser.add_argument("--matrix", choices=("axes", "cross"), default="axes")
    parser.add_argument("--lanes", nargs="+", choices=LANES, default=list(LANES))
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--compile-cache-dir",
        type=Path,
        default=Path(".runtime_cache/13_qwen3_reranker/prefix_throughput"),
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--task", default=DEFAULT_TASK)
    return parser.parse_args()


def source_hash() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for name in (
        "benchmark_prefix_cache_throughput.py",
        "local_modeling_qwen3_reranker.py",
    ):
        digest.update((root / name).read_bytes())
    return digest.hexdigest()[:12]


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
        ).strip()
    except Exception:
        return None


def synchronize(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize()


def timed_call(device: torch.device, fn: Callable[[], torch.Tensor]) -> tuple[float, torch.Tensor]:
    synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = fn()
    synchronize(device)
    return time.perf_counter() - started, output


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_timings(
    values: list[float],
    *,
    batch_size: int,
    served_tokens: int,
    executed_tokens: int,
    attention_query_tokens: int,
) -> dict[str, float | int]:
    median_s = statistics.median(values)
    mean_s = statistics.mean(values)
    return {
        "runs": len(values),
        "mean_s": mean_s,
        "median_s": median_s,
        "min_s": min(values),
        "p90_s": percentile(values, 0.90),
        "max_s": max(values),
        "pairs_s": batch_size / median_s,
        "served_input_tok_s": served_tokens / median_s,
        "executed_model_tok_s": executed_tokens / median_s,
        "physical_attention_q_tok_s": attention_query_tokens / median_s,
    }


class FullLastHiddenStage(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.model.forward_hidden_states_prepared(
            input_ids,
            position_ids,
            attention_mask,
        )
        return hidden_states[:, -1]


class PrefixLastHiddenStage(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.layer_count = len(model.layers)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *flat_prefix_caches: torch.Tensor,
    ) -> torch.Tensor:
        key_caches = tuple(flat_prefix_caches[: self.layer_count])
        value_caches = tuple(flat_prefix_caches[self.layer_count :])
        hidden_states = self.model.forward_cached_suffix_prepared(
            input_ids,
            position_ids,
            attention_mask,
            key_caches,
            value_caches,
        )
        return hidden_states[:, -1]


def set_attention_impl(model: nn.Module, implementation: str) -> None:
    model.attention_impl = implementation
    for layer in model.layers:
        layer.self_attn.attention_impl = implementation


def make_prefix_inputs(tokenizer, *, task: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, int]:
    fixed_body_prefix = f"<Instruct>: {task}\n<Query>:"
    prefix_ids = tokenizer.encode(PREFIX, add_special_tokens=False)
    prefix_ids += tokenizer.encode(fixed_body_prefix, add_special_tokens=False)
    if len(prefix_ids) > PREFIX_BLOCK:
        raise ValueError(f"prefix uses {len(prefix_ids)} tokens, exceeding block {PREFIX_BLOCK}")
    padding = PREFIX_BLOCK - len(prefix_ids)
    input_ids = torch.tensor(
        [[int(tokenizer.pad_token_id)] * padding + prefix_ids],
        device=device,
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [[0] * padding + [1] * len(prefix_ids)],
        device=device,
        dtype=torch.long,
    )
    return input_ids, attention_mask, len(prefix_ids)


def make_continuation_inputs(
    tokenizer,
    *,
    batch_size: int,
    continuation_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    suffix_ids = tokenizer.encode(SUFFIX, add_special_tokens=False)
    filler_ids = tokenizer.encode(" benchmark", add_special_tokens=False)
    if not filler_ids:
        raise RuntimeError("tokenizer produced no benchmark filler tokens")
    filler_length = continuation_length - len(suffix_ids)
    if filler_length <= 0:
        raise ValueError("continuation length is too small for the reranker suffix")
    repeated = (filler_ids * ((filler_length + len(filler_ids) - 1) // len(filler_ids)))[
        :filler_length
    ]
    row = repeated + suffix_ids
    input_ids = torch.tensor([row] * batch_size, device=device, dtype=torch.long)
    attention_mask = torch.ones(
        (batch_size, continuation_length),
        device=device,
        dtype=torch.long,
    )
    return input_ids, attention_mask


def build_prefix_cache(
    model: nn.Module,
    prefix_input_ids: torch.Tensor,
    prefix_attention_mask: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[float, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    position_ids = prefix_attention_mask.cumsum(dim=-1) - 1
    position_ids = position_ids.clamp(min=0)
    prefix_mask = build_left_padded_causal_bool_mask(prefix_attention_mask)
    elapsed, caches = timed_call(
        device,
        lambda: model.build_prefix_cache_eager(
            prefix_input_ids,
            position_ids,
            prefix_mask,
        ),
    )
    key_caches, value_caches = caches
    return elapsed, key_caches, value_caches


def benchmark_lane(
    *,
    lane: str,
    fn: Callable[[], torch.Tensor],
    device: torch.device,
    warmups: int,
    repeats: int,
    batch_size: int,
    served_tokens: int,
    executed_tokens: int,
    attention_query_tokens: int,
    compile_wrapper_s: float | None,
    cache_dir: Path | None,
) -> tuple[dict, torch.Tensor]:
    first_call_s, output = timed_call(device, fn)
    warmup_seconds = []
    for _ in range(max(0, warmups - 1)):
        elapsed, output = timed_call(device, fn)
        warmup_seconds.append(elapsed)
    measured = []
    for _ in range(repeats):
        elapsed, output = timed_call(device, fn)
        measured.append(elapsed)
    summary = summarize_timings(
        measured,
        batch_size=batch_size,
        served_tokens=served_tokens,
        executed_tokens=executed_tokens,
        attention_query_tokens=attention_query_tokens,
    )
    summary.update(
        {
            "lane": lane,
            "first_call_s": first_call_s,
            "additional_warmup_s": warmup_seconds,
            "compile_wrapper_s": compile_wrapper_s,
            "cache_dir": None if cache_dir is None else str(cache_dir),
        }
    )
    return summary, output


def compile_stage(
    stage: nn.Module,
    *,
    cache_dir: Path,
    device: torch.device,
) -> tuple[Callable, float]:
    from torch_npu.dynamo.torchair.configs.compiler_config import CompilerConfig

    cache_dir.mkdir(parents=True, exist_ok=True)
    synchronize(device)
    started = time.perf_counter()
    compiled = _import_cache_compile()(
        stage.forward,
        config=CompilerConfig(),
        dynamic=False,
        cache_dir=str(cache_dir),
        ge_cache=True,
        fullgraph=True,
    )
    synchronize(device)
    return compiled, time.perf_counter() - started


def shape_matrix(args: argparse.Namespace) -> tuple[tuple[int, int], ...]:
    if args.matrix == "cross":
        shapes = {
            (batch_size, continuation_length)
            for batch_size in args.batch_sizes
            for continuation_length in args.continuation_lengths
        }
    else:
        shapes = {
            (batch_size, args.batch_sweep_continuation)
            for batch_size in args.batch_sizes
        }
        shapes.update(
            (args.length_sweep_batch, continuation_length)
            for continuation_length in args.continuation_lengths
        )
    return tuple(sorted(shapes, key=lambda item: (item[1], item[0])))


def print_result(result: dict) -> None:
    print(
        "THROUGHPUT "
        f"lane={result['lane']} "
        f"B={result['batch_size']} C={result['continuation_length']} "
        f"S={result['full_physical_length']} "
        f"median_ms={result['median_s'] * 1000.0:.3f} "
        f"pairs_s={result['pairs_s']:.2f} "
        f"served_tok_s={result['served_input_tok_s']:.2f} "
        f"executed_tok_s={result['executed_model_tok_s']:.2f} "
        f"attention_q_tok_s={result['physical_attention_q_tok_s']:.2f} "
        f"first_call_s={result['first_call_s']:.3f}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.warmups <= 0 or args.repeats <= 0:
        raise ValueError("warmups and repeats must be positive")
    all_lengths = set(args.continuation_lengths) | {args.batch_sweep_continuation}
    if any(length % 128 != 0 for length in all_lengths):
        raise ValueError("all continuation lengths must be multiples of 128")

    import torch_npu

    device = torch.device(args.device)
    torch.npu.set_device(device)
    torch.npu.set_compile_mode(jit_compile=False)
    args.compile_cache_dir = args.compile_cache_dir.expanduser().resolve()
    args.compile_cache_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"ENV host={platform.node()} device={torch.npu.get_device_name(device)!r} "
        f"torch={torch.__version__} torch_npu={torch_npu.__version__} commit={git_commit()}",
        flush=True,
    )

    maximum_batch = max(max(args.batch_sizes), args.length_sweep_batch)
    print("loading model", flush=True)
    runner = LocalQwen3RerankerRunner(
        args.model_dir,
        device=device,
        dtype=torch.float16,
        max_length=PREFIX_BLOCK + min(all_lengths),
        batch_size=maximum_batch,
        compile_forward=False,
        attention_impl="prompt_flash_attention",
        ffn_weight_mode="dense",
    )
    model = runner.model
    tokenizer = runner.tokenizer
    prefix_input_ids, prefix_attention_mask, prefix_valid_tokens = make_prefix_inputs(
        tokenizer,
        task=args.task,
        device=device,
    )
    prefix_build_s, prefix_key_caches, prefix_value_caches = build_prefix_cache(
        model,
        prefix_input_ids,
        prefix_attention_mask,
        device=device,
    )
    print(
        f"PREFIX_CACHE_BUILD valid_tokens={prefix_valid_tokens} physical_tokens={PREFIX_BLOCK} "
        f"wall_s={prefix_build_s:.6f}",
        flush=True,
    )

    full_stage = FullLastHiddenStage(model).eval()
    prefix_stage = PrefixLastHiddenStage(model).eval()
    source_key = source_hash()
    results = []
    for batch_size, continuation_length in shape_matrix(args):
        full_length = PREFIX_BLOCK + continuation_length
        continuation_ids, continuation_attention = make_continuation_inputs(
            tokenizer,
            batch_size=batch_size,
            continuation_length=continuation_length,
            device=device,
        )
        prefix_ids = prefix_input_ids.expand(batch_size, -1)
        prefix_attention = prefix_attention_mask.expand(batch_size, -1)
        full_input_ids = torch.cat((prefix_ids, continuation_ids), dim=1).contiguous()
        full_attention = torch.cat((prefix_attention, continuation_attention), dim=1)
        full_position_ids = (full_attention.cumsum(dim=-1) - 1).clamp(min=0)
        full_bool_mask = build_left_padded_causal_bool_mask(full_attention)
        full_additive_mask = build_left_padded_causal_mask(full_attention, torch.float16)
        continuation_position_ids = full_position_ids[:, PREFIX_BLOCK:].contiguous()
        continuation_mask = build_left_padded_causal_bool_mask_chunk(
            full_attention,
            query_start=PREFIX_BLOCK,
            query_end=full_length,
        )
        batched_key_caches = tuple(
            cache.expand(batch_size, -1, -1, -1).contiguous()
            for cache in prefix_key_caches
        )
        batched_value_caches = tuple(
            cache.expand(batch_size, -1, -1, -1).contiguous()
            for cache in prefix_value_caches
        )
        served_tokens = batch_size * (prefix_valid_tokens + continuation_length)
        attention_query_tokens = batch_size * full_length
        lane_outputs: dict[str, torch.Tensor] = {}

        if "full_manual" in args.lanes:
            set_attention_impl(model, "eager")
            summary, output = benchmark_lane(
                lane="full_manual",
                fn=lambda: full_stage(full_input_ids, full_position_ids, full_additive_mask),
                device=device,
                warmups=args.warmups,
                repeats=args.repeats,
                batch_size=batch_size,
                served_tokens=served_tokens,
                executed_tokens=batch_size * full_length,
                attention_query_tokens=attention_query_tokens,
                compile_wrapper_s=None,
                cache_dir=None,
            )
            lane_outputs["full_manual"] = output
            summary.update(
                batch_size=batch_size,
                continuation_length=continuation_length,
                full_physical_length=full_length,
            )
            results.append(summary)
            print_result(summary)

        if "full_promptfa_compiled" in args.lanes:
            set_attention_impl(model, "prompt_flash_attention")
            cache_dir = args.compile_cache_dir / (
                f"full_promptfa_b{batch_size}_s{full_length}_fp16_src{source_key}"
            )
            compiled, wrapper_s = compile_stage(full_stage, cache_dir=cache_dir, device=device)
            summary, output = benchmark_lane(
                lane="full_promptfa_compiled",
                fn=lambda: compiled(full_input_ids, full_position_ids, full_bool_mask),
                device=device,
                warmups=args.warmups,
                repeats=args.repeats,
                batch_size=batch_size,
                served_tokens=served_tokens,
                executed_tokens=batch_size * full_length,
                attention_query_tokens=attention_query_tokens,
                compile_wrapper_s=wrapper_s,
                cache_dir=cache_dir,
            )
            lane_outputs["full_promptfa_compiled"] = output
            summary.update(
                batch_size=batch_size,
                continuation_length=continuation_length,
                full_physical_length=full_length,
            )
            results.append(summary)
            print_result(summary)
            del compiled

        if "prefix_promptfa_compiled" in args.lanes:
            set_attention_impl(model, "prompt_flash_attention")
            cache_dir = args.compile_cache_dir / (
                f"prefix_promptfa_b{batch_size}_q{full_length}_kv{full_length}_"
                f"realq{continuation_length}_fp16_src{source_key}"
            )
            compiled, wrapper_s = compile_stage(prefix_stage, cache_dir=cache_dir, device=device)
            flat_caches = batched_key_caches + batched_value_caches
            summary, output = benchmark_lane(
                lane="prefix_promptfa_compiled",
                fn=lambda: compiled(
                    continuation_ids,
                    continuation_position_ids,
                    continuation_mask,
                    *flat_caches,
                ),
                device=device,
                warmups=args.warmups,
                repeats=args.repeats,
                batch_size=batch_size,
                served_tokens=served_tokens,
                executed_tokens=batch_size * continuation_length,
                attention_query_tokens=attention_query_tokens,
                compile_wrapper_s=wrapper_s,
                cache_dir=cache_dir,
            )
            lane_outputs["prefix_promptfa_compiled"] = output
            summary.update(
                batch_size=batch_size,
                continuation_length=continuation_length,
                full_physical_length=full_length,
            )
            results.append(summary)
            print_result(summary)
            del compiled

        if "full_manual" in lane_outputs:
            reference = lane_outputs["full_manual"].float()
            for lane, output in lane_outputs.items():
                if lane == "full_manual":
                    continue
                max_abs = float((output.float() - reference).abs().max().detach().cpu())
                for result in reversed(results):
                    if (
                        result["lane"] == lane
                        and result["batch_size"] == batch_size
                        and result["continuation_length"] == continuation_length
                    ):
                        result["max_abs_hidden_vs_full_manual"] = max_abs
                        break
                print(
                    f"CORRECTNESS lane={lane} B={batch_size} C={continuation_length} "
                    f"max_abs_hidden_vs_full_manual={max_abs:.6f}",
                    flush=True,
                )

        del lane_outputs
        gc.collect()
        torch.npu.empty_cache()

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
            "batch_sizes": list(args.batch_sizes),
            "continuation_lengths": list(args.continuation_lengths),
            "batch_sweep_continuation": args.batch_sweep_continuation,
            "length_sweep_batch": args.length_sweep_batch,
            "matrix": args.matrix,
            "lanes": list(args.lanes),
            "warmups": args.warmups,
            "repeats": args.repeats,
            "prefix_block": PREFIX_BLOCK,
            "prefix_valid_tokens": prefix_valid_tokens,
            "prefix_build_s": prefix_build_s,
            "source_hash": source_key,
        },
        "metric_definitions": {
            "served_input_tok_s": "valid prefix plus continuation tokens served per median call second",
            "executed_model_tok_s": "token rows executing decoder QKV and MLP per median call second",
            "physical_attention_q_tok_s": "physical attention Q rows, including square-padding dummies, per median call second",
        },
        "results": results,
    }
    encoded = json.dumps(result, indent=2)
    print("RESULT_JSON_BEGIN")
    print(encoded)
    print("RESULT_JSON_END")
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(encoded + "\n")
        print(f"OUTPUT_JSON {args.json_out}")


if __name__ == "__main__":
    main()
