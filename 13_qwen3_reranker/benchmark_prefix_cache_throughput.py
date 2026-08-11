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
import types
from pathlib import Path
from typing import Callable

import torch
from torch import nn

from local_modeling_qwen3_reranker import (
    PREFILL_OPTIMIZATION_PRESETS,
    RERANKER_LINEAR_WEIGHT_FORMAT_CHOICES,
    build_310p_square_promptfa_mask,
    build_left_padded_causal_bool_mask,
    build_left_padded_causal_bool_mask_chunk,
    build_left_padded_causal_mask,
    fuse_reranker_qkv_projections_inplace,
    prepare_reranker_linear_weight_format,
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
    parser.add_argument(
        "--full-lengths-are-total",
        action="store_true",
        help=(
            "Interpret the length sweep as total valid sequence lengths for "
            "full-prefill-only runs. The fixed instruction tokens are included "
            "inside each requested length."
        ),
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--compile-cache-dir",
        type=Path,
        default=Path(".runtime_cache/13_qwen3_reranker/prefix_throughput"),
    )
    parser.add_argument(
        "--compile-source-key",
        help="Reuse a known-compatible graph source key after benchmark-only changes.",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument(
        "--linear-weight-format",
        choices=RERANKER_LINEAR_WEIGHT_FORMAT_CHOICES,
        default="native",
    )
    parser.add_argument(
        "--ffn-weight-mode",
        choices=("dense", "gate_up_w8a8", "down_w8a8", "w8a8", "qkv_w8a8", "packed_qkv_w8a8", "o_w8a8", "full_w8a8"),
        default="dense",
        help=(
            "Use dense projections, isolated or combined FFN W8A8, separate Q/K/V W8A8 "
            "matmuls sharing one activation quantization, or one packed QKV "
            "W8A8 matmul, O-only W8A8, or full packed-QKV/O/FFN W8A8."
        ),
    )
    parser.add_argument(
        "--enable-internal-format",
        action="store_true",
        help="Enable torch-npu internal tensor formats before the first NPU allocation.",
    )
    parser.add_argument(
        "--fuse-qkv-projections",
        action="store_true",
        help="Replace each layer's three FP16 Q/K/V linears with one concatenated linear.",
    )
    parser.add_argument(
        "--prefill-optimizations",
        nargs="+",
        choices=tuple(PREFILL_OPTIMIZATION_PRESETS),
        default=["baseline"],
        help="Compiled full- and prefix-prefill implementation presets to compare.",
    )
    return parser.parse_args()


def source_hash() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for name in (
        "benchmark_prefix_cache_throughput.py",
        "local_modeling_qwen3_reranker.py",
        "local_reranker_w8a8.py",
        "run_local_qwen3_reranker.py",
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


def attach_output_signature(
    summary: dict[str, object],
    output: torch.Tensor,
    *,
    model: nn.Module,
    false_token_id: int,
    true_token_id: int,
) -> None:
    yes_no_ids = torch.tensor(
        [false_token_id, true_token_id],
        device=output.device,
        dtype=torch.long,
    )
    with torch.no_grad():
        weight = model.lm_head.weight.index_select(0, yes_no_ids)
        logits = torch.nn.functional.linear(output, weight).float()
        scores = torch.softmax(logits, dim=-1)[:, 1]
    summary["yes_no_logits"] = logits.detach().cpu().tolist()
    summary["yes_scores"] = scores.detach().cpu().tolist()
    summary["yes_no_choices"] = logits.argmax(dim=-1).detach().cpu().tolist()


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
    cache_was_warm: bool | None,
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
            "cache_was_warm": cache_was_warm,
        }
    )
    return summary, output


def compile_stage(
    stage: nn.Module,
    *,
    entrypoint_name: str,
    cache_dir: Path,
    device: torch.device,
) -> tuple[Callable, float, bool]:
    from torch_npu.dynamo.torchair.configs.compiler_config import CompilerConfig

    cache_was_warm = cache_dir.is_dir() and any(cache_dir.iterdir())
    cache_dir.mkdir(parents=True, exist_ok=True)
    original = stage.forward.__func__
    function = types.FunctionType(
        original.__code__.replace(co_name=entrypoint_name),
        original.__globals__,
        entrypoint_name,
        original.__defaults__,
        original.__closure__,
    )
    function.__annotations__ = dict(original.__annotations__)
    function.__kwdefaults__ = original.__kwdefaults__
    entrypoint = types.MethodType(function, stage)
    synchronize(device)
    started = time.perf_counter()
    compiled = _import_cache_compile()(
        entrypoint,
        config=CompilerConfig(),
        dynamic=False,
        cache_dir=str(cache_dir),
        ge_cache=True,
        fullgraph=True,
    )
    synchronize(device)
    return compiled, time.perf_counter() - started, cache_was_warm


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
    optimization = result.get("prefill_optimization")
    optimization_text = "" if optimization is None else f"optimization={optimization} "
    requested_length = result.get("requested_sequence_length")
    requested_text = "" if requested_length is None else f"T={requested_length} "
    print(
        "THROUGHPUT "
        f"lane={result['lane']} "
        f"{optimization_text}"
        f"{requested_text}"
        f"B={result['batch_size']} C={result['continuation_length']} "
        f"S={result['full_physical_length']} "
        f"median_ms={result['median_s'] * 1000.0:.3f} "
        f"pairs_s={result['pairs_s']:.2f} "
        f"served_tok_s={result['served_input_tok_s']:.2f} "
        f"executed_tok_s={result['executed_model_tok_s']:.2f} "
        f"attention_q_tok_s={result['physical_attention_q_tok_s']:.2f} "
        f"first_call_s={result['first_call_s']:.3f} "
        f"cache_was_warm={result['cache_was_warm']}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.warmups <= 0 or args.repeats <= 0:
        raise ValueError("warmups and repeats must be positive")
    all_lengths = set(args.continuation_lengths) | {args.batch_sweep_continuation}
    if any(length % 128 != 0 for length in all_lengths):
        raise ValueError("all continuation lengths must be multiples of 128")
    if args.full_lengths_are_total and "prefix_promptfa_compiled" in args.lanes:
        raise ValueError(
            "--full-lengths-are-total is only valid when the prefix-cached lane "
            "is excluded"
        )

    import torch_npu

    internal_format_enabled = bool(
        args.enable_internal_format or args.linear_weight_format == "fractal_nz"
    )
    torch.npu.config.allow_internal_format = internal_format_enabled
    device = torch.device(args.device)
    torch.npu.set_device(device)
    torch.npu.set_compile_mode(jit_compile=False)
    args.compile_cache_dir = args.compile_cache_dir.expanduser().resolve()
    args.compile_cache_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"ENV host={platform.node()} device={torch.npu.get_device_name(device)!r} "
        f"torch={torch.__version__} torch_npu={torch_npu.__version__} commit={git_commit()} "
        f"internal_format={internal_format_enabled}",
        flush=True,
    )

    maximum_batch = max(max(args.batch_sizes), args.length_sweep_batch)
    print("loading model", flush=True)
    model_load_started = time.perf_counter()
    runner = LocalQwen3RerankerRunner(
        args.model_dir,
        device=device,
        dtype=torch.float16,
        max_length=(
            min(all_lengths)
            if args.full_lengths_are_total
            else PREFIX_BLOCK + min(all_lengths)
        ),
        batch_size=maximum_batch,
        compile_forward=False,
        attention_impl="prompt_flash_attention",
        ffn_weight_mode=args.ffn_weight_mode,
    )
    model_load_s = time.perf_counter() - model_load_started
    print(
        "MODEL_LOAD "
        f"wall_s={model_load_s:.6f} "
        f"weight_quantization_s={runner.weight_quantization_s:.6f}",
        flush=True,
    )
    model = runner.model
    tokenizer = runner.tokenizer
    qkv_fusion_s = 0.0
    fused_qkv_layer_count = 0
    if args.fuse_qkv_projections:
        qkv_fusion_started = time.perf_counter()
        fused_qkv_layer_count = fuse_reranker_qkv_projections_inplace(model)
        synchronize(device)
        qkv_fusion_s = time.perf_counter() - qkv_fusion_started
        print(
            "QKV_FUSION "
            f"layers={fused_qkv_layer_count} wall_s={qkv_fusion_s:.6f}",
            flush=True,
        )
    weight_format_started = time.perf_counter()
    weight_format = prepare_reranker_linear_weight_format(
        model,
        requested=args.linear_weight_format,
    )
    synchronize(device)
    weight_format["setup_s"] = time.perf_counter() - weight_format_started
    print("LINEAR_WEIGHT_FORMAT " + json.dumps(weight_format, sort_keys=True), flush=True)
    quant_weight_format = None
    if args.ffn_weight_mode != "dense":
        from local_reranker_w8a8 import prepare_w8a8_weight_format

        quant_weight_format = prepare_w8a8_weight_format(
            model,
            requested=args.linear_weight_format,
        )
        synchronize(device)
        print("W8A8_WEIGHT_FORMAT " + json.dumps(quant_weight_format, sort_keys=True), flush=True)
    weight_cache_key = (
        f"ffn{args.ffn_weight_mode}_weights{weight_format['effective_mode']}_"
        f"internal{int(internal_format_enabled)}_"
        f"qkvfused{int(args.fuse_qkv_projections)}"
    )
    prefix_input_ids, prefix_attention_mask, prefix_valid_tokens = make_prefix_inputs(
        tokenizer,
        task=args.task,
        device=device,
    )
    if args.full_lengths_are_total and min(all_lengths) <= prefix_valid_tokens:
        raise ValueError(
            "every total full-prefill length must exceed the valid instruction "
            f"length {prefix_valid_tokens}"
        )
    if args.ffn_weight_mode != "dense":
        calibration_length = (
            max(all_lengths) - prefix_valid_tokens
            if args.full_lengths_are_total
            else max(all_lengths)
        )
        calibration_ids, calibration_attention = make_continuation_inputs(
            tokenizer,
            batch_size=maximum_batch,
            continuation_length=calibration_length,
            device=device,
        )
        if args.full_lengths_are_total:
            calibration_prefix_ids = prefix_input_ids[
                :, PREFIX_BLOCK - prefix_valid_tokens:
            ].expand(maximum_batch, -1)
            calibration_prefix_attention = torch.ones(
                (maximum_batch, prefix_valid_tokens),
                device=device,
                dtype=torch.long,
            )
        else:
            calibration_prefix_ids = prefix_input_ids.expand(maximum_batch, -1)
            calibration_prefix_attention = prefix_attention_mask.expand(
                maximum_batch,
                -1,
            )
        calibration_full_ids = torch.cat(
            (calibration_prefix_ids, calibration_ids),
            dim=1,
        ).contiguous()
        calibration_full_attention = torch.cat(
            (calibration_prefix_attention, calibration_attention),
            dim=1,
        )
        calibration_started = time.perf_counter()
        runner.calibrate_ffn_input_scales(calibration_full_ids, calibration_full_attention)
        synchronize(device)
        print(
            "W8A8_CALIBRATION "
            f"batch={maximum_batch} sequence={calibration_full_ids.shape[1]} "
            f"wall_s={time.perf_counter() - calibration_started:.6f}",
            flush=True,
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
    source_key = args.compile_source_key or source_hash()
    results = []
    for batch_size, requested_length in shape_matrix(args):
        continuation_length = (
            requested_length - prefix_valid_tokens
            if args.full_lengths_are_total
            else requested_length
        )
        cached_full_length = PREFIX_BLOCK + continuation_length
        compact_full_length = prefix_valid_tokens + continuation_length
        continuation_ids, continuation_attention = make_continuation_inputs(
            tokenizer,
            batch_size=batch_size,
            continuation_length=continuation_length,
            device=device,
        )
        cached_prefix_attention = prefix_attention_mask.expand(batch_size, -1)
        cached_full_attention = torch.cat(
            (cached_prefix_attention, continuation_attention),
            dim=1,
        )
        cached_full_position_ids = (
            cached_full_attention.cumsum(dim=-1) - 1
        ).clamp(min=0)
        continuation_position_ids = cached_full_position_ids[
            :, PREFIX_BLOCK:
        ].contiguous()
        continuation_mask = build_left_padded_causal_bool_mask_chunk(
            cached_full_attention,
            query_start=PREFIX_BLOCK,
            query_end=cached_full_length,
        )

        compact_prefix_ids = prefix_input_ids[
            :, PREFIX_BLOCK - prefix_valid_tokens:
        ].expand(batch_size, -1)
        compact_input_ids = torch.cat(
            (compact_prefix_ids, continuation_ids),
            dim=1,
        ).contiguous()
        compact_attention = torch.ones(
            (batch_size, compact_full_length),
            device=device,
            dtype=torch.long,
        )
        compact_position_ids = torch.arange(
            compact_full_length,
            device=device,
            dtype=torch.long,
        ).view(1, -1).expand(batch_size, -1).contiguous()
        compact_bool_mask = build_left_padded_causal_bool_mask(compact_attention)
        compact_additive_mask = build_left_padded_causal_mask(
            compact_attention,
            torch.float16,
        )
        served_tokens = batch_size * (prefix_valid_tokens + continuation_length)
        compact_attention_query_tokens = batch_size * compact_full_length
        cached_attention_query_tokens = batch_size * cached_full_length
        lane_outputs: dict[str, torch.Tensor] = {}

        if "full_manual" in args.lanes:
            set_attention_impl(model, "eager")
            summary, output = benchmark_lane(
                lane="full_manual",
                fn=lambda: full_stage(
                    compact_input_ids,
                    compact_position_ids,
                    compact_additive_mask,
                ),
                device=device,
                warmups=args.warmups,
                repeats=args.repeats,
                batch_size=batch_size,
                served_tokens=served_tokens,
                executed_tokens=batch_size * compact_full_length,
                attention_query_tokens=compact_attention_query_tokens,
                compile_wrapper_s=None,
                cache_dir=None,
                cache_was_warm=None,
            )
            lane_outputs["full_manual"] = output
            summary.update(
                batch_size=batch_size,
                continuation_length=continuation_length,
                full_physical_length=compact_full_length,
                requested_sequence_length=(
                    requested_length if args.full_lengths_are_total else None
                ),
                linear_weight_format=weight_format["effective_mode"],
            )
            attach_output_signature(
                summary,
                output,
                model=model,
                false_token_id=runner.false_token_id,
                true_token_id=runner.true_token_id,
            )
            results.append(summary)
            print_result(summary)

        if "full_promptfa_compiled" in args.lanes:
            set_attention_impl(model, "prompt_flash_attention")
            for optimization_name in args.prefill_optimizations:
                optimization = model.set_prefill_optimization(optimization_name)
                compiled_mask = (
                    build_310p_square_promptfa_mask(compact_bool_mask)
                    if optimization.prebuilt_square_mask
                    else compact_bool_mask
                )
                cache_dir = args.compile_cache_dir / (
                    f"full_promptfa_{optimization.name}_b{batch_size}_"
                    f"s{compact_full_length}_{weight_cache_key}_fp16_src{source_key}"
                )
                compiled, wrapper_s, cache_was_warm = compile_stage(
                    full_stage,
                    entrypoint_name=(
                        f"reranker_full_promptfa_{optimization.name}_"
                        f"b{batch_size}_s{compact_full_length}_{weight_cache_key}"
                    ),
                    cache_dir=cache_dir,
                    device=device,
                )
                summary, output = benchmark_lane(
                    lane="full_promptfa_compiled",
                    fn=lambda: compiled(
                        compact_input_ids,
                        compact_position_ids,
                        compiled_mask,
                    ),
                    device=device,
                    warmups=args.warmups,
                    repeats=args.repeats,
                    batch_size=batch_size,
                    served_tokens=served_tokens,
                    executed_tokens=batch_size * compact_full_length,
                    attention_query_tokens=compact_attention_query_tokens,
                    compile_wrapper_s=wrapper_s,
                    cache_dir=cache_dir,
                    cache_was_warm=cache_was_warm,
                )
                output_key = f"full_promptfa_compiled:{optimization.name}"
                lane_outputs[output_key] = output
                summary.update(
                    batch_size=batch_size,
                    continuation_length=continuation_length,
                    full_physical_length=compact_full_length,
                    requested_sequence_length=(
                        requested_length if args.full_lengths_are_total else None
                    ),
                    prefill_optimization=optimization.name,
                    linear_weight_format=weight_format["effective_mode"],
                )
                attach_output_signature(
                    summary,
                    output,
                    model=model,
                    false_token_id=runner.false_token_id,
                    true_token_id=runner.true_token_id,
                )
                results.append(summary)
                print_result(summary)
                del compiled
            model.set_prefill_optimization("baseline")

        if "prefix_promptfa_compiled" in args.lanes:
            set_attention_impl(model, "prompt_flash_attention")
            for optimization_name in args.prefill_optimizations:
                optimization = model.set_prefill_optimization(optimization_name)
                prepared_keys, prepared_values = model.prepare_prefix_caches(
                    prefix_key_caches,
                    prefix_value_caches,
                )
                batched_key_caches = tuple(
                    cache.expand(batch_size, -1, -1, -1).contiguous()
                    for cache in prepared_keys
                )
                batched_value_caches = tuple(
                    cache.expand(batch_size, -1, -1, -1).contiguous()
                    for cache in prepared_values
                )
                compiled_mask = (
                    build_310p_square_promptfa_mask(continuation_mask)
                    if optimization.prebuilt_square_mask
                    else continuation_mask
                )
                cache_dir = args.compile_cache_dir / (
                    f"prefix_promptfa_{optimization.name}_b{batch_size}_"
                    f"q{cached_full_length}_kv{cached_full_length}_"
                    f"realq{continuation_length}_"
                    f"{weight_cache_key}_fp16_src{source_key}"
                )
                compiled, wrapper_s, cache_was_warm = compile_stage(
                    prefix_stage,
                    entrypoint_name=(
                        f"reranker_prefix_promptfa_{optimization.name}_b{batch_size}_"
                        f"realq{continuation_length}_s{cached_full_length}_"
                        f"{weight_cache_key}"
                    ),
                    cache_dir=cache_dir,
                    device=device,
                )
                flat_caches = batched_key_caches + batched_value_caches
                summary, output = benchmark_lane(
                    lane="prefix_promptfa_compiled",
                    fn=lambda: compiled(
                        continuation_ids,
                        continuation_position_ids,
                        compiled_mask,
                        *flat_caches,
                    ),
                    device=device,
                    warmups=args.warmups,
                    repeats=args.repeats,
                    batch_size=batch_size,
                    served_tokens=served_tokens,
                    executed_tokens=batch_size * continuation_length,
                    attention_query_tokens=cached_attention_query_tokens,
                    compile_wrapper_s=wrapper_s,
                    cache_dir=cache_dir,
                    cache_was_warm=cache_was_warm,
                )
                output_key = f"prefix_promptfa_compiled:{optimization.name}"
                lane_outputs[output_key] = output
                summary.update(
                    batch_size=batch_size,
                    continuation_length=continuation_length,
                    full_physical_length=cached_full_length,
                    requested_sequence_length=(
                        requested_length if args.full_lengths_are_total else None
                    ),
                    prefill_optimization=optimization.name,
                    linear_weight_format=weight_format["effective_mode"],
                )
                attach_output_signature(
                    summary,
                    output,
                    model=model,
                    false_token_id=runner.false_token_id,
                    true_token_id=runner.true_token_id,
                )
                results.append(summary)
                print_result(summary)
                del compiled
            model.set_prefill_optimization("baseline")

        if "full_manual" in lane_outputs:
            reference = lane_outputs["full_manual"].float()
            for output_key, output in lane_outputs.items():
                if output_key == "full_manual":
                    continue
                max_abs = float((output.float() - reference).abs().max().detach().cpu())
                for result in reversed(results):
                    if (
                        output_key.startswith(result["lane"])
                        and result["batch_size"] == batch_size
                        and result["continuation_length"] == continuation_length
                        and (
                            result.get("prefill_optimization") is None
                            or output_key.endswith(f":{result['prefill_optimization']}")
                        )
                    ):
                        result["max_abs_hidden_vs_full_manual"] = max_abs
                        break
                print(
                    f"CORRECTNESS lane={output_key} B={batch_size} C={continuation_length} "
                    f"max_abs_hidden_vs_full_manual={max_abs:.6f}",
                    flush=True,
                )

        baseline_prefix = lane_outputs.get("prefix_promptfa_compiled:baseline")
        if baseline_prefix is not None:
            yes_no_ids = torch.tensor(
                [runner.false_token_id, runner.true_token_id],
                device=device,
                dtype=torch.long,
            )
            yes_no_weight = model.lm_head.weight.index_select(0, yes_no_ids)
            with torch.no_grad():
                baseline_logits = torch.nn.functional.linear(
                    baseline_prefix,
                    yes_no_weight,
                ).float()
                baseline_scores = torch.softmax(baseline_logits, dim=-1)[:, 1]
            for output_key, output in lane_outputs.items():
                if not output_key.startswith("prefix_promptfa_compiled:") or output_key.endswith(":baseline"):
                    continue
                max_abs = float((output.float() - baseline_prefix.float()).abs().max().detach().cpu())
                with torch.no_grad():
                    candidate_logits = torch.nn.functional.linear(
                        output,
                        yes_no_weight,
                    ).float()
                    candidate_scores = torch.softmax(candidate_logits, dim=-1)[:, 1]
                max_abs_logits = float(
                    (candidate_logits - baseline_logits).abs().max().detach().cpu()
                )
                max_abs_scores = float(
                    (candidate_scores - baseline_scores).abs().max().detach().cpu()
                )
                same_binary_choice = bool(
                    torch.equal(
                        candidate_logits.argmax(dim=-1),
                        baseline_logits.argmax(dim=-1),
                    )
                )
                optimization_name = output_key.rsplit(":", 1)[1]
                for result in reversed(results):
                    if (
                        result["lane"] == "prefix_promptfa_compiled"
                        and result.get("prefill_optimization") == optimization_name
                        and result["batch_size"] == batch_size
                        and result["continuation_length"] == continuation_length
                    ):
                        result["max_abs_hidden_vs_prefix_baseline"] = max_abs
                        result["max_abs_yes_no_logits_vs_prefix_baseline"] = max_abs_logits
                        result["max_abs_yes_score_vs_prefix_baseline"] = max_abs_scores
                        result["same_yes_no_choice_vs_prefix_baseline"] = same_binary_choice
                        break
                print(
                    f"CORRECTNESS lane={output_key} B={batch_size} C={continuation_length} "
                    f"max_abs_hidden_vs_prefix_baseline={max_abs:.6f} "
                    f"max_abs_yes_no_logits={max_abs_logits:.6f} "
                    f"max_abs_yes_score={max_abs_scores:.8f} "
                    f"same_yes_no_choice={same_binary_choice}",
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
            "full_lengths_are_total": args.full_lengths_are_total,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "prefix_block": PREFIX_BLOCK,
            "prefix_valid_tokens": prefix_valid_tokens,
            "prefix_build_s": prefix_build_s,
            "source_hash": source_key,
            "compile_source_key_override": args.compile_source_key,
            "prefill_optimizations": list(args.prefill_optimizations),
            "internal_format_enabled": internal_format_enabled,
            "linear_weight_format": weight_format,
            "ffn_weight_mode": args.ffn_weight_mode,
            "fuse_qkv_projections": args.fuse_qkv_projections,
            "fused_qkv_layer_count": fused_qkv_layer_count,
            "qkv_fusion_s": qkv_fusion_s,
            "quant_weight_format": quant_weight_format,
            "model_load_s": model_load_s,
            "weight_quantization_s": runner.weight_quantization_s,
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
