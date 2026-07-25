#!/usr/bin/env python3
"""Benchmark direct Scatter-PA + FIA in one full Paddle NPUGraph decode."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import torch
import torch_npu
from torch import nn

HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXPERIMENT_ROOT))

from benchmark_paged_fia_full_decoder import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_MODEL,
    OPTIMIZATION,
    PagedFIATextDecodeStage,
    _allocate_matching_caches,
    _cache_delta_stats,
    _create_random_model,
    _delta_stats,
    _dense_cache_written_values,
    _page_cache_written_values,
    _parameter_counts,
    _timed_decode,
)
from paddleocr_vl.model.config import PaddleOCRVLConfig
from paddleocr_vl.model.text_decode import (
    TextDecodeRuntime,
    cast_decode_linear_weights_to_nz,
    prepare_decode_optimization_modules,
)
from utils.timing import synchronize


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "tmp/09_persistent_page_engine/text_decode_lab"
    / "npugraph_paged_fia_full_decoder_b1_k1024_p768.json"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cache-length", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--position", type=int, default=768)
    parser.add_argument(
        "--min-position",
        type=int,
        default=None,
        help=(
            "Linearly spread batch-row positions from this value through "
            "--position. Omit for a uniform-length batch."
        ),
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.cache_length <= 0:
        parser.error("--cache-length must be positive")
    if args.block_size <= 0 or args.cache_length % args.block_size:
        parser.error("--block-size must evenly divide --cache-length")
    if not 0 <= args.position < args.cache_length:
        parser.error("--position must be inside the selected cache capacity")
    if args.min_position is not None and not (
        0 <= args.min_position <= args.position
    ):
        parser.error(
            "--min-position must be between zero and --position"
        )
    if args.warmup < 0 or args.repeats <= 0:
        parser.error("--warmup must be non-negative and --repeats positive")
    return args


class DecodeAndArgmax(nn.Module):
    def __init__(self, stage: PagedFIATextDecodeStage):
        super().__init__()
        self.stage = stage

    def forward(self, *args: torch.Tensor):
        logits = self.stage(*args)
        tokens = torch.argmax(
            logits[:, -1, :].float(),
            dim=-1,
            keepdim=True,
        )
        return logits, tokens


def _cast_paged_cache_to_nd(cache) -> None:
    torch.npu.config.allow_internal_format = True
    for tensor in cache.flat_tensors():
        torch_npu.npu_format_cast(tensor, 2)


def _capture(
    stage: DecodeAndArgmax,
    args: tuple[torch.Tensor, ...],
    *,
    device: torch.device,
) -> tuple[torch.npu.NPUGraph, tuple[torch.Tensor, torch.Tensor], float]:
    stage(*args)
    synchronize(device)
    graph = torch.npu.NPUGraph()
    started = time.perf_counter()
    with torch.npu.graph(graph):
        output = stage(*args)
    synchronize(device)
    return graph, output, time.perf_counter() - started


def _timed_replay(
    graph: torch.npu.NPUGraph,
    *,
    batch_size: int,
    warmup: int,
    repeats: int,
) -> dict[str, float]:
    for _ in range(warmup):
        graph.replay()
    torch.npu.synchronize()
    start = torch_npu.npu.Event(enable_timing=True)
    end = torch_npu.npu.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    total_s = float(start.elapsed_time(end)) / 1000.0
    return {
        "total_s": total_s,
        "mean_ms": total_s * 1000.0 / repeats,
        "decode_steps_per_s": repeats / total_s,
        "raw_tokens_per_s": repeats * batch_size / total_s,
    }


def _batch_positions(args: argparse.Namespace) -> torch.Tensor:
    if args.min_position is None or args.batch_size == 1:
        return torch.full(
            (args.batch_size,),
            args.position,
            dtype=torch.int64,
        )
    return torch.linspace(
        args.min_position,
        args.position,
        steps=args.batch_size,
        dtype=torch.float64,
    ).round().to(torch.int64)


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not torch.npu.is_available():
        raise RuntimeError("benchmark requires an available Ascend NPU")

    device = torch.device("npu:0")
    dtype = torch.float16
    torch.npu.set_compile_mode(jit_compile=False)
    config = PaddleOCRVLConfig.from_model_dir(args.model)

    setup_started = time.perf_counter()
    model_started = time.perf_counter()
    model = _create_random_model(
        config,
        device=device,
        dtype=dtype,
        seed=args.seed,
    )
    model_create_s = time.perf_counter() - model_started
    parameters_before_packing = _parameter_counts(model)
    optimization = prepare_decode_optimization_modules(
        model,
        OPTIMIZATION,
    )
    parameters_after_packing = _parameter_counts(model)
    weight_format_started = time.perf_counter()
    weight_format = cast_decode_linear_weights_to_nz(model)
    synchronize(device)
    weight_format_s = time.perf_counter() - weight_format_started

    incre_started = time.perf_counter()
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
    incre_setup_s = time.perf_counter() - incre_started

    dense_cache, paged_cache = _allocate_matching_caches(
        config.text_config,
        batch_size=args.batch_size,
        cache_length=args.cache_length,
        block_size=args.block_size,
        device=device,
        dtype=dtype,
        seed=args.seed + 1000 + args.position,
    )
    _cast_paged_cache_to_nd(paged_cache)
    input_ids = (
        torch.arange(
            args.batch_size,
            device=device,
            dtype=torch.int64,
        )
        .add(17)
        .remainder(config.text_config.vocab_size)
        .view(args.batch_size, 1)
    )
    cache_position = _batch_positions(args).to(device)
    rope_deltas = torch.zeros(
        (args.batch_size, 1),
        device=device,
        dtype=torch.int64,
    )
    paged_args = (
        input_ids,
        cache_position,
        rope_deltas,
        paged_cache.block_table,
        *paged_cache.flat_tensors(),
    )
    page_values_before = _page_cache_written_values(
        paged_cache,
        cache_position,
    )

    incre_logits = incre_runtime.fn(
        input_ids,
        cache_position,
        rope_deltas,
        *dense_cache.flat_tensors(),
    )
    incre_tokens = torch.argmax(
        incre_logits[:, -1, :].float(),
        dim=-1,
        keepdim=True,
    )

    paged_stage = PagedFIATextDecodeStage(
        model,
        block_size=args.block_size,
        cache_update_mode="scatter_pa_inplace",
        optimization=optimization,
        native_fia=True,
        fixed_actual_kv_lengths=tuple(
            int(position) + 1 for position in cache_position.cpu().tolist()
        ),
    ).eval()
    captured_stage = DecodeAndArgmax(paged_stage).eval()
    graph, graph_output, capture_s = _capture(
        captured_stage,
        paged_args,
        device=device,
    )
    graph.replay()
    synchronize(device)
    paged_logits, paged_tokens = graph_output

    cache_delta = _cache_delta_stats(
        _dense_cache_written_values(dense_cache, cache_position),
        _page_cache_written_values(paged_cache, cache_position),
    )
    page_pool_change = _cache_delta_stats(
        _page_cache_written_values(paged_cache, cache_position),
        page_values_before,
    )
    logits_delta = _delta_stats(paged_logits, incre_logits)
    first_layer_delta = {
        "key": cache_delta["per_tensor"][0],
        "value": cache_delta["per_tensor"][
            config.text_config.num_hidden_layers
        ],
    }

    incre_timing = _timed_decode(
        incre_runtime.fn,
        (
            input_ids,
            cache_position,
            rope_deltas,
            *dense_cache.flat_tensors(),
        ),
        warmup=args.warmup,
        repeats=args.repeats,
        device=device,
    )
    paged_timing = _timed_replay(
        graph,
        batch_size=args.batch_size,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    setup_s = time.perf_counter() - setup_started

    passed = (
        first_layer_delta["key"]["max_abs"] == 0.0
        and first_layer_delta["value"]["max_abs"] == 0.0
        and page_pool_change["max_abs"] > 0.0
        and bool((paged_tokens == incre_tokens).all().cpu())
    )
    result = {
        "schema_version": 1,
        "kind": "random_full_decoder_npugraph_paged_fia_benchmark",
        "passed": passed,
        "configuration": {
            "model_config": str(args.model / "config.json"),
            "random_weights": True,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "cache_length": args.cache_length,
            "block_size": args.block_size,
            "cache_positions": [
                int(position) for position in cache_position.cpu().tolist()
            ],
            "actual_kv_lengths": [
                int(position) + 1
                for position in cache_position.cpu().tolist()
            ],
            "position_pattern": (
                "uniform"
                if args.min_position is None
                else "linearly_staggered"
            ),
            "paged_cache_layout": "PA_NZ",
            "paged_cache_update": "npu_scatter_pa_kv_cache",
            "paged_attention": "npu_fused_infer_attention_score_v2",
            "graph": "torch.npu.NPUGraph",
            "warmup": args.warmup,
            "repeats": args.repeats,
            "dtype": str(dtype),
            "optimization": OPTIMIZATION,
            "full_step": (
                "embedding_18_layers_final_norm_lm_head_argmax"
            ),
        },
        "architecture": {
            "hidden_size": config.text_config.hidden_size,
            "intermediate_size": config.text_config.intermediate_size,
            "num_hidden_layers": config.text_config.num_hidden_layers,
            "num_attention_heads": config.text_config.num_attention_heads,
            "num_key_value_heads": config.text_config.num_key_value_heads,
            "head_dim": config.text_config.head_dim,
            "vocab_size": config.text_config.vocab_size,
            "parameters_before_packed_qkv": parameters_before_packing,
            "parameters_after_packed_qkv": parameters_after_packing,
        },
        "versions": {
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
        },
        "setup_s": {
            "total": setup_s,
            "random_model_create_and_transfer": model_create_s,
            "weight_format": weight_format_s,
            "increfa_runtime": incre_setup_s,
            "npugraph_capture": capture_s,
        },
        "weight_format": weight_format,
        "correctness": {
            "logits": logits_delta,
            "argmax_matches": int(
                (paged_tokens == incre_tokens).sum().cpu()
            ),
            "argmax_total": args.batch_size,
            "written_kv": cache_delta,
            "first_layer_written_kv": first_layer_delta,
            "input_page_pool_change": page_pool_change,
        },
        "timing": {
            "increfa": incre_timing,
            "npugraph_paged_fia": paged_timing,
            "paged_vs_increfa_speedup": (
                incre_timing["mean_ms"] / paged_timing["mean_ms"]
            ),
        },
        "anchor": {
            "saved_b1_k1024_full_production_step_tok_per_s": 742.6,
            "saved_b1_k1024_model_and_argmax_tok_per_s": 749.6,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
