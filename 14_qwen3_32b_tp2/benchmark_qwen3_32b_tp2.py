#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu
from torch_npu.dynamo import torchair
from torch_npu.dynamo.torchair.configs.compiler_config import CompilerConfig

from checkpoint import load_tp_checkpoint
from modeling_qwen3_tp2 import (
    Qwen3TPConfig,
    Qwen3TPForCausalLM,
    Qwen3TPStaticCache,
)


OFFICIAL_QWEN3_32B_SHAPE = (5120, 25600, 64, 64, 8, 128, 151936)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline static-KV Qwen3-32B BF16 TP2 decode benchmark."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--cache-length", type=int, default=4096)
    parser.add_argument("--prefix-length", type=int, default=512)
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument(
        "--layers",
        type=int,
        help="Load only the first N layers for compile bring-up. Default: all 64.",
    )
    parser.add_argument(
        "--backend", choices=("raw_eager", "torchair"), default="torchair"
    )
    parser.add_argument("--json-out")
    return parser.parse_args()


def log(rank: int, message: str) -> None:
    print(f"[rank {rank}] {message}", flush=True)


def memory_snapshot() -> dict[str, float]:
    torch.npu.synchronize()
    free_bytes, total_bytes = torch.npu.mem_get_info()
    return {
        "allocated_gib": torch.npu.memory_allocated() / 1024**3,
        "reserved_gib": torch.npu.memory_reserved() / 1024**3,
        "free_gib": free_bytes / 1024**3,
        "total_gib": total_bytes / 1024**3,
    }


def validate_official_shape(config: Qwen3TPConfig) -> None:
    actual = (
        config.hidden_size,
        config.intermediate_size,
        config.num_hidden_layers,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        config.vocab_size,
    )
    if actual != OFFICIAL_QWEN3_32B_SHAPE:
        raise ValueError(
            "This experiment is intentionally fixed to official Qwen3-32B. "
            f"Expected {OFFICIAL_QWEN3_32B_SHAPE}, got {actual}."
        )


def build_model(
    config: Qwen3TPConfig,
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Qwen3TPForCausalLM:
    with torch.device("meta"):
        model = Qwen3TPForCausalLM(config, tp_rank=rank, tp_size=world_size)
    model = model.to(dtype=dtype)
    model.to_empty(device=device)
    return model


def reduce_max_seconds(elapsed: float, device: torch.device) -> float:
    value = torch.tensor([elapsed], dtype=torch.float32, device=device)
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return float(value.item())


def run_decode_steps(
    decode_one,
    next_token: torch.Tensor,
    key_caches: tuple[torch.Tensor, ...],
    value_caches: tuple[torch.Tensor, ...],
    *,
    first_position: int,
    steps: int,
    device: torch.device,
) -> torch.Tensor:
    for offset in range(steps):
        cache_position = torch.tensor(
            [first_position + offset], dtype=torch.int64, device=device
        )
        next_token = decode_one(
            next_token,
            cache_position,
            key_caches,
            value_caches,
        )
    return next_token


def main() -> None:
    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise ValueError(f"This experiment requires TP2, got WORLD_SIZE={world_size}")
    if args.prefix_length < 1:
        raise ValueError("prefix-length must be positive")
    total_positions = 1 + args.warmup_steps + args.decode_steps
    if args.prefix_length + total_positions > args.cache_length:
        raise ValueError(
            "prefix plus compile/warmup/measured decode steps exceeds cache length: "
            f"{args.prefix_length} + {total_positions} > {args.cache_length}"
        )

    torch.npu.set_device(local_rank)
    torch.npu.set_compile_mode(jit_compile=False)
    dist.init_process_group("hccl")
    torchair.patch_for_hcom()
    device = torch.device(f"npu:{local_rank}")
    dtype = torch.bfloat16

    full_config = Qwen3TPConfig.from_model_dir(args.model_dir)
    validate_official_shape(full_config)
    config = (
        full_config.with_num_hidden_layers(args.layers)
        if args.layers is not None
        else full_config
    )
    log(rank, f"constructing {config.num_hidden_layers}-layer TP2 shard")
    model = build_model(
        config,
        rank=rank,
        world_size=world_size,
        device=device,
        dtype=dtype,
    )
    log(rank, f"allocated model shard: {json.dumps(memory_snapshot(), sort_keys=True)}")
    load_tp_checkpoint(
        model,
        args.model_dir,
        device=device,
        progress=lambda message: log(rank, message),
    )
    model.prepare_decode(cache_length=args.cache_length)
    model.eval()
    log(rank, f"loaded model shard: {json.dumps(memory_snapshot(), sort_keys=True)}")

    cache = Qwen3TPStaticCache(
        config,
        tp_size=world_size,
        batch_size=1,
        cache_length=args.cache_length,
        device=device,
        dtype=dtype,
    )
    next_token = torch.tensor([[1]], dtype=torch.int64, device=device)
    log(rank, f"allocated static KV: {json.dumps(memory_snapshot(), sort_keys=True)}")

    eager_decode = model.decode
    if args.backend == "torchair":
        compiler_config = CompilerConfig()
        decode_one = torch.compile(
            eager_decode,
            backend=torchair.get_npu_backend(compiler_config=compiler_config),
            dynamic=False,
            fullgraph=True,
        )
    else:
        decode_one = eager_decode

    dist.barrier()
    torch.npu.synchronize()
    compile_started = time.perf_counter()
    next_token = run_decode_steps(
        decode_one,
        next_token,
        cache.key_caches,
        cache.value_caches,
        first_position=args.prefix_length - 1,
        steps=1,
        device=device,
    )
    torch.npu.synchronize()
    compile_first_call_sec = reduce_max_seconds(
        time.perf_counter() - compile_started, device
    )
    log(rank, f"first decode call complete in {compile_first_call_sec:.3f}s")

    next_token = run_decode_steps(
        decode_one,
        next_token,
        cache.key_caches,
        cache.value_caches,
        first_position=args.prefix_length,
        steps=args.warmup_steps,
        device=device,
    )
    torch.npu.synchronize()
    dist.barrier()

    measured_first_position = args.prefix_length + args.warmup_steps
    started = time.perf_counter()
    next_token = run_decode_steps(
        decode_one,
        next_token,
        cache.key_caches,
        cache.value_caches,
        first_position=measured_first_position,
        steps=args.decode_steps,
        device=device,
    )
    torch.npu.synchronize()
    elapsed_sec = reduce_max_seconds(time.perf_counter() - started, device)
    result = {
        "model": "Qwen/Qwen3-32B",
        "dtype": "bfloat16",
        "tensor_parallel_size": world_size,
        "layers": config.num_hidden_layers,
        "full_model_layers": full_config.num_hidden_layers,
        "backend": args.backend,
        "batch_size": 1,
        "cache_length": args.cache_length,
        "prefix_length": args.prefix_length,
        "decode_steps": args.decode_steps,
        "compile_first_call_sec": compile_first_call_sec,
        "decode_elapsed_sec": elapsed_sec,
        "decode_tok_s": args.decode_steps / elapsed_sec,
        "mean_tpot_ms": 1000.0 * elapsed_sec / args.decode_steps,
        "final_token_id": int(next_token.item()),
        "rank0_memory": memory_snapshot() if rank == 0 else None,
        "contracts": {
            "q_heads_per_rank": config.num_attention_heads // world_size,
            "kv_heads_per_rank": config.num_key_value_heads // world_size,
            "kv_heads_replicated": False,
            "qkv": "packed_column_parallel",
            "gate_up": "packed_column_parallel",
            "o_proj": "row_parallel_all_reduce",
            "down_proj": "row_parallel_all_reduce",
            "embedding": "vocab_parallel_all_reduce",
            "lm_head": "vocab_parallel_local_argmax_pair_all_gather",
            "kv_cache": "contiguous_static_per_rank",
            "paged_attention": False,
        },
    }
    if rank == 0:
        print("QWEN3_32B_TP2_RESULT " + json.dumps(result, sort_keys=True), flush=True)
        if args.json_out:
            output_path = Path(args.json_out)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
