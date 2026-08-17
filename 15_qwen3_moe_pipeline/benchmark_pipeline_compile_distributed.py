#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu

from modeling_qwen3_moe_pipeline import Qwen3MoeConfig
from runtime import build_stage, memory_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Two-process HCCL pipeline benchmark for static compiled "
            "Qwen3-30B-A3B B1 decode."
        )
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--capture", required=True)
    parser.add_argument(
        "--cache-length",
        type=int,
        help="Static KV capacity. Default: use the capture capacity.",
    )
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--decode-steps", type=int, default=20)
    parser.add_argument(
        "--compile-cache-dir",
        type=Path,
        default=Path(".runtime_cache/15_qwen3_moe_pipeline"),
    )
    parser.add_argument("--summary-out", type=Path, required=True)
    return parser.parse_args()


def source_hash() -> str:
    model_file = Path(__file__).resolve().parent / "modeling_qwen3_moe_pipeline.py"
    return hashlib.sha256(model_file.read_bytes()).hexdigest()[:12]


def import_cache_compile():
    try:
        from torch_npu.dynamo.torchair.inference import cache_compile
    except ImportError:
        from torchair.inference import cache_compile
    return cache_compile


def compile_static(method, *, cache_dir: Path):
    from torch_npu.dynamo.torchair.configs.compiler_config import CompilerConfig

    cache_was_warm = cache_dir.is_dir() and any(cache_dir.iterdir())
    cache_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    compiled = import_cache_compile()(
        method,
        config=CompilerConfig(),
        dynamic=False,
        cache_dir=str(cache_dir),
        ge_cache=True,
        fullgraph=True,
    )
    return compiled, time.perf_counter() - started, cache_was_warm


def dynamo_stats() -> dict[str, int]:
    counters = torch._dynamo.utils.counters
    return {
        "unique_graphs": int(counters["stats"]["unique_graphs"]),
        "calls_captured": int(counters["stats"]["calls_captured"]),
        "frames_total": int(counters["frames"]["total"]),
        "frames_ok": int(counters["frames"]["ok"]),
    }


def rank1_summary_path(path: Path) -> Path:
    return path.with_name(path.stem + ".rank1" + path.suffix)


def main() -> None:
    args = parse_args()
    if args.warmup_steps < 0 or args.decode_steps < 1:
        raise ValueError("Invalid warmup/decode step count")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2 or rank not in (0, 1):
        raise ValueError(f"This benchmark requires exactly two ranks, got {world_size}")

    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")
    dist.init_process_group("hccl")
    capture = torch.load(args.capture, map_location="cpu", weights_only=False)
    if capture["format"] != "qwen3_30b_a3b_stage2_replay_v1":
        raise ValueError(f"Unsupported capture format: {capture['format']}")
    config = Qwen3MoeConfig.from_model_dir(args.model_dir)
    config.validate_qwen3_30b_a3b()
    cache_length = (
        int(args.cache_length)
        if args.cache_length is not None
        else int(capture["cache_length"])
    )
    prompt_token_ids = list(capture["prompt_token_ids"])
    expected_token_ids = list(capture["generated_token_ids"])
    total_steps = (
        len(prompt_token_ids) - 1
        + len(expected_token_ids)
        + args.warmup_steps
        + args.decode_steps
    )
    if total_steps > cache_length:
        raise ValueError(f"Run needs {total_steps} cache rows, capacity is {cache_length}")

    if rank == 0:
        layer_start, layer_end = 0, 24
        with_embedding, with_lm_head = True, False
        stage_name = "distributed-stage0"
    else:
        layer_start, layer_end = 24, 48
        with_embedding, with_lm_head = False, True
        stage_name = "distributed-stage1"
    stage, stage_metadata = build_stage(
        config,
        args.model_dir,
        layer_start=layer_start,
        layer_end=layer_end,
        with_embedding=with_embedding,
        with_lm_head=with_lm_head,
        device=device,
        name=stage_name,
        cache_length=cache_length,
    )
    cache = stage.make_cache(cache_length=cache_length)

    cache_dir = args.compile_cache_dir.expanduser().resolve() / (
        f"distributed_stage{rank}_l24_b1_kv{cache_length}_bf16_"
        f"src{source_hash()}"
    )
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    method = (
        stage.decode_static_input_ids if rank == 0 else stage.decode_static_output
    )
    compiled, wrapper_sec, cache_was_warm = compile_static(
        method,
        cache_dir=cache_dir,
    )
    dist.barrier()

    boundary_buffer = torch.empty(
        (1, 1, config.hidden_size), dtype=torch.bfloat16, device=device
    )
    token_buffer = torch.empty((1, 1), dtype=torch.int64, device=device)
    generated_token_ids = []
    first_call_sec = None
    stats_after_prompt = None
    measured_started = None
    measured_sec = None

    for position in range(total_steps):
        if rank == 0:
            if position < len(prompt_token_ids):
                input_ids = torch.tensor(
                    [[prompt_token_ids[position]]], dtype=torch.int64, device=device
                )
            else:
                input_ids = token_buffer
            cache_position = torch.tensor(
                [position], dtype=torch.int64, device=device
            )
            if position == 0:
                torch.npu.synchronize()
                first_started = time.perf_counter()
            if position == (
                len(prompt_token_ids) - 1
                + len(expected_token_ids)
                + args.warmup_steps
            ):
                dist.barrier()
                torch.npu.synchronize()
                measured_started = time.perf_counter()
            boundary = compiled(
                input_ids,
                cache_position,
                cache.key_caches,
                cache.value_caches,
            )
            dist.send(boundary.contiguous(), dst=1)
            dist.recv(token_buffer, src=1)
            if position == 0:
                torch.npu.synchronize()
                first_call_sec = time.perf_counter() - first_started
            if position == len(prompt_token_ids) - 2:
                stats_after_prompt = dynamo_stats()
            generation_index = position - (len(prompt_token_ids) - 1)
            if 0 <= generation_index < len(expected_token_ids):
                token_id = int(token_buffer.item())
                generated_token_ids.append(token_id)
                expected = int(expected_token_ids[generation_index])
                if token_id != expected:
                    raise RuntimeError(
                        f"Token mismatch at generation step {generation_index}: "
                        f"got {token_id}, expected {expected}"
                    )
        else:
            cache_position = torch.tensor(
                [position], dtype=torch.int64, device=device
            )
            if position == 0:
                torch.npu.synchronize()
                first_started = time.perf_counter()
            if position == (
                len(prompt_token_ids) - 1
                + len(expected_token_ids)
                + args.warmup_steps
            ):
                dist.barrier()
            dist.recv(boundary_buffer, src=0)
            logits = compiled(
                boundary_buffer,
                cache_position,
                cache.key_caches,
                cache.value_caches,
            )
            token_buffer.copy_(logits.argmax(dim=-1, keepdim=True))
            dist.send(token_buffer, dst=0)
            if position == 0:
                torch.npu.synchronize()
                first_call_sec = time.perf_counter() - first_started
            if position == len(prompt_token_ids) - 2:
                stats_after_prompt = dynamo_stats()

    torch.npu.synchronize()
    if rank == 0:
        if measured_started is None:
            raise RuntimeError("Measured window did not start")
        measured_sec = time.perf_counter() - measured_started
    stats_final = dynamo_stats()
    if stats_after_prompt is None:
        raise RuntimeError("Prompt graph statistics were not captured")
    graph_delta = stats_final["unique_graphs"] - stats_after_prompt["unique_graphs"]
    rank_summary = {
        "rank": rank,
        "device": str(device),
        "stage": stage_metadata,
        "compile_contract": {
            "fullgraph": True,
            "dynamic": False,
            "ge_cache": True,
            "cache_dir": str(cache_dir),
            "cache_was_warm": cache_was_warm,
            "wrapper_sec": wrapper_sec,
            "first_call_sec": first_call_sec,
            "dynamo_after_prompt": stats_after_prompt,
            "dynamo_final": stats_final,
            "unique_graph_delta_after_prompt": graph_delta,
            "no_recompilations_after_prompt": graph_delta == 0,
        },
        "memory": memory_snapshot(device),
    }

    if rank == 1:
        rank1_path = rank1_summary_path(args.summary_out)
        rank1_path.parent.mkdir(parents=True, exist_ok=True)
        rank1_path.write_text(json.dumps(rank_summary, indent=2, sort_keys=True) + "\n")
    dist.barrier()

    if rank == 0:
        rank1_payload = json.loads(rank1_summary_path(args.summary_out).read_text())
        summary = {
            "model": capture["model"],
            "chip": "Ascend 910B2",
            "dtype": "bfloat16",
            "batch_size": 1,
            "pipeline_parallel_size": 2,
            "transport": "two_process_hccl_send_recv",
            "cache_length": cache_length,
            "parity": {
                "passed": generated_token_ids == expected_token_ids,
                "generated_token_ids": generated_token_ids,
                "expected_token_ids": expected_token_ids,
            },
            "benchmark": {
                "warmup_steps": args.warmup_steps,
                "decode_steps": args.decode_steps,
                "elapsed_sec": measured_sec,
                "mean_tpot_ms": 1000.0 * measured_sec / args.decode_steps,
                "tok_s": args.decode_steps / measured_sec,
            },
            "rank0": rank_summary,
            "rank1": rank1_payload,
            "no_recompilations_after_prompt": bool(
                rank_summary["compile_contract"]["no_recompilations_after_prompt"]
                and rank1_payload["compile_contract"][
                    "no_recompilations_after_prompt"
                ]
            ),
        }
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print("QWEN3_MOE_DISTRIBUTED_PIPELINE " + json.dumps(summary, sort_keys=True))
        if not summary["no_recompilations_after_prompt"]:
            raise RuntimeError("Additional static graphs appeared after prompt setup")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
