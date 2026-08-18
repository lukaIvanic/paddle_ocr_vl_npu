#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu
from torch_npu.dynamo import torchair
from torch_npu.dynamo.torchair.configs.compiler_config import CompilerConfig

from checkpoint_tp2 import load_tp_stage_checkpoint
from modeling_qwen3_moe_pipeline import Qwen3MoeConfig
from modeling_qwen3_moe_tp2 import Qwen3MoeTPStage
from runtime import memory_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Complete static-KV Qwen3-30B-A3B BF16 TP2 decode."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--cache-length", type=int, default=4096)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--decode-steps", type=int, default=20)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument(
        "--profile-dir",
        type=Path,
        help="Optional torch-npu profiler output root, with one subdirectory per rank.",
    )
    return parser.parse_args()


def log(rank: int, message: str) -> None:
    print(f"[tp rank {rank}] {message}", flush=True)


def reduce_max_seconds(elapsed: float, device: torch.device) -> float:
    value = torch.tensor([elapsed], dtype=torch.float32, device=device)
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return float(value.item())


def gather_full_logits(
    local_logits: torch.Tensor, world_size: int
) -> torch.Tensor:
    batch, local_vocab = local_logits.shape
    gathered = torch.empty(
        (world_size * batch, local_vocab),
        dtype=local_logits.dtype,
        device=local_logits.device,
    )
    dist.all_gather_into_tensor(gathered, local_logits.contiguous())
    return gathered.view(world_size, batch, local_vocab).transpose(0, 1).reshape(
        batch, world_size * local_vocab
    )


def distributed_local_argmax(
    local_logits: torch.Tensor,
    *,
    vocab_start: int,
    world_size: int,
) -> torch.Tensor:
    local_value, local_index = local_logits.float().max(dim=-1)
    local_pair = torch.stack(
        (local_value, (local_index + int(vocab_start)).float()), dim=-1
    )
    gathered = torch.empty(
        (world_size * local_pair.shape[0], 2),
        dtype=local_pair.dtype,
        device=local_pair.device,
    )
    dist.all_gather_into_tensor(gathered, local_pair.contiguous())
    candidates = gathered.view(world_size, local_pair.shape[0], 2).transpose(0, 1)
    winner = candidates[:, :, 0].argmax(dim=-1, keepdim=True)
    return candidates[:, :, 1].gather(-1, winner).to(torch.int64).view(-1, 1)


def timed_decode(
    decode,
    token: torch.Tensor,
    position: int,
    cache,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    cache_position = torch.tensor([position], dtype=torch.int64, device=device)
    torch.npu.synchronize()
    started = time.perf_counter()
    output = decode(
        token,
        cache_position,
        cache.key_caches,
        cache.value_caches,
    )
    torch.npu.synchronize()
    return output, time.perf_counter() - started


def validate_capture(
    decode,
    stage: Qwen3MoeTPStage,
    capture,
    *,
    cache_length: int,
    world_size: int,
    device: torch.device,
) -> tuple[object, torch.Tensor, list[float], list[dict[str, object]]]:
    cache = stage.make_cache(cache_length=cache_length)
    prompt_ids = [int(token) for token in capture["prompt_token_ids"]]
    call_times = []
    for position, token_id in enumerate(prompt_ids[:-1]):
        token = torch.tensor([[token_id]], dtype=torch.int64, device=device)
        _output, elapsed = timed_decode(
            decode, token, position, cache, device=device
        )
        call_times.append(elapsed)

    current_token = torch.tensor(
        [[prompt_ids[-1]]], dtype=torch.int64, device=device
    )
    checks = []
    first_generation_position = len(prompt_ids) - 1
    for step, expected_token_id in enumerate(capture["generated_token_ids"]):
        local_logits, elapsed = timed_decode(
            decode,
            current_token,
            first_generation_position + step,
            cache,
            device=device,
        )
        call_times.append(elapsed)
        full_logits = gather_full_logits(local_logits, world_size)
        current_token = full_logits.argmax(dim=-1, keepdim=True)
        expected_logits = capture["expected_logits"][step].to(
            device=device, dtype=full_logits.dtype
        )
        diff = (full_logits.float() - expected_logits.float()).abs()
        token_id = int(current_token.item())
        checks.append(
            {
                "step": step,
                "token_id": token_id,
                "expected_token_id": int(expected_token_id),
                "token_match": token_id == int(expected_token_id),
                "logit_max_abs": float(diff.max().item()),
                "logit_mean_abs": float(diff.mean().item()),
            }
        )
    return cache, current_token, call_times, checks


def benchmark_continuation(
    decode,
    cache,
    current_token: torch.Tensor,
    stage: Qwen3MoeTPStage,
    capture,
    *,
    world_size: int,
    device: torch.device,
    warmup_steps: int,
    decode_steps: int,
) -> dict[str, float]:
    first_position = (
        len(capture["prompt_token_ids"])
        - 1
        + len(capture["generated_token_ids"])
    )
    for offset in range(warmup_steps):
        cache_position = torch.tensor(
            [first_position + offset], dtype=torch.int64, device=device
        )
        local_logits = decode(
            current_token,
            cache_position,
            cache.key_caches,
            cache.value_caches,
        )
        current_token = distributed_local_argmax(
            local_logits,
            vocab_start=stage.vocab_start,
            world_size=world_size,
        )
    torch.npu.synchronize()
    dist.barrier()
    started = time.perf_counter()
    for offset in range(decode_steps):
        cache_position = torch.tensor(
            [first_position + warmup_steps + offset],
            dtype=torch.int64,
            device=device,
        )
        local_logits = decode(
            current_token,
            cache_position,
            cache.key_caches,
            cache.value_caches,
        )
        current_token = distributed_local_argmax(
            local_logits,
            vocab_start=stage.vocab_start,
            world_size=world_size,
        )
    torch.npu.synchronize()
    elapsed = reduce_max_seconds(time.perf_counter() - started, device)
    return {
        "steps": decode_steps,
        "elapsed_sec": elapsed,
        "mean_tpot_ms": 1000.0 * elapsed / decode_steps,
        "tok_s": decode_steps / elapsed,
        "final_token_id": int(current_token.item()),
    }


def profile_compiled_step(
    decode,
    cache,
    stage: Qwen3MoeTPStage,
    capture,
    benchmark: dict[str, float],
    *,
    profile_root: Path,
    rank: int,
    world_size: int,
    device: torch.device,
    warmup_steps: int,
    decode_steps: int,
) -> dict[str, object]:
    import torch_npu.profiler as npu_prof

    profile_dir = profile_root / f"rank{rank}"
    if profile_dir.exists():
        shutil.rmtree(profile_dir)
    profile_dir.mkdir(parents=True)
    current_token = torch.tensor(
        [[int(benchmark["final_token_id"])]], dtype=torch.int64, device=device
    )
    first_position = (
        len(capture["prompt_token_ids"])
        - 1
        + len(capture["generated_token_ids"])
        + warmup_steps
        + decode_steps
    )
    synchronized_seconds = []
    schedule = npu_prof.schedule(wait=0, warmup=1, active=1, repeat=1)
    experimental_config = npu_prof._ExperimentalConfig(
        profiler_level=npu_prof.ProfilerLevel.Level1,
        aic_metrics=npu_prof.AiCMetrics.PipeUtilization,
        export_type=npu_prof.ExportType.Text,
    )
    dist.barrier()
    with npu_prof.profile(
        activities=[npu_prof.ProfilerActivity.CPU, npu_prof.ProfilerActivity.NPU],
        schedule=schedule,
        on_trace_ready=npu_prof.tensorboard_trace_handler(
            str(profile_dir), analyse_flag=True
        ),
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
        experimental_config=experimental_config,
    ) as prof:
        for offset in range(2):
            cache_position = torch.tensor(
                [first_position + offset], dtype=torch.int64, device=device
            )
            torch.npu.synchronize()
            started = time.perf_counter()
            local_logits = decode(
                current_token,
                cache_position,
                cache.key_caches,
                cache.value_caches,
            )
            current_token = distributed_local_argmax(
                local_logits,
                vocab_start=stage.vocab_start,
                world_size=world_size,
            )
            torch.npu.synchronize()
            synchronized_seconds.append(time.perf_counter() - started)
            prof.step()
    torch.npu.synchronize()
    return {
        "rank": rank,
        "profile_dir": str(profile_dir),
        "synchronized_seconds": synchronized_seconds,
    }


def main() -> None:
    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise ValueError(f"TP2 requires two ranks, got {world_size}")
    torch.npu.set_device(local_rank)
    torch.npu.set_compile_mode(jit_compile=False)
    dist.init_process_group("hccl")
    torchair.patch_for_hcom()
    device = torch.device(f"npu:{local_rank}")

    capture = torch.load(args.capture, map_location="cpu", weights_only=False)
    if capture["format"] != "qwen3_30b_a3b_stage2_replay_v1":
        raise ValueError(f"Unsupported capture format: {capture['format']}")
    final_position = (
        len(capture["prompt_token_ids"])
        - 1
        + len(capture["generated_token_ids"])
        + args.warmup_steps
        + args.decode_steps
    )
    if final_position >= args.cache_length:
        raise ValueError("Validation and benchmark exceed the static KV cache")

    config = Qwen3MoeConfig.from_model_dir(args.model_dir)
    config.validate_qwen3_30b_a3b()
    with torch.device("meta"):
        stage = Qwen3MoeTPStage(
            config,
            tp_rank=rank,
            tp_size=world_size,
            layer_start=0,
            layer_end=config.num_hidden_layers,
            with_lm_head=True,
            with_embedding=True,
        )
    stage = stage.to(dtype=torch.bfloat16)
    stage.to_empty(device=device)
    log(rank, "allocated full TP shard: " + json.dumps(memory_snapshot(device)))
    load_started = time.perf_counter()
    load_tp_stage_checkpoint(
        stage,
        args.model_dir,
        device=device,
        progress=lambda message: log(rank, message),
    )
    stage.prepare_decode(cache_length=args.cache_length)
    stage.eval()
    load_sec = reduce_max_seconds(time.perf_counter() - load_started, device)
    log(rank, "loaded full TP shard: " + json.dumps(memory_snapshot(device)))

    with torch.inference_mode():
        eager_cache, eager_token, eager_times, eager_checks = validate_capture(
            stage.decode_input_ids_local_output,
            stage,
            capture,
            cache_length=args.cache_length,
            world_size=world_size,
            device=device,
        )
        eager_benchmark = benchmark_continuation(
            stage.decode_input_ids_local_output,
            eager_cache,
            eager_token,
            stage,
            capture,
            world_size=world_size,
            device=device,
            warmup_steps=args.warmup_steps,
            decode_steps=args.decode_steps,
        )

        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        compiled = torch.compile(
            stage.decode_input_ids_local_output,
            backend=torchair.get_npu_backend(
                compiler_config=CompilerConfig()
            ),
            dynamic=False,
            fullgraph=True,
        )
        compiled_cache, compiled_token, compiled_times, compiled_checks = (
            validate_capture(
                compiled,
                stage,
                capture,
                cache_length=args.cache_length,
                world_size=world_size,
                device=device,
            )
        )
        stats_after_capture = {
            "unique_graphs": int(
                torch._dynamo.utils.counters["stats"]["unique_graphs"]
            ),
            "calls_captured": int(
                torch._dynamo.utils.counters["stats"]["calls_captured"]
            ),
        }
        compiled_benchmark = benchmark_continuation(
            compiled,
            compiled_cache,
            compiled_token,
            stage,
            capture,
            world_size=world_size,
            device=device,
            warmup_steps=args.warmup_steps,
            decode_steps=args.decode_steps,
        )
        stats_final = {
            "unique_graphs": int(
                torch._dynamo.utils.counters["stats"]["unique_graphs"]
            ),
            "calls_captured": int(
                torch._dynamo.utils.counters["stats"]["calls_captured"]
            ),
        }
        profile_summary = None
        if args.profile_dir is not None:
            profile_summary = profile_compiled_step(
                compiled,
                compiled_cache,
                stage,
                capture,
                compiled_benchmark,
                profile_root=args.profile_dir,
                rank=rank,
                world_size=world_size,
                device=device,
                warmup_steps=args.warmup_steps,
                decode_steps=args.decode_steps,
            )

    parity_passed = all(
        bool(check["token_match"])
        for check in eager_checks + compiled_checks
    )
    no_recompilations = (
        stats_final["unique_graphs"] == stats_after_capture["unique_graphs"]
    )
    summary = {
        "model": capture["model"],
        "chip": "Ascend 910B2",
        "dtype": "bfloat16",
        "tensor_parallel_size": world_size,
        "layers": config.num_hidden_layers,
        "cache_length": args.cache_length,
        "collectives_per_token_in_graph": 1 + 2 * config.num_hidden_layers,
        "load_sec": load_sec,
        "parity_passed": parity_passed,
        "eager": {
            "call_times_sec": eager_times,
            "checks": eager_checks,
            "benchmark": eager_benchmark,
        },
        "compiled": {
            "call_times_sec": compiled_times,
            "checks": compiled_checks,
            "benchmark": compiled_benchmark,
        },
        "compile_first_call_sec": reduce_max_seconds(
            compiled_times[0], device
        ),
        "speedup": eager_benchmark["mean_tpot_ms"]
        / compiled_benchmark["mean_tpot_ms"],
        "dynamo_after_capture": stats_after_capture,
        "dynamo_final": stats_final,
        "no_recompilations_after_capture": no_recompilations,
        "profile": profile_summary,
        "rank0_memory": memory_snapshot(device) if rank == 0 else None,
        "contracts": {
            "embedding": "vocab_parallel_all_reduce",
            "q_heads_per_rank": config.num_attention_heads // world_size,
            "kv_heads_per_rank": config.num_key_value_heads // world_size,
            "moe_intermediate_per_rank": config.moe_intermediate_size
            // world_size,
            "attention_o": "row_parallel_all_reduce",
            "expert_routing": "fused_gating_init_routing_v2",
            "expert_compute": "persistent_bf16_grouped_matmul",
            "expert_finalize": "npu_moe_finalize_routing_then_all_reduce",
            "expert_down": "row_parallel_all_reduce",
            "qk_norm": "fresh_per_call_npu_add_rms_norm_zero_banks",
            "lm_head": "vocab_parallel_local_pair_all_gather",
        },
    }
    if rank == 0:
        print("QWEN3_MOE_FULL_TP2 " + json.dumps(summary, sort_keys=True))
        if args.summary_out:
            args.summary_out.parent.mkdir(parents=True, exist_ok=True)
            args.summary_out.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n"
            )
    if not parity_passed or not no_recompilations:
        raise RuntimeError("Full TP2 parity or static-graph contract failed")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
