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

from checkpoint_tp2 import load_tp_stage_checkpoint
from modeling_qwen3_moe_pipeline import Qwen3MoeConfig
from modeling_qwen3_moe_tp2 import Qwen3MoeTPStage
from runtime import memory_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the captured Qwen3-30B-A3B second half as static TP2 and "
            "compare eager versus TorchAir."
        )
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--decode-steps", type=int, default=20)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--logit-atol", type=float, default=1.0)
    parser.add_argument("--logit-rtol", type=float, default=5e-2)
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
    global_index = local_index + int(vocab_start)
    local_pair = torch.stack((local_value, global_index.float()), dim=-1)
    gathered = torch.empty(
        (world_size * local_pair.shape[0], 2),
        dtype=local_pair.dtype,
        device=local_pair.device,
    )
    dist.all_gather_into_tensor(gathered, local_pair.contiguous())
    candidates = gathered.view(world_size, local_pair.shape[0], 2).transpose(0, 1)
    winner = candidates[:, :, 0].argmax(dim=-1, keepdim=True)
    return candidates[:, :, 1].gather(-1, winner).to(torch.int64)


def restore_prefix(cache, capture, *, rank: int, world_size: int) -> None:
    restored = cache.restore_full_prefix(
        capture["stage2_prefix_cache"],
        tp_rank=rank,
        tp_size=world_size,
    )
    if restored != int(capture["prefix_length"]):
        raise RuntimeError(
            f"Restored prefix {restored}, expected {capture['prefix_length']}"
        )


def router_debug_check(
    stage: Qwen3MoeTPStage,
    capture,
    *,
    rank: int,
    world_size: int,
    device: torch.device,
) -> dict[str, object]:
    cache = stage.make_cache(cache_length=int(capture["cache_length"]))
    restore_prefix(cache, capture, rank=rank, world_size=world_size)
    boundary = capture["boundary_hidden_states"][0].to(
        device=device, dtype=torch.bfloat16
    )
    position = torch.tensor(
        [capture["cache_positions"][0]], dtype=torch.int64, device=device
    )
    with torch.inference_mode():
        output, indices, weights = stage.decode_debug(
            boundary,
            position,
            cache.key_caches,
            cache.value_caches,
        )
    local_indices = torch.stack(indices).reshape(stage.num_layers, -1).to(torch.int64)
    local_weights = torch.stack(weights).reshape(stage.num_layers, -1).float()
    gathered_indices = torch.empty(
        (world_size * stage.num_layers, local_indices.shape[1]),
        dtype=local_indices.dtype,
        device=device,
    )
    gathered_weights = torch.empty(
        (world_size * stage.num_layers, local_weights.shape[1]),
        dtype=local_weights.dtype,
        device=device,
    )
    dist.all_gather_into_tensor(gathered_indices, local_indices.contiguous())
    dist.all_gather_into_tensor(gathered_weights, local_weights.contiguous())
    gathered_indices = gathered_indices.view(
        world_size, stage.num_layers, -1
    )
    gathered_weights = gathered_weights.view(
        world_size, stage.num_layers, -1
    )
    expected_indices = torch.stack(
        capture["stage2_router_indices"][0][: stage.num_layers]
    ).reshape(stage.num_layers, -1).to(device=device, dtype=torch.int64)
    expected_weights = torch.stack(
        capture["stage2_router_weights"][0][: stage.num_layers]
    ).reshape(stage.num_layers, -1).to(device=device, dtype=torch.float32)
    indices_cross_rank = bool(
        torch.equal(gathered_indices[0], gathered_indices[1])
    )
    weights_cross_rank = bool(
        torch.equal(gathered_weights[0], gathered_weights[1])
    )
    indices_match_capture = bool(torch.equal(local_indices, expected_indices))
    weight_diff = (local_weights - expected_weights).abs()
    result = {
        "indices_cross_rank_exact": indices_cross_rank,
        "weights_cross_rank_exact": weights_cross_rank,
        "indices_match_capture": indices_match_capture,
        "weights_vs_capture_max_abs": float(weight_diff.max().item()),
        "weights_vs_capture_mean_abs": float(weight_diff.mean().item()),
    }
    if stage.norm is not None and stage.lm_head is not None:
        full_logits = gather_full_logits(output, world_size)
        expected_logits = capture["expected_logits"][0].to(
            device=device, dtype=full_logits.dtype
        )
        diff = (full_logits.float() - expected_logits.float()).abs()
        result.update(
            {
                "token_id": int(full_logits.argmax(dim=-1).item()),
                "expected_token_id": int(capture["generated_token_ids"][0]),
                "logit_max_abs": float(diff.max().item()),
                "logit_mean_abs": float(diff.mean().item()),
            }
        )
    if not (
        indices_cross_rank
        and weights_cross_rank
        and indices_match_capture
    ):
        raise RuntimeError(f"TP2 router debug parity failed: {result}")
    return result


def run_capture_sequence(
    decode,
    cache,
    stage: Qwen3MoeTPStage,
    capture,
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    complete_stage: bool,
) -> tuple[list[dict[str, object]], list[float]]:
    checks = []
    timings = []
    for step, (boundary_cpu, position_value) in enumerate(
        zip(capture["boundary_hidden_states"], capture["cache_positions"])
    ):
        boundary = boundary_cpu.to(device=device, dtype=torch.bfloat16)
        position = torch.tensor(
            [position_value], dtype=torch.int64, device=device
        )
        torch.npu.synchronize()
        started = time.perf_counter()
        local_output = decode(
            boundary,
            position,
            cache.key_caches,
            cache.value_caches,
        )
        if complete_stage:
            full_output = gather_full_logits(local_output, world_size)
        else:
            full_output = local_output
        torch.npu.synchronize()
        timings.append(time.perf_counter() - started)
        check = {"step": step}
        if complete_stage:
            expected = capture["expected_logits"][step].to(
                device=device, dtype=full_output.dtype
            )
            diff = (full_output.float() - expected.float()).abs()
            token_id = int(full_output.argmax(dim=-1).item())
            check.update(
                {
                    "token_id": token_id,
                    "expected_token_id": int(capture["generated_token_ids"][step]),
                    "token_match": token_id
                    == int(capture["generated_token_ids"][step]),
                    "logit_max_abs": float(diff.max().item()),
                    "logit_mean_abs": float(diff.mean().item()),
                }
            )
        else:
            rank_copy = torch.empty(
                (world_size * full_output.shape[0], *full_output.shape[1:]),
                dtype=full_output.dtype,
                device=device,
            )
            dist.all_gather_into_tensor(rank_copy, full_output.contiguous())
            rank_copy = rank_copy.view(world_size, *full_output.shape)
            check["cross_rank_max_abs"] = float(
                (rank_copy[0].float() - rank_copy[1].float()).abs().max().item()
            )
        checks.append(check)
    return checks, timings


def benchmark_continuation(
    decode,
    cache,
    stage: Qwen3MoeTPStage,
    capture,
    *,
    world_size: int,
    device: torch.device,
    complete_stage: bool,
    warmup_steps: int,
    decode_steps: int,
) -> dict[str, float]:
    boundary = capture["boundary_hidden_states"][-1].to(
        device=device, dtype=torch.bfloat16
    )
    first_position = int(capture["cache_positions"][-1]) + 1
    for offset in range(warmup_steps):
        position = torch.tensor(
            [first_position + offset], dtype=torch.int64, device=device
        )
        output = decode(
            boundary, position, cache.key_caches, cache.value_caches
        )
        if complete_stage:
            distributed_local_argmax(
                output,
                vocab_start=stage.vocab_start,
                world_size=world_size,
            )
    torch.npu.synchronize()
    dist.barrier()
    started = time.perf_counter()
    for offset in range(decode_steps):
        position = torch.tensor(
            [first_position + warmup_steps + offset],
            dtype=torch.int64,
            device=device,
        )
        output = decode(
            boundary, position, cache.key_caches, cache.value_caches
        )
        if complete_stage:
            distributed_local_argmax(
                output,
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
    }


def main() -> None:
    args = parse_args()
    if not 1 <= args.layers <= 24:
        raise ValueError("--layers must be in [1, 24]")
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
    if int(capture["stage2_layer_start"]) != 24:
        raise ValueError(
            "This TP2 replay expects the verified layer-24 stage boundary, got "
            f"{capture['stage2_layer_start']}"
        )
    final_position = (
        int(capture["cache_positions"][-1])
        + args.warmup_steps
        + args.decode_steps
    )
    if final_position >= int(capture["cache_length"]):
        raise ValueError(
            "Capture plus warmup and benchmark exceeds the static KV cache: "
            f"final_position={final_position}, "
            f"cache_length={capture['cache_length']}"
        )
    config = Qwen3MoeConfig.from_model_dir(args.model_dir)
    config.validate_qwen3_30b_a3b()
    complete_stage = args.layers == 24

    with torch.device("meta"):
        stage = Qwen3MoeTPStage(
            config,
            tp_rank=rank,
            tp_size=world_size,
            layer_start=24,
            layer_end=24 + args.layers,
            with_lm_head=complete_stage,
        )
    stage = stage.to(dtype=torch.bfloat16)
    stage.to_empty(device=device)
    log(rank, f"allocated TP shard: {json.dumps(memory_snapshot(device), sort_keys=True)}")
    load_started = time.perf_counter()
    load_tp_stage_checkpoint(
        stage,
        args.model_dir,
        device=device,
        progress=lambda message: log(rank, message),
    )
    stage.prepare_decode(cache_length=int(capture["cache_length"]))
    stage.eval()
    load_sec = time.perf_counter() - load_started
    log(rank, f"loaded TP shard: {json.dumps(memory_snapshot(device), sort_keys=True)}")

    debug = router_debug_check(
        stage,
        capture,
        rank=rank,
        world_size=world_size,
        device=device,
    )
    log(rank, "router debug: " + json.dumps(debug, sort_keys=True))

    eager_cache = stage.make_cache(cache_length=int(capture["cache_length"]))
    restore_prefix(eager_cache, capture, rank=rank, world_size=world_size)
    eager_checks, eager_capture_times = run_capture_sequence(
        stage.decode_local_output,
        eager_cache,
        stage,
        capture,
        rank=rank,
        world_size=world_size,
        device=device,
        complete_stage=complete_stage,
    )
    eager_benchmark = benchmark_continuation(
        stage.decode_local_output,
        eager_cache,
        stage,
        capture,
        world_size=world_size,
        device=device,
        complete_stage=complete_stage,
        warmup_steps=args.warmup_steps,
        decode_steps=args.decode_steps,
    )

    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    compiled = torch.compile(
        stage.decode_local_output,
        backend=torchair.get_npu_backend(compiler_config=CompilerConfig()),
        dynamic=False,
        fullgraph=True,
    )
    compiled_cache = stage.make_cache(cache_length=int(capture["cache_length"]))
    restore_prefix(compiled_cache, capture, rank=rank, world_size=world_size)
    compiled_checks, compiled_capture_times = run_capture_sequence(
        compiled,
        compiled_cache,
        stage,
        capture,
        rank=rank,
        world_size=world_size,
        device=device,
        complete_stage=complete_stage,
    )
    stats_after_capture = {
        "unique_graphs": int(torch._dynamo.utils.counters["stats"]["unique_graphs"]),
        "calls_captured": int(torch._dynamo.utils.counters["stats"]["calls_captured"]),
    }
    compiled_benchmark = benchmark_continuation(
        compiled,
        compiled_cache,
        stage,
        capture,
        world_size=world_size,
        device=device,
        complete_stage=complete_stage,
        warmup_steps=args.warmup_steps,
        decode_steps=args.decode_steps,
    )
    stats_final = {
        "unique_graphs": int(torch._dynamo.utils.counters["stats"]["unique_graphs"]),
        "calls_captured": int(torch._dynamo.utils.counters["stats"]["calls_captured"]),
    }
    graph_delta = stats_final["unique_graphs"] - stats_after_capture["unique_graphs"]
    parity_passed = True
    if complete_stage:
        parity_passed = all(
            bool(check["token_match"])
            for check in eager_checks + compiled_checks
        )
    else:
        parity_passed = all(
            float(check["cross_rank_max_abs"]) == 0.0
            for check in eager_checks + compiled_checks
        )

    summary = {
        "model": capture["model"],
        "chip": "Ascend 910B2",
        "dtype": "bfloat16",
        "tensor_parallel_size": world_size,
        "layers": args.layers,
        "complete_stage": complete_stage,
        "load_sec": reduce_max_seconds(load_sec, device),
        "router_debug": debug,
        "parity_passed": parity_passed,
        "eager": {
            "capture_step_times_sec": eager_capture_times,
            "checks": eager_checks,
            "benchmark": eager_benchmark,
        },
        "compiled": {
            "capture_step_times_sec": compiled_capture_times,
            "checks": compiled_checks,
            "benchmark": compiled_benchmark,
        },
        "compile_first_call_sec": reduce_max_seconds(
            compiled_capture_times[0], device
        ),
        "speedup": eager_benchmark["mean_tpot_ms"]
        / compiled_benchmark["mean_tpot_ms"],
        "dynamo_after_capture": stats_after_capture,
        "dynamo_final": stats_final,
        "no_recompilations_after_capture": graph_delta == 0,
        "rank0_memory": memory_snapshot(device) if rank == 0 else None,
        "contracts": {
            "q_heads_per_rank": config.num_attention_heads // world_size,
            "kv_heads_per_rank": config.num_key_value_heads // world_size,
            "moe_intermediate_per_rank": config.moe_intermediate_size
            // world_size,
            "router": "replicated_fp32_softmax_top8",
            "attention_o": "row_parallel_all_reduce",
            "expert_gate_up": "column_parallel_selected_expert_bmm",
            "expert_down": "row_parallel_selected_expert_bmm_all_reduce",
            "lm_head": (
                "vocab_parallel_local_pair_all_gather"
                if complete_stage
                else "absent"
            ),
        },
    }
    if rank == 0:
        print("QWEN3_MOE_STAGE2_TP2 " + json.dumps(summary, sort_keys=True), flush=True)
        if args.summary_out:
            args.summary_out.parent.mkdir(parents=True, exist_ok=True)
            args.summary_out.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n"
            )
    if not parity_passed or graph_delta != 0:
        raise RuntimeError("TP2 parity or static-graph contract failed")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
