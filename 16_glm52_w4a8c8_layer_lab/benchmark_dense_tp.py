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

from modeling_glm52_dense_tp import GLM52DenseTPStack, shard_bounds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GLM-5.2 W4A8C8 dense layers 0-2 TP1/TP2 benchmark."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--cache-length", type=int, default=4096)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--decode-steps", type=int, default=200)
    parser.add_argument("--validation-steps", type=int, default=8)
    parser.add_argument(
        "--backend", choices=("raw_eager", "torchair"), default="torchair"
    )
    parser.add_argument("--reference-out", type=Path)
    parser.add_argument("--reference-in", type=Path)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--summary-out", type=Path)
    return parser.parse_args()


def log(rank: int, message: str) -> None:
    print(f"[dense-tp rank {rank}] {message}", flush=True)


def memory_snapshot(device: torch.device) -> dict[str, float]:
    torch.npu.synchronize()
    free_bytes, total_bytes = torch.npu.mem_get_info(device)
    return {
        "allocated_gib": torch.npu.memory_allocated(device) / 2**30,
        "reserved_gib": torch.npu.memory_reserved(device) / 2**30,
        "free_gib": free_bytes / 2**30,
        "total_gib": total_bytes / 2**30,
    }


def barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def reduce_max_seconds(
    elapsed: float, *, world_size: int, device: torch.device
) -> float:
    if world_size == 1:
        return elapsed
    value = torch.tensor([elapsed], dtype=torch.float32, device=device)
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return float(value.item())


def run_steps(
    decode,
    hidden_states: torch.Tensor,
    caches,
    *,
    first_position: int,
    steps: int,
) -> torch.Tensor:
    output = None
    for offset in range(steps):
        position = torch.tensor(
            [first_position + offset],
            dtype=torch.int64,
            device=hidden_states.device,
        )
        output = decode(hidden_states.clone(), position, *caches)
    if output is None:
        raise ValueError("steps must be positive")
    return output


def used_cache_cpu(caches, used_length: int) -> dict[str, tuple[torch.Tensor, ...]]:
    keys, values, indices = caches
    return {
        "keys": tuple(cache[:, :, :used_length].cpu() for cache in keys),
        "values": tuple(cache[:, :, :used_length].cpu() for cache in values),
        "indices": tuple(cache[:, :used_length].cpu() for cache in indices),
    }


def save_reference(
    path: Path,
    output: torch.Tensor,
    caches,
    *,
    used_length: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "format": "glm52_dense_layers0_2_tp1_reference_v1",
        "used_length": used_length,
        "output": output.cpu(),
        **used_cache_cpu(caches, used_length),
    }
    torch.save(record, path)


def compare_reference(
    path: Path,
    output: torch.Tensor,
    caches,
    *,
    rank: int,
    world_size: int,
    total_heads: int,
) -> dict[str, float | bool]:
    reference = torch.load(path, map_location="cpu", weights_only=False)
    if reference["format"] != "glm52_dense_layers0_2_tp1_reference_v1":
        raise ValueError(f"Unsupported reference format: {reference['format']}")
    used_length = int(reference["used_length"])
    output_diff = (output.cpu().float() - reference["output"].float()).abs()
    head_start, head_end = shard_bounds(total_heads, rank, world_size)
    keys, values, indices = caches
    key_max = 0.0
    value_max = 0.0
    index_max = 0.0
    for layer_index in range(3):
        key_diff = (
            keys[layer_index][:, :, :used_length].cpu().float()
            - reference["keys"][layer_index][
                :, head_start:head_end, :used_length
            ].float()
        ).abs()
        value_diff = (
            values[layer_index][:, :, :used_length].cpu().float()
            - reference["values"][layer_index][
                :, head_start:head_end, :used_length
            ].float()
        ).abs()
        index_diff = (
            indices[layer_index][:, :used_length].cpu().float()
            - reference["indices"][layer_index][:, :used_length].float()
        ).abs()
        key_max = max(key_max, float(key_diff.max().item()))
        value_max = max(value_max, float(value_diff.max().item()))
        index_max = max(index_max, float(index_diff.max().item()))
    return {
        "output_max_abs": float(output_diff.max().item()),
        "output_mean_abs": float(output_diff.mean().item()),
        "key_cache_max_abs": key_max,
        "value_cache_max_abs": value_max,
        "index_cache_max_abs": index_max,
        "output_allclose_atol_5e_2_rtol_5e_2": bool(
            torch.allclose(
                output.cpu(),
                reference["output"],
                atol=5e-2,
                rtol=5e-2,
            )
        ),
    }


def main() -> None:
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size not in (1, 2):
        raise ValueError(f"Expected TP1 or TP2, got world_size={world_size}")
    if min(args.warmup_steps, args.decode_steps, args.validation_steps) < 1:
        raise ValueError("warmup, decode, and validation steps must be positive")
    required = max(
        args.validation_steps, args.warmup_steps + args.decode_steps
    )
    if required > args.cache_length:
        raise ValueError("Requested positions exceed the static cache")

    torch.npu.set_device(local_rank)
    torch.npu.set_compile_mode(jit_compile=False)
    if world_size > 1:
        dist.init_process_group("hccl")
        torchair.patch_for_hcom()
    device = torch.device(f"npu:{local_rank}")

    load_started = time.perf_counter()
    stack = GLM52DenseTPStack.from_checkpoint(
        args.model_dir,
        rank=rank,
        world_size=world_size,
        cache_length=args.cache_length,
        device=device,
        progress=lambda message: log(rank, message),
    )
    stack.eval()
    torch.npu.synchronize()
    load_sec = time.perf_counter() - load_started
    weights_memory = memory_snapshot(device)
    log(rank, "weights loaded " + json.dumps(weights_memory, sort_keys=True))

    generator = torch.Generator(device="cpu").manual_seed(52)
    hidden_states = torch.randn(
        1,
        1,
        stack.config.hidden_size,
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=torch.bfloat16)
    eager_decode = stack.forward_decode
    if args.backend == "torchair":
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        decode = torch.compile(
            eager_decode,
            backend=torchair.get_npu_backend(
                compiler_config=CompilerConfig()
            ),
            dynamic=False,
            fullgraph=True,
        )
    else:
        decode = eager_decode

    with torch.inference_mode():
        validation_caches = stack.make_cache(device=device)
        validation_output = run_steps(
            decode,
            hidden_states,
            validation_caches,
            first_position=0,
            steps=args.validation_steps,
        )
        torch.npu.synchronize()
    if not bool(torch.isfinite(validation_output).all().item()):
        raise RuntimeError("TP validation output is not finite")
    reference_comparison = None
    if args.reference_in is not None:
        reference_comparison = compare_reference(
            args.reference_in,
            validation_output,
            validation_caches,
            rank=rank,
            world_size=world_size,
            total_heads=stack.config.num_attention_heads,
        )
        if not reference_comparison["output_allclose_atol_5e_2_rtol_5e_2"]:
            raise RuntimeError(
                "TP output failed TP1 reference parity: "
                + json.dumps(reference_comparison, sort_keys=True)
            )
    if args.reference_out is not None and rank == 0:
        if world_size != 1:
            raise ValueError("Only TP1 may write the full reference")
        save_reference(
            args.reference_out,
            validation_output,
            validation_caches,
            used_length=args.validation_steps,
        )

    with torch.inference_mode():
        benchmark_caches = stack.make_cache(device=device)
        _, warmup_sec_local = timed_steps(
            decode,
            hidden_states,
            benchmark_caches,
            first_position=0,
            steps=args.warmup_steps,
            world_size=world_size,
            device=device,
        )
        dynamo_after_warmup = None
        if args.backend == "torchair":
            dynamo_after_warmup = {
                "unique_graphs": int(
                    torch._dynamo.utils.counters["stats"]["unique_graphs"]
                ),
                "calls_captured": int(
                    torch._dynamo.utils.counters["stats"]["calls_captured"]
                ),
            }
        final_output, elapsed_sec = timed_steps(
            decode,
            hidden_states,
            benchmark_caches,
            first_position=args.warmup_steps,
            steps=args.decode_steps,
            world_size=world_size,
            device=device,
        )
        dynamo_after_measurement = None
        if args.backend == "torchair":
            dynamo_after_measurement = {
                "unique_graphs": int(
                    torch._dynamo.utils.counters["stats"]["unique_graphs"]
                ),
                "calls_captured": int(
                    torch._dynamo.utils.counters["stats"]["calls_captured"]
                ),
            }
            if (
                dynamo_after_measurement["unique_graphs"]
                != dynamo_after_warmup["unique_graphs"]
            ):
                raise RuntimeError("TorchAir captured a graph during measurement")

    summary = {
        "model": "Eco-Tech/GLM-5.2-w4a8c8",
        "chip": "Ascend 910B2",
        "layers": [0, 1, 2],
        "tensor_parallel_size": world_size,
        "backend": args.backend,
        "batch_size": 1,
        "cache_length": args.cache_length,
        "validation_steps": args.validation_steps,
        "warmup_steps": args.warmup_steps,
        "warmup_elapsed_sec_excluded": warmup_sec_local,
        "decode_steps": args.decode_steps,
        "elapsed_sec": elapsed_sec,
        "mean_stack_ms": 1000.0 * elapsed_sec / args.decode_steps,
        "stack_calls_per_sec": args.decode_steps / elapsed_sec,
        "effective_layer_calls_per_sec": 3 * args.decode_steps / elapsed_sec,
        "load_sec": load_sec,
        "memory": memory_snapshot(device),
        "reference_comparison": reference_comparison,
        "dynamo": {
            "after_warmup": dynamo_after_warmup,
            "after_measurement": dynamo_after_measurement,
        }
        if args.backend == "torchair"
        else None,
        "contracts": {
            "replicated_qkv_a_and_indexer": True,
            "attention_heads_per_rank": stack.local_heads,
            "q_b_and_kv_b": "column_parallel_by_head",
            "kv_cache": "head_sharded",
            "o_proj": "row_parallel_all_reduce",
            "dense_gate_up": "column_parallel_intermediate",
            "dense_down": "row_parallel_all_reduce",
            "all_reduces_per_layer": 2 if world_size > 1 else 0,
        },
        "final_output_abs_max": float(final_output.float().abs().max().item()),
    }
    if args.profile_dir is not None:
        if args.backend != "torchair":
            raise ValueError("Profiling is supported only for TorchAir")
        with torch.inference_mode():
            profile_caches = stack.make_cache(device=device)
            summary["profile"] = profile_compiled_steps(
                decode,
                hidden_states,
                profile_caches,
                profile_root=args.profile_dir,
                rank=rank,
                world_size=world_size,
                device=device,
                first_position=0,
            )
    if rank == 0:
        result = {
            "rank0": summary,
            "timing_uses_slowest_rank": True,
        }
        if args.summary_out is not None:
            args.summary_out.parent.mkdir(parents=True, exist_ok=True)
            args.summary_out.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n"
            )
        print("GLM52_DENSE_TP_SUMMARY " + json.dumps(result, sort_keys=True), flush=True)
    barrier(world_size)
    if world_size > 1:
        dist.destroy_process_group()


def timed_steps(
    decode,
    hidden_states: torch.Tensor,
    caches,
    *,
    first_position: int,
    steps: int,
    world_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    barrier(world_size)
    torch.npu.synchronize()
    started = time.perf_counter()
    output = run_steps(
        decode,
        hidden_states,
        caches,
        first_position=first_position,
        steps=steps,
    )
    torch.npu.synchronize()
    elapsed = time.perf_counter() - started
    return output, reduce_max_seconds(
        elapsed, world_size=world_size, device=device
    )


def profile_compiled_steps(
    decode,
    hidden_states: torch.Tensor,
    caches,
    *,
    profile_root: Path,
    rank: int,
    world_size: int,
    device: torch.device,
    first_position: int,
) -> dict[str, object]:
    import torch_npu.profiler as npu_prof

    profile_dir = profile_root / f"rank{rank}"
    profile_dir.mkdir(parents=True, exist_ok=False)
    schedule = npu_prof.schedule(wait=0, warmup=1, active=1, repeat=1)
    experimental_config = npu_prof._ExperimentalConfig(
        profiler_level=npu_prof.ProfilerLevel.Level1,
        aic_metrics=npu_prof.AiCMetrics.PipeUtilization,
        export_type=npu_prof.ExportType.Text,
    )
    synchronized_seconds = []
    barrier(world_size)
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
            position = torch.tensor(
                [first_position + offset], dtype=torch.int64, device=device
            )
            torch.npu.synchronize()
            started = time.perf_counter()
            decode(hidden_states.clone(), position, *caches)
            torch.npu.synchronize()
            synchronized_seconds.append(time.perf_counter() - started)
            prof.step()
    torch.npu.synchronize()
    barrier(world_size)
    return {
        "profile_dir": str(profile_dir),
        "synchronized_seconds": synchronized_seconds,
    }


if __name__ == "__main__":
    main()
