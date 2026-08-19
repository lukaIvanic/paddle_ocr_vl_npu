#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch_npu
from torch_npu.dynamo import torchair
from torch_npu.dynamo.torchair.configs.compiler_config import CompilerConfig

from modeling_glm52_layer import configure_grouped_matmul_scale_conversion
from modeling_glm52_stack import GLM52LayerStack


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Owned GLM-5.2 W4A8C8 layers 0-6 on one Ascend NPU."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--cache-length", type=int, default=4096)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--eager-only", action="store_true")
    parser.add_argument("--summary-out", type=Path)
    return parser.parse_args()


def memory_snapshot(device: torch.device) -> dict[str, float | str]:
    free_bytes, total_bytes = torch.npu.mem_get_info(device)
    return {
        "device": str(device),
        "allocated_gib": torch.npu.memory_allocated(device) / 2**30,
        "reserved_gib": torch.npu.memory_reserved(device) / 2**30,
        "free_gib": free_bytes / 2**30,
        "total_gib": total_bytes / 2**30,
    }


def run_loop(
    decode,
    hidden_states: torch.Tensor,
    caches,
    *,
    first_position: int,
    steps: int,
) -> tuple[torch.Tensor, float]:
    torch.npu.synchronize()
    started = time.perf_counter()
    output = None
    for offset in range(steps):
        position = torch.tensor(
            [first_position + offset],
            dtype=torch.int64,
            device=hidden_states.device,
        )
        output = decode(hidden_states.clone(), position, *caches)
    torch.npu.synchronize()
    if output is None:
        raise ValueError("steps must be positive")
    return output, time.perf_counter() - started


def summarize_timing(elapsed: float, steps: int) -> dict[str, float | int]:
    return {
        "steps": steps,
        "elapsed_sec": elapsed,
        "mean_stack_ms": 1000.0 * elapsed / steps,
        "stack_calls_per_sec": steps / elapsed,
        "effective_layer_calls_per_sec": 7 * steps / elapsed,
        "fresh_stack_input_clone_in_timing": True,
    }


def max_cache_diff(left, right, used_length: int) -> float:
    maximum = 0.0
    for left_cache, right_cache in zip(left, right):
        if left_cache.shape[1] == 1 and left_cache.shape[-1] == 1:
            continue
        if left_cache.dim() == 4:
            left_used = left_cache[:, :, :used_length]
            right_used = right_cache[:, :, :used_length]
        else:
            left_used = left_cache[:, :used_length]
            right_used = right_cache[:, :used_length]
        maximum = max(
            maximum,
            float((left_used.float() - right_used.float()).abs().max().item()),
        )
    return maximum


def main() -> None:
    args = parse_args()
    if args.warmup_steps < 1 or args.decode_steps < 1:
        raise ValueError("warmup and measured steps must both be positive")
    used_length = args.warmup_steps + args.decode_steps
    if used_length > args.cache_length:
        raise ValueError("warmup plus measured steps exceed the static cache")

    torch.npu.set_compile_mode(jit_compile=False)
    device = torch.device(args.device)
    torch.npu.set_device(device)

    def progress(message: str) -> None:
        record = {"message": message}
        if message.startswith("loaded layer"):
            record["memory"] = memory_snapshot(device)
        print("[layers0-6] " + json.dumps(record, sort_keys=True), flush=True)

    load_started = time.perf_counter()
    stack = GLM52LayerStack.from_checkpoint(
        args.model_dir,
        first_layer=0,
        last_layer=6,
        cache_length=args.cache_length,
        device=device,
        progress=progress,
    )
    stack.eval()
    torch.npu.synchronize()
    load_sec = time.perf_counter() - load_started
    weights_memory = memory_snapshot(device)
    print(
        "[layers0-6] loaded " + json.dumps(weights_memory, sort_keys=True),
        flush=True,
    )

    with torch.inference_mode():
        generator = torch.Generator(device="cpu").manual_seed(52)
        hidden_states = torch.randn(
            1,
            1,
            stack.config.hidden_size,
            generator=generator,
            dtype=torch.float32,
        ).to(device=device, dtype=torch.bfloat16)
        eager_caches = stack.make_cache(device=device)
        caches_memory = memory_snapshot(device)
        _, eager_warmup_sec = run_loop(
            stack.forward_decode,
            hidden_states,
            eager_caches,
            first_position=0,
            steps=args.warmup_steps,
        )
        eager_output, eager_elapsed = run_loop(
            stack.forward_decode,
            hidden_states,
            eager_caches,
            first_position=args.warmup_steps,
            steps=args.decode_steps,
        )
    eager_finite = bool(torch.isfinite(eager_output).all().item())
    if not eager_finite:
        raise RuntimeError("Layers 0-6 eager output is not finite")
    eager_summary = summarize_timing(eager_elapsed, args.decode_steps)
    eager_summary.update(
        {
            "warmup_steps": args.warmup_steps,
            "warmup_elapsed_sec_excluded": eager_warmup_sec,
            "finite": eager_finite,
            "output_mean": float(eager_output.float().mean().item()),
            "output_std": float(eager_output.float().std().item()),
            "output_abs_max": float(eager_output.float().abs().max().item()),
        }
    )

    compiled_summary = None
    parity = None
    if not args.eager_only:
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        configure_grouped_matmul_scale_conversion()
        compiled = torch.compile(
            stack.forward_decode,
            backend=torchair.get_npu_backend(
                compiler_config=CompilerConfig()
            ),
            dynamic=False,
            fullgraph=True,
        )
        with torch.inference_mode():
            compiled_caches = stack.make_cache(device=device)
            _, compiled_warmup_sec = run_loop(
                compiled,
                hidden_states,
                compiled_caches,
                first_position=0,
                steps=args.warmup_steps,
            )
            stats_after_warmup = {
                "unique_graphs": int(
                    torch._dynamo.utils.counters["stats"]["unique_graphs"]
                ),
                "calls_captured": int(
                    torch._dynamo.utils.counters["stats"]["calls_captured"]
                ),
            }
            compiled_output, compiled_elapsed = run_loop(
                compiled,
                hidden_states,
                compiled_caches,
                first_position=args.warmup_steps,
                steps=args.decode_steps,
            )
            stats_after_measurement = {
                "unique_graphs": int(
                    torch._dynamo.utils.counters["stats"]["unique_graphs"]
                ),
                "calls_captured": int(
                    torch._dynamo.utils.counters["stats"]["calls_captured"]
                ),
            }
        new_graphs = (
            stats_after_measurement["unique_graphs"]
            - stats_after_warmup["unique_graphs"]
        )
        if new_graphs:
            raise RuntimeError("TorchAir captured a graph inside measurement")
        output_diff = (compiled_output.float() - eager_output.float()).abs()
        parity = {
            "output_max_abs": float(output_diff.max().item()),
            "output_mean_abs": float(output_diff.mean().item()),
            "key_cache_max_abs": max_cache_diff(
                compiled_caches[0], eager_caches[0], used_length
            ),
            "value_cache_max_abs": max_cache_diff(
                compiled_caches[1], eager_caches[1], used_length
            ),
            "index_cache_max_abs": max_cache_diff(
                compiled_caches[2], eager_caches[2], used_length
            ),
            "allclose_atol_5e_2_rtol_5e_2": bool(
                torch.allclose(
                    compiled_output, eager_output, atol=5e-2, rtol=5e-2
                )
            ),
        }
        if not parity["allclose_atol_5e_2_rtol_5e_2"]:
            raise RuntimeError(
                "Compiled layers 0-6 failed eager output parity: "
                + json.dumps(parity, sort_keys=True)
            )
        compiled_summary = summarize_timing(
            compiled_elapsed, args.decode_steps
        )
        compiled_summary.update(
            {
                "warmup_steps": args.warmup_steps,
                "warmup_elapsed_sec_excluded": compiled_warmup_sec,
                "dynamo": {
                    "after_warmup": stats_after_warmup,
                    "after_measurement": stats_after_measurement,
                    "new_graphs_during_measurement": new_graphs,
                },
                "memory_after_compile": memory_snapshot(device),
            }
        )

    summary = {
        "model": "Eco-Tech/GLM-5.2-w4a8c8",
        "chip": "Ascend 910B2",
        "layers": [0, 1, 2, 3, 4, 5, 6],
        "layer_types": {
            "0-2": "dense_mlp_full_dsa_indexer",
            "3-5": "w4a8_moe_shared_dsa_indices",
            "6": "w4a8_moe_full_dsa_indexer",
        },
        "attention": "owned_dsa_topk_manual_sparse_attention",
        "index_topk": stack.top_k,
        "measured_positions_with_dsa_pruning": max(
            0,
            used_length - max(args.warmup_steps, stack.top_k),
        ),
        "cache_length": args.cache_length,
        "dtype": "bfloat16",
        "load_sec": load_sec,
        "memory_after_weights": weights_memory,
        "memory_after_eager_caches": caches_memory,
        "eager": eager_summary,
        "compiled": compiled_summary,
        "parity": parity,
    }
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
    print("GLM52_LAYERS0_6_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
