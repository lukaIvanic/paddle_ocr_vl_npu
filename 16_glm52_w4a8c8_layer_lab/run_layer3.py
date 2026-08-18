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

from modeling_glm52_layer import GLM52Layer3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Owned eager and static-compiled GLM-5.2 W4A8C8 layer-3 smoke."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--cache-length", type=int, default=4096)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--decode-steps", type=int, default=20)
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


def benchmark(
    decode,
    hidden_states: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    warmup_steps: int,
    decode_steps: int,
    capture_dynamo_stats: bool = False,
) -> tuple[dict[str, object], torch.Tensor]:
    torch.npu.synchronize()
    warmup_started = time.perf_counter()
    for step in range(warmup_steps):
        cache_position = torch.tensor(
            [step], dtype=torch.int64, device=hidden_states.device
        )
        decode(hidden_states.clone(), cache_position, key_cache, value_cache)
    torch.npu.synchronize()
    warmup_elapsed = time.perf_counter() - warmup_started
    dynamo_after_warmup = None
    if capture_dynamo_stats:
        dynamo_after_warmup = {
            "unique_graphs": int(
                torch._dynamo.utils.counters["stats"]["unique_graphs"]
            ),
            "calls_captured": int(
                torch._dynamo.utils.counters["stats"]["calls_captured"]
            ),
        }

    started = time.perf_counter()
    output = None
    for step in range(decode_steps):
        cache_position = torch.tensor(
            [warmup_steps + step],
            dtype=torch.int64,
            device=hidden_states.device,
        )
        output = decode(
            hidden_states.clone(), cache_position, key_cache, value_cache
        )
    torch.npu.synchronize()
    elapsed = time.perf_counter() - started
    if output is None:
        raise ValueError("decode_steps must be positive")
    dynamo_summary = None
    if capture_dynamo_stats:
        dynamo_after_measurement = {
            "unique_graphs": int(
                torch._dynamo.utils.counters["stats"]["unique_graphs"]
            ),
            "calls_captured": int(
                torch._dynamo.utils.counters["stats"]["calls_captured"]
            ),
        }
        dynamo_summary = {
            "after_warmup": dynamo_after_warmup,
            "after_measurement": dynamo_after_measurement,
            "new_graphs_during_measurement": (
                dynamo_after_measurement["unique_graphs"]
                - dynamo_after_warmup["unique_graphs"]
            ),
        }
    summary = {
        "warmup_steps": warmup_steps,
        "warmup_elapsed_sec_excluded": warmup_elapsed,
        "decode_steps": decode_steps,
        "elapsed_sec": elapsed,
        "mean_layer_ms": 1000.0 * elapsed / decode_steps,
        "layer_calls_per_sec": decode_steps / elapsed,
        "fresh_input_clone_in_timing": True,
    }
    if dynamo_summary is not None:
        summary["dynamo"] = dynamo_summary
    return (
        summary,
        output,
    )


def main() -> None:
    args = parse_args()
    if args.warmup_steps < 1:
        raise ValueError("At least one normal warmup step is required")
    if args.decode_steps < 1:
        raise ValueError("At least one measured decode step is required")
    if args.cache_length < args.warmup_steps + args.decode_steps + 1:
        raise ValueError("Static cache is too short for the requested run")
    torch.npu.set_compile_mode(jit_compile=False)
    device = torch.device(args.device)
    torch.npu.set_device(device)

    load_started = time.perf_counter()
    model = GLM52Layer3.from_checkpoint(
        args.model_dir,
        layer_index=3,
        cache_length=args.cache_length,
        device=device,
        progress=lambda message: print("[layer3] " + message, flush=True),
    )
    model.eval()
    torch.npu.synchronize()
    load_sec = time.perf_counter() - load_started
    print("[layer3] loaded " + json.dumps(memory_snapshot(device), sort_keys=True), flush=True)

    with torch.inference_mode():
        generator = torch.Generator(device="cpu").manual_seed(52)
        hidden_states = torch.randn(
            1,
            1,
            model.config.hidden_size,
            generator=generator,
            dtype=torch.float32,
        ).to(device=device, dtype=torch.bfloat16)
        eager_key, eager_value = model.make_cache(device=device)
        eager_summary, eager_output = benchmark(
            model.forward_decode,
            hidden_states,
            eager_key,
            eager_value,
            warmup_steps=args.warmup_steps,
            decode_steps=args.decode_steps,
        )
    eager_finite = bool(torch.isfinite(eager_output).all().item())
    eager_summary.update({
        "finite": eager_finite,
        "output_mean": float(eager_output.float().mean().item()),
        "output_std": float(eager_output.float().std().item()),
        "output_abs_max": float(eager_output.float().abs().max().item()),
    })
    if not eager_finite:
        raise RuntimeError("Owned eager layer-3 output is not finite")

    compiled_summary = None
    parity = None
    if not args.eager_only:
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        compiled = torch.compile(
            model.forward_decode,
            backend=torchair.get_npu_backend(
                compiler_config=CompilerConfig()
            ),
            dynamic=False,
            fullgraph=True,
        )
        with torch.inference_mode():
            compiled_key, compiled_value = model.make_cache(device=device)
            compiled_hidden_before = hidden_states.clone()
            compiled_summary, compiled_output = benchmark(
                compiled,
                hidden_states,
                compiled_key,
                compiled_value,
                warmup_steps=args.warmup_steps,
                decode_steps=args.decode_steps,
                capture_dynamo_stats=True,
            )
            compiled_summary["hidden_input_max_abs_after"] = float(
                (hidden_states.float() - compiled_hidden_before.float())
                .abs()
                .max()
                .item()
            )
        if compiled_summary["dynamo"]["new_graphs_during_measurement"] != 0:
            raise RuntimeError(
                "TorchAir recompiled inside the measured window: "
                + json.dumps(compiled_summary["dynamo"], sort_keys=True)
            )
        output_diff = (compiled_output.float() - eager_output.float()).abs()
        used_cache_length = args.warmup_steps + args.decode_steps
        key_diff = (
            compiled_key[:, :, :used_cache_length].float()
            - eager_key[:, :, :used_cache_length].float()
        ).abs()
        value_diff = (
            compiled_value[:, :, :used_cache_length].float()
            - eager_value[:, :, :used_cache_length].float()
        ).abs()
        parity = {
            "output_max_abs": float(output_diff.max().item()),
            "output_mean_abs": float(output_diff.mean().item()),
            "key_row_max_abs": float(key_diff.max().item()),
            "value_row_max_abs": float(value_diff.max().item()),
            "allclose_atol_5e_2_rtol_5e_2": bool(
                torch.allclose(
                    compiled_output,
                    eager_output,
                    atol=5e-2,
                    rtol=5e-2,
                )
            ),
        }
        parity["cache_row_abs_max"] = {
            "eager_key": eager_key[:, :, :used_cache_length]
            .float()
            .abs()
            .amax(dim=(0, 1, 3))
            .cpu()
            .tolist(),
            "compiled_key": compiled_key[:, :, :used_cache_length]
            .float()
            .abs()
            .amax(dim=(0, 1, 3))
            .cpu()
            .tolist(),
            "eager_value": eager_value[:, :, :used_cache_length]
            .float()
            .abs()
            .amax(dim=(0, 1, 3))
            .cpu()
            .tolist(),
            "compiled_value": compiled_value[:, :, :used_cache_length]
            .float()
            .abs()
            .amax(dim=(0, 1, 3))
            .cpu()
            .tolist(),
        }
        compiled_summary["measured_window_graph_stable"] = True
        print("[layer3] parity " + json.dumps(parity, sort_keys=True), flush=True)
        if not parity["allclose_atol_5e_2_rtol_5e_2"]:
            raise RuntimeError("Compiled layer-3 output failed eager parity")

    summary = {
        "model": "Eco-Tech/GLM-5.2-w4a8c8",
        "chip": "Ascend 910B2",
        "layer_index": 3,
        "dtype": "bfloat16",
        "batch_size": 1,
        "cache_length": args.cache_length,
        "architecture": {
            "owned_modeling": True,
            "vllm_runtime_imported": False,
            "attention": "dense_unabsorbed_mla_increfa",
            "dsa_topk_indexer": "skipped_by_layer_pattern",
            "attention_linears": "modelslim_w8a8_dynamic",
            "shared_expert": "modelslim_w8a8_dynamic",
            "routed_experts": "modelslim_w4a8_dynamic_public_npu_ops",
            "num_experts": model.config.num_experts,
            "top_k": model.config.top_k,
        },
        "load_sec": load_sec,
        "memory": memory_snapshot(device),
        "eager": eager_summary,
        "compiled": compiled_summary,
        "parity": parity,
    }
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("GLM52_LAYER3_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
