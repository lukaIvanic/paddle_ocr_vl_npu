#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time

import torch
import torch_npu

from modeling_qwen3_moe_pipeline import Qwen3MoeConfig
from runtime import build_stage, memory_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-weight Qwen3-30B-A3B stage decode probe."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--cache-length", type=int, default=256)
    parser.add_argument("--device", default="npu:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.npu.set_compile_mode(jit_compile=False)
    device = torch.device(args.device)
    torch.npu.set_device(device)
    config = Qwen3MoeConfig.from_model_dir(args.model_dir)
    config.validate_qwen3_30b_a3b()
    if args.num_layers < 1 or args.layer + args.num_layers > config.num_hidden_layers:
        raise ValueError("Requested layer range is outside the model")
    stage, metadata = build_stage(
        config,
        args.model_dir,
        layer_start=args.layer,
        layer_end=args.layer + args.num_layers,
        with_embedding=False,
        with_lm_head=False,
        device=device,
        name=f"layers-{args.layer}-{args.layer + args.num_layers}",
        cache_length=args.cache_length,
    )
    cache = stage.make_cache(cache_length=args.cache_length)
    torch.manual_seed(17)
    hidden_states = torch.randn(
        (1, 1, config.hidden_size), dtype=torch.bfloat16, device=device
    )
    cache_position = torch.tensor([0], dtype=torch.int64, device=device)
    with torch.inference_mode():
        torch.npu.synchronize(device)
        started = time.perf_counter()
        output, router_indices, router_weights = stage.decode_hidden_states(
            hidden_states,
            cache_position,
            cache,
            capture_router=True,
        )
        torch.npu.synchronize(device)
        elapsed = time.perf_counter() - started
    result = {
        "layer": args.layer,
        "num_layers": args.num_layers,
        "elapsed_sec": elapsed,
        "output_shape": list(output.shape),
        "output_finite": bool(torch.isfinite(output).all().item()),
        "output_max_abs": float(output.float().abs().max().item()),
        "selected_experts": router_indices[0].cpu().tolist(),
        "routing_weights": router_weights[0].float().cpu().tolist(),
        "routing_weight_sum": float(router_weights[0].float().sum().item()),
        "stage": metadata,
        "memory": memory_snapshot(device),
    }
    print("QWEN3_MOE_REAL_LAYER_PROBE " + json.dumps(result, sort_keys=True))
    if not result["output_finite"]:
        raise RuntimeError("One-layer output contains non-finite values")


if __name__ == "__main__":
    main()
