#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch_npu

from modeling_qwen3_moe_pipeline import Qwen3MoeConfig
from runtime import build_stage, memory_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay captured layer-24 boundary states through the second half "
            "of Qwen3-30B-A3B on one NPU."
        )
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--summary-out")
    parser.add_argument("--logit-atol", type=float, default=1e-3)
    parser.add_argument("--logit-rtol", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.npu.set_compile_mode(jit_compile=False)
    device = torch.device(args.device)
    torch.npu.set_device(device)
    capture = torch.load(args.capture, map_location="cpu", weights_only=False)
    if capture["format"] != "qwen3_30b_a3b_stage2_replay_v1":
        raise ValueError(f"Unsupported capture format: {capture['format']}")

    config = Qwen3MoeConfig.from_model_dir(args.model_dir)
    config.validate_qwen3_30b_a3b()
    stage2, stage_metadata = build_stage(
        config,
        args.model_dir,
        layer_start=int(capture["stage2_layer_start"]),
        layer_end=int(capture["stage2_layer_end"]),
        with_embedding=False,
        with_lm_head=True,
        device=device,
        name="stage2-replay",
        cache_length=int(capture["cache_length"]),
    )
    cache = stage2.make_cache(cache_length=int(capture["cache_length"]))
    restored_prefix_length = cache.restore_prefix(capture["stage2_prefix_cache"])
    if restored_prefix_length != int(capture["prefix_length"]):
        raise RuntimeError(
            f"Restored prefix {restored_prefix_length} does not match capture "
            f"{capture['prefix_length']}"
        )

    step_results = []
    all_tokens_match = True
    all_topk_match = True
    all_router_indices_match = True
    all_logits_close = True
    with torch.inference_mode():
        for step, (boundary_cpu, position) in enumerate(
            zip(capture["boundary_hidden_states"], capture["cache_positions"])
        ):
            boundary = boundary_cpu.to(device=device, dtype=torch.bfloat16)
            cache_position = torch.tensor([position], dtype=torch.int64, device=device)
            torch.npu.synchronize(device)
            started = time.perf_counter()
            final_hidden, router_indices, _router_weights = (
                stage2.decode_hidden_states(
                    boundary,
                    cache_position,
                    cache,
                    capture_router=True,
                )
            )
            logits = stage2.logits(final_hidden)[:, -1, :]
            next_token = logits.argmax(dim=-1)
            torch.npu.synchronize(device)
            elapsed = time.perf_counter() - started

            expected_logits = capture["expected_logits"][step].to(
                device=device, dtype=logits.dtype
            )
            logit_diff = (logits.float() - expected_logits.float()).abs()
            logits_close = bool(
                torch.allclose(
                    logits,
                    expected_logits,
                    atol=args.logit_atol,
                    rtol=args.logit_rtol,
                )
            )
            topk_ids = torch.topk(logits.float(), 10, dim=-1).indices.cpu()
            topk_match = torch.equal(topk_ids, capture["expected_topk_ids"][step])
            token_id = int(next_token.item())
            expected_token_id = int(capture["generated_token_ids"][step])
            token_match = token_id == expected_token_id
            router_match = all(
                torch.equal(actual.cpu(), expected)
                for actual, expected in zip(
                    router_indices, capture["stage2_router_indices"][step]
                )
            )
            all_tokens_match &= token_match
            all_topk_match &= topk_match
            all_router_indices_match &= router_match
            all_logits_close &= logits_close
            step_results.append(
                {
                    "step": step,
                    "cache_position": int(position),
                    "elapsed_sec": elapsed,
                    "token_id": token_id,
                    "expected_token_id": expected_token_id,
                    "token_match": token_match,
                    "top10_ids_match": topk_match,
                    "router_indices_match": router_match,
                    "logits_close": logits_close,
                    "logit_max_abs": float(logit_diff.max().item()),
                    "logit_mean_abs": float(logit_diff.mean().item()),
                }
            )

    summary = {
        "model": capture["model"],
        "capture": str(Path(args.capture)),
        "stage": stage_metadata,
        "steps": len(step_results),
        "all_tokens_match": all_tokens_match,
        "all_top10_ids_match": all_topk_match,
        "all_router_indices_match": all_router_indices_match,
        "all_logits_close": all_logits_close,
        "mean_step_sec": sum(row["elapsed_sec"] for row in step_results)
        / len(step_results),
        "memory": memory_snapshot(device),
        "step_results": step_results,
    }
    print("QWEN3_MOE_STAGE2_REPLAY " + json.dumps(summary, sort_keys=True))
    if args.summary_out:
        output_path = Path(args.summary_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if not (
        all_tokens_match
        and all_topk_match
        and all_router_indices_match
        and all_logits_close
    ):
        raise RuntimeError("Stage-2 replay parity failed")


if __name__ == "__main__":
    main()
