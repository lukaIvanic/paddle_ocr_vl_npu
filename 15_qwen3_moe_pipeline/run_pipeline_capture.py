#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch_npu
from transformers import AutoTokenizer

from modeling_qwen3_moe_pipeline import Qwen3MoeConfig
from runtime import build_stage, memory_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Qwen3-30B-A3B as a two-NPU sequential pipeline and capture "
            "a replayable layer-24 boundary package."
        )
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--cache-length", type=int, default=256)
    parser.add_argument("--stage0-device", default="npu:0")
    parser.add_argument("--stage1-device", default="npu:1")
    parser.add_argument("--split-layer", type=int, default=24)
    parser.add_argument("--reference-json")
    parser.add_argument("--capture-out", required=True)
    parser.add_argument("--summary-out")
    return parser.parse_args()


def cpu_tuple(values: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    return tuple(value.detach().cpu() for value in values)


def main() -> None:
    args = parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("max-new-tokens must be positive")
    torch.npu.set_compile_mode(jit_compile=False)
    stage0_device = torch.device(args.stage0_device)
    stage1_device = torch.device(args.stage1_device)
    torch.npu.set_device(stage0_device)

    config = Qwen3MoeConfig.from_model_dir(args.model_dir)
    config.validate_qwen3_30b_a3b()
    if args.split_layer != 24:
        raise ValueError("The first parity package is fixed to split-layer=24")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    encoded = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=True)
    prompt_token_ids = encoded["input_ids"][0].tolist()
    if len(prompt_token_ids) < 2:
        raise ValueError("The capture prompt must contain at least two tokens")
    if len(prompt_token_ids) + args.max_new_tokens > args.cache_length:
        raise ValueError("Prompt plus generated tokens exceeds cache-length")

    reference = None
    if args.reference_json:
        reference = json.loads(Path(args.reference_json).read_text())
        if reference["prompt_token_ids"] != prompt_token_ids:
            raise RuntimeError("Reference prompt token IDs do not match tokenizer output")

    stage0, stage0_metadata = build_stage(
        config,
        args.model_dir,
        layer_start=0,
        layer_end=args.split_layer,
        with_embedding=True,
        with_lm_head=False,
        device=stage0_device,
        name="stage0",
        cache_length=args.cache_length,
    )
    stage1, stage1_metadata = build_stage(
        config,
        args.model_dir,
        layer_start=args.split_layer,
        layer_end=config.num_hidden_layers,
        with_embedding=False,
        with_lm_head=True,
        device=stage1_device,
        name="stage1",
        cache_length=args.cache_length,
    )
    stage0_cache = stage0.make_cache(cache_length=args.cache_length)
    stage1_cache = stage1.make_cache(cache_length=args.cache_length)

    prefix_length = len(prompt_token_ids) - 1
    with torch.inference_mode():
        for position, token_id in enumerate(prompt_token_ids[:-1]):
            input_id = torch.tensor(
                [[token_id]], dtype=torch.int64, device=stage0_device
            )
            cache_position = torch.tensor(
                [position], dtype=torch.int64, device=stage0_device
            )
            boundary, _indices, _weights = stage0.decode_input_ids(
                input_id, cache_position, stage0_cache
            )
            boundary_stage1 = boundary.to(stage1_device)
            stage1_position = cache_position.to(stage1_device)
            stage1.decode_hidden_states(
                boundary_stage1, stage1_position, stage1_cache
            )

        stage1_prefix_cache = stage1_cache.snapshot_prefix(prefix_length)
        generated_token_ids = []
        input_token_ids = []
        cache_positions = []
        boundary_hidden_states = []
        expected_logits = []
        expected_topk_ids = []
        expected_topk_values = []
        stage1_router_indices = []
        stage1_router_weights = []
        step_times = []

        current_input_id = prompt_token_ids[-1]
        for step in range(args.max_new_tokens):
            position = prefix_length + step
            input_token_ids.append(current_input_id)
            cache_positions.append(position)
            input_id = torch.tensor(
                [[current_input_id]], dtype=torch.int64, device=stage0_device
            )
            cache_position = torch.tensor(
                [position], dtype=torch.int64, device=stage0_device
            )
            torch.npu.synchronize(stage0_device)
            torch.npu.synchronize(stage1_device)
            started = time.perf_counter()
            boundary, _stage0_indices, _stage0_weights = stage0.decode_input_ids(
                input_id,
                cache_position,
                stage0_cache,
            )
            boundary_stage1 = boundary.to(stage1_device)
            stage1_position = cache_position.to(stage1_device)
            final_hidden, router_indices, router_weights = (
                stage1.decode_hidden_states(
                    boundary_stage1,
                    stage1_position,
                    stage1_cache,
                    capture_router=True,
                )
            )
            logits = stage1.logits(final_hidden)[:, -1, :]
            next_token = logits.argmax(dim=-1)
            torch.npu.synchronize(stage1_device)
            step_times.append(time.perf_counter() - started)

            topk_values, topk_ids = torch.topk(logits.float(), 10, dim=-1)
            next_token_id = int(next_token.item())
            generated_token_ids.append(next_token_id)
            boundary_hidden_states.append(boundary.detach().cpu())
            expected_logits.append(logits.detach().cpu())
            expected_topk_ids.append(topk_ids.detach().cpu())
            expected_topk_values.append(topk_values.detach().cpu())
            stage1_router_indices.append(cpu_tuple(router_indices))
            stage1_router_weights.append(cpu_tuple(router_weights))
            current_input_id = next_token_id

    token_match = None
    if reference is not None:
        expected_tokens = reference["generated_token_ids"][: args.max_new_tokens]
        token_match = generated_token_ids == expected_tokens
        if not token_match:
            raise RuntimeError(
                "Pipeline token mismatch versus reference: "
                f"pipeline={generated_token_ids} reference={expected_tokens}"
            )

    capture = {
        "format": "qwen3_30b_a3b_stage2_replay_v1",
        "model": "Qwen/Qwen3-30B-A3B",
        "dtype": "bfloat16",
        "split_layer": args.split_layer,
        "stage2_layer_start": args.split_layer,
        "stage2_layer_end": config.num_hidden_layers,
        "cache_length": args.cache_length,
        "prompt": args.prompt,
        "prompt_token_ids": prompt_token_ids,
        "prefix_length": prefix_length,
        "input_token_ids": input_token_ids,
        "cache_positions": cache_positions,
        "generated_token_ids": generated_token_ids,
        "boundary_hidden_states": tuple(boundary_hidden_states),
        "stage2_prefix_cache": stage1_prefix_cache,
        "expected_logits": tuple(expected_logits),
        "expected_topk_ids": tuple(expected_topk_ids),
        "expected_topk_values": tuple(expected_topk_values),
        "stage2_router_indices": tuple(stage1_router_indices),
        "stage2_router_weights": tuple(stage1_router_weights),
    }
    capture_path = Path(args.capture_out)
    capture_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(capture, capture_path)

    summary = {
        "model": capture["model"],
        "prompt": args.prompt,
        "prompt_token_ids": prompt_token_ids,
        "generated_token_ids": generated_token_ids,
        "generated_text": tokenizer.decode(
            generated_token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
        "reference_token_match": token_match,
        "split_layer": args.split_layer,
        "stage0": stage0_metadata,
        "stage1": stage1_metadata,
        "step_times_sec": step_times,
        "mean_step_sec": sum(step_times) / len(step_times),
        "capture_path": str(capture_path),
        "capture_size_bytes": capture_path.stat().st_size,
        "final_memory": {
            "stage0": memory_snapshot(stage0_device),
            "stage1": memory_snapshot(stage1_device),
        },
    }
    print("QWEN3_MOE_PIPELINE_CAPTURE " + json.dumps(summary, sort_keys=True))
    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
