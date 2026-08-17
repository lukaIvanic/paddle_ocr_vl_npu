#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import torch_npu

from modeling_qwen3_moe_pipeline import Qwen3MoeConfig
from runtime import build_stage, memory_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Static TorchAir two-NPU Qwen3-30B-A3B B1 decode benchmark."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--stage0-device", default="npu:0")
    parser.add_argument("--stage1-device", default="npu:1")
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--decode-steps", type=int, default=20)
    parser.add_argument("--stage-timing-steps", type=int, default=5)
    parser.add_argument(
        "--token-handoff",
        choices=("host", "direct"),
        default="host",
        help="Return the sampled B1 token through a host scalar or direct NPU copy.",
    )
    parser.add_argument(
        "--compile-cache-dir",
        type=Path,
        default=Path(".runtime_cache/15_qwen3_moe_pipeline"),
    )
    parser.add_argument("--summary-out", type=Path)
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


def synchronize(*devices: torch.device) -> None:
    for device in devices:
        torch.npu.synchronize(device)


def dynamo_stats() -> dict[str, int]:
    counters = torch._dynamo.utils.counters
    return {
        "unique_graphs": int(counters["stats"]["unique_graphs"]),
        "calls_captured": int(counters["stats"]["calls_captured"]),
        "frames_total": int(counters["frames"]["total"]),
        "frames_ok": int(counters["frames"]["ok"]),
    }


def compile_static(
    method,
    *,
    cache_dir: Path,
):
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


def pipeline_step(
    stage0_decode,
    stage1_decode,
    input_ids: torch.Tensor,
    position: int,
    stage0_cache,
    stage1_cache,
    *,
    stage0_device: torch.device,
    stage1_device: torch.device,
    token_handoff: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    stage0_position = torch.tensor(
        [position], dtype=torch.int64, device=stage0_device
    )
    boundary = stage0_decode(
        input_ids,
        stage0_position,
        stage0_cache.key_caches,
        stage0_cache.value_caches,
    )
    logits = stage1_decode(
        boundary.to(stage1_device),
        stage0_position.to(stage1_device),
        stage1_cache.key_caches,
        stage1_cache.value_caches,
    )
    next_token_stage1 = logits.argmax(dim=-1, keepdim=True)
    if token_handoff == "host":
        next_token_id = int(next_token_stage1.item())
        next_token_stage0 = torch.tensor(
            [[next_token_id]], dtype=torch.int64, device=stage0_device
        )
    else:
        next_token_stage0 = next_token_stage1.to(stage0_device)
    return next_token_stage0, logits


def main() -> None:
    args = parse_args()
    if (
        args.warmup_steps < 0
        or args.decode_steps < 1
        or args.stage_timing_steps < 0
    ):
        raise ValueError("Invalid warmup/decode step count")
    torch.npu.set_compile_mode(jit_compile=False)
    stage0_device = torch.device(args.stage0_device)
    stage1_device = torch.device(args.stage1_device)
    torch.npu.set_device(stage0_device)
    capture = torch.load(args.capture, map_location="cpu", weights_only=False)
    if capture["format"] != "qwen3_30b_a3b_stage2_replay_v1":
        raise ValueError(f"Unsupported capture format: {capture['format']}")

    config = Qwen3MoeConfig.from_model_dir(args.model_dir)
    config.validate_qwen3_30b_a3b()
    cache_length = int(capture["cache_length"])
    final_position = (
        len(capture["prompt_token_ids"])
        + len(capture["generated_token_ids"])
        + args.warmup_steps
        + args.decode_steps
        + args.stage_timing_steps
        - 1
    )
    if final_position >= cache_length:
        raise ValueError(
            f"Run needs position {final_position}, cache length is {cache_length}"
        )

    stage0, stage0_metadata = build_stage(
        config,
        args.model_dir,
        layer_start=0,
        layer_end=24,
        with_embedding=True,
        with_lm_head=False,
        device=stage0_device,
        name="compiled-pipeline-stage0",
        cache_length=cache_length,
    )
    stage1, stage1_metadata = build_stage(
        config,
        args.model_dir,
        layer_start=24,
        layer_end=48,
        with_embedding=False,
        with_lm_head=True,
        device=stage1_device,
        name="compiled-pipeline-stage1",
        cache_length=cache_length,
    )
    stage0_cache = stage0.make_cache(cache_length=cache_length)
    stage1_cache = stage1.make_cache(cache_length=cache_length)

    cache_root = args.compile_cache_dir.expanduser().resolve()
    code_hash = source_hash()
    stage0_cache_dir = cache_root / (
        f"pipeline_stage0_l24_b1_kv{cache_length}_bf16_src{code_hash}"
    )
    stage1_cache_dir = cache_root / (
        f"pipeline_stage1_l24_b1_kv{cache_length}_bf16_src{code_hash}"
    )
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    stage0_decode, stage0_wrapper_sec, stage0_cache_warm = compile_static(
        stage0.decode_static_input_ids,
        cache_dir=stage0_cache_dir,
    )
    stage1_decode, stage1_wrapper_sec, stage1_cache_warm = compile_static(
        stage1.decode_static_output,
        cache_dir=stage1_cache_dir,
    )

    prompt_token_ids = list(capture["prompt_token_ids"])
    first_input = torch.tensor(
        [[prompt_token_ids[0]]], dtype=torch.int64, device=stage0_device
    )
    synchronize(stage0_device, stage1_device)
    stage0_first_started = time.perf_counter()
    first_position = torch.tensor([0], dtype=torch.int64, device=stage0_device)
    first_boundary = stage0_decode(
        first_input,
        first_position,
        stage0_cache.key_caches,
        stage0_cache.value_caches,
    )
    synchronize(stage0_device)
    stage0_first_call_sec = time.perf_counter() - stage0_first_started
    stage1_first_started = time.perf_counter()
    stage1_decode(
        first_boundary.to(stage1_device),
        first_position.to(stage1_device),
        stage1_cache.key_caches,
        stage1_cache.value_caches,
    )
    synchronize(stage1_device)
    stage1_first_call_sec = time.perf_counter() - stage1_first_started

    for position, token_id in enumerate(prompt_token_ids[1:-1], start=1):
        input_ids = torch.tensor(
            [[token_id]], dtype=torch.int64, device=stage0_device
        )
        pipeline_step(
            stage0_decode,
            stage1_decode,
            input_ids,
            position,
            stage0_cache,
            stage1_cache,
            stage0_device=stage0_device,
            stage1_device=stage1_device,
            token_handoff=args.token_handoff,
        )
    synchronize(stage0_device, stage1_device)
    stats_after_prompt = dynamo_stats()

    current_input = torch.tensor(
        [[prompt_token_ids[-1]]], dtype=torch.int64, device=stage0_device
    )
    generated_token_ids = []
    parity_step_times = []
    position = len(prompt_token_ids) - 1
    for expected_token_id in capture["generated_token_ids"]:
        synchronize(stage0_device, stage1_device)
        started = time.perf_counter()
        current_input, _logits = pipeline_step(
            stage0_decode,
            stage1_decode,
            current_input,
            position,
            stage0_cache,
            stage1_cache,
            stage0_device=stage0_device,
            stage1_device=stage1_device,
            token_handoff=args.token_handoff,
        )
        synchronize(stage0_device, stage1_device)
        parity_step_times.append(time.perf_counter() - started)
        token_id = int(current_input.item())
        generated_token_ids.append(token_id)
        if token_id != int(expected_token_id):
            raise RuntimeError(
                f"Compiled pipeline token mismatch at position {position}: "
                f"got {token_id}, expected {expected_token_id}"
            )
        position += 1

    for _ in range(args.warmup_steps):
        current_input, _logits = pipeline_step(
            stage0_decode,
            stage1_decode,
            current_input,
            position,
            stage0_cache,
            stage1_cache,
            stage0_device=stage0_device,
            stage1_device=stage1_device,
            token_handoff=args.token_handoff,
        )
        position += 1
    synchronize(stage0_device, stage1_device)
    measured_started = time.perf_counter()
    for _ in range(args.decode_steps):
        current_input, _logits = pipeline_step(
            stage0_decode,
            stage1_decode,
            current_input,
            position,
            stage0_cache,
            stage1_cache,
            stage0_device=stage0_device,
            stage1_device=stage1_device,
            token_handoff=args.token_handoff,
        )
        position += 1
    synchronize(stage0_device, stage1_device)
    measured_sec = time.perf_counter() - measured_started

    stage_timing_samples = []
    for _ in range(args.stage_timing_steps):
        stage0_position = torch.tensor(
            [position], dtype=torch.int64, device=stage0_device
        )
        synchronize(stage0_device, stage1_device)
        total_started = time.perf_counter()
        stage_started = time.perf_counter()
        boundary = stage0_decode(
            current_input,
            stage0_position,
            stage0_cache.key_caches,
            stage0_cache.value_caches,
        )
        synchronize(stage0_device)
        stage0_sec = time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        boundary_stage1 = boundary.to(stage1_device)
        stage1_position = stage0_position.to(stage1_device)
        synchronize(stage1_device)
        boundary_copy_sec = time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        logits = stage1_decode(
            boundary_stage1,
            stage1_position,
            stage1_cache.key_caches,
            stage1_cache.value_caches,
        )
        synchronize(stage1_device)
        stage1_sec = time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        next_token_stage1 = logits.argmax(dim=-1, keepdim=True)
        synchronize(stage1_device)
        argmax_sec = time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        if args.token_handoff == "host":
            next_token_id = int(next_token_stage1.item())
            current_input = torch.tensor(
                [[next_token_id]], dtype=torch.int64, device=stage0_device
            )
        else:
            current_input = next_token_stage1.to(stage0_device)
        synchronize(stage0_device)
        token_copy_sec = time.perf_counter() - stage_started
        total_sec = time.perf_counter() - total_started
        stage_timing_samples.append(
            {
                "stage0_ms": 1000.0 * stage0_sec,
                "boundary_copy_ms": 1000.0 * boundary_copy_sec,
                "stage1_ms": 1000.0 * stage1_sec,
                "argmax_ms": 1000.0 * argmax_sec,
                "token_copy_ms": 1000.0 * token_copy_sec,
                "total_ms": 1000.0 * total_sec,
            }
        )
        position += 1

    stage_timing_mean_ms = {
        key: sum(sample[key] for sample in stage_timing_samples)
        / len(stage_timing_samples)
        for key in stage_timing_samples[0]
    } if stage_timing_samples else {}
    stats_final = dynamo_stats()
    unique_graph_delta = (
        stats_final["unique_graphs"] - stats_after_prompt["unique_graphs"]
    )
    summary = {
        "model": capture["model"],
        "chip": "Ascend 910B2",
        "dtype": "bfloat16",
        "batch_size": 1,
        "pipeline_parallel_size": 2,
        "token_handoff": args.token_handoff,
        "cache_length": cache_length,
        "compile_contract": {
            "fullgraph": True,
            "dynamic": False,
            "ge_cache": True,
            "stage0_cache_dir": str(stage0_cache_dir),
            "stage1_cache_dir": str(stage1_cache_dir),
            "stage0_cache_was_warm": stage0_cache_warm,
            "stage1_cache_was_warm": stage1_cache_warm,
            "stage0_wrapper_sec": stage0_wrapper_sec,
            "stage1_wrapper_sec": stage1_wrapper_sec,
            "stage0_first_call_sec": stage0_first_call_sec,
            "stage1_first_call_sec": stage1_first_call_sec,
            "dynamo_after_prompt": stats_after_prompt,
            "dynamo_final": stats_final,
            "unique_graph_delta_after_prompt": unique_graph_delta,
            "no_recompilations_after_prompt": unique_graph_delta == 0,
        },
        "parity": {
            "passed": generated_token_ids == capture["generated_token_ids"],
            "generated_token_ids": generated_token_ids,
            "expected_token_ids": capture["generated_token_ids"],
            "step_times_sec": parity_step_times,
        },
        "benchmark": {
            "warmup_steps": args.warmup_steps,
            "decode_steps": args.decode_steps,
            "elapsed_sec": measured_sec,
            "mean_tpot_ms": 1000.0 * measured_sec / args.decode_steps,
            "tok_s": args.decode_steps / measured_sec,
        },
        "stage_timing_diagnostic": {
            "steps": args.stage_timing_steps,
            "mean_ms": stage_timing_mean_ms,
            "samples": stage_timing_samples,
        },
        "stage0": stage0_metadata,
        "stage1": stage1_metadata,
        "memory": {
            "stage0": memory_snapshot(stage0_device),
            "stage1": memory_snapshot(stage1_device),
        },
    }
    print("QWEN3_MOE_PIPELINE_COMPILE " + json.dumps(summary, sort_keys=True))
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if unique_graph_delta != 0:
        raise RuntimeError("Additional static graphs were compiled after prompt setup")


if __name__ == "__main__":
    main()
