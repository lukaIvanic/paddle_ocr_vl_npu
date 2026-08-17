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
        description=(
            "Compile a fixed-shape Qwen3-30B-A3B second-stage decode graph and "
            "compare it with the captured eager path."
        )
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument(
        "--expert-impl",
        choices=("selected_bmm", "grouped_matmul"),
        default="selected_bmm",
    )
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
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=5e-2)
    return parser.parse_args()


def source_hash() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    digest.update((root / "modeling_qwen3_moe_pipeline.py").read_bytes())
    return digest.hexdigest()[:12]


def import_cache_compile():
    try:
        from torch_npu.dynamo.torchair.inference import cache_compile
    except ImportError:
        from torchair.inference import cache_compile
    return cache_compile


def restore_capture_prefix(cache, capture: dict[str, object]) -> None:
    restored = cache.restore_prefix(capture["stage2_prefix_cache"])
    expected = int(capture["prefix_length"])
    if restored != expected:
        raise RuntimeError(f"Restored prefix {restored}, expected {expected}")


def synchronize(device: torch.device) -> None:
    torch.npu.synchronize(device)


def run_capture_sequence(
    decode,
    cache,
    capture: dict[str, object],
    *,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[float]]:
    outputs = []
    elapsed = []
    for boundary_cpu, position in zip(
        capture["boundary_hidden_states"], capture["cache_positions"]
    ):
        boundary = boundary_cpu.to(device=device, dtype=torch.bfloat16)
        cache_position = torch.tensor([position], dtype=torch.int64, device=device)
        synchronize(device)
        started = time.perf_counter()
        output = decode(
            boundary,
            cache_position,
            cache.key_caches,
            cache.value_caches,
        )
        synchronize(device)
        elapsed.append(time.perf_counter() - started)
        outputs.append(output.detach().cpu())
    return outputs, elapsed


def benchmark_continuation(
    decode,
    cache,
    capture: dict[str, object],
    *,
    device: torch.device,
    warmup_steps: int,
    decode_steps: int,
) -> dict[str, float]:
    boundary = capture["boundary_hidden_states"][-1].to(
        device=device, dtype=torch.bfloat16
    )
    first_position = int(capture["cache_positions"][-1]) + 1
    required_position = first_position + warmup_steps + decode_steps - 1
    cache_length = int(cache.key_caches[0].shape[2])
    if required_position >= cache_length:
        raise ValueError(
            f"Benchmark needs position {required_position}, cache length is {cache_length}"
        )
    for offset in range(warmup_steps):
        position = torch.tensor(
            [first_position + offset], dtype=torch.int64, device=device
        )
        decode(boundary, position, cache.key_caches, cache.value_caches)
    synchronize(device)
    started = time.perf_counter()
    for offset in range(decode_steps):
        position = torch.tensor(
            [first_position + warmup_steps + offset],
            dtype=torch.int64,
            device=device,
        )
        decode(boundary, position, cache.key_caches, cache.value_caches)
    synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "elapsed_sec": elapsed,
        "steps": decode_steps,
        "tok_s": decode_steps / elapsed,
        "mean_tpot_ms": 1000.0 * elapsed / decode_steps,
    }


def dynamo_stats() -> dict[str, int]:
    counters = torch._dynamo.utils.counters
    return {
        "unique_graphs": int(counters["stats"]["unique_graphs"]),
        "calls_captured": int(counters["stats"]["calls_captured"]),
        "frames_total": int(counters["frames"]["total"]),
        "frames_ok": int(counters["frames"]["ok"]),
    }


def main() -> None:
    args = parse_args()
    if not 1 <= args.layers <= 24:
        raise ValueError("--layers must be in [1, 24]")
    if args.warmup_steps < 0 or args.decode_steps < 1:
        raise ValueError("Invalid warmup/decode step count")
    torch.npu.set_compile_mode(jit_compile=False)
    device = torch.device(args.device)
    torch.npu.set_device(device)
    capture = torch.load(args.capture, map_location="cpu", weights_only=False)
    if capture["format"] != "qwen3_30b_a3b_stage2_replay_v1":
        raise ValueError(f"Unsupported capture format: {capture['format']}")

    config = Qwen3MoeConfig.from_model_dir(args.model_dir)
    config.validate_qwen3_30b_a3b()
    layer_start = int(capture["stage2_layer_start"])
    layer_end = layer_start + args.layers
    complete_stage = layer_end == int(capture["stage2_layer_end"])
    cache_length = (
        int(args.cache_length)
        if args.cache_length is not None
        else int(capture["cache_length"])
    )
    stage, stage_metadata = build_stage(
        config,
        args.model_dir,
        layer_start=layer_start,
        layer_end=layer_end,
        with_embedding=False,
        with_lm_head=complete_stage,
        device=device,
        name=f"stage2-compile-l{args.layers}",
        cache_length=cache_length,
        expert_impl=args.expert_impl,
    )

    eager_cache = stage.make_cache(cache_length=cache_length)
    restore_capture_prefix(eager_cache, capture)
    eager_outputs, eager_capture_times = run_capture_sequence(
        stage.decode_static_output,
        eager_cache,
        capture,
        device=device,
    )
    eager_benchmark = benchmark_continuation(
        stage.decode_static_output,
        eager_cache,
        capture,
        device=device,
        warmup_steps=args.warmup_steps,
        decode_steps=args.decode_steps,
    )

    shape_key = (
        f"stage2_{args.expert_impl}_l{args.layers}_b1_kv{cache_length}_bf16_"
        f"src{source_hash()}"
    )
    cache_dir = args.compile_cache_dir.expanduser().resolve() / shape_key
    cache_was_warm = cache_dir.is_dir() and any(cache_dir.iterdir())
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    from torch_npu.dynamo.torchair.configs.compiler_config import CompilerConfig

    wrapper_started = time.perf_counter()
    compiled_decode = import_cache_compile()(
        stage.decode_static_output,
        config=CompilerConfig(),
        dynamic=False,
        cache_dir=str(cache_dir),
        ge_cache=True,
        fullgraph=True,
    )
    compile_wrapper_sec = time.perf_counter() - wrapper_started
    compiled_cache = stage.make_cache(cache_length=cache_length)
    restore_capture_prefix(compiled_cache, capture)
    compiled_outputs, compiled_capture_times = run_capture_sequence(
        compiled_decode,
        compiled_cache,
        capture,
        device=device,
    )
    stats_after_capture = dynamo_stats()
    compiled_benchmark = benchmark_continuation(
        compiled_decode,
        compiled_cache,
        capture,
        device=device,
        warmup_steps=args.warmup_steps,
        decode_steps=args.decode_steps,
    )
    stats_final = dynamo_stats()

    step_checks = []
    all_outputs_close = True
    all_tokens_match = True
    for step, (eager, compiled) in enumerate(zip(eager_outputs, compiled_outputs)):
        diff = (eager.float() - compiled.float()).abs()
        close = bool(
            torch.allclose(eager, compiled, atol=args.atol, rtol=args.rtol)
        )
        eager_top2 = torch.topk(
            eager.float().reshape(-1, eager.shape[-1]), 2, dim=-1
        ).values
        compiled_top2 = torch.topk(
            compiled.float().reshape(-1, compiled.shape[-1]), 2, dim=-1
        ).values
        check = {
            "step": step,
            "allclose": close,
            "max_abs": float(diff.max().item()),
            "mean_abs": float(diff.mean().item()),
            "eager_top1_margin": float(
                (eager_top2[0, 0] - eager_top2[0, 1]).item()
            ),
            "compiled_top1_margin": float(
                (compiled_top2[0, 0] - compiled_top2[0, 1]).item()
            ),
        }
        if complete_stage:
            expected = capture["expected_logits"][step]
            check["expected_logits_allclose"] = bool(
                torch.allclose(compiled, expected, atol=args.atol, rtol=args.rtol)
            )
            check["token_id"] = int(compiled.argmax(dim=-1).item())
            check["expected_token_id"] = int(capture["generated_token_ids"][step])
            check["token_match"] = check["token_id"] == check["expected_token_id"]
            check["top10_ids_match"] = torch.equal(
                torch.topk(compiled.float(), 10, dim=-1).indices,
                capture["expected_topk_ids"][step],
            )
            all_tokens_match &= check["token_match"]
        all_outputs_close &= bool(close)
        step_checks.append(check)

    parity_passed = all_tokens_match if complete_stage else all_outputs_close

    unique_graph_delta = (
        stats_final["unique_graphs"] - stats_after_capture["unique_graphs"]
    )
    no_recompilations_after_capture = unique_graph_delta == 0
    summary = {
        "model": capture["model"],
        "chip": "Ascend 910B2",
        "dtype": "bfloat16",
        "layers": args.layers,
        "complete_stage": complete_stage,
        "cache_length": cache_length,
        "expert_impl": args.expert_impl,
        "capture_steps": len(compiled_capture_times),
        "compile_contract": {
            "fullgraph": True,
            "dynamic": False,
            "ge_cache": True,
            "cache_dir": str(cache_dir),
            "cache_was_warm": cache_was_warm,
            "compile_wrapper_sec": compile_wrapper_sec,
            "first_call_sec": compiled_capture_times[0],
            "dynamo_after_capture": stats_after_capture,
            "dynamo_final": stats_final,
            "unique_graph_delta_after_capture": unique_graph_delta,
            "no_recompilations_after_capture": no_recompilations_after_capture,
        },
        "parity": {
            "gate": (
                "exact_greedy_token_ids" if complete_stage else "output_allclose"
            ),
            "passed": parity_passed,
            "all_outputs_close": all_outputs_close,
            "all_tokens_match": all_tokens_match if complete_stage else None,
            "step_checks": step_checks,
        },
        "eager": {
            "capture_step_times_sec": eager_capture_times,
            "benchmark": eager_benchmark,
        },
        "compiled": {
            "capture_step_times_sec": compiled_capture_times,
            "benchmark": compiled_benchmark,
        },
        "speedup": eager_benchmark["mean_tpot_ms"]
        / compiled_benchmark["mean_tpot_ms"],
        "stage": stage_metadata,
        "memory": memory_snapshot(device),
    }
    print("QWEN3_MOE_STAGE2_COMPILE " + json.dumps(summary, sort_keys=True))
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if not parity_passed:
        raise RuntimeError("Compiled stage output parity failed")
    if not no_recompilations_after_capture:
        raise RuntimeError("Additional static graphs were compiled after capture parity")


if __name__ == "__main__":
    main()
