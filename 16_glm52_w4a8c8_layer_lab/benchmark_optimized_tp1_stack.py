#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
import torch_npu
from torch_npu.dynamo import torchair
from torch_npu.dynamo.torchair.configs.compiler_config import CompilerConfig

from modeling_glm52_dense_tp import prepare_w8a8_weight_format
from modeling_glm52_layer import configure_grouped_matmul_scale_conversion
from modeling_glm52_tp1 import GLM52OptimizedTP1Stack


PROFILE_METRICS = ("pipe", "memory", "memory_access", "l2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Integrated optimized GLM-5.2 W4A8C8 TP1 decoder-stack benchmark."
        )
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--first-layer", type=int, default=2)
    parser.add_argument("--last-layer", type=int, default=10)
    parser.add_argument("--cache-length", type=int, default=4096)
    parser.add_argument("--validation-steps", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--decode-steps", type=int, default=200)
    parser.add_argument("--measurement-repeats", type=int, default=3)
    parser.add_argument(
        "--compile-cache-dir",
        type=Path,
        default=Path(".runtime_cache/16_glm52_w4a8c8_layer_lab"),
    )
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--profile-metric", choices=PROFILE_METRICS, default="pipe")
    parser.add_argument("--profile-warmup-steps", type=int, default=20)
    parser.add_argument("--profile-active-steps", type=int, default=5)
    parser.add_argument("--summary-out", type=Path)
    return parser.parse_args()


def source_hash() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for name in (
        "absorbed_mla.py",
        "modeling_glm52_layer.py",
        "modeling_glm52_dense_tp.py",
        "modeling_glm52_stack.py",
        "modeling_glm52_tp1.py",
        "benchmark_optimized_tp1_stack.py",
    ):
        digest.update((root / name).read_bytes())
    return digest.hexdigest()[:12]


def import_cache_compile():
    try:
        from torch_npu.dynamo.torchair.inference import cache_compile
    except ImportError:
        from torchair.inference import cache_compile
    return cache_compile


def memory_snapshot(device: torch.device) -> dict[str, float]:
    torch.npu.synchronize(device)
    free_bytes, total_bytes = torch.npu.mem_get_info(device)
    return {
        "allocated_gib": torch.npu.memory_allocated(device) / 2**30,
        "reserved_gib": torch.npu.memory_reserved(device) / 2**30,
        "free_gib": free_bytes / 2**30,
        "total_gib": total_bytes / 2**30,
    }


def make_hidden_rows(
    *,
    steps: int,
    hidden_size: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    bank = torch.randn(
        steps,
        1,
        1,
        hidden_size,
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=torch.bfloat16)
    return bank.unbind(0)


def make_position_rows(
    *, first_position: int, steps: int, device: torch.device
) -> tuple[torch.Tensor, ...]:
    return torch.arange(
        first_position,
        first_position + steps,
        dtype=torch.int64,
        device=device,
    ).view(steps, 1).unbind(0)


def make_prefilled_caches(
    stack: GLM52OptimizedTP1Stack,
    *,
    device: torch.device,
    prefix_length: int,
    seed: int,
):
    caches = stack.make_cache(device=device)
    return fill_prefilled_caches(
        stack,
        caches,
        device=device,
        prefix_length=prefix_length,
        seed=seed,
    )


def fill_prefilled_caches(
    stack: GLM52OptimizedTP1Stack,
    caches,
    *,
    device: torch.device,
    prefix_length: int,
    seed: int,
):
    if not 0 <= prefix_length < stack.cache_length:
        raise ValueError("prefix_length must fit inside the static cache")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for group in caches:
        for cache in group:
            cache.zero_()
            if cache.shape[1] == 1 and stack.cache_length != 1:
                continue
            prefix = torch.randn(
                cache[:, :prefix_length].shape,
                generator=generator,
                dtype=torch.float32,
            ).to(device=device, dtype=cache.dtype)
            cache[:, :prefix_length].copy_(prefix)
    return caches


def clone_caches(caches):
    return tuple(tuple(cache.clone() for cache in group) for group in caches)


def run_rows(
    decode,
    hidden_rows: tuple[torch.Tensor, ...],
    position_rows: tuple[torch.Tensor, ...],
    caches,
    shared_topk: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = None
    for hidden_row, position_row in zip(hidden_rows, position_rows, strict=True):
        output, shared_topk = decode(
            hidden_row,
            position_row,
            *caches,
            shared_topk,
        )
    if output is None:
        raise ValueError("input rows must not be empty")
    return output, shared_topk


def timed_rows(
    decode,
    hidden_rows: tuple[torch.Tensor, ...],
    position_rows: tuple[torch.Tensor, ...],
    caches,
    shared_topk: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    torch.npu.synchronize(device)
    started = time.perf_counter()
    output, shared_topk = run_rows(
        decode,
        hidden_rows,
        position_rows,
        caches,
        shared_topk,
    )
    torch.npu.synchronize(device)
    return output, shared_topk, time.perf_counter() - started


def tensor_max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max().item())


def cache_max_abs(left, right) -> dict[str, float]:
    names = ("latent", "rope", "index")
    result = {}
    for name, left_group, right_group in zip(names, left, right, strict=True):
        result[name + "_max_abs"] = max(
            tensor_max_abs(left_tensor, right_tensor)
            for left_tensor, right_tensor in zip(
                left_group, right_group, strict=True
            )
        )
    return result


def topk_comparison(left: torch.Tensor, right: torch.Tensor) -> dict[str, object]:
    left_cpu = left.reshape(-1).cpu()
    right_cpu = right.reshape(-1).cpu()
    if left_cpu.numel() != right_cpu.numel():
        raise ValueError("top-k tensors have different sizes")
    overlap = int(torch.isin(left_cpu, right_cpu).sum().item())
    total = int(left_cpu.numel())
    return {
        "count": total,
        "ordered_match_count": int((left_cpu == right_cpu).sum().item()),
        "ordered_exact": bool(torch.equal(left_cpu, right_cpu)),
        "set_exact": bool(
            torch.equal(torch.sort(left_cpu).values, torch.sort(right_cpu).values)
        ),
        "set_overlap_count": overlap,
        "set_overlap_ratio": overlap / total,
    }


def dynamo_stats() -> dict[str, int]:
    return {
        "unique_graphs": int(
            torch._dynamo.utils.counters["stats"]["unique_graphs"]
        ),
        "calls_captured": int(
            torch._dynamo.utils.counters["stats"]["calls_captured"]
        ),
    }


def profile_rows(
    decode,
    *,
    stack: GLM52OptimizedTP1Stack,
    caches,
    profile_root: Path,
    device: torch.device,
    metric: str,
    first_position: int,
    shared_topk: torch.Tensor,
    ordinary_warmup_steps: int,
    active_steps: int,
) -> dict[str, object]:
    import torch_npu.profiler as npu_prof

    profile_root.mkdir(parents=True, exist_ok=False)
    warmup_hidden = make_hidden_rows(
        steps=ordinary_warmup_steps,
        hidden_size=stack.config.hidden_size,
        device=device,
        seed=52120,
    )
    warmup_positions = make_position_rows(
        first_position=first_position,
        steps=ordinary_warmup_steps,
        device=device,
    )
    _, shared_topk, ordinary_warmup_sec = timed_rows(
        decode,
        warmup_hidden,
        warmup_positions,
        caches,
        shared_topk,
        device=device,
    )
    profile_hidden = make_hidden_rows(
        steps=1 + active_steps,
        hidden_size=stack.config.hidden_size,
        device=device,
        seed=52121,
    )
    profile_positions = make_position_rows(
        first_position=first_position + ordinary_warmup_steps,
        steps=1 + active_steps,
        device=device,
    )
    metric_value = {
        "pipe": npu_prof.AiCMetrics.PipeUtilization,
        "memory": npu_prof.AiCMetrics.Memory,
        "memory_access": npu_prof.AiCMetrics.MemoryAccess,
        "l2": npu_prof.AiCMetrics.L2Cache,
    }[metric]
    schedule = npu_prof.schedule(wait=0, warmup=1, active=active_steps, repeat=1)
    experimental_config = npu_prof._ExperimentalConfig(
        profiler_level=npu_prof.ProfilerLevel.Level1,
        aic_metrics=metric_value,
        export_type=npu_prof.ExportType.Text,
    )
    torch.npu.synchronize(device)
    started = time.perf_counter()
    with npu_prof.profile(
        activities=[npu_prof.ProfilerActivity.CPU, npu_prof.ProfilerActivity.NPU],
        schedule=schedule,
        on_trace_ready=npu_prof.tensorboard_trace_handler(
            str(profile_root), analyse_flag=True
        ),
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
        experimental_config=experimental_config,
    ) as prof:
        for offset, (hidden_row, position_row) in enumerate(
            zip(profile_hidden, profile_positions, strict=True)
        ):
            _output, shared_topk = decode(
                hidden_row,
                position_row,
                *caches,
                shared_topk,
            )
            if offset in (0, active_steps):
                torch.npu.synchronize(device)
            prof.step()
    torch.npu.synchronize(device)
    return {
        "profile_root": str(profile_root),
        "metric": metric,
        "ordinary_warmup_steps": ordinary_warmup_steps,
        "ordinary_warmup_sec_excluded": ordinary_warmup_sec,
        "profiler_schedule_warmup_steps": 1,
        "active_steps": active_steps,
        "profiled_loop_sec_including_profiler_overhead": (
            time.perf_counter() - started
        ),
        "expected_calls_per_active_step": {
            "sparse_flash_attention": len(stack.layers),
            "lightning_indexer": len(stack.full_indexer_layers),
            "moe_init_routing": sum(
                layer.layer_index >= stack.config.first_k_dense_replace
                for layer in stack.layers
            ),
            "grouped_matmul": 2
            * sum(
                layer.layer_index >= stack.config.first_k_dense_replace
                for layer in stack.layers
            ),
        },
    }


def main() -> None:
    args = parse_args()
    if min(
        args.validation_steps,
        args.warmup_steps,
        args.decode_steps,
        args.measurement_repeats,
        args.profile_warmup_steps,
        args.profile_active_steps,
    ) < 1:
        raise ValueError("all step and repeat counts must be positive")
    profile_steps = (
        args.profile_warmup_steps + 1 + args.profile_active_steps
        if args.profile_dir is not None
        else 0
    )
    continuation_steps = (
        args.validation_steps
        + args.warmup_steps
        + args.measurement_repeats * args.decode_steps
        + profile_steps
    )
    if continuation_steps > args.cache_length:
        raise ValueError("continuous validation and measurement exceed the cache")

    torch.npu.config.allow_internal_format = True
    torch.npu.set_compile_mode(jit_compile=False)
    device = torch.device(args.device)
    torch.npu.set_device(device)
    configure_grouped_matmul_scale_conversion("bitcast")

    load_started = time.perf_counter()
    stack = GLM52OptimizedTP1Stack.from_checkpoint(
        args.model_dir,
        first_layer=args.first_layer,
        last_layer=args.last_layer,
        cache_length=args.cache_length,
        device=device,
        progress=lambda message: print("[optimized-tp1] " + message, flush=True),
        w4_weight_format="fractal_nz",
    )
    stack.eval()
    weight_format = prepare_w8a8_weight_format(
        stack,
        requested="fractal_nz",
    )
    expected_w8_count = sum(
        6 if layer.indexer is not None else 5 for layer in stack.layers
    )
    if int(weight_format["quant_linear_count"]) != expected_w8_count:
        raise RuntimeError(
            f"expected {expected_w8_count} W8A8 linears, got "
            f"{weight_format['quant_linear_count']}"
        )
    torch.npu.synchronize(device)
    load_sec = time.perf_counter() - load_started
    memory_after_weights = memory_snapshot(device)
    print(
        "[optimized-tp1] weights loaded "
        + json.dumps(memory_after_weights, sort_keys=True),
        flush=True,
    )

    validation_first_position = args.cache_length - continuation_steps
    with torch.inference_mode():
        validation_hidden = make_hidden_rows(
            steps=args.validation_steps,
            hidden_size=stack.config.hidden_size,
            device=device,
            seed=52100,
        )
        validation_positions = make_position_rows(
            first_position=validation_first_position,
            steps=args.validation_steps,
            device=device,
        )
        eager_caches = make_prefilled_caches(
            stack,
            device=device,
            prefix_length=validation_first_position,
            seed=52099,
        )
        compiled_caches = clone_caches(eager_caches)
        eager_output, eager_topk = run_rows(
            stack.forward_decode,
            validation_hidden,
            validation_positions,
            eager_caches,
            stack.initial_topk(device=device),
        )
        torch.npu.synchronize(device)

        shape_key = (
            f"tp1_l{args.first_layer}_{args.last_layer}_b1_kv{args.cache_length}_"
            f"bf16_src{source_hash()}"
        )
        cache_dir = args.compile_cache_dir.expanduser().resolve() / shape_key
        cache_was_warm = cache_dir.is_dir() and any(cache_dir.iterdir())
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        compiled = import_cache_compile()(
            stack.forward_decode,
            config=CompilerConfig(),
            dynamic=False,
            cache_dir=str(cache_dir),
            ge_cache=True,
            fullgraph=True,
        )
        compiled_output, compiled_topk = run_rows(
            compiled,
            validation_hidden,
            validation_positions,
            compiled_caches,
            stack.initial_topk(device=device),
        )
        torch.npu.synchronize(device)
        stats_after_validation = dynamo_stats()

        output_diff = tensor_max_abs(compiled_output, eager_output)
        output_mean_diff = float(
            (compiled_output.float() - eager_output.float()).abs().mean().item()
        )
        parity = {
            "output_max_abs": output_diff,
            "output_mean_abs": output_mean_diff,
            "output_allclose_atol_5e_2_rtol_5e_2": bool(
                torch.allclose(
                    compiled_output,
                    eager_output,
                    atol=5e-2,
                    rtol=5e-2,
                )
            ),
            "shared_topk": topk_comparison(compiled_topk, eager_topk),
            **cache_max_abs(compiled_caches, eager_caches),
        }
        if not parity["output_allclose_atol_5e_2_rtol_5e_2"]:
            raise RuntimeError("compiled output failed eager parity")
        if parity["shared_topk"]["set_overlap_ratio"] < 0.99:
            raise RuntimeError(
                "compiled shared top-k overlap fell below 99%: "
                + json.dumps(parity["shared_topk"], sort_keys=True)
            )
        if max(
            parity["latent_max_abs"],
            parity["rope_max_abs"],
            parity["index_max_abs"],
        ) > 5e-2:
            raise RuntimeError(
                "compiled cache state failed eager parity: "
                + json.dumps(parity, sort_keys=True)
            )

        warmup_first_position = validation_first_position + args.validation_steps
        warmup_hidden = make_hidden_rows(
            steps=args.warmup_steps,
            hidden_size=stack.config.hidden_size,
            device=device,
            seed=52101,
        )
        warmup_positions = make_position_rows(
            first_position=warmup_first_position,
            steps=args.warmup_steps,
            device=device,
        )
        _warmup_output, continuation_topk, warmup_sec = timed_rows(
            compiled,
            warmup_hidden,
            warmup_positions,
            compiled_caches,
            compiled_topk,
            device=device,
        )
        stats_after_warmup = dynamo_stats()

        repeat_elapsed_sec = []
        final_output = None
        for repeat in range(args.measurement_repeats):
            measured_first_position = (
                warmup_first_position
                + args.warmup_steps
                + repeat * args.decode_steps
            )
            measured_hidden = make_hidden_rows(
                steps=args.decode_steps,
                hidden_size=stack.config.hidden_size,
                device=device,
                seed=52102 + repeat,
            )
            measured_positions = make_position_rows(
                first_position=measured_first_position,
                steps=args.decode_steps,
                device=device,
            )
            final_output, _final_topk, elapsed_sec = timed_rows(
                compiled,
                measured_hidden,
                measured_positions,
                compiled_caches,
                continuation_topk,
                device=device,
            )
            continuation_topk = _final_topk
            repeat_elapsed_sec.append(elapsed_sec)
        stats_after_measurement = dynamo_stats()
        if (
            stats_after_measurement["unique_graphs"]
            != stats_after_warmup["unique_graphs"]
        ):
            raise RuntimeError("TorchAir captured a graph during measurement")
        if final_output is None:
            raise RuntimeError("measurement produced no output")

    median_elapsed_sec = statistics.median(repeat_elapsed_sec)
    layer_count = len(stack.layers)
    moe_layer_count = sum(
        layer.layer_index >= stack.config.first_k_dense_replace
        for layer in stack.layers
    )
    summary = {
        "model": "Eco-Tech/GLM-5.2-w4a8c8",
        "chip": "Ascend 910B2",
        "tensor_parallel_size": 1,
        "layers": list(range(args.first_layer, args.last_layer + 1)),
        "layer_count": layer_count,
        "dense_layer_count": layer_count - moe_layer_count,
        "moe_layer_count": moe_layer_count,
        "full_indexer_layers": stack.full_indexer_layers,
        "shared_indexer_layers": stack.shared_indexer_layers,
        "backend": "torchair_fullgraph_static_cache_compile",
        "batch_size": 1,
        "cache_length": args.cache_length,
        "cache_prefix_mode": "deterministic_varied_bfloat16_prefix",
        "attention_path": "absorbed_contiguous_bsnd_sparse_flash",
        "indexer_path": "contiguous_bsnd_lightning_indexer_with_four_layer_reuse",
        "rope_path": "block_layout_npu_interleave_rope",
        "moe_path": "w4a8_grouped_matmul_bitcast_scale_fractal_nz",
        "w8_weight_format": weight_format,
        "compile_cache_dir": str(cache_dir),
        "compile_cache_was_warm": cache_was_warm,
        "load_sec": load_sec,
        "memory_after_weights": memory_after_weights,
        "validation_steps": args.validation_steps,
        "validation_first_position": validation_first_position,
        "continuous_last_position": args.cache_length - 1,
        "parity": parity,
        "warmup_steps": args.warmup_steps,
        "warmup_elapsed_sec_excluded": warmup_sec,
        "decode_steps": args.decode_steps,
        "measurement_repeats": args.measurement_repeats,
        "repeat_elapsed_sec": repeat_elapsed_sec,
        "repeat_mean_stack_ms": [
            1000.0 * elapsed / args.decode_steps
            for elapsed in repeat_elapsed_sec
        ],
        "median_mean_stack_ms": 1000.0 * median_elapsed_sec / args.decode_steps,
        "stack_calls_per_sec": args.decode_steps / median_elapsed_sec,
        "effective_layer_calls_per_sec": (
            layer_count * args.decode_steps / median_elapsed_sec
        ),
        "final_output_abs_max": float(final_output.float().abs().max().item()),
        "dynamo": {
            "after_validation": stats_after_validation,
            "after_warmup": stats_after_warmup,
            "after_measurement": stats_after_measurement,
        },
        "profile": None,
    }
    if args.profile_dir is not None:
        with torch.inference_mode():
            summary["profile"] = profile_rows(
                compiled,
                stack=stack,
                caches=compiled_caches,
                profile_root=args.profile_dir,
                device=device,
                metric=args.profile_metric,
                first_position=(
                    validation_first_position
                    + args.validation_steps
                    + args.warmup_steps
                    + args.measurement_repeats * args.decode_steps
                ),
                shared_topk=continuation_topk,
                ordinary_warmup_steps=args.profile_warmup_steps,
                active_steps=args.profile_active_steps,
            )
        summary["dynamo"]["after_profile"] = dynamo_stats()
        if (
            summary["dynamo"]["after_profile"]["unique_graphs"]
            != stats_after_measurement["unique_graphs"]
        ):
            raise RuntimeError("TorchAir captured a graph during profiling")
    result = {"tp1": summary}
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
    print(
        "GLM52_OPTIMIZED_TP1_STACK_SUMMARY "
        + json.dumps(result, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
