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

from modeling_glm52_dense_tp import prepare_w8a8_weight_format
from modeling_glm52_moe_tp1 import GLM52MoEMLPStack


PROFILE_METRICS = ("pipe", "memory", "memory_access", "l2")


def configure_grouped_matmul_scale_conversion(mode: str) -> None:
    if mode == "cast":
        return
    if mode != "bitcast":
        raise ValueError(f"Unsupported grouped-matmul scale conversion: {mode}")
    from torch_npu.dynamo.torchair._ge_concrete_graph.ge_converter.custom import (
        grouped_matmul as grouped_matmul_converter,
    )

    def bitcast_int64_scales(scales):
        if scales[0].dtype != grouped_matmul_converter.DataType.DT_INT64:
            return scales
        return [
            grouped_matmul_converter.ge.Bitcast(
                scale,
                type=grouped_matmul_converter.DataType.DT_UINT64,
                keep_dim=True,
            )
            for scale in scales
        ]

    grouped_matmul_converter.convert_scale_tensorlist = bitcast_int64_scales


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GLM-5.2 W4A8C8 TP1 MoE MLP stack benchmark."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--first-layer", type=int, default=3)
    parser.add_argument("--last-layer", type=int, default=5)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--decode-steps", type=int, default=1000)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--profile-metric", choices=PROFILE_METRICS, default="pipe")
    parser.add_argument("--profile-warmup-steps", type=int, default=20)
    parser.add_argument("--profile-active-steps", type=int, default=5)
    parser.add_argument(
        "--gmm-scale-conversion",
        choices=("cast", "bitcast"),
        default="bitcast",
    )
    parser.add_argument(
        "--w4-weight-format",
        choices=("native", "fractal_nz"),
        default="native",
    )
    parser.add_argument("--summary-out", type=Path)
    return parser.parse_args()


def memory_snapshot(device: torch.device) -> dict[str, float]:
    torch.npu.synchronize()
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
    if steps < 1:
        raise ValueError("steps must be positive")
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


def run_rows(forward, rows: tuple[torch.Tensor, ...]) -> torch.Tensor:
    output = None
    for row in rows:
        output = forward(row)
    if output is None:
        raise ValueError("input rows must not be empty")
    return output


def timed_rows(
    forward,
    rows: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, float]:
    torch.npu.synchronize()
    started = time.perf_counter()
    output = run_rows(forward, rows)
    torch.npu.synchronize()
    return output, time.perf_counter() - started


def profile_rows(
    forward,
    *,
    stack: GLM52MoEMLPStack,
    profile_root: Path,
    device: torch.device,
    metric: str,
    ordinary_warmup_steps: int,
    active_steps: int,
) -> dict[str, object]:
    import torch_npu.profiler as npu_prof

    profile_root.mkdir(parents=True, exist_ok=False)
    ordinary_rows = make_hidden_rows(
        steps=ordinary_warmup_steps,
        hidden_size=stack.config.hidden_size,
        device=device,
        seed=52003,
    )
    torch.npu.synchronize()
    _, ordinary_warmup_sec = timed_rows(forward, ordinary_rows)
    profile_rows_all = make_hidden_rows(
        steps=1 + active_steps,
        hidden_size=stack.config.hidden_size,
        device=device,
        seed=52004,
    )
    torch.npu.synchronize()
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
    torch.npu.synchronize()
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
        for index, row in enumerate(profile_rows_all):
            forward(row)
            if index in (0, active_steps):
                torch.npu.synchronize()
            prof.step()
    torch.npu.synchronize()
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
        "execution_mode": (
            "preallocated_varied_inputs_contiguous_active_graph_calls"
        ),
    }


def main() -> None:
    args = parse_args()
    if min(
        args.warmup_steps,
        args.decode_steps,
        args.profile_warmup_steps,
        args.profile_active_steps,
    ) < 1:
        raise ValueError("warmup, decode, and profile steps must be positive")
    torch.npu.config.allow_internal_format = True
    torch.npu.set_compile_mode(jit_compile=False)
    device = torch.device(args.device)
    torch.npu.set_device(device)

    load_started = time.perf_counter()
    stack = GLM52MoEMLPStack.from_checkpoint(
        args.model_dir,
        first_layer=args.first_layer,
        last_layer=args.last_layer,
        device=device,
        progress=lambda message: print("[moe-tp1] " + message, flush=True),
        w4_weight_format=args.w4_weight_format,
    )
    stack.eval()
    shared_weight_format = prepare_w8a8_weight_format(
        stack, requested="fractal_nz"
    )
    torch.npu.synchronize()
    load_sec = time.perf_counter() - load_started
    weights_memory = memory_snapshot(device)

    with torch.inference_mode():
        reference_row = make_hidden_rows(
            steps=1,
            hidden_size=stack.config.hidden_size,
            device=device,
            seed=52000,
        )[0]
        eager_output = stack(reference_row)
        torch.npu.synchronize()
        if not bool(torch.isfinite(eager_output).all().item()):
            raise RuntimeError("eager MoE output is not finite")

        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        configure_grouped_matmul_scale_conversion(args.gmm_scale_conversion)
        compiled = torch.compile(
            stack.forward,
            backend=torchair.get_npu_backend(
                compiler_config=CompilerConfig()
            ),
            dynamic=False,
            fullgraph=True,
        )
        compiled_reference = compiled(reference_row.clone())
        torch.npu.synchronize()
        reference_diff = (compiled_reference.float() - eager_output.float()).abs()
        if not torch.allclose(
            compiled_reference, eager_output, atol=5e-2, rtol=5e-2
        ):
            raise RuntimeError(
                "compiled MoE output failed eager parity: "
                f"max_abs={float(reference_diff.max().item())}"
            )

        warmup_rows = make_hidden_rows(
            steps=args.warmup_steps,
            hidden_size=stack.config.hidden_size,
            device=device,
            seed=52001,
        )
        torch.npu.synchronize()
        _, warmup_sec = timed_rows(compiled, warmup_rows)
        dynamo_after_warmup = {
            "unique_graphs": int(
                torch._dynamo.utils.counters["stats"]["unique_graphs"]
            ),
            "calls_captured": int(
                torch._dynamo.utils.counters["stats"]["calls_captured"]
            ),
        }
        measured_rows = make_hidden_rows(
            steps=args.decode_steps,
            hidden_size=stack.config.hidden_size,
            device=device,
            seed=52002,
        )
        torch.npu.synchronize()
        final_output, elapsed_sec = timed_rows(compiled, measured_rows)
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
        "layers": list(range(args.first_layer, args.last_layer + 1)),
        "scope": "post_attention_norm_plus_routed_and_shared_moe_mlp",
        "batch_size": 1,
        "hidden_size": stack.config.hidden_size,
        "num_experts": stack.config.num_experts,
        "experts_per_token": stack.config.top_k,
        "moe_intermediate_size": stack.config.moe_intermediate_size,
        "backend": "torchair_fullgraph_static",
        "gmm_scale_conversion": args.gmm_scale_conversion,
        "requested_w4_weight_format": args.w4_weight_format,
        "input_mode": "preallocated_varied_bfloat16_hidden_rows",
        "shared_w8a8_weight_format": shared_weight_format,
        "routed_w4a8_weight_storage": {
            "w13_dtype": str(stack.blocks[0].routed.w13_weight.dtype),
            "w13_format": int(
                torch_npu.get_npu_format(stack.blocks[0].routed.w13_weight)
            ),
            "w2_dtype": str(stack.blocks[0].routed.w2_weight.dtype),
            "w2_format": int(
                torch_npu.get_npu_format(stack.blocks[0].routed.w2_weight)
            ),
        },
        "load_sec": load_sec,
        "memory_after_weights": weights_memory,
        "warmup_steps": args.warmup_steps,
        "warmup_elapsed_sec_excluded": warmup_sec,
        "decode_steps": args.decode_steps,
        "elapsed_sec": elapsed_sec,
        "mean_stack_ms": 1000.0 * elapsed_sec / args.decode_steps,
        "stack_calls_per_sec": args.decode_steps / elapsed_sec,
        "effective_moe_layer_calls_per_sec": (
            len(stack.blocks) * args.decode_steps / elapsed_sec
        ),
        "reference_parity": {
            "output_max_abs": float(reference_diff.max().item()),
            "output_mean_abs": float(reference_diff.mean().item()),
            "allclose_atol_5e_2_rtol_5e_2": True,
        },
        "final_output_abs_max": float(final_output.float().abs().max().item()),
        "dynamo": {
            "after_warmup": dynamo_after_warmup,
            "after_measurement": dynamo_after_measurement,
        },
    }
    if args.profile_dir is not None:
        summary["profile"] = profile_rows(
            compiled,
            stack=stack,
            profile_root=args.profile_dir,
            device=device,
            metric=args.profile_metric,
            ordinary_warmup_steps=args.profile_warmup_steps,
            active_steps=args.profile_active_steps,
        )
    result = {"tp1": summary}
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
    print("GLM52_MOE_TP1_SUMMARY " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
