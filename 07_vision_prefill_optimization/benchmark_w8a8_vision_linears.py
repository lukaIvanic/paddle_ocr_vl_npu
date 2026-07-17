#!/usr/bin/env python3
"""Benchmark the PaddleOCR-VL vision-transformer matmuls with native W8A8.

The benchmark mirrors the four projection calls used by the optimized vision
encoder boundary: grouped QKV, attention output, MLP fc1, and MLP fc2. Weight
quantization and packing happen once, outside timed regions. Both candidates
include activation quantization plus the quantized matmul in their headlines:

* dynamic per-token A8, matching the modern A2/A3 vLLM-Ascend path;
* calibrated static per-tensor A8, matching the portable 310P path.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F


FRACTAL_NZ = 29


@dataclass(frozen=True)
class LinearSpec:
    name: str
    in_features: int
    out_features: int


VISION_LINEAR_SPECS = (
    LinearSpec("qkv", 1152, 3 * 1152),
    LinearSpec("out_proj", 1152, 1152),
    LinearSpec("fc1", 1152, 4304),
    LinearSpec("fc2", 4304, 1152),
)


def maybe_sync(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def quantize_weight_per_output_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return symmetric INT8 [N,K] weights and FP32 per-output scales."""
    weight_fp32 = weight.float()
    scale = weight_fp32.abs().amax(dim=1).clamp_min(torch.finfo(torch.float32).eps) / 127.0
    quantized = torch.round(weight_fp32 / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return quantized, scale


def pack_weight(
    weight_nk: torch.Tensor,
    *,
    device: torch.device,
    layout: str,
) -> torch.Tensor:
    """Pack a logical [N,K] checkpoint weight for QuantMatmul's [K,N] input."""
    import torch_npu

    if layout == "nd_kn":
        return weight_nk.transpose(0, 1).contiguous().to(device)
    if layout == "nz_kn":
        weight_kn = weight_nk.transpose(0, 1).contiguous().to(device)
        return torch_npu.npu_format_cast(weight_kn, FRACTAL_NZ)
    if layout == "nz_nk_transposed":
        # This is the packing order used by vLLM-Ascend's 310P W8A8 path:
        # checkpoint [N,K] -> FRACTAL_NZ -> logical transpose to [K,N].
        weight_nk_nz = torch_npu.npu_format_cast(weight_nk.contiguous().to(device), FRACTAL_NZ)
        return weight_nk_nz.transpose(0, 1)
    raise ValueError(f"unknown weight layout: {layout}")


def elapsed_per_call(
    fn: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> tuple[float, torch.Tensor]:
    output = fn()
    for _ in range(max(0, int(warmup) - 1)):
        output = fn()
    maybe_sync(device)
    start = time.perf_counter()
    for _ in range(int(iterations)):
        output = fn()
    maybe_sync(device)
    return float((time.perf_counter() - start) / int(iterations)), output


def diff_stats(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    ref = reference.float().flatten()
    cand = candidate.float().flatten()
    diff = (ref - cand).abs()
    ref_norm = torch.linalg.vector_norm(ref)
    cand_norm = torch.linalg.vector_norm(cand)
    cosine = torch.dot(ref, cand) / (ref_norm * cand_norm).clamp_min(torch.finfo(torch.float32).eps)
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "rmse": float(torch.sqrt(torch.mean((ref - cand).square())).item()),
        "cosine": float(cosine.item()),
    }


def benchmark_one(
    spec: LinearSpec,
    *,
    rows: int,
    device: torch.device,
    dtype: torch.dtype,
    weight_layout: str,
    warmup: int,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    if device.type != "npu":
        raise ValueError("native W8A8 benchmark requires an NPU device")
    import torch_npu

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    # A transformer-like scale avoids unrealistic saturation while preserving
    # the operator's performance characteristics.
    x_cpu = torch.randn(rows, spec.in_features, generator=generator, dtype=torch.float32).to(dtype)
    weight_cpu = (
        torch.randn(spec.out_features, spec.in_features, generator=generator, dtype=torch.float32)
        / math.sqrt(spec.in_features)
    ).to(dtype)
    bias_cpu = torch.zeros(spec.out_features, dtype=dtype)

    weight_int8_nk, weight_scale = quantize_weight_per_output_channel(weight_cpu)
    input_scale_cpu = x_cpu.float().abs().max().clamp_min(torch.finfo(torch.float32).eps) / 127.0
    dequant_scale_cpu = input_scale_cpu * weight_scale
    quant_bias_cpu = torch.round(bias_cpu.float() / dequant_scale_cpu).to(torch.int32)
    x = x_cpu.to(device)
    weight_fp = weight_cpu.to(device)
    bias = bias_cpu.to(device)
    weight_int8_kn = pack_weight(weight_int8_nk, device=device, layout=weight_layout)
    weight_scale = weight_scale.to(device=device, dtype=torch.float32)
    input_scale = input_scale_cpu.repeat(spec.in_features).to(device=device, dtype=torch.float32)
    input_zero_point = torch.zeros(spec.in_features, device=device, dtype=torch.int8)
    dequant_scale = torch_npu.npu_trans_quant_param(dequant_scale_cpu.to(device=device, dtype=torch.float32))
    quant_bias = quant_bias_cpu.to(device=device)

    def fp16_linear() -> torch.Tensor:
        return F.linear(x, weight_fp, bias)

    group_list = torch.full((1,), rows, dtype=torch.int64, device=device)
    grouped_weight = weight_fp.transpose(0, 1).contiguous().unsqueeze(0)
    grouped_bias = [bias.contiguous().unsqueeze(0)]

    def fp16_grouped() -> torch.Tensor:
        return torch_npu.npu_grouped_matmul(
            [x],
            [grouped_weight],
            bias=grouped_bias,
            group_list=group_list,
            split_item=2,
            group_type=0,
            group_list_type=1,
        )[0]

    def dynamic_w8a8() -> torch.Tensor:
        quantized_x, pertoken_scale = torch_npu.npu_dynamic_quant(x)
        return torch_npu.npu_quant_matmul(
            quantized_x,
            weight_int8_kn,
            weight_scale,
            pertoken_scale=pertoken_scale,
            bias=bias,
            output_dtype=dtype,
        )

    def static_quantize() -> torch.Tensor:
        return torch_npu.npu_quantize(
            x,
            input_scale,
            input_zero_point,
            torch.qint8,
            axis=-1,
            div_mode=True,
        )

    def static_w8a8() -> torch.Tensor:
        quantized_static_x = static_quantize()
        return torch_npu.npu_quant_matmul(
            quantized_static_x,
            weight_int8_kn,
            dequant_scale,
            bias=quant_bias,
            output_dtype=dtype,
        )

    quantized_x, pertoken_scale = torch_npu.npu_dynamic_quant(x)
    quantized_static_x = static_quantize()

    def dynamic_quant_only() -> torch.Tensor:
        return torch_npu.npu_dynamic_quant(x)[0]

    def prequantized_matmul() -> torch.Tensor:
        return torch_npu.npu_quant_matmul(
            quantized_x,
            weight_int8_kn,
            weight_scale,
            pertoken_scale=pertoken_scale,
            bias=bias,
            output_dtype=dtype,
        )

    def static_prequantized_matmul() -> torch.Tensor:
        return torch_npu.npu_quant_matmul(
            quantized_static_x,
            weight_int8_kn,
            dequant_scale,
            bias=quant_bias,
            output_dtype=dtype,
        )

    fp16_s, reference = elapsed_per_call(
        fp16_linear, device=device, warmup=warmup, iterations=iterations
    )
    fp16_grouped_s, grouped_reference = elapsed_per_call(
        fp16_grouped, device=device, warmup=warmup, iterations=iterations
    )
    use_grouped_baseline = spec.name in {"qkv", "fc1"}
    optimized_fp16_baseline_s = fp16_grouped_s if use_grouped_baseline else fp16_s
    w8a8_s, candidate = elapsed_per_call(
        dynamic_w8a8, device=device, warmup=warmup, iterations=iterations
    )
    quant_s, _ = elapsed_per_call(
        dynamic_quant_only, device=device, warmup=warmup, iterations=iterations
    )
    qmm_s, _ = elapsed_per_call(
        prequantized_matmul, device=device, warmup=warmup, iterations=iterations
    )
    static_w8a8_s, static_candidate = elapsed_per_call(
        static_w8a8, device=device, warmup=warmup, iterations=iterations
    )
    static_quant_s, _ = elapsed_per_call(
        static_quantize, device=device, warmup=warmup, iterations=iterations
    )
    static_qmm_s, _ = elapsed_per_call(
        static_prequantized_matmul, device=device, warmup=warmup, iterations=iterations
    )

    maybe_sync(device)
    return {
        "name": spec.name,
        "rows": int(rows),
        "in_features": int(spec.in_features),
        "out_features": int(spec.out_features),
        "weight_layout": str(weight_layout),
        "packed_weight_format": int(torch_npu.get_npu_format(weight_int8_kn)),
        "fp16_linear_s": fp16_s,
        "fp16_grouped_s": fp16_grouped_s,
        "optimized_fp16_baseline": "grouped_matmul" if use_grouped_baseline else "linear",
        "optimized_fp16_baseline_s": optimized_fp16_baseline_s,
        "grouped_vs_linear_diff": diff_stats(reference, grouped_reference),
        "dynamic_w8a8_s": w8a8_s,
        "dynamic_quant_only_s": quant_s,
        "prequantized_matmul_s": qmm_s,
        "dynamic_speedup": float(optimized_fp16_baseline_s / w8a8_s),
        "dynamic_w8a8_time_fraction": float(w8a8_s / optimized_fp16_baseline_s),
        "dynamic_quant_fraction_of_w8a8": float(quant_s / w8a8_s),
        "dynamic_diff": diff_stats(reference, candidate),
        "dynamic_output_nonfinite_count": int((~torch.isfinite(candidate.float())).sum().item()),
        "static_input_scale": float(input_scale_cpu.item()),
        "static_w8a8_s": static_w8a8_s,
        "static_quant_only_s": static_quant_s,
        "static_prequantized_matmul_s": static_qmm_s,
        "static_speedup": float(optimized_fp16_baseline_s / static_w8a8_s),
        "static_w8a8_time_fraction": float(static_w8a8_s / optimized_fp16_baseline_s),
        "static_quant_fraction_of_w8a8": float(static_quant_s / static_w8a8_s),
        "static_diff": diff_stats(reference, static_candidate),
        "static_output_nonfinite_count": int((~torch.isfinite(static_candidate.float())).sum().item()),
    }


def aggregate_layer(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fp16_s = float(sum(float(row["fp16_linear_s"]) for row in rows))
    optimized_fp16_s = float(sum(float(row["optimized_fp16_baseline_s"]) for row in rows))
    dynamic_w8a8_s = float(sum(float(row["dynamic_w8a8_s"]) for row in rows))
    static_w8a8_s = float(sum(float(row["static_w8a8_s"]) for row in rows))
    return {
        "fp16_four_projection_s": fp16_s,
        "optimized_fp16_four_projection_s": optimized_fp16_s,
        "dynamic_w8a8_four_projection_s": dynamic_w8a8_s,
        "dynamic_four_projection_speedup": float(optimized_fp16_s / dynamic_w8a8_s),
        "static_w8a8_four_projection_s": static_w8a8_s,
        "static_four_projection_speedup": float(optimized_fp16_s / static_w8a8_s),
        "estimated_fp16_27_layer_projection_s": float(27 * fp16_s),
        "estimated_optimized_fp16_27_layer_projection_s": float(27 * optimized_fp16_s),
        "estimated_dynamic_w8a8_27_layer_projection_s": float(27 * dynamic_w8a8_s),
        "estimated_static_w8a8_27_layer_projection_s": float(27 * static_w8a8_s),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", choices=("fp16",), default="fp16")
    parser.add_argument("--rows", default="32,64,128,256,512,1024,2048")
    parser.add_argument(
        "--weight-layouts",
        default="nd_kn,nz_kn,nz_nk_transposed",
        help="comma-separated subset of nd_kn,nz_kn,nz_nk_transposed",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch_npu

    device = torch.device(args.device)
    dtype = torch.float16
    rows_values = [int(value) for value in str(args.rows).split(",") if value.strip()]
    layouts = [value.strip() for value in str(args.weight_layouts).split(",") if value.strip()]
    allowed_layouts = {"nd_kn", "nz_kn", "nz_nk_transposed"}
    unknown_layouts = set(layouts) - allowed_layouts
    if unknown_layouts:
        raise ValueError(f"unknown layouts: {sorted(unknown_layouts)}")
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")

    results: list[dict[str, Any]] = []
    aggregates: list[dict[str, Any]] = []
    for layout in layouts:
        for rows in rows_values:
            shape_rows = []
            for spec_index, spec in enumerate(VISION_LINEAR_SPECS):
                row = benchmark_one(
                    spec,
                    rows=rows,
                    device=device,
                    dtype=dtype,
                    weight_layout=layout,
                    warmup=int(args.warmup),
                    iterations=int(args.iterations),
                    seed=int(args.seed + rows * 10 + spec_index),
                )
                results.append(row)
                shape_rows.append(row)
                print(
                    f"W8A8_LINEAR layout={layout} rows={rows} name={spec.name} "
                    f"fp16_ms={1000.0 * row['fp16_linear_s']:.4f} "
                    f"fp16_grouped_ms={1000.0 * row['fp16_grouped_s']:.4f} "
                    f"baseline={row['optimized_fp16_baseline']} "
                    f"dynamic_ms={1000.0 * row['dynamic_w8a8_s']:.4f} "
                    f"dynamic_speedup={row['dynamic_speedup']:.3f} "
                    f"static_ms={1000.0 * row['static_w8a8_s']:.4f} "
                    f"static_speedup={row['static_speedup']:.3f} "
                    f"dynamic_cosine={row['dynamic_diff']['cosine']:.7f} "
                    f"static_cosine={row['static_diff']['cosine']:.7f}",
                    flush=True,
                )
            aggregate = {
                "rows": int(rows),
                "weight_layout": str(layout),
                **aggregate_layer(shape_rows),
            }
            aggregates.append(aggregate)
            print(
                f"W8A8_AGGREGATE layout={layout} rows={rows} "
                f"dynamic_speedup={aggregate['dynamic_four_projection_speedup']:.3f} "
                f"static_speedup={aggregate['static_four_projection_speedup']:.3f}",
                flush=True,
            )

    output = {
        "schema_version": 1,
        "experiment": "07_vision_prefill_optimization",
        "kind": "native_dynamic_w8a8_vision_linear_benchmark",
        "torch_version": str(torch.__version__),
        "torch_npu_version": str(torch_npu.__version__),
        "device": str(device),
        "device_name": str(
            torch.npu.get_device_name(torch.npu.current_device() if device.index is None else int(device.index))
        ),
        "dtype": str(dtype),
        "warmup": int(args.warmup),
        "iterations": int(args.iterations),
        "rows": rows_values,
        "weight_layouts": layouts,
        "quantization": {
            "weight": "offline_symmetric_per_output_channel_int8",
            "weight_scale_dtype": "float32",
            "output_dtype": "float16",
            "dynamic_activation": "symmetric_per_token_int8",
            "static_activation": "calibrated_symmetric_per_tensor_int8",
            "static_bias": "folded_int32",
            "timed_region": "activation_quantization_plus_npu_quant_matmul",
        },
        "results": results,
        "aggregates": aggregates,
    }
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"W8A8_OUTPUT={output_path}")


if __name__ == "__main__":
    main()
