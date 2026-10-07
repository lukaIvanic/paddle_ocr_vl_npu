#!/usr/bin/env python3
"""Compile an isolated shared-activation gate/up W8A8 pair on Ascend NPU."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from torch import nn

from local_modeling_qwen3_reranker import FRACTAL_NZ
from local_reranker_w8a8 import (
    W8A8GateUp,
    prepare_w8a8_weight_format,
    restore_w8a8_scale_dtypes,
)
from run_local_qwen3_reranker import _import_cache_compile


class DenseGateUp(nn.Module):
    def __init__(self, gate: nn.Linear, up: nn.Linear):
        super().__init__()
        self.gate = gate
        self.up = up

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.gate(hidden_states), self.up(hidden_states)


class QuantGateUp(nn.Module):
    def __init__(self, gate_up: W8A8GateUp):
        super().__init__()
        self.gate_up = gate_up

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.gate_up(hidden_states)


class QuantizeOnly(nn.Module):
    def __init__(self, input_scale: torch.Tensor):
        super().__init__()
        self.register_buffer("input_scale", input_scale.reshape(1).to(torch.float32))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        import torch_npu

        return torch_npu.npu_quantize(
            hidden_states,
            scales=self.input_scale,
            zero_points=None,
            dtype=torch.qint8,
            axis=0,
            div_mode=True,
        )


class PrequantizedGateUp(nn.Module):
    def __init__(self, gate_up: W8A8GateUp):
        super().__init__()
        self.gate_up = gate_up

    def forward(self, hidden_states_q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        input_scale = self.gate_up.gate_proj.static_input_scale.reshape(())
        return (
            self.gate_up.gate_proj.quant_matmul_from_quantized(hidden_states_q, input_scale),
            self.gate_up.up_proj.quant_matmul_from_quantized(hidden_states_q, input_scale),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--compile-cache-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument(
        "--weight-format",
        choices=("native", "fractal_nz", "fractal_nz_inference_doc"),
        default="fractal_nz",
    )
    return parser.parse_args()


def synchronize() -> None:
    torch.npu.synchronize()


def timed(fn) -> tuple[float, tuple[torch.Tensor, torch.Tensor]]:
    synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = fn()
    synchronize()
    return time.perf_counter() - started, output


def benchmark(fn, *, warmups: int, repeats: int) -> tuple[dict[str, float], tuple[torch.Tensor, torch.Tensor]]:
    first_call_s, output = timed(fn)
    for _ in range(max(0, warmups - 1)):
        _elapsed, output = timed(fn)
    measured = []
    for _ in range(repeats):
        elapsed, output = timed(fn)
        measured.append(elapsed)
    ordered = sorted(measured)
    return {
        "first_call_s": first_call_s,
        "median_s": statistics.median(measured),
        "mean_s": statistics.fmean(measured),
        "min_s": min(measured),
        "p90_s": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
        "max_s": max(measured),
    }, output


def main() -> None:
    args = parse_args()
    if args.tokens <= 0 or args.warmups <= 0 or args.repeats <= 0:
        raise ValueError("tokens, warmups, and repeats must be positive")

    import torch_npu
    from torch_npu.dynamo.torchair.configs.compiler_config import CompilerConfig

    required_ops = ("npu_quantize", "npu_quant_matmul", "npu_trans_quant_param", "npu_format_cast")
    missing_ops = [name for name in required_ops if not callable(getattr(torch_npu, name, None))]
    if missing_ops:
        raise RuntimeError(f"required torch_npu operations are unavailable: {missing_ops}")

    torch.npu.config.allow_internal_format = True
    device = torch.device(args.device)
    torch.npu.set_device(device)
    torch.npu.set_compile_mode(jit_compile=False)

    config = json.loads((args.model_dir / "config.json").read_text())
    hidden_size = int(config["hidden_size"])
    intermediate_size = int(config["intermediate_size"])
    torch.manual_seed(310)
    gate = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=torch.float16)
    up = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=torch.float16)
    with torch.no_grad():
        gate.weight.normal_(mean=0.0, std=0.02)
        up.weight.normal_(mean=0.0, std=0.02)

    quantization_started = time.perf_counter()
    quant_gate_up = W8A8GateUp(gate, up, out_dtype=torch.float16)
    weight_quantization_s = time.perf_counter() - quantization_started

    dense = DenseGateUp(gate, up).to(device=device, dtype=torch.float16).eval()
    quant = QuantGateUp(quant_gate_up).to(device=device).eval()
    restore_w8a8_scale_dtypes(quant)
    for linear in (dense.gate, dense.up):
        linear.weight.data = torch_npu.npu_format_cast(linear.weight.data, FRACTAL_NZ)
    quant_format = prepare_w8a8_weight_format(quant, requested=args.weight_format)

    hidden_states = torch.randn(
        args.tokens,
        hidden_size,
        device=device,
        dtype=torch.float16,
    )
    input_scale = hidden_states.abs().amax().to(torch.float32).reshape(1) / 127.0
    quant.gate_up.gate_proj.set_static_input_scale(input_scale)
    quant.gate_up.up_proj.set_static_input_scale(input_scale)
    quantize_only = QuantizeOnly(input_scale).to(device=device).eval()
    prequantized_gate_up = PrequantizedGateUp(quant.gate_up).to(device=device).eval()
    with torch.inference_mode():
        hidden_states_q = quantize_only(hidden_states)
    synchronize()

    args.compile_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_compile = _import_cache_compile()
    compiler_config = CompilerConfig()
    dense_compiled = cache_compile(
        dense.forward,
        config=compiler_config,
        dynamic=False,
        cache_dir=str(args.compile_cache_dir / "dense"),
        ge_cache=True,
        fullgraph=True,
    )
    quant_compiled = cache_compile(
        quant.forward,
        config=compiler_config,
        dynamic=False,
        cache_dir=str(args.compile_cache_dir / "w8a8"),
        ge_cache=True,
        fullgraph=True,
    )
    quantize_compiled = cache_compile(
        quantize_only.forward,
        config=compiler_config,
        dynamic=False,
        cache_dir=str(args.compile_cache_dir / "quantize_only"),
        ge_cache=True,
        fullgraph=True,
    )
    prequantized_compiled = cache_compile(
        prequantized_gate_up.forward,
        config=compiler_config,
        dynamic=False,
        cache_dir=str(args.compile_cache_dir / "prequantized_gate_up"),
        ge_cache=True,
        fullgraph=True,
    )

    dense_timing, dense_output = benchmark(
        lambda: dense_compiled(hidden_states),
        warmups=args.warmups,
        repeats=args.repeats,
    )
    quant_timing, quant_output = benchmark(
        lambda: quant_compiled(hidden_states),
        warmups=args.warmups,
        repeats=args.repeats,
    )
    quantize_timing, _quantized_output = benchmark(
        lambda: quantize_compiled(hidden_states),
        warmups=args.warmups,
        repeats=args.repeats,
    )
    prequantized_timing, prequantized_output = benchmark(
        lambda: prequantized_compiled(hidden_states_q),
        warmups=args.warmups,
        repeats=args.repeats,
    )
    gate_diff = (quant_output[0].float() - dense_output[0].float()).abs()
    up_diff = (quant_output[1].float() - dense_output[1].float()).abs()
    result = {
        "environment": {
            "device": torch.npu.get_device_name(device),
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
        },
        "shape": {
            "tokens": args.tokens,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
        },
        "operator_registration": {name: True for name in required_ops},
        "weight_quantization_s": weight_quantization_s,
        "quant_weight_format": quant_format,
        "dense": dense_timing,
        "w8a8": quant_timing,
        "quantize_only": quantize_timing,
        "prequantized_gate_up": prequantized_timing,
        "estimated_quantize_plus_prequantized_s": (
            quantize_timing["median_s"] + prequantized_timing["median_s"]
        ),
        "speedup": dense_timing["median_s"] / quant_timing["median_s"],
        "output_diff": {
            "gate_max_abs": float(gate_diff.max().cpu()),
            "gate_mean_abs": float(gate_diff.mean().cpu()),
            "up_max_abs": float(up_diff.max().cpu()),
            "up_mean_abs": float(up_diff.mean().cpu()),
            "prequantized_gate_max_abs": float(
                (prequantized_output[0].float() - quant_output[0].float()).abs().max().cpu()
            ),
            "prequantized_up_max_abs": float(
                (prequantized_output[1].float() - quant_output[1].float()).abs().max().cpu()
            ),
        },
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2) + "\n")
    print("W8A8_OP_PROBE " + json.dumps(result, sort_keys=True), flush=True)
    print(f"OUTPUT_JSON {args.json_out}", flush=True)


if __name__ == "__main__":
    main()
