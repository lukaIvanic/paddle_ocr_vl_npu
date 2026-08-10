from __future__ import annotations

import torch
from torch import nn

from local_modeling_qwen3_reranker import FRACTAL_NZ, LocalQwen3RerankerMLP, linear_tokenwise


def quantize_per_output_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weight_fp32 = weight.detach().to(torch.float32)
    scale = weight_fp32.abs().amax(dim=-1).clamp_min(1e-6) / 127.0
    weight_q = torch.round(weight_fp32 / scale.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    return weight_q.contiguous(), scale.to(torch.float16)


class W8A8Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, *, out_dtype: torch.dtype):
        super().__init__()
        if out_dtype is not torch.float16:
            raise ValueError("W8A8Linear is FP16-only")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.out_dtype = out_dtype
        self.register_buffer("weight_q", torch.empty(out_features, in_features, dtype=torch.int8))
        self.register_buffer("weight_scale", torch.ones(out_features, dtype=torch.float16))
        self.register_buffer("static_input_scale", torch.ones(1, dtype=torch.float32))
        self.register_buffer("static_packed_deq_scale", torch.empty(out_features, dtype=torch.int64))
        self.use_static_input_scale = False
        self.use_static_packed_deq_scale = False
        self.weight_is_matmul_ready = False
        self.weight_requires_graph_transpose = False
        self.collect_input_scale_stats = False
        self._observed_input_scale = 0.0

    @classmethod
    def from_linear(cls, linear: nn.Linear, *, out_dtype: torch.dtype) -> "W8A8Linear":
        if linear.bias is not None:
            raise ValueError("Qwen3 reranker FFN linears are expected to be bias-free")
        quantized = cls(linear.in_features, linear.out_features, out_dtype=out_dtype)
        weight_q, weight_scale = quantize_per_output_channel(linear.weight)
        quantized.weight_q.copy_(weight_q)
        quantized.weight_scale.copy_(weight_scale)
        return quantized

    def begin_input_scale_calibration(self) -> None:
        self.collect_input_scale_stats = True
        self._observed_input_scale = 0.0

    def finish_input_scale_calibration(self) -> None:
        scale = max(self._observed_input_scale, 1e-6 / 127.0)
        self.set_static_input_scale(torch.tensor([scale], dtype=torch.float32, device=self.static_input_scale.device))
        self.collect_input_scale_stats = False

    def set_static_input_scale(self, input_scale: torch.Tensor) -> None:
        self.static_input_scale.copy_(input_scale.reshape(1).to(dtype=torch.float32, device=self.static_input_scale.device))
        self.use_static_packed_deq_scale = False
        if self.weight_scale.device.type == "npu":
            import torch_npu

            deq_scale = self.weight_scale.to(torch.float32) * self.static_input_scale.reshape(())
            self.static_packed_deq_scale.copy_(torch_npu.npu_trans_quant_param(deq_scale, None))
            self.use_static_packed_deq_scale = True
        self.use_static_input_scale = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import torch_npu

        shape = x.shape[:-1]
        x_fp16 = x.reshape(-1, self.in_features).to(torch.float16)
        if self.use_static_input_scale:
            input_scale = self.static_input_scale.reshape(())
        else:
            input_scale = x_fp16.abs().amax().clamp_min(1e-6) / 127.0
        if self.collect_input_scale_stats:
            self._observed_input_scale = max(self._observed_input_scale, float(input_scale.detach().cpu().item()))

        x_q = torch_npu.npu_quantize(
            x_fp16,
            scales=input_scale.reshape(1).to(torch.float32),
            zero_points=None,
            dtype=torch.qint8,
            axis=0,
            div_mode=True,
        )
        out = self.quant_matmul_from_quantized(x_q, input_scale)
        return out.reshape(*shape, self.out_features)

    def quant_matmul_from_quantized(self, x_q: torch.Tensor, input_scale: torch.Tensor) -> torch.Tensor:
        import torch_npu

        if self.use_static_packed_deq_scale:
            deq_scale = self.static_packed_deq_scale
        else:
            deq_scale = self.weight_scale.to(torch.float32) * input_scale.to(torch.float32)
            deq_scale = torch_npu.npu_trans_quant_param(deq_scale, None)
        weight = (
            self.weight_q.transpose(0, 1)
            if self.weight_requires_graph_transpose or not self.weight_is_matmul_ready
            else self.weight_q
        )
        return torch_npu.npu_quant_matmul(
            x_q,
            weight,
            scale=deq_scale,
            bias=None,
            output_dtype=self.out_dtype,
        )


class W8A8GateUp(nn.Module):
    def __init__(self, gate_proj: nn.Linear, up_proj: nn.Linear, *, out_dtype: torch.dtype):
        super().__init__()
        if gate_proj.in_features != up_proj.in_features:
            raise ValueError("gate/up input sizes must match")
        if gate_proj.out_features != up_proj.out_features:
            raise ValueError("gate/up output sizes must match")
        self.gate_proj = W8A8Linear.from_linear(gate_proj, out_dtype=out_dtype)
        self.up_proj = W8A8Linear.from_linear(up_proj, out_dtype=out_dtype)

    def begin_input_scale_calibration(self) -> None:
        self.gate_proj.begin_input_scale_calibration()

    def finish_input_scale_calibration(self) -> None:
        self.gate_proj.finish_input_scale_calibration()
        self.up_proj.set_static_input_scale(self.gate_proj.static_input_scale)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        import torch_npu

        shape = x.shape[:-1]
        x_fp16 = x.reshape(-1, self.gate_proj.in_features).to(torch.float16)
        if self.gate_proj.use_static_input_scale:
            input_scale = self.gate_proj.static_input_scale.reshape(())
        else:
            input_scale = x_fp16.abs().amax().clamp_min(1e-6) / 127.0
        if self.gate_proj.collect_input_scale_stats:
            self.gate_proj._observed_input_scale = max(
                self.gate_proj._observed_input_scale,
                float(input_scale.detach().cpu().item()),
            )
        x_q = torch_npu.npu_quantize(
            x_fp16,
            scales=input_scale.reshape(1).to(torch.float32),
            zero_points=None,
            dtype=torch.qint8,
            axis=0,
            div_mode=True,
        )
        gate = self.gate_proj.quant_matmul_from_quantized(x_q, input_scale)
        up = self.up_proj.quant_matmul_from_quantized(x_q, input_scale)
        return gate.reshape(*shape, self.gate_proj.out_features), up.reshape(*shape, self.up_proj.out_features)


class W8A8MLP(nn.Module):
    def __init__(self, dense_mlp: LocalQwen3RerankerMLP, *, out_dtype: torch.dtype):
        super().__init__()
        self.gate_up = W8A8GateUp(dense_mlp.gate_proj, dense_mlp.up_proj, out_dtype=out_dtype)
        self.down_proj = W8A8Linear.from_linear(dense_mlp.down_proj, out_dtype=out_dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F

        gate, up = self.gate_up(hidden_states)
        return self.down_proj(F.silu(gate) * up)


class W8A8GateUpOnlyMLP(nn.Module):
    """Quantize the two wide FFN projections and keep the down projection FP16."""

    def __init__(self, dense_mlp: LocalQwen3RerankerMLP, *, out_dtype: torch.dtype):
        super().__init__()
        self.gate_up = W8A8GateUp(dense_mlp.gate_proj, dense_mlp.up_proj, out_dtype=out_dtype)
        self.down_proj = dense_mlp.down_proj

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F

        gate, up = self.gate_up(hidden_states)
        return linear_tokenwise(self.down_proj, F.silu(gate) * up)


def quantize_reranker_gate_up_inplace(module: nn.Module, *, out_dtype: torch.dtype) -> None:
    for parent in module.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, LocalQwen3RerankerMLP):
                setattr(parent, name, W8A8GateUpOnlyMLP(child, out_dtype=out_dtype))


def quantize_reranker_ffn_inplace(module: nn.Module, *, out_dtype: torch.dtype) -> None:
    for parent in module.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, LocalQwen3RerankerMLP):
                setattr(parent, name, W8A8MLP(child, out_dtype=out_dtype))


def quantize_reranker_all_linears_inplace(
    module: nn.Module,
    *,
    out_dtype: torch.dtype,
    prefix: str = "",
) -> None:
    for name, child in list(module.named_children()):
        child_prefix = f"{prefix}.{name}" if prefix else name
        if child_prefix == "lm_head":
            continue
        if isinstance(child, LocalQwen3RerankerMLP):
            setattr(module, name, W8A8MLP(child, out_dtype=out_dtype))
            continue
        if isinstance(child, nn.Linear):
            setattr(module, name, W8A8Linear.from_linear(child, out_dtype=out_dtype))
            continue
        quantize_reranker_all_linears_inplace(child, out_dtype=out_dtype, prefix=child_prefix)


def iter_w8a8_linears(module: nn.Module):
    for child in module.modules():
        if isinstance(child, W8A8Linear):
            yield child


def iter_w8a8_gate_up(module: nn.Module):
    for child in module.modules():
        if isinstance(child, W8A8GateUp):
            yield child


def restore_w8a8_scale_dtypes(module: nn.Module) -> None:
    """Keep quantizer scales FP32 after the surrounding model is cast to FP16."""
    for linear in iter_w8a8_linears(module):
        linear.static_input_scale = linear.static_input_scale.to(torch.float32)


def prepare_w8a8_weight_format(module: nn.Module, *, requested: str) -> dict[str, object]:
    """Prepare INT8 weights once for the product-specific QuantMatmul contract.

    Atlas 310P and training-series devices require a different ordering of the
    format cast and logical transpose. Both branches preserve the public
    QuantMatmul [M, K] x [K, N] contract.
    """
    if requested not in {"native", "fractal_nz", "fractal_nz_inference_doc"}:
        raise ValueError(f"unsupported W8A8 weight format {requested!r}")
    linears = list(iter_w8a8_linears(module))
    if any(linear.weight_q.device.type != "npu" for linear in linears):
        raise RuntimeError("W8A8 weight preparation requires NPU-resident weights")

    import torch_npu

    device_name = torch.npu.get_device_name(linears[0].weight_q.device) if linears else ""
    is_310p = "310P" in device_name.upper()
    before = [int(torch_npu.get_npu_format(linear.weight_q)) for linear in linears]
    for linear in linears:
        if linear.weight_is_matmul_ready:
            continue
        if is_310p and requested in {"fractal_nz", "fractal_nz_inference_doc"}:
            # Keep the formatted buffer in physical/logical [N,K] form. The
            # compiled forward must express the transpose to [K,N], matching
            # Huawei's Atlas-inference high-performance example exactly.
            prepared = torch_npu.npu_format_cast(
                linear.weight_q.contiguous(),
                FRACTAL_NZ,
            )
            linear.weight_requires_graph_transpose = True
        elif requested == "fractal_nz_inference_doc":
            raise ValueError("fractal_nz_inference_doc is only for Atlas inference products")
        elif requested == "fractal_nz":
            prepared = torch_npu.npu_format_cast(
                linear.weight_q.transpose(0, 1).contiguous(),
                FRACTAL_NZ,
            )
        else:
            prepared = linear.weight_q.transpose(0, 1).contiguous()
        linear.weight_q = prepared
        linear.weight_is_matmul_ready = True
    after = [int(torch_npu.get_npu_format(linear.weight_q)) for linear in linears]
    return {
        "requested": requested,
        "effective_mode": requested,
        "device_name": device_name,
        "device_layout_branch": (
            "native_logical_k_n"
            if requested == "native"
            else "310p_n_k_format_then_graph_transpose"
            if requested in {"fractal_nz", "fractal_nz_inference_doc"} and is_310p
            else "a2_logical_k_n_then_format"
        ),
        "quant_linear_count": len(linears),
        "before_format_histogram": {str(code): before.count(code) for code in sorted(set(before))},
        "after_format_histogram": {str(code): after.count(code) for code in sorted(set(after))},
        "prepared_weight_metadata": [
            {
                "shape": list(linear.weight_q.shape),
                "stride": list(linear.weight_q.stride()),
                "contiguous": bool(linear.weight_q.is_contiguous()),
                "graph_transpose": bool(linear.weight_requires_graph_transpose),
            }
            for linear in linears
        ],
        "all_matmul_ready": all(linear.weight_is_matmul_ready for linear in linears),
    }


def calibrate_w8a8_input_scales(model: nn.Module, forward_fn) -> None:
    gate_up_modules = list(iter_w8a8_gate_up(model))
    linears = [
        linear
        for linear in iter_w8a8_linears(model)
        if not any(linear is gate_up.gate_proj or linear is gate_up.up_proj for gate_up in gate_up_modules)
    ]
    for gate_up in gate_up_modules:
        gate_up.begin_input_scale_calibration()
    for linear in linears:
        linear.begin_input_scale_calibration()
    forward_fn()
    for gate_up in gate_up_modules:
        gate_up.finish_input_scale_calibration()
    for linear in linears:
        linear.finish_input_scale_calibration()
