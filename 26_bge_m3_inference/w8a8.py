"""Unfused static W8A8, following experiment 13's Qwen projection path."""
from __future__ import annotations

import torch
from torch import nn


@torch.inference_mode()
def calibrate(model: nn.Module, batches: list[dict[str, torch.Tensor]]) -> dict[str, float]:
    """Observe FP16 linear inputs once; no reductions or hooks in the hot path."""
    maxima, hooks = {}, []

    def observer(name):
        def observe(_module, inputs):
            value = inputs[0].detach().abs().amax().float()
            maxima[name] = torch.maximum(maxima[name], value) if name in maxima else value
        return observe

    try:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_pre_hook(observer(name)))
        for batch in batches:
            model(**batch)
    finally:
        for hook in hooks:
            hook.remove()
    return {name: max(float(value.cpu()), 1e-6) / 127.0 for name, value in maxima.items()}


class W8A8Linear(nn.Module):
    def __init__(self, linear: nn.Linear, input_scale: float):
        super().__init__()
        import torch_npu
        if linear.weight.device.type != "npu" or linear.weight.dtype != torch.float16:
            raise ValueError("W8A8 conversion requires NPU FP16 weights")
        weight = linear.weight.detach().float()
        weight_scale = weight.abs().amax(dim=1).clamp_min(1e-6) / 127.0
        quantized = (weight / weight_scale[:, None]).round().clamp(-127, 127).to(torch.int8)
        self.in_features, self.out_features = linear.in_features, linear.out_features
        # Match the product-specific layout established in the Qwen experiments.
        self.graph_transpose = "310P" in torch.npu.get_device_name(linear.weight.device).upper()
        stored = quantized if self.graph_transpose else quantized.T.contiguous()
        self.register_buffer("weight_q", torch_npu.npu_format_cast(stored, 29))
        self.register_buffer("input_scale", torch.tensor([input_scale], device=weight.device, dtype=torch.float32))
        self.register_buffer("deq_scale", torch_npu.npu_trans_quant_param(weight_scale * input_scale, None))
        # Keep the original floating bias. QuantMatmul INT32 bias would round it.
        self.register_buffer("bias", None if linear.bias is None else linear.bias.detach().clone())

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        import torch_npu
        return torch_npu.npu_quantize(x.reshape(-1, self.in_features), scales=self.input_scale,
                                     zero_points=None, dtype=torch.qint8, axis=0, div_mode=True)

    def matmul(self, quantized: torch.Tensor) -> torch.Tensor:
        import torch_npu
        weight = self.weight_q.T if self.graph_transpose else self.weight_q
        result = torch_npu.npu_quant_matmul(quantized, weight, scale=self.deq_scale,
                                          bias=None, output_dtype=torch.float16)
        return result if self.bias is None else result + self.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.matmul(self.quantize(x)).reshape(*x.shape[:-1], self.out_features)


class SharedQKV(nn.Module):
    def __init__(self, attention, scales: dict[str, float], prefix: str):
        super().__init__()
        names = ("query", "key", "value")
        scale = scales[prefix + ".query"]
        if any(scales[prefix + "." + name] != scale for name in names):
            raise ValueError("Q/K/V calibration must observe the same input")
        self.projections = nn.ModuleList([W8A8Linear(getattr(attention, name), scale) for name in names])

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, ...]:
        quantized = self.projections[0].quantize(hidden)
        return tuple(p.matmul(quantized).reshape(*hidden.shape[:-1], p.out_features) for p in self.projections)


@torch.inference_mode()
def convert(model, scales: dict[str, float], mode: str) -> dict:
    """FFN-only or all six encoder linears; norms/embeddings/attention stay FP16."""
    if mode not in {"ffn_w8a8", "full_w8a8"}:
        raise ValueError(f"Unknown W8A8 mode: {mode}")
    for index, layer in enumerate(model.encoder.layer):
        prefix = f"encoder.layer.{index}"
        layer.intermediate.dense = W8A8Linear(layer.intermediate.dense, scales[prefix + ".intermediate.dense"])
        layer.output.dense = W8A8Linear(layer.output.dense, scales[prefix + ".output.dense"])
        if mode == "full_w8a8":
            attention = layer.attention.self
            attention.qkv = SharedQKV(attention, scales, prefix + ".attention.self")
            attention.query = attention.key = attention.value = None
            layer.attention.output.dense = W8A8Linear(layer.attention.output.dense,
                                                       scales[prefix + ".attention.output.dense"])
    quantized = sum(isinstance(module, W8A8Linear) for module in model.modules())
    expected = len(model.encoder.layer) * (2 if mode == "ffn_w8a8" else 6)
    if quantized != expected:
        raise RuntimeError(f"Expected {expected} quantized linears, got {quantized}")
    return {"mode": mode, "quantized_linears": quantized,
            "activation_quantizations_per_layer": 2 if mode == "ffn_w8a8" else 4,
            "weight_quantization": "symmetric_per_output_channel", "activation_quantization": "static_per_tensor",
            "bias": "original_fp16_after_dequantization", "weight_format": "FRACTAL_NZ"}


@torch.inference_mode()
def prepare_dense_weights(model: nn.Module) -> int:
    """Give all lanes the same NZ optimization for their remaining FP16 linears."""
    import torch_npu
    count = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            module.weight = nn.Parameter(torch_npu.npu_format_cast(module.weight.detach(), 29), requires_grad=False)
            count += 1
    return count
