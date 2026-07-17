"""Native Ascend W8A8 Linear helpers for the PaddleOCR-VL vision encoder."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


FRACTAL_NZ = 29
VISION_LINEAR_QUANTIZATION_CHOICES = (
    "none",
    "w8a8_dynamic",
    "w8a8_static",
    "w8a8_fused_pertoken",
    "a16w8_weight_only",
)
VISION_LINEAR_SITES = ("qkv", "out_proj", "fc1", "fc2")
W8A8_WEIGHT_LAYOUT_CHOICES = ("auto", "nd_kn", "nz_kn", "nz_nk_transposed")


def is_310p_device(device: torch.device) -> bool:
    if device.type != "npu":
        return False
    device_index = torch.npu.current_device() if device.index is None else int(device.index)
    name = str(torch.npu.get_device_name(device_index)).lower()
    return "310p" in name or "300i" in name or "200i" in name


def resolve_weight_layout(device: torch.device, requested: str) -> str:
    if requested not in W8A8_WEIGHT_LAYOUT_CHOICES:
        raise ValueError(f"unsupported W8A8 weight layout={requested!r}")
    if requested != "auto":
        return requested
    return "nz_nk_transposed" if is_310p_device(device) else "nz_kn"


def quantize_weight_per_output_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weight_fp32 = weight.detach().float()
    scale = weight_fp32.abs().amax(dim=1).clamp_min(torch.finfo(torch.float32).eps) / 127.0
    quantized = torch.round(weight_fp32 / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return quantized, scale


def pack_weight(weight_nk: torch.Tensor, *, layout: str) -> torch.Tensor:
    import torch_npu

    if layout == "nd_kn":
        return weight_nk.transpose(0, 1).contiguous()
    if layout == "nz_kn":
        return torch_npu.npu_format_cast(weight_nk.transpose(0, 1).contiguous(), FRACTAL_NZ)
    if layout == "nz_nk_transposed":
        return torch_npu.npu_format_cast(weight_nk.contiguous(), FRACTAL_NZ).transpose(0, 1)
    raise ValueError(f"unsupported resolved W8A8 weight layout={layout!r}")


class PackedW8A8Linear(torch.nn.Module):
    """Prepacked INT8 Linear using torch-npu quantize + QuantMatmul operators."""

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        *,
        mode: str,
        weight_layout: str,
        static_input_scale: float | None = None,
    ) -> None:
        super().__init__()
        if mode not in {"w8a8_dynamic", "w8a8_static"}:
            raise ValueError(f"PackedW8A8Linear requires a W8A8 mode, got {mode!r}")
        if weight.device.type != "npu":
            raise ValueError("PackedW8A8Linear requires weights already resident on NPU")
        self.mode = str(mode)
        self.weight_layout = str(weight_layout)
        self.in_features = int(weight.shape[1])
        self.out_features = int(weight.shape[0])

        weight_int8_nk, weight_scale = quantize_weight_per_output_channel(weight)
        self.register_buffer("weight_int8", pack_weight(weight_int8_nk, layout=weight_layout), persistent=False)
        self.register_buffer("weight_scale", weight_scale.to(dtype=torch.float32), persistent=False)
        if bias is None:
            self.fp_bias = None
        else:
            self.register_buffer("fp_bias", bias.detach().contiguous(), persistent=False)

        self.static_input_scale_value = None if static_input_scale is None else float(static_input_scale)
        if self.mode == "w8a8_static":
            if static_input_scale is None or not float(static_input_scale) > 0:
                raise ValueError("static W8A8 requires a positive calibrated input scale")
            import torch_npu

            input_scale_scalar = torch.tensor(
                float(static_input_scale), device=weight.device, dtype=torch.float32
            )
            self.register_buffer(
                "input_scale",
                input_scale_scalar.to(dtype=weight.dtype).repeat(self.in_features),
                persistent=False,
            )
            self.register_buffer(
                "input_scale_reciprocal",
                (1.0 / input_scale_scalar).to(dtype=weight.dtype).repeat(self.in_features),
                persistent=False,
            )
            self.register_buffer(
                "input_offset",
                torch.zeros(self.in_features, device=weight.device, dtype=weight.dtype),
                persistent=False,
            )
            dequant_scale = input_scale_scalar * self.weight_scale
            self.register_buffer(
                "dequant_scale",
                torch_npu.npu_trans_quant_param(dequant_scale.contiguous()),
                persistent=False,
            )
            if bias is None:
                quant_bias = torch.zeros(self.out_features, device=weight.device, dtype=torch.int32)
            else:
                quant_bias = torch.round(bias.detach().float() / dequant_scale).to(torch.int32)
            self.register_buffer("quant_bias", quant_bias.contiguous(), persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        leading_shape = tuple(hidden_states.shape[:-1])
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        if self.mode == "w8a8_dynamic":
            quantized_x, pertoken_scale = torch.ops.npu.npu_dynamic_quant(flat)
            output = torch.ops.npu.npu_quant_matmul(
                quantized_x,
                self.weight_int8,
                self.weight_scale,
                pertoken_scale=pertoken_scale,
                bias=self.fp_bias,
                output_dtype=hidden_states.dtype,
            )
        else:
            quantized_x = torch.ops.npu.npu_quantize(
                flat,
                self.input_scale_reciprocal,
                self.input_offset,
                torch.qint8,
                axis=-1,
                div_mode=False,
            )
            output = torch.ops.npu.npu_quant_matmul(
                quantized_x,
                self.weight_int8,
                self.dequant_scale,
                bias=self.quant_bias,
                output_dtype=hidden_states.dtype,
            )
        return output.reshape(*leading_shape, self.out_features)

    def metadata(self) -> dict[str, Any]:
        import torch_npu

        return {
            "mode": self.mode,
            "in_features": int(self.in_features),
            "out_features": int(self.out_features),
            "weight_layout": self.weight_layout,
            "packed_weight_format": int(torch_npu.get_npu_format(self.weight_int8)),
            "static_input_scale": self.static_input_scale_value,
            "static_quantize_div_mode": False if self.mode == "w8a8_static" else None,
        }


class PackedFusedPertokenW8A8Linear(torch.nn.Module):
    """INT8 weights with fused per-token activation quantize/matmul/dequant."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None) -> None:
        super().__init__()
        if weight.device.type != "npu":
            raise ValueError("PackedFusedPertokenW8A8Linear requires NPU weights")
        self.in_features = int(weight.shape[1])
        self.out_features = int(weight.shape[0])
        weight_int8_nk, weight_scale = quantize_weight_per_output_channel(weight)
        # QuantMatmulDequant takes the logical transposed Linear weight [N, K].
        self.register_buffer("weight_int8_nk", weight_int8_nk.contiguous(), persistent=False)
        self.register_buffer("weight_scale", weight_scale.to(dtype=torch.float32), persistent=False)
        if bias is None:
            self.fp_bias = None
        else:
            self.register_buffer("fp_bias", bias.detach().contiguous(), persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        leading_shape = tuple(hidden_states.shape[:-1])
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        output = torch.ops.npu.npu_quant_matmul_dequant(
            flat,
            self.weight_int8_nk,
            self.weight_scale,
            quant_mode="pertoken",
        )
        if self.fp_bias is not None:
            output = output + self.fp_bias
        return output.reshape(*leading_shape, self.out_features)

    def metadata(self) -> dict[str, Any]:
        import torch_npu

        return {
            "mode": "w8a8_fused_pertoken",
            "in_features": int(self.in_features),
            "out_features": int(self.out_features),
            "weight_layout": "nd_nk_transposed_linear_weight",
            "packed_weight_format": int(torch_npu.get_npu_format(self.weight_int8_nk)),
            "activation_quantization": "fused_per_token",
            "bias_application": "fp16_add_after_fused_op" if self.fp_bias is not None else "none",
        }


class PackedA16W8Linear(torch.nn.Module):
    """FP16 activations with prepacked per-channel INT8 weights."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None) -> None:
        super().__init__()
        if weight.device.type != "npu":
            raise ValueError("PackedA16W8Linear requires NPU weights")
        self.in_features = int(weight.shape[1])
        self.out_features = int(weight.shape[0])
        weight_int8_nk, weight_scale = quantize_weight_per_output_channel(weight)
        self.register_buffer("weight_int8_nk", weight_int8_nk.contiguous(), persistent=False)
        self.register_buffer("antiquant_scale", weight_scale.to(dtype=weight.dtype), persistent=False)
        if bias is None:
            self.fp_bias = None
        else:
            self.register_buffer("fp_bias", bias.detach().contiguous(), persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        leading_shape = tuple(hidden_states.shape[:-1])
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        output = torch.ops.npu.npu_weight_quant_batchmatmul(
            flat,
            self.weight_int8_nk.transpose(-1, -2),
            self.antiquant_scale,
            bias=self.fp_bias,
        )
        return output.reshape(*leading_shape, self.out_features)

    def metadata(self) -> dict[str, Any]:
        import torch_npu

        return {
            "mode": "a16w8_weight_only",
            "in_features": int(self.in_features),
            "out_features": int(self.out_features),
            "weight_layout": "nd_nk_transposed_at_call",
            "packed_weight_format": int(torch_npu.get_npu_format(self.weight_int8_nk)),
            "activation_quantization": "none_fp16",
        }


def packed_from_linears(
    linears: Sequence[torch.nn.Linear],
    *,
    mode: str,
    weight_layout: str,
    static_input_scale: float | None,
) -> PackedW8A8Linear:
    if not linears:
        raise ValueError("at least one Linear is required")
    weight = torch.cat([linear.weight.detach() for linear in linears], dim=0).contiguous()
    if any(linear.bias is None for linear in linears):
        if not all(linear.bias is None for linear in linears):
            raise ValueError("cannot combine a mixture of biased and bias-free Linears")
        bias = None
    else:
        bias = torch.cat([linear.bias.detach() for linear in linears if linear.bias is not None], dim=0).contiguous()
    if mode == "w8a8_fused_pertoken":
        return PackedFusedPertokenW8A8Linear(weight, bias)
    if mode == "a16w8_weight_only":
        return PackedA16W8Linear(weight, bias)
    return PackedW8A8Linear(
        weight,
        bias,
        mode=mode,
        weight_layout=weight_layout,
        static_input_scale=static_input_scale,
    )
