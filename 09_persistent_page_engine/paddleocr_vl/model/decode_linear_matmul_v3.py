"""B1 decode Linear identity lowered to CANN's AscendC MatMulV3."""

from __future__ import annotations

import importlib
from typing import Any

import torch
import torch.nn.functional as F

from .compile_utils import import_torchair


GE_OP_NAME = "MatMulV3"
PYTORCH_OP_NAME = "paddleocr_vl::decode_linear_matmul_v3"


@torch.library.custom_op(PYTORCH_OP_NAME, mutates_args=())
def _decode_linear_matmul_v3(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Eager reference; TorchAir lowers this identity to AscendC MatMulV3."""
    return F.linear(x, weight)


@_decode_linear_matmul_v3.register_fake
def _decode_linear_matmul_v3_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(
        (*x.shape[:-1], weight.shape[0]),
        dtype=x.dtype,
        device=x.device,
    )


_CONVERTER_REGISTERED = False


def register_decode_linear_matmul_v3_converter() -> None:
    """Lower the explicit graph identity to installed AscendC MatMulV3."""
    global _CONVERTER_REGISTERED
    if _CONVERTER_REGISTERED:
        return

    torchair, _CompilerConfig = import_torchair()
    converter_module = importlib.import_module(
        f"{torchair.__name__}._ge_concrete_graph.fx2ge_converter"
    )
    ge_module = importlib.import_module(f"{torchair.__name__}.ge")
    ge_attr = importlib.import_module("torchair.ge.attr")
    register_converter = converter_module.register_fx_node_ge_converter
    ge_custom_op = ge_module.custom_op
    op = torch.ops.paddleocr_vl.decode_linear_matmul_v3.default

    @register_converter(op)
    def _convert_decode_linear_matmul_v3(
        x: Any,
        weight: Any,
        meta_outputs: Any = None,
    ) -> Any:
        del meta_outputs
        return ge_custom_op(
            GE_OP_NAME,
            inputs={"x1": x, "x2": weight},
            attrs={
                "transpose_x1": ge_attr.Bool(False),
                "transpose_x2": ge_attr.Bool(True),
                "offset_x": ge_attr.Int(0),
                "enable_hf32": ge_attr.Bool(False),
            },
            outputs=["y"],
        )

    _CONVERTER_REGISTERED = True


def decode_linear_matmul_v3(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Apply a no-bias FP16 Linear through installed AscendC MatMulV3."""
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError("decode_linear_matmul_v3 requires rank-2 x and weight")
    if x.shape[0] != 1 or x.shape[1] != weight.shape[1]:
        raise ValueError(
            "decode_linear_matmul_v3 requires x[1,K] and weight[N,K]"
        )
    if x.dtype != torch.float16 or weight.dtype != torch.float16:
        raise ValueError("decode_linear_matmul_v3 requires FP16 tensors")
    return _decode_linear_matmul_v3(x, weight)
