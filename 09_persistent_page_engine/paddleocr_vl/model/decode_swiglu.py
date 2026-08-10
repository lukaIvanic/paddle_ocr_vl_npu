"""Fixed-shape Paddle decoder SwiGLU backed by an independent AIV op."""

from __future__ import annotations

import importlib
from typing import Any

import torch
import torch.nn.functional as F

from .compile_utils import import_torchair


PYTORCH_OP_NAME = "paddleocr_vl::decode_swiglu_v1"
GE_OP_NAME = "PaddleDecodeSwiGluV1"


@torch.library.custom_op(PYTORCH_OP_NAME, mutates_args=())
def _decode_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate) * up


@_decode_swiglu.register_fake
def _decode_swiglu_fake(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    del up
    return torch.empty_like(gate)


_CONVERTER_REGISTERED = False


def register_decode_swiglu_converter() -> None:
    """Lower the graph identity to the independent fixed-shape AIV op."""
    global _CONVERTER_REGISTERED
    if _CONVERTER_REGISTERED:
        return
    torchair, _CompilerConfig = import_torchair()
    converter_module = importlib.import_module(
        f"{torchair.__name__}._ge_concrete_graph.fx2ge_converter"
    )
    ge_module = importlib.import_module(f"{torchair.__name__}.ge")
    register_converter = converter_module.register_fx_node_ge_converter
    ge_custom_op = ge_module.custom_op

    @register_converter(torch.ops.paddleocr_vl.decode_swiglu_v1.default)
    def _convert_decode_swiglu(
        gate: Any,
        up: Any,
        meta_outputs: Any = None,
    ) -> Any:
        del meta_outputs
        return ge_custom_op(
            GE_OP_NAME,
            inputs={"gate": gate, "up": up},
            outputs=["output"],
        )

    _CONVERTER_REGISTERED = True


def decode_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Apply the Paddle B1 FP16 gate at the fixed intermediate width."""
    if gate.shape != (1, 1, 3072) or up.shape != gate.shape:
        raise ValueError("decode_swiglu requires gate/up[1,1,3072]")
    if gate.dtype != torch.float16 or up.dtype != torch.float16:
        raise ValueError("decode_swiglu requires FP16 gate/up")
    return _decode_swiglu(gate.contiguous(), up.contiguous())
