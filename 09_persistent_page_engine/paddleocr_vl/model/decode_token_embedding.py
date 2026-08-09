"""One-token embedding lookup backed by the installed AscendC GatherV3 op."""

from __future__ import annotations

import importlib
from typing import Any

import torch
import torch.nn.functional as F

from .compile_utils import import_torchair


GE_OP_NAME = "GatherV3"
PYTORCH_OP_NAME = "paddleocr_vl::decode_token_embedding"


@torch.library.custom_op(PYTORCH_OP_NAME, mutates_args=())
def _decode_token_embedding(
    weight: torch.Tensor,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Eager reference; TorchAir lowers this identity to AscendC GatherV3."""
    return F.embedding(input_ids, weight)


@_decode_token_embedding.register_fake
def _decode_token_embedding_fake(
    weight: torch.Tensor,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(
        (*input_ids.shape, weight.shape[1]),
        dtype=weight.dtype,
        device=weight.device,
    )


_CONVERTER_REGISTERED = False


def register_decode_token_embedding_converter() -> None:
    """Lower the graph identity to CANN's installed AscendC GatherV3."""
    global _CONVERTER_REGISTERED
    if _CONVERTER_REGISTERED:
        return

    torchair, _CompilerConfig = import_torchair()
    converter_module = importlib.import_module(
        f"{torchair.__name__}._ge_concrete_graph.fx2ge_converter"
    )
    ge_module = importlib.import_module(f"{torchair.__name__}.ge")
    register_converter = converter_module.register_fx_node_ge_converter
    ge_const = ge_module.Const
    compat_ir = importlib.import_module(
        f"{torchair.__name__}._ge_concrete_graph.compat_ir"
    )
    ge_op = compat_ir.ge_op
    op = torch.ops.paddleocr_vl.decode_token_embedding.default

    @register_converter(op)
    def _convert_decode_token_embedding(
        weight: Any,
        input_ids: Any,
        meta_outputs: Any = None,
    ) -> Any:
        del meta_outputs
        return ge_op(
            op_type=GE_OP_NAME,
            inputs={
                "x": weight,
                "indices": input_ids,
                "axis": ge_const([0]),
            },
            outputs=["y"],
        )

    _CONVERTER_REGISTERED = True


def decode_token_embedding(
    weight: torch.Tensor,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Run the specialized B1, one-token embedding graph identity."""
    if weight.ndim != 2 or input_ids.shape != (1, 1):
        raise ValueError(
            "decode_token_embedding requires weight[V, H] and B1/S1 ids"
        )
    if weight.dtype != torch.float16 or input_ids.dtype != torch.int64:
        raise ValueError(
            "decode_token_embedding requires FP16 weight and INT64 ids"
        )
    return _decode_token_embedding(weight, input_ids)
