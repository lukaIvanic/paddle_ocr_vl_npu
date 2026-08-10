"""B1 ND KV-cache update through CANN's AscendC ScatterPaKvCache op."""

from __future__ import annotations

import importlib
from typing import Any

import torch

from .compile_utils import import_torchair


PYTORCH_OP_NAME = "paddleocr_vl::decode_kv_scatter_v1"
GE_OP_NAME = "ScatterPaKvCache"


def _reference(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_position: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    import torch_npu

    key_output = key_cache.clone()
    value_output = value_cache.clone()
    torch_npu.npu_scatter_pa_kv_cache(
        key_states,
        value_states,
        key_output,
        value_output,
        cache_position.reshape(-1).contiguous(),
        cache_mode="Norm",
    )
    return key_output, value_output


@torch.library.custom_op(PYTORCH_OP_NAME, mutates_args=())
def _decode_kv_scatter(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_position: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _reference(
        key_cache,
        value_cache,
        cache_position,
        key_states,
        value_states,
    )


@_decode_kv_scatter.register_fake
def _decode_kv_scatter_fake(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_position: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    del cache_position, key_states, value_states
    return torch.empty_like(key_cache), torch.empty_like(value_cache)


_CONVERTER_REGISTERED = False


def register_decode_kv_scatter_converter() -> None:
    """Lower the functional graph identity to CANN's AscendC ND ref op.

    CANN 9.0's generated ``ScatterPaKvCache`` GE wrapper hard-codes ``PA_NZ``
    even though its public signature accepts ``Norm``. Build the registered IR
    node directly with canonical GE attribute classes. The GE op remains the
    built-in ref op, so TorchAir can reuse each output buffer with its cache
    input and preserve persistent state without a full-cache copy.
    """
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

    @register_converter(torch.ops.paddleocr_vl.decode_kv_scatter_v1.default)
    def _convert_decode_kv_scatter(
        key_cache: Any,
        value_cache: Any,
        cache_position: Any,
        key_states: Any,
        value_states: Any,
        meta_outputs: Any = None,
    ) -> Any:
        del meta_outputs
        return ge_custom_op(
            GE_OP_NAME,
            inputs={
                "key": key_states,
                "key_cache": key_cache,
                "slot_mapping": cache_position,
                "value": value_states,
                "value_cache": value_cache,
                "compress_lens": None,
                "compress_seq_offset": None,
                "seq_lens": None,
            },
            attrs={
                "cache_mode": ge_attr.Str("Norm"),
                "scatter_mode": ge_attr.Str("None"),
                "strides": ge_attr.ListInt([1, 1]),
                "offsets": ge_attr.ListInt([0, 0]),
            },
            outputs=["key_cache", "value_cache"],
        )

    _CONVERTER_REGISTERED = True


def decode_kv_scatter(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_position: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Write and return one B1/GQA K/V row in the BSND decode arena."""
    if key_cache.shape != (1, 1024, 2, 128):
        raise ValueError("specialized KV scatter requires K cache[1,1024,2,128]")
    if value_cache.shape != key_cache.shape:
        raise ValueError("specialized KV scatter requires matching K/V caches")
    if key_states.shape != (1, 2, 1, 128):
        raise ValueError("specialized KV scatter requires K state[1,2,1,128]")
    if value_states.shape != key_states.shape:
        raise ValueError("specialized KV scatter requires matching K/V states")
    if cache_position.shape not in ((1,), (1, 1)):
        raise ValueError("specialized KV scatter requires one cache position")
    if key_cache.dtype != torch.float16 or key_states.dtype != torch.float16:
        raise ValueError("specialized KV scatter requires FP16 K/V tensors")
    if cache_position.dtype != torch.int64:
        raise ValueError("specialized KV scatter requires an INT64 position")
    key_input = key_states.transpose(1, 2).reshape(1, 2, 128).contiguous()
    value_input = value_states.transpose(1, 2).reshape(1, 2, 128).contiguous()
    position = cache_position.reshape(-1).contiguous()
    return _decode_kv_scatter(
        key_cache,
        value_cache,
        position,
        key_input,
        value_input,
    )
