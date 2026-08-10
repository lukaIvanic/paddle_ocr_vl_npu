"""B1 ND KV-cache update through CANN's AscendC ScatterPaKvCache op."""

from __future__ import annotations

import importlib
from typing import Any

import torch

from .compile_utils import import_torchair


_CONVERTER_REGISTERED = False


def register_decode_kv_scatter_converter() -> None:
    """Override torch-npu's PA_NZ-only converter with the requested ND mode.

    torch-npu 2.10 exposes ``cache_mode`` in the functional operator schema,
    but its installed TorchAir converter omits that argument and hard-codes
    ``PA_NZ``. The Paddle decode arena is ND. Keep the public mutating PyTorch
    op and lower its functionalized identity to the same built-in
    ``ScatterPaKvCache`` GE ref op with ``cache_mode=Norm``.
    """
    global _CONVERTER_REGISTERED
    if _CONVERTER_REGISTERED:
        return
    import torch_npu  # noqa: F401

    torchair, _CompilerConfig = import_torchair()
    # Force TorchAir's lazy custom-converter package to load first. Otherwise
    # its PA_NZ-only registration runs after this function and overwrites the
    # ND converter below on the first graph conversion.
    importlib.import_module(
        f"{torchair.__name__}._ge_concrete_graph.ge_converter.custom."
        "npu_scatter_pa_kv_cache"
    )
    converter_module = importlib.import_module(
        f"{torchair.__name__}._ge_concrete_graph.fx2ge_converter"
    )
    ge_apis = importlib.import_module(
        f"{torchair.__name__}._ge_concrete_graph.ge_apis"
    )
    ge_module = importlib.import_module(f"{torchair.__name__}.ge")
    # The compat-IR type checks use the canonical ``torchair.ge.attr`` class
    # objects. Importing the namespaced torch_npu module creates distinct
    # Python class identities that the builder rejects.
    ge_attr = importlib.import_module("torchair.ge.attr")
    ge_custom_op = ge_module.custom_op
    register_converter = converter_module.register_fx_node_ge_converter

    @register_converter(
        torch.ops.npu.npu_scatter_pa_kv_cache_functional.default
    )
    def _convert_scatter_pa_kv_cache_norm(
        key: Any,
        value: Any,
        key_cache: Any,
        value_cache: Any,
        slot_mapping: Any,
        *,
        compress_lens: Any = None,
        compress_seq_offsets: Any = None,
        seq_lens: Any = None,
        cache_mode: str = "PA_NZ",
        meta_outputs: Any = None,
    ) -> Any:
        del meta_outputs
        if cache_mode != "Norm":
            raise ValueError(
                "the Paddle B1 decoder KV scatter requires cache_mode='Norm'"
            )
        key_cache_copy = ge_apis.TensorMove(key_cache)
        value_cache_copy = ge_apis.TensorMove(value_cache)
        # The generated CANN 9.0 GE wrapper also hard-codes PA_NZ even though
        # its public signature defaults to Norm. Build the registered IR node
        # directly so the selected attribute is the requested ND contract.
        return ge_custom_op(
            "ScatterPaKvCache",
            inputs={
                "key": key,
                "key_cache": key_cache_copy,
                "slot_mapping": slot_mapping,
                "value": value,
                "value_cache": value_cache_copy,
                "compress_lens": compress_lens,
                "compress_seq_offset": compress_seq_offsets,
                "seq_lens": seq_lens,
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


def update_decode_kv_cache_with_scatter_pa_(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_position: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
) -> None:
    """Write one B1/GQA K/V row into the persistent ND decode arena."""
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

    import torch_npu

    torch_npu.npu_scatter_pa_kv_cache(
        key_states.transpose(1, 2).reshape(1, 2, 128).contiguous(),
        value_states.transpose(1, 2).reshape(1, 2, 128).contiguous(),
        key_cache,
        value_cache,
        cache_position.reshape(-1).contiguous(),
        cache_mode="Norm",
    )
