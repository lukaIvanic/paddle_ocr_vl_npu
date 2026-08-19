"""Decode-only model rewrites shared by UniRec labs and production runners."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

import modeling_optimized_unirec as unirec
from modeling_optimized_unirec import OptimizedUniRecRunner, synchronize_device


UNIREC_SEMANTIC_VOCAB_SIZE = 56_371


class SemanticVocabDecodeStepModule(unirec.LocalUniRecCachedDecodeStepModule):
    """Hide padded LM-head rows from token selection."""

    def forward(
        self,
        decoder_input_ids: torch.Tensor,
        cache_position: torch.Tensor,
        active_length: int,
        self_key_cache: tuple[torch.Tensor, ...],
        self_value_cache: tuple[torch.Tensor, ...],
        cross_key_cache: tuple[torch.Tensor, ...],
        cross_value_cache: tuple[torch.Tensor, ...],
        cross_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        logits = super().forward(
            decoder_input_ids,
            cache_position,
            active_length,
            self_key_cache,
            self_value_cache,
            cross_key_cache,
            cross_value_cache,
            cross_attention_mask,
        )
        return logits[..., :UNIREC_SEMANTIC_VOCAB_SIZE]


def decode_cache_variant_root(
    base: str | Path,
    *,
    weight_format: str,
    lm_head_rows: int,
) -> Path:
    root = Path(base).expanduser().resolve()
    if weight_format == "native" and lm_head_rows == 0:
        return root
    normalized_rows = lm_head_rows or UNIREC_SEMANTIC_VOCAB_SIZE
    return root / (
        f"decode_weight_{weight_format}_lmhead{normalized_rows}_"
        f"semantic{UNIREC_SEMANTIC_VOCAB_SIZE}"
    )


@torch.inference_mode()
def apply_decode_model_optimizations(
    runner: OptimizedUniRecRunner,
    *,
    weight_format: str,
    lm_head_rows: int,
) -> dict[str, Any]:
    if weight_format not in ("native", "nz"):
        raise ValueError(f"unsupported decode weight format: {weight_format}")
    semantic_vocab = int(runner.config.vocab_size)
    if semantic_vocab != UNIREC_SEMANTIC_VOCAB_SIZE:
        raise ValueError(
            f"expected UniRec vocab {UNIREC_SEMANTIC_VOCAB_SIZE}, "
            f"got {semantic_vocab}"
        )
    target_rows = lm_head_rows or semantic_vocab
    if target_rows < semantic_vocab:
        raise ValueError("decode LM-head rows cannot be smaller than the vocabulary")
    if runner._compiled_decode_modules:
        raise RuntimeError("apply decode model optimizations before compiling")

    padded_rows = target_rows - semantic_vocab
    if padded_rows:
        head = runner.model.lm_head
        if int(head.weight.shape[0]) != semantic_vocab:
            raise ValueError("LM head no longer has the semantic vocabulary shape")
        weight = torch.nn.functional.pad(
            head.weight.detach(),
            (0, 0, 0, padded_rows),
        )
        head.weight = nn.Parameter(weight, requires_grad=False)
        head.out_features = target_rows
        unirec.LocalUniRecCachedDecodeStepModule = SemanticVocabDecodeStepModule
        synchronize_device(runner.device)

    nz_tensor_count = 0
    if weight_format == "nz":
        nz_tensor_count = runner.cast_decoder_weights_nz()

    return {
        "weight_format": weight_format,
        "weights_nz": bool(runner.weights_nz),
        "nz_tensor_count": int(nz_tensor_count),
        "semantic_vocab_size": semantic_vocab,
        "lm_head_rows": target_rows,
        "lm_head_padded_rows": padded_rows,
        "decode_module": (
            "SemanticVocabDecodeStepModule"
            if padded_rows
            else "LocalUniRecCachedDecodeStepModule"
        ),
    }
