#!/usr/bin/env python3
"""Small-boundary TorchAir repro for PaddleOCR-VL vision attention.

This intentionally keeps patch embedding, LayerNorm, the QKV linear projection,
MLP, residuals, and output projection out of the compiled graph. The default
boundary compiles only attention over already-materialized BNSD Q/K/V tensors.
Optional boundaries progressively move SNHD layout conversion, RoPE, and QKV
chunk/view into the compiled graph so PromptFA GE drift and padded nonfinites
can be isolated without returning to a whole vision layer.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from local_modeling_paddleocr_vl import (
    LocalPaddleOCRVLForConditionalGeneration,
    apply_rotary_pos_emb_vision,
)
from repro_inline_single_layer_compile import (
    DEFAULT_BASELINE,
    DEFAULT_MODEL,
    _resolve_model_dir,
    build_static_abs_pos_embed,
    build_static_vision_rope,
    configure_npu_jit_compile,
    diff_stats,
    load_baseline_item,
    maybe_sync,
    parse_dtype,
    resolve_device,
    row_split_diff_stats,
    static_pad_tokens,
    tensor_summary,
    torchair_backend,
)


SCRIPT_DIR = Path(__file__).resolve().parent


def manual_layer_norm(layer_norm: torch.nn.LayerNorm, hidden_states: torch.Tensor, *, fp32_reduce: bool) -> torch.Tensor:
    if fp32_reduce:
        x = hidden_states.float()
        mean = x.mean(dim=-1, keepdim=True)
        centered = x - mean
        var = centered.pow(2).mean(dim=-1, keepdim=True)
        y = centered * torch.rsqrt(var + float(layer_norm.eps))
        y = y.to(dtype=hidden_states.dtype)
    else:
        mean = hidden_states.mean(dim=-1, keepdim=True)
        centered = hidden_states - mean
        var = centered.pow(2).mean(dim=-1, keepdim=True)
        y = centered * torch.rsqrt(var + float(layer_norm.eps))
    if layer_norm.weight is not None:
        y = y * layer_norm.weight
    if layer_norm.bias is not None:
        y = y + layer_norm.bias
    return y


def make_mask(
    *,
    kind: str,
    real_seq_len: int,
    physical_seq_len: int,
    rank: int,
    device: torch.device,
) -> torch.Tensor | None:
    if kind == "none" or int(real_seq_len) == int(physical_seq_len):
        return None
    if rank not in (2, 3, 4):
        raise ValueError(f"unsupported mask rank={rank}")
    real = int(real_seq_len)
    physical = int(physical_seq_len)
    mask = torch.zeros((physical, physical), device=device, dtype=torch.bool)
    if kind == "current":
        mask[:real, real:physical] = True
        mask[real:physical, :real] = True
    elif kind == "real_to_pad":
        mask[:real, real:physical] = True
    elif kind == "pad_to_real":
        mask[real:physical, :real] = True
    elif kind == "all_false":
        pass
    elif kind == "all_true_pad_rows":
        mask[real:physical, :] = True
        mask[:, real:physical] = True
    else:
        raise ValueError(f"unsupported mask kind={kind!r}")
    if rank == 2:
        return mask.contiguous()
    if rank == 3:
        return mask.unsqueeze(0).contiguous()
    return mask.unsqueeze(0).unsqueeze(0).contiguous()


def mask_summary(mask: torch.Tensor | None, *, kind: str, rank: int) -> dict[str, Any]:
    if mask is None:
        return {
            "kind": kind,
            "rank_requested": int(rank),
            "present": False,
            "shape": None,
            "dtype": None,
            "true_count": 0,
            "numel": 0,
            "true_fraction": None,
        }
    true_count = int(mask.sum().item())
    numel = int(mask.numel())
    return {
        "kind": kind,
        "rank_requested": int(rank),
        "present": True,
        "shape": [int(dim) for dim in mask.shape],
        "dtype": str(mask.dtype),
        "true_count": true_count,
        "numel": numel,
        "true_fraction": float(true_count / numel) if numel else None,
    }


def infer_promptfa_call_shape(
    *,
    input_layout: str,
    num_heads: int,
    physical_seq_len: int,
    call_head_dim: int,
) -> list[int]:
    if input_layout == "BNSD":
        return [1, int(num_heads), int(physical_seq_len), int(call_head_dim)]
    if input_layout == "BSND":
        return [1, int(physical_seq_len), int(num_heads), int(call_head_dim)]
    raise ValueError(f"unsupported PromptFA input_layout={input_layout!r}")


class AttentionOnly(torch.nn.Module):
    def __init__(
        self,
        *,
        attention: str,
        input_layout: str,
        input_boundary: str,
        num_heads: int,
        head_dim: int,
        scaling: float,
        sparse_mode: int,
        mask: torch.Tensor | None,
        rope_cos: torch.Tensor | None,
        rope_sin: torch.Tensor | None,
        promptfa_pad_head_dim_to: int,
    ):
        super().__init__()
        if attention not in ("manual", "prompt_flash_attention"):
            raise ValueError(f"unsupported attention={attention!r}")
        if input_layout not in ("BNSD", "BSND"):
            raise ValueError(f"unsupported input_layout={input_layout!r}")
        if input_boundary not in ("bnsd", "snhd_rope_done", "snhd_pre_rope", "qkv_flat_pre_rope"):
            raise ValueError(f"unsupported input_boundary={input_boundary!r}")
        self.attention = str(attention)
        self.input_layout = str(input_layout)
        self.input_boundary = str(input_boundary)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.scaling = float(scaling)
        self.sparse_mode = int(sparse_mode)
        self.promptfa_pad_head_dim_to = int(promptfa_pad_head_dim_to)
        if self.promptfa_pad_head_dim_to and self.promptfa_pad_head_dim_to < self.head_dim:
            raise ValueError(
                "--promptfa-pad-head-dim-to must be 0 or >= the real head_dim "
                f"({self.head_dim}), got {self.promptfa_pad_head_dim_to}"
            )
        self.register_buffer("atten_mask", mask, persistent=False)
        self.register_buffer(
            "rope_cos",
            torch.empty(0) if rope_cos is None else rope_cos.detach().clone(),
            persistent=False,
        )
        self.register_buffer(
            "rope_sin",
            torch.empty(0) if rope_sin is None else rope_sin.detach().clone(),
            persistent=False,
        )

    def _to_bnsd(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.input_layout == "BNSD":
            return tensor
        return tensor.transpose(1, 2).contiguous()

    def _from_snhd_to_call_layout(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.input_layout == "BNSD":
            return (
                query.transpose(0, 1).unsqueeze(0).contiguous(),
                key.transpose(0, 1).unsqueeze(0).contiguous(),
                value.transpose(0, 1).unsqueeze(0).contiguous(),
            )
        return (
            query.unsqueeze(0).contiguous(),
            key.unsqueeze(0).contiguous(),
            value.unsqueeze(0).contiguous(),
        )

    def _prepare_call_inputs(self, q_source: torch.Tensor, k_source: torch.Tensor, v_source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.input_boundary == "bnsd":
            return q_source, k_source, v_source
        if self.input_boundary == "snhd_rope_done":
            return self._from_snhd_to_call_layout(q_source, k_source, v_source)
        if self.input_boundary == "snhd_pre_rope":
            query, key = apply_rotary_pos_emb_vision(q_source, k_source, self.rope_cos, self.rope_sin)
            return self._from_snhd_to_call_layout(query, key, v_source)
        if self.input_boundary == "qkv_flat_pre_rope":
            seq_len = q_source.shape[0]
            query, key, value = q_source.chunk(3, dim=-1)
            query = query.view(seq_len, self.num_heads, self.head_dim)
            key = key.view(seq_len, self.num_heads, self.head_dim)
            value = value.view(seq_len, self.num_heads, self.head_dim)
            query, key = apply_rotary_pos_emb_vision(query, key, self.rope_cos, self.rope_sin)
            return self._from_snhd_to_call_layout(query, key, value)
        raise RuntimeError(f"unreachable input_boundary={self.input_boundary!r}")

    def _pad_promptfa_head_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        target = int(self.promptfa_pad_head_dim_to)
        if target <= 0 or target == int(tensor.shape[-1]):
            return tensor
        if target < int(tensor.shape[-1]):
            raise RuntimeError(f"cannot pad PromptFA head dim from {tensor.shape[-1]} down to {target}")
        return F.pad(tensor, (0, target - int(tensor.shape[-1])))

    def forward(self, q_source: torch.Tensor, k_source: torch.Tensor, v_source: torch.Tensor) -> torch.Tensor:
        q_states, k_states, v_states = self._prepare_call_inputs(q_source, k_source, v_source)
        if self.attention == "manual":
            q_bnsd = self._to_bnsd(q_states)
            k_bnsd = self._to_bnsd(k_states)
            v_bnsd = self._to_bnsd(v_states)
            scores = torch.matmul(q_bnsd, k_bnsd.transpose(2, 3)) * self.scaling
            if self.atten_mask is not None:
                scores = scores.masked_fill(self.atten_mask, torch.finfo(scores.dtype).min)
            probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q_bnsd.dtype)
            return torch.matmul(probs, v_bnsd)

        import torch_npu

        mask_kwargs = {}
        sparse_mode = 0
        if self.atten_mask is not None:
            mask_kwargs["atten_mask"] = self.atten_mask.to(torch.bool).contiguous()
            sparse_mode = int(self.sparse_mode)
        q_call = self._pad_promptfa_head_dim(q_states).contiguous()
        k_call = self._pad_promptfa_head_dim(k_states).contiguous()
        v_call = self._pad_promptfa_head_dim(v_states).contiguous()
        output = torch_npu.npu_prompt_flash_attention(
            q_call,
            k_call,
            v_call,
            num_heads=int(self.num_heads),
            input_layout=self.input_layout,
            scale_value=float(self.scaling),
            sparse_mode=sparse_mode,
            **mask_kwargs,
        )
        if int(output.shape[-1]) != int(self.head_dim):
            output = output[..., : int(self.head_dim)].contiguous()
        if self.input_layout == "BSND":
            return output.transpose(1, 2).contiguous()
        return output


@torch.inference_mode()
def build_real_qkv_inputs(args: argparse.Namespace, model_dir: Path, device: torch.device, dtype: torch.dtype) -> tuple[
    LocalPaddleOCRVLForConditionalGeneration,
    dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    dict[str, Any],
]:
    model = LocalPaddleOCRVLForConditionalGeneration.from_pretrained(model_dir, dtype=dtype, device=device).eval()
    pixel_values_cpu, image_grid_thw_cpu, item_meta = load_baseline_item(args, model_dir)
    pixel_values = pixel_values_cpu.to(device=device, dtype=model.visual.dtype)
    image_grid_thw = image_grid_thw_cpu.to(device=device)

    transformer = model.visual.vision_model
    embeddings = transformer.embeddings
    layer = transformer.encoder.layers[int(args.layer_index)]
    attention = layer.self_attn

    real_seq_len = int(image_grid_thw.prod().item())
    pad_tokens = static_pad_tokens(real_seq_len, no_padding=bool(args.no_padding))
    physical_seq_len = int(real_seq_len + pad_tokens)

    patch_flat = embeddings.patch_embedding(pixel_values.to(dtype=embeddings.patch_embedding.weight.dtype)).flatten(-2).squeeze(-1)
    if pad_tokens:
        patch_pad = torch.cat(
            [
                patch_flat,
                torch.zeros(pad_tokens, patch_flat.shape[-1], device=device, dtype=patch_flat.dtype),
            ],
            dim=0,
        )
    else:
        patch_pad = patch_flat

    abs_pos = build_static_abs_pos_embed(model, image_grid_thw, device)
    rope_cos, rope_sin = build_static_vision_rope(model, image_grid_thw, device)
    if pad_tokens:
        abs_pos = torch.cat(
            [abs_pos, torch.zeros(pad_tokens, abs_pos.shape[-1], device=device, dtype=abs_pos.dtype)],
            dim=0,
        ).contiguous()
        rope_cos = torch.cat(
            [rope_cos, torch.ones(pad_tokens, rope_cos.shape[-1], device=device, dtype=rope_cos.dtype)],
            dim=0,
        ).contiguous()
        rope_sin = torch.cat(
            [rope_sin, torch.zeros(pad_tokens, rope_sin.shape[-1], device=device, dtype=rope_sin.dtype)],
            dim=0,
        ).contiguous()

    patch_pos = patch_pad + abs_pos
    ln1 = manual_layer_norm(layer.layer_norm1, patch_pos, fp32_reduce=bool(args.ln_fp32_reduce))
    qkv = torch.cat(
        [
            attention.q_proj(ln1),
            attention.k_proj(ln1),
            attention.v_proj(ln1),
        ],
        dim=-1,
    )
    query_pre, key_pre, value_snhd = qkv.chunk(3, dim=-1)
    query_pre = query_pre.view(physical_seq_len, attention.num_heads, attention.head_dim)
    key_pre = key_pre.view(physical_seq_len, attention.num_heads, attention.head_dim)
    value_snhd = value_snhd.view(physical_seq_len, attention.num_heads, attention.head_dim)
    query_rope, key_rope = apply_rotary_pos_emb_vision(query_pre, key_pre, rope_cos, rope_sin)
    q_bnsd = query_rope.transpose(0, 1).unsqueeze(0).contiguous()
    k_bnsd = key_rope.transpose(0, 1).unsqueeze(0).contiguous()
    v_bnsd = value_snhd.transpose(0, 1).unsqueeze(0).contiguous()

    boundary_inputs = {
        "bnsd": (q_bnsd, k_bnsd, v_bnsd),
        "snhd_rope_done": (query_rope.contiguous(), key_rope.contiguous(), value_snhd.contiguous()),
        "snhd_pre_rope": (query_pre.contiguous(), key_pre.contiguous(), value_snhd.contiguous()),
        "qkv_flat_pre_rope": (qkv.contiguous(), qkv.contiguous(), qkv.contiguous()),
    }

    meta = {
        "item": item_meta,
        "layer_index": int(args.layer_index),
        "ln_fp32_reduce": bool(args.ln_fp32_reduce),
        "real_seq_len": int(real_seq_len),
        "pad_tokens": int(pad_tokens),
        "physical_seq_len": int(physical_seq_len),
        "physical_seq_len_mod16": int(physical_seq_len % 16),
        "physical_seq_len_mod128": int(physical_seq_len % 128),
        "num_heads": int(attention.num_heads),
        "head_dim": int(attention.head_dim),
        "scaling": float(attention.scaling),
        "rope_cos_shape": [int(dim) for dim in rope_cos.shape],
        "rope_sin_shape": [int(dim) for dim in rope_sin.shape],
        "boundary_input_shapes": {
            name: [[int(dim) for dim in tensor.shape] for tensor in tensors]
            for name, tensors in boundary_inputs.items()
        },
        "q_summary": tensor_summary(q_bnsd.detach().cpu()),
        "k_summary": tensor_summary(k_bnsd.detach().cpu()),
        "v_summary": tensor_summary(v_bnsd.detach().cpu()),
    }
    meta_tensors = {
        **boundary_inputs,
        "_rope": (rope_cos.contiguous(), rope_sin.contiguous(), rope_sin.contiguous()),
    }
    return model, meta_tensors, meta


def compare_attention_output(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    real_seq_len: int,
    physical_seq_len: int,
) -> dict[str, Any]:
    return {
        "overall": diff_stats(candidate.detach().cpu(), reference.detach().cpu()),
        "row_split": row_split_diff_stats(
            candidate.detach().cpu(),
            reference.detach().cpu(),
            real_seq_len=int(real_seq_len),
            physical_seq_len=int(physical_seq_len),
        ),
        "candidate_summary": tensor_summary(candidate.detach().cpu()),
        "reference_summary": tensor_summary(reference.detach().cpu()),
    }


def top_diff_locations(lhs: torch.Tensor, rhs: torch.Tensor, *, limit: int = 8) -> list[dict[str, Any]]:
    lhs_f = lhs.detach().float().cpu()
    rhs_f = rhs.detach().float().cpu()
    if tuple(lhs_f.shape) != tuple(rhs_f.shape):
        return []
    diff = torch.abs(lhs_f - rhs_f)
    finite = torch.isfinite(lhs_f) & torch.isfinite(rhs_f)
    finite_diff = diff.masked_fill(~finite, -1.0).reshape(-1)
    positive = finite_diff > 0
    if not bool(positive.any().item()):
        return []
    k = min(int(limit), int(positive.sum().item()))
    values, flat_indices = torch.topk(finite_diff, k=k)
    out: list[dict[str, Any]] = []
    shape = tuple(int(dim) for dim in lhs_f.shape)
    for value, flat_idx in zip(values.tolist(), flat_indices.tolist()):
        idx = np.unravel_index(int(flat_idx), shape)
        out.append(
            {
                "index": [int(part) for part in idx],
                "abs_diff": float(value),
                "candidate": float(lhs_f[idx].item()),
                "reference": float(rhs_f[idx].item()),
            }
        )
    return out


def nonfinite_locations(tensor: torch.Tensor, *, limit: int = 8) -> list[dict[str, Any]]:
    tensor_f = tensor.detach().float().cpu()
    nonfinite = ~torch.isfinite(tensor_f)
    if not bool(nonfinite.any().item()):
        return []
    flat_indices = nonfinite.reshape(-1).nonzero(as_tuple=False).flatten()[: int(limit)]
    shape = tuple(int(dim) for dim in tensor_f.shape)
    out: list[dict[str, Any]] = []
    for flat_idx in flat_indices.tolist():
        idx = np.unravel_index(int(flat_idx), shape)
        value = tensor_f[idx]
        out.append(
            {
                "index": [int(part) for part in idx],
                "value": str(float(value.item())),
                "is_nan": bool(torch.isnan(value).item()),
                "is_posinf": bool(torch.isposinf(value).item()),
                "is_neginf": bool(torch.isneginf(value).item()),
            }
        )
    return out


def nonfinite_mask_compare(lhs: torch.Tensor, rhs: torch.Tensor) -> dict[str, Any]:
    lhs_f = lhs.detach().float().cpu()
    rhs_f = rhs.detach().float().cpu()
    if tuple(lhs_f.shape) != tuple(rhs_f.shape):
        return {
            "shape_match": False,
            "lhs_shape": [int(dim) for dim in lhs_f.shape],
            "rhs_shape": [int(dim) for dim in rhs_f.shape],
        }
    lhs_nonfinite = ~torch.isfinite(lhs_f)
    rhs_nonfinite = ~torch.isfinite(rhs_f)
    mismatch = lhs_nonfinite ^ rhs_nonfinite
    both_nonfinite = lhs_nonfinite & rhs_nonfinite
    lhs_nan = torch.isnan(lhs_f)
    rhs_nan = torch.isnan(rhs_f)
    nan_mismatch = lhs_nan ^ rhs_nan
    return {
        "shape_match": True,
        "shape": [int(dim) for dim in lhs_f.shape],
        "lhs_nonfinite_count": int(lhs_nonfinite.sum().item()),
        "rhs_nonfinite_count": int(rhs_nonfinite.sum().item()),
        "both_nonfinite_count": int(both_nonfinite.sum().item()),
        "nonfinite_mask_match": bool(not mismatch.any().item()),
        "nonfinite_mask_mismatch_count": int(mismatch.sum().item()),
        "nan_mask_match": bool(not nan_mismatch.any().item()),
        "nan_mask_mismatch_count": int(nan_mismatch.sum().item()),
        "lhs_nan_count": int(lhs_nan.sum().item()),
        "rhs_nan_count": int(rhs_nan.sum().item()),
    }


def _ranges_from_bool_mask(mask_1d: torch.Tensor, *, limit: int = 12) -> list[list[int]]:
    indices = mask_1d.nonzero(as_tuple=False).flatten().tolist()
    if not indices:
        return []
    ranges: list[list[int]] = []
    start = prev = int(indices[0])
    for raw_idx in indices[1:]:
        idx = int(raw_idx)
        if idx == prev + 1:
            prev = idx
            continue
        ranges.append([start, prev])
        if len(ranges) >= int(limit):
            return ranges
        start = prev = idx
    ranges.append([start, prev])
    return ranges[: int(limit)]


def _index_count_samples(counts_1d: torch.Tensor, *, limit: int = 12) -> dict[str, list[dict[str, int]]]:
    active = counts_1d > 0
    indices = active.nonzero(as_tuple=False).flatten().tolist()
    if not indices:
        return {"first": [], "last": []}

    def _rows(raw_indices: list[int]) -> list[dict[str, int]]:
        return [
            {"index": int(idx), "count": int(counts_1d[int(idx)].item())}
            for idx in raw_indices
        ]

    limit = int(limit)
    return {
        "first": _rows(indices[:limit]),
        "last": _rows(indices[-limit:]),
    }


def nonfinite_pattern_summary(
    tensor: torch.Tensor,
    *,
    real_seq_len: int,
    physical_seq_len: int,
    limit_locations: int = 8,
) -> dict[str, Any]:
    tensor_f = tensor.detach().float().cpu()
    nonfinite = ~torch.isfinite(tensor_f)
    out: dict[str, Any] = {
        "shape": [int(dim) for dim in tensor_f.shape],
        "total_nonfinite_count": int(nonfinite.sum().item()),
        "nan_count": int(torch.isnan(tensor_f).sum().item()),
        "posinf_count": int(torch.isposinf(tensor_f).sum().item()),
        "neginf_count": int(torch.isneginf(tensor_f).sum().item()),
        "layout_assumption": "BNSD" if tensor_f.ndim == 4 else "unknown",
    }
    if tensor_f.ndim != 4:
        out["locations"] = nonfinite_locations(tensor_f, limit=int(limit_locations))
        return out

    _batch, num_heads, seq_len, _head_dim = tensor_f.shape
    real = min(int(real_seq_len), int(seq_len))
    out["real_nonfinite_count"] = int(nonfinite[:, :, :real, :].sum().item())
    out["pad_nonfinite_count"] = int(nonfinite[:, :, real:, :].sum().item())

    per_head: list[dict[str, Any]] = []
    for head_idx in range(int(num_heads)):
        head_mask = nonfinite[:, head_idx, :, :]
        count = int(head_mask.sum().item())
        if count == 0:
            continue
        nz = head_mask.nonzero(as_tuple=False)
        seq_mask = head_mask.any(dim=(0, 2))
        dim_mask = head_mask.any(dim=(0, 1))
        seq_counts = head_mask.sum(dim=(0, 2))
        dim_counts = head_mask.sum(dim=(0, 1))
        max_per_seq = int(head_mask.shape[0] * head_mask.shape[2])
        max_per_dim = int(head_mask.shape[0] * head_mask.shape[1])
        partial_seq_mask = (seq_counts > 0) & (seq_counts < max_per_seq)
        full_seq_mask = seq_counts == max_per_seq
        partial_dim_mask = (dim_counts > 0) & (dim_counts < max_per_dim)
        full_dim_mask = dim_counts == max_per_dim
        active_seq_counts = seq_counts[seq_counts > 0]
        active_dim_counts = dim_counts[dim_counts > 0]
        per_head.append(
            {
                "head": int(head_idx),
                "count": count,
                "real_count": int(head_mask[:, :real, :].sum().item()),
                "pad_count": int(head_mask[:, real:, :].sum().item()),
                "seq_min": int(nz[:, 1].min().item()),
                "seq_max": int(nz[:, 1].max().item()),
                "dim_min": int(nz[:, 2].min().item()),
                "dim_max": int(nz[:, 2].max().item()),
                "active_seq_count": int((seq_counts > 0).sum().item()),
                "full_seq_count": int(full_seq_mask.sum().item()),
                "partial_seq_count": int(partial_seq_mask.sum().item()),
                "active_seq_min_count": int(active_seq_counts.min().item()) if active_seq_counts.numel() else 0,
                "active_seq_max_count": int(active_seq_counts.max().item()) if active_seq_counts.numel() else 0,
                "seq_ranges_first": _ranges_from_bool_mask(seq_mask),
                "full_seq_ranges_first": _ranges_from_bool_mask(full_seq_mask),
                "partial_seq_ranges_first": _ranges_from_bool_mask(partial_seq_mask),
                "seq_count_samples": _index_count_samples(seq_counts),
                "active_dim_count": int((dim_counts > 0).sum().item()),
                "full_dim_count": int(full_dim_mask.sum().item()),
                "partial_dim_count": int(partial_dim_mask.sum().item()),
                "active_dim_min_count": int(active_dim_counts.min().item()) if active_dim_counts.numel() else 0,
                "active_dim_max_count": int(active_dim_counts.max().item()) if active_dim_counts.numel() else 0,
                "dim_ranges_first": _ranges_from_bool_mask(dim_mask),
                "full_dim_ranges_first": _ranges_from_bool_mask(full_dim_mask),
                "partial_dim_ranges_first": _ranges_from_bool_mask(partial_dim_mask),
                "dim_count_samples": _index_count_samples(dim_counts),
            }
        )
    out["per_head"] = sorted(per_head, key=lambda row: -int(row["count"]))
    out["locations"] = nonfinite_locations(tensor_f, limit=int(limit_locations))
    return out


def per_head_diff_summary(lhs: torch.Tensor, rhs: torch.Tensor) -> list[dict[str, Any]]:
    lhs_f = lhs.detach().float().cpu()
    rhs_f = rhs.detach().float().cpu()
    if tuple(lhs_f.shape) != tuple(rhs_f.shape) or lhs_f.ndim != 4:
        return []
    rows: list[dict[str, Any]] = []
    for head_idx in range(int(lhs_f.shape[1])):
        diff = diff_stats(lhs_f[:, head_idx], rhs_f[:, head_idx])
        rows.append(
            {
                "head": int(head_idx),
                "max_abs_diff": diff.get("max_abs_diff"),
                "mean_abs_diff": diff.get("mean_abs_diff"),
                "p99": diff.get("abs_diff_quantiles", {}).get("p99"),
                "candidate_nonfinite_count": diff.get("lhs_nonfinite_count"),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            -int(row["candidate_nonfinite_count"] or 0),
            -1.0 if row["max_abs_diff"] is None else -float(row["max_abs_diff"]),
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    parser.add_argument("--dataset-dir", default="")
    parser.add_argument("--item-index", type=int, default=0)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", default="fp16", choices=("fp16", "float16", "fp32", "float32", "bf16", "bfloat16"))
    parser.add_argument("--npu-jit-compile", default="off", choices=("default", "off", "on"))
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--attention", default="prompt_flash_attention", choices=("manual", "prompt_flash_attention"))
    parser.add_argument("--promptfa-layout", default="bnsd", choices=("bnsd", "bsnd"))
    parser.add_argument(
        "--promptfa-pad-head-dim-to",
        type=int,
        default=0,
        help=(
            "PromptFA-only zero-padding target for the last/head_dim axis. "
            "0 keeps the real head_dim. If set above the real head_dim, Q/K/V "
            "are padded only for the PromptFA call, scale stays based on the "
            "real head_dim, and the output is sliced back before comparison."
        ),
    )
    parser.add_argument(
        "--input-boundary",
        default="bnsd",
        choices=("bnsd", "snhd_rope_done", "snhd_pre_rope", "qkv_flat_pre_rope"),
    )
    parser.add_argument("--compile-backend", default="torchair", choices=("none", "torchair", "aot_eager", "inductor"))
    parser.add_argument("--torchair-mode", default="default", choices=("default", "max-autotune"))
    parser.add_argument("--torchair-run-eagerly", action="store_true")
    parser.add_argument("--mask-kind", default="current", choices=("none", "current", "real_to_pad", "pad_to_real", "all_false", "all_true_pad_rows"))
    parser.add_argument("--mask-rank", type=int, default=4, choices=(2, 3, 4))
    parser.add_argument("--promptfa-sparse-mode", type=int, default=1, choices=(0, 1))
    parser.add_argument("--no-padding", action="store_true")
    parser.add_argument("--ln-fp32-reduce", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", default="")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype)
    configure_npu_jit_compile(args.npu_jit_compile, device)
    model_dir = _resolve_model_dir(args.model)

    maybe_sync(device)
    start = time.perf_counter()
    model, qkv_tensors, qkv_meta = build_real_qkv_inputs(args, model_dir, device, dtype)
    maybe_sync(device)
    setup_s = float(time.perf_counter() - start)
    boundary = str(args.input_boundary)
    q_boundary, k_boundary, v_boundary = qkv_tensors[boundary]
    rope_cos, rope_sin, _unused_rope = qkv_tensors["_rope"]

    mask = make_mask(
        kind=str(args.mask_kind),
        real_seq_len=int(qkv_meta["real_seq_len"]),
        physical_seq_len=int(qkv_meta["physical_seq_len"]),
        rank=int(args.mask_rank),
        device=device,
    )
    promptfa_layout = str(args.promptfa_layout).upper()
    promptfa_pad_head_dim_to = int(args.promptfa_pad_head_dim_to)
    if promptfa_pad_head_dim_to and promptfa_pad_head_dim_to < int(qkv_meta["head_dim"]):
        raise ValueError(
            "--promptfa-pad-head-dim-to must be 0 or >= the real head_dim "
            f"({qkv_meta['head_dim']}), got {promptfa_pad_head_dim_to}"
        )
    promptfa_call_head_dim = int(qkv_meta["head_dim"])
    if promptfa_pad_head_dim_to > promptfa_call_head_dim:
        promptfa_call_head_dim = promptfa_pad_head_dim_to
    if boundary == "bnsd" and promptfa_layout == "BSND":
        q_input = q_boundary.transpose(1, 2).contiguous()
        k_input = k_boundary.transpose(1, 2).contiguous()
        v_input = v_boundary.transpose(1, 2).contiguous()
    else:
        q_input = q_boundary
        k_input = k_boundary
        v_input = v_boundary

    manual_ref_module = AttentionOnly(
        attention="manual",
        input_layout=promptfa_layout,
        input_boundary=boundary,
        num_heads=int(qkv_meta["num_heads"]),
        head_dim=int(qkv_meta["head_dim"]),
        scaling=float(qkv_meta["scaling"]),
        sparse_mode=int(args.promptfa_sparse_mode),
        mask=mask,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        promptfa_pad_head_dim_to=0,
    ).eval()
    candidate_module = AttentionOnly(
        attention=str(args.attention),
        input_layout=promptfa_layout,
        input_boundary=boundary,
        num_heads=int(qkv_meta["num_heads"]),
        head_dim=int(qkv_meta["head_dim"]),
        scaling=float(qkv_meta["scaling"]),
        sparse_mode=int(args.promptfa_sparse_mode),
        mask=mask,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        promptfa_pad_head_dim_to=promptfa_pad_head_dim_to,
    ).eval()
    eager_promptfa_module = AttentionOnly(
        attention="prompt_flash_attention",
        input_layout=promptfa_layout,
        input_boundary=boundary,
        num_heads=int(qkv_meta["num_heads"]),
        head_dim=int(qkv_meta["head_dim"]),
        scaling=float(qkv_meta["scaling"]),
        sparse_mode=int(args.promptfa_sparse_mode),
        mask=mask,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        promptfa_pad_head_dim_to=promptfa_pad_head_dim_to,
    ).eval()

    maybe_sync(device)
    start = time.perf_counter()
    manual_ref = manual_ref_module(q_input, k_input, v_input)
    maybe_sync(device)
    manual_ref_s = float(time.perf_counter() - start)

    eager_promptfa_ref = None
    eager_promptfa_s = None
    if str(args.attention) == "prompt_flash_attention":
        maybe_sync(device)
        start = time.perf_counter()
        eager_promptfa_ref = eager_promptfa_module(q_input, k_input, v_input)
        maybe_sync(device)
        eager_promptfa_s = float(time.perf_counter() - start)

    backend_meta: dict[str, Any] = {"backend_kind": "none"}
    compile_wrapper_s = 0.0
    candidate_callable = candidate_module
    old_capture_scalar_outputs: bool | None = None
    if args.compile_backend != "none":
        import torch._dynamo

        old_capture_scalar_outputs = bool(torch._dynamo.config.capture_scalar_outputs)
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.reset()
        compile_kwargs: dict[str, Any] = {"fullgraph": True, "dynamic": False}
        if args.compile_backend == "torchair":
            compile_kwargs["backend"] = torchair_backend(
                run_eagerly=bool(args.torchair_run_eagerly),
                mode=str(args.torchair_mode),
            )
            backend_meta = {
                "backend_kind": "torchair",
                "torchair_mode": str(args.torchair_mode),
                "torchair_run_eagerly": bool(args.torchair_run_eagerly),
            }
        else:
            compile_kwargs["backend"] = str(args.compile_backend)
            backend_meta = {"backend_kind": str(args.compile_backend)}
        maybe_sync(device)
        start = time.perf_counter()
        candidate_callable = torch.compile(candidate_module, **compile_kwargs)
        maybe_sync(device)
        compile_wrapper_s = float(time.perf_counter() - start)

    try:
        maybe_sync(device)
        start = time.perf_counter()
        candidate_first = candidate_callable(q_input, k_input, v_input)
        maybe_sync(device)
        candidate_first_s = float(time.perf_counter() - start)

        maybe_sync(device)
        start = time.perf_counter()
        candidate_second = candidate_callable(q_input, k_input, v_input)
        maybe_sync(device)
        candidate_second_s = float(time.perf_counter() - start)
    finally:
        if old_capture_scalar_outputs is not None:
            import torch._dynamo

            torch._dynamo.config.capture_scalar_outputs = old_capture_scalar_outputs

    compare_kwargs = {
        "real_seq_len": int(qkv_meta["real_seq_len"]),
        "physical_seq_len": int(qkv_meta["physical_seq_len"]),
    }
    candidate_vs_manual = compare_attention_output(candidate_second, manual_ref, **compare_kwargs)
    candidate_stability = compare_attention_output(candidate_first, candidate_second, **compare_kwargs)
    candidate_stability_nonfinite_mask = nonfinite_mask_compare(candidate_first, candidate_second)
    if eager_promptfa_ref is not None:
        candidate_vs_eager_promptfa = compare_attention_output(candidate_second, eager_promptfa_ref, **compare_kwargs)
        eager_promptfa_vs_manual = compare_attention_output(eager_promptfa_ref, manual_ref, **compare_kwargs)
    else:
        candidate_vs_eager_promptfa = None
        eager_promptfa_vs_manual = None

    output = {
        "schema_version": 1,
        "experiment": "07_vision_prefill_optimization",
        "kind": "attention_only_compile_repro",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(model_dir),
        "device": str(device),
        "dtype": str(dtype),
        "config": {
            "attention": str(args.attention),
            "promptfa_layout": str(args.promptfa_layout),
            "promptfa_pad_head_dim_to": int(promptfa_pad_head_dim_to),
            "promptfa_real_head_dim": int(qkv_meta["head_dim"]),
            "promptfa_call_head_dim": int(promptfa_call_head_dim),
            "promptfa_head_dim_pad_extra": int(promptfa_call_head_dim - int(qkv_meta["head_dim"])),
            "promptfa_call_head_dim_fp16_bytes": int(promptfa_call_head_dim * 2),
            "promptfa_call_head_dim_fp16_32b_aligned": bool((promptfa_call_head_dim * 2) % 32 == 0),
            "input_boundary": boundary,
            "attention_output_layout": "BNSD_normalized",
            "compile_backend": str(args.compile_backend),
            "fullgraph": bool(args.compile_backend != "none"),
            "dynamic": False if args.compile_backend != "none" else None,
            "backend_meta": backend_meta,
            "mask": mask_summary(mask, kind=str(args.mask_kind), rank=int(args.mask_rank)),
            "promptfa_sparse_mode": int(args.promptfa_sparse_mode),
            "no_padding": bool(args.no_padding),
        },
        "qkv_meta": qkv_meta,
        "attention_input_meta": {
            "layout": promptfa_layout,
            "input_boundary": boundary,
            "q_shape": [int(dim) for dim in q_input.shape],
            "k_shape": [int(dim) for dim in k_input.shape],
            "v_shape": [int(dim) for dim in v_input.shape],
            "q_stride": [int(dim) for dim in q_input.stride()],
            "k_stride": [int(dim) for dim in k_input.stride()],
            "v_stride": [int(dim) for dim in v_input.stride()],
            "q_is_contiguous": bool(q_input.is_contiguous()),
            "k_is_contiguous": bool(k_input.is_contiguous()),
            "v_is_contiguous": bool(v_input.is_contiguous()),
            "promptfa_call_shape_is_static_inferred": True,
            "promptfa_call_shape_note": "Shape after boundary prep/RoPE/layout conversion, before PromptFA. The final axis includes PromptFA-only D padding if enabled.",
            "promptfa_call_q_shape": infer_promptfa_call_shape(
                input_layout=promptfa_layout,
                num_heads=int(qkv_meta["num_heads"]),
                physical_seq_len=int(qkv_meta["physical_seq_len"]),
                call_head_dim=int(promptfa_call_head_dim),
            ),
            "promptfa_call_k_shape": infer_promptfa_call_shape(
                input_layout=promptfa_layout,
                num_heads=int(qkv_meta["num_heads"]),
                physical_seq_len=int(qkv_meta["physical_seq_len"]),
                call_head_dim=int(promptfa_call_head_dim),
            ),
            "promptfa_call_v_shape": infer_promptfa_call_shape(
                input_layout=promptfa_layout,
                num_heads=int(qkv_meta["num_heads"]),
                physical_seq_len=int(qkv_meta["physical_seq_len"]),
                call_head_dim=int(promptfa_call_head_dim),
            ),
            "promptfa_output_sliced_back_to_head_dim": bool(promptfa_call_head_dim != int(qkv_meta["head_dim"])),
        },
        "timing_s": {
            "setup_materialize_qkv": setup_s,
            "manual_reference": manual_ref_s,
            "eager_promptfa_reference": eager_promptfa_s,
            "compile_wrapper": compile_wrapper_s,
            "candidate_first": candidate_first_s,
            "candidate_second": candidate_second_s,
        },
        "summary": {
            "candidate_vs_manual_allclose_5e_2": bool(candidate_vs_manual["overall"].get("allclose_atol_5e_2_rtol_5e_2")),
            "candidate_vs_manual_max_abs": candidate_vs_manual["overall"].get("max_abs_diff"),
            "candidate_vs_manual_nonfinite": candidate_vs_manual["overall"].get("lhs_nonfinite_count"),
            "candidate_vs_manual_real_max_abs": (candidate_vs_manual.get("row_split") or {}).get("real_rows", {}).get("max_abs_diff"),
            "candidate_vs_manual_real_nonfinite": (candidate_vs_manual.get("row_split") or {}).get("real_rows", {}).get("lhs_nonfinite_count"),
            "candidate_vs_manual_pad_max_abs": (candidate_vs_manual.get("row_split") or {}).get("pad_rows", {}).get("max_abs_diff"),
            "candidate_vs_manual_pad_nonfinite": (candidate_vs_manual.get("row_split") or {}).get("pad_rows", {}).get("lhs_nonfinite_count"),
            "candidate_first_vs_second_allclose_5e_2": bool(candidate_stability["overall"].get("allclose_atol_5e_2_rtol_5e_2")),
            "candidate_first_vs_second_max_abs": candidate_stability["overall"].get("max_abs_diff"),
            "candidate_first_vs_second_nonfinite_mask_match": candidate_stability_nonfinite_mask.get("nonfinite_mask_match"),
            "candidate_first_vs_second_nonfinite_mask_mismatch_count": candidate_stability_nonfinite_mask.get("nonfinite_mask_mismatch_count"),
            "candidate_first_vs_second_nan_mask_match": candidate_stability_nonfinite_mask.get("nan_mask_match"),
            "candidate_first_vs_second_nan_mask_mismatch_count": candidate_stability_nonfinite_mask.get("nan_mask_mismatch_count"),
            "candidate_vs_eager_promptfa_max_abs": None
            if candidate_vs_eager_promptfa is None
            else candidate_vs_eager_promptfa["overall"].get("max_abs_diff"),
            "candidate_vs_eager_promptfa_nonfinite": None
            if candidate_vs_eager_promptfa is None
            else candidate_vs_eager_promptfa["overall"].get("lhs_nonfinite_count"),
            "candidate_vs_eager_promptfa_real_max_abs": None
            if candidate_vs_eager_promptfa is None
            else (candidate_vs_eager_promptfa.get("row_split") or {}).get("real_rows", {}).get("max_abs_diff"),
            "candidate_vs_eager_promptfa_real_nonfinite": None
            if candidate_vs_eager_promptfa is None
            else (candidate_vs_eager_promptfa.get("row_split") or {}).get("real_rows", {}).get("lhs_nonfinite_count"),
            "eager_promptfa_vs_manual_max_abs": None
            if eager_promptfa_vs_manual is None
            else eager_promptfa_vs_manual["overall"].get("max_abs_diff"),
        },
        "candidate_vs_manual": candidate_vs_manual,
        "candidate_vs_eager_promptfa": candidate_vs_eager_promptfa,
        "eager_promptfa_vs_manual": eager_promptfa_vs_manual,
        "candidate_first_vs_second": candidate_stability,
        "candidate_first_vs_second_nonfinite_mask": candidate_stability_nonfinite_mask,
        "top_diff_locations_candidate_vs_manual": top_diff_locations(candidate_second, manual_ref),
        "nonfinite_locations_candidate": nonfinite_locations(candidate_second),
        "nonfinite_locations_manual_reference": nonfinite_locations(manual_ref),
        "nonfinite_pattern_candidate_first": nonfinite_pattern_summary(candidate_first, **compare_kwargs),
        "nonfinite_pattern_candidate_second": nonfinite_pattern_summary(candidate_second, **compare_kwargs),
        "per_head_candidate_vs_manual": per_head_diff_summary(candidate_second, manual_ref),
    }

    output_path_raw = str(args.output or "").strip()
    if output_path_raw:
        output_path = Path(output_path_raw).expanduser().resolve()
    else:
        out_dir = SCRIPT_DIR / "outputs" / f"attention_only_repro_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        output_path = out_dir / "attention_only_repro.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")

    print("EXP07_ATTENTION_ONLY_REPRO SUMMARY")
    print(f"output={output_path}")
    print(f"config={output['config']}")
    print(f"qkv_meta={qkv_meta}")
    print(f"attention_input_meta={output['attention_input_meta']}")
    print(f"summary={output['summary']}")
    row_split = candidate_vs_manual.get("row_split") or {}
    print(f"candidate_vs_manual_overall={candidate_vs_manual['overall']}")
    print(f"candidate_vs_manual_real_rows={row_split.get('real_rows')}")
    print(f"candidate_vs_manual_pad_rows={row_split.get('pad_rows')}")
    if candidate_vs_eager_promptfa is not None:
        eager_row_split = candidate_vs_eager_promptfa.get("row_split") or {}
        print(f"candidate_vs_eager_promptfa_overall={candidate_vs_eager_promptfa['overall']}")
        print(f"candidate_vs_eager_promptfa_real_rows={eager_row_split.get('real_rows')}")
    if eager_promptfa_vs_manual is not None:
        promptfa_row_split = eager_promptfa_vs_manual.get("row_split") or {}
        print(f"eager_promptfa_vs_manual_overall={eager_promptfa_vs_manual['overall']}")
        print(f"eager_promptfa_vs_manual_real_rows={promptfa_row_split.get('real_rows')}")
    print(f"top_diff_locations_candidate_vs_manual={output['top_diff_locations_candidate_vs_manual']}")
    print(f"nonfinite_locations_candidate={output['nonfinite_locations_candidate']}")
    print(f"candidate_first_vs_second_nonfinite_mask={output['candidate_first_vs_second_nonfinite_mask']}")
    print(f"nonfinite_pattern_candidate_second={output['nonfinite_pattern_candidate_second']}")
    print(f"per_head_candidate_vs_manual_top4={output['per_head_candidate_vs_manual'][:4]}")


if __name__ == "__main__":
    main()
