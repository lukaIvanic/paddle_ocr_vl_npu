#!/usr/bin/env python3
"""Real-QKV attention-only TorchAir repro for PaddleOCR-VL vision attention.

This intentionally compiles only the attention operation. Q/K/V tensors are
materialized eagerly from one real crop using the same layer-0 preprocessing,
manual LayerNorm, QKV projection, and vision RoPE used in the inline layer repro.
The compiled graph therefore cannot include patch embedding, LayerNorm, QKV,
MLP, residuals, or output projection.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


class AttentionOnly(torch.nn.Module):
    def __init__(
        self,
        *,
        attention: str,
        num_heads: int,
        scaling: float,
        sparse_mode: int,
        mask: torch.Tensor | None,
    ):
        super().__init__()
        if attention not in ("manual", "prompt_flash_attention"):
            raise ValueError(f"unsupported attention={attention!r}")
        self.attention = str(attention)
        self.num_heads = int(num_heads)
        self.scaling = float(scaling)
        self.sparse_mode = int(sparse_mode)
        self.register_buffer("atten_mask", mask, persistent=False)

    def forward(self, q_bnsd: torch.Tensor, k_bnsd: torch.Tensor, v_bnsd: torch.Tensor) -> torch.Tensor:
        if self.attention == "manual":
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
        return torch_npu.npu_prompt_flash_attention(
            q_bnsd.contiguous(),
            k_bnsd.contiguous(),
            v_bnsd.contiguous(),
            num_heads=int(self.num_heads),
            input_layout="BNSD",
            scale_value=float(self.scaling),
            sparse_mode=sparse_mode,
            **mask_kwargs,
        )


@torch.inference_mode()
def build_real_qkv_inputs(args: argparse.Namespace, model_dir: Path, device: torch.device, dtype: torch.dtype) -> tuple[
    LocalPaddleOCRVLForConditionalGeneration,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
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
    query, key, value = qkv.chunk(3, dim=-1)
    query = query.view(physical_seq_len, attention.num_heads, attention.head_dim)
    key = key.view(physical_seq_len, attention.num_heads, attention.head_dim)
    value = value.view(physical_seq_len, attention.num_heads, attention.head_dim)
    query, key = apply_rotary_pos_emb_vision(query, key, rope_cos, rope_sin)
    q_bnsd = query.transpose(0, 1).unsqueeze(0).contiguous()
    k_bnsd = key.transpose(0, 1).unsqueeze(0).contiguous()
    v_bnsd = value.transpose(0, 1).unsqueeze(0).contiguous()

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
        "q_summary": tensor_summary(q_bnsd.detach().cpu()),
        "k_summary": tensor_summary(k_bnsd.detach().cpu()),
        "v_summary": tensor_summary(v_bnsd.detach().cpu()),
    }
    return model, q_bnsd, k_bnsd, v_bnsd, meta


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
    model, q_bnsd, k_bnsd, v_bnsd, qkv_meta = build_real_qkv_inputs(args, model_dir, device, dtype)
    maybe_sync(device)
    setup_s = float(time.perf_counter() - start)

    mask = make_mask(
        kind=str(args.mask_kind),
        real_seq_len=int(qkv_meta["real_seq_len"]),
        physical_seq_len=int(qkv_meta["physical_seq_len"]),
        rank=int(args.mask_rank),
        device=device,
    )
    manual_ref_module = AttentionOnly(
        attention="manual",
        num_heads=int(qkv_meta["num_heads"]),
        scaling=float(qkv_meta["scaling"]),
        sparse_mode=int(args.promptfa_sparse_mode),
        mask=mask,
    ).eval()
    candidate_module = AttentionOnly(
        attention=str(args.attention),
        num_heads=int(qkv_meta["num_heads"]),
        scaling=float(qkv_meta["scaling"]),
        sparse_mode=int(args.promptfa_sparse_mode),
        mask=mask,
    ).eval()

    maybe_sync(device)
    start = time.perf_counter()
    manual_ref = manual_ref_module(q_bnsd, k_bnsd, v_bnsd)
    maybe_sync(device)
    manual_ref_s = float(time.perf_counter() - start)

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
        candidate_first = candidate_callable(q_bnsd, k_bnsd, v_bnsd)
        maybe_sync(device)
        candidate_first_s = float(time.perf_counter() - start)

        maybe_sync(device)
        start = time.perf_counter()
        candidate_second = candidate_callable(q_bnsd, k_bnsd, v_bnsd)
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
            "compile_backend": str(args.compile_backend),
            "fullgraph": bool(args.compile_backend != "none"),
            "dynamic": False if args.compile_backend != "none" else None,
            "backend_meta": backend_meta,
            "mask": mask_summary(mask, kind=str(args.mask_kind), rank=int(args.mask_rank)),
            "promptfa_sparse_mode": int(args.promptfa_sparse_mode),
            "no_padding": bool(args.no_padding),
        },
        "qkv_meta": qkv_meta,
        "timing_s": {
            "setup_materialize_qkv": setup_s,
            "manual_reference": manual_ref_s,
            "compile_wrapper": compile_wrapper_s,
            "candidate_first": candidate_first_s,
            "candidate_second": candidate_second_s,
        },
        "summary": {
            "candidate_vs_manual_allclose_5e_2": bool(candidate_vs_manual["overall"].get("allclose_atol_5e_2_rtol_5e_2")),
            "candidate_vs_manual_max_abs": candidate_vs_manual["overall"].get("max_abs_diff"),
            "candidate_vs_manual_real_max_abs": (candidate_vs_manual.get("row_split") or {}).get("real_rows", {}).get("max_abs_diff"),
            "candidate_vs_manual_real_nonfinite": (candidate_vs_manual.get("row_split") or {}).get("real_rows", {}).get("lhs_nonfinite_count"),
            "candidate_vs_manual_pad_max_abs": (candidate_vs_manual.get("row_split") or {}).get("pad_rows", {}).get("max_abs_diff"),
            "candidate_vs_manual_pad_nonfinite": (candidate_vs_manual.get("row_split") or {}).get("pad_rows", {}).get("lhs_nonfinite_count"),
            "candidate_first_vs_second_allclose_5e_2": bool(candidate_stability["overall"].get("allclose_atol_5e_2_rtol_5e_2")),
            "candidate_first_vs_second_max_abs": candidate_stability["overall"].get("max_abs_diff"),
        },
        "candidate_vs_manual": candidate_vs_manual,
        "candidate_first_vs_second": candidate_stability,
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
    print(f"summary={output['summary']}")
    row_split = candidate_vs_manual.get("row_split") or {}
    print(f"candidate_vs_manual_overall={candidate_vs_manual['overall']}")
    print(f"candidate_vs_manual_real_rows={row_split.get('real_rows')}")
    print(f"candidate_vs_manual_pad_rows={row_split.get('pad_rows')}")


if __name__ == "__main__":
    main()
