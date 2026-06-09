#!/usr/bin/env python3
"""Probe PromptFlashAttention call variants for PaddleOCR-VL vision attention.

The full encoder output can explode after 27 layers if a single attention call
is wrong. This probe compares one first-layer vision attention call before the
output projection, using the real crop hidden states and real PaddleOCR-VL
q/k/v/vision RoPE tensors.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from bench_stage_timing import build_cohort_inputs, load_manifest, select_manifest_entries
from local_modeling_paddleocr_vl import (
    LocalPaddleOCRVLForConditionalGeneration,
    _resolve_model_dir,
    apply_rotary_pos_emb_vision,
)
from probe_static_compile import maybe_sync
from profile_vision_encoder import DEFAULT_CROP_ID
from run_local_recognition import (
    NPU_JIT_COMPILE_CHOICES,
    configure_npu_jit_compile,
    load_preprocessor_config,
    parse_dtype,
    resolve_device,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return str(value)


def timed(device: torch.device, fn: Callable[[], torch.Tensor]) -> tuple[torch.Tensor, float]:
    maybe_sync(device)
    start = time.perf_counter()
    result = fn()
    maybe_sync(device)
    return result, time.perf_counter() - start


def diff_stats(lhs: torch.Tensor, rhs: torch.Tensor) -> dict[str, float | bool]:
    lhs_cpu = lhs.detach().to(dtype=torch.float32).cpu()
    rhs_cpu = rhs.detach().to(dtype=torch.float32).cpu()
    diff = (lhs_cpu - rhs_cpu).abs()
    return {
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "allclose_2e_2": bool(torch.allclose(lhs_cpu, rhs_cpu, atol=2e-2, rtol=2e-2)),
        "allclose_5e_2": bool(torch.allclose(lhs_cpu, rhs_cpu, atol=5e-2, rtol=5e-2)),
        "allclose_1e_1": bool(torch.allclose(lhs_cpu, rhs_cpu, atol=1e-1, rtol=1e-1)),
    }


@torch.inference_mode()
def make_first_layer_qkv(
    *,
    model: LocalPaddleOCRVLForConditionalGeneration,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
) -> dict[str, torch.Tensor | int | list[int]]:
    pixel_values = pixel_values.type(model.visual.dtype).unsqueeze(0)
    vision_model = model.visual.vision_model
    hidden_states = vision_model.embeddings(pixel_values, image_grid_thw=image_grid_thw)
    encoder = vision_model.encoder

    device = hidden_states.device
    split_hids = []
    split_wids = []
    for t, h, w in image_grid_thw:
        image_pids = torch.arange(int(t * h * w), device=device) % int(h * w)
        split_hids.append(image_pids // int(w))
        split_wids.append(image_pids % int(w))
    pids = torch.stack([torch.cat(split_hids), torch.cat(split_wids)], dim=-1)
    max_grid_size = pids.max() + 1
    rotary_max = encoder.rotary_pos_emb(max_grid_size)
    rotary_embeddings = rotary_max[pids].flatten(1).repeat(1, 2)
    position_embeddings = (rotary_embeddings.cos(), rotary_embeddings.sin())

    layer = encoder.layers[0]
    attn = layer.self_attn
    normed = layer.layer_norm1(hidden_states)
    seq_length = int(normed.shape[0])
    query_states = attn.q_proj(normed).view(seq_length, attn.num_heads, attn.head_dim)
    key_states = attn.k_proj(normed).view(seq_length, attn.num_heads, attn.head_dim)
    value_states = attn.v_proj(normed).view(seq_length, attn.num_heads, attn.head_dim)
    query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, *position_embeddings)
    q_bnsd = query_states.transpose(0, 1).unsqueeze(0).contiguous()
    k_bnsd = key_states.transpose(0, 1).unsqueeze(0).contiguous()
    v_bnsd = value_states.transpose(0, 1).unsqueeze(0).contiguous()
    cu_seqlens = F.pad(
        torch.repeat_interleave(
            image_grid_thw[:, 1] * image_grid_thw[:, 2],
            image_grid_thw[:, 0],
        ).cumsum(dim=0, dtype=torch.int32),
        (1, 0),
        value=0,
    )
    lengths = [int(v) for v in (cu_seqlens[1:] - cu_seqlens[:-1]).detach().cpu().tolist()]
    if len(lengths) != 1:
        raise ValueError(f"expected one crop sequence, got lengths={lengths}")
    return {
        "q_bnsd": q_bnsd,
        "k_bnsd": k_bnsd,
        "v_bnsd": v_bnsd,
        "length": int(lengths[0]),
        "num_heads": int(attn.num_heads),
        "head_dim": int(attn.head_dim),
        "scaling": float(attn.scaling),
        "hidden_size": int(attn.embed_dim),
    }


def manual_attention(q_bnsd: torch.Tensor, k_bnsd: torch.Tensor, v_bnsd: torch.Tensor, scale: float) -> torch.Tensor:
    scores = torch.matmul(q_bnsd, k_bnsd.transpose(2, 3)) * float(scale)
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q_bnsd.dtype)
    return torch.matmul(probs, v_bnsd).contiguous()


def prompt_fa_variants(
    *,
    q_bnsd: torch.Tensor,
    k_bnsd: torch.Tensor,
    v_bnsd: torch.Tensor,
    length: int,
    num_heads: int,
    scale: float,
) -> dict[str, Callable[[], torch.Tensor]]:
    import torch_npu

    def bnsd_lengths() -> torch.Tensor:
        return torch_npu.npu_prompt_flash_attention(
            q_bnsd,
            k_bnsd,
            v_bnsd,
            actual_seq_lengths=[int(length)],
            actual_seq_lengths_kv=[int(length)],
            num_heads=int(num_heads),
            num_key_value_heads=int(num_heads),
            input_layout="BNSD",
            scale_value=float(scale),
            sparse_mode=0,
        )

    def bnsd_lengths_pre_next() -> torch.Tensor:
        return torch_npu.npu_prompt_flash_attention(
            q_bnsd,
            k_bnsd,
            v_bnsd,
            actual_seq_lengths=[int(length)],
            actual_seq_lengths_kv=[int(length)],
            num_heads=int(num_heads),
            num_key_value_heads=int(num_heads),
            input_layout="BNSD",
            scale_value=float(scale),
            pre_tokens=65535,
            next_tokens=65535,
            sparse_mode=0,
        )

    def bnsd_no_lengths() -> torch.Tensor:
        return torch_npu.npu_prompt_flash_attention(
            q_bnsd,
            k_bnsd,
            v_bnsd,
            num_heads=int(num_heads),
            num_key_value_heads=int(num_heads),
            input_layout="BNSD",
            scale_value=float(scale),
            sparse_mode=0,
        )

    def bnsd_no_kv_heads() -> torch.Tensor:
        return torch_npu.npu_prompt_flash_attention(
            q_bnsd,
            k_bnsd,
            v_bnsd,
            actual_seq_lengths=[int(length)],
            actual_seq_lengths_kv=[int(length)],
            num_heads=int(num_heads),
            input_layout="BNSD",
            scale_value=float(scale),
            pre_tokens=65535,
            next_tokens=65535,
            sparse_mode=0,
        )

    q_bsnd = q_bnsd.transpose(1, 2).contiguous()
    k_bsnd = k_bnsd.transpose(1, 2).contiguous()
    v_bsnd = v_bnsd.transpose(1, 2).contiguous()

    def bsnd_lengths_pre_next_to_bnsd() -> torch.Tensor:
        out = torch_npu.npu_prompt_flash_attention(
            q_bsnd,
            k_bsnd,
            v_bsnd,
            actual_seq_lengths=[int(length)],
            actual_seq_lengths_kv=[int(length)],
            num_heads=int(num_heads),
            num_key_value_heads=int(num_heads),
            input_layout="BSND",
            scale_value=float(scale),
            pre_tokens=65535,
            next_tokens=65535,
            sparse_mode=0,
        )
        return out.transpose(1, 2).contiguous()

    def bsnd_no_lengths_to_bnsd() -> torch.Tensor:
        out = torch_npu.npu_prompt_flash_attention(
            q_bsnd,
            k_bsnd,
            v_bsnd,
            num_heads=int(num_heads),
            input_layout="BSND",
            scale_value=float(scale),
            sparse_mode=0,
        )
        return out.transpose(1, 2).contiguous()

    q_bsh = q_bsnd.reshape(1, int(length), int(num_heads) * q_bnsd.shape[-1]).contiguous()
    k_bsh = k_bsnd.reshape(1, int(length), int(num_heads) * k_bnsd.shape[-1]).contiguous()
    v_bsh = v_bsnd.reshape(1, int(length), int(num_heads) * v_bnsd.shape[-1]).contiguous()

    def bsh_no_lengths_to_bnsd() -> torch.Tensor:
        out = torch_npu.npu_prompt_flash_attention(
            q_bsh,
            k_bsh,
            v_bsh,
            num_heads=int(num_heads),
            input_layout="BSH",
            scale_value=float(scale),
            sparse_mode=0,
        )
        return out.view(1, int(length), int(num_heads), q_bnsd.shape[-1]).transpose(1, 2).contiguous()

    return {
        "bnsd_lengths": bnsd_lengths,
        "bnsd_lengths_pre_next": bnsd_lengths_pre_next,
        "bnsd_no_lengths": bnsd_no_lengths,
        "bnsd_no_kv_heads": bnsd_no_kv_heads,
        "bsnd_lengths_pre_next": bsnd_lengths_pre_next_to_bnsd,
        "bsnd_no_lengths": bsnd_no_lengths_to_bnsd,
        "bsh_no_lengths": bsh_no_lengths_to_bnsd,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="PaddlePaddle/PaddleOCR-VL-1.6")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "crops" / "hotswap_100_manifest.json")
    parser.add_argument("--crop-id", default=DEFAULT_CROP_ID)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "float16", "bf16", "bfloat16"])
    parser.add_argument("--npu-jit-compile", default="off", choices=NPU_JIT_COMPILE_CHOICES)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if device.type != "npu":
        raise ValueError("PromptFlashAttention variant probe requires --device npu:0")
    configure_npu_jit_compile(args.npu_jit_compile, device)
    dtype = parse_dtype(args.dtype, device)
    model_dir = _resolve_model_dir(args.model)
    pre_cfg = load_preprocessor_config(model_dir)
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    manifest = load_manifest(args.manifest)
    entries = select_manifest_entries(manifest, num_items=1, crop_ids=[str(args.crop_id)])
    cohort = build_cohort_inputs(
        entries=entries,
        manifest_path=args.manifest,
        tokenizer=tokenizer,
        pre_cfg=pre_cfg,
        prompt_override=None,
    )
    item = cohort[0]
    model = LocalPaddleOCRVLForConditionalGeneration.from_pretrained(model_dir, dtype=dtype, device=device)
    maybe_sync(device)
    tensors = make_first_layer_qkv(
        model=model,
        pixel_values=item.pixel_values.to(device),
        image_grid_thw=item.image_grid_thw.to(device),
    )
    q_bnsd = tensors["q_bnsd"]
    k_bnsd = tensors["k_bnsd"]
    v_bnsd = tensors["v_bnsd"]
    length = int(tensors["length"])
    num_heads = int(tensors["num_heads"])
    scale = float(tensors["scaling"])

    manual, manual_s = timed(device, lambda: manual_attention(q_bnsd, k_bnsd, v_bnsd, scale))
    results = []
    for name, fn in prompt_fa_variants(
        q_bnsd=q_bnsd,
        k_bnsd=k_bnsd,
        v_bnsd=v_bnsd,
        length=length,
        num_heads=num_heads,
        scale=scale,
    ).items():
        try:
            out, elapsed_s = timed(device, fn)
            results.append(
                {
                    "name": name,
                    "ok": True,
                    "elapsed_s": float(elapsed_s),
                    "output_shape": [int(v) for v in out.shape],
                    **diff_stats(out, manual),
                }
            )
        except Exception as exc:
            results.append(
                {
                    "name": name,
                    "ok": False,
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                }
            )

    summary = {
        "probe": "vision_prompt_fa_single_layer_variants",
        "model": str(model_dir),
        "device": str(device),
        "dtype": str(dtype),
        "crop_id": str(item.entry.get("id")),
        "crop_size": item.entry.get("crop_size"),
        "length": length,
        "num_heads": num_heads,
        "head_dim": int(tensors["head_dim"]),
        "hidden_size": int(tensors["hidden_size"]),
        "manual_elapsed_s": float(manual_s),
        "manual_shape": [int(v) for v in manual.shape],
        "results": results,
    }
    print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
