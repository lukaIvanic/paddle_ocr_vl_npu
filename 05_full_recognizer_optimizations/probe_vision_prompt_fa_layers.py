#!/usr/bin/env python3
"""Layer-by-layer PromptFlashAttention diff probe for the vision encoder."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

from bench_stage_timing import build_cohort_inputs, load_manifest, select_manifest_entries
from local_modeling_paddleocr_vl import (
    LocalPaddleOCRVLForConditionalGeneration,
    _resolve_model_dir,
    apply_rotary_pos_emb_vision,
)
from probe_static_compile import maybe_sync
from profile_vision_encoder import DEFAULT_CROP_ID
from probe_vision_prompt_fa_variants import diff_stats, manual_attention
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


def timed(device: torch.device, fn):
    maybe_sync(device)
    start = time.perf_counter()
    result = fn()
    maybe_sync(device)
    return result, time.perf_counter() - start


def prepare_initial_hidden_and_positions(
    model: LocalPaddleOCRVLForConditionalGeneration,
    *,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
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
    rotary_max = encoder.rotary_pos_emb(pids.max() + 1)
    rotary_embeddings = rotary_max[pids].flatten(1).repeat(1, 2)
    return hidden_states, (rotary_embeddings.cos(), rotary_embeddings.sin())


def qkv_bnsd(layer, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor]):
    attn = layer.self_attn
    normed = layer.layer_norm1(hidden_states)
    seq_length = int(normed.shape[0])
    query_states = attn.q_proj(normed).view(seq_length, attn.num_heads, attn.head_dim)
    key_states = attn.k_proj(normed).view(seq_length, attn.num_heads, attn.head_dim)
    value_states = attn.v_proj(normed).view(seq_length, attn.num_heads, attn.head_dim)
    query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, *position_embeddings)
    return (
        query_states.transpose(0, 1).unsqueeze(0).contiguous(),
        key_states.transpose(0, 1).unsqueeze(0).contiguous(),
        value_states.transpose(0, 1).unsqueeze(0).contiguous(),
        seq_length,
        float(attn.scaling),
        int(attn.num_heads),
    )


def prompt_fa_minimal(q_bnsd: torch.Tensor, k_bnsd: torch.Tensor, v_bnsd: torch.Tensor, *, num_heads: int, scale: float):
    import torch_npu

    return torch_npu.npu_prompt_flash_attention(
        q_bnsd,
        k_bnsd,
        v_bnsd,
        num_heads=int(num_heads),
        input_layout="BNSD",
        scale_value=float(scale),
        sparse_mode=0,
    )


def merge_heads(raw_bnsd: torch.Tensor, seq_length: int) -> torch.Tensor:
    return raw_bnsd.transpose(1, 2).contiguous().view(seq_length, -1)


def layer_output_from_attention(layer, hidden_states: torch.Tensor, attn_projected: torch.Tensor) -> torch.Tensor:
    hidden_states = hidden_states + attn_projected
    return hidden_states + layer.mlp(layer.layer_norm2(hidden_states))


def attention_outputs(layer, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor]):
    q, k, v, seq_length, scale, num_heads = qkv_bnsd(layer, hidden_states, position_embeddings)
    manual_raw = manual_attention(q, k, v, scale)
    prompt_raw = prompt_fa_minimal(q, k, v, num_heads=num_heads, scale=scale)
    manual_merged = merge_heads(manual_raw, seq_length)
    prompt_merged = merge_heads(prompt_raw, seq_length)
    manual_projected = layer.self_attn.out_proj(manual_merged)
    prompt_projected = layer.self_attn.out_proj(prompt_merged)
    return {
        "manual_raw": manual_raw,
        "prompt_raw": prompt_raw,
        "manual_merged": manual_merged,
        "prompt_merged": prompt_merged,
        "manual_projected": manual_projected,
        "prompt_projected": prompt_projected,
        "manual_layer": layer_output_from_attention(layer, hidden_states, manual_projected),
        "prompt_layer": layer_output_from_attention(layer, hidden_states, prompt_projected),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="PaddlePaddle/PaddleOCR-VL-1.6")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "crops" / "hotswap_100_manifest.json")
    parser.add_argument("--crop-id", default=DEFAULT_CROP_ID)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "float16", "bf16", "bfloat16"])
    parser.add_argument("--npu-jit-compile", default="off", choices=NPU_JIT_COMPILE_CHOICES)
    parser.add_argument("--max-layers", type=int, default=27)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if device.type != "npu":
        raise ValueError("Layer PromptFlashAttention probe requires --device npu:0")
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
    hidden0, position_embeddings = prepare_initial_hidden_and_positions(
        model,
        pixel_values=item.pixel_values.to(device),
        image_grid_thw=item.image_grid_thw.to(device),
    )
    layers = model.visual.vision_model.encoder.layers
    max_layers = min(int(args.max_layers), len(layers))
    manual_state = hidden0
    prompt_state = hidden0
    rows = []
    for layer_idx in range(max_layers):
        layer = layers[layer_idx]
        common, common_s = timed(device, lambda layer=layer, manual_state=manual_state: attention_outputs(layer, manual_state, position_embeddings))
        propagated_prompt, prompt_s = timed(device, lambda layer=layer, prompt_state=prompt_state: attention_outputs(layer, prompt_state, position_embeddings))
        manual_next = common["manual_layer"]
        prompt_next_same_input = common["prompt_layer"]
        prompt_next_propagated = propagated_prompt["prompt_layer"]
        rows.append(
            {
                "layer": int(layer_idx),
                "same_input_raw": diff_stats(common["prompt_raw"], common["manual_raw"]),
                "same_input_merged": diff_stats(common["prompt_merged"], common["manual_merged"]),
                "same_input_projected": diff_stats(common["prompt_projected"], common["manual_projected"]),
                "same_input_layer": diff_stats(prompt_next_same_input, manual_next),
                "propagated_input": diff_stats(prompt_state, manual_state),
                "propagated_layer": diff_stats(prompt_next_propagated, manual_next),
                "manual_common_s": float(common_s),
                "prompt_propagated_s": float(prompt_s),
            }
        )
        manual_state = manual_next
        prompt_state = prompt_next_propagated

    summary = {
        "probe": "vision_prompt_fa_layer_diffs",
        "model": str(model_dir),
        "device": str(device),
        "dtype": str(dtype),
        "crop_id": str(item.entry.get("id")),
        "crop_size": item.entry.get("crop_size"),
        "vision_tokens": int(hidden0.shape[0]),
        "hidden_size": int(hidden0.shape[-1]),
        "layers_checked": int(max_layers),
        "final_propagated": diff_stats(prompt_state, manual_state),
        "layers": rows,
    }
    print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
