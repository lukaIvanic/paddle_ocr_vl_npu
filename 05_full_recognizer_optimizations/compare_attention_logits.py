#!/usr/bin/env python3
"""Compare real-crop prefill logits under two attention configurations.

This is a compact numerical-drift harness, not a throughput benchmark. It runs
the same local PaddleOCR-VL recognizer on real crop inputs twice and compares:

1. Projected image embeddings after the native-resolution vision encoder and
   adaptive MLP projector.
2. Final prefill LM logits for the first generated token.
3. The first generated token argmax.

Use this when generation text still matches but we need to know whether an
attention implementation is quietly changing logits or vision embeddings.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from bench_stage_timing import build_cohort_inputs, load_manifest, select_manifest_entries, stats
from local_modeling_paddleocr_vl import (
    SOFTMAX_DTYPE_CHOICES,
    TEXT_SOFTMAX_DTYPE_ENV,
    VISION_ATTENTION_CHOICES,
    VISION_ATTENTION_ENV,
    VISION_PROMPT_FA_LAYOUT_CHOICES,
    VISION_PROMPT_FA_LAYOUT_ENV,
    VISION_SOFTMAX_DTYPE_ENV,
    LocalPaddleOCRVLForConditionalGeneration,
    _resolve_model_dir,
)
from probe_static_compile import maybe_sync
from run_local_recognition import (
    NPU_JIT_COMPILE_CHOICES,
    configure_npu_jit_compile,
    load_preprocessor_config,
    parse_dtype,
    resolve_device,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AttentionConfig:
    name: str
    vision_attention: str
    vision_prompt_fa_layout: str
    text_softmax_dtype: str
    vision_softmax_dtype: str


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return str(value)


def apply_attention_config(config: AttentionConfig) -> None:
    os.environ[VISION_ATTENTION_ENV] = str(config.vision_attention)
    os.environ[VISION_PROMPT_FA_LAYOUT_ENV] = str(config.vision_prompt_fa_layout)
    os.environ[TEXT_SOFTMAX_DTYPE_ENV] = str(config.text_softmax_dtype)
    os.environ[VISION_SOFTMAX_DTYPE_ENV] = str(config.vision_softmax_dtype)


def diff_stats(lhs: torch.Tensor, rhs: torch.Tensor) -> dict[str, Any]:
    lhs_cpu = lhs.detach().to(dtype=torch.float32).cpu()
    rhs_cpu = rhs.detach().to(dtype=torch.float32).cpu()
    diff = (lhs_cpu - rhs_cpu).abs()
    flat_idx = int(diff.argmax().item()) if diff.numel() else 0
    return {
        "shape": [int(value) for value in lhs.shape],
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
        "rms_abs_diff": float(torch.sqrt(torch.mean(diff * diff)).item()) if diff.numel() else 0.0,
        "max_abs_flat_index": flat_idx,
        "allclose_atol_1e_2_rtol_1e_2": bool(torch.allclose(lhs_cpu, rhs_cpu, atol=1e-2, rtol=1e-2)),
        "allclose_atol_5e_2_rtol_5e_2": bool(torch.allclose(lhs_cpu, rhs_cpu, atol=5e-2, rtol=5e-2)),
        "allclose_atol_1e_1_rtol_1e_1": bool(torch.allclose(lhs_cpu, rhs_cpu, atol=1e-1, rtol=1e-1)),
    }


def aggregate_diff(items: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return {
        "max_abs_diff": stats([float(item[key]["max_abs_diff"]) for item in items]),
        "mean_abs_diff": stats([float(item[key]["mean_abs_diff"]) for item in items]),
        "rms_abs_diff": stats([float(item[key]["rms_abs_diff"]) for item in items]),
        "allclose_5e_2_count": int(sum(1 for item in items if item[key]["allclose_atol_5e_2_rtol_5e_2"])),
    }


@torch.inference_mode()
def run_prefill_logits(
    *,
    model: LocalPaddleOCRVLForConditionalGeneration,
    item,
    config: AttentionConfig,
    device: torch.device,
    cache_length: int,
) -> dict[str, Any]:
    apply_attention_config(config)
    maybe_sync(device)
    start = time.perf_counter()

    input_ids = item.input_ids.to(device)
    attention_mask = item.attention_mask.to(device)
    pixel_values = item.pixel_values.to(device=device, dtype=model.visual.dtype)
    image_grid_thw = item.image_grid_thw

    pixel_values_batched = pixel_values.type(model.visual.dtype).unsqueeze(0)
    cu_seqlens = torch.repeat_interleave(
        image_grid_thw[:, 1] * image_grid_thw[:, 2],
        image_grid_thw[:, 0],
    ).cumsum(dim=0, dtype=torch.int32)
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
    image_features = model.visual(
        pixel_values=pixel_values_batched,
        image_grid_thw=image_grid_thw,
        cu_seqlens=cu_seqlens,
    )
    image_embeds = model.mlp_AR(image_features, image_grid_thw)

    inputs_embeds = model.model.embed_tokens(input_ids)
    image_mask = (input_ids == model.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
    image_token_count = int((input_ids == model.config.image_token_id).sum().item())
    if image_token_count * int(inputs_embeds.shape[-1]) != int(image_embeds.numel()):
        raise ValueError(
            "image features and image tokens do not match: "
            f"tokens={image_token_count} features={int(image_embeds.shape[0])}"
        )
    inputs_embeds = inputs_embeds.masked_scatter(
        image_mask,
        image_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype),
    )

    position_ids, rope_deltas = model.get_rope_index(input_ids, image_grid_thw, attention_mask)
    cache = model.allocate_static_cache(
        batch_size=int(input_ids.shape[0]),
        cache_length=int(cache_length),
        device=inputs_embeds.device,
        dtype=inputs_embeds.dtype,
        init_mode="zeros",
    )
    hidden_states = model.model.forward_prefill_static(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        cache=cache,
    )
    logits = model.lm_head(hidden_states[:, -1:, :])
    next_token = torch.argmax(logits[:, -1, :].float(), dim=-1, keepdim=True)
    maybe_sync(device)
    elapsed_s = time.perf_counter() - start
    return {
        "image_embeds": image_embeds.detach(),
        "hidden_last": hidden_states[:, -1:, :].detach(),
        "logits": logits.detach(),
        "next_token": next_token.detach(),
        "rope_deltas": rope_deltas.detach(),
        "elapsed_s": float(elapsed_s),
        "vision_tokens": int(image_features.shape[0]),
        "projected_image_tokens": int(image_embeds.shape[0]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="PaddlePaddle/PaddleOCR-VL-1.6")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "crops" / "hotswap_100_manifest.json")
    parser.add_argument("--num-items", type=int, default=8)
    parser.add_argument("--crop-ids", nargs="*", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "float16", "bf16", "bfloat16"])
    parser.add_argument("--cache-length", type=int, default=2048)
    parser.add_argument("--npu-jit-compile", default="off", choices=NPU_JIT_COMPILE_CHOICES)
    parser.add_argument("--baseline-name", default="manual_fp32")
    parser.add_argument("--baseline-vision-attention", default="manual", choices=VISION_ATTENTION_CHOICES)
    parser.add_argument("--baseline-vision-prompt-fa-layout", default="bnsd", choices=VISION_PROMPT_FA_LAYOUT_CHOICES)
    parser.add_argument("--baseline-text-softmax-dtype", default="fp32", choices=SOFTMAX_DTYPE_CHOICES)
    parser.add_argument("--baseline-vision-softmax-dtype", default="fp32", choices=SOFTMAX_DTYPE_CHOICES)
    parser.add_argument("--candidate-name", default="manual_model")
    parser.add_argument("--candidate-vision-attention", default="manual", choices=VISION_ATTENTION_CHOICES)
    parser.add_argument("--candidate-vision-prompt-fa-layout", default="bnsd", choices=VISION_PROMPT_FA_LAYOUT_CHOICES)
    parser.add_argument("--candidate-text-softmax-dtype", default="model", choices=SOFTMAX_DTYPE_CHOICES)
    parser.add_argument("--candidate-vision-softmax-dtype", default="model", choices=SOFTMAX_DTYPE_CHOICES)
    parser.add_argument("--fail-on-argmax-mismatch", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if int(args.num_items) <= 0:
        raise ValueError(f"--num-items must be positive, got {args.num_items}")
    if int(args.cache_length) <= 0:
        raise ValueError(f"--cache-length must be positive, got {args.cache_length}")

    device = resolve_device(args.device)
    configure_npu_jit_compile(args.npu_jit_compile, device)
    dtype = parse_dtype(args.dtype, device)
    model_dir = _resolve_model_dir(args.model)
    pre_cfg = load_preprocessor_config(model_dir)
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    manifest = load_manifest(args.manifest)
    entries = select_manifest_entries(manifest, num_items=int(args.num_items), crop_ids=args.crop_ids)
    items = build_cohort_inputs(
        entries=entries,
        manifest_path=args.manifest,
        tokenizer=tokenizer,
        pre_cfg=pre_cfg,
        prompt_override=args.prompt,
    )

    model_load_start = time.perf_counter()
    model = LocalPaddleOCRVLForConditionalGeneration.from_pretrained(model_dir, dtype=dtype, device=device)
    maybe_sync(device)
    model_load_s = time.perf_counter() - model_load_start

    baseline_config = AttentionConfig(
        name=str(args.baseline_name),
        vision_attention=str(args.baseline_vision_attention),
        vision_prompt_fa_layout=str(args.baseline_vision_prompt_fa_layout),
        text_softmax_dtype=str(args.baseline_text_softmax_dtype),
        vision_softmax_dtype=str(args.baseline_vision_softmax_dtype),
    )
    candidate_config = AttentionConfig(
        name=str(args.candidate_name),
        vision_attention=str(args.candidate_vision_attention),
        vision_prompt_fa_layout=str(args.candidate_vision_prompt_fa_layout),
        text_softmax_dtype=str(args.candidate_text_softmax_dtype),
        vision_softmax_dtype=str(args.candidate_vision_softmax_dtype),
    )

    rows = []
    for idx, item in enumerate(items):
        baseline = run_prefill_logits(
            model=model,
            item=item,
            config=baseline_config,
            device=device,
            cache_length=int(args.cache_length),
        )
        candidate = run_prefill_logits(
            model=model,
            item=item,
            config=candidate_config,
            device=device,
            cache_length=int(args.cache_length),
        )
        baseline_token = int(baseline["next_token"].detach().cpu().view(-1)[0].item())
        candidate_token = int(candidate["next_token"].detach().cpu().view(-1)[0].item())
        row = {
            "idx": int(idx),
            "id": str(item.entry.get("id")),
            "category_type": item.entry.get("category_type"),
            "crop_size": item.entry.get("crop_size"),
            "input_tokens": int(item.input_ids.shape[1]),
            "vision_tokens": int(baseline["vision_tokens"]),
            "projected_image_tokens": int(baseline["projected_image_tokens"]),
            "baseline_elapsed_s": float(baseline["elapsed_s"]),
            "candidate_elapsed_s": float(candidate["elapsed_s"]),
            "projected_image_embeddings": diff_stats(candidate["image_embeds"], baseline["image_embeds"]),
            "prefill_hidden_last": diff_stats(candidate["hidden_last"], baseline["hidden_last"]),
            "prefill_logits": diff_stats(candidate["logits"], baseline["logits"]),
            "baseline_next_token": baseline_token,
            "candidate_next_token": candidate_token,
            "next_token_match": bool(baseline_token == candidate_token),
        }
        rows.append(row)

    argmax_mismatches = [row for row in rows if not row["next_token_match"]]
    output = {
        "comparison": "attention_prefill_logits",
        "model": str(model_dir),
        "device": str(device),
        "dtype": str(dtype),
        "cache_length": int(args.cache_length),
        "num_items": int(len(rows)),
        "model_load_s": float(model_load_s),
        "baseline": baseline_config.__dict__,
        "candidate": candidate_config.__dict__,
        "summary": {
            "next_token_mismatch_count": int(len(argmax_mismatches)),
            "projected_image_embeddings": aggregate_diff(rows, "projected_image_embeddings"),
            "prefill_hidden_last": aggregate_diff(rows, "prefill_hidden_last"),
            "prefill_logits": aggregate_diff(rows, "prefill_logits"),
            "baseline_elapsed_s": stats([float(row["baseline_elapsed_s"]) for row in rows]),
            "candidate_elapsed_s": stats([float(row["candidate_elapsed_s"]) for row in rows]),
            "sample_argmax_mismatches": argmax_mismatches[:8],
        },
        "items": rows,
    }
    print(json.dumps(output, indent=2 if args.json else None, sort_keys=True, default=json_default))
    if args.fail_on_argmax_mismatch and argmax_mismatches:
        raise SystemExit(f"candidate changed first-token argmax for {len(argmax_mismatches)} item(s)")


if __name__ == "__main__":
    main()
