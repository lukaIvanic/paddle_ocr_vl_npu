#!/usr/bin/env python3
"""Compare compiled visual output after the projector, prefill, and OCR decode.

This is a correctness diagnostic, not a throughput benchmark. It runs one real
OmniDocBench crop through the same static_visual wrapper twice:

1. eager static_visual
2. compiled static_visual

Then it feeds both visual outputs through the same adaptive MLP projector, text
prefill, LM head, and static OCR decode path. The purpose is to answer whether a
raw visual feature mismatch actually changes decoder-facing image embeddings,
prefill logits, first-token argmax, or generated OCR text.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
EXP5_DIR = REPO_ROOT / "05_full_recognizer_optimizations"
if str(EXP5_DIR) not in sys.path:
    sys.path.insert(0, str(EXP5_DIR))

from bench_page_pipeline_e2e import (  # noqa: E402
    build_detected_crops,
    build_omnidocbench_gt_layout_pages,
    build_queue_inputs_from_crops,
    clean_json,
    load_pages_result,
)
from bench_recognizer_queue import QueueInput  # noqa: E402
from local_modeling_paddleocr_vl import (  # noqa: E402
    VISION_ATTENTION_CHOICES,
    VISION_ATTENTION_ENV,
    VISION_PROMPT_FA_LAYOUT_CHOICES,
    VISION_PROMPT_FA_LAYOUT_ENV,
    LocalPaddleOCRVLForConditionalGeneration,
    _resolve_model_dir,
    get_vision_attention_impl,
    get_vision_prompt_fa_layout,
)
from probe_static_compile import maybe_sync  # noqa: E402
from run_local_recognition import (  # noqa: E402
    NPU_JIT_COMPILE_CHOICES,
    configure_npu_jit_compile,
    load_preprocessor_config,
    resolve_device,
)

from bench_vision_prefill_only import (  # noqa: E402
    CROP_SAMPLE_CHOICES,
    VISION_COMPILE_BACKEND_CHOICES,
    SingleCropVisionFeatureModule,
    compile_single_crop_vision_forward,
    parse_vision_dtype,
    select_profile_crop_sample,
    tensor_grid,
    vision_tokens,
)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return str(value)


def diff_stats(lhs: torch.Tensor, rhs: torch.Tensor) -> dict[str, Any]:
    lhs_cpu = lhs.detach().to(dtype=torch.float32).cpu()
    rhs_cpu = rhs.detach().to(dtype=torch.float32).cpu()
    diff = (lhs_cpu - rhs_cpu).abs()
    if diff.numel():
        flat_idx = int(diff.argmax().item())
        max_abs = float(diff.max().item())
        mean_abs = float(diff.mean().item())
        rms_abs = float(torch.sqrt(torch.mean(diff * diff)).item())
        p99_abs = float(torch.quantile(diff.flatten(), 0.99).item())
        p999_abs = float(torch.quantile(diff.flatten(), 0.999).item())
    else:
        flat_idx = 0
        max_abs = mean_abs = rms_abs = p99_abs = p999_abs = 0.0
    return {
        "shape": [int(value) for value in lhs.shape],
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "rms_abs_diff": rms_abs,
        "p99_abs_diff": p99_abs,
        "p999_abs_diff": p999_abs,
        "max_abs_flat_index": flat_idx,
        "allclose_atol_5e_2_rtol_5e_2": bool(torch.allclose(lhs_cpu, rhs_cpu, atol=5e-2, rtol=5e-2)),
        "allclose_atol_1e_1_rtol_1e_1": bool(torch.allclose(lhs_cpu, rhs_cpu, atol=1e-1, rtol=1e-1)),
        "allclose_atol_1e_0_rtol_1e_0": bool(torch.allclose(lhs_cpu, rhs_cpu, atol=1.0, rtol=1.0)),
    }


def trim_after_eos(tokens: list[int], eos_token_id: int) -> list[int]:
    if eos_token_id in tokens:
        return tokens[: tokens.index(eos_token_id) + 1]
    return tokens


def first_mismatch(lhs: list[int], rhs: list[int]) -> dict[str, Any] | None:
    limit = min(len(lhs), len(rhs))
    for idx in range(limit):
        if int(lhs[idx]) != int(rhs[idx]):
            return {"position": int(idx), "lhs": int(lhs[idx]), "rhs": int(rhs[idx])}
    if len(lhs) != len(rhs):
        return {"position": int(limit), "lhs": lhs[limit:] if len(lhs) > limit else None, "rhs": rhs[limit:] if len(rhs) > limit else None}
    return None


def scatter_projected_image_embeds(
    *,
    model: LocalPaddleOCRVLForConditionalGeneration,
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    image_embeds: torch.Tensor,
) -> torch.Tensor:
    if int(input_ids.shape[0]) != 1:
        raise ValueError("downstream compiled-visual comparison currently expects batch size 1")
    projected = image_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
    positions = torch.nonzero(input_ids[0] == int(model.config.image_token_id), as_tuple=False).flatten()
    if int(positions.numel()) != int(projected.shape[0]):
        raise ValueError(
            "image features and image tokens do not match: "
            f"tokens={int(positions.numel())} features={int(projected.shape[0])}"
        )
    flat_embeds = inputs_embeds[0].clone()
    flat_embeds.index_copy_(0, positions.to(flat_embeds.device), projected)
    return flat_embeds.unsqueeze(0)


@torch.inference_mode()
def prefill_from_visual_features(
    *,
    model: LocalPaddleOCRVLForConditionalGeneration,
    item: QueueInput,
    image_features: torch.Tensor,
    device: torch.device,
    cache_length: int,
) -> dict[str, Any]:
    input_ids = item.input_ids.to(device)
    attention_mask = item.attention_mask.to(device)
    image_grid_thw = item.image_grid_thw
    image_embeds = model.mlp_AR(image_features, image_grid_thw)

    inputs_embeds = model.model.embed_tokens(input_ids)
    inputs_embeds = scatter_projected_image_embeds(
        model=model,
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        image_embeds=image_embeds,
    )

    position_ids_cpu, rope_deltas_cpu = model.get_rope_index(item.input_ids, item.image_grid_thw, item.attention_mask)
    position_ids = position_ids_cpu.to(device)
    rope_deltas = rope_deltas_cpu.to(device)
    cache = model.allocate_static_cache(
        batch_size=int(inputs_embeds.shape[0]),
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
    cache_position = torch.full((int(input_ids.shape[0]),), int(input_ids.shape[1]), device=device, dtype=torch.int64)
    return {
        "image_embeds": image_embeds.detach(),
        "inputs_embeds": inputs_embeds.detach(),
        "hidden_last": hidden_states[:, -1:, :].detach(),
        "logits": logits.detach(),
        "next_token": next_token.detach(),
        "cache": cache,
        "rope_deltas": rope_deltas,
        "cache_position": cache_position,
    }


@torch.inference_mode()
def generate_from_prefill(
    *,
    model: LocalPaddleOCRVLForConditionalGeneration,
    prefill: dict[str, Any],
    max_new_tokens: int,
    eos_token_id: int,
) -> torch.Tensor:
    next_token = prefill["next_token"]
    cache = prefill["cache"]
    rope_deltas = prefill["rope_deltas"]
    cache_position = prefill["cache_position"]
    generated = [next_token]
    finished = next_token.squeeze(1) == int(eos_token_id)
    for _ in range(max(0, int(max_new_tokens) - 1)):
        if bool(finished.all().item()):
            break
        outputs_decode = model.forward_static_decode(
            input_ids=next_token,
            cache=cache,
            cache_position=cache_position,
            rope_deltas=rope_deltas,
            logits_to_keep=1,
        )
        next_token = torch.argmax(outputs_decode.logits[:, -1, :].float(), dim=-1, keepdim=True)
        next_token = torch.where(finished.view(-1, 1), torch.full_like(next_token, int(eos_token_id)), next_token)
        generated.append(next_token)
        finished |= next_token.squeeze(1) == int(eos_token_id)
        cache_position = cache_position + 1
    return torch.cat(generated, dim=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="PaddlePaddle/PaddleOCR-VL-1.6")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--page-start", type=int, default=0)
    parser.add_argument("--num-pages", type=int, default=8)
    parser.add_argument("--max-crops", type=int, default=0)
    parser.add_argument("--crop-padding", type=int, default=0)
    parser.add_argument("--min-crop-side", type=int, default=4)
    parser.add_argument("--skip-labels", default="")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "float16", "fp32", "float32", "bf16", "bfloat16"])
    parser.add_argument("--npu-jit-compile", default="off", choices=NPU_JIT_COMPILE_CHOICES)
    parser.add_argument("--vision-attention", default=os.environ.get(VISION_ATTENTION_ENV, "manual"), choices=VISION_ATTENTION_CHOICES)
    parser.add_argument(
        "--vision-prompt-fa-layout",
        default=os.environ.get(VISION_PROMPT_FA_LAYOUT_ENV, "bnsd"),
        choices=VISION_PROMPT_FA_LAYOUT_CHOICES,
    )
    parser.add_argument("--vision-compile-backend", default="torchair", choices=VISION_COMPILE_BACKEND_CHOICES)
    parser.add_argument("--crop-sample", default="small_only", choices=CROP_SAMPLE_CHOICES)
    parser.add_argument("--cache-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--fail-on-token-mismatch", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if int(args.cache_length) <= 0:
        raise ValueError("--cache-length must be positive")
    if int(args.max_new_tokens) <= 0:
        raise ValueError("--max-new-tokens must be positive")

    os.environ[VISION_ATTENTION_ENV] = str(args.vision_attention)
    os.environ[VISION_PROMPT_FA_LAYOUT_ENV] = str(args.vision_prompt_fa_layout)

    device = resolve_device(args.device)
    configure_npu_jit_compile(args.npu_jit_compile, device)
    dtype = parse_vision_dtype(args.dtype)
    model_dir = _resolve_model_dir(args.model)
    pre_cfg = load_preprocessor_config(model_dir)
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))

    page_load = load_pages_result(args.dataset_dir, page_start=int(args.page_start), num_pages=int(args.num_pages))
    layout_pages, _layout_timing = build_omnidocbench_gt_layout_pages(page_load.pages, include_ignored=False, include_empty_gt=False)
    crops, crop_summary, _crop_timing = build_detected_crops(pages=page_load.pages, layout_pages=layout_pages, args=args)
    if int(args.max_crops) > 0:
        crops = crops[: int(args.max_crops)]
    queue_inputs, input_build_summary = build_queue_inputs_from_crops(
        crops=crops,
        tokenizer=tokenizer,
        pre_cfg=pre_cfg,
        prompt_override=args.prompt,
    )
    queue_inputs, crop_sample_summary = select_profile_crop_sample(queue_inputs, strategy=str(args.crop_sample))
    if len(queue_inputs) != 1:
        raise ValueError(
            "compiled visual downstream compare is shape-specialized and expects exactly one selected crop; "
            f"got {len(queue_inputs)} from --crop-sample {args.crop_sample!r}"
        )
    item = queue_inputs[0]

    model_load_start = time.perf_counter()
    model = LocalPaddleOCRVLForConditionalGeneration.from_pretrained(model_dir, dtype=dtype, device=device)
    maybe_sync(device)
    model_load_s = time.perf_counter() - model_load_start

    wrapper = SingleCropVisionFeatureModule(model, item.image_grid_thw, boundary="static_visual", device=device).eval()
    pixel_values = item.pixel_values.to(device=device, dtype=model.visual.dtype)

    maybe_sync(device)
    eager_visual_start = time.perf_counter()
    eager_visual = wrapper(pixel_values)
    maybe_sync(device)
    eager_visual_s = time.perf_counter() - eager_visual_start

    compiled_visual_fn, compile_meta = compile_single_crop_vision_forward(
        model=model,
        item=item,
        device=device,
        backend_name=str(args.vision_compile_backend),
        boundary="static_visual",
        wrapper=wrapper,
    )
    if compiled_visual_fn is None:
        compiled_visual_fn = wrapper

    maybe_sync(device)
    compiled_visual_start = time.perf_counter()
    compiled_visual = compiled_visual_fn(pixel_values)
    maybe_sync(device)
    compiled_visual_s = time.perf_counter() - compiled_visual_start
    compile_meta["compiled_first_call_s"] = float(compiled_visual_s)

    maybe_sync(device)
    eager_prefill_start = time.perf_counter()
    eager_prefill = prefill_from_visual_features(
        model=model,
        item=item,
        image_features=eager_visual,
        device=device,
        cache_length=int(args.cache_length),
    )
    maybe_sync(device)
    eager_prefill_s = time.perf_counter() - eager_prefill_start

    maybe_sync(device)
    compiled_prefill_start = time.perf_counter()
    compiled_prefill = prefill_from_visual_features(
        model=model,
        item=item,
        image_features=compiled_visual,
        device=device,
        cache_length=int(args.cache_length),
    )
    maybe_sync(device)
    compiled_prefill_s = time.perf_counter() - compiled_prefill_start

    eos_token_id = int(model.config.eos_token_id)
    maybe_sync(device)
    eager_generate_start = time.perf_counter()
    eager_ids = generate_from_prefill(
        model=model,
        prefill=eager_prefill,
        max_new_tokens=int(args.max_new_tokens),
        eos_token_id=eos_token_id,
    )
    maybe_sync(device)
    eager_generate_s = time.perf_counter() - eager_generate_start

    maybe_sync(device)
    compiled_generate_start = time.perf_counter()
    compiled_ids = generate_from_prefill(
        model=model,
        prefill=compiled_prefill,
        max_new_tokens=int(args.max_new_tokens),
        eos_token_id=eos_token_id,
    )
    maybe_sync(device)
    compiled_generate_s = time.perf_counter() - compiled_generate_start

    eager_token_list = [int(v) for v in eager_ids[0].detach().cpu().tolist()]
    compiled_token_list = [int(v) for v in compiled_ids[0].detach().cpu().tolist()]
    eager_trimmed = trim_after_eos(eager_token_list, eos_token_id)
    compiled_trimmed = trim_after_eos(compiled_token_list, eos_token_id)
    eager_text = tokenizer.decode(eager_trimmed, skip_special_tokens=True)
    compiled_text = tokenizer.decode(compiled_trimmed, skip_special_tokens=True)
    token_mismatch = first_mismatch(eager_trimmed, compiled_trimmed)

    eager_next = int(eager_prefill["next_token"].detach().cpu().view(-1)[0].item())
    compiled_next = int(compiled_prefill["next_token"].detach().cpu().view(-1)[0].item())
    output = {
        "comparison": "compiled_visual_downstream",
        "model": str(model_dir),
        "device": str(device),
        "dtype": str(dtype),
        "npu_jit_compile": str(args.npu_jit_compile),
        "vision_attention": get_vision_attention_impl(),
        "vision_prompt_fa_layout": get_vision_prompt_fa_layout(),
        "vision_compile": clean_json(compile_meta),
        "page_start": int(args.page_start),
        "num_pages": int(args.num_pages),
        "crop_sample": str(args.crop_sample),
        "crop_summary": clean_json(crop_summary),
        "crop_sample_summary": clean_json(crop_sample_summary),
        "input_build_summary": clean_json(input_build_summary),
        "item": {
            "id": str(item.entry.get("id")),
            "category_type": item.entry.get("category_type"),
            "crop_size": item.entry.get("crop_size"),
            "input_tokens": int(item.input_ids.shape[1]),
            "image_grid_thw": tensor_grid(item),
            "vision_tokens": int(vision_tokens(item)),
            "projected_image_tokens": int(item.image_grid_thw.prod().item() // 4),
        },
        "timing_s": {
            "model_load": float(model_load_s),
            "eager_visual": float(eager_visual_s),
            "compiled_visual_first_call": float(compiled_visual_s),
            "eager_projector_prefill": float(eager_prefill_s),
            "compiled_projector_prefill": float(compiled_prefill_s),
            "eager_decode": float(eager_generate_s),
            "compiled_visual_decode": float(compiled_generate_s),
        },
        "diffs": {
            "visual_post_layernorm": diff_stats(compiled_visual, eager_visual),
            "projected_image_embeddings": diff_stats(compiled_prefill["image_embeds"], eager_prefill["image_embeds"]),
            "text_inputs_embeds_after_scatter": diff_stats(compiled_prefill["inputs_embeds"], eager_prefill["inputs_embeds"]),
            "prefill_hidden_last": diff_stats(compiled_prefill["hidden_last"], eager_prefill["hidden_last"]),
            "prefill_logits": diff_stats(compiled_prefill["logits"], eager_prefill["logits"]),
        },
        "tokens": {
            "eager_first_token": eager_next,
            "compiled_visual_first_token": compiled_next,
            "first_token_match": bool(eager_next == compiled_next),
            "eager_generated_trimmed": eager_trimmed,
            "compiled_visual_generated_trimmed": compiled_trimmed,
            "generated_trimmed_match": bool(eager_trimmed == compiled_trimmed),
            "first_mismatch": token_mismatch,
        },
        "texts": {
            "eager": eager_text,
            "compiled_visual": compiled_text,
            "match": bool(eager_text == compiled_text),
        },
    }
    print(json.dumps(output, indent=2 if args.json else None, sort_keys=True, default=json_default))
    if bool(args.fail_on_token_mismatch) and eager_trimmed != compiled_trimmed:
        raise SystemExit(f"compiled visual changed generated tokens: {token_mismatch}")


if __name__ == "__main__":
    main()
