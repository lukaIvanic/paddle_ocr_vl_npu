#!/usr/bin/env python3
"""Probe the inherited static-cache decode path with torch.compile(fullgraph=True, dynamic=False)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = EXPERIMENT_ROOT.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

from paddleocr_vl.model.text_decode import (
    TextDecodeStage,
    compile_text_decode_stage,
    decode_attention_label,
    decode_cache_update_label,
)
from paddleocr_vl.model.modeling import (
    LocalPaddleOCRVLForConditionalGeneration,
    _resolve_model_dir,
    cast_decode_linear_weights_to_nz,
)
from paddleocr_vl.model.preprocessing import (
    build_inputs,
    load_preprocessor_config,
    preprocess_image,
)
from utils.timing import synchronize


DEFAULT_TORCHAIR_CACHE_DIR = Path("outputs") / "torchair_cache"


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="PaddlePaddle/PaddleOCR-VL-1.6")
    parser.add_argument("--crop", default="crops/crop_01_text_block_en.png")
    parser.add_argument("--prompt", default="OCR:")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--cache-length", type=int, default=None)
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "float16", "bf16", "bfloat16"])
    parser.add_argument("--backend", default="eager", choices=["raw_eager", "eager", "aot_eager", "inductor", "default", "torchair"])
    parser.add_argument("--torchair-cache-dir", type=Path, default=DEFAULT_TORCHAIR_CACHE_DIR)
    args = parser.parse_args()

    model_dir = _resolve_model_dir(args.model)
    crop = Path(args.crop)
    if not crop.exists():
        crop = REPO_ROOT / args.crop
    import torch_npu  # noqa: F401

    device = torch.device("npu:0")
    if not torch.npu.is_available():
        raise RuntimeError("The decode compile probe requires an available NPU")
    dtype = (
        torch.float16
        if args.dtype in {"fp16", "float16"}
        else torch.bfloat16
    )
    torch.npu.set_compile_mode(jit_compile=False)

    pre_cfg = load_preprocessor_config(model_dir)
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    pixel_values, image_grid_thw = preprocess_image(crop, pre_cfg)
    input_ids, attention_mask = build_inputs(tokenizer, image_grid_thw, args.prompt, merge_size=int(pre_cfg["merge_size"]))

    model = LocalPaddleOCRVLForConditionalGeneration.from_pretrained(model_dir, dtype=dtype, device=device)
    synchronize(device)
    weight_format_start = time.perf_counter()
    weight_format_meta = cast_decode_linear_weights_to_nz(model)
    synchronize(device)
    weight_format_meta["setup_s"] = time.perf_counter() - weight_format_start
    pixel_values = pixel_values.to(device=device, dtype=model.visual.dtype)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    prompt_length = int(input_ids.shape[1])
    min_cache_length = prompt_length + max(0, int(args.max_new_tokens) - 1)
    cache_length = int(
        args.cache_length
        if args.cache_length is not None
        else (prompt_length + int(args.max_new_tokens))
    )
    if cache_length < min_cache_length:
        raise ValueError(
            f"--cache-length={cache_length} is too small for prompt length {prompt_length} "
            f"and --max-new-tokens={args.max_new_tokens}; need at least {min_cache_length}"
        )

    dynamic_ids = model.generate_ids(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        max_new_tokens=args.max_new_tokens,
    )
    static_ids = model.generate_ids_static(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        max_new_tokens=args.max_new_tokens,
        cache_length=cache_length,
    )
    print(f"static_matches_dynamic={bool(torch.equal(static_ids, dynamic_ids))}")
    print(f"cache_update=prefill_slice_decode_{decode_cache_update_label(device)} npu_jit_compile=False")
    print(f"dynamic_text={tokenizer.decode(dynamic_ids[0].detach().cpu().tolist(), skip_special_tokens=True)!r}")
    print(f"static_text={tokenizer.decode(static_ids[0].detach().cpu().tolist(), skip_special_tokens=True)!r}")

    prefill = model.forward_static_prefill(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        cache_length=cache_length,
        logits_to_keep=1,
    )
    next_token = torch.argmax(prefill.logits[:, -1, :].float(), dim=-1, keepdim=True)
    decode_stage = TextDecodeStage(model).eval()

    synchronize(device)
    start = time.perf_counter()
    compiled_decode, compile_meta = compile_text_decode_stage(
        decode_stage,
        backend_name=args.backend,
        device=device,
        cache_root=args.torchair_cache_dir,
        batch_size=int(input_ids.shape[0]),
        cache_length=cache_length,
        dtype=dtype,
        model_dir=model_dir,
        linear_weight_format=str(weight_format_meta["effective_mode"]),
    )
    synchronize(device)
    compile_setup_s = time.perf_counter() - start

    flat_cache = prefill.cache.flat_tensors()
    synchronize(device)
    start = time.perf_counter()
    eager_logits = decode_stage(next_token, prefill.next_cache_position, prefill.rope_deltas, *flat_cache)
    synchronize(device)
    eager_s = time.perf_counter() - start

    synchronize(device)
    start = time.perf_counter()
    compiled_logits = compiled_decode(next_token, prefill.next_cache_position, prefill.rope_deltas, *flat_cache)
    synchronize(device)
    compiled_first_s = time.perf_counter() - start

    diff = (eager_logits.float() - compiled_logits.float()).abs()
    print(f"compile_backend={args.backend} fullgraph=True dynamic=False")
    print("compile_meta=" + repr(compile_meta))
    print(f"decode_attention={decode_attention_label(device)}")
    print("linear_weight_format=" + repr(weight_format_meta))
    print(f"compile_setup_s={compile_setup_s:.6f} eager_decode_s={eager_s:.6f} compiled_first_s={compiled_first_s:.6f}")
    print(f"compiled_matches_eager=max_abs:{float(diff.max())} mean_abs:{float(diff.mean())}")
    print(f"compiled_next_token={int(torch.argmax(compiled_logits[:, -1, :].float(), dim=-1).item())}")


if __name__ == "__main__":
    main()
