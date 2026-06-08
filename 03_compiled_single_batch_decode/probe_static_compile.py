#!/usr/bin/env python3
"""Probe experiment-3 static-cache decode with torch.compile(fullgraph=True, dynamic=False)."""

from __future__ import annotations

import argparse
import importlib
import time
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

from local_modeling_paddleocr_vl import (
    DECODE_ATTENTION_MODE_CHOICES,
    DECODE_LINEAR_WEIGHT_FORMAT,
    LocalPaddleOCRVLForConditionalGeneration,
    _resolve_model_dir,
    cast_decode_linear_weights_to_nz,
)
from run_local_recognition import (
    NPU_JIT_COMPILE_CHOICES,
    build_inputs,
    configure_npu_jit_compile,
    load_preprocessor_config,
    parse_dtype,
    preprocess_image,
    resolve_device,
)


DEFAULT_TORCHAIR_CACHE_DIR = Path("outputs") / "torchair_cache"


def import_torchair():
    try:
        import torchair

        CompilerConfig = torchair.CompilerConfig
    except Exception as direct_error:
        try:
            from torch_npu.dynamo import torchair
            from torch_npu.dynamo.torchair.configs.compiler_config import CompilerConfig

        except Exception as fallback_error:
            raise RuntimeError(
                "TorchAir is unavailable: direct `import torchair` failed with "
                f"{direct_error!r}, and `from torch_npu.dynamo import torchair` "
                f"failed with {fallback_error!r}."
            ) from fallback_error

    if not hasattr(torchair, "inference"):
        torchair.inference = importlib.import_module(f"{torchair.__name__}.inference")
    return torchair, CompilerConfig


def compile_backend(name: str):
    if name == "default":
        return None
    if name == "torchair":
        torchair, CompilerConfig = import_torchair()
        config = CompilerConfig()

        return torchair.get_npu_backend(compiler_config=config)
    return name


def torchair_cache_dir_for_shape(cache_root: Path, *, batch_size: int, cache_length: int, decode_attention: str) -> Path:
    if decode_attention not in DECODE_ATTENTION_MODE_CHOICES:
        raise ValueError(f"unsupported decode_attention={decode_attention!r}")
    shape_key = f"{DECODE_LINEAR_WEIGHT_FORMAT}_{decode_attention}_bs{int(batch_size)}_cache{int(cache_length)}"
    return cache_root.expanduser().resolve() / shape_key


def compile_decode_module(
    flat_decode: torch.nn.Module,
    *,
    backend_name: str,
    device: torch.device,
    cache_root: Path,
    batch_size: int,
    cache_length: int,
    decode_attention: str,
) -> tuple[Any, dict[str, Any]]:
    if backend_name == "torchair":
        if device.type != "npu":
            raise ValueError("--backend torchair requires an NPU device.")
        torchair, CompilerConfig = import_torchair()
        config = CompilerConfig()
        shape_cache_dir = torchair_cache_dir_for_shape(
            cache_root,
            batch_size=batch_size,
            cache_length=cache_length,
            decode_attention=decode_attention,
        )
        shape_cache_dir.mkdir(parents=True, exist_ok=True)
        compiled_decode = torchair.inference.cache_compile(
            flat_decode.forward,
            config=config,
            dynamic=False,
            cache_dir=str(shape_cache_dir),
            ge_cache=True,
        )
        return compiled_decode, {
            "backend": backend_name,
            "torchair_cache_dir": str(shape_cache_dir),
            "torchair_ge_cache": True,
            "compile_api": "torchair.inference.cache_compile",
            "linear_weight_format": DECODE_LINEAR_WEIGHT_FORMAT,
            "decode_attention": decode_attention,
        }

    backend = compile_backend(backend_name)
    compile_kwargs = {"fullgraph": True, "dynamic": False}
    if backend is not None:
        compile_kwargs["backend"] = backend
    return torch.compile(flat_decode, **compile_kwargs), {
        "backend": backend_name,
        "compile_api": "torch.compile",
        "linear_weight_format": DECODE_LINEAR_WEIGHT_FORMAT,
        "decode_attention": decode_attention,
    }


def maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "npu":
        import torch_npu

        torch_npu.npu.synchronize()


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="PaddlePaddle/PaddleOCR-VL-1.6")
    parser.add_argument("--crop", default="crops/crop_01_text_block_en.png")
    parser.add_argument("--prompt", default="OCR:")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--cache-length", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "float16", "bf16", "bfloat16"])
    parser.add_argument("--backend", default="eager", choices=["eager", "aot_eager", "inductor", "default", "torchair"])
    parser.add_argument("--npu-jit-compile", default="off", choices=NPU_JIT_COMPILE_CHOICES)
    parser.add_argument("--torchair-cache-dir", type=Path, default=DEFAULT_TORCHAIR_CACHE_DIR)
    parser.add_argument("--decode-attention", default="manual", choices=DECODE_ATTENTION_MODE_CHOICES)
    args = parser.parse_args()

    model_dir = _resolve_model_dir(args.model)
    crop = Path(args.crop)
    if not crop.exists():
        crop = Path(__file__).resolve().parents[1] / args.crop
    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype, device)
    configure_npu_jit_compile(args.npu_jit_compile, device)

    pre_cfg = load_preprocessor_config(model_dir)
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    pixel_values, image_grid_thw = preprocess_image(crop, pre_cfg)
    input_ids, attention_mask = build_inputs(tokenizer, image_grid_thw, args.prompt, merge_size=int(pre_cfg["merge_size"]))

    model = LocalPaddleOCRVLForConditionalGeneration.from_pretrained(model_dir, dtype=dtype, device=device)
    model.set_decode_attention_mode(args.decode_attention)
    maybe_sync(device)
    weight_format_start = time.perf_counter()
    weight_format_meta = cast_decode_linear_weights_to_nz(model)
    maybe_sync(device)
    weight_format_meta["setup_s"] = time.perf_counter() - weight_format_start
    pixel_values = pixel_values.to(device)
    image_grid_thw = image_grid_thw.to(device)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    cache_length = int(args.cache_length or (input_ids.shape[1] + args.max_new_tokens))

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
    print(f"cache_update=prefill_slice_decode_npu_scatter npu_jit_compile={args.npu_jit_compile}")
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
    flat_decode = model.make_flat_static_decode_module().eval()

    maybe_sync(device)
    start = time.perf_counter()
    compiled_decode, compile_meta = compile_decode_module(
        flat_decode,
        backend_name=args.backend,
        device=device,
        cache_root=args.torchair_cache_dir,
        batch_size=int(input_ids.shape[0]),
        cache_length=cache_length,
        decode_attention=args.decode_attention,
    )
    maybe_sync(device)
    compile_setup_s = time.perf_counter() - start

    flat_cache = prefill.cache.flat_tensors()
    maybe_sync(device)
    start = time.perf_counter()
    eager_logits = flat_decode(next_token, prefill.next_cache_position, prefill.rope_deltas, *flat_cache)
    maybe_sync(device)
    eager_s = time.perf_counter() - start

    maybe_sync(device)
    start = time.perf_counter()
    compiled_logits = compiled_decode(next_token, prefill.next_cache_position, prefill.rope_deltas, *flat_cache)
    maybe_sync(device)
    compiled_first_s = time.perf_counter() - start

    diff = (eager_logits.float() - compiled_logits.float()).abs()
    print(f"compile_backend={args.backend} fullgraph=True dynamic=False")
    print("compile_meta=" + repr(compile_meta))
    print(f"decode_attention={args.decode_attention}")
    print("linear_weight_format=" + repr(weight_format_meta))
    print(f"compile_setup_s={compile_setup_s:.6f} eager_decode_s={eager_s:.6f} compiled_first_s={compiled_first_s:.6f}")
    print(f"compiled_matches_eager=max_abs:{float(diff.max())} mean_abs:{float(diff.mean())}")
    print(f"compiled_next_token={int(torch.argmax(compiled_logits[:, -1, :].float(), dim=-1).item())}")


if __name__ == "__main__":
    main()
