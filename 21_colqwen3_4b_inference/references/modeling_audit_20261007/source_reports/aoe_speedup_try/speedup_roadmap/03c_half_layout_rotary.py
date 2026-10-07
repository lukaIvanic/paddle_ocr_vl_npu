from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from speedup_roadmap.common.roadmap import (  # noqa: E402
    add_common_args,
    benchmark_decode,
    compile_decode_variant,
    decode_preview,
    ensure_basic_shape,
    load_live_runner,
    precompute_decode_masks,
    prefill_once,
    profile_decode,
    prompt_penalty_ids_for,
    require_npu,
    reset_dir,
    synthetic_method_decode_loop,
    tensor_ids,
    to_half_layout_key_cache_state,
    trim_after_first_eos,
    warm_compiled_decode,
    write_manifest,
)


ARTIFACT_DIR = REPO_ROOT / "artifacts" / "speedup_roadmap" / "03c_half_layout_rotary"
VARIANTS = [
    {
        "name": "current_interleave_manual",
        "rotary_impl": "manual_hoisted_noslice",
        "qkv_impl": "fused",
        "norm_impl": "npu",
        "qk_layout": "interleaved",
    },
    {
        "name": "half_layout_manual",
        "rotary_impl": "manual_hoisted_half_layout",
        "qkv_impl": "fused",
        "norm_impl": "npu",
        "qk_layout": "half",
    },
    {
        "name": "half_layout_npu_half",
        "rotary_impl": "npu_rotary_mul_half_layout",
        "qkv_impl": "fused",
        "norm_impl": "npu",
        "qk_layout": "half",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 3c: half-layout Q/K plus npu_rotary_mul half mode.")
    add_common_args(parser)
    parser.set_defaults(repeats=5)
    parser.add_argument("--eos-mode", choices=["none", "sync", "overlap_event_flags"], default="none")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_basic_shape(args)
    device = require_npu(args.device)
    dtype = torch.float16
    runner, inputs, prompt_tokens, repetition_penalty, clean_output_text = load_live_runner(args, device=device, dtype=dtype)
    compiled = {
        variant["name"]: compile_decode_variant(args, device=device, dtype=dtype, **{k: variant[k] for k in ("rotary_impl", "qkv_impl", "norm_impl")})
        for variant in VARIANTS
    }

    def base_prefill(decode_steps: int = args.decode_steps) -> Any:
        return prefill_once(runner, inputs, static_kv_cache_len=args.cache_len, decode_steps=decode_steps, repetition_penalty=repetition_penalty)

    def make_state(variant: dict[str, str], decode_steps: int = args.decode_steps) -> Any:
        state = base_prefill(decode_steps)
        if variant["qk_layout"] == "half":
            state = to_half_layout_key_cache_state(state)
        return state

    for variant in VARIANTS:
        warm_compiled_decode(compiled[variant["name"]], make_state(variant, 1), args.cache_len)
    masks = precompute_decode_masks(args, prompt_tokens=prompt_tokens, device=device)
    prompt_penalty_ids = prompt_penalty_ids_for(args, base_prefill(1), repetition_penalty=repetition_penalty, device=device)
    eos_ids = tuple(int(token_id) for token_id in runner.config.eos_token_ids)

    def decode_with(name: str, state: Any, decode_steps: int = args.decode_steps, masks_for_run: tuple[torch.Tensor, ...] | None = masks) -> torch.Tensor:
        return synthetic_method_decode_loop(
            runner,
            compiled[name],
            state,
            decode_steps=decode_steps,
            cache_len=args.cache_len,
            repetition_penalty=repetition_penalty,
            eos_mode=args.eos_mode,
            precomputed_decode_masks=masks_for_run,
            prompt_penalty_ids=prompt_penalty_ids,
        )

    validation = {}
    for variant in VARIANTS:
        name = variant["name"]
        output = decode_with(name, make_state(variant))
        trimmed = trim_after_first_eos(output, eos_ids)
        validation[name] = {
            "ids": tensor_ids(output),
            "trimmed_ids": tensor_ids(trimmed),
            "text_preview": decode_preview(runner.processor, clean_output_text, output),
        }
    first_ids = next(iter(validation.values()))["ids"]
    first_trimmed_ids = next(iter(validation.values()))["trimmed_ids"]
    results = [
        benchmark_decode(
            variant["name"],
            lambda variant=variant: make_state(variant),
            lambda state, name=variant["name"]: decode_with(name, state),
            warmup=args.warmup,
            repeats=args.repeats,
            decode_steps=args.decode_steps,
            batch_size=args.batch_size,
            eos_ids=eos_ids,
            measure_npu_events=True,
        )
        for variant in VARIANTS
    ]
    profiles = {}
    if args.profile:
        profile_root = ARTIFACT_DIR / "profiles"
        reset_dir(profile_root)
        profile_masks = precompute_decode_masks(args, prompt_tokens=prompt_tokens, device=device, decode_steps=args.profile_steps)
        for variant in VARIANTS:
            name = variant["name"]
            profiles[name] = profile_decode(
                profile_root=profile_root,
                name=name,
                metric=args.profile_metric,
                make_state=lambda steps, variant=variant: make_state(variant, steps),
                decode_fn=lambda state, steps, name=name: decode_with(name, state, steps, profile_masks),
                decode_steps=args.profile_steps,
                outer_warmup=args.profile_warmup,
                profiler_warmup=args.profiler_warmup,
                topn=args.topn,
            )
    manifest = {
        "step": "03c_half_layout_rotary",
        "shape": {
            "prompt_tokens": prompt_tokens,
            "cache_len": args.cache_len,
            "decode_steps": args.decode_steps,
            "mask_mode": args.mask_mode,
            "penalty_history_mode": args.penalty_history_mode,
            "prompt_penalty_ids": None if prompt_penalty_ids is None else int(prompt_penalty_ids.numel()),
        },
        "variants": VARIANTS,
        "validation": {
            "all_exact_match": all(item["ids"] == first_ids for item in validation.values()),
            "all_eos_trimmed_exact_match": all(item["trimmed_ids"] == first_trimmed_ids for item in validation.values()),
            "outputs": validation,
        },
        "results": results,
        "profiles": profiles,
    }
    write_manifest(ARTIFACT_DIR / "manifest.json", manifest)


if __name__ == "__main__":
    main()
