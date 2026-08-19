#!/usr/bin/env python3
"""Profile the optimized UniRec lane-A decode graph at one batch size."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

import modeling_optimized_unirec as unirec
from decode_lm_head_padding_b1_lab import (
    ALIGNED_VOCAB_SIZE,
    SEMANTIC_VOCAB_SIZE,
    SemanticVocabDecodeStepModule,
    pad_lm_head,
)
from modeling_optimized_unirec import OptimizedUniRecRunner, synchronize_device
from text_decode_lab import (
    make_state,
    profile_compiled_lane,
    profile_compiled_timing,
    reset_state_,
    run_steps,
    step,
)


SELF_CACHE_LENGTH = 256
CROSS_CACHE_LENGTH = 256
INITIAL_CACHE_POSITION = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, choices=(1, 16, 64, 128), required=True)
    parser.add_argument("--source-length", type=int, default=56)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--measure-steps", type=int, default=100)
    parser.add_argument("--timing-steps", type=int, default=100)
    parser.add_argument("--validation-steps", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.source_length <= CROSS_CACHE_LENGTH:
        parser.error("--source-length must be within C256")
    phase_calls = {
        "warmup": args.warmup_steps,
        "measure": args.measure_steps,
        "timing": 2 * args.timing_steps,
        "validation": args.validation_steps,
        "profile": 1,
    }
    for phase, calls in phase_calls.items():
        if calls < 1:
            parser.error(f"{phase} steps must be positive")
        if INITIAL_CACHE_POSITION + calls > SELF_CACHE_LENGTH:
            parser.error(
                f"{phase} would advance beyond S256: "
                f"start={INITIAL_CACHE_POSITION} calls={calls}"
            )
    return args


def progress(event: str, **fields: Any) -> None:
    print(
        "UNIREC_DECODE_LANE_A_SCALING_PROGRESS "
        + json.dumps({"event": event, **fields}, sort_keys=True),
        flush=True,
    )


def configure_cross_mask_(state: dict[str, Any], source_length: int) -> None:
    mask = state["cross_mask"]
    mask.fill_(torch.finfo(mask.dtype).min)
    mask[..., :source_length].zero_()
    if not bool((~mask.bool()).reshape(mask.shape[0], -1).any(dim=1).all().item()):
        raise RuntimeError("lane-A scaling lab constructed a fully masked row")


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.reset_peak_memory_stats(args.device)
    progress("lane_begin", batch_size=args.batch_size)

    runner = OptimizedUniRecRunner(
        model_path=args.model,
        device=args.device,
        dtype="float16",
        compile_cache_dir=args.cache_dir,
    )
    if int(runner.config.vocab_size) != SEMANTIC_VOCAB_SIZE:
        raise ValueError(
            f"expected config vocab {SEMANTIC_VOCAB_SIZE}, "
            f"got {runner.config.vocab_size}"
        )
    padded_rows = pad_lm_head(runner)
    unirec.LocalUniRecCachedDecodeStepModule = SemanticVocabDecodeStepModule
    nz_started = time.perf_counter()
    nz_tensor_count = runner.cast_decoder_weights_nz()
    nz_format_s = time.perf_counter() - nz_started

    module, compile_meta = runner._compile_decode_module(
        backend="torchair",
        self_attention_backend="increfa_all",
        compile_dynamic=False,
        cross_cache_len=CROSS_CACHE_LENGTH,
        batch_size=args.batch_size,
        self_cache_len=SELF_CACHE_LENGTH,
    )

    state = make_state(
        runner,
        batch_size=args.batch_size,
        self_cache_length=SELF_CACHE_LENGTH,
        cross_cache_length=CROSS_CACHE_LENGTH,
        cache_position=INITIAL_CACHE_POSITION,
        seed=7,
    )
    configure_cross_mask_(state, args.source_length)

    def reset(seed: int) -> dict[str, Any]:
        reset_state_(
            state,
            runner,
            seed=seed,
            cache_position=INITIAL_CACHE_POSITION,
        )
        configure_cross_mask_(state, args.source_length)
        return state

    progress("first_call_begin", batch_size=args.batch_size)
    first_started = time.perf_counter()
    step(module, state)
    synchronize_device(args.device)
    first_call_s = time.perf_counter() - first_started
    progress(
        "first_call_end",
        batch_size=args.batch_size,
        first_call_s=first_call_s,
    )
    run_steps(module, state, args.warmup_steps - 1)
    synchronize_device(args.device)

    reset(7)
    synchronize_device(args.device)
    measured_started = time.perf_counter()
    run_steps(module, state, args.measure_steps)
    synchronize_device(args.device)
    measured_s = time.perf_counter() - measured_started

    reset(17)
    compiled_timing = profile_compiled_timing(
        fn=module,
        state=state,
        device=args.device,
        steps=args.timing_steps,
        stepper=step,
    )

    reset(11)
    validation_logits, validation_tokens = run_steps(
        module,
        state,
        args.validation_steps,
        collect=True,
    )
    synchronize_device(args.device)
    validation_logits_cpu = validation_logits.detach().float().cpu()
    flat_tokens = [token for row in validation_tokens for token in row]
    if not flat_tokens:
        raise RuntimeError("validation returned no tokens")
    if min(flat_tokens) < 0 or max(flat_tokens) >= SEMANTIC_VOCAB_SIZE:
        raise RuntimeError("validation selected a padded vocabulary row")

    reset(19)
    compiled_profile = profile_compiled_lane(
        backend="increfa_all",
        fn=module,
        state=state,
        device=args.device,
        output_root=args.output.parent,
        steps=1,
        metric="pipe",
        stepper=step,
    )

    raw_tokens = args.batch_size * args.measure_steps
    payload = {
        "schema_version": 1,
        "kind": "unirec_decode_lane_a_batch_scaling_lab",
        "status": "ok",
        "shape": {
            "batch_size": args.batch_size,
            "semantic_heads": 6,
            "qkv_fused": False,
            "self_cache_length": SELF_CACHE_LENGTH,
            "cross_cache_length": CROSS_CACHE_LENGTH,
            "cache_position": INITIAL_CACHE_POSITION,
            "source_length": args.source_length,
            "semantic_vocab_size": SEMANTIC_VOCAB_SIZE,
            "lm_head_rows": ALIGNED_VOCAB_SIZE,
        },
        "padded_rows": padded_rows,
        "logits_rows_returned": int(validation_logits_cpu.shape[-1]),
        "weights_nz": bool(runner.weights_nz),
        "nz_tensor_count": nz_tensor_count,
        "nz_format_s": nz_format_s,
        "first_call_s": first_call_s,
        "measure_steps": args.measure_steps,
        "measure_s": measured_s,
        "step_ms": measured_s * 1000.0 / args.measure_steps,
        "raw_tok_s": raw_tokens / measured_s,
        "batch_steps_s": args.measure_steps / measured_s,
        "compiled_timing": compiled_timing,
        "compiled_profile": compiled_profile,
        "compile": compile_meta,
        "validation": {
            "steps": args.validation_steps,
            "rows": len(validation_tokens),
            "token_count": len(flat_tokens),
            "min_token": min(flat_tokens),
            "max_token": max(flat_tokens),
            "finite_logits": bool(torch.isfinite(validation_logits_cpu).all().item()),
        },
        "npu_memory": {
            "allocated_bytes": int(torch.npu.memory_allocated(args.device)),
            "reserved_bytes": int(torch.npu.memory_reserved(args.device)),
            "max_allocated_bytes": int(torch.npu.max_memory_allocated(args.device)),
            "max_reserved_bytes": int(torch.npu.max_memory_reserved(args.device)),
        },
    }
    if payload["logits_rows_returned"] != SEMANTIC_VOCAB_SIZE:
        raise RuntimeError("aligned graph returned padded vocabulary rows")
    if nz_tensor_count != 49:
        raise RuntimeError(f"expected 49 NZ tensors, got {nz_tensor_count}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    progress(
        "lane_end",
        batch_size=args.batch_size,
        step_ms=payload["step_ms"],
        raw_tok_s=payload["raw_tok_s"],
    )
    print("UNIREC_DECODE_LANE_A_SCALING: PASS")
    print("UNIREC_DECODE_LANE_A_SCALING_RESULT " + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
