#!/usr/bin/env python3
"""Compare native six-head and zero-padded eight-head UniRec B1 decode."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

import modeling_optimized_unirec as unirec
from modeling_optimized_unirec import OptimizedUniRecRunner, synchronize_device
from text_decode_lab import (
    profile_compiled_lane,
    profile_compiled_timing,
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
    parser.add_argument("--physical-heads", type=int, choices=(6, 8), required=True)
    parser.add_argument("--weights-nz", action="store_true")
    parser.add_argument("--source-length", type=int, default=56)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--measure-steps", type=int, default=100)
    parser.add_argument("--timing-steps", type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.source_length <= CROSS_CACHE_LENGTH:
        parser.error("--source-length must be within C256")
    phase_calls = {
        "warmup": args.warmup_steps,
        "measure": args.measure_steps,
        "timing": 2 * args.timing_steps,
    }
    for phase, calls in phase_calls.items():
        if calls < 1:
            parser.error(f"--{phase}-steps must be positive")
        if INITIAL_CACHE_POSITION + calls > SELF_CACHE_LENGTH:
            parser.error(
                f"{phase} would advance cache position beyond S256: "
                f"start={INITIAL_CACHE_POSITION} calls={calls}"
            )
    return args


def progress(event: str, **fields: Any) -> None:
    print(
        "UNIREC_DECODE_HEAD_PADDING_B1_PROGRESS "
        + json.dumps({"event": event, **fields}, sort_keys=True),
        flush=True,
    )


def pad_linear_output(linear: nn.Linear, target_output: int) -> None:
    current_output = int(linear.weight.shape[0])
    if target_output <= current_output:
        raise ValueError("target linear output must be larger")
    rows = target_output - current_output
    weight = torch.nn.functional.pad(linear.weight.detach(), (0, 0, 0, rows))
    linear.weight = nn.Parameter(weight, requires_grad=False)
    if linear.bias is not None:
        bias = torch.nn.functional.pad(linear.bias.detach(), (0, rows))
        linear.bias = nn.Parameter(bias, requires_grad=False)
    linear.out_features = target_output


def pad_linear_input(linear: nn.Linear, target_input: int) -> None:
    current_input = int(linear.weight.shape[1])
    if target_input <= current_input:
        raise ValueError("target linear input must be larger")
    columns = target_input - current_input
    weight = torch.nn.functional.pad(linear.weight.detach(), (0, columns, 0, 0))
    linear.weight = nn.Parameter(weight, requires_grad=False)
    linear.in_features = target_input


def padded_attend_increfa(
    self: unirec.LocalDecoderAttention,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    batch_size, _, target_length, _ = query_states.shape
    if target_length != 1:
        raise ValueError(
            f"Local UniRec IncreFA path expects q_len == 1, got {target_length}"
        )
    if attention_mask.shape != (batch_size, 1, 1, key_states.shape[2]):
        raise ValueError("padded-head attention mask shape mismatch")
    attention_output = unirec.torch_npu.npu_incre_flash_attention(
        query_states.contiguous(),
        key_states.contiguous(),
        value_states.contiguous(),
        atten_mask=attention_mask.to(dtype=torch.bool).contiguous(),
        num_heads=int(self.num_heads),
        num_key_value_heads=int(self.num_heads),
        input_layout="BNSD",
        scale_value=float(self.scaling),
    )
    attention_output = attention_output.to(dtype=output_dtype)
    attention_width = int(self.num_heads) * int(self.head_dim)
    attention_output = (
        attention_output.transpose(1, 2)
        .contiguous()
        .view(batch_size, target_length, attention_width)
    )
    return self.apply_linear_3d(self.out_proj, attention_output)


def pad_runner_attention_heads(runner: OptimizedUniRecRunner) -> int:
    if runner._compiled_decode_modules:
        raise RuntimeError("pad attention heads before compiling decode")
    semantic_heads = int(runner.config.decoder_attention_heads)
    if semantic_heads != 6:
        raise ValueError(f"expected six semantic heads, got {semantic_heads}")
    head_dim = int(runner.config.d_model) // semantic_heads
    target_heads = 8
    target_width = target_heads * head_dim
    padded_modules = 0
    for layer in runner.model.decoder.layers:
        for attention in (layer.self_attn, layer.encoder_attn):
            if attention.qkv_weight is not None:
                raise RuntimeError("head-padding lab requires unfused QKV")
            for linear in (attention.q_proj, attention.k_proj, attention.v_proj):
                pad_linear_output(linear, target_width)
            pad_linear_input(attention.out_proj, target_width)
            attention.num_heads = target_heads
            # Preserve the trained 128-wide head and its original scaling.
            attention.head_dim = head_dim
            attention.scaling = head_dim**-0.5
            padded_modules += 1
    unirec.LocalDecoderAttention.attend_increfa = padded_attend_increfa
    synchronize_device(runner.device)
    return padded_modules


def make_state(
    runner: OptimizedUniRecRunner,
    *,
    physical_heads: int,
    seed: int,
    source_length: int,
) -> dict[str, Any]:
    config = runner.config
    semantic_heads = int(config.decoder_attention_heads)
    head_dim = int(config.d_model) // semantic_heads
    layers = int(config.decoder_layers)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    next_token = torch.randint(
        0,
        int(config.vocab_size),
        (1, 1),
        generator=generator,
        dtype=torch.int64,
    ).to(runner.device)
    self_keys = tuple(
        torch.zeros(
            (1, physical_heads, SELF_CACHE_LENGTH, head_dim),
            device=runner.device,
            dtype=runner.dtype,
        )
        for _ in range(layers)
    )
    self_values = tuple(torch.zeros_like(value) for value in self_keys)
    packed_cross_kv = torch.zeros(
        (2 * layers, 1, physical_heads, CROSS_CACHE_LENGTH, head_dim),
        device=runner.device,
        dtype=runner.dtype,
    )
    cross_mask = torch.full(
        (1, 1, 1, CROSS_CACHE_LENGTH),
        torch.finfo(torch.float32).min,
        device=runner.device,
        dtype=torch.float32,
    )
    cross_mask[..., :source_length].zero_()
    return {
        "next_token": next_token,
        "cache_position": torch.full(
            (1,), INITIAL_CACHE_POSITION, device=runner.device, dtype=torch.int64
        ),
        "self_keys": self_keys,
        "self_values": self_values,
        "cross_keys": tuple(packed_cross_kv[layer] for layer in range(layers)),
        "cross_values": tuple(
            packed_cross_kv[layers + layer] for layer in range(layers)
        ),
        "cross_mask": cross_mask,
    }


def eager_logits(
    runner: OptimizedUniRecRunner, state: dict[str, Any]
) -> torch.Tensor:
    logits = runner.model.forward_cached_logits(
        decoder_input_ids=state["next_token"],
        cache_position=state["cache_position"],
        active_length=0,
        key_cache=state["self_keys"],
        value_cache=state["self_values"],
        cross_key_cache=state["cross_keys"],
        cross_value_cache=state["cross_values"],
        cross_attention_mask=state["cross_mask"],
        self_attention_backend="increfa_all",
    )
    synchronize_device(runner.device)
    return logits.detach().float().cpu()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    progress(
        "lane_begin",
        physical_heads=args.physical_heads,
        weights_nz=args.weights_nz,
    )
    runner = OptimizedUniRecRunner(
        model_path=args.model,
        device=args.device,
        dtype="float16",
        compile_cache_dir=args.cache_dir,
    )
    padding_started = time.perf_counter()
    padded_modules = (
        pad_runner_attention_heads(runner) if args.physical_heads == 8 else 0
    )
    padding_s = time.perf_counter() - padding_started
    nz_started = time.perf_counter()
    nz_tensor_count = runner.cast_decoder_weights_nz() if args.weights_nz else 0
    nz_format_s = time.perf_counter() - nz_started if args.weights_nz else 0.0

    eager_state = make_state(
        runner,
        physical_heads=args.physical_heads,
        seed=11,
        source_length=args.source_length,
    )
    reference_logits = eager_logits(runner, eager_state)

    module, compile_meta = runner._compile_decode_module(
        backend="torchair",
        self_attention_backend="increfa_all",
        compile_dynamic=False,
        cross_cache_len=CROSS_CACHE_LENGTH,
        batch_size=1,
        self_cache_len=SELF_CACHE_LENGTH,
    )

    def new_state(seed: int) -> dict[str, Any]:
        return make_state(
            runner,
            physical_heads=args.physical_heads,
            seed=seed,
            source_length=args.source_length,
        )

    warm_state = new_state(7)
    progress("first_call_begin", physical_heads=args.physical_heads)
    first_started = time.perf_counter()
    step(module, warm_state)
    synchronize_device(args.device)
    first_call_s = time.perf_counter() - first_started
    progress(
        "first_call_end",
        physical_heads=args.physical_heads,
        first_call_s=first_call_s,
    )
    run_steps(module, warm_state, args.warmup_steps - 1)
    synchronize_device(args.device)

    measure_state = new_state(7)
    synchronize_device(args.device)
    measured_started = time.perf_counter()
    run_steps(module, measure_state, args.measure_steps)
    synchronize_device(args.device)
    measured_s = time.perf_counter() - measured_started

    timing_state = new_state(17)
    compiled_timing = profile_compiled_timing(
        fn=module,
        state=timing_state,
        device=args.device,
        steps=args.timing_steps,
        stepper=step,
    )

    validation_state = new_state(11)
    compiled_logits, validation_tokens = run_steps(
        module, validation_state, 1, collect=True
    )
    synchronize_device(args.device)
    compiled_logits_cpu = compiled_logits.detach().float().cpu()
    local_delta = (reference_logits - compiled_logits_cpu).abs()

    lane_root = args.output.parent / f"heads{args.physical_heads}"
    compiled_profile = profile_compiled_lane(
        backend="increfa_all",
        fn=module,
        state=new_state(19),
        device=args.device,
        output_root=lane_root,
        steps=1,
        metric="pipe",
        stepper=step,
    )

    logits_path = args.output.with_suffix(".validation_logits.npy")
    eager_path = args.output.with_suffix(".eager_logits.npy")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(logits_path, compiled_logits_cpu.numpy())
    np.save(eager_path, reference_logits.numpy())
    payload = {
        "schema_version": 1,
        "kind": "unirec_decode_head_padding_b1_lab",
        "status": "ok",
        "shape": {
            "batch_size": 1,
            "semantic_heads": 6,
            "physical_heads": args.physical_heads,
            "head_dim": 128,
            "attention_width": args.physical_heads * 128,
            "self_cache_length": SELF_CACHE_LENGTH,
            "cross_cache_length": CROSS_CACHE_LENGTH,
            "cache_position": INITIAL_CACHE_POSITION,
            "source_length": args.source_length,
        },
        "padded_attention_modules": padded_modules,
        "padding_s": padding_s,
        "weights_nz": bool(runner.weights_nz),
        "nz_tensor_count": nz_tensor_count,
        "nz_format_s": nz_format_s,
        "first_call_s": first_call_s,
        "measure_steps": args.measure_steps,
        "measure_s": measured_s,
        "step_ms": measured_s * 1000.0 / args.measure_steps,
        "raw_tok_s": args.measure_steps / measured_s,
        "compiled_timing": compiled_timing,
        "compiled_profile": compiled_profile,
        "compile": compile_meta,
        "compiled_vs_same_lane_eager": {
            "max_abs": float(local_delta.max().item()),
            "mean_abs": float(local_delta.mean().item()),
            "argmax_exact": bool(
                compiled_logits_cpu.argmax().item() == reference_logits.argmax().item()
            ),
        },
        "validation_tokens": validation_tokens,
        "validation_logits_npy": str(logits_path),
        "eager_logits_npy": str(eager_path),
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    progress(
        "lane_end",
        physical_heads=args.physical_heads,
        step_ms=payload["step_ms"],
        raw_tok_s=payload["raw_tok_s"],
    )
    print("UNIREC_DECODE_HEAD_PADDING_B1: PASS")
    print("UNIREC_DECODE_HEAD_PADDING_B1_RESULT " + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
