#!/usr/bin/env python3
"""Faithful UniRec decoder-only throughput, parity, and profiler lab.

The lab excludes image processing and prefill.  It executes the real six-layer
decoder, static self-KV updates, static cross-attention, LM head, and token
selection.  Synthetic cache contents preserve production tensor shapes while
making the experiment independent of page frontend work.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from modeling_optimized_unirec import (
    LOCAL_UNIREC_STATIC_CACHE_LEN,
    OptimizedUniRecRunner,
    synchronize_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/workspace/models/unirec-0.1b"))
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", choices=("float16",), default="float16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--self-cache-length", type=int, default=2048)
    parser.add_argument("--cross-cache-length", type=int, default=1320)
    parser.add_argument("--cache-position", type=int, default=1023)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--measure-steps", type=int, default=100)
    parser.add_argument("--validation-steps", type=int, default=8)
    parser.add_argument("--profile-steps", type=int, default=2)
    parser.add_argument(
        "--profile-metric",
        choices=("pipe", "memory", "l2", "memory_access"),
        default="pipe",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".runtime_cache/12_unirec_0_1b_inference/text_decode_lab"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tmp/12_unirec_0_1b_inference/text_decode_lab/result.json"),
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if not 0 <= args.cache_position < args.self_cache_length:
        parser.error("--cache-position must be inside the self KV cache")
    return args


def progress(event: str, **fields: Any) -> None:
    print(
        "UNIREC_DECODE_LAB "
        + json.dumps({"event": event, **fields}, sort_keys=True),
        flush=True,
    )


def profiler_config(metric: str):
    import torch_npu.profiler as npu_prof

    metrics = {
        "pipe": npu_prof.AiCMetrics.PipeUtilization,
        "memory": npu_prof.AiCMetrics.Memory,
        "l2": npu_prof.AiCMetrics.L2Cache,
        "memory_access": npu_prof.AiCMetrics.MemoryAccess,
    }
    return npu_prof._ExperimentalConfig(
        profiler_level=npu_prof.ProfilerLevel.Level1,
        aic_metrics=metrics[metric],
        l2_cache=metric == "l2",
        export_type=npu_prof.ExportType.Text,
    )


def make_state(
    runner: OptimizedUniRecRunner,
    *,
    batch_size: int,
    self_cache_length: int,
    cross_cache_length: int,
    cache_position: int,
    seed: int,
) -> dict[str, Any]:
    config = runner.config
    heads = int(config.decoder_attention_heads)
    head_dim = int(config.d_model) // heads
    layers = int(config.decoder_layers)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    input_ids = torch.randint(
        0,
        int(config.vocab_size),
        (batch_size, 1),
        generator=generator,
        dtype=torch.int64,
    ).to(runner.device)
    self_keys = tuple(
        torch.zeros(
            (batch_size, heads, self_cache_length, head_dim),
            device=runner.device,
            dtype=runner.dtype,
        )
        for _ in range(layers)
    )
    self_values = tuple(torch.zeros_like(value) for value in self_keys)
    cross_keys = tuple(
        torch.zeros(
            (batch_size, heads, cross_cache_length, head_dim),
            device=runner.device,
            dtype=runner.dtype,
        )
        for _ in range(layers)
    )
    cross_values = tuple(torch.zeros_like(value) for value in cross_keys)
    cross_mask = torch.zeros(
        (batch_size, 1, 1, cross_cache_length),
        device=runner.device,
        dtype=torch.float32,
    )
    return {
        "next_token": input_ids,
        "cache_position": torch.full(
            (batch_size,), cache_position, device=runner.device, dtype=torch.int64
        ),
        "self_keys": self_keys,
        "self_values": self_values,
        "cross_keys": cross_keys,
        "cross_values": cross_values,
        "cross_mask": cross_mask,
    }


def step(fn: Any, state: dict[str, Any]) -> torch.Tensor:
    logits = fn(
        state["next_token"],
        state["cache_position"],
        0,
        state["self_keys"],
        state["self_values"],
        state["cross_keys"],
        state["cross_values"],
        state["cross_mask"],
    )
    state["next_token"] = torch.argmax(
        logits[:, -1, :].float(), dim=-1, keepdim=True
    ).long()
    state["cache_position"] = state["cache_position"] + 1
    return logits


def run_steps(fn: Any, state: dict[str, Any], count: int, *, collect: bool = False):
    tokens: list[torch.Tensor] = []
    logits = None
    for _ in range(count):
        logits = step(fn, state)
        if collect:
            tokens.append(state["next_token"].detach().cpu())
    return logits, None if not collect else torch.cat(tokens, dim=1).tolist()


def profile_eager_lane(
    *,
    runner: OptimizedUniRecRunner,
    backend: str,
    state: dict[str, Any],
    output_root: Path,
    steps: int,
    metric: str,
) -> dict[str, Any]:
    import torch_npu.profiler as npu_prof

    profile_dir = output_root / f"profile_eager_{backend}_{metric}"
    shutil.rmtree(profile_dir, ignore_errors=True)
    profile_dir.mkdir(parents=True, exist_ok=True)

    def eager_decode(*inputs):
        return runner.model.forward_cached_logits(
            decoder_input_ids=inputs[0],
            cache_position=inputs[1],
            active_length=0,
            key_cache=inputs[3],
            value_cache=inputs[4],
            cross_key_cache=inputs[5],
            cross_value_cache=inputs[6],
            cross_attention_mask=inputs[7],
            self_attention_backend=backend,
        )

    schedule = npu_prof.schedule(wait=0, warmup=0, active=1, repeat=1)
    synchronize_device(runner.device)
    started = time.perf_counter()
    with npu_prof.profile(
        activities=[npu_prof.ProfilerActivity.CPU, npu_prof.ProfilerActivity.NPU],
        schedule=schedule,
        experimental_config=profiler_config(metric),
        on_trace_ready=npu_prof.tensorboard_trace_handler(
            str(profile_dir), analyse_flag=True
        ),
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
    ) as profiler:
        with torch.profiler.record_function(f"unirec.decode.eager.{backend}"):
            run_steps(eager_decode, state, steps)
        synchronize_device(runner.device)
        profiler.step()
    synchronize_device(runner.device)
    return {
        "profile_dir": str(profile_dir),
        "profile_steps": int(steps),
        "profile_wall_s": time.perf_counter() - started,
        "metric": metric,
        "note": "Eager faithful decoder profile exposes operators; throughput uses compiled lanes.",
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    import torch_npu  # noqa: F401

    if args.self_cache_length != LOCAL_UNIREC_STATIC_CACHE_LEN:
        raise ValueError(
            "Set UNIREC_STATIC_CACHE_LEN to match --self-cache-length before "
            f"launch: env={LOCAL_UNIREC_STATIC_CACHE_LEN} arg={args.self_cache_length}"
        )
    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    progress("model_load_begin", model=str(args.model))
    load_started = time.perf_counter()
    runner = OptimizedUniRecRunner(
        model_path=args.model,
        device=args.device,
        dtype=args.dtype,
        compile_cache_dir=args.cache_dir,
    )
    progress("model_load_end", seconds=time.perf_counter() - load_started)

    lanes: dict[str, Any] = {}
    validations: dict[str, Any] = {}
    for backend in ("eager", "increfa", "increfa_all"):
        progress("lane_begin", backend=backend)
        compiled, compile_meta = runner._compile_decode_module(
            backend="torchair",
            self_attention_backend=backend,
            compile_dynamic=False,
            cross_cache_len=args.cross_cache_length,
            batch_size=args.batch_size,
        )
        state = make_state(
            runner,
            batch_size=args.batch_size,
            self_cache_length=args.self_cache_length,
            cross_cache_length=args.cross_cache_length,
            cache_position=args.cache_position,
            seed=7,
        )
        progress("first_call_begin", backend=backend)
        first_started = time.perf_counter()
        step(compiled, state)
        synchronize_device(runner.device)
        first_call_s = time.perf_counter() - first_started
        progress("first_call_end", backend=backend, seconds=first_call_s)
        run_steps(compiled, state, max(0, args.warmup_steps - 1))
        synchronize_device(runner.device)

        state = make_state(
            runner,
            batch_size=args.batch_size,
            self_cache_length=args.self_cache_length,
            cross_cache_length=args.cross_cache_length,
            cache_position=args.cache_position,
            seed=7,
        )
        synchronize_device(runner.device)
        measured_started = time.perf_counter()
        run_steps(compiled, state, args.measure_steps)
        synchronize_device(runner.device)
        measured_s = time.perf_counter() - measured_started

        state = make_state(
            runner,
            batch_size=args.batch_size,
            self_cache_length=args.self_cache_length,
            cross_cache_length=args.cross_cache_length,
            cache_position=args.cache_position,
            seed=11,
        )
        validation_logits, validation_tokens = run_steps(
            compiled, state, args.validation_steps, collect=True
        )
        synchronize_device(runner.device)
        validations[backend] = {
            "tokens": validation_tokens,
            "logits": validation_logits.detach().float().cpu(),
        }
        raw_tokens = args.batch_size * args.measure_steps
        lanes[backend] = {
            "compile": compile_meta,
            "first_call_s": first_call_s,
            "measure": {
                "steps": args.measure_steps,
                "decode_s": measured_s,
                "step_ms": measured_s * 1000.0 / args.measure_steps,
                "raw_tok_s": raw_tokens / measured_s,
                "batch_s": args.measure_steps / measured_s,
            },
        }
        progress("lane_end", backend=backend, raw_tok_s=raw_tokens / measured_s)

    left = validations["eager"]
    comparison = {}
    for backend in ("increfa", "increfa_all"):
        right = validations[backend]
        delta = (left["logits"] - right["logits"]).abs()
        comparison[f"eager_vs_{backend}"] = {
            "token_exact": left["tokens"] == right["tokens"],
            "eager_tokens": left["tokens"],
            f"{backend}_tokens": right["tokens"],
            "final_logits_max_abs": float(delta.max()),
            "final_logits_mean_abs": float(delta.mean()),
            "final_logits_cosine": float(
                F.cosine_similarity(
                    left["logits"].flatten(), right["logits"].flatten(), dim=0
                )
            ),
            "compiled_speedup": (
                lanes[backend]["measure"]["raw_tok_s"]
                / lanes["eager"]["measure"]["raw_tok_s"]
            ),
        }

    profiles: dict[str, Any] = {}
    if args.profile_steps > 0:
        for backend in ("eager", "increfa", "increfa_all"):
            progress("profile_begin", backend=backend)
            state = make_state(
                runner,
                batch_size=args.batch_size,
                self_cache_length=args.self_cache_length,
                cross_cache_length=args.cross_cache_length,
                cache_position=args.cache_position,
                seed=13,
            )
            profiles[backend] = profile_eager_lane(
                runner=runner,
                backend=backend,
                state=state,
                output_root=output.parent,
                steps=args.profile_steps,
                metric=args.profile_metric,
            )
            progress("profile_end", backend=backend, **profiles[backend])

    payload = {
        "schema_version": 1,
        "kind": "unirec_text_decode_lab",
        "scope": "warmed full six-layer decoder; prefill excluded",
        "shape": {
            "batch_size": args.batch_size,
            "self_cache_length": args.self_cache_length,
            "cross_cache_length": args.cross_cache_length,
            "initial_cache_position": args.cache_position,
        },
        "lanes": lanes,
        "comparison": comparison,
        "profiles": profiles,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
