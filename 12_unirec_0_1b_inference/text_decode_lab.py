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
        "--profile-compiled-steps",
        type=int,
        default=0,
        help=(
            "Profile the compiled decode graph at kernel level for each "
            "selected lane. Runs inside the lane loop on the warmed graph."
        ),
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("eager", "increfa", "increfa_all"),
        default=("eager", "increfa", "increfa_all"),
        help="Compiled decoder lanes to run; select one lane for shape sweeps.",
    )
    parser.add_argument(
        "--compiled-timing-steps",
        type=int,
        default=0,
        help=(
            "Measure the compiled graph with NPU events, separating queued "
            "device execution, host submission, final synchronization, and "
            "production-like sampled-token D2H wait."
        ),
    )
    parser.add_argument(
        "--profile-metric",
        choices=("pipe", "memory", "l2", "memory_access"),
        default="pipe",
    )
    parser.add_argument(
        "--qkv-projection",
        choices=("separate", "fused"),
        default="separate",
        help=(
            "separate runs the three per-projection matmuls; fused builds one "
            "concatenated QKV weight at load time and runs a single matmul."
        ),
    )
    parser.add_argument(
        "--mask-mode",
        choices=("per_step", "persistent"),
        default="per_step",
        help=(
            "per_step rebuilds the self-attention KV mask once per decode "
            "step inside the graph; persistent keeps a caller-owned mask "
            "and marks only the current position valid in place."
        ),
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
    mask_backend: str | None = None,
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
    state: dict[str, Any] = {
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
    if mask_backend is not None:
        # Persistent self-attention mask: positions before the initial cache
        # position are valid; the decode graph marks each newly written
        # position valid in place. Eager takes an additive float mask,
        # IncreFA backends take a boolean mask (True = masked).
        kv_positions = torch.arange(self_cache_length, device=runner.device).view(
            1, 1, 1, self_cache_length
        )
        invalid = (kv_positions >= cache_position).expand(
            batch_size, 1, 1, self_cache_length
        )
        if mask_backend == "eager":
            self_mask = torch.zeros(
                (batch_size, 1, 1, self_cache_length),
                device=runner.device,
                dtype=torch.float32,
            ).masked_fill(invalid, torch.finfo(torch.float32).min)
        else:
            self_mask = invalid.contiguous()
        state["self_mask"] = self_mask
    return state


def call_args(state: dict[str, Any]) -> tuple[Any, ...]:
    args: list[Any] = [
        state["next_token"],
        state["cache_position"],
        0,
        state["self_keys"],
        state["self_values"],
        state["cross_keys"],
        state["cross_values"],
        state["cross_mask"],
    ]
    if "self_mask" in state:
        args.append(state["self_mask"])
    return tuple(args)


def step(fn: Any, state: dict[str, Any]) -> torch.Tensor:
    logits = fn(*call_args(state))
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


def profile_compiled_timing(
    *,
    fn: Any,
    state: dict[str, Any],
    device: str,
    steps: int,
) -> dict[str, Any]:
    """Separate compiled-device work from host submission and token D2H wait."""
    import torch_npu

    synchronize_device(device)
    queued_start = torch_npu.npu.Event(enable_timing=True)
    queued_end = torch_npu.npu.Event(enable_timing=True)
    queued_start.record()
    host_started = time.perf_counter()
    run_steps(fn, state, steps)
    queued_end.record()
    host_enqueue_s = time.perf_counter() - host_started
    wait_started = time.perf_counter()
    queued_end.synchronize()
    final_sync_wait_s = time.perf_counter() - wait_started
    queued_device_s = float(queued_start.elapsed_time(queued_end)) / 1000.0
    queued_wall_s = host_enqueue_s + final_sync_wait_s

    synchronize_device(device)
    production_device_s = 0.0
    production_submit_s = 0.0
    production_d2h_wait_s = 0.0
    production_wall_started = time.perf_counter()
    for _ in range(steps):
        start_event = torch_npu.npu.Event(enable_timing=True)
        end_event = torch_npu.npu.Event(enable_timing=True)
        start_event.record()
        submit_started = time.perf_counter()
        logits = fn(*call_args(state))
        predicted = torch.argmax(logits[:, -1, :].float(), dim=-1, keepdim=True).long()
        state["next_token"] = predicted
        state["cache_position"] = state["cache_position"] + 1
        end_event.record()
        production_submit_s += time.perf_counter() - submit_started
        wait_started = time.perf_counter()
        predicted.detach().cpu()
        production_d2h_wait_s += time.perf_counter() - wait_started
        production_device_s += float(start_event.elapsed_time(end_event)) / 1000.0
    production_wall_s = time.perf_counter() - production_wall_started

    def per_step(seconds: float) -> float:
        return seconds * 1000.0 / steps

    return {
        "steps": int(steps),
        "queued": {
            "device_step_ms": per_step(queued_device_s),
            "host_enqueue_step_ms": per_step(host_enqueue_s),
            "final_sync_wait_step_ms": per_step(final_sync_wait_s),
            "wall_step_ms": per_step(queued_wall_s),
            "device_share_of_wall": queued_device_s / queued_wall_s,
        },
        "production_like_d2h": {
            "device_step_ms": per_step(production_device_s),
            "host_submit_step_ms": per_step(production_submit_s),
            "sampled_token_d2h_wait_step_ms": per_step(production_d2h_wait_s),
            "wall_step_ms": per_step(production_wall_s),
            "device_share_of_wall": production_device_s / production_wall_s,
        },
    }


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
        self_attention_mask = inputs[8] if len(inputs) > 8 else None
        if self_attention_mask is not None:
            # Persistent mask: mark the current position valid in place,
            # mirroring LocalUniRecPersistentMaskDecodeStepModule.
            index = inputs[1].view(-1, 1, 1, 1)
            if backend == "eager":
                self_attention_mask.scatter_(3, index, 0.0)
            else:
                self_attention_mask.scatter_(3, index, False)
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
            self_attention_mask=self_attention_mask,
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


def profile_compiled_lane(
    *,
    backend: str,
    fn: Any,
    state: dict[str, Any],
    device: str,
    output_root: Path,
    steps: int,
    metric: str,
) -> dict[str, Any]:
    import torch_npu.profiler as npu_prof

    profile_dir = output_root / f"profile_compiled_{backend}_{metric}"
    shutil.rmtree(profile_dir, ignore_errors=True)
    profile_dir.mkdir(parents=True, exist_ok=True)

    schedule = npu_prof.schedule(wait=0, warmup=0, active=1, repeat=1)
    synchronize_device(device)
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
        with torch.profiler.record_function(f"unirec.decode.compiled.{backend}"):
            run_steps(fn, state, steps)
        synchronize_device(device)
        profiler.step()
    synchronize_device(device)
    return {
        "profile_dir": str(profile_dir),
        "profile_steps": int(steps),
        "profile_wall_s": time.perf_counter() - started,
        "metric": metric,
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
    if args.qkv_projection == "fused":
        runner.fuse_decoder_self_qkv()
        progress("qkv_fused")

    def lane_state(backend: str, seed: int) -> dict[str, Any]:
        return make_state(
            runner,
            batch_size=args.batch_size,
            self_cache_length=args.self_cache_length,
            cross_cache_length=args.cross_cache_length,
            cache_position=args.cache_position,
            seed=seed,
            mask_backend=backend if args.mask_mode == "persistent" else None,
        )

    lanes: dict[str, Any] = {}
    validations: dict[str, Any] = {}
    for backend in args.backends:
        progress("lane_begin", backend=backend)
        compiled, compile_meta = runner._compile_decode_module(
            backend="torchair",
            self_attention_backend=backend,
            compile_dynamic=False,
            cross_cache_len=args.cross_cache_length,
            batch_size=args.batch_size,
            mask_mode=args.mask_mode,
        )
        state = lane_state(backend, 7)
        progress("first_call_begin", backend=backend)
        first_started = time.perf_counter()
        step(compiled, state)
        synchronize_device(runner.device)
        first_call_s = time.perf_counter() - first_started
        progress("first_call_end", backend=backend, seconds=first_call_s)
        run_steps(compiled, state, max(0, args.warmup_steps - 1))
        synchronize_device(runner.device)

        state = lane_state(backend, 7)
        synchronize_device(runner.device)
        measured_started = time.perf_counter()
        run_steps(compiled, state, args.measure_steps)
        synchronize_device(runner.device)
        measured_s = time.perf_counter() - measured_started

        state = lane_state(backend, 11)
        validation_logits, validation_tokens = run_steps(
            compiled, state, args.validation_steps, collect=True
        )
        synchronize_device(runner.device)
        validations[backend] = {
            "tokens": validation_tokens,
            "logits": validation_logits.detach().float().cpu(),
        }
        compiled_timing = None
        if args.compiled_timing_steps > 0:
            progress("compiled_timing_begin", backend=backend)
            timing_state = lane_state(backend, 17)
            compiled_timing = profile_compiled_timing(
                fn=compiled,
                state=timing_state,
                device=runner.device,
                steps=args.compiled_timing_steps,
            )
            progress(
                "compiled_timing_end",
                backend=backend,
                **compiled_timing["production_like_d2h"],
            )
        compiled_profile = None
        if args.profile_compiled_steps > 0:
            progress("compiled_profile_begin", backend=backend)
            profile_state = lane_state(backend, 19)
            compiled_profile = profile_compiled_lane(
                backend=backend,
                fn=compiled,
                state=profile_state,
                device=runner.device,
                output_root=output.parent,
                steps=args.profile_compiled_steps,
                metric=args.profile_metric,
            )
            progress("compiled_profile_end", backend=backend, **compiled_profile)
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
            "compiled_timing": compiled_timing,
            "compiled_profile": compiled_profile,
        }
        progress("lane_end", backend=backend, raw_tok_s=raw_tokens / measured_s)

    comparison = {}
    if "eager" in validations:
        left = validations["eager"]
        for backend in ("increfa", "increfa_all"):
            if backend not in validations:
                continue
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
        for backend in args.backends:
            progress("profile_begin", backend=backend)
            state = lane_state(backend, 13)
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
