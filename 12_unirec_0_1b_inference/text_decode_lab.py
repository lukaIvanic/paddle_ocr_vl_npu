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
        "--graph-mode",
        choices=("ge", "acl", "npugraph", "ge_capture", "npugraph_ex", "npugraph_ex_full"),
        default="ge",
        help=(
            "ge executes the compiled decode step through the GE graph "
            "engine (cache_compile); acl uses TorchAir reduce-overhead "
            "ACLGraph capture/replay for dispatch; npugraph bypasses "
            "torch.compile entirely and captures the eager decode step "
            "(forward + argmax + state advance) into a torch.npu.NPUGraph "
            "replayed per step over static buffers; ge_capture attempts the "
            "hybrid - capturing calls to the GE-compiled step into an "
            "NPUGraph, combining GE's fused device graph with replay "
            "dispatch (may fail if GE submission is not stream-capturable); "
            "npugraph_ex uses TorchAir's supported FX+ACLGraph backend, and "
            "npugraph_ex_full adds static-kernel compile, SuperKernel, and "
            "frozen_parameter."
        ),
    )
    parser.add_argument(
        "--ge-tuning",
        nargs="*",
        default=[],
        choices=("frozen_parameter", "ref_data", "single_stream", "tiling_schedule"),
        help=(
            "GE-mode tuning knobs from the TorchAir advanced docs: "
            "frozen_parameter fixes weight input addresses (host dispatch), "
            "ref_data avoids copies for in-place KV scatter, single_stream "
            "removes inter-stream switching, tiling_schedule sinks tiling to "
            "device (fused attention ops only, so increfa lanes)."
        ),
    )
    parser.add_argument(
        "--static-kernel",
        action="store_true",
        help=(
            "For --graph-mode npugraph: before capture, run one eager step "
            "under StaticKernelCompiler so aclnn selects static-shape kernel "
            "binaries compiled for these exact shapes. Installs a kernel "
            "package into the shared CANN opp tree (auto-uninstalled at "
            "clean exit; uninstall path recorded in the result)."
        ),
    )
    parser.add_argument(
        "--weight-format",
        choices=("nd", "nz"),
        default="nd",
        help=(
            "nz casts decode-path weights (fused QKV, out/cross-Q "
            "projections, MLP, LM head) to FRACTAL_NZ before compile or "
            "capture - the internal format GE uses, measured ~85%% faster "
            "for the same eager matmuls at M=1."
        ),
    )
    parser.add_argument(
        "--prefetch-mode",
        choices=("none", "weights", "weights_kv", "staged"),
        default="none",
        help=(
            "npu_prefetch strategy: weights/weights_kv issue one bulk "
            "prefetch at step start; staged warms each layer's MLP weights "
            "during its attention and the next layer's attention weights "
            "during its MLP, leaving the LM head as an unprefetched control. "
            "per_step mask mode only."
        ),
    )
    parser.add_argument(
        "--lm-head-rows",
        type=int,
        default=0,
        help=(
            "SYNTHETIC timing probe: slice the LM head to the first N vocab "
            "rows before compile. Output tokens are not valid model output. "
            "0 keeps the full head. Use a dedicated --cache-dir per value."
        ),
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


def step_static(fn: Any, state: dict[str, Any]) -> torch.Tensor:
    """Step with in-place state advance so input addresses never change.

    npugraph_ex keys captured ACLGraphs by input tensor address; the default
    step() allocates fresh next_token/cache_position tensors every call, which
    makes every step look like a new graph and forces eager fallback.
    """
    logits = fn(*call_args(state))
    predicted = torch.argmax(logits[:, -1, :].float(), dim=-1, keepdim=True)
    state["next_token"].copy_(predicted)
    state["cache_position"].add_(1)
    return logits


def run_steps(
    fn: Any,
    state: dict[str, Any],
    count: int,
    *,
    collect: bool = False,
    stepper: Any = step,
):
    tokens: list[torch.Tensor] = []
    logits = None
    for _ in range(count):
        logits = stepper(fn, state)
        if collect:
            tokens.append(state["next_token"].detach().cpu())
    return logits, None if not collect else torch.cat(tokens, dim=1).tolist()


def reset_state_(
    state: dict[str, Any],
    runner: OptimizedUniRecRunner,
    *,
    seed: int,
    cache_position: int,
) -> None:
    """Restore a state to make_state(seed) values without reallocating.

    NPUGraph replay is bound to the captured buffer addresses, so measurement
    phases must reuse one state and reset its contents in place.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    input_ids = torch.randint(
        0,
        int(runner.config.vocab_size),
        tuple(state["next_token"].shape),
        generator=generator,
        dtype=torch.int64,
    )
    state["next_token"].copy_(input_ids.to(runner.device))
    state["cache_position"].fill_(cache_position)
    for key in ("self_keys", "self_values", "cross_keys", "cross_values"):
        for tensor in state[key]:
            tensor.zero_()
    state["cross_mask"].zero_()
    if "self_mask" in state:
        length = int(state["self_mask"].shape[-1])
        kv_positions = torch.arange(length, device=runner.device).view(1, 1, 1, length)
        invalid = (kv_positions >= cache_position).expand_as(state["self_mask"])
        if state["self_mask"].dtype == torch.bool:
            state["self_mask"].copy_(invalid)
        else:
            state["self_mask"].zero_()
            state["self_mask"].masked_fill_(invalid, torch.finfo(torch.float32).min)


def capture_npugraph(
    module: Any,
    state: dict[str, Any],
    device: str,
    *,
    warmup_iters: int = 3,
    capture_stream: Any = None,
) -> tuple[Any, torch.Tensor, float]:
    """Capture one full decode step into a torch.npu.NPUGraph.

    The captured region is forward + argmax + in-place next_token/cache_position
    advance, so a replay consumes zero host-side tensor work. Returns the graph,
    the static logits buffer, and capture wall seconds (including warmup).

    capture_stream pins both warmup and capture to one caller-owned stream.
    The GE executor binds to the stream of its first call and refuses to run
    on any other, so capturing a GE-compiled callable requires warming and
    capturing on the same stream.
    """
    import torch_npu

    static_args = call_args(state)

    def one_step() -> torch.Tensor:
        logits = module(*static_args)
        predicted = torch.argmax(logits[:, -1, :].float(), dim=-1, keepdim=True)
        state["next_token"].copy_(predicted)
        state["cache_position"].add_(1)
        return logits

    started = time.perf_counter()
    side_stream = capture_stream if capture_stream is not None else torch_npu.npu.Stream()
    side_stream.wait_stream(torch_npu.npu.current_stream())
    with torch_npu.npu.stream(side_stream):
        for _ in range(warmup_iters):
            one_step()
    torch_npu.npu.current_stream().wait_stream(side_stream)
    synchronize_device(device)
    graph = torch_npu.npu.NPUGraph()
    if capture_stream is not None:
        with torch_npu.npu.graph(graph, stream=capture_stream):
            static_logits = one_step()
    else:
        with torch_npu.npu.graph(graph):
            static_logits = one_step()
    synchronize_device(device)
    return graph, static_logits, time.perf_counter() - started


def replay_steps(
    graph: Any, state: dict[str, Any], count: int, *, collect: bool = False
):
    tokens: list[torch.Tensor] = []
    for _ in range(count):
        graph.replay()
        if collect:
            tokens.append(state["next_token"].detach().cpu())
    return None if not collect else torch.cat(tokens, dim=1).tolist()


def profile_replay_timing(
    *,
    graph: Any,
    state: dict[str, Any],
    device: str,
    steps: int,
    reset: Any,
) -> dict[str, Any]:
    """profile_compiled_timing's protocol with graph.replay() as the step."""
    import torch_npu

    reset()
    synchronize_device(device)
    queued_start = torch_npu.npu.Event(enable_timing=True)
    queued_end = torch_npu.npu.Event(enable_timing=True)
    queued_start.record()
    host_started = time.perf_counter()
    replay_steps(graph, state, steps)
    queued_end.record()
    host_enqueue_s = time.perf_counter() - host_started
    wait_started = time.perf_counter()
    queued_end.synchronize()
    final_sync_wait_s = time.perf_counter() - wait_started
    queued_device_s = float(queued_start.elapsed_time(queued_end)) / 1000.0
    queued_wall_s = host_enqueue_s + final_sync_wait_s

    reset()
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
        graph.replay()
        end_event.record()
        production_submit_s += time.perf_counter() - submit_started
        wait_started = time.perf_counter()
        state["next_token"].detach().cpu()
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


def profile_replay_lane(
    *,
    backend: str,
    graph: Any,
    state: dict[str, Any],
    device: str,
    output_root: Path,
    steps: int,
    metric: str,
) -> dict[str, Any]:
    import torch_npu.profiler as npu_prof

    profile_dir = output_root / f"profile_replay_{backend}_{metric}"
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
        with torch.profiler.record_function(f"unirec.decode.replay.{backend}"):
            replay_steps(graph, state, steps)
        synchronize_device(device)
        profiler.step()
    synchronize_device(device)
    return {
        "profile_dir": str(profile_dir),
        "profile_steps": int(steps),
        "profile_wall_s": time.perf_counter() - started,
        "metric": metric,
    }


def run_npugraph_lane(
    *,
    runner: OptimizedUniRecRunner,
    backend: str,
    args: argparse.Namespace,
    lane_state: Any,
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    compile_meta: dict[str, Any]
    underlying_mode = "ge" if args.graph_mode == "ge_capture" else "npugraph"
    module, compile_meta = runner._compile_decode_module(
        backend="torchair",
        self_attention_backend=backend,
        compile_dynamic=False,
        cross_cache_len=args.cross_cache_length,
        batch_size=args.batch_size,
        mask_mode=args.mask_mode,
        prefetch_mode=args.prefetch_mode,
        graph_mode=underlying_mode,
    )
    state = lane_state(backend, 7)

    def reset(seed: int) -> None:
        reset_state_(state, runner, seed=seed, cache_position=args.cache_position)

    static_kernel_meta = None
    if args.static_kernel and args.graph_mode == "npugraph":
        # Static-shape kernel compilation: dump the step's op shapes, compile
        # per-shape binaries with op_compiler, install, and reselect aclnn
        # kernels. Subsequent eager calls - and the capture below - pick up
        # the static kernels.
        from torch_npu._inductor import npu_static_kernel as nsk

        build_root = (args.cache_dir.expanduser().resolve() / "static_kernel_build")
        build_root.mkdir(parents=True, exist_ok=True)
        progress("static_kernel_begin", backend=backend, build_dir=str(build_root))
        sk_started = time.perf_counter()
        with nsk.StaticKernelCompiler(build_dir=str(build_root)):
            logits = module(*call_args(state))
            torch.argmax(logits[:, -1, :].float(), dim=-1, keepdim=True)
            synchronize_device(runner.device)
        synchronize_device(runner.device)
        static_kernel_meta = {
            "build_dir": str(build_root),
            "compile_s": time.perf_counter() - sk_started,
            "uninstall_path": nsk._uninstall_path,
        }
        progress("static_kernel_end", backend=backend, **static_kernel_meta)
        # This StaticKernelCompiler variant does not auto-uninstall at exit
        # (unlike npugraph_ex's atexit hook), and the box is shared: remove
        # the installed package ourselves once the process ends. The kernels
        # stay selected in-process; uninstalling only removes the on-disk
        # package for other users.
        if nsk._uninstall_path:
            import atexit
            import subprocess

            atexit.register(
                subprocess.run,
                ["bash", nsk._uninstall_path],
                check=False,
                capture_output=True,
            )
        reset(7)

    ge_first_call_s = None
    capture_stream = None
    if args.graph_mode == "ge_capture":
        # Compile/load the GE graph eagerly before any stream capture so the
        # capture records only steady-state submission. The GE executor binds
        # to the stream of its first call and refuses any other, so the
        # prewarm, warmup, and capture all run on one caller-owned stream.
        import torch_npu

        compile_meta = dict(compile_meta)
        compile_meta["graph_mode"] = "ge_capture"
        capture_stream = torch_npu.npu.Stream()
        progress("ge_capture_prewarm_begin", backend=backend)
        first_started = time.perf_counter()
        capture_stream.wait_stream(torch_npu.npu.current_stream())
        with torch_npu.npu.stream(capture_stream):
            module(*call_args(state))
        synchronize_device(runner.device)
        ge_first_call_s = time.perf_counter() - first_started
        progress("ge_capture_prewarm_end", backend=backend, seconds=ge_first_call_s)
        reset(7)

    progress("npugraph_capture_begin", backend=backend)
    graph, static_logits, capture_s = capture_npugraph(
        module, state, runner.device, capture_stream=capture_stream
    )
    progress("npugraph_capture_end", backend=backend, seconds=capture_s)

    reset(7)
    replay_steps(graph, state, max(0, args.warmup_steps))
    synchronize_device(runner.device)

    reset(7)
    synchronize_device(runner.device)
    measured_started = time.perf_counter()
    replay_steps(graph, state, args.measure_steps)
    synchronize_device(runner.device)
    measured_s = time.perf_counter() - measured_started

    reset(11)
    validation_tokens = replay_steps(
        graph, state, args.validation_steps, collect=True
    )
    synchronize_device(runner.device)
    validation = {
        "tokens": validation_tokens,
        "logits": static_logits.detach().float().cpu(),
    }
    # Replay-integrity evidence: the degenerate validation sequence (a
    # repeated token) cannot distinguish a live graph from a frozen one, so
    # check the device-side state actually advanced and the KV slots the
    # replays should have written are populated.
    written = state["self_keys"][0][
        0, 0, args.cache_position : args.cache_position + args.validation_steps, :
    ]
    replay_checks = {
        "final_cache_position": int(state["cache_position"][0].item()),
        "expected_final_cache_position": args.cache_position + args.validation_steps,
        "validation_kv_slots_written": bool((written.abs().sum(dim=-1) > 0).all().item()),
    }
    replay_checks["state_advances"] = (
        replay_checks["final_cache_position"]
        == replay_checks["expected_final_cache_position"]
    )
    progress("replay_checks", **replay_checks)

    compiled_timing = None
    if args.compiled_timing_steps > 0:
        progress("compiled_timing_begin", backend=backend)
        compiled_timing = profile_replay_timing(
            graph=graph,
            state=state,
            device=runner.device,
            steps=args.compiled_timing_steps,
            reset=lambda: reset(17),
        )
        progress(
            "compiled_timing_end",
            backend=backend,
            **compiled_timing["production_like_d2h"],
        )
    compiled_profile = None
    if args.profile_compiled_steps > 0:
        progress("compiled_profile_begin", backend=backend)
        reset(19)
        compiled_profile = profile_replay_lane(
            backend=backend,
            graph=graph,
            state=state,
            device=runner.device,
            output_root=output_root,
            steps=args.profile_compiled_steps,
            metric=args.profile_metric,
        )
        progress("compiled_profile_end", backend=backend, **compiled_profile)

    raw_tokens = args.batch_size * args.measure_steps
    lane = {
        "compile": compile_meta,
        "first_call_s": capture_s,
        "ge_prewarm_s": ge_first_call_s,
        "static_kernel": static_kernel_meta,
        "replay_checks": replay_checks,
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
    return lane, validation


def profile_compiled_timing(
    *,
    fn: Any,
    state: dict[str, Any],
    device: str,
    steps: int,
    stepper: Any = step,
) -> dict[str, Any]:
    """Separate compiled-device work from host submission and token D2H wait."""
    import torch_npu

    synchronize_device(device)
    queued_start = torch_npu.npu.Event(enable_timing=True)
    queued_end = torch_npu.npu.Event(enable_timing=True)
    queued_start.record()
    host_started = time.perf_counter()
    run_steps(fn, state, steps, stepper=stepper)
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
        stepper(fn, state)
        end_event.record()
        production_submit_s += time.perf_counter() - submit_started
        wait_started = time.perf_counter()
        state["next_token"].detach().cpu()
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
    stepper: Any = step,
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
            run_steps(fn, state, steps, stepper=stepper)
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
    if args.lm_head_rows > 0:
        # Synthetic residency probe: replace the head with its first N rows.
        # Tokens decoded through this head are not valid model output.
        full_head = runner.model.lm_head
        sliced = torch.nn.Linear(
            full_head.in_features, args.lm_head_rows, bias=False
        ).to(device=runner.device, dtype=runner.dtype)
        with torch.no_grad():
            sliced.weight.copy_(full_head.weight[: args.lm_head_rows])
        runner.model.lm_head = sliced
        progress("lm_head_sliced", rows=args.lm_head_rows)
    if args.qkv_projection == "fused":
        runner.fuse_decoder_self_qkv()
        progress("qkv_fused")
    if args.weight_format == "nz":
        cast_count = runner.cast_decoder_weights_nz()
        progress("weights_nz", tensors=cast_count)

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
        if args.graph_mode in ("npugraph", "ge_capture"):
            lanes[backend], validations[backend] = run_npugraph_lane(
                runner=runner,
                backend=backend,
                args=args,
                lane_state=lane_state,
                output_root=output.parent,
            )
            progress(
                "lane_end",
                backend=backend,
                raw_tok_s=lanes[backend]["measure"]["raw_tok_s"],
            )
            continue
        compiled, compile_meta = runner._compile_decode_module(
            backend="torchair",
            self_attention_backend=backend,
            compile_dynamic=False,
            cross_cache_len=args.cross_cache_length,
            batch_size=args.batch_size,
            mask_mode=args.mask_mode,
            prefetch_mode=args.prefetch_mode,
            graph_mode=args.graph_mode,
            ge_tuning=tuple(args.ge_tuning),
        )
        # npugraph_ex keys captured graphs by input address: advance state in
        # place so every step presents the same tensor addresses.
        stepper = step_static if args.graph_mode.startswith("npugraph_ex") else step
        state = lane_state(backend, 7)

        def phase_state(seed: int) -> dict[str, Any]:
            # Address-stable modes must reuse one state across phases (a fresh
            # state would change input addresses and force a re-capture inside
            # the measurement); other modes keep fresh per-phase states.
            nonlocal state
            if stepper is step_static:
                reset_state_(
                    state, runner, seed=seed, cache_position=args.cache_position
                )
            else:
                state = lane_state(backend, seed)
            return state

        progress("first_call_begin", backend=backend)
        first_started = time.perf_counter()
        stepper(compiled, state)
        synchronize_device(runner.device)
        first_call_s = time.perf_counter() - first_started
        progress("first_call_end", backend=backend, seconds=first_call_s)
        run_steps(compiled, state, max(0, args.warmup_steps - 1), stepper=stepper)
        synchronize_device(runner.device)

        state = phase_state(7)
        synchronize_device(runner.device)
        measured_started = time.perf_counter()
        run_steps(compiled, state, args.measure_steps, stepper=stepper)
        synchronize_device(runner.device)
        measured_s = time.perf_counter() - measured_started

        state = phase_state(11)
        validation_logits, validation_tokens = run_steps(
            compiled, state, args.validation_steps, collect=True, stepper=stepper
        )
        synchronize_device(runner.device)
        validations[backend] = {
            "tokens": validation_tokens,
            "logits": validation_logits.detach().float().cpu(),
        }
        compiled_timing = None
        if args.compiled_timing_steps > 0:
            progress("compiled_timing_begin", backend=backend)
            timing_state = phase_state(17)
            compiled_timing = profile_compiled_timing(
                fn=compiled,
                state=timing_state,
                device=runner.device,
                steps=args.compiled_timing_steps,
                stepper=stepper,
            )
            progress(
                "compiled_timing_end",
                backend=backend,
                **compiled_timing["production_like_d2h"],
            )
        compiled_profile = None
        if args.profile_compiled_steps > 0:
            progress("compiled_profile_begin", backend=backend)
            profile_state = phase_state(19)
            compiled_profile = profile_compiled_lane(
                backend=backend,
                fn=compiled,
                state=profile_state,
                device=runner.device,
                output_root=output.parent,
                steps=args.profile_compiled_steps,
                metric=args.profile_metric,
                stepper=stepper,
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
            "lm_head_rows": args.lm_head_rows or None,
            "synthetic_head": bool(args.lm_head_rows),
        },
        "lanes": lanes,
        "comparison": comparison,
        "profiles": profiles,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
