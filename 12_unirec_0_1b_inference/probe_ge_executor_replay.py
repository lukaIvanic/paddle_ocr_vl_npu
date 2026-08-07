"""Decompose the GE compiled call's ~0.39 ms/step host cost.

Two measurements over the same warmed GE-compiled decode step:

A. cProfile of the normal compiled call loop - how much host time is inside
   the C++ executor entry (TorchNpuGraph.run / _backend) vs the Python layers
   above it (dynamo guard/bytecode, arg flattening, ge kernel preamble).
B. Direct executor replay - hook GeGraph.run to capture the exact
   (graph, inputs, assigned_outputs, stream) of a steady-state call, then
   time graph.run(...) invoked directly in a loop, skipping every Python
   layer above it. This is the host-cost floor for "GE as replay".

TIMING-ONLY: the direct loop reuses the captured input tensors verbatim, so
cache_position does not advance across iterations; device work per call is
shape-identical to a real step but tokens are meaningless.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import time
from pathlib import Path

import torch

from modeling_optimized_unirec import OptimizedUniRecRunner, synchronize_device
from text_decode_lab import call_args, make_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/workspace/models/unirec-0.1b"))
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--self-cache-length", type=int, default=256)
    parser.add_argument("--cross-cache-length", type=int, default=128)
    parser.add_argument("--cache-position", type=int, default=127)
    parser.add_argument("--backend", default="eager")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch_npu
    import torch_npu.dynamo.torchair.ge._ge_graph as ge_graph_mod

    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)

    runner = OptimizedUniRecRunner(
        model_path=args.model,
        device=args.device,
        dtype=args.dtype,
        compile_cache_dir=args.cache_dir,
    )
    runner.fuse_decoder_self_qkv()
    compiled, meta = runner._compile_decode_module(
        backend="torchair",
        self_attention_backend=args.backend,
        compile_dynamic=False,
        cross_cache_len=args.cross_cache_length,
        batch_size=1,
        mask_mode="per_step",
        prefetch_mode="none",
        graph_mode="ge",
    )
    state = make_state(
        runner,
        batch_size=1,
        self_cache_length=args.self_cache_length,
        cross_cache_length=args.cross_cache_length,
        cache_position=args.cache_position,
        seed=7,
    )

    with torch.inference_mode():
        # Warm: first call compiles/loads the GE model.
        compiled(*call_args(state))
        synchronize_device(runner.device)

        # Capture the steady-state executor call.
        captured: dict = {}
        original_run = ge_graph_mod.GeGraph.run

        def hooked_run(self, inputs, assigned_outputs=[], stream=None):  # noqa: B006
            captured["graph"] = self
            captured["inputs"] = list(inputs)
            captured["outputs"] = list(assigned_outputs)
            captured["stream"] = stream
            return original_run(self, inputs, assigned_outputs, stream)

        ge_graph_mod.GeGraph.run = hooked_run
        try:
            compiled(*call_args(state))
        finally:
            ge_graph_mod.GeGraph.run = original_run
        synchronize_device(runner.device)
        if "graph" not in captured:
            raise RuntimeError("GeGraph.run hook never fired; call path changed")

        input_summary = [
            (
                tuple(item.shape),
                str(item.dtype).replace("torch.", ""),
            )
            if isinstance(item, torch.Tensor)
            else repr(item)
            for item in captured["inputs"]
        ]

        # A. Normal compiled-call loop: wall + device + cProfile decomposition.
        synchronize_device(runner.device)
        start_evt = torch_npu.npu.Event(enable_timing=True)
        end_evt = torch_npu.npu.Event(enable_timing=True)
        start_evt.record()
        wall_started = time.perf_counter()
        for _ in range(args.steps):
            compiled(*call_args(state))
        wrapper_host_s = time.perf_counter() - wall_started
        end_evt.record()
        end_evt.synchronize()
        wrapper_device_s = float(start_evt.elapsed_time(end_evt)) / 1000.0

        profiler = cProfile.Profile()
        synchronize_device(runner.device)
        profiler.enable()
        for _ in range(args.steps):
            compiled(*call_args(state))
        profiler.disable()
        synchronize_device(runner.device)
        stats_stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stats_stream)
        stats.sort_stats("cumulative").print_stats(25)
        profile_text = stats_stream.getvalue()
        executor_tt = 0.0
        total_tt = 0.0
        for (filename, _lineno, funcname), (_cc, _nc, tt, ct, _callers) in stats.stats.items():
            total_tt += tt
            if "_backend.py" in filename and funcname == "run":
                executor_tt += ct  # cumulative: includes the C++ super().run

        # B. Direct executor replay loop.
        graph = captured["graph"]
        direct_inputs = captured["inputs"]
        direct_outputs = captured["outputs"]
        direct_stream = captured["stream"]
        # Warm the direct path once.
        graph.run(direct_inputs, direct_outputs, direct_stream)
        synchronize_device(runner.device)
        start_evt2 = torch_npu.npu.Event(enable_timing=True)
        end_evt2 = torch_npu.npu.Event(enable_timing=True)
        start_evt2.record()
        direct_started = time.perf_counter()
        for _ in range(args.steps):
            graph.run(direct_inputs, direct_outputs, direct_stream)
        direct_host_s = time.perf_counter() - direct_started
        end_evt2.record()
        end_evt2.synchronize()
        direct_device_s = float(start_evt2.elapsed_time(end_evt2)) / 1000.0

    steps = args.steps
    payload = {
        "kind": "probe_ge_executor_replay",
        "note": (
            "Direct loop replays captured static inputs; cache_position frozen; "
            "timing-only, tokens invalid"
        ),
        "compile_meta": {k: v for k, v in meta.items() if isinstance(v, (str, int, bool, type(None)))},
        "captured": {
            "num_inputs": len(captured["inputs"]),
            "num_outputs": len(captured["outputs"]),
            "stream_is_none": captured["stream"] is None,
            "input_summary_first_10": input_summary[:10],
        },
        "wrapper_call": {
            "steps": steps,
            "host_per_step_ms": wrapper_host_s * 1000.0 / steps,
            "device_per_step_ms": wrapper_device_s / steps,
        },
        "executor_cprofile": {
            "total_python_tottime_s": total_tt,
            "cumulative_in_backend_run_s": executor_tt,
            "backend_run_share_of_wrapper_host": (
                executor_tt / wrapper_host_s if wrapper_host_s else None
            ),
        },
        "direct_executor_call": {
            "steps": steps,
            "host_per_step_ms": direct_host_s * 1000.0 / steps,
            "device_per_step_ms": direct_device_s / steps,
        },
        "profile_top25": profile_text,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "profile_top25"}, indent=2))
    print(profile_text)


if __name__ == "__main__":
    main()
