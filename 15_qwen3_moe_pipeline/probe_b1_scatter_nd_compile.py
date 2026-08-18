#!/usr/bin/env python3

"""Screen direct 1-D ScatterNd group-count construction for B1 MoE routing."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch_npu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--expert-num", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--benchmark-steps", type=int, default=10000)
    parser.add_argument(
        "--compile-cache-dir",
        type=Path,
        default=Path(".runtime_cache/15_qwen3_moe_pipeline/scatter_nd_probe"),
    )
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--summary-out", type=Path)
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    torch.npu.synchronize(device)


def cache_compile():
    try:
        from torch_npu.dynamo.torchair.inference import cache_compile as compile_fn
    except ImportError:
        from torchair.inference import cache_compile as compile_fn
    return compile_fn


class ScatterNdRoutingGraph(nn.Module):
    def __init__(
        self,
        *,
        layers: int,
        hidden_size: int,
        expert_num: int,
        top_k: int,
    ) -> None:
        super().__init__()
        self.layers = layers
        self.hidden_size = hidden_size
        self.expert_num = expert_num
        self.top_k = top_k
        self.register_buffer(
            "count_updates",
            torch.ones(top_k, dtype=torch.int64),
            persistent=False,
        )

    def forward(
        self,
        hidden_stack: torch.Tensor,
        selected_stack: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        expanded_outputs = []
        row_idx_outputs = []
        count_outputs = []
        for layer in range(self.layers):
            hidden = hidden_stack[layer]
            ids = selected_stack[layer].reshape(self.top_k)
            expanded = hidden.expand(self.top_k, self.hidden_size).contiguous()
            row_idx = (ids.view(self.top_k, 1) > ids.view(1, self.top_k)).sum(
                dim=1, dtype=torch.int32
            )
            counts = torch.zeros(
                self.expert_num,
                dtype=torch.int64,
                device=hidden_stack.device,
            )
            torch_npu.npu_scatter_nd_update_(
                counts,
                ids.view(self.top_k, 1),
                self.count_updates,
            )
            expanded_outputs.append(expanded)
            row_idx_outputs.append(row_idx)
            count_outputs.append(counts)
        return (
            torch.stack(expanded_outputs),
            torch.stack(row_idx_outputs),
            torch.stack(count_outputs),
        )


def make_inputs(
    *,
    layers: int,
    hidden_size: int,
    expert_num: int,
    top_k: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260818)
    hidden = torch.randn(
        layers, 1, hidden_size, dtype=torch.bfloat16, generator=generator
    )
    selected = torch.stack(
        [torch.randperm(expert_num, generator=generator)[:top_k] for _ in range(layers)]
    ).view(layers, 1, top_k)
    if layers >= 1:
        selected[0, 0] = torch.arange(top_k)
    if layers >= 2:
        selected[1, 0] = torch.arange(expert_num - top_k, expert_num)
    if layers >= 3 and expert_num >= 128 and top_k == 8:
        selected[2, 0] = torch.tensor([127, 0, 64, 1, 126, 63, 2, 125])
    return hidden.to(device), selected.to(device=device, dtype=torch.int32)


def reference_outputs(
    hidden_stack: torch.Tensor,
    selected_stack: torch.Tensor,
    *,
    expert_num: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    expanded_outputs = []
    row_idx_outputs = []
    count_outputs = []
    for layer in range(hidden_stack.shape[0]):
        expanded, row_idx, counts, _scale = torch_npu.npu_moe_init_routing_v2(
            hidden_stack[layer],
            selected_stack[layer],
            scale=None,
            offset=None,
            active_num=top_k,
            expert_capacity=-1,
            expert_num=expert_num,
            drop_pad_mode=0,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            quant_mode=-1,
            active_expert_range=[0, expert_num],
            row_idx_type=0,
        )
        expanded_outputs.append(expanded)
        row_idx_outputs.append(row_idx)
        count_outputs.append(counts)
    return (
        torch.stack(expanded_outputs),
        torch.stack(row_idx_outputs),
        torch.stack(count_outputs),
    )


def compare_outputs(
    expected: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    actual: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, object]:
    names = ("expanded_hidden", "expanded_row_idx", "group_counts")
    outputs = {}
    all_exact = True
    for name, expected_tensor, actual_tensor in zip(names, expected, actual):
        exact = torch.equal(expected_tensor, actual_tensor)
        outputs[name] = {
            "exact": exact,
            "shape": list(actual_tensor.shape),
            "dtype": str(actual_tensor.dtype),
        }
        all_exact = all_exact and exact
    return {"all_exact": all_exact, "outputs": outputs}


def profile_once(
    compiled,
    hidden: torch.Tensor,
    selected: torch.Tensor,
    *,
    profile_dir: Path,
    device: torch.device,
) -> str:
    import torch_npu.profiler as npu_prof

    shutil.rmtree(profile_dir, ignore_errors=True)
    profile_dir.mkdir(parents=True, exist_ok=True)
    schedule = npu_prof.schedule(wait=0, warmup=1, active=1, repeat=1)
    experimental_config = npu_prof._ExperimentalConfig(
        profiler_level=npu_prof.ProfilerLevel.Level1,
        aic_metrics=npu_prof.AiCMetrics.PipeUtilization,
        export_type=npu_prof.ExportType.Text,
    )
    with npu_prof.profile(
        activities=[npu_prof.ProfilerActivity.CPU, npu_prof.ProfilerActivity.NPU],
        schedule=schedule,
        on_trace_ready=npu_prof.tensorboard_trace_handler(
            str(profile_dir), analyse_flag=True
        ),
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
        experimental_config=experimental_config,
    ) as prof:
        for _ in range(2):
            compiled(hidden, selected)
            synchronize(device)
            prof.step()
    return str(profile_dir)


def main() -> None:
    args = parse_args()
    torch.npu.set_compile_mode(jit_compile=False)
    device = torch.device(args.device)
    torch.npu.set_device(device)
    hidden, selected = make_inputs(
        layers=args.layers,
        hidden_size=args.hidden_size,
        expert_num=args.expert_num,
        top_k=args.top_k,
        device=device,
    )
    alternate_selected = torch.roll(selected, shifts=17, dims=-1)
    module = ScatterNdRoutingGraph(
        layers=args.layers,
        hidden_size=args.hidden_size,
        expert_num=args.expert_num,
        top_k=args.top_k,
    ).to(device)

    raw_expected = reference_outputs(
        hidden, selected, expert_num=args.expert_num, top_k=args.top_k
    )
    raw_actual = module(hidden, selected)
    synchronize(device)
    raw_correctness = compare_outputs(raw_expected, raw_actual)
    if not raw_correctness["all_exact"]:
        raise RuntimeError("Raw ScatterNd routing output does not match InitRoutingV2")

    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    cache_dir = args.compile_cache_dir.expanduser().resolve() / (
        f"l{args.layers}_h{args.hidden_size}_e{args.expert_num}_k{args.top_k}_"
        f"src{source_hash}"
    )
    cache_was_warm = cache_dir.is_dir() and any(cache_dir.iterdir())
    cache_dir.mkdir(parents=True, exist_ok=True)
    from torch_npu.dynamo.torchair.configs.compiler_config import CompilerConfig

    compiled = cache_compile()(
        module.forward,
        config=CompilerConfig(),
        dynamic=False,
        cache_dir=str(cache_dir),
        ge_cache=True,
        fullgraph=True,
    )
    first_started = time.perf_counter()
    compiled_actual = compiled(hidden, selected)
    synchronize(device)
    first_call_sec = time.perf_counter() - first_started
    compiled_correctness = compare_outputs(raw_expected, compiled_actual)

    # Alternate the indices, then return to the original input. This catches a
    # stale in-place count buffer that a single correctness call would miss.
    alternate_expected = reference_outputs(
        hidden,
        alternate_selected,
        expert_num=args.expert_num,
        top_k=args.top_k,
    )
    alternate_actual = compiled(hidden, alternate_selected)
    original_again = compiled(hidden, selected)
    synchronize(device)
    state_reset_correctness = {
        "alternate": compare_outputs(alternate_expected, alternate_actual),
        "original_again": compare_outputs(raw_expected, original_again),
    }
    if not (
        compiled_correctness["all_exact"]
        and state_reset_correctness["alternate"]["all_exact"]
        and state_reset_correctness["original_again"]["all_exact"]
    ):
        raise RuntimeError("Compiled ScatterNd routing failed exact or reset parity")

    for _ in range(args.warmup_steps):
        compiled(hidden, selected)
    synchronize(device)
    started = time.perf_counter()
    for _ in range(args.benchmark_steps):
        compiled(hidden, selected)
    synchronize(device)
    elapsed = time.perf_counter() - started

    profile_path = None
    if args.profile_dir is not None:
        profile_path = profile_once(
            compiled,
            hidden,
            selected,
            profile_dir=args.profile_dir.expanduser().resolve(),
            device=device,
        )

    summary = {
        "format": "qwen3_moe_b1_scatter_nd_probe_v1",
        "shape": {
            "layers": args.layers,
            "hidden_size": args.hidden_size,
            "expert_num": args.expert_num,
            "top_k": args.top_k,
        },
        "correctness": {
            "raw": raw_correctness,
            "compiled": compiled_correctness,
            "state_reset": state_reset_correctness,
        },
        "compile": {
            "fullgraph": True,
            "dynamic": False,
            "cache_dir": str(cache_dir),
            "cache_was_warm": cache_was_warm,
            "first_call_sec": first_call_sec,
        },
        "benchmark": {
            "warmup_steps": args.warmup_steps,
            "steps": args.benchmark_steps,
            "elapsed_sec": elapsed,
            "graph_calls_per_s": args.benchmark_steps / elapsed,
            "us_per_24_layer_graph": elapsed * 1e6 / args.benchmark_steps,
            "us_per_routing_call": elapsed * 1e6 / args.benchmark_steps / args.layers,
        },
        "profile_dir": profile_path,
    }
    print(json.dumps(summary, indent=2))
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
