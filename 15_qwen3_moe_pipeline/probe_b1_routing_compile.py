#!/usr/bin/env python3

"""Compile-screen B1/top-8 replacements for ``npu_moe_init_routing_v2``.

This is deliberately a routing-only probe.  Every graph contains 24 routing
calls, which is the number exercised by one pipeline stage.  Candidate outputs
are checked exactly against InitRoutingV2 before their timing is reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu


VARIANTS = (
    "v2_active8",
    "v2_dropless0",
    "manual_scatter_rank_expand",
    "manual_scatter_rank_repeat",
    "manual_one_hot_rank",
    "manual_compare_rank",
    "manual_compare_buffer_rank",
    "manual_identity_rank",
    "manual_npu_scatter_rank",
    "manual_bincount_rank",
    "manual_scatter_sort_order",
    "legacy_pair",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Static TorchAir B1/top-8 MoE routing primitive probe."
    )
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--expert-num", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--benchmark-steps", type=int, default=2000)
    parser.add_argument(
        "--compile-cache-dir",
        type=Path,
        default=Path(".runtime_cache/15_qwen3_moe_pipeline/routing_probe"),
    )
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--summary-out", type=Path)
    return parser.parse_args()


def import_cache_compile():
    try:
        from torch_npu.dynamo.torchair.inference import cache_compile
    except ImportError:
        from torchair.inference import cache_compile
    return cache_compile


def synchronize(device: torch.device) -> None:
    torch.npu.synchronize(device)


def dynamo_stats() -> dict[str, int]:
    counters = torch._dynamo.utils.counters
    return {
        "unique_graphs": int(counters["stats"]["unique_graphs"]),
        "calls_captured": int(counters["stats"]["calls_captured"]),
        "frames_total": int(counters["frames"]["total"]),
        "frames_ok": int(counters["frames"]["ok"]),
    }


class RoutingGraph(nn.Module):
    """Run one routing implementation repeatedly inside one static graph."""

    def __init__(
        self,
        variant: str,
        *,
        layers: int,
        hidden_size: int,
        expert_num: int,
        top_k: int,
    ) -> None:
        super().__init__()
        self.variant = variant
        self.layers = layers
        self.hidden_size = hidden_size
        self.expert_num = expert_num
        self.top_k = top_k
        self.register_buffer(
            "expert_axis",
            torch.arange(expert_num, dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "expert_identity",
            torch.eye(expert_num, dtype=torch.int64),
            persistent=False,
        )

    def _v2(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        *,
        active_num: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        expanded, row_idx, counts, _scale = torch_npu.npu_moe_init_routing_v2(
            hidden_states,
            selected_experts,
            scale=None,
            offset=None,
            active_num=active_num,
            expert_capacity=-1,
            expert_num=self.expert_num,
            drop_pad_mode=0,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            quant_mode=-1,
            active_expert_range=[0, self.expert_num],
            row_idx_type=0,
        )
        return expanded, row_idx, counts

    def _manual(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ids = selected_experts.reshape(self.top_k)
        ids_i64 = ids.to(dtype=torch.int64)

        if self.variant == "manual_scatter_rank_repeat":
            expanded = hidden_states.repeat(self.top_k, 1)
        else:
            expanded = hidden_states.expand(self.top_k, self.hidden_size).contiguous()

        if self.variant == "manual_scatter_sort_order":
            row_idx = torch.sort(ids).indices.to(dtype=torch.int32)
        else:
            # For each source-order route, compute its position in expert order.
            row_idx = (ids.view(self.top_k, 1) > ids.view(1, self.top_k)).sum(
                dim=1, dtype=torch.int32
            )

        if self.variant in (
            "manual_scatter_rank_expand",
            "manual_scatter_rank_repeat",
            "manual_scatter_sort_order",
        ):
            counts = torch.zeros(
                self.expert_num, dtype=torch.int64, device=hidden_states.device
            ).scatter(
                0,
                ids_i64,
                torch.ones(self.top_k, dtype=torch.int64, device=hidden_states.device),
            )
        elif self.variant == "manual_one_hot_rank":
            counts = F.one_hot(ids_i64, num_classes=self.expert_num).sum(
                dim=0, dtype=torch.int64
            )
        elif self.variant in ("manual_compare_rank", "manual_compare_buffer_rank"):
            if self.variant == "manual_compare_rank":
                expert_axis = torch.arange(
                    self.expert_num, dtype=torch.int32, device=hidden_states.device
                )
            else:
                expert_axis = self.expert_axis
            counts = (ids.view(self.top_k, 1) == expert_axis.view(1, -1)).sum(
                dim=0, dtype=torch.int64
            )
        elif self.variant == "manual_identity_rank":
            counts = torch.index_select(self.expert_identity, 0, ids_i64).sum(
                dim=0, dtype=torch.int64
            )
        elif self.variant == "manual_npu_scatter_rank":
            # The NPU ScatterUpdate kernel requires a 2D-8D int32 data tensor.
            # Top-k expert IDs are unique, so assigning one gives exact counts.
            count_rows = torch.zeros(
                (self.top_k, self.expert_num),
                dtype=torch.int32,
                device=hidden_states.device,
            )
            torch_npu.scatter_update_(
                count_rows,
                ids_i64,
                torch.ones(
                    (self.top_k, 1), dtype=torch.int32, device=hidden_states.device
                ),
                1,
            )
            counts = count_rows.sum(dim=0, dtype=torch.int64)
        elif self.variant == "manual_bincount_rank":
            counts = torch.bincount(ids_i64, minlength=self.expert_num)
        else:
            raise AssertionError(f"Unhandled manual variant: {self.variant}")
        return expanded, row_idx, counts

    def _legacy_pair(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row_idx = torch.arange(
            self.top_k, dtype=torch.int32, device=hidden_states.device
        ).view(1, self.top_k)
        expanded, expanded_row_idx, expanded_expert_idx = (
            torch_npu.npu_moe_init_routing(
                hidden_states,
                row_idx,
                selected_experts,
                active_num=1,
            )
        )
        counts = torch_npu.npu_moe_compute_expert_tokens(
            expanded_expert_idx.reshape(-1), self.expert_num
        ).to(dtype=torch.int64)
        return expanded, expanded_row_idx, counts

    def _route(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.variant == "v2_active8":
            return self._v2(hidden_states, selected_experts, active_num=self.top_k)
        if self.variant == "v2_dropless0":
            return self._v2(hidden_states, selected_experts, active_num=0)
        if self.variant.startswith("manual_"):
            return self._manual(hidden_states, selected_experts)
        if self.variant == "legacy_pair":
            return self._legacy_pair(hidden_states, selected_experts)
        raise AssertionError(f"Unhandled variant: {self.variant}")

    def forward(
        self,
        hidden_stack: torch.Tensor,
        selected_stack: torch.Tensor,
    ) -> tuple[
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
    ]:
        expanded_outputs = []
        row_idx_outputs = []
        count_outputs = []
        for layer in range(self.layers):
            expanded, row_idx, counts = self._route(
                hidden_stack[layer], selected_stack[layer]
            )
            expanded_outputs.append(expanded)
            row_idx_outputs.append(row_idx)
            count_outputs.append(counts)
        return (
            tuple(expanded_outputs),
            tuple(row_idx_outputs),
            tuple(count_outputs),
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
    if layers >= 1 and expert_num >= top_k:
        selected[0, 0] = torch.arange(top_k)
    if layers >= 2 and expert_num >= top_k:
        selected[1, 0] = torch.arange(expert_num - top_k, expert_num)
    if layers >= 3 and expert_num >= 128 and top_k == 8:
        selected[2, 0] = torch.tensor([127, 0, 64, 1, 126, 63, 2, 125])
    return (
        hidden.to(device=device),
        selected.to(device=device, dtype=torch.int32),
    )


def compare_outputs(
    reference: tuple[tuple[torch.Tensor, ...], ...],
    candidate: tuple[tuple[torch.Tensor, ...], ...],
) -> dict[str, object]:
    names = ("expanded_hidden", "expanded_row_idx", "group_counts")
    result: dict[str, object] = {"all_exact": True, "outputs": {}}
    for name, reference_group, candidate_group in zip(names, reference, candidate):
        exact = all(
            torch.equal(expected, actual)
            for expected, actual in zip(reference_group, candidate_group)
        )
        result["outputs"][name] = {
            "exact": exact,
            "reference_dtype": str(reference_group[0].dtype),
            "candidate_dtype": str(candidate_group[0].dtype),
            "reference_shape": list(reference_group[0].shape),
            "candidate_shape": list(candidate_group[0].shape),
            "reference_first": reference_group[0].reshape(-1)[:16].tolist(),
            "candidate_first": candidate_group[0].reshape(-1)[:16].tolist(),
        }
        result["all_exact"] = bool(result["all_exact"]) and exact
    return result


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
    if args.layers < 1 or args.hidden_size < 1 or args.expert_num < args.top_k:
        raise ValueError("Invalid static shape")
    if args.warmup_steps < 0 or args.benchmark_steps < 1:
        raise ValueError("Invalid benchmark step count")

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
    reference_module = RoutingGraph(
        "v2_active8",
        layers=args.layers,
        hidden_size=args.hidden_size,
        expert_num=args.expert_num,
        top_k=args.top_k,
    ).to(device)
    candidate_module = RoutingGraph(
        args.variant,
        layers=args.layers,
        hidden_size=args.hidden_size,
        expert_num=args.expert_num,
        top_k=args.top_k,
    ).to(device)

    reference = reference_module(hidden, selected)
    candidate = candidate_module(hidden, selected)
    synchronize(device)
    correctness = compare_outputs(reference, candidate)
    if args.variant == "manual_npu_scatter_rank" and not correctness["all_exact"]:
        raise RuntimeError(
            "NPU ScatterUpdate candidate failed synchronized exact routing parity"
        )

    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    shape_key = (
        f"{args.variant}_l{args.layers}_h{args.hidden_size}_e{args.expert_num}_"
        f"k{args.top_k}_src{source_hash}"
    )
    cache_dir = args.compile_cache_dir.expanduser().resolve() / shape_key
    cache_was_warm = cache_dir.is_dir() and any(cache_dir.iterdir())
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    from torch_npu.dynamo.torchair.configs.compiler_config import CompilerConfig

    wrapper_started = time.perf_counter()
    compiled = import_cache_compile()(
        candidate_module.forward,
        config=CompilerConfig(),
        dynamic=False,
        cache_dir=str(cache_dir),
        ge_cache=True,
        fullgraph=True,
    )
    compile_wrapper_sec = time.perf_counter() - wrapper_started

    first_started = time.perf_counter()
    compiled_output = compiled(hidden, selected)
    synchronize(device)
    first_call_sec = time.perf_counter() - first_started
    compiled_correctness = compare_outputs(reference, compiled_output)
    stats_after_first = dynamo_stats()

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
        "format": "qwen3_moe_b1_routing_compile_probe_v1",
        "variant": args.variant,
        "shape": {
            "layers": args.layers,
            "hidden_size": args.hidden_size,
            "expert_num": args.expert_num,
            "top_k": args.top_k,
            "dtype": "bfloat16",
        },
        "correctness": {
            "raw_candidate_vs_v2_active8": correctness,
            "compiled_candidate_vs_v2_active8": compiled_correctness,
        },
        "compile": {
            "fullgraph": True,
            "dynamic": False,
            "cache_dir": str(cache_dir),
            "cache_was_warm": cache_was_warm,
            "wrapper_sec": compile_wrapper_sec,
            "first_call_sec": first_call_sec,
            "dynamo_stats_after_first": stats_after_first,
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
