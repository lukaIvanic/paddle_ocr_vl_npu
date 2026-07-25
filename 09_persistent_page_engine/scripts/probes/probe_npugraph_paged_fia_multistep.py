#!/usr/bin/env python3
"""Replay one paged-FIA graph across advancing batched decode iterations."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch_npu

HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXPERIMENT_ROOT))

import benchmark_paged_fia_full_decoder as bench
from benchmark_npugraph_paged_fia_full_decoder import (
    DecodeAndArgmax,
    _cast_paged_cache_to_nd,
)
from paddleocr_vl.model.text_decode import (
    TextDecodeRuntime,
    cast_decode_linear_weights_to_nz,
    prepare_decode_optimization_modules,
)
from utils.timing import synchronize


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "tmp/09_persistent_page_engine/text_decode_lab"
    / "npugraph_paged_fia_multistep.json"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=bench.DEFAULT_MODEL)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=bench.DEFAULT_CACHE_ROOT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cache-length", type=int, default=4096)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--min-position", type=int, default=128)
    parser.add_argument("--max-position", type=int, default=768)
    parser.add_argument("--correctness-steps", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--timing-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.cache_length <= 0:
        parser.error("--cache-length must be positive")
    if args.block_size <= 0 or args.cache_length % args.block_size:
        parser.error("--block-size must evenly divide --cache-length")
    if not 0 <= args.min_position <= args.max_position:
        parser.error("positions must be non-negative and ordered")
    total_steps = (
        args.correctness_steps
        + args.warmup_steps
        + args.timing_steps
    )
    if args.max_position + total_steps >= args.cache_length:
        parser.error("decode progression exceeds --cache-length")
    if (
        args.correctness_steps <= 0
        or args.warmup_steps < 0
        or args.timing_steps <= 0
    ):
        parser.error("step counts must be positive except warmup may be zero")
    return args


@dataclass
class _FIAGraphTask:
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    block_table: torch.Tensor
    output: torch.Tensor
    softmax_lse: torch.Tensor
    workspace: torch.Tensor
    event: torch.npu.ExternalEvent
    handle: object
    block_size: int
    num_query_heads: int
    num_key_value_heads: int
    softmax_scale: float


class _FIAGraphTaskRecorder:
    """Mirror vLLM-Ascend's FIA graph-task capture/update mechanism."""

    def __init__(self, *, batch_size: int, device: torch.device):
        self.batch_size = int(batch_size)
        self.device = device
        self.tasks: list[_FIAGraphTask] = []
        self.shared_workspace: torch.Tensor | None = None
        self.update_stream = torch_npu.npu.Stream(device=device)

    def __call__(
        self,
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        block_table: torch.Tensor,
        block_size: int,
        actual_seq_kv_lengths: tuple[int, ...],
        num_query_heads: int,
        num_key_value_heads: int,
        softmax_scale: float,
    ) -> torch.Tensor:
        common = {
            "query": query,
            "key": key,
            "value": value,
            "block_table": block_table,
            "input_layout": "BNSD",
            "block_size": int(block_size),
            "actual_seq_qlen": [1] * self.batch_size,
            "actual_seq_kvlen": list(actual_seq_kv_lengths),
            "num_key_value_heads": int(num_key_value_heads),
            "num_query_heads": int(num_query_heads),
            "sparse_mode": 0,
            "softmax_scale": float(softmax_scale),
            "inner_precise": 1,
        }
        if self.shared_workspace is None:
            self.shared_workspace = (
                torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(
                    **common
                )
            )
        output = torch.empty_like(query)
        softmax_lse = torch.empty(
            (1,),
            device=query.device,
            dtype=query.dtype,
        )
        stream = torch_npu.npu.current_stream()
        event = torch.npu.ExternalEvent()
        event.wait(stream)
        event.reset(stream)
        torch.npu.graph_task_group_begin(stream)
        torch_npu.npu_fused_infer_attention_score_v2.out(
            **common,
            workspace=self.shared_workspace,
            out=[output, softmax_lse],
        )
        handle = torch.npu.graph_task_group_end(stream)
        self.tasks.append(
            _FIAGraphTask(
                query=query,
                key=key,
                value=value,
                block_table=block_table,
                output=output,
                softmax_lse=softmax_lse,
                workspace=self.shared_workspace,
                event=event,
                handle=handle,
                block_size=int(block_size),
                num_query_heads=int(num_query_heads),
                num_key_value_heads=int(num_key_value_heads),
                softmax_scale=float(softmax_scale),
            )
        )
        return output

    def update(self, actual_seq_kv_lengths: Sequence[int]) -> None:
        lengths = [int(length) for length in actual_seq_kv_lengths]
        if len(lengths) != self.batch_size:
            raise ValueError("one sequence length is required per batch row")
        with torch_npu.npu.stream(self.update_stream):
            for task in self.tasks:
                torch.npu.graph_task_update_begin(
                    self.update_stream,
                    task.handle,
                )
                torch_npu.npu_fused_infer_attention_score_v2.out(
                    query=task.query,
                    key=task.key,
                    value=task.value,
                    block_table=task.block_table,
                    input_layout="BNSD",
                    block_size=task.block_size,
                    actual_seq_qlen=[1] * self.batch_size,
                    actual_seq_kvlen=lengths,
                    num_key_value_heads=task.num_key_value_heads,
                    num_query_heads=task.num_query_heads,
                    sparse_mode=0,
                    softmax_scale=task.softmax_scale,
                    inner_precise=1,
                    workspace=task.workspace,
                    out=[task.output, task.softmax_lse],
                )
                torch.npu.graph_task_update_end(self.update_stream)
                task.event.record(self.update_stream)


def _positions(args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    if args.batch_size == 1:
        values = torch.tensor([args.max_position], dtype=torch.int64)
    else:
        values = torch.linspace(
            args.min_position,
            args.max_position,
            steps=args.batch_size,
            dtype=torch.float64,
        ).round().to(torch.int64)
    return values.to(device)


def _reset_paged_from_dense(
    dense: bench.LocalPaddleOCRVLStaticCache,
    paged: bench.PagedCache,
) -> None:
    for dense_tensor, paged_tensor in zip(
        dense.flat_tensors(),
        paged.flat_tensors(),
    ):
        paged_tensor.copy_(
            bench._dense_to_paged_nz(
                dense_tensor,
                block_size=paged.block_size,
            )
        )


def _timed_increfa(
    fn,
    state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    flat_cache: tuple[torch.Tensor, ...],
    *,
    batch_size: int,
    warmup: int,
    steps: int,
) -> dict[str, float]:
    input_ids, positions, rope_deltas = state
    for _ in range(warmup):
        logits = fn(input_ids, positions, rope_deltas, *flat_cache)
        input_ids.copy_(
            torch.argmax(logits[:, -1, :].float(), dim=-1, keepdim=True)
        )
        positions.add_(1)
    synchronize(input_ids.device)
    start = torch_npu.npu.Event(enable_timing=True)
    end = torch_npu.npu.Event(enable_timing=True)
    wall_started = time.perf_counter()
    start.record()
    for _ in range(steps):
        logits = fn(input_ids, positions, rope_deltas, *flat_cache)
        input_ids.copy_(
            torch.argmax(logits[:, -1, :].float(), dim=-1, keepdim=True)
        )
        positions.add_(1)
    end.record()
    end.synchronize()
    wall_s = time.perf_counter() - wall_started
    device_s = float(start.elapsed_time(end)) / 1000.0
    return {
        "device_s": device_s,
        "wall_s": wall_s,
        "mean_device_ms": device_s * 1000.0 / steps,
        "raw_device_tokens_per_s": batch_size * steps / device_s,
        "raw_wall_tokens_per_s": batch_size * steps / wall_s,
    }


def _timed_paged(
    graph: torch.npu.NPUGraph,
    output: tuple[torch.Tensor, torch.Tensor],
    recorder: _FIAGraphTaskRecorder,
    state: tuple[torch.Tensor, torch.Tensor],
    host_positions: list[int],
    *,
    batch_size: int,
    warmup: int,
    steps: int,
) -> dict[str, float]:
    input_ids, positions = state
    _logits, tokens = output

    def one_step() -> None:
        recorder.update([position + 1 for position in host_positions])
        graph.replay()
        input_ids.copy_(tokens)
        positions.add_(1)
        for index in range(len(host_positions)):
            host_positions[index] += 1

    for _ in range(warmup):
        one_step()
    synchronize(input_ids.device)
    start = torch_npu.npu.Event(enable_timing=True)
    end = torch_npu.npu.Event(enable_timing=True)
    wall_started = time.perf_counter()
    start.record()
    for _ in range(steps):
        one_step()
    end.record()
    end.synchronize()
    wall_s = time.perf_counter() - wall_started
    device_s = float(start.elapsed_time(end)) / 1000.0
    return {
        "device_s": device_s,
        "wall_s": wall_s,
        "mean_device_ms": device_s * 1000.0 / steps,
        "raw_device_tokens_per_s": batch_size * steps / device_s,
        "raw_wall_tokens_per_s": batch_size * steps / wall_s,
        "graph_task_updates_per_step": len(recorder.tasks),
    }


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not torch.npu.is_available():
        raise RuntimeError("probe requires an available Ascend NPU")
    device = torch.device("npu:0")
    dtype = torch.float16
    torch.npu.set_compile_mode(jit_compile=False)
    config = bench.PaddleOCRVLConfig.from_model_dir(args.model)
    model = bench._create_random_model(
        config,
        device=device,
        dtype=dtype,
        seed=args.seed,
    )
    optimization = prepare_decode_optimization_modules(
        model,
        bench.OPTIMIZATION,
    )
    weight_format = cast_decode_linear_weights_to_nz(model)
    synchronize(device)
    incre_runtime = TextDecodeRuntime(
        model,
        backend="torchair",
        device=device,
        cache_root=args.cache_dir,
        batch_size=args.batch_size,
        cache_length=args.cache_length,
        dtype=dtype,
        model_dir=args.model,
        linear_weight_format=str(weight_format["effective_mode"]),
        optimization=optimization,
    )
    dense_cache, paged_cache = bench._allocate_matching_caches(
        config.text_config,
        batch_size=args.batch_size,
        cache_length=args.cache_length,
        block_size=args.block_size,
        device=device,
        dtype=dtype,
        seed=args.seed + 1000,
    )
    _cast_paged_cache_to_nd(paged_cache)
    initial_positions = _positions(args, device)
    dense_positions = initial_positions.clone()
    paged_positions = initial_positions.clone()
    dense_input = (
        torch.arange(args.batch_size, device=device, dtype=torch.int64)
        .add_(17)
        .view(args.batch_size, 1)
    )
    paged_input = dense_input.clone()
    dense_rope = torch.zeros(
        (args.batch_size, 1),
        device=device,
        dtype=torch.int64,
    )
    paged_rope = dense_rope.clone()
    initial_lengths = tuple(
        int(position) + 1
        for position in initial_positions.cpu().tolist()
    )
    paged_stage = bench.PagedFIATextDecodeStage(
        model,
        block_size=args.block_size,
        cache_update_mode="scatter_pa_inplace",
        optimization=optimization,
        native_fia=True,
        fixed_actual_kv_lengths=initial_lengths,
    ).eval()
    captured_stage = DecodeAndArgmax(paged_stage).eval()
    paged_args = (
        paged_input,
        paged_positions,
        paged_rope,
        paged_cache.block_table,
        *paged_cache.flat_tensors(),
    )
    captured_stage(*paged_args)
    synchronize(device)
    recorder = _FIAGraphTaskRecorder(
        batch_size=args.batch_size,
        device=device,
    )
    paged_stage.native_fia_runner = recorder
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        graph_output = captured_stage(*paged_args)
    paged_stage.native_fia_runner = None
    synchronize(device)
    if len(recorder.tasks) != config.text_config.num_hidden_layers:
        raise RuntimeError(
            f"captured {len(recorder.tasks)} FIA tasks, expected "
            f"{config.text_config.num_hidden_layers}"
        )
    _reset_paged_from_dense(dense_cache, paged_cache)
    synchronize(device)

    host_positions = [int(value) for value in initial_positions.cpu().tolist()]
    step_results = []
    correctness_passed = True
    for step in range(args.correctness_steps):
        dense_logits = incre_runtime.fn(
            dense_input,
            dense_positions,
            dense_rope,
            *dense_cache.flat_tensors(),
        )
        recorder.update([position + 1 for position in host_positions])
        graph.replay()
        synchronize(device)
        paged_logits, paged_tokens = graph_output
        dense_tokens = torch.argmax(
            dense_logits[:, -1, :].float(),
            dim=-1,
            keepdim=True,
        )
        logits_delta = bench._delta_stats(paged_logits, dense_logits)
        kv_delta = bench._cache_delta_stats(
            bench._dense_cache_written_values(
                dense_cache,
                dense_positions,
            ),
            bench._page_cache_written_values(
                paged_cache,
                paged_positions,
            ),
        )
        argmax_matches = int((paged_tokens == dense_tokens).sum().cpu())
        step_passed = (
            argmax_matches == args.batch_size
            and kv_delta["mean_abs"] < 1e-3
        )
        correctness_passed = correctness_passed and step_passed
        step_results.append(
            {
                "step": step,
                "positions": list(host_positions),
                "argmax_matches": argmax_matches,
                "argmax_total": args.batch_size,
                "logits": logits_delta,
                "written_kv": {
                    "max_abs": kv_delta["max_abs"],
                    "mean_abs": kv_delta["mean_abs"],
                },
                "passed": step_passed,
            }
        )
        dense_input.copy_(dense_tokens)
        paged_input.copy_(paged_tokens)
        dense_positions.add_(1)
        paged_positions.add_(1)
        for index in range(len(host_positions)):
            host_positions[index] += 1

    incre_timing = _timed_increfa(
        incre_runtime.fn,
        (dense_input, dense_positions, dense_rope),
        dense_cache.flat_tensors(),
        batch_size=args.batch_size,
        warmup=args.warmup_steps,
        steps=args.timing_steps,
    )
    paged_timing = _timed_paged(
        graph,
        graph_output,
        recorder,
        (paged_input, paged_positions),
        host_positions,
        batch_size=args.batch_size,
        warmup=args.warmup_steps,
        steps=args.timing_steps,
    )
    result = {
        "schema_version": 1,
        "kind": "npugraph_paged_fia_multistep_replay",
        "passed": correctness_passed,
        "configuration": {
            "batch_size": args.batch_size,
            "cache_length": args.cache_length,
            "block_size": args.block_size,
            "initial_positions": [
                int(value) for value in initial_positions.cpu().tolist()
            ],
            "correctness_steps": args.correctness_steps,
            "warmup_steps": args.warmup_steps,
            "timing_steps": args.timing_steps,
            "model_weights": "random",
            "architecture": "full_paddle_text_decoder",
            "graph": "torch.npu.NPUGraph",
            "fia_metadata_update": (
                "graph_task_update_begin_end_per_layer_per_replay"
            ),
            "paged_cache_layout": "PA_NZ",
            "paged_cache_update": "npu_scatter_pa_kv_cache",
            "paged_attention": "npu_fused_infer_attention_score_v2.out",
        },
        "versions": {
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
        },
        "captured_fia_tasks": len(recorder.tasks),
        "correctness": step_results,
        "timing": {
            "increfa": incre_timing,
            "npugraph_paged_fia": paged_timing,
            "fia_vs_increfa_raw_wall_speedup": (
                paged_timing["raw_wall_tokens_per_s"]
                / incre_timing["raw_wall_tokens_per_s"]
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if correctness_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
