#!/usr/bin/env python3
"""Compare serial and stream-pipelined embedding inference on Ascend NPU.

The workload accepts token IDs shaped [batch, sequence] and returns token
embeddings shaped [batch, sequence, hidden].  It is deliberately independent
of PaddleOCR so the transfer pattern can be copied into another project.

The serial lane performs H2D, model compute, D2H, and a host wait for every
batch.  The pipelined lane uses pinned host buffers, separate H2D/compute/D2H
streams, events, and a small ring so that adjacent independent batches may
overlap.  Stream submission expresses the dependency graph; whether transfers
actually overlap compute is still hardware/runtime dependent.

Example:

    source npu-setup

    # Small smoke test.  Stream overhead may be larger than any saved time.
    python 09_persistent_page_engine/scripts/demo_embedding_stream_pipeline.py \
        --seq-len 512 --hidden-dim 1024 --compute-layers 1

    # Large-output, roughly 10 ms-class demonstration on a 910B2: each result
    # is 64 MiB in FP16, making D2H large enough to expose possible overlap.
    python 09_persistent_page_engine/scripts/demo_embedding_stream_pipeline.py \
        --seq-len 8192 --hidden-dim 4096 --vocab-size 8192 \
        --compute-layers 8 --batches 16 --warmup-batches 3
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F


class FakeEmbeddingEncoder(torch.nn.Module):
    """A small stand-in for a token-level embedding/encoder service."""

    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_dim: int,
        compute_layers: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_dim, dtype=dtype)
        self.projections = torch.nn.ModuleList(
            torch.nn.Linear(hidden_dim, hidden_dim, bias=False, dtype=dtype)
            for _ in range(compute_layers)
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(token_ids)
        for projection in self.projections:
            hidden = F.silu(projection(hidden))
        return hidden


@dataclass
class PipelineSlot:
    device_input: torch.Tensor
    host_output: torch.Tensor
    device_output: torch.Tensor | None = None
    d2h_done: object | None = None


@dataclass(frozen=True)
class ComponentTimes:
    h2d_us: float
    compute_us: float
    d2h_us: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument(
        "--compute-layers",
        type=int,
        default=1,
        help="Extra hidden-to-hidden projections after the embedding lookup; use 0 for lookup only.",
    )
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--warmup-batches", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--ring-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "batch_size",
        "seq_len",
        "hidden_dim",
        "vocab_size",
        "batches",
        "warmup_batches",
        "rounds",
    )
    for name in positive:
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.compute_layers < 0:
        raise ValueError("--compute-layers must be non-negative")
    if args.ring_size < 2:
        raise ValueError("--ring-size must be at least 2")


def make_host_inputs(args: argparse.Namespace) -> torch.Tensor:
    inputs = torch.empty(
        (args.batches, args.batch_size, args.seq_len),
        dtype=torch.int64,
        pin_memory=True,
    )
    inputs.random_(0, args.vocab_size)
    return inputs


@torch.inference_mode()
def measure_components(
    *,
    model: torch.nn.Module,
    host_input: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    iterations: int,
) -> ComponentTimes:
    """Measure isolated device-stream time for each pipeline stage."""
    h2d_stream = torch_npu.npu.Stream(device=device)
    compute_stream = torch_npu.npu.current_stream()
    d2h_stream = torch_npu.npu.Stream(device=device)
    device_input = torch.empty_like(host_input, device=device)
    host_output = torch.empty(
        (*host_input.shape, model.embedding.embedding_dim),
        dtype=dtype,
        pin_memory=True,
    )

    device_input.copy_(host_input)
    device_output = model(device_input)
    host_output.copy_(device_output)
    torch_npu.npu.synchronize()

    with torch_npu.npu.stream(h2d_stream):
        start = torch_npu.npu.Event(enable_timing=True)
        end = torch_npu.npu.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            device_input.copy_(host_input, non_blocking=True)
        end.record()
    end.synchronize()
    h2d_us = float(start.elapsed_time(end)) * 1000.0 / iterations

    with torch_npu.npu.stream(compute_stream):
        start = torch_npu.npu.Event(enable_timing=True)
        end = torch_npu.npu.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            device_output = model(device_input)
        end.record()
    end.synchronize()
    compute_us = float(start.elapsed_time(end)) * 1000.0 / iterations

    with torch_npu.npu.stream(d2h_stream):
        start = torch_npu.npu.Event(enable_timing=True)
        end = torch_npu.npu.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            host_output.copy_(device_output, non_blocking=True)
        end.record()
    device_output.record_stream(d2h_stream)
    end.synchronize()
    d2h_us = float(start.elapsed_time(end)) * 1000.0 / iterations
    return ComponentTimes(h2d_us=h2d_us, compute_us=compute_us, d2h_us=d2h_us)


@torch.inference_mode()
def run_serial(
    *,
    model: torch.nn.Module,
    host_inputs: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    """Run one strictly serialized H2D -> compute -> D2H schedule."""
    device_input = torch.empty_like(host_inputs[0], device=device)
    host_output = torch.empty(
        (*host_inputs.shape[1:], model.embedding.embedding_dim),
        dtype=dtype,
        pin_memory=True,
    )
    compute_stream = torch_npu.npu.current_stream()
    d2h_done = torch_npu.npu.Event()
    torch_npu.npu.synchronize()
    started = time.perf_counter()
    for host_input in host_inputs:
        with torch_npu.npu.stream(compute_stream):
            device_input.copy_(host_input, non_blocking=True)
            device_output = model(device_input)
            host_output.copy_(device_output, non_blocking=True)
            d2h_done.record(compute_stream)
        # This is intentionally strict.  The next loop iteration cannot submit
        # any work until the complete output is in pinned CPU memory and has
        # been observed by the host.
        d2h_done.synchronize()
        _ = host_output.reshape(-1)[0].item()
    return time.perf_counter() - started


@torch.inference_mode()
def run_pipelined(
    *,
    model: torch.nn.Module,
    host_inputs: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    ring_size: int,
) -> float:
    """Pipeline independent batches across H2D, compute, and D2H streams."""
    h2d_stream = torch_npu.npu.Stream(device=device)
    compute_stream = torch_npu.npu.current_stream()
    d2h_stream = torch_npu.npu.Stream(device=device)
    output_shape = (*host_inputs.shape[1:], model.embedding.embedding_dim)
    slots = [
        PipelineSlot(
            device_input=torch.empty_like(host_inputs[0], device=device),
            host_output=torch.empty(output_shape, dtype=dtype, pin_memory=True),
        )
        for _ in range(ring_size)
    ]

    torch_npu.npu.synchronize()
    started = time.perf_counter()
    for batch_index, host_input in enumerate(host_inputs):
        slot = slots[batch_index % ring_size]

        # A completed D2H event transitively proves that this slot's previous
        # H2D and compute work are also complete, so every buffer is reusable.
        if slot.d2h_done is not None:
            slot.d2h_done.synchronize()
            _ = slot.host_output.reshape(-1)[0].item()
            slot.device_output = None

        with torch_npu.npu.stream(h2d_stream):
            slot.device_input.copy_(host_input, non_blocking=True)
            h2d_done = h2d_stream.record_event()

        with torch_npu.npu.stream(compute_stream):
            compute_stream.wait_event(h2d_done)
            slot.device_output = model(slot.device_input)
            compute_done = compute_stream.record_event()

        with torch_npu.npu.stream(d2h_stream):
            d2h_stream.wait_event(compute_done)
            slot.host_output.copy_(slot.device_output, non_blocking=True)
            slot.d2h_done = d2h_stream.record_event()

        # The output tensor was allocated on the compute stream and is consumed
        # by the D2H stream.  Retain it in the slot and tell the allocator about
        # the side-stream use until the D2H event completes.
        slot.device_output.record_stream(d2h_stream)

    for slot in slots:
        if slot.d2h_done is not None:
            slot.d2h_done.synchronize()
            _ = slot.host_output.reshape(-1)[0].item()
    return time.perf_counter() - started


@torch.inference_mode()
def validate_pipeline(
    *,
    model: torch.nn.Module,
    host_input: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Check one event-linked transfer against the ordinary result."""
    reference = model(host_input.to(device)).cpu()

    h2d_stream = torch_npu.npu.Stream(device=device)
    compute_stream = torch_npu.npu.current_stream()
    d2h_stream = torch_npu.npu.Stream(device=device)
    device_input = torch.empty_like(host_input, device=device)
    host_output = torch.empty(
        (*host_input.shape, model.embedding.embedding_dim),
        dtype=dtype,
        pin_memory=True,
    )

    with torch_npu.npu.stream(h2d_stream):
        device_input.copy_(host_input, non_blocking=True)
        h2d_done = h2d_stream.record_event()
    with torch_npu.npu.stream(compute_stream):
        compute_stream.wait_event(h2d_done)
        device_output = model(device_input)
        compute_done = compute_stream.record_event()
    with torch_npu.npu.stream(d2h_stream):
        d2h_stream.wait_event(compute_done)
        host_output.copy_(device_output, non_blocking=True)
        done = d2h_stream.record_event()
    device_output.record_stream(d2h_stream)
    done.synchronize()

    torch.testing.assert_close(host_output, reference, rtol=0.0, atol=0.0)


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def main() -> None:
    global torch_npu
    args = parse_args()
    validate_args(args)

    try:
        import torch_npu as imported_torch_npu
    except ImportError as exc:
        raise RuntimeError("torch_npu is required for this Ascend demonstration") from exc
    torch_npu = imported_torch_npu

    if not torch.npu.is_available():
        raise RuntimeError("an Ascend NPU is required; CPU fallback is intentionally disabled")

    device = torch.device(args.device)
    torch.npu.set_device(device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    torch.manual_seed(args.seed)

    model = FakeEmbeddingEncoder(
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_dim,
        compute_layers=args.compute_layers,
        dtype=dtype,
    ).to(device).eval()
    host_inputs = make_host_inputs(args)

    validate_pipeline(
        model=model,
        host_input=host_inputs[0],
        device=device,
        dtype=dtype,
    )

    warmup_inputs = host_inputs[: min(args.warmup_batches, args.batches)]
    run_serial(
        model=model,
        host_inputs=warmup_inputs,
        device=device,
        dtype=dtype,
    )
    run_pipelined(
        model=model,
        host_inputs=warmup_inputs,
        device=device,
        dtype=dtype,
        ring_size=args.ring_size,
    )
    components = measure_components(
        model=model,
        host_input=host_inputs[0],
        device=device,
        dtype=dtype,
        iterations=args.batches,
    )

    results: dict[str, list[float]] = {"serial": [], "pipelined": []}
    for round_index in range(args.rounds):
        order = (
            ("serial", "pipelined")
            if round_index % 2 == 0
            else ("pipelined", "serial")
        )
        for lane in order:
            if lane == "serial":
                elapsed = run_serial(
                    model=model,
                    host_inputs=host_inputs,
                    device=device,
                    dtype=dtype,
                )
            else:
                elapsed = run_pipelined(
                    model=model,
                    host_inputs=host_inputs,
                    device=device,
                    dtype=dtype,
                    ring_size=args.ring_size,
                )
            results[lane].append(elapsed)
            print(
                f"round={round_index + 1} lane={lane} "
                f"wall_s={elapsed:.6f} ms_per_batch={elapsed * 1e3 / args.batches:.3f}",
                flush=True,
            )

    serial_s = median(results["serial"])
    pipelined_s = median(results["pipelined"])
    speedup = serial_s / pipelined_s
    output_bytes_per_batch = (
        args.batch_size
        * args.seq_len
        * args.hidden_dim
        * torch.empty((), dtype=dtype).element_size()
    )
    total_output_gib = output_bytes_per_batch * args.batches / (1024**3)
    output_mib_per_batch = output_bytes_per_batch / (1024**2)

    print("\nEMBEDDING_STREAM_PIPELINE_RESULT")
    print("scope: synthetic_schedule_demonstration_not_a_310p_performance_claim")
    print(
        "shape: "
        f"input=[{args.batch_size},{args.seq_len}] int64 "
        f"output=[{args.batch_size},{args.seq_len},{args.hidden_dim}] {args.dtype}"
    )
    print(
        f"workload: batches={args.batches} compute_layers={args.compute_layers} "
        f"ring_size={args.ring_size} rounds={args.rounds} "
        f"output_MiB_per_batch={output_mib_per_batch:.3f}"
    )
    print(
        "correctness: one_batch_exact_match=true "
        "naive_waits_for_each_complete_d2h=true"
    )
    component_sum_us = components.h2d_us + components.compute_us + components.d2h_us
    ideal_pipeline_floor_us = max(
        components.h2d_us,
        components.compute_us,
        components.d2h_us,
    )
    print(
        "isolated_device_stage_us_per_batch: "
        f"h2d={components.h2d_us:.3f} "
        f"compute={components.compute_us:.3f} "
        f"d2h={components.d2h_us:.3f}"
    )
    print(
        f"isolated_stage_sum_us={component_sum_us:.3f} "
        f"ideal_full_overlap_floor_us={ideal_pipeline_floor_us:.3f} "
        f"idealized_max_speedup={component_sum_us / ideal_pipeline_floor_us:.3f}x"
    )
    print(
        f"serial: wall_s={serial_s:.6f} "
        f"ms_per_batch={serial_s * 1e3 / args.batches:.3f} "
        f"request_qps={args.batches * args.batch_size / serial_s:.3f} "
        f"output_GiB_per_s={total_output_gib / serial_s:.3f}"
    )
    print(
        f"pipelined: wall_s={pipelined_s:.6f} "
        f"ms_per_batch={pipelined_s * 1e3 / args.batches:.3f} "
        f"request_qps={args.batches * args.batch_size / pipelined_s:.3f} "
        f"output_GiB_per_s={total_output_gib / pipelined_s:.3f}"
    )
    print(
        f"schedule_speedup={speedup:.3f}x "
        f"wall_reduction_pct={(1.0 - pipelined_s / serial_s) * 100.0:.2f}"
    )
    if speedup > 1.03:
        print("interpretation: this run shows a measurable benefit from pipelining")
    else:
        print(
            "interpretation: this run does not show a clear overlap benefit; "
            "inspect an NPU timeline before attributing the result"
        )


if __name__ == "__main__":
    main()
