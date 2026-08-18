#!/usr/bin/env python3
"""Minimal 310P TorchAir probe for UniRec masked global context reduction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn


PROCESS_STARTED = time.perf_counter()
CHANNELS = 192
PHYSICAL_HEIGHT = 88
PHYSICAL_WIDTH = 128
VALID_HEIGHT = 80
VALID_WIDTH = 120


def phase(name: str, **fields: Any) -> None:
    print(
        "UNIREC_GLOBAL_CONTEXT_PHASE "
        + json.dumps(
            {
                "phase": name,
                "process_elapsed_s": time.perf_counter() - PROCESS_STARTED,
                **fields,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--require-cache", action="store_true")
    args = parser.parse_args()
    if args.timing_repeats < 1:
        parser.error("--timing-repeats must be positive")
    if not args.device.startswith("npu"):
        parser.error("this probe requires an NPU device")
    return args


class GlobalContextComparison(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.act = nn.GELU()

    def forward(
        self,
        ctx: torch.Tensor,
        valid_mask: torch.Tensor,
        accumulated_context: torch.Tensor,
        global_gate: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        masked = ctx * valid_mask

        # Current production formulation: mean across width, then mean across
        # valid rows. This is the first divergent boundary on 310P.
        row_counts = valid_mask.sum(dim=3, keepdim=True).clamp_min(1)
        row_means = masked.sum(dim=3, keepdim=True) / row_counts
        valid_rows = (valid_mask.sum(dim=3, keepdim=True) > 0).to(ctx.dtype)
        original = (row_means * valid_rows).sum(
            dim=2,
            keepdim=True,
        ) / valid_rows.sum(dim=2, keepdim=True).clamp_min(1)

        # Equivalent for UniRec rectangular crop masks: one masked sum and one
        # pixel count across both spatial dimensions.
        pixel_count = valid_mask.sum(dim=(2, 3), keepdim=True).clamp_min(1)
        direct_4d = masked.sum(dim=(2, 3), keepdim=True) / pixel_count

        # Keep the channel vector in unambiguous ND form. [N,C,1,1] is a known
        # ambiguous format family on Ascend; [N,C] cannot be mistaken for a 4D
        # NCHW/NHWC/internal-format activation.
        flat_pixel_count = valid_mask.sum(dim=(2, 3), keepdim=False).clamp_min(1)
        direct_2d = masked.sum(dim=(2, 3), keepdim=False) / flat_pixel_count

        original_broadcast = (
            accumulated_context + self.act(original) * global_gate
        )
        flat_broadcast = accumulated_context + (
            self.act(direct_2d).unsqueeze(-1).unsqueeze(-1) * global_gate
        )
        return original, direct_4d, direct_2d, original_broadcast, flat_broadcast


def difference(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    left = reference.float()
    right = candidate.float()
    delta = right - left
    count = delta.numel()
    denominator = (
        left.square().sum().sqrt() * right.square().sum().sqrt()
    ).clamp_min(1e-12)
    return {
        "shape": list(reference.shape),
        "exact": bool(torch.equal(reference, candidate)),
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().sum(dtype=torch.float64).item() / count),
        "rmse": float((delta.square().sum(dtype=torch.float64) / count).sqrt().item()),
        "cosine": float(((left * right).sum() / denominator).item()),
    }


def inventory(path: Path) -> dict[str, Any]:
    files = sorted(
        file
        for file in path.rglob("*")
        if file.is_file() and (file.name == "compiled_module" or file.suffix == ".om")
    )
    return {
        "compiled_module_count": sum(file.name == "compiled_module" for file in files),
        "om_count": sum(file.suffix == ".om" for file in files),
        "files": [
            {
                "path": str(file.relative_to(path)),
                "size": file.stat().st_size,
                "mtime_ns": file.stat().st_mtime_ns,
            }
            for file in files
        ],
    }


def timed(
    fn: Callable[[], tuple[torch.Tensor, ...]],
) -> tuple[tuple[torch.Tensor, ...], float]:
    torch.npu.synchronize()
    started = time.perf_counter()
    output = fn()
    torch.npu.synchronize()
    return output, (time.perf_counter() - started) * 1000.0


def main() -> None:
    args = parse_args()
    import torch_npu
    from torch_npu.dynamo.torchair.configs.compiler_config import CompilerConfig

    try:
        from torch_npu.dynamo.torchair.inference import cache_compile
    except ImportError:
        from torchair.inference import cache_compile

    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    if not visible:
        raise RuntimeError("set ASCEND_RT_VISIBLE_DEVICES before running")
    physical_devices = [int(value) for value in visible.split(",") if value.strip()]
    if any(value in {5, 6} for value in physical_devices):
        raise RuntimeError("physical NPU 5 and 6 are excluded")
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.manual_seed(args.seed)
    phase(
        "setup",
        physical_devices=physical_devices,
        npu_jit_compile=False,
    )

    module = GlobalContextComparison().to(args.device).eval()
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    host_ctx = torch.randn(
        (1, CHANNELS, PHYSICAL_HEIGHT, PHYSICAL_WIDTH),
        generator=generator,
        dtype=torch.float32,
    ).mul_(0.2)
    ctx = host_ctx.to(args.device, dtype=torch.float16)
    mask = torch.zeros(
        (1, 1, PHYSICAL_HEIGHT, PHYSICAL_WIDTH),
        dtype=torch.float16,
    )
    mask[:, :, :VALID_HEIGHT, :VALID_WIDTH] = 1
    mask = mask.to(args.device)
    ctx.mul_(mask)
    host_accumulated_context = torch.randn(
        (1, CHANNELS, PHYSICAL_HEIGHT, PHYSICAL_WIDTH),
        generator=generator,
        dtype=torch.float32,
    ).mul_(0.2)
    accumulated_context = host_accumulated_context.to(
        args.device,
        dtype=torch.float16,
    )
    accumulated_context.mul_(mask)
    host_global_gate = torch.randn(
        (1, 1, PHYSICAL_HEIGHT, PHYSICAL_WIDTH),
        generator=generator,
        dtype=torch.float32,
    ).mul_(0.2)
    global_gate = host_global_gate.to(args.device, dtype=torch.float16)
    global_gate.mul_(mask)
    phase(
        "inputs_ready",
        ctx_shape=list(ctx.shape),
        mask_shape=list(mask.shape),
        valid_shape=[VALID_HEIGHT, VALID_WIDTH],
    )

    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    cache_dir = args.cache_root.expanduser().resolve() / (
        f"masked_global_context_c192_h88_w128_fp16_src{source_hash}"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    before = inventory(cache_dir)
    if args.require_cache and not before["compiled_module_count"]:
        raise RuntimeError(f"required cache is absent: {cache_dir}")
    config = CompilerConfig()
    config.mode.value = "max-autotune"
    compiled = cache_compile(
        module.forward,
        config=config,
        dynamic=False,
        cache_dir=str(cache_dir),
        ge_cache=True,
        fullgraph=True,
    )
    phase("graph_registered", cache_before=before, cache_dir=str(cache_dir))

    call = lambda fn: fn(ctx, mask, accumulated_context, global_gate)
    with torch.inference_mode():
        eager_first, eager_first_ms = timed(lambda: call(module))
        phase("eager_first", synchronized_ms=eager_first_ms)
        phase("compiled_first_begin", cache_dir=str(cache_dir))
        compiled_first, compiled_first_ms = timed(lambda: call(compiled))
        phase("compiled_first", synchronized_ms=compiled_first_ms)
        eager_times = []
        compiled_times = []
        eager_output = eager_first
        compiled_output = compiled_first
        for repeat in range(args.timing_repeats):
            ordered = (("eager", module), ("compiled", compiled))
            if repeat % 2:
                ordered = tuple(reversed(ordered))
            for name, fn in ordered:
                output, elapsed_ms = timed(lambda fn=fn: call(fn))
                if name == "eager":
                    eager_output = output
                    eager_times.append(elapsed_ms)
                else:
                    compiled_output = output
                    compiled_times.append(elapsed_ms)

    (
        eager_original,
        eager_direct_4d,
        eager_direct_2d,
        eager_original_broadcast,
        eager_flat_broadcast,
    ) = eager_output
    (
        compiled_original,
        compiled_direct_4d,
        compiled_direct_2d,
        compiled_original_broadcast,
        compiled_flat_broadcast,
    ) = compiled_output
    comparisons = {
        "original_eager_vs_compiled": difference(eager_original, compiled_original),
        "direct_4d_eager_vs_compiled": difference(
            eager_direct_4d,
            compiled_direct_4d,
        ),
        "direct_2d_eager_vs_compiled": difference(
            eager_direct_2d,
            compiled_direct_2d,
        ),
        "original_broadcast_eager_vs_compiled": difference(
            eager_original_broadcast,
            compiled_original_broadcast,
        ),
        "flat_broadcast_eager_vs_compiled": difference(
            eager_flat_broadcast,
            compiled_flat_broadcast,
        ),
        "eager_original_4d_vs_direct_4d": difference(
            eager_original,
            eager_direct_4d,
        ),
        "eager_original_4d_vs_direct_2d": difference(
            eager_original.flatten(1),
            eager_direct_2d,
        ),
        "compiled_original_4d_vs_direct_2d": difference(
            compiled_original.flatten(1),
            compiled_direct_2d,
        ),
        "eager_original_vs_flat_broadcast": difference(
            eager_original_broadcast,
            eager_flat_broadcast,
        ),
        "compiled_original_vs_flat_broadcast": difference(
            compiled_original_broadcast,
            compiled_flat_broadcast,
        ),
    }
    after = inventory(cache_dir)
    report = {
        "schema": "unirec_masked_global_context_probe_v1",
        "status": "pass",
        "physical_devices": physical_devices,
        "ctx_shape": list(ctx.shape),
        "mask_shape": list(mask.shape),
        "valid_shape": [VALID_HEIGHT, VALID_WIDTH],
        "npu_jit_compile": False,
        "first_call_ms": {
            "eager": eager_first_ms,
            "compiled": compiled_first_ms,
        },
        "steady_p50_ms": {
            "eager": statistics.median(eager_times),
            "compiled": statistics.median(compiled_times),
        },
        "comparisons": comparisons,
        "cache_dir": str(cache_dir),
        "cache_before": before,
        "cache_after": after,
        "cache_changed": before != after,
        "process_wall_s": time.perf_counter() - PROCESS_STARTED,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, row in comparisons.items():
        print(
            "UNIREC_GLOBAL_CONTEXT_COMPARISON "
            f"name={name} max_abs={row['max_abs']:.9g} "
            f"rmse={row['rmse']:.9g} cosine={row['cosine']:.9g}",
            flush=True,
        )
    print("UNIREC_GLOBAL_CONTEXT_RESULT " + json.dumps(report), flush=True)
    print(f"OUTPUT_JSON={output}", flush=True)


if __name__ == "__main__":
    main()
