#!/usr/bin/env python3
"""Execute one production-shaped FRACTAL_NZ vision Linear for ``msprof op``.

This is intentionally not a throughput benchmark.  It gives ``msprof op`` one
ordinary MatMulV2 launch with the exact dtype, activation format, weight
format, and dimensions used by the optimized B1xS2048 production graph.
The full 27-layer graph remains the source of truth for end-to-end timing.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch.nn import functional as F


FRACTAL_NZ = 29
SEQUENCE_LENGTH = 2048


@dataclass(frozen=True)
class LinearSpec:
    role: str
    m: int
    k: int
    n: int
    production_roles: tuple[str, ...]

    @property
    def flops(self) -> int:
        return 2 * self.m * self.k * self.n


SPECS = {
    "square": LinearSpec(
        role="square",
        m=SEQUENCE_LENGTH,
        k=1152,
        n=1152,
        production_roles=("q_proj", "k_proj", "v_proj", "out_proj"),
    ),
    "fc1": LinearSpec(
        role="fc1",
        m=SEQUENCE_LENGTH,
        k=1152,
        n=4352,
        production_roles=("fc1",),
    ),
    "fc2": LinearSpec(
        role="fc2",
        m=SEQUENCE_LENGTH,
        k=4352,
        n=1152,
        production_roles=("fc2",),
    ),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=tuple(SPECS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260730)
    return parser.parse_args(argv)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "target_summary.json"
    spec = SPECS[args.role]

    import torch_npu

    # This gate must be enabled before the first NPU allocation.
    torch.npu.config.allow_internal_format = True
    if not torch.npu.is_available():
        raise RuntimeError("the msprof vision Linear target requires an NPU")
    torch.npu.set_compile_mode(jit_compile=False)
    device = torch.device("npu:0")
    dtype = torch.float16

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    activation_cpu = (
        torch.randn((spec.m, spec.k), generator=generator, dtype=dtype) * 0.02
    )
    weight_cpu = (
        torch.randn((spec.n, spec.k), generator=generator, dtype=dtype) * 0.02
    )
    bias_cpu = torch.randn((spec.n,), generator=generator, dtype=dtype) * 0.02

    activation = activation_cpu.to(device)
    weight = torch_npu.npu_format_cast(weight_cpu.to(device), FRACTAL_NZ)
    bias = bias_cpu.to(device)
    weight_format = int(torch_npu.get_npu_format(weight))
    if weight_format != FRACTAL_NZ:
        raise RuntimeError(
            "FRACTAL_NZ materialization failed: "
            f"expected {FRACTAL_NZ}, got {weight_format}"
        )

    torch.npu.synchronize()
    start_event = torch_npu.npu.Event(enable_timing=True)
    end_event = torch_npu.npu.Event(enable_timing=True)
    wall_started = time.perf_counter()
    start_event.record()
    result = F.linear(activation, weight, bias)
    end_event.record()
    torch.npu.synchronize()
    wall_ms = (time.perf_counter() - wall_started) * 1000.0
    device_ms = float(start_event.elapsed_time(end_event))
    result_cpu = result.cpu()

    summary = {
        "schema_version": 1,
        "status": "completed",
        "purpose": (
            "single production-shaped MatMulV2 target for deep msprof-op "
            "mechanics; not an end-to-end throughput measurement"
        ),
        "spec": {
            **asdict(spec),
            "flops": spec.flops,
            "dtype": str(dtype),
            "activation_format_expected": "ND",
            "weight_format_expected": "FRACTAL_NZ",
            "weight_format_code_expected": FRACTAL_NZ,
            "bias": True,
        },
        "observed": {
            "activation_format_code": int(
                torch_npu.get_npu_format(activation)
            ),
            "weight_format_code": weight_format,
            "output_format_code": int(torch_npu.get_npu_format(result)),
            "output_shape": list(result.shape),
            "output_dtype": str(result.dtype),
            "output_finite": bool(torch.isfinite(result_cpu.float()).all()),
            "device_event_ms": device_ms,
            "host_wall_ms": wall_ms,
            "kernel_local_tflop_per_s_estimate": (
                spec.flops / (device_ms / 1000.0) / 1e12
            ),
        },
        "environment": {
            "hostname": platform.node(),
            "python": sys.version,
            "torch": torch.__version__,
            "torch_npu": getattr(torch_npu, "__version__", None),
            "device": torch.npu.get_device_name(0),
            "ascend_rt_visible_devices": os.environ.get(
                "ASCEND_RT_VISIBLE_DEVICES"
            ),
            "cann_root": os.environ.get("ASCEND_HOME_PATH"),
        },
    }
    _write_json(summary_path, summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
