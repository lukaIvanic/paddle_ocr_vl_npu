#!/usr/bin/env python3
"""Execute the one production-equivalent Linear visible to ``msprof op``.

The full compiled 27-layer graph is the source of truth for timing and every
Linear role. This target exists only for the square attention/output projection
shape whose eager launch was verified to match production's MatMulV2 dispatch,
dtype, ND/FRACTAL_NZ/ND formats, dimensions, and Block Dim. TorchAir's cached
graph hides its inner MatMul from ``msprof op``; FC1/FC2 are therefore not
represented by a misleading eager surrogate here.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Sequence

import torch
from torch.nn import functional as F


FRACTAL_NZ = 29
SEQUENCE_LENGTH = 2048
HIDDEN_SIZE = 1152


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("square",), required=True)
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

    import torch_npu

    torch.npu.config.allow_internal_format = True
    if not torch.npu.is_available():
        raise RuntimeError("the msprof vision Linear target requires an NPU")
    torch.npu.set_compile_mode(jit_compile=False)
    device = torch.device("npu:0")
    dtype = torch.float16

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    activation_cpu = (
        torch.randn(
            (SEQUENCE_LENGTH, HIDDEN_SIZE),
            generator=generator,
            dtype=dtype,
        )
        * 0.02
    )
    weight_cpu = (
        torch.randn(
            (HIDDEN_SIZE, HIDDEN_SIZE),
            generator=generator,
            dtype=dtype,
        )
        * 0.02
    )
    bias_cpu = (
        torch.randn(
            (HIDDEN_SIZE,),
            generator=generator,
            dtype=dtype,
        )
        * 0.02
    )

    activation = activation_cpu.to(device)
    weight = torch_npu.npu_format_cast(weight_cpu.to(device), FRACTAL_NZ)
    bias = bias_cpu.to(device)
    weight_format = int(torch_npu.get_npu_format(weight))
    if weight_format != FRACTAL_NZ:
        raise RuntimeError(
            "FRACTAL_NZ materialization failed: "
            f"expected {FRACTAL_NZ}, got {weight_format}"
        )

    # The first identical launch pays one-time eager dispatch/kernel setup.
    # msprof selects this MatMulV2 and applies its own replay warm-ups.
    warmup_result = F.linear(activation, weight, bias)
    torch.npu.synchronize()
    del warmup_result
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
    flops = 2 * SEQUENCE_LENGTH * HIDDEN_SIZE * HIDDEN_SIZE

    summary = {
        "schema_version": 2,
        "status": "completed",
        "purpose": (
            "production-equivalent square MatMulV2 target for deep msprof-op "
            "mechanics; not an end-to-end throughput measurement"
        ),
        "scope": {
            "supported_direct_roles": [
                "q_proj",
                "k_proj",
                "v_proj",
                "out_proj",
            ],
            "excluded_roles": {
                "fc1": (
                    "eager dispatch is MatMulV3 while production is MatMulV2; "
                    "compiled MatMulV2 is not visible to msprof op"
                ),
                "fc2": (
                    "compiled graph internals are not visible to msprof op; "
                    "no unvalidated eager surrogate is used"
                ),
            },
        },
        "spec": {
            "role": "square",
            "m": SEQUENCE_LENGTH,
            "k": HIDDEN_SIZE,
            "n": HIDDEN_SIZE,
            "production_roles": [
                "q_proj",
                "k_proj",
                "v_proj",
                "out_proj",
            ],
            "flops": flops,
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
            "identical_warmup_launches_before_measurement": 1,
            "device_event_ms": device_ms,
            "host_wall_ms": wall_ms,
            "kernel_local_tflop_per_s_estimate": (
                flops / (device_ms / 1000.0) / 1e12
            ),
        },
        "environment": {
            "hostname": platform.node(),
            "python": sys.version,
            "torch": torch.__version__,
            "torch_npu": getattr(torch_npu, "__version__", None),
            "device": torch.npu.get_device_name(0),
            "soc_version": str(torch_npu.npu.get_soc_version()),
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
