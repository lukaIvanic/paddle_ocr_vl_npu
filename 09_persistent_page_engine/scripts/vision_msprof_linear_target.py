#!/usr/bin/env python3
"""Execute one production-shaped compiled vision Linear for ``msprof op``.

This is intentionally not a throughput benchmark. It gives ``msprof op`` one
TorchAir-compiled MatMulV2 with the exact dtype, activation format, weight
format, bias, and dimensions used by the optimized B1xS2048 production graph.
The full 27-layer graph remains the source of truth for end-to-end timing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import nn


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

from paddleocr_vl.model.compile_utils import import_torchair  # noqa: E402


FRACTAL_NZ = 29
BATCH_SIZE = 1
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
        m=BATCH_SIZE * SEQUENCE_LENGTH,
        k=1152,
        n=1152,
        production_roles=("q_proj", "k_proj", "v_proj", "out_proj"),
    ),
    "fc1": LinearSpec(
        role="fc1",
        m=BATCH_SIZE * SEQUENCE_LENGTH,
        k=1152,
        n=4352,
        production_roles=("fc1",),
    ),
    "fc2": LinearSpec(
        role="fc2",
        m=BATCH_SIZE * SEQUENCE_LENGTH,
        k=4352,
        n=1152,
        production_roles=("fc2",),
    ),
}


class TorchAirLinearTarget(nn.Module):
    """One bound ``nn.Linear`` matching the production compiler boundary."""

    def __init__(
        self,
        spec: LinearSpec,
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator,
    ) -> None:
        super().__init__()
        weight_cpu = (
            torch.randn(
                (spec.n, spec.k),
                generator=generator,
                dtype=dtype,
            )
            * 0.02
        )
        bias_cpu = (
            torch.randn((spec.n,), generator=generator, dtype=dtype) * 0.02
        )
        self.linear = nn.Linear(
            spec.k,
            spec.n,
            bias=True,
            device="cpu",
            dtype=dtype,
        )
        with torch.no_grad():
            self.linear.weight.copy_(weight_cpu)
            self.linear.bias.copy_(bias_cpu)
        self.linear = self.linear.to(device)
        import torch_npu

        self.linear.weight.data = torch_npu.npu_format_cast(
            self.linear.weight.data,
            FRACTAL_NZ,
        )
        self.eval()
        self.requires_grad_(False)

    def forward(self, activation: torch.Tensor) -> torch.Tensor:
        return self.linear(activation)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=tuple(SPECS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--prepare-cache-only",
        action="store_true",
        help="Compile/load the graph once, validate it, and exit.",
    )
    parser.add_argument(
        "--allow-compile-if-missing",
        action="store_true",
        help="Permit creation of a missing TorchAir graph cache.",
    )
    return parser.parse_args(argv)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _cache_populated(path: Path) -> bool:
    return path.is_dir() and any(item.is_file() for item in path.rglob("*"))


def _cache_dir(
    root: Path,
    spec: LinearSpec,
    *,
    torch_npu: object,
    torchair: object,
) -> Path:
    soc = str(torch_npu.npu.get_soc_version())
    identity = {
        "role": spec.role,
        "batch_size": BATCH_SIZE,
        "sequence_length": SEQUENCE_LENGTH,
        "k": spec.k,
        "n": spec.n,
        "dtype": "float16",
        "activation_format": "ND",
        "weight_format": "FRACTAL_NZ",
        "bias": True,
        "torch": torch.__version__,
        "torch_npu": getattr(torch_npu, "__version__", "unknown"),
        "torchair": getattr(torchair, "__version__", "unknown"),
        "source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    safe_soc = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in soc
    )
    return (
        root.expanduser().resolve()
        / safe_soc
        / (
            f"linear_b{BATCH_SIZE}_s{SEQUENCE_LENGTH}_"
            f"k{spec.k}_n{spec.n}_fp16_nd_nz_bias_{digest}"
        )
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
    torchair, CompilerConfig = import_torchair()
    cache_dir = _cache_dir(
        args.cache_root,
        spec,
        torch_npu=torch_npu,
        torchair=torchair,
    )

    cache_existed_before = _cache_populated(cache_dir)
    if not cache_existed_before and not args.allow_compile_if_missing:
        raise RuntimeError(
            "the exact TorchAir target cache is missing; prepare it outside "
            f"msprof first: {cache_dir}"
        )
    cache_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    activation_cpu = (
        torch.randn(
            (BATCH_SIZE, SEQUENCE_LENGTH, spec.k),
            generator=generator,
            dtype=dtype,
        )
        * 0.02
    )
    target = TorchAirLinearTarget(
        spec,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    activation = activation_cpu.to(device)
    weight_format = int(torch_npu.get_npu_format(target.linear.weight))
    if weight_format != FRACTAL_NZ:
        raise RuntimeError(
            "FRACTAL_NZ materialization failed: "
            f"expected {FRACTAL_NZ}, got {weight_format}"
        )

    compiled = torchair.inference.cache_compile(
        target.forward,
        config=CompilerConfig(),
        dynamic=False,
        cache_dir=str(cache_dir),
        ge_cache=True,
    )

    first_started = time.perf_counter()
    first_result = compiled(activation)
    torch.npu.synchronize()
    first_call_ms = (time.perf_counter() - first_started) * 1000.0
    first_result_cpu = first_result.cpu()
    if not _cache_populated(cache_dir):
        raise RuntimeError(f"TorchAir did not populate cache: {cache_dir}")
    if not bool(torch.isfinite(first_result_cpu.float()).all()):
        raise RuntimeError("compiled target produced non-finite output")

    observed: dict[str, object] = {
        "activation_format_code": int(
            torch_npu.get_npu_format(activation)
        ),
        "weight_format_code": weight_format,
        "output_format_code": int(torch_npu.get_npu_format(first_result)),
        "output_shape": list(first_result.shape),
        "output_dtype": str(first_result.dtype),
        "output_finite": True,
        "first_compiled_call_ms": first_call_ms,
    }
    status = "cache_ready"
    if not args.prepare_cache_only:
        start_event = torch_npu.npu.Event(enable_timing=True)
        end_event = torch_npu.npu.Event(enable_timing=True)
        wall_started = time.perf_counter()
        start_event.record()
        result = compiled(activation)
        end_event.record()
        torch.npu.synchronize()
        wall_ms = (time.perf_counter() - wall_started) * 1000.0
        device_ms = float(start_event.elapsed_time(end_event))
        result_cpu = result.cpu()
        observed.update(
            {
                "output_format_code": int(
                    torch_npu.get_npu_format(result)
                ),
                "output_shape": list(result.shape),
                "output_dtype": str(result.dtype),
                "output_finite": bool(
                    torch.isfinite(result_cpu.float()).all()
                ),
                "compiled_launches_before_measurement": 1,
                "msprof_selected_launch": (
                    "first compiled MatMulV2; same graph and tensors as "
                    "the event-timed launch"
                ),
                "device_event_ms": device_ms,
                "host_wall_ms": wall_ms,
                "kernel_local_tflop_per_s_estimate": (
                    spec.flops / (device_ms / 1000.0) / 1e12
                ),
            }
        )
        status = "completed"

    summary = {
        "schema_version": 2,
        "status": status,
        "purpose": (
            "single production-shaped TorchAir MatMulV2 target for deep "
            "msprof-op mechanics; not an end-to-end throughput measurement"
        ),
        "execution": {
            "api": "torchair.inference.cache_compile",
            "dynamic": False,
            "ge_cache": True,
            "cache_dir": str(cache_dir),
            "cache_existed_before": cache_existed_before,
            "compile_was_permitted": bool(args.allow_compile_if_missing),
            "prepare_cache_only": bool(args.prepare_cache_only),
        },
        "spec": {
            **asdict(spec),
            "batch_size": BATCH_SIZE,
            "sequence_length": SEQUENCE_LENGTH,
            "input_shape": [BATCH_SIZE, SEQUENCE_LENGTH, spec.k],
            "flops": spec.flops,
            "dtype": str(dtype),
            "activation_format_expected": "ND",
            "weight_format_expected": "FRACTAL_NZ",
            "weight_format_code_expected": FRACTAL_NZ,
            "bias": True,
        },
        "observed": observed,
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
