#!/usr/bin/env python3
"""Inspect and, when available, smoke the installed MSDA Python wrapper."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
import traceback
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    import torch_npu

    exposed = sorted(
        name
        for name in dir(torch_npu)
        if "deform" in name.lower() or "grid" in name.lower()
    )
    npu_namespace = sorted(
        name
        for name in dir(torch.ops.npu)
        if "deform" in name.lower() or "grid" in name.lower()
    )
    result: dict[str, object] = {
        "format": "unirec_layout_msda_runtime_preflight_v1",
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "torch_npu_exposed_names": exposed,
        "torch_ops_npu_names": npu_namespace,
        "mx_driving_installed": importlib.util.find_spec("mx_driving") is not None,
        "production_shape_contract": {
            "value": [1, 13125, 8, 32],
            "spatial_shapes": [[100, 100], [50, 50], [25, 25]],
            "level_start_index": [0, 10000, 12500],
            "sampling_locations": [1, 300, 8, 3, 4, 2],
            "attention_weights": [1, 300, 8, 3, 4],
            "output": [1, 300, 256],
        },
    }

    if not result["mx_driving_installed"]:
        result["status"] = "no_python_wrapper"
    else:
        try:
            import mx_driving

            operation = getattr(mx_driving, "multi_scale_deformable_attn")
            torch.manual_seed(20260816)
            device = torch.device("npu:0")
            dtype = torch.float16
            spatial_shapes = torch.tensor(
                [[100, 100], [50, 50], [25, 25]],
                dtype=torch.int32,
                device=device,
            )
            level_start_index = torch.tensor(
                [0, 10000, 12500], dtype=torch.int32, device=device
            )
            value = torch.randn(
                (1, 13125, 8, 32), dtype=dtype, device=device
            )
            sampling_locations = torch.rand(
                (1, 300, 8, 3, 4, 2), dtype=dtype, device=device
            )
            attention_weights = torch.rand(
                (1, 300, 8, 3, 4), dtype=dtype, device=device
            )
            attention_weights = torch.softmax(
                attention_weights.flatten(-2), dim=-1
            ).view(1, 300, 8, 3, 4)
            torch.npu.synchronize()
            started = time.perf_counter()
            output = operation(
                value,
                spatial_shapes,
                level_start_index,
                sampling_locations,
                attention_weights,
            )
            torch.npu.synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000
            result.update(
                {
                    "status": "pass",
                    "elapsed_ms_first_call": elapsed_ms,
                    "output_shape": list(output.shape),
                    "output_dtype": str(output.dtype),
                    "output_all_finite": bool(torch.isfinite(output).all().item()),
                    "output_abs_mean": float(output.abs().mean().item()),
                }
            )
        except Exception as exc:  # The exact unsupported-op error is evidence.
            result.update(
                {
                    "status": "runtime_call_failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        "UNIREC_LAYOUT_MSDA_RUNTIME "
        f"status={result['status']} "
        f"mx_driving={str(result['mx_driving_installed']).lower()} "
        f"output={args.output.resolve()}",
        flush=True,
    )
    if result["status"] == "runtime_call_failed":
        print(
            "UNIREC_LAYOUT_MSDA_RUNTIME_ERROR "
            f"type={result['error_type']} error={result['error']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
