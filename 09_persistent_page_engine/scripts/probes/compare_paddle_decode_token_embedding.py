#!/usr/bin/env python3
"""Validate the specialized B1 decode embedding through TorchAir."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch.nn.functional as F

from paddleocr_vl.model.compile_utils import import_torchair
from paddleocr_vl.model.decode_token_embedding import (
    GE_OP_NAME,
    PYTORCH_OP_NAME,
    decode_token_embedding,
    register_decode_token_embedding_converter,
)


VOCAB_SIZE = 103424
HIDDEN_SIZE = 1024


class CustomEmbedding(torch.nn.Module):
    def forward(
        self, weight: torch.Tensor, input_ids: torch.Tensor
    ) -> torch.Tensor:
        return decode_token_embedding(weight, input_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--token-id", type=int, default=1024)
    args = parser.parse_args()
    if not 0 <= args.token_id < VOCAB_SIZE:
        parser.error(f"--token-id must be in [0, {VOCAB_SIZE})")
    if args.warmup < 0 or args.repeats <= 0:
        parser.error("--warmup must be nonnegative and --repeats positive")
    return args


def main() -> int:
    args = parse_args()
    import torch_npu

    if not torch.npu.is_available():
        raise RuntimeError("an Ascend NPU is required")
    torch.npu.set_compile_mode(jit_compile=False)
    register_decode_token_embedding_converter()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260810)
    weight = torch.randn(
        (VOCAB_SIZE, HIDDEN_SIZE), generator=generator, dtype=torch.float16
    ).to("npu:0")
    input_ids = torch.tensor([[args.token_id]], dtype=torch.int64, device="npu:0")
    reference = F.embedding(input_ids, weight)

    torchair, CompilerConfig = import_torchair()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    step = torchair.inference.cache_compile(
        CustomEmbedding().forward,
        config=CompilerConfig(),
        dynamic=False,
        cache_dir=str(args.cache_dir),
        ge_cache=True,
    )

    first_start = time.perf_counter()
    output = step(weight, input_ids)
    torch.npu.synchronize()
    first_call_s = time.perf_counter() - first_start
    for _ in range(args.warmup):
        output = step(weight, input_ids)
    torch.npu.synchronize()

    call_us: list[float] = []
    for _ in range(args.repeats):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        output = step(weight, input_ids)
        end.record()
        end.synchronize()
        call_us.append(float(start.elapsed_time(end)) * 1000.0)

    output_cpu = output.float().cpu()
    reference_cpu = reference.float().cpu()
    max_abs = float((output_cpu - reference_cpu).abs().max().item())
    exact = bool(torch.equal(output_cpu, reference_cpu))
    result = {
        "kind": "paddle_decode_token_embedding_torchair_probe",
        "operator": {"pytorch": PYTORCH_OP_NAME, "ge": GE_OP_NAME},
        "contract": {
            "weight_shape": list(weight.shape),
            "input_ids_shape": list(input_ids.shape),
            "output_shape": list(output.shape),
            "dtype": str(weight.dtype),
            "token_id": args.token_id,
        },
        "environment": {
            "device": torch.npu.get_device_name(0),
            "visible_device": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            "custom_opp_path": os.environ.get("ASCEND_CUSTOM_OPP_PATH"),
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
        },
        "correctness": {"exact": exact, "max_abs": max_abs},
        "timing": {
            "first_call_s": first_call_s,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "npu_event_us": {
                "mean": statistics.fmean(call_us),
                "median": statistics.median(call_us),
                "minimum": min(call_us),
                "maximum": max(call_us),
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not exact:
        raise RuntimeError(f"custom embedding mismatch: max_abs={max_abs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
