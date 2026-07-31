#!/usr/bin/env python3
"""Standalone 310P IncreFA masked-GQA discriminator.

This probe deliberately imports no project modules and reads no model files.
It runs exactly one eager ``torch_npu.npu_incre_flash_attention`` call followed
by one device synchronization.  Run each lane in a separate process under an
external shell timeout because a failing 310P kernel may never complete.

The PaddleOCR-VL decoder geometry is fixed here: 16 query heads, 2 KV heads,
and head dimension 128.  The ``mha_masked`` control instead stores 16 KV heads
and passes ``num_key_value_heads=0``, the documented MHA representation for
Atlas inference-series accelerator cards.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path
from typing import Any, Sequence

import torch


QUERY_HEADS = 16
GQA_KV_HEADS = 2
HEAD_DIM = 128
SCALE_VALUE = 1.0 / math.sqrt(HEAD_DIM)

LANES = {
    "gqa_masked": {
        "stored_kv_heads": GQA_KV_HEADS,
        "op_num_key_value_heads": GQA_KV_HEADS,
        "masked": True,
    },
    "gqa_nomask": {
        "stored_kv_heads": GQA_KV_HEADS,
        "op_num_key_value_heads": GQA_KV_HEADS,
        "masked": False,
    },
    "mha_masked": {
        "stored_kv_heads": QUERY_HEADS,
        "op_num_key_value_heads": 0,
        "masked": True,
    },
    "gqa_pse_only": {
        "stored_kv_heads": GQA_KV_HEADS,
        "op_num_key_value_heads": GQA_KV_HEADS,
        "masked": False,
        "pse_only": True,
    },
    "gqa_masked_static_actual": {
        "stored_kv_heads": GQA_KV_HEADS,
        "op_num_key_value_heads": GQA_KV_HEADS,
        "masked": True,
        "static_actual": True,
    },
    "gqa_masked_pse_sentinel": {
        "stored_kv_heads": GQA_KV_HEADS,
        "op_num_key_value_heads": GQA_KV_HEADS,
        "masked": True,
        "pse_sentinel": True,
    },
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", required=True, choices=tuple(LANES))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cache-length", type=int, default=4096)
    parser.add_argument("--effective-length", type=int, default=1280)
    parser.add_argument(
        "--batch-pattern",
        choices=("uniform", "mixed"),
        default="uniform",
        help="mixed gives only --target-row the requested effective length",
    )
    parser.add_argument("--target-row", type=int, default=0)
    parser.add_argument("--inactive-effective-length", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.cache_length < 1:
        parser.error("--cache-length must be positive")
    if not 1 <= args.effective_length <= args.cache_length:
        parser.error("--effective-length must be in [1, cache_length]")
    if not 1 <= args.inactive_effective_length <= args.cache_length:
        parser.error("--inactive-effective-length must be in [1, cache_length]")
    if not 0 <= args.target_row < args.batch_size:
        parser.error("--target-row must be inside the batch")
    if args.batch_pattern == "mixed" and args.batch_size == 1:
        parser.error("--batch-pattern mixed requires --batch-size > 1")
    return args


class Recorder:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def emit(self, event: str, **fields: Any) -> None:
        row = {"event": event, "monotonic_s": time.monotonic(), **fields}
        line = json.dumps(row, sort_keys=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
        print("INCREFA_GQA " + line, flush=True)


def torch_npu_git_version(torch_npu: Any) -> str | None:
    version_module = getattr(torch_npu, "version", None)
    value = getattr(version_module, "git_version", None)
    return None if value is None else str(value)


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    import torch_npu

    if not torch.npu.is_available():
        raise RuntimeError("an available Ascend NPU is required")

    torch.npu.set_compile_mode(jit_compile=False)
    torch.manual_seed(args.seed)
    device = torch.device("npu:0")
    dtype = torch.float16

    output = args.output.expanduser().resolve()
    recorder = Recorder(output.with_suffix(".progress.jsonl"))
    lane = LANES[args.lane]

    effective_lengths = torch.full(
        (args.batch_size,),
        args.effective_length,
        dtype=torch.int64,
    )
    if args.batch_pattern == "mixed":
        effective_lengths.fill_(args.inactive_effective_length)
        effective_lengths[args.target_row] = args.effective_length

    setup = {
        "lane": args.lane,
        "batch_size": args.batch_size,
        "batch_pattern": args.batch_pattern,
        "target_row": args.target_row,
        "effective_lengths": effective_lengths.tolist(),
        "cache_length": args.cache_length,
        "query_heads": QUERY_HEADS,
        "stored_kv_heads": lane["stored_kv_heads"],
        "op_num_key_value_heads": lane["op_num_key_value_heads"],
        "head_dim": HEAD_DIM,
        "masked": lane["masked"],
        "mask_dtype": "bool" if lane["masked"] else None,
        "input_layout": "BNSD",
        "actual_seq_lengths": None,
        "torch": torch.__version__,
        "torch_npu": getattr(torch_npu, "__version__", None),
        "torch_npu_git_version": torch_npu_git_version(torch_npu),
        "python": platform.python_version(),
    }
    recorder.emit("setup_begin", **setup)

    query = torch.randn(
        (args.batch_size, QUERY_HEADS, 1, HEAD_DIM),
        device=device,
        dtype=dtype,
    )
    cache_shape = (
        args.batch_size,
        lane["stored_kv_heads"],
        args.cache_length,
        HEAD_DIM,
    )
    key = torch.zeros(cache_shape, device=device, dtype=dtype)
    value = torch.zeros(cache_shape, device=device, dtype=dtype)

    atten_mask = None
    pse_shift = None
    physical_positions = None
    lengths_npu = None
    if lane["masked"] or lane.get("pse_only"):
        physical_positions = torch.arange(
            args.cache_length,
            device=device,
            dtype=torch.int64,
        ).view(1, 1, 1, args.cache_length)
        lengths_npu = effective_lengths.to(device).view(args.batch_size, 1, 1, 1)
    if lane["masked"]:
        atten_mask = (physical_positions >= lengths_npu).contiguous()

    if lane.get("pse_only"):
        pse_shift = torch.zeros(
            (args.batch_size, QUERY_HEADS, 1, args.cache_length),
            device=device,
            dtype=dtype,
        ).masked_fill(
            (physical_positions >= lengths_npu).expand(
                args.batch_size, QUERY_HEADS, 1, args.cache_length
            ),
            float("-inf"),
        )

    if lane.get("pse_sentinel"):
        sentinel_positions = effective_lengths.to(device).view(
            args.batch_size, 1, 1, 1
        )
        is_sentinel = physical_positions == sentinel_positions
        atten_mask = atten_mask & ~is_sentinel
        pse_shift = torch.zeros(
            (args.batch_size, QUERY_HEADS, 1, args.cache_length),
            device=device,
            dtype=dtype,
        ).masked_fill(
            is_sentinel.expand(
                args.batch_size, QUERY_HEADS, 1, args.cache_length
            ),
            float("-inf"),
        )

    actual_seq_lengths = None
    if lane.get("static_actual"):
        actual_seq_lengths = [args.cache_length] * args.batch_size

    torch.npu.synchronize()
    recorder.emit("setup_end")
    recorder.emit("call_begin")
    started = time.perf_counter()
    attention_output = torch_npu.npu_incre_flash_attention(
        query,
        key,
        value,
        pse_shift=pse_shift,
        atten_mask=atten_mask,
        actual_seq_lengths=actual_seq_lengths,
        num_heads=QUERY_HEADS,
        num_key_value_heads=lane["op_num_key_value_heads"],
        input_layout="BNSD",
        scale_value=SCALE_VALUE,
    )
    recorder.emit("call_returned")
    recorder.emit("sync_begin")
    torch.npu.synchronize()
    elapsed_s = time.perf_counter() - started
    recorder.emit("sync_end", elapsed_s=elapsed_s)

    output_cpu = attention_output.float().cpu()
    result = {
        "schema_version": 1,
        "kind": "standalone_increfa_masked_gqa_discriminator",
        "passed": bool(torch.isfinite(output_cpu).all()),
        "configuration": setup,
        "elapsed_s": elapsed_s,
        "output_shape": list(output_cpu.shape),
        "output_abs_mean": float(output_cpu.abs().mean()),
        "progress_jsonl": str(recorder.path),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    recorder.emit("probe_end", passed=result["passed"], output=str(output))
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
