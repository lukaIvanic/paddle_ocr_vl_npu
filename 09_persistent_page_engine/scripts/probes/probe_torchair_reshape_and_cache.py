#!/usr/bin/env python3
"""Validate stateful _npu_reshape_and_cache through TorchAir cache_compile."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import torch
import torch_npu

HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

from paddleocr_vl.model.compile_utils import import_torchair
from utils.timing import synchronize


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=(
            REPO_ROOT
            / ".runtime_cache/09_persistent_page_engine_torchair"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPO_ROOT
            / "tmp/09_persistent_page_engine/text_decode_lab"
            / "torchair_reshape_and_cache.json"
        ),
    )
    args = parser.parse_args(argv)
    if args.steps < 2:
        parser.error("--steps must be at least 2")
    return args


def _source_hash() -> str:
    return hashlib.sha1(Path(__file__).read_bytes()).hexdigest()[:12]


class ReshapeAndCache(torch.nn.Module):
    def forward(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        torch_npu._npu_reshape_and_cache(
            key=key,
            value=value,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_indices=slot_mapping,
        )
        return key_cache, value_cache


def _slot(
    cache: torch.Tensor,
    slot: int,
    *,
    block_size: int,
) -> torch.Tensor:
    block, offset = divmod(slot, block_size)
    return cache[block, offset]


def _run_steps(
    stage,
    keys: list[torch.Tensor],
    values: list[torch.Tensor],
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    start_slot: int,
    block_size: int,
    device: torch.device,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    expected: list[tuple[int, torch.Tensor, torch.Tensor]] = []
    for step, (key, value) in enumerate(zip(keys, values, strict=True)):
        slot = start_slot + step
        slot_mapping.fill_(slot)
        key_out, value_out = stage(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
        )
        synchronize(device)
        expected.append((slot, key, value))
        retained_key_error = max(
            float(
                (
                    _slot(key_cache, old_slot, block_size=block_size).float()
                    - old_key[0].float()
                )
                .abs()
                .max()
                .cpu()
            )
            for old_slot, old_key, _ in expected
        )
        retained_value_error = max(
            float(
                (
                    _slot(
                        value_cache,
                        old_slot,
                        block_size=block_size,
                    ).float()
                    - old_value[0].float()
                )
                .abs()
                .max()
                .cpu()
            )
            for old_slot, _, old_value in expected
        )
        rows.append(
            {
                "step": step,
                "slot": slot,
                "key_error": retained_key_error,
                "value_error": retained_value_error,
                "input_output_alias": (
                    key_cache.data_ptr() == key_out.data_ptr()
                    and value_cache.data_ptr() == value_out.data_ptr()
                ),
                "key_nonzero": int((key_cache != 0).sum().cpu()),
                "value_nonzero": int((value_cache != 0).sum().cpu()),
            }
        )
    return rows


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not torch.npu.is_available():
        raise RuntimeError("probe requires an available Ascend NPU")

    device = torch.device("npu:0")
    torch.npu.set_compile_mode(jit_compile=False)
    torch.manual_seed(7)

    block_size = 128
    num_blocks = 8
    num_kv_heads = 2
    head_dim = 128
    start_slot = block_size - 1
    cache_shape = (
        num_blocks,
        block_size,
        num_kv_heads,
        head_dim,
    )
    keys = [
        torch.randn(
            (1, num_kv_heads, head_dim),
            device=device,
            dtype=torch.float16,
        )
        for _ in range(args.steps)
    ]
    values = [torch.randn_like(key) for key in keys]

    eager_key_cache = torch.zeros(
        cache_shape,
        device=device,
        dtype=torch.float16,
    )
    eager_value_cache = torch.zeros_like(eager_key_cache)
    compiled_key_cache = torch.zeros_like(eager_key_cache)
    compiled_value_cache = torch.zeros_like(eager_value_cache)
    eager_slot = torch.zeros((1,), device=device, dtype=torch.int32)
    compiled_slot = torch.zeros_like(eager_slot)

    stage = ReshapeAndCache().eval()
    eager_rows = _run_steps(
        stage,
        keys,
        values,
        eager_key_cache,
        eager_value_cache,
        eager_slot,
        start_slot=start_slot,
        block_size=block_size,
        device=device,
    )

    torchair, CompilerConfig = import_torchair()
    cache_dir = (
        args.cache_dir.expanduser().resolve()
        / f"reshape_and_cache_src{_source_hash()}"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    compile_started = time.perf_counter()
    compiled = torchair.inference.cache_compile(
        stage.forward,
        config=CompilerConfig(),
        dynamic=False,
        cache_dir=str(cache_dir),
        ge_cache=True,
    )
    compile_wrapper_s = time.perf_counter() - compile_started
    first_call_started = time.perf_counter()
    compiled_rows = _run_steps(
        compiled,
        keys,
        values,
        compiled_key_cache,
        compiled_value_cache,
        compiled_slot,
        start_slot=start_slot,
        block_size=block_size,
        device=device,
    )
    first_calls_s = time.perf_counter() - first_call_started

    passed = all(
        row["key_error"] == 0.0
        and row["value_error"] == 0.0
        and row["input_output_alias"]
        for row in eager_rows + compiled_rows
    )
    result = {
        "schema_version": 1,
        "passed": passed,
        "operator": "torch_npu._npu_reshape_and_cache",
        "route": "torchair.inference.cache_compile",
        "cache_shape": list(cache_shape),
        "cache_format": int(torch_npu.get_npu_format(compiled_key_cache)),
        "cache_dir": str(cache_dir),
        "compile_wrapper_s": compile_wrapper_s,
        "compile_and_first_calls_s": first_calls_s,
        "eager_steps": eager_rows,
        "compiled_steps": compiled_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"OUTPUT_JSON={args.output}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
