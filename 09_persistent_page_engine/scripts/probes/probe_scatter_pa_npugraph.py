#!/usr/bin/env python3
"""Validate persistent PA_NZ cache writes through direct NPU graph replay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import torch
import torch_npu

HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

from utils.timing import synchronize


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPO_ROOT
            / "tmp/09_persistent_page_engine/text_decode_lab"
            / "scatter_pa_npugraph_probe.json"
        ),
    )
    return parser.parse_args(argv)


def _allocate_omniinfer_cache_pair(
    shape: tuple[int, ...],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.npu.config.allow_internal_format = True
    pair = torch.zeros(
        (2, *shape),
        device=device,
        dtype=torch.float16,
    )
    pair = torch_npu.npu_format_cast(pair, 2)
    return pair[0], pair[1]


def _slot_error(
    update: torch.Tensor,
    cache: torch.Tensor,
    *,
    slot: int,
    block_size: int,
    tile_size: int,
) -> float:
    physical_block, offset = divmod(slot, block_size)
    expected = update.reshape(-1, tile_size)
    actual = cache[physical_block, :, offset, :]
    return float((actual.float() - expected.float()).abs().max().cpu())


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not torch.npu.is_available():
        raise RuntimeError("probe requires an available Ascend NPU")
    if args.steps < 2:
        raise ValueError("--steps must be at least 2")

    device = torch.device("npu:0")
    torch.npu.set_compile_mode(jit_compile=False)
    torch.manual_seed(7)

    block_size = 128
    num_blocks = 8
    num_kv_heads = 2
    head_dim = 128
    tile_size = 16
    hidden_size = num_kv_heads * head_dim
    cache_shape = (
        num_blocks,
        hidden_size // tile_size,
        block_size,
        tile_size,
    )
    start_slot = 768
    keys = [
        torch.randn(
            (1, num_kv_heads, head_dim),
            device=device,
            dtype=torch.float16,
        )
        for _ in range(args.steps)
    ]
    values = [torch.randn_like(key) for key in keys]

    key_cache, value_cache = _allocate_omniinfer_cache_pair(
        cache_shape,
        device=device,
    )
    stable_key = torch.zeros_like(keys[0])
    stable_value = torch.zeros_like(values[0])
    stable_slot = torch.zeros((1,), device=device, dtype=torch.int32)

    synchronize(device)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        torch_npu.npu_scatter_pa_kv_cache(
            key=stable_key,
            value=stable_value,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mapping=stable_slot,
            cache_mode="PA_NZ",
        )
    synchronize(device)

    step_results: list[dict[str, object]] = []
    input_addresses = {
        "key": stable_key.data_ptr(),
        "value": stable_value.data_ptr(),
        "slot_mapping": stable_slot.data_ptr(),
        "key_cache": key_cache.data_ptr(),
        "value_cache": value_cache.data_ptr(),
    }
    for step, (key, value) in enumerate(zip(keys, values, strict=True)):
        slot = start_slot + step
        stable_key.copy_(key)
        stable_value.copy_(value)
        stable_slot.fill_(slot)
        graph.replay()
        synchronize(device)
        step_results.append(
            {
                "step": step,
                "slot": slot,
                "key_error": _slot_error(
                    key,
                    key_cache,
                    slot=slot,
                    block_size=block_size,
                    tile_size=tile_size,
                ),
                "value_error": _slot_error(
                    value,
                    value_cache,
                    slot=slot,
                    block_size=block_size,
                    tile_size=tile_size,
                ),
                "key_nonzero": int((key_cache != 0).sum().cpu()),
                "value_nonzero": int((value_cache != 0).sum().cpu()),
            }
        )

    expected_nonzero = args.steps * hidden_size
    passed = all(
        step["key_error"] == 0.0 and step["value_error"] == 0.0
        for step in step_results
    )
    passed = (
        passed
        and step_results[-1]["key_nonzero"] == expected_nonzero
        and step_results[-1]["value_nonzero"] == expected_nonzero
    )
    result = {
        "schema_version": 1,
        "passed": passed,
        "route": "torch.npu.NPUGraph",
        "cache_mode": "PA_NZ",
        "cache_shape": list(cache_shape),
        "cache_format": int(torch_npu.get_npu_format(key_cache)),
        "stable_input_addresses": input_addresses,
        "steps_requested": args.steps,
        "expected_nonzero_per_cache": expected_nonzero,
        "steps": step_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
