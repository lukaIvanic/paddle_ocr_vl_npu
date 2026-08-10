#!/usr/bin/env python3
"""Validate the independent B1 K/V-scatter ordering operator through TorchAir."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch

from paddleocr_vl.model.compile_utils import import_torchair
from paddleocr_vl.model.decode_kv_scatter_query import (
    decode_kv_scatter_query,
    register_decode_kv_scatter_query_converter,
)


class DecodeKvScatterQuery(torch.nn.Module):
    def __init__(self, strict_scope: bool) -> None:
        super().__init__()
        self.scope = None
        if strict_scope:
            self.scope = __import__(
                "torchair.scope", fromlist=["super_kernel"]
            ).super_kernel

    def _forward_impl(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cache_position: torch.Tensor,
        key_state: torch.Tensor,
        value_state: torch.Tensor,
    ) -> torch.Tensor:
        return decode_kv_scatter_query(
            query,
            key_cache,
            value_cache,
            cache_position,
            key_state,
            value_state,
        )

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        if self.scope is None:
            return self._forward_impl(*args)
        with self.scope(
            "paddle_decode_kv_scatter_query_probe",
            "feed-sync-all=0:stream-fusion=0:strict-scope-check=abort:"
            "preload-code=none:early-start=0:split-mode=1",
        ):
            return self._forward_impl(*args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict-scope", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import torch_npu

    if not torch.npu.is_available():
        raise RuntimeError("an Ascend NPU is required")
    torch.npu.set_compile_mode(jit_compile=False)
    register_decode_kv_scatter_query_converter()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260810)
    query = torch.randn(
        (1, 16, 1, 128), generator=generator, dtype=torch.float16
    ).to("npu:0")
    key_cache = torch.zeros(
        (1, 2, 1024, 128), dtype=torch.float16, device="npu:0"
    )
    value_cache = torch.zeros_like(key_cache)
    key_states = [
        torch.randn((1, 2, 1, 128), generator=generator, dtype=torch.float16).to("npu:0")
        for _ in range(2)
    ]
    value_states = [torch.randn_like(state) for state in key_states]
    expected_keys = [state.cpu().clone() for state in key_states]
    expected_values = [state.cpu().clone() for state in value_states]
    positions = [128, 129]

    torchair, CompilerConfig = import_torchair()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    compiler_config = CompilerConfig()
    graph_dump_dir = args.cache_dir / "graph_dump"
    graph_dump_dir.mkdir(parents=True, exist_ok=True)
    compiler_config.debug.graph_dump.type = "pbtxt"
    compiler_config.debug.graph_dump.path = str(graph_dump_dir)
    step = torchair.inference.cache_compile(
        DecodeKvScatterQuery(args.strict_scope).forward,
        config=compiler_config,
        dynamic=False,
        cache_dir=str(args.cache_dir),
        ge_cache=True,
    )
    timings = []
    ordered_outputs = []
    ordered_aliases = []
    for position, key_state, value_state in zip(
        positions, key_states, value_states, strict=True
    ):
        position_tensor = torch.tensor(
            [position], dtype=torch.int64, device="npu:0"
        )
        started = time.perf_counter()
        ordered = step(
            query,
            key_cache,
            value_cache,
            position_tensor,
            key_state,
            value_state,
        )
        torch.npu.synchronize()
        timings.append(time.perf_counter() - started)
        ordered_aliases.append(
            {
                "query": ordered.data_ptr() == query.data_ptr(),
                "key_cache": ordered.data_ptr() == key_cache.data_ptr(),
                "value_cache": ordered.data_ptr() == value_cache.data_ptr(),
                "key_state": ordered.data_ptr() == key_state.data_ptr(),
                "value_state": ordered.data_ptr() == value_state.data_ptr(),
            }
        )
        ordered_outputs.append(ordered.cpu().clone())

    checks = []
    all_exact = True
    state_candidates = {
        "key_0": expected_keys[0],
        "key_1": expected_keys[1],
        "value_0": expected_values[0],
        "value_1": expected_values[1],
    }
    for index, position in enumerate(positions):
        query_expected = query.cpu()
        key_actual = key_cache[:, :, position : position + 1, :].cpu()
        key_expected = expected_keys[index]
        value_actual = value_cache[:, :, position : position + 1, :].cpu()
        value_expected = expected_values[index]
        query_exact = bool(torch.equal(ordered_outputs[index], query_expected))
        key_exact = bool(
            torch.equal(key_actual, key_expected)
        )
        value_exact = bool(
            torch.equal(value_actual, value_expected)
        )
        all_exact = all_exact and query_exact and key_exact and value_exact
        checks.append(
            {
                "position": position,
                "query_exact": query_exact,
                "ordered_output_aliases": ordered_aliases[index],
                "query_max_abs": float(
                    (ordered_outputs[index].float() - query_expected.float())
                    .abs()
                    .max()
                ),
                "key_exact": key_exact,
                "key_max_abs": float(
                    (key_actual.float() - key_expected.float()).abs().max()
                ),
                "value_exact": value_exact,
                "value_max_abs": float(
                    (value_actual.float() - value_expected.float()).abs().max()
                ),
                "key_candidate_max_abs": {
                    name: float(
                        (key_actual.float() - candidate.float()).abs().max()
                    )
                    for name, candidate in state_candidates.items()
                },
                "value_candidate_max_abs": {
                    name: float(
                        (value_actual.float() - candidate.float()).abs().max()
                    )
                    for name, candidate in state_candidates.items()
                },
            }
        )
    result = {
        "kind": "paddle_decode_kv_scatter_query_torchair_probe",
        "operator": {
            "pytorch": "paddleocr_vl::decode_kv_scatter_query_v1",
            "ge": "PaddleDecodeKvScatterQueryV1",
            "kernel": "paddle_decode_kv_scatter_query_v1",
        },
        "contract": {
            "query_shape": list(query.shape),
            "cache_shape": list(key_cache.shape),
            "state_shape": list(key_states[0].shape),
            "positions": positions,
            "strict_scope": args.strict_scope,
            "block_dim": 1,
            "core_type": "AIV_ONLY",
        },
        "environment": {
            "device": torch.npu.get_device_name(0),
            "visible_device": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
        },
        "correctness": {
            "all_exact": all_exact,
            "steps": checks,
            "key_nonzero_count": int(torch.count_nonzero(key_cache).cpu()),
            "value_nonzero_count": int(torch.count_nonzero(value_cache).cpu()),
            "key_state_inputs_preserved": [
                bool(torch.equal(actual.cpu(), expected))
                for actual, expected in zip(key_states, expected_keys, strict=True)
            ],
            "value_state_inputs_preserved": [
                bool(torch.equal(actual.cpu(), expected))
                for actual, expected in zip(value_states, expected_values, strict=True)
            ],
        },
        "timing": {"call_s": timings},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not all_exact:
        raise RuntimeError("PaddleDecodeKvScatterQueryV1 failed exact parity")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
