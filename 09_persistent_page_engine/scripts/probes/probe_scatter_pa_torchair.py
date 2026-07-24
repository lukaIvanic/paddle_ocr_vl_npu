#!/usr/bin/env python3
"""Check ScatterPaKvCache functional outputs through TorchAir cache_compile."""

from __future__ import annotations

import argparse
import hashlib
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

from paddleocr_vl.model.compile_utils import import_torchair
from utils.timing import synchronize


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
            / "scatter_pa_torchair_probe.json"
        ),
    )
    return parser.parse_args(argv)


def _source_hash() -> str:
    return hashlib.sha1(Path(__file__).read_bytes()).hexdigest()[:12]


class FunctionalUpdate(torch.nn.Module):
    def forward(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch_npu.npu_scatter_pa_kv_cache_functional(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
        )


def _stats(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    physical_block: int,
    offset: int,
) -> dict[str, float | int]:
    expected_key = key.reshape(1, 16, 16)[0]
    expected_value = value.reshape(1, 16, 16)[0]
    return {
        "key_slot_max_abs": float(
            (
                key_cache[physical_block, :, offset, :].float()
                - expected_key.float()
            )
            .abs()
            .max()
            .cpu()
        ),
        "value_slot_max_abs": float(
            (
                value_cache[physical_block, :, offset, :].float()
                - expected_value.float()
            )
            .abs()
            .max()
            .cpu()
        ),
        "key_nonzero": int((key_cache != 0).sum().cpu()),
        "value_nonzero": int((value_cache != 0).sum().cpu()),
    }


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not torch.npu.is_available():
        raise RuntimeError("probe requires an available Ascend NPU")
    device = torch.device("npu:0")
    torch.npu.set_compile_mode(jit_compile=False)
    torch.manual_seed(7)

    shape = (8, 16, 128, 16)
    key = torch.randn((1, 2, 128), device=device, dtype=torch.float16)
    value = torch.randn_like(key)
    slot_mapping = torch.tensor([768], device=device, dtype=torch.int64)
    eager_key_cache = torch.zeros(shape, device=device, dtype=torch.float16)
    eager_value_cache = torch.zeros_like(eager_key_cache)
    compiled_key_cache = torch.zeros_like(eager_key_cache)
    compiled_value_cache = torch.zeros_like(eager_key_cache)

    stage = FunctionalUpdate().eval()
    eager_key_out, eager_value_out = stage(
        key,
        value,
        eager_key_cache,
        eager_value_cache,
        slot_mapping,
    )

    torchair, CompilerConfig = import_torchair()
    cache_dir = (
        args.cache_dir.expanduser().resolve()
        / f"scatter_pa_functional_probe_src{_source_hash()}"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    compiled = torchair.inference.cache_compile(
        stage.forward,
        config=CompilerConfig(),
        dynamic=False,
        cache_dir=str(cache_dir),
        ge_cache=True,
    )
    compiled_key_out, compiled_value_out = compiled(
        key,
        value,
        compiled_key_cache,
        compiled_value_cache,
        slot_mapping,
    )
    synchronize(device)

    result = {
        "schema_version": 1,
        "cache_dir": str(cache_dir),
        "eager_output": _stats(
            key,
            value,
            eager_key_out,
            eager_value_out,
            physical_block=6,
            offset=0,
        ),
        "compiled_output": _stats(
            key,
            value,
            compiled_key_out,
            compiled_value_out,
            physical_block=6,
            offset=0,
        ),
        "compiled_input_after_call": _stats(
            key,
            value,
            compiled_key_cache,
            compiled_value_cache,
            physical_block=6,
            offset=0,
        ),
    }
    result["passed"] = (
        result["eager_output"]["key_slot_max_abs"] == 0.0
        and result["eager_output"]["value_slot_max_abs"] == 0.0
        and result["compiled_output"]["key_slot_max_abs"] == 0.0
        and result["compiled_output"]["value_slot_max_abs"] == 0.0
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
