#!/usr/bin/env python3
"""Validate persistent PA_NZ cache updates through TorchAir RefData."""

from __future__ import annotations

import argparse
import importlib
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


def _install_scatter_pa_metadata_converter(torchair) -> None:
    """Preserve concrete PA cache descriptors across the converter.

    The installed ScatterPaKvCache converter creates TensorMove nodes without
    assigning their output metadata. ``set_meta`` alone fills TorchAir's
    symbolic metadata but not the protobuf shape used by CANN tiling. Materialize
    the static descriptor here because this probe compiles a static full graph.
    """
    del torchair
    # TorchAir imports custom converters lazily from _get_converter().  Load
    # the stock registration before installing the override so that the lazy
    # import cannot replace this converter after compilation starts.
    importlib.import_module(
        "torchair._ge_concrete_graph.ge_converter.custom."
        "npu_scatter_pa_kv_cache"
    )

    from torchair._ge_concrete_graph import ge_apis as ge
    from torchair._ge_concrete_graph.fx2ge_converter import (
        register_fx_node_ge_converter,
    )
    from torchair.ge._ge_graph import Tensor, TensorSpec

    def preserve_static_descriptor(
        target: Tensor,
        source: Tensor,
    ) -> Tensor:
        target.set_meta(source.meta)
        target.desc.shape.dim[:] = [
            int(dimension) for dimension in source.meta.shape
        ]
        target.desc.layout = source.desc.layout or "ND"
        target.desc.device_type = source.desc.device_type or "NPU"
        return target

    def convert_scatter_pa(
        key: Tensor,
        value: Tensor,
        key_cache: Tensor,
        value_cache: Tensor,
        slot_mapping: Tensor,
        *,
        compress_lens: Tensor | None = None,
        compress_seq_offset: Tensor | None = None,
        seq_lens: Tensor | None = None,
        meta_outputs: TensorSpec | None = None,
    ):
        del meta_outputs
        key_cache_copy = preserve_static_descriptor(
            ge.TensorMove(key_cache),
            key_cache,
        )
        value_cache_copy = preserve_static_descriptor(
            ge.TensorMove(value_cache),
            value_cache,
        )
        key_cache_out, value_cache_out = ge.ScatterPaKvCache(
            key,
            key_cache_copy,
            slot_mapping,
            value,
            value_cache_copy,
            compress_lens=compress_lens,
            compress_seq_offset=compress_seq_offset,
            seq_lens=seq_lens,
            cache_mode="PA_NZ",
            scatter_mode="None",
            strides=[1, 1],
            offsets=[0, 0],
        )
        return (
            preserve_static_descriptor(key_cache_out, key_cache),
            preserve_static_descriptor(value_cache_out, value_cache),
        )

    register_fx_node_ge_converter(
        torch.ops.npu.npu_scatter_pa_kv_cache_functional.default
    )(convert_scatter_pa)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument(
        "--tile-size",
        type=int,
        choices=(16, 32),
        default=16,
        help="PA_NZ hidden-dimension tile used by the cache view.",
    )
    parser.add_argument(
        "--acl-format",
        type=int,
        choices=(0, 29),
        default=0,
        help=(
            "0 uses the 910B PA_NZ logical tiled buffer; 29 additionally "
            "requests an ACL FRACTAL_NZ tensor as used by the 310P path."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPO_ROOT
            / "tmp/09_persistent_page_engine/text_decode_lab"
            / "scatter_pa_refdata_probe.json"
        ),
    )
    parser.add_argument(
        "--skip-eager",
        action="store_true",
        help=(
            "Skip the eager control. This is needed when testing an ACL "
            "internal format that the eager ACLNN route rejects but GE may "
            "accept."
        ),
    )
    parser.add_argument(
        "--compiled-call",
        choices=("inplace", "functional"),
        default="inplace",
        help=(
            "Compile the public in-place op with RefData, or call its "
            "functionalized overload directly and thread returned caches."
        ),
    )
    parser.add_argument(
        "--omniinfer-cache-allocation",
        action="store_true",
        help=(
            "Allocate K/V as one 5-D tensor, cast it to ACL format 2 (ND), "
            "then split it exactly as OmniInfer does."
        ),
    )
    parser.add_argument(
        "--graph-dump-dir",
        type=Path,
        default=None,
        help="Optional directory for TorchAir pbtxt graph dumps.",
    )
    return parser.parse_args(argv)


class InplaceUpdate(torch.nn.Module):
    def forward(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        torch_npu.npu_scatter_pa_kv_cache(
            key=key,
            value=value,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mapping=slot_mapping,
            cache_mode="PA_NZ",
        )
        return key_cache, value_cache


class FunctionalUpdate(torch.nn.Module):
    def forward(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.npu.npu_scatter_pa_kv_cache_functional.default(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
        )


def _allocate_cache(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    acl_format: int,
) -> torch.Tensor:
    if acl_format == 0:
        return torch.zeros(shape, device=device, dtype=torch.float16)
    cache = torch_npu.empty_with_format(
        shape,
        dtype=torch.float16,
        device=device,
        acl_format=acl_format,
    )
    cache.zero_()
    return cache


def _allocate_cache_pair(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    acl_format: int,
    omniinfer_allocation: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not omniinfer_allocation:
        return (
            _allocate_cache(shape, device=device, acl_format=acl_format),
            _allocate_cache(shape, device=device, acl_format=acl_format),
        )
    if acl_format != 0:
        raise ValueError(
            "OmniInfer allocation owns the cache format; use --acl-format 0"
        )
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


def _run_steps(
    stage,
    keys: list[torch.Tensor],
    values: list[torch.Tensor],
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    start_slot: int,
    block_size: int,
    tile_size: int,
    device: torch.device,
) -> dict[str, object]:
    step_results: list[dict[str, object]] = []
    key_in = key_cache
    value_in = value_cache
    for step, (key, value) in enumerate(zip(keys, values, strict=True)):
        slot = start_slot + step
        slot_mapping = torch.tensor(
            [slot],
            device=device,
            dtype=torch.int32,
        )
        key_out, value_out = stage(
            key,
            value,
            key_in,
            value_in,
            slot_mapping,
        )
        synchronize(device)
        step_results.append(
            {
                "step": step,
                "slot": slot,
                "input_key_error": _slot_error(
                    key,
                    key_in,
                    slot=slot,
                    block_size=block_size,
                    tile_size=tile_size,
                ),
                "input_value_error": _slot_error(
                    value,
                    value_in,
                    slot=slot,
                    block_size=block_size,
                    tile_size=tile_size,
                ),
                "output_key_error": _slot_error(
                    key,
                    key_out,
                    slot=slot,
                    block_size=block_size,
                    tile_size=tile_size,
                ),
                "output_value_error": _slot_error(
                    value,
                    value_out,
                    slot=slot,
                    block_size=block_size,
                    tile_size=tile_size,
                ),
                "input_output_alias": (
                    key_in.data_ptr() == key_out.data_ptr()
                    and value_in.data_ptr() == value_out.data_ptr()
                ),
                "input_key_nonzero": int((key_in != 0).sum().cpu()),
                "input_value_nonzero": int((value_in != 0).sum().cpu()),
                "output_key_nonzero": int((key_out != 0).sum().cpu()),
                "output_value_nonzero": int((value_out != 0).sum().cpu()),
            }
        )
        key_in = key_out
        value_in = value_out
    return {
        "key_cache_format": int(torch_npu.get_npu_format(key_cache)),
        "value_cache_format": int(torch_npu.get_npu_format(value_cache)),
        "steps": step_results,
    }


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not torch.npu.is_available():
        raise RuntimeError("probe requires an available Ascend NPU")
    if args.steps < 2:
        raise ValueError("--steps must be at least 2")

    device = torch.device("npu:0")
    torch.npu.set_compile_mode(jit_compile=False)
    if args.omniinfer_cache_allocation:
        torch.npu.config.allow_internal_format = True
    torch.manual_seed(7)

    block_size = 128
    num_blocks = 8
    num_kv_heads = 2
    head_dim = 128
    hidden_size = num_kv_heads * head_dim
    if hidden_size % args.tile_size:
        raise ValueError("KV hidden size must be divisible by tile size")
    shape = (
        num_blocks,
        hidden_size // args.tile_size,
        block_size,
        args.tile_size,
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

    eager_key_cache, eager_value_cache = _allocate_cache_pair(
        shape,
        device=device,
        acl_format=args.acl_format,
        omniinfer_allocation=args.omniinfer_cache_allocation,
    )
    compiled_key_cache, compiled_value_cache = _allocate_cache_pair(
        shape,
        device=device,
        acl_format=args.acl_format,
        omniinfer_allocation=args.omniinfer_cache_allocation,
    )

    eager_stage = InplaceUpdate().eval()
    eager = None
    if not args.skip_eager:
        eager = _run_steps(
            eager_stage,
            keys,
            values,
            eager_key_cache,
            eager_value_cache,
            start_slot=start_slot,
            block_size=block_size,
            tile_size=args.tile_size,
            device=device,
        )

    torchair, CompilerConfig = import_torchair()
    _install_scatter_pa_metadata_converter(torchair)
    compiler_config = CompilerConfig()
    compiler_config.experimental_config.enable_ref_data = True
    if args.graph_dump_dir is not None:
        graph_dump_dir = args.graph_dump_dir.expanduser().resolve()
        graph_dump_dir.mkdir(parents=True, exist_ok=True)
        compiler_config.debug.graph_dump.type = "pbtxt"
        compiler_config.debug.graph_dump.path = str(graph_dump_dir)
    backend = torchair.get_npu_backend(compiler_config=compiler_config)
    torch._dynamo.reset()
    compiled_module = (
        InplaceUpdate()
        if args.compiled_call == "inplace"
        else FunctionalUpdate()
    ).eval()
    compiled_stage = torch.compile(
        compiled_module.forward,
        backend=backend,
        dynamic=False,
        fullgraph=True,
    )
    compiled = _run_steps(
        compiled_stage,
        keys,
        values,
        compiled_key_cache,
        compiled_value_cache,
        start_slot=start_slot,
        block_size=block_size,
        tile_size=args.tile_size,
        device=device,
    )

    expected_nonzero = args.steps * hidden_size
    passed = all(
        step["output_key_error"] == 0.0
        and step["output_value_error"] == 0.0
        for step in compiled["steps"]
    )
    passed = (
        passed
        and compiled["steps"][-1]["output_key_nonzero"] == expected_nonzero
        and compiled["steps"][-1]["output_value_nonzero"] == expected_nonzero
    )
    if args.compiled_call == "inplace":
        passed = passed and all(
            step["input_key_error"] == 0.0
            and step["input_value_error"] == 0.0
            and step["input_output_alias"]
            for step in compiled["steps"]
        )
    if eager is not None:
        passed = passed and all(
            step["input_key_error"] == 0.0
            and step["input_value_error"] == 0.0
            and step["output_key_error"] == 0.0
            and step["output_value_error"] == 0.0
            and step["input_output_alias"]
            for step in eager["steps"]
        )
        passed = (
            passed
            and eager["steps"][-1]["output_key_nonzero"] == expected_nonzero
            and eager["steps"][-1]["output_value_nonzero"] == expected_nonzero
        )
    result = {
        "schema_version": 1,
        "passed": passed,
        "cache_shape": list(shape),
        "acl_format_requested": args.acl_format,
        "omniinfer_cache_allocation": args.omniinfer_cache_allocation,
        "tile_size": args.tile_size,
        "steps_requested": args.steps,
        "expected_nonzero_per_cache": expected_nonzero,
        "compiler": {
            "route": "torch.compile",
            "compiled_call": args.compiled_call,
            "fullgraph": True,
            "dynamic": False,
            "enable_ref_data": True,
            "scatter_pa_metadata_converter": True,
        },
        "eager": eager,
        "compiled": compiled,
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
