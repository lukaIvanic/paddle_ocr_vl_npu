#!/usr/bin/env python3
"""Standalone synthetic reproducer for UniRec vision TorchAir divergence.

This file deliberately imports no project module and loads no model checkpoint
or image.  It recreates a suffix of the masked focal vision encoder with
deterministic synthetic FP16 weights and activations, then compares the exact
same module in raw eager and TorchAir-compiled execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn


PROCESS_STARTED = time.perf_counter()
CANVAS_WIDTH = 1024
CANVAS_HEIGHT = 704
VALID_WIDTH = 960
VALID_HEIGHT = 640
STAGE_FACTORS = (4, 8, 16, 32)
STAGE_DIMS = (96, 192, 384, 768)
STAGE_DEPTHS = (2, 2, 9, 2)


def phase(name: str, **fields: Any) -> None:
    print(
        "UNIREC_STANDALONE_VISION_PHASE "
        + json.dumps(
            {
                "phase": name,
                "process_elapsed_s": time.perf_counter() - PROCESS_STARTED,
                **fields,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-stage", type=int, choices=(0, 1, 2, 3), default=3)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--require-cache", action="store_true")
    parser.add_argument(
        "--trace-boundaries",
        action="store_true",
        help=(
            "return intermediate stage/block boundaries from the same compiled "
            "graph so eager-vs-TorchAir divergence can be localized"
        ),
    )
    parser.add_argument(
        "--weight-format",
        choices=("native", "torchair_internal"),
        default="torchair_internal",
    )
    args = parser.parse_args()
    if args.timing_repeats < 1:
        parser.error("--timing-repeats must be positive")
    if not args.device.startswith("npu"):
        parser.error("the standalone reproducer requires an NPU device")
    return args


class Mlp(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * 4, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class FocalModulation(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.focal_level = 3
        self.f = nn.Linear(dim, 2 * dim + self.focal_level + 1)
        self.h = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.focal_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        dim,
                        dim,
                        kernel_size=kernel,
                        stride=1,
                        padding=kernel // 2,
                        groups=dim,
                        bias=False,
                    ),
                    nn.GELU(),
                )
                for kernel in (3, 5, 7)
            ]
        )


class FocalBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.modulation = FocalModulation(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim)


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(2, 2),
            stride=(2, 2),
        )
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return self.norm(x.flatten(2).transpose(1, 2))


class Stage(nn.Module):
    def __init__(self, index: int) -> None:
        super().__init__()
        dim = STAGE_DIMS[index]
        self.blocks = nn.ModuleList(
            [FocalBlock(dim) for _ in range(STAGE_DEPTHS[index])]
        )
        self.downsample = (
            PatchEmbed(dim, STAGE_DIMS[index + 1]) if index < 3 else None
        )


def mask_nhwc(x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    return x * valid_mask.permute(0, 2, 3, 1)


def masked_global_context(
    ctx: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    row_counts = valid_mask.sum(dim=3, keepdim=True).clamp_min(1)
    row_means = (ctx * valid_mask).sum(dim=3, keepdim=True) / row_counts
    valid_rows = (valid_mask.sum(dim=3, keepdim=True) > 0).to(ctx.dtype)
    return (row_means * valid_rows).sum(dim=2, keepdim=True) / valid_rows.sum(
        dim=2,
        keepdim=True,
    ).clamp_min(1)


def run_masked_focal_block(
    block: FocalBlock,
    x: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    channels = x.shape[1]
    shortcut = x * valid_mask
    normalized = block.norm1(shortcut.permute(0, 2, 3, 1))
    modulation = block.modulation
    projected = modulation.f(normalized).permute(0, 3, 1, 2).contiguous()
    q, ctx, gates = torch.split(
        projected,
        (channels, channels, modulation.focal_level + 1),
        dim=1,
    )
    q = q * valid_mask
    ctx = ctx * valid_mask
    gates = gates * valid_mask
    ctx_all = None
    for level, focal_layer in enumerate(modulation.focal_layers):
        ctx = focal_layer(ctx) * valid_mask
        contribution = ctx * gates[:, level : level + 1]
        ctx_all = contribution if ctx_all is None else ctx_all + contribution
    if ctx_all is None:
        raise RuntimeError("focal modulation unexpectedly has no focal layers")
    global_context = modulation.act(masked_global_context(ctx, valid_mask))
    ctx_all = ctx_all + global_context * gates[:, modulation.focal_level :]
    modulator = modulation.h(ctx_all) * valid_mask
    modulated = q * modulator
    modulated = modulated.permute(0, 2, 3, 1).contiguous()
    modulated = mask_nhwc(modulation.proj(modulated), valid_mask)
    residual = shortcut.permute(0, 2, 3, 1) + modulated
    output = residual + block.mlp(block.norm2(residual))
    return mask_nhwc(output, valid_mask).permute(0, 3, 1, 2).contiguous()


class MaskedVisionSuffix(nn.Module):
    def __init__(self, start_stage: int, *, trace_boundaries: bool = False) -> None:
        super().__init__()
        self.start_stage = int(start_stage)
        self.trace_boundaries = bool(trace_boundaries)
        self.stages = nn.ModuleList(
            [Stage(index) for index in range(self.start_stage, 4)]
        )
        self.projection = nn.Linear(768, 768)

        boundary_specs: list[tuple[str, int, str]] = []
        for stage_index in range(self.start_stage, 4):
            for block_index in range(STAGE_DEPTHS[stage_index]):
                boundary_specs.append(
                    (f"stage_{stage_index}_block_{block_index}", stage_index, "nchw")
                )
            if stage_index < 3:
                boundary_specs.append(
                    (
                        f"stage_{stage_index}_downsample_to_{stage_index + 1}",
                        stage_index + 1,
                        "nchw",
                    )
                )
        boundary_specs.append(("projection", 3, "tokens"))
        self.boundary_specs = tuple(boundary_specs)

    @staticmethod
    def tokens_to_chw(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch = tokens.shape[0]
        height, width = mask.shape[2:]
        tokens = tokens * mask.flatten(2).transpose(1, 2)
        return tokens.transpose(1, 2).reshape(batch, -1, height, width)

    def forward(
        self,
        x: torch.Tensor,
        mask4: torch.Tensor,
        mask8: torch.Tensor,
        mask16: torch.Tensor,
        mask32: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        masks = (mask4, mask8, mask16, mask32)
        boundaries: list[torch.Tensor] = []
        for offset, stage in enumerate(self.stages):
            stage_index = self.start_stage + offset
            stage_mask = masks[stage_index]
            for block in stage.blocks:
                x = run_masked_focal_block(block, x, stage_mask)
                if self.trace_boundaries:
                    boundaries.append(x)
            if stage.downsample is not None:
                x = stage.downsample(x)
                x = self.tokens_to_chw(x, masks[stage_index + 1])
                if self.trace_boundaries:
                    boundaries.append(x)
        tokens = x.flatten(2).transpose(1, 2).contiguous()
        tokens = self.projection(tokens)
        tokens = tokens * mask32.flatten(2).transpose(1, 2)
        if self.trace_boundaries:
            boundaries.append(tokens)
            return tuple(boundaries)
        return tokens


def compact_boundary(
    tensor: torch.Tensor,
    *,
    stage_index: int,
    layout: str,
) -> torch.Tensor:
    factor = STAGE_FACTORS[stage_index]
    valid_height = VALID_HEIGHT // factor
    valid_width = VALID_WIDTH // factor
    if layout == "nchw":
        return tensor[:, :, :valid_height, :valid_width].contiguous()
    if layout == "tokens":
        grid = tensor.reshape(
            tensor.shape[0],
            CANVAS_HEIGHT // factor,
            CANVAS_WIDTH // factor,
            tensor.shape[-1],
        )
        return grid[:, :valid_height, :valid_width].reshape(
            tensor.shape[0], -1, tensor.shape[-1]
        )
    raise ValueError(f"unsupported boundary layout: {layout}")


def difference(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    left = reference.float()
    right = candidate.float()
    delta = right - left
    count = delta.numel()
    denominator = (
        left.square().sum().sqrt() * right.square().sum().sqrt()
    ).clamp_min(1e-12)
    return {
        "shape": list(reference.shape),
        "exact": bool(torch.equal(reference, candidate)),
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().sum(dtype=torch.float64).item() / count),
        "rmse": float((delta.square().sum(dtype=torch.float64) / count).sqrt().item()),
        "cosine": float(((left * right).sum() / denominator).item()),
    }


def inventory(path: Path) -> dict[str, Any]:
    files = sorted(
        file
        for file in path.rglob("*")
        if file.is_file() and (file.name == "compiled_module" or file.suffix == ".om")
    )
    return {
        "compiled_module_count": sum(file.name == "compiled_module" for file in files),
        "om_count": sum(file.suffix == ".om" for file in files),
        "files": [
            {
                "path": str(file.relative_to(path)),
                "size": file.stat().st_size,
                "mtime_ns": file.stat().st_mtime_ns,
            }
            for file in files
        ],
    }


def timed(fn: Callable[[], torch.Tensor], device: str) -> tuple[torch.Tensor, float]:
    torch.npu.synchronize()
    started = time.perf_counter()
    output = fn()
    torch.npu.synchronize()
    return output, (time.perf_counter() - started) * 1000.0


def main() -> None:
    args = parse_args()
    import torch_npu
    from torch_npu.dynamo.torchair.configs.compiler_config import CompilerConfig

    try:
        from torch_npu.dynamo.torchair.inference import cache_compile
    except ImportError:
        from torchair.inference import cache_compile

    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    if not visible:
        raise RuntimeError("set ASCEND_RT_VISIBLE_DEVICES before running")
    physical_devices = [int(value) for value in visible.split(",") if value.strip()]
    if any(value in {5, 6} for value in physical_devices):
        raise RuntimeError("physical NPU 5 and 6 are excluded")
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.manual_seed(args.seed)
    phase(
        "setup",
        physical_devices=physical_devices,
        npu_jit_compile=False,
        start_stage=args.start_stage,
    )

    module = MaskedVisionSuffix(
        args.start_stage,
        trace_boundaries=args.trace_boundaries,
    ).to(dtype=torch.float16)
    torch_npu.npu.config.allow_internal_format = False
    module = module.to(args.device).eval()
    torch.npu.synchronize()
    if args.weight_format == "torchair_internal":
        torch_npu.npu.config.allow_internal_format = True
        try:
            from torch_npu.dynamo.torchair import use_internal_format_weight
        except ImportError:
            from torchair import use_internal_format_weight
        use_internal_format_weight(module.stages)
        torch.npu.synchronize()
    phase("module_ready", weight_format=args.weight_format)

    factor = STAGE_FACTORS[args.start_stage]
    channels = STAGE_DIMS[args.start_stage]
    physical_height = CANVAS_HEIGHT // factor
    physical_width = CANVAS_WIDTH // factor
    valid_height = VALID_HEIGHT // factor
    valid_width = VALID_WIDTH // factor
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    host_input = torch.randn(
        (1, channels, physical_height, physical_width),
        generator=generator,
        dtype=torch.float32,
    ).mul_(0.2)
    x = host_input.to(args.device, dtype=torch.float16)
    masks = []
    for stage_factor in STAGE_FACTORS:
        mask = torch.zeros(
            (1, 1, CANVAS_HEIGHT // stage_factor, CANVAS_WIDTH // stage_factor),
            dtype=torch.float16,
        )
        mask[
            :,
            :,
            : VALID_HEIGHT // stage_factor,
            : VALID_WIDTH // stage_factor,
        ] = 1
        masks.append(mask.to(args.device))
    x.mul_(masks[args.start_stage])
    masks_tuple = tuple(masks)
    phase(
        "inputs_ready",
        input_shape=list(x.shape),
        valid_shape=[valid_height, valid_width],
    )

    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    cache_dir = args.cache_root.expanduser().resolve() / (
        f"standalone_vision_suffix_s{args.start_stage}_1024x704_fp16_"
        f"trace{int(args.trace_boundaries)}_"
        f"w{args.weight_format}_src{source_hash}"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    before = inventory(cache_dir)
    if args.require_cache and not before["compiled_module_count"]:
        raise RuntimeError(f"required cache is absent: {cache_dir}")
    config = CompilerConfig()
    config.mode.value = "max-autotune"
    config.experimental_config.frozen_parameter.value = True
    compiled = cache_compile(
        module.forward,
        config=config,
        dynamic=False,
        cache_dir=str(cache_dir),
        ge_cache=True,
        fullgraph=True,
    )
    phase("graph_registered", cache_before=before, cache_dir=str(cache_dir))

    call = lambda fn: fn(x, *masks_tuple)
    with torch.inference_mode():
        eager_first, eager_first_ms = timed(lambda: call(module), args.device)
        phase("eager_first", synchronized_ms=eager_first_ms)
        phase("compiled_first_begin", cache_dir=str(cache_dir))
        compiled_first, compiled_first_ms = timed(lambda: call(compiled), args.device)
        phase("compiled_first", synchronized_ms=compiled_first_ms)
        eager_times = []
        compiled_times = []
        eager_output = eager_first
        compiled_output = compiled_first
        for repeat in range(args.timing_repeats):
            ordered = (("eager", module), ("compiled", compiled))
            if repeat % 2:
                ordered = tuple(reversed(ordered))
            for name, fn in ordered:
                output, elapsed_ms = timed(lambda fn=fn: call(fn), args.device)
                if name == "eager":
                    eager_output = output
                    eager_times.append(elapsed_ms)
                else:
                    compiled_output = output
                    compiled_times.append(elapsed_ms)

    if args.trace_boundaries:
        if not isinstance(eager_output, tuple) or not isinstance(compiled_output, tuple):
            raise RuntimeError("boundary trace requested but graph did not return tuples")
        if len(eager_output) != len(module.boundary_specs):
            raise RuntimeError(
                f"boundary count mismatch: {len(eager_output)} != "
                f"{len(module.boundary_specs)}"
            )
        boundary_comparison = {}
        first_divergent_boundary = None
        for spec, eager_tensor, compiled_tensor in zip(
            module.boundary_specs,
            eager_output,
            compiled_output,
            strict=True,
        ):
            name, stage_index, layout = spec
            full = difference(eager_tensor, compiled_tensor)
            valid = difference(
                compact_boundary(
                    eager_tensor,
                    stage_index=stage_index,
                    layout=layout,
                ),
                compact_boundary(
                    compiled_tensor,
                    stage_index=stage_index,
                    layout=layout,
                ),
            )
            boundary_comparison[name] = {
                "stage_index": stage_index,
                "layout": layout,
                "full_physical": full,
                "valid_compact": valid,
            }
            diverged = float(valid["cosine"]) < 0.999 or float(valid["max_abs"]) > 0.5
            if diverged and first_divergent_boundary is None:
                first_divergent_boundary = name
        final_name = module.boundary_specs[-1][0]
        full_difference = boundary_comparison[final_name]["full_physical"]
        compact_difference = boundary_comparison[final_name]["valid_compact"]
    else:
        if isinstance(eager_output, tuple) or isinstance(compiled_output, tuple):
            raise RuntimeError("unexpected tuple output without boundary trace")
        full_difference = difference(eager_output, compiled_output)
        compact_eager = compact_boundary(
            eager_output,
            stage_index=3,
            layout="tokens",
        )
        compact_compiled = compact_boundary(
            compiled_output,
            stage_index=3,
            layout="tokens",
        )
        compact_difference = difference(compact_eager, compact_compiled)
        boundary_comparison = None
        first_divergent_boundary = None
    after = inventory(cache_dir)
    report = {
        "schema": "unirec_standalone_vision_torchair_divergence_v1",
        "status": "pass",
        "physical_devices": physical_devices,
        "canvas": [CANVAS_WIDTH, CANVAS_HEIGHT],
        "valid_pixels": [VALID_WIDTH, VALID_HEIGHT],
        "start_stage": args.start_stage,
        "trace_boundaries": args.trace_boundaries,
        "input_shape": list(x.shape),
        "weight_format": args.weight_format,
        "npu_jit_compile": False,
        "first_call_ms": {
            "eager": eager_first_ms,
            "compiled": compiled_first_ms,
        },
        "steady_p50_ms": {
            "eager": statistics.median(eager_times),
            "compiled": statistics.median(compiled_times),
        },
        "comparison": {
            "full_physical": full_difference,
            "valid_compact": compact_difference,
        },
        "boundary_comparison": boundary_comparison,
        "first_divergent_boundary": first_divergent_boundary,
        "cache_dir": str(cache_dir),
        "cache_before": before,
        "cache_after": after,
        "cache_changed": before != after,
        "process_wall_s": time.perf_counter() - PROCESS_STARTED,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if boundary_comparison is not None:
        for name, row in boundary_comparison.items():
            valid = row["valid_compact"]
            print(
                "UNIREC_STANDALONE_VISION_BOUNDARY "
                f"name={name} shape={valid['shape']} "
                f"max_abs={valid['max_abs']:.9g} rmse={valid['rmse']:.9g} "
                f"cosine={valid['cosine']:.9g}",
                flush=True,
            )
        print(
            "UNIREC_STANDALONE_VISION_FIRST_DIVERGENCE "
            f"boundary={first_divergent_boundary or 'none'}",
            flush=True,
        )
    print("UNIREC_STANDALONE_VISION_RESULT " + json.dumps(report), flush=True)
    print(f"OUTPUT_JSON={output}", flush=True)


if __name__ == "__main__":
    main()
