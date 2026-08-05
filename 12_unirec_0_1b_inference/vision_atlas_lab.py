#!/usr/bin/env python3
"""Guarded-atlas experiments for UniRec FocalSVTR vision stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from torch import nn


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from modeling_optimized_unirec import (  # noqa: E402
    OptimizedUniRecRunner,
    import_torchair_cache_compile,
    synchronize_device,
)


DEFAULT_MODEL = Path("/workspace/models/unirec-0.1b")
DEFAULT_TRACE = (
    REPO_ROOT
    / "tmp/12_unirec_0_1b_inference"
    / "packed_text_s1024_32p_warm_044dc53/output/recognition_trace.jsonl"
)
DEFAULT_CACHE_ROOT = REPO_ROOT / ".runtime_cache/12_unirec_0_1b_inference/vision_atlas_lab"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "tmp/12_unirec_0_1b_inference/vision_atlas_lab/result.json"
)
STAGE_FACTORS = (4, 8, 16, 32)
STAGE_CHANNELS = (96, 192, 384, 768)


@dataclass(frozen=True)
class CropShape:
    source_index: int
    request_id: str
    height: int
    width: int

    @property
    def tokens(self) -> int:
        return self.height * self.width


@dataclass(frozen=True)
class Placement:
    crop: CropShape
    member: int
    y: int
    x: int
    guard: int

    @property
    def inner_y(self) -> int:
        return self.y + self.guard

    @property
    def inner_x(self) -> int:
        return self.x + self.guard


@dataclass
class Shelf:
    y: int
    height: int
    next_x: int = 0


@dataclass
class AtlasPack:
    height: int
    width: int
    guard: int
    max_members: int
    shelves: list[Shelf]
    placements: list[Placement]

    @classmethod
    def empty(
        cls,
        *,
        height: int,
        width: int,
        guard: int,
        max_members: int,
    ) -> "AtlasPack":
        return cls(
            height=height,
            width=width,
            guard=guard,
            max_members=max_members,
            shelves=[],
            placements=[],
        )

    def try_add(self, crop: CropShape) -> bool:
        if len(self.placements) >= self.max_members:
            return False
        physical_h = crop.height + 2 * self.guard
        physical_w = crop.width + 2 * self.guard
        if physical_h > self.height or physical_w > self.width:
            return False
        selected: Shelf | None = None
        for shelf in self.shelves:
            if physical_h <= shelf.height and shelf.next_x + physical_w <= self.width:
                selected = shelf
                break
        if selected is None:
            next_y = sum(shelf.height for shelf in self.shelves)
            if next_y + physical_h > self.height:
                return False
            selected = Shelf(y=next_y, height=physical_h)
            self.shelves.append(selected)
        placement = Placement(
            crop=crop,
            member=len(self.placements),
            y=selected.y,
            x=selected.next_x,
            guard=self.guard,
        )
        selected.next_x += physical_w
        self.placements.append(placement)
        return True

    @property
    def real_tokens(self) -> int:
        return sum(item.crop.tokens for item in self.placements)


class GuardedAtlasStage(nn.Module):
    """Run one FocalSVTR stage on a masked multi-crop atlas."""

    def __init__(self, stage: nn.Module):
        super().__init__()
        self.blocks = stage.blocks

    @staticmethod
    def _mask_nhwc(x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        return x * valid_mask.permute(0, 2, 3, 1)

    @staticmethod
    def _global_context(
        ctx: torch.Tensor,
        membership: torch.Tensor,
        normalized_membership: torch.Tensor,
    ) -> torch.Tensor:
        batch, channels, height, width = ctx.shape
        if batch != 1:
            raise ValueError(f"guarded atlas requires batch size 1, got {batch}")
        flat = ctx.reshape(channels, height * width)
        means = torch.mm(
            flat,
            normalized_membership.transpose(0, 1).contiguous(),
        )
        return torch.mm(means, membership).reshape(
            1,
            channels,
            height,
            width,
        )

    def forward(
        self,
        atlas: torch.Tensor,
        valid_mask: torch.Tensor,
        membership: torch.Tensor,
        normalized_membership: torch.Tensor,
    ) -> torch.Tensor:
        x = atlas * valid_mask
        valid_nhwc = valid_mask.permute(0, 2, 3, 1)
        for block in self.blocks:
            shortcut = x
            normalized = block.norm1(x.permute(0, 2, 3, 1))
            modulation = block.modulation
            projected = modulation.f(normalized).permute(0, 3, 1, 2).contiguous()
            channels = normalized.shape[-1]
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
                ctx_all = (
                    contribution
                    if ctx_all is None
                    else ctx_all + contribution
                )
            assert ctx_all is not None
            global_context = self._global_context(
                ctx,
                membership,
                normalized_membership,
            )
            global_context = modulation.act(global_context) * valid_mask
            ctx_all = ctx_all + global_context * gates[:, modulation.focal_level :]
            modulator = modulation.h(ctx_all) * valid_mask
            modulated = q * modulator
            modulated = modulated.permute(0, 2, 3, 1).contiguous()
            modulated = self._mask_nhwc(modulation.proj(modulated), valid_mask)
            x_nhwc = shortcut.permute(0, 2, 3, 1) + modulated
            mlp_out = block.mlp(block.norm2(x_nhwc))
            x = ((x_nhwc + mlp_out) * valid_nhwc).permute(0, 3, 1, 2).contiguous()
        return x


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--stage", type=int, choices=range(4), default=2)
    parser.add_argument("--atlas-height", type=int, default=64)
    parser.add_argument("--atlas-width", type=int, default=256)
    parser.add_argument("--guard", type=int, default=6)
    parser.add_argument("--max-members", type=int, default=16)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--packing", choices=("fifo", "ffd"), default="ffd")
    parser.add_argument(
        "--execution",
        action="append",
        choices=("eager", "torchair"),
        help="Repeat to test multiple lanes; default: eager.",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    args.execution = tuple(dict.fromkeys(args.execution or ["eager"]))
    if args.atlas_height <= 0 or args.atlas_width <= 0:
        parser.error("atlas dimensions must be positive")
    if args.guard < 6:
        parser.error(
            "--guard must be at least 6 because the sequential 3x3, 5x5, "
            "and 7x7 focal kernels have a cumulative six-pixel radius"
        )
    if args.max_members <= 0 or args.limit <= 0:
        parser.error("--max-members and --limit must be positive")
    if args.warmup < 0 or args.repeats <= 0:
        parser.error("--warmup must be non-negative and --repeats positive")
    return args


def _read_shapes(path: Path, *, stage: int, limit: int) -> list[CropShape]:
    factor = STAGE_FACTORS[stage]
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    shapes = []
    for source_index, row in enumerate(rows[:limit]):
        width, height = (int(value) for value in row["processed_image_size"])
        if width % factor or height % factor:
            raise ValueError(
                f"processed size {(width, height)} is not divisible by stage factor {factor}"
            )
        shapes.append(
            CropShape(
                source_index=source_index,
                request_id=str(row["request_id"]),
                height=height // factor,
                width=width // factor,
            )
        )
    if not shapes:
        raise ValueError(f"no crop shapes found in {path}")
    return shapes


def _pack_shapes(
    shapes: list[CropShape],
    *,
    atlas_height: int,
    atlas_width: int,
    guard: int,
    max_members: int,
    policy: str,
) -> tuple[list[AtlasPack], list[CropShape]]:
    order = list(shapes)
    if policy == "ffd":
        order.sort(
            key=lambda item: (
                (item.height + 2 * guard) * (item.width + 2 * guard),
                item.height,
                item.width,
            ),
            reverse=True,
        )
    packs: list[AtlasPack] = []
    overflow: list[CropShape] = []
    for crop in order:
        if crop.height + 2 * guard > atlas_height or crop.width + 2 * guard > atlas_width:
            overflow.append(crop)
            continue
        placed = False
        for pack in packs:
            if pack.try_add(crop):
                placed = True
                break
        if not placed:
            pack = AtlasPack.empty(
                height=atlas_height,
                width=atlas_width,
                guard=guard,
                max_members=max_members,
            )
            if not pack.try_add(crop):
                raise AssertionError("fresh atlas rejected a fitting crop")
            packs.append(pack)
    return packs, overflow


def _make_inputs(
    shapes: list[CropShape],
    *,
    channels: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> dict[int, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return {
        crop.source_index: torch.randn(
            (1, crop.tokens, channels),
            dtype=torch.float32,
            generator=generator,
        ).to(device=device, dtype=dtype)
        for crop in shapes
    }


def _materialize_pack(
    pack: AtlasPack,
    inputs: dict[int, torch.Tensor],
    *,
    channels: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    atlas = torch.zeros(
        (1, channels, pack.height, pack.width),
        dtype=dtype,
        device=device,
    )
    valid_mask = torch.zeros(
        (1, 1, pack.height, pack.width),
        dtype=dtype,
        device=device,
    )
    membership = torch.zeros(
        (pack.max_members, pack.height * pack.width),
        dtype=dtype,
        device=device,
    )
    for placement in pack.placements:
        crop = placement.crop
        y, x = placement.inner_y, placement.inner_x
        feature = inputs[crop.source_index].reshape(
            1,
            crop.height,
            crop.width,
            channels,
        ).permute(0, 3, 1, 2)
        atlas[:, :, y : y + crop.height, x : x + crop.width] = feature
        valid_mask[:, :, y : y + crop.height, x : x + crop.width] = 1
        member_map = membership[placement.member].view(pack.height, pack.width)
        member_map[y : y + crop.height, x : x + crop.width] = 1
    counts = membership.sum(dim=1, keepdim=True).clamp_min(1)
    normalized_membership = membership / counts
    return atlas, valid_mask, membership, normalized_membership


def _baseline_forward(
    stage: nn.Module,
    crop: CropShape,
    value: torch.Tensor,
) -> torch.Tensor:
    x = value
    for block in stage.blocks:
        block.H = crop.height
        block.W = crop.width
        x = block(x)
    return x


def _extract_pack(
    pack: AtlasPack,
    output: torch.Tensor,
) -> dict[int, torch.Tensor]:
    extracted = {}
    for placement in pack.placements:
        crop = placement.crop
        y, x = placement.inner_y, placement.inner_x
        extracted[crop.source_index] = output[
            :,
            :,
            y : y + crop.height,
            x : x + crop.width,
        ].permute(0, 2, 3, 1).reshape(1, crop.tokens, -1).contiguous()
    return extracted


def _synchronize(device: torch.device) -> None:
    synchronize_device(str(device))


def _measure_ms(device: torch.device, fn: Callable[[], Any]) -> tuple[float, Any]:
    if device.type != "npu":
        started = time.perf_counter()
        result = fn()
        _synchronize(device)
        return (time.perf_counter() - started) * 1000.0, result
    import torch_npu

    start = torch_npu.npu.Event(enable_timing=True)
    end = torch_npu.npu.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)), result


def _compile_atlas(
    module: GuardedAtlasStage,
    *,
    cache_root: Path,
    stage: int,
    atlas_height: int,
    atlas_width: int,
    max_members: int,
    dtype_name: str,
) -> tuple[Callable[..., torch.Tensor], dict[str, Any]]:
    source = Path(__file__).read_bytes()
    source += Path(__file__).with_name("modeling_optimized_unirec.py").read_bytes()
    digest = hashlib.sha256(source).hexdigest()[:12]
    cache_dir = cache_root.expanduser().resolve() / (
        f"stage{stage}_atlas{atlas_height}x{atlas_width}_m{max_members}_"
        f"{dtype_name}_src{digest}"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    from torch_npu.dynamo.torchair.configs.compiler_config import CompilerConfig

    config = CompilerConfig()
    config.mode.value = "max-autotune"
    cache_compile, import_path = import_torchair_cache_compile()
    compiled = cache_compile(
        module.forward,
        config=config,
        dynamic=False,
        cache_dir=str(cache_dir),
        ge_cache=True,
        fullgraph=True,
    )
    return compiled, {
        "compile_api": import_path,
        "cache_dir": str(cache_dir),
        "dynamic": False,
        "fullgraph": True,
    }


def _correctness(
    reference: dict[int, torch.Tensor],
    candidate: dict[int, torch.Tensor],
) -> dict[str, Any]:
    per_crop = []
    for source_index, expected in reference.items():
        actual = candidate[source_index]
        delta = actual.float() - expected.float()
        expected_f = expected.float()
        per_crop.append(
            {
                "source_index": source_index,
                "max_abs": float(delta.abs().max().item()),
                "mean_abs": float(delta.abs().mean().item()),
                "relative_l2": float(
                    torch.linalg.vector_norm(delta).item()
                    / max(torch.linalg.vector_norm(expected_f).item(), 1e-12)
                ),
                "cosine": float(
                    torch.nn.functional.cosine_similarity(
                        expected_f.reshape(1, -1),
                        actual.float().reshape(1, -1),
                    ).item()
                ),
            }
        )
    return {
        "crops": len(per_crop),
        "worst_max_abs": max(item["max_abs"] for item in per_crop),
        "mean_of_mean_abs": statistics.fmean(item["mean_abs"] for item in per_crop),
        "worst_relative_l2": max(item["relative_l2"] for item in per_crop),
        "worst_cosine": min(item["cosine"] for item in per_crop),
        "worst_crops": sorted(
            per_crop,
            key=lambda item: item["relative_l2"],
            reverse=True,
        )[:5],
    }


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    device = torch.device("npu:0")
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    shapes = _read_shapes(
        args.trace.expanduser().resolve(),
        stage=args.stage,
        limit=args.limit,
    )
    packs, overflow = _pack_shapes(
        shapes,
        atlas_height=args.atlas_height,
        atlas_width=args.atlas_width,
        guard=args.guard,
        max_members=args.max_members,
        policy=args.packing,
    )
    packed_shapes = [placement.crop for pack in packs for placement in pack.placements]
    if not packed_shapes:
        raise RuntimeError("no crop fits the selected atlas")
    print(
        "UNIREC_VISION_ATLAS_CORPUS "
        f"stage={args.stage} crops={len(shapes)} packed={len(packed_shapes)} "
        f"overflow={len(overflow)} packs={len(packs)} "
        f"real_tokens={sum(item.tokens for item in packed_shapes)} "
        f"physical_tokens={len(packs) * args.atlas_height * args.atlas_width}",
        flush=True,
    )

    runner = OptimizedUniRecRunner(
        model_path=args.model,
        device=str(device),
        dtype=args.dtype,
        compile_cache_dir=args.cache_dir,
    )
    stage = runner.model.encoder.vision_encoder.layers[args.stage]
    atlas_module = GuardedAtlasStage(stage).eval()
    inputs = _make_inputs(
        packed_shapes,
        channels=STAGE_CHANNELS[args.stage],
        dtype=dtype,
        device=device,
        seed=args.seed,
    )
    materialized = [
        _materialize_pack(
            pack,
            inputs,
            channels=STAGE_CHANNELS[args.stage],
            dtype=dtype,
            device=device,
        )
        for pack in packs
    ]

    print("UNIREC_VISION_ATLAS_REFERENCE_BEGIN", flush=True)
    with torch.inference_mode():
        reference = {
            crop.source_index: _baseline_forward(
                stage,
                crop,
                inputs[crop.source_index],
            )
            for crop in packed_shapes
        }
    _synchronize(device)
    print("UNIREC_VISION_ATLAS_REFERENCE_END", flush=True)

    def baseline_corpus() -> list[torch.Tensor]:
        return [
            _baseline_forward(stage, crop, inputs[crop.source_index])
            for crop in packed_shapes
        ]

    for _ in range(args.warmup):
        with torch.inference_mode():
            baseline_corpus()
        _synchronize(device)
    baseline_samples = []
    for _ in range(args.repeats):
        with torch.inference_mode():
            elapsed_ms, _ = _measure_ms(device, baseline_corpus)
        baseline_samples.append(elapsed_ms)
    real_tokens = sum(item.tokens for item in packed_shapes)
    baseline_median_ms = statistics.median(baseline_samples)
    baseline_result = {
        "calls": len(packed_shapes),
        "samples_ms": baseline_samples,
        "median_ms": baseline_median_ms,
        "effective_tokens_per_s": real_tokens / (baseline_median_ms / 1000.0),
    }
    print(
        "UNIREC_VISION_ATLAS_BASELINE "
        + json.dumps(baseline_result, sort_keys=True),
        flush=True,
    )

    lane_results = []
    for execution in args.execution:
        if execution == "eager":
            run_atlas = atlas_module.forward
            compile_metadata = None
        else:
            run_atlas, compile_metadata = _compile_atlas(
                atlas_module,
                cache_root=args.cache_dir,
                stage=args.stage,
                atlas_height=args.atlas_height,
                atlas_width=args.atlas_width,
                max_members=args.max_members,
                dtype_name=args.dtype,
            )
            print(
                "UNIREC_VISION_ATLAS_FIRST_GRAPH_BEGIN "
                + json.dumps(compile_metadata, sort_keys=True),
                flush=True,
            )
            started = time.perf_counter()
            with torch.inference_mode():
                run_atlas(*materialized[0])
            _synchronize(device)
            compile_metadata["first_call_wall_s"] = time.perf_counter() - started
            print(
                "UNIREC_VISION_ATLAS_FIRST_GRAPH_END "
                + json.dumps(compile_metadata, sort_keys=True),
                flush=True,
            )

        def atlas_corpus() -> list[torch.Tensor]:
            return [run_atlas(*values) for values in materialized]

        for _ in range(args.warmup):
            with torch.inference_mode():
                atlas_corpus()
            _synchronize(device)
        with torch.inference_mode():
            candidate_outputs = atlas_corpus()
        _synchronize(device)
        candidate: dict[int, torch.Tensor] = {}
        for pack, output in zip(packs, candidate_outputs):
            candidate.update(_extract_pack(pack, output))
        correctness = _correctness(reference, candidate)
        samples = []
        for _ in range(args.repeats):
            with torch.inference_mode():
                elapsed_ms, _ = _measure_ms(device, atlas_corpus)
            samples.append(elapsed_ms)
        median_ms = statistics.median(samples)
        physical_tokens = len(packs) * args.atlas_height * args.atlas_width
        result = {
            "execution": execution,
            "calls": len(packs),
            "samples_ms": samples,
            "median_ms": median_ms,
            "real_tokens": real_tokens,
            "physical_tokens": physical_tokens,
            "fill_fraction": real_tokens / physical_tokens,
            "physical_tokens_per_s": physical_tokens / (median_ms / 1000.0),
            "effective_tokens_per_s": real_tokens / (median_ms / 1000.0),
            "speedup_vs_per_crop": baseline_median_ms / median_ms,
            "correctness": correctness,
            "compile": compile_metadata,
        }
        lane_results.append(result)
        print(
            "UNIREC_VISION_ATLAS_LANE " + json.dumps(result, sort_keys=True),
            flush=True,
        )

    payload = {
        "schema_version": 1,
        "git_commit": _git_commit(),
        "model": str(args.model.expanduser().resolve()),
        "trace": str(args.trace.expanduser().resolve()),
        "device": str(device),
        "dtype": args.dtype,
        "torch": torch.__version__,
        "stage": args.stage,
        "stage_depth": len(stage.blocks),
        "stage_channels": STAGE_CHANNELS[args.stage],
        "stage_downsample_factor": STAGE_FACTORS[args.stage],
        "atlas": {
            "height": args.atlas_height,
            "width": args.atlas_width,
            "guard": args.guard,
            "max_members": args.max_members,
            "packing": args.packing,
            "packs": len(packs),
            "member_histogram": {
                str(value): sum(len(pack.placements) == value for pack in packs)
                for value in sorted({len(pack.placements) for pack in packs})
            },
        },
        "corpus": {
            "requested": len(shapes),
            "packed": len(packed_shapes),
            "overflow": [item.request_id for item in overflow],
            "real_tokens": real_tokens,
        },
        "baseline": baseline_result,
        "lanes": lane_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"UNIREC_VISION_ATLAS_DONE output={args.output}", flush=True)


if __name__ == "__main__":
    main()
