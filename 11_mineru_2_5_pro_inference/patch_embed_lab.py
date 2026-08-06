#!/usr/bin/env python3
"""Compare exact MinerU patch-embedding implementations on one real page."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image

from local_modeling_mineru import LocalMinerU2_5ForConditionalGeneration
from prefill_timing import PrefillDeviceTimeline
from run_transformers_recognition_smoke import configure_npu, synchronize
from vision_prefill_compile import (
    DEFAULT_VISION_BUCKETS,
    MinerUVisionPrefillRuntime,
)


DEFAULT_MODEL = Path("/workspace/models/MinerU2.5-Pro-2605-1.2B")
DEFAULT_DATASET_JSON = Path("/workspace/datasets/OmniDocBench/OmniDocBench.json")
DEFAULT_IMAGES_DIR = Path("/workspace/datasets/OmniDocBench/images")
DEFAULT_CACHE_DIR = Path(
    ".runtime_cache/11_mineru_2_5_pro_inference/vision_prefill_b1_fp16"
)
DEFAULT_OUTPUT = Path(
    "tmp/11_mineru_2_5_pro_inference/patch_embed_lab/layout_1036/result.json"
)
MODES = ("conv3d", "flat_linear", "temporal_fused_linear")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset-json", type=Path, default=DEFAULT_DATASET_JSON)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--page-index", type=int, default=0)
    parser.add_argument("--layout-size", type=int, nargs=2, default=(1036, 1036))
    parser.add_argument(
        "--patch-token-sweep",
        type=str,
        help=(
            "Comma-separated raw patch-token counts. When set, benchmark only "
            "Conv3D and the production flat-linear replacement on synthetic "
            "fp16 patch rows, without running the vision transformer."
        ),
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--patch-warmup", type=int, default=10)
    parser.add_argument("--patch-blocks", type=int, default=7)
    parser.add_argument("--patch-calls-per-block", type=int, default=20)
    parser.add_argument("--full-warmup", type=int, default=5)
    parser.add_argument("--full-samples", type=int, default=10)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.page_index < 0:
        parser.error("--page-index must be non-negative")
    if min(
        args.patch_warmup,
        args.patch_blocks,
        args.patch_calls_per_block,
        args.full_warmup,
        args.full_samples,
    ) <= 0:
        parser.error("all warm-up and measurement counts must be positive")
    if args.patch_token_sweep:
        try:
            args.patch_token_sweep = [
                int(value) for value in args.patch_token_sweep.split(",")
            ]
        except ValueError:
            parser.error("--patch-token-sweep must contain comma-separated integers")
        if not args.patch_token_sweep or min(args.patch_token_sweep) <= 0:
            parser.error("--patch-token-sweep values must be positive")
    return args


def summary(values: list[float]) -> dict[str, float]:
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "p50": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def compare(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    left_float = left.float()
    right_float = right.float()
    delta = (right_float - left_float).abs()
    left_flat = left_float.flatten()
    right_flat = right_float.flatten()
    return {
        "shape": list(left.shape),
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
        "relative_l2": float(
            torch.linalg.vector_norm(right_flat - left_flat).item()
            / max(torch.linalg.vector_norm(left_flat).item(), 1e-12)
        ),
        "cosine": float(F.cosine_similarity(left_flat, right_flat, dim=0).item()),
        "nonfinite_left": int((~torch.isfinite(left)).sum().item()),
        "nonfinite_right": int((~torch.isfinite(right)).sum().item()),
    }


class PatchEmbedVariants(nn.Module):
    """Lab-only exact algebraic alternatives to the checkpoint Conv3D."""

    def __init__(self, source: nn.Module):
        super().__init__()
        self.proj = source.proj
        self.patch_size = int(source.patch_size)
        self.temporal_patch_size = int(source.temporal_patch_size)
        self.in_channels = int(source.in_channels)
        self.embed_dim = int(source.embed_dim)
        if self.temporal_patch_size != 2:
            raise ValueError(
                "temporal fusion requires temporal_patch_size=2, got "
                f"{self.temporal_patch_size}"
            )
        fused = (
            self.proj.weight.detach()[:, :, 0]
            + self.proj.weight.detach()[:, :, 1]
        ).contiguous()
        self.register_buffer("temporal_fused_weight", fused, persistent=False)
        self.mode = "conv3d"

    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown patch mode {mode!r}; expected {MODES}")
        self.mode = mode

    def _patches(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        ).to(dtype=self.proj.weight.dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        patches = self._patches(hidden_states)
        if self.mode == "conv3d":
            return self.proj(patches).view(-1, self.embed_dim)
        if self.mode == "flat_linear":
            return F.linear(
                patches.flatten(1),
                self.proj.weight.flatten(1),
            )
        return F.linear(
            patches[:, :, 0].flatten(1),
            self.temporal_fused_weight.flatten(1),
        )


def device_block(fn: Callable[[], torch.Tensor], calls: int) -> tuple[torch.Tensor, float]:
    import torch_npu

    start = torch_npu.npu.Event(enable_timing=True)
    end = torch_npu.npu.Event(enable_timing=True)
    start.record()
    output = None
    for _ in range(calls):
        output = fn()
    end.record()
    end.synchronize()
    if output is None:
        raise RuntimeError("device block produced no output")
    return output, float(start.elapsed_time(end)) / calls


def run_full(
    model: LocalMinerU2_5ForConditionalGeneration,
    pixel_values: torch.Tensor,
    grid: torch.Tensor,
) -> tuple[torch.Tensor, float, dict[str, float]]:
    timeline = PrefillDeviceTimeline(model.device)
    started = time.perf_counter()
    output = model.get_image_features(
        pixel_values,
        grid,
        device_timeline=timeline,
    )
    stages = timeline.resolve()
    return output, time.perf_counter() - started, stages


def run_patch_sweep(
    model: LocalMinerU2_5ForConditionalGeneration,
    token_counts: list[int],
    args: argparse.Namespace,
) -> None:
    original = model.visual.patch_embed
    variants = PatchEmbedVariants(original).to(device=model.device, dtype=model.dtype)
    rows: list[dict[str, Any]] = []
    feature_width = (
        variants.in_channels
        * variants.temporal_patch_size
        * variants.patch_size
        * variants.patch_size
    )
    torch.manual_seed(17)
    for token_count in token_counts:
        hidden_states = torch.randn(
            token_count,
            feature_width,
            device=model.device,
            dtype=model.dtype,
        )
        mode_outputs: dict[str, torch.Tensor] = {}
        mode_ms: dict[str, dict[str, float]] = {}
        for mode in ("conv3d", "flat_linear"):
            variants.set_mode(mode)
            for _ in range(args.patch_warmup):
                variants(hidden_states)
            synchronize()
            block_ms: list[float] = []
            output = None
            for _ in range(args.patch_blocks):
                output, milliseconds = device_block(
                    lambda: variants(hidden_states),
                    args.patch_calls_per_block,
                )
                block_ms.append(milliseconds)
            if output is None:
                raise RuntimeError("patch sweep produced no output")
            mode_outputs[mode] = output.clone()
            mode_ms[mode] = summary(block_ms)
        parity = compare(mode_outputs["conv3d"], mode_outputs["flat_linear"])
        conv_ms = mode_ms["conv3d"]["p50"]
        linear_ms = mode_ms["flat_linear"]["p50"]
        row = {
            "raw_tokens": token_count,
            "conv3d_ms": mode_ms["conv3d"],
            "flat_linear_ms": mode_ms["flat_linear"],
            "conv3d_raw_tokens_per_s": token_count / (conv_ms / 1000),
            "flat_linear_raw_tokens_per_s": token_count / (linear_ms / 1000),
            "speedup": conv_ms / linear_ms,
            "parity": parity,
        }
        rows.append(row)
        print(
            f"[sweep] tokens={token_count} conv3d_ms={conv_ms:.6f} "
            f"flat_linear_ms={linear_ms:.6f} speedup={row['speedup']:.2f} "
            f"max_abs={parity['max_abs']:.6g}",
            flush=True,
        )
    payload = {
        "schema_version": 1,
        "kind": "mineru_patch_embed_token_sweep",
        "device": "Ascend NPU",
        "dtype": "fp16",
        "model": str(args.model.expanduser().resolve()),
        "feature_width": feature_width,
        "measurement": {
            "patch_warmup": args.patch_warmup,
            "patch_blocks": args.patch_blocks,
            "patch_calls_per_block": args.patch_calls_per_block,
        },
        "rows": rows,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[output] {output}", flush=True)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    configure_npu()
    from transformers import AutoProcessor

    model_dir = args.model.expanduser().resolve()
    processor = AutoProcessor.from_pretrained(
        model_dir,
        use_fast=True,
        local_files_only=True,
    )
    model = LocalMinerU2_5ForConditionalGeneration.from_pretrained(
        model_dir,
        dtype=torch.float16,
        device="npu:0",
    )
    if args.patch_token_sweep:
        run_patch_sweep(model, args.patch_token_sweep, args)
        return
    model.set_vision_attention_impl("prompt_flash_attention")
    runtime = MinerUVisionPrefillRuntime(
        model.visual,
        buckets=DEFAULT_VISION_BUCKETS,
        cache_root=args.cache_dir,
        model_dir=model_dir,
        device=model.device,
        dtype=model.dtype,
    )
    model.set_vision_prefill_runtime(runtime)

    samples = json.loads(args.dataset_json.read_text(encoding="utf-8"))
    sample = samples[args.page_index]
    image_name = Path(sample["page_info"]["image_path"]).name
    image_path = args.images_dir / image_name
    with Image.open(image_path) as source:
        image = source.convert("RGB").resize(
            tuple(args.layout_size),
            Image.Resampling.BICUBIC,
        )
    inputs = processor.image_processor(images=[image], return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device=model.device, dtype=model.dtype)
    grid = inputs["image_grid_thw"].to(device=model.device)
    synchronize()

    original = model.visual.patch_embed
    variants = PatchEmbedVariants(original).to(device=model.device, dtype=model.dtype)
    model.visual.patch_embed = variants
    patches = variants._patches(pixel_values)
    temporal_delta = (patches[:, :, 0] - patches[:, :, 1]).abs()
    temporal = {
        "exact_equal": bool(torch.equal(patches[:, :, 0], patches[:, :, 1])),
        "max_abs": float(temporal_delta.max().item()),
        "mean_abs": float(temporal_delta.mean().item()),
    }
    print(f"[temporal] {json.dumps(temporal, sort_keys=True)}", flush=True)
    if not temporal["exact_equal"]:
        raise RuntimeError("real image temporal slices are not exactly identical")

    results: dict[str, Any] = {}
    patch_outputs: dict[str, torch.Tensor] = {}
    full_outputs: dict[str, torch.Tensor] = {}
    for mode in MODES:
        variants.set_mode(mode)
        print(f"[{mode}] patch warmup={args.patch_warmup}", flush=True)
        patch_output = None
        for _ in range(args.patch_warmup):
            patch_output = variants(pixel_values)
        synchronize()
        patch_ms: list[float] = []
        for block in range(args.patch_blocks):
            patch_output, milliseconds = device_block(
                lambda: variants(pixel_values),
                args.patch_calls_per_block,
            )
            patch_ms.append(milliseconds)
            print(
                f"[{mode}] patch block={block + 1}/{args.patch_blocks} "
                f"ms={milliseconds:.6f}",
                flush=True,
            )
        if patch_output is None:
            raise RuntimeError("patch measurement produced no output")
        patch_outputs[mode] = patch_output.clone()

        print(f"[{mode}] full warmup={args.full_warmup}", flush=True)
        full_output = None
        for _ in range(args.full_warmup):
            full_output, _wall, _stages = run_full(model, pixel_values, grid)
        synchronize()
        wall_samples: list[float] = []
        stage_samples: dict[str, list[float]] = defaultdict(list)
        for index in range(args.full_samples):
            full_output, wall_s, stages = run_full(model, pixel_values, grid)
            wall_samples.append(wall_s)
            for name, value in stages.items():
                stage_samples[name].append(float(value))
            print(
                f"[{mode}] full sample={index + 1}/{args.full_samples} "
                f"wall_ms={wall_s * 1000:.3f}",
                flush=True,
            )
        if full_output is None:
            raise RuntimeError("full measurement produced no output")
        full_outputs[mode] = full_output.clone()
        patch_median = statistics.median(patch_ms)
        results[mode] = {
            "patch_ms": summary(patch_ms),
            "patch_raw_tokens_per_s": int(pixel_values.shape[0]) / (patch_median / 1000),
            "full_wall_s": summary(wall_samples),
            "device_stages_s": {
                name: summary(values) for name, values in sorted(stage_samples.items())
            },
        }

    baseline = "conv3d"
    for mode in MODES:
        results[mode]["patch_parity_vs_conv3d"] = compare(
            patch_outputs[baseline], patch_outputs[mode]
        )
        results[mode]["final_parity_vs_conv3d"] = compare(
            full_outputs[baseline], full_outputs[mode]
        )
        results[mode]["patch_speedup_vs_conv3d"] = (
            results[baseline]["patch_ms"]["p50"]
            / results[mode]["patch_ms"]["p50"]
        )
        results[mode]["full_speedup_vs_conv3d"] = (
            results[baseline]["full_wall_s"]["p50"]
            / results[mode]["full_wall_s"]["p50"]
        )

    payload = {
        "schema_version": 1,
        "kind": "mineru_patch_embed_lab",
        "device": "Ascend NPU",
        "dtype": "fp16",
        "model": str(model_dir),
        "page_index": args.page_index,
        "image": str(image_path),
        "layout_size_wh": list(args.layout_size),
        "grid_thw": inputs["image_grid_thw"].tolist(),
        "raw_tokens": int(pixel_values.shape[0]),
        "physical_transformer_tokens": runtime.metadata()["buckets"][-1],
        "temporal_slices": temporal,
        "measurement": {
            "patch_warmup": args.patch_warmup,
            "patch_blocks": args.patch_blocks,
            "patch_calls_per_block": args.patch_calls_per_block,
            "full_warmup": args.full_warmup,
            "full_samples": args.full_samples,
        },
        "results": results,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                mode: {
                    "patch_ms": results[mode]["patch_ms"]["p50"],
                    "patch_speedup": results[mode]["patch_speedup_vs_conv3d"],
                    "full_ms": results[mode]["full_wall_s"]["p50"] * 1000,
                    "full_speedup": results[mode]["full_speedup_vs_conv3d"],
                    "patch_cosine": results[mode]["patch_parity_vs_conv3d"]["cosine"],
                    "final_cosine": results[mode]["final_parity_vs_conv3d"]["cosine"],
                }
                for mode in MODES
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"[output] {output}", flush=True)


if __name__ == "__main__":
    main()
