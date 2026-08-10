#!/usr/bin/env python3
"""Validate and benchmark one masked fixed-width UniRec vision-prefix bucket."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from modeling_optimized_unirec import (  # noqa: E402
    OptimizedUniRecRunner,
    import_torchair_cache_compile,
    synchronize_device,
)
from prefill_artifact import read_jsonl  # noqa: E402
from vision_atlas import UniRecVisionAtlasRuntime  # noqa: E402
from vision_prefix_crop_lab import _reconstruct_crops  # noqa: E402
from vision_static_shape import (  # noqa: E402
    _new_static_prefix_module,
    _source_hash as static_prefix_source_hash,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openocr-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--page-manifest", type=Path, required=True)
    parser.add_argument("--crop-manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bucket-width", type=int, default=960)
    parser.add_argument("--bucket-height", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--page-limit",
        type=int,
        default=0,
        help="Use only the first N page-manifest rows; zero keeps every page.",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.bucket_width % 16 or args.bucket_height % 16:
        parser.error("bucket dimensions must be divisible by 16")
    if args.batch_size < 1 or args.page_limit < 0 or args.warmup < 0 or args.repeats < 1:
        parser.error("batch size and repeats must be positive; warmup cannot be negative")
    return args


def _physical_devices() -> list[int]:
    value = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if not value:
        raise RuntimeError("source npu-setup before launching the width-bucket lab")
    devices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if 5 in devices:
        raise RuntimeError("physical NPU 5 is excluded from UniRec experiments")
    return devices


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary_ms(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "p50": statistics.median(values),
        "mean": statistics.fmean(values),
        "p90": _percentile(values, 0.9),
        "max": max(values),
    }


def _measure_ms(fn: Callable[[], Any]) -> float:
    import torch_npu

    start = torch_npu.npu.Event(enable_timing=True)
    end = torch_npu.npu.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _mask_nhwc(x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    return x * valid_mask.permute(0, 2, 3, 1)


def _masked_per_row_global_context(
    ctx: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Reproduce mean(width), then mean(height), independently per batch row."""
    row_counts = valid_mask.sum(dim=3, keepdim=True).clamp_min(1)
    row_means = (ctx * valid_mask).sum(dim=3, keepdim=True) / row_counts
    valid_rows = (valid_mask.sum(dim=3, keepdim=True) > 0).to(ctx.dtype)
    return (row_means * valid_rows).sum(dim=2, keepdim=True) / valid_rows.sum(
        dim=2, keepdim=True
    ).clamp_min(1)


def _run_masked_focal_block(
    block: nn.Module,
    x: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Run one focal block while keeping right-padding mathematically inert."""
    batch, channels, height, width = x.shape
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
        raise RuntimeError("UniRec focal modulation unexpectedly has no focal layers")
    global_context = modulation.act(
        _masked_per_row_global_context(ctx, valid_mask)
    )
    ctx_all = ctx_all + global_context * gates[:, modulation.focal_level :]
    modulator = modulation.h(ctx_all) * valid_mask
    modulated = q * modulator
    modulated = modulated.permute(0, 2, 3, 1).contiguous()
    modulated = _mask_nhwc(modulation.proj(modulated), valid_mask)
    shortcut_nhwc = shortcut.permute(0, 2, 3, 1)
    residual = shortcut_nhwc + modulated
    output = residual + block.mlp(block.norm2(residual))
    return _mask_nhwc(output, valid_mask).permute(0, 3, 1, 2).contiguous()


class MaskedWidthBucketPrefix(nn.Module):
    """Stages 0-1 over one fixed canvas with a native-width mask per row."""

    def __init__(self, runner: OptimizedUniRecRunner) -> None:
        super().__init__()
        vision = runner.model.encoder.vision_encoder
        self.stem0 = vision.patch_embed[0]
        self.stem1 = vision.patch_embed[1]
        self.pos_drop = vision.pos_drop
        self.stage0_blocks = vision.layers[0].blocks
        self.stage0_downsample = vision.layers[0].downsample
        self.stage1_blocks = vision.layers[1].blocks
        self.stage1_downsample = vision.layers[1].downsample
        if self.stage0_downsample is None or self.stage1_downsample is None:
            raise ValueError("UniRec vision stages 0 and 1 must have downsamplers")

    def forward(
        self,
        pixel_values: torch.Tensor,
        mask2: torch.Tensor,
        mask4: torch.Tensor,
        mask8: torch.Tensor,
        mask16: torch.Tensor,
    ) -> torch.Tensor:
        x = self.stem0(pixel_values) * mask2
        x = self.stem1(x) * mask4
        batch, channels, height4, width4 = x.shape
        x = self.pos_drop(x.flatten(2).transpose(1, 2))
        x = x.transpose(1, 2).reshape(batch, channels, height4, width4)
        for block in self.stage0_blocks:
            x = _run_masked_focal_block(block, x, mask4)
        x = self.stage0_downsample(x)[0]
        height8, width8 = mask8.shape[2:]
        x = x * mask8.flatten(2).transpose(1, 2)
        x = x.transpose(1, 2).reshape(batch, -1, height8, width8)
        for block in self.stage1_blocks:
            x = _run_masked_focal_block(block, x, mask8)
        x = self.stage1_downsample(x)[0]
        return x * mask16.flatten(2).transpose(1, 2)


def _make_masks(
    widths: list[int],
    *,
    bucket_height: int,
    bucket_width: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    masks = []
    for factor in (2, 4, 8, 16):
        mask = torch.zeros(
            (len(widths), 1, bucket_height // factor, bucket_width // factor),
            dtype=dtype,
            device=device,
        )
        for row, width in enumerate(widths):
            mask[row, :, :, : width // factor] = 1
        masks.append(mask)
    return tuple(masks)  # type: ignore[return-value]


def _select_rows(
    rows: list[dict[str, Any]],
    *,
    bucket_height: int,
    bucket_width: int,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[int, int], int]:
    fitting = [
        row
        for row in rows
        if int(row["prefill"]["prep"]["processed_image_size"][1]) == bucket_height
        and int(row["prefill"]["prep"]["processed_image_size"][0]) <= bucket_width
    ]
    counts: dict[int, int] = {}
    by_width: dict[int, list[dict[str, Any]]] = {}
    for row in fitting:
        width = int(row["prefill"]["prep"]["processed_image_size"][0])
        counts[width] = counts.get(width, 0) + 1
        by_width.setdefault(width, []).append(row)
    selected = [by_width[width][0] for width in sorted(by_width)]
    if len(selected) > batch_size:
        raise RuntimeError(
            f"bucket has {len(selected)} distinct widths but batch size is only {batch_size}"
        )
    remaining = [row for row in fitting if row not in selected]
    remaining.sort(
        key=lambda row: (
            -counts[int(row["prefill"]["prep"]["processed_image_size"][0])],
            int(row["page_index"]),
            int(row["crop_index"]),
        )
    )
    selected.extend(remaining[: batch_size - len(selected)])
    if len(selected) != batch_size:
        raise RuntimeError(
            f"bucket has only {len(fitting)} crops, cannot form batch {batch_size}"
        )
    return selected, counts, len(fitting)


def _extract_valid_tokens(
    output: torch.Tensor,
    *,
    row: int,
    width: int,
    bucket_height: int,
    bucket_width: int,
) -> torch.Tensor:
    height16 = bucket_height // 16
    width16 = bucket_width // 16
    return output[row].reshape(height16, width16, -1)[:, : width // 16].reshape(
        1, -1, output.shape[-1]
    )


def main() -> None:
    args = parse_args()
    physical_devices = _physical_devices()
    torch._dynamo.config.cache_size_limit = max(
        int(torch._dynamo.config.cache_size_limit), args.batch_size + 32
    )
    torch._dynamo.config.recompile_limit = max(
        int(torch._dynamo.config.recompile_limit), args.batch_size + 32
    )
    torch._dynamo.config.accumulated_cache_size_limit = max(
        int(torch._dynamo.config.accumulated_cache_size_limit), (args.batch_size + 32) * 4
    )
    torch._dynamo.config.accumulated_recompile_limit = max(
        int(torch._dynamo.config.accumulated_recompile_limit), (args.batch_size + 32) * 4
    )

    sys.path.insert(0, str(args.openocr_root.expanduser().resolve()))
    from tools.utils.opendoc_onnx_utils.utils import (  # noqa: PLC0415
        crop_margin,
        tokenize_figure_of_table,
    )

    page_rows = read_jsonl(args.page_manifest.expanduser().resolve())
    all_rows = read_jsonl(args.crop_manifest.expanduser().resolve())
    if args.page_limit:
        selected_page_indices = {
            int(row["page_index"]) for row in page_rows[: args.page_limit]
        }
        all_rows = [
            row for row in all_rows if int(row["page_index"]) in selected_page_indices
        ]
    selected_rows, width_counts, fitting_count = _select_rows(
        all_rows,
        bucket_height=args.bucket_height,
        bucket_width=args.bucket_width,
        batch_size=args.batch_size,
    )
    images = _reconstruct_crops(
        page_manifest=args.page_manifest.expanduser().resolve(),
        selected_rows=selected_rows,
        crop_margin=crop_margin,
        tokenize_figure_of_table=tokenize_figure_of_table,
    )

    cache_root = args.cache_dir.expanduser().resolve()
    runner = OptimizedUniRecRunner(
        model_path=args.model_path.expanduser().resolve(),
        device="npu:0",
        dtype="float16",
        compile_cache_dir=cache_root,
    )
    eager_runtime = UniRecVisionAtlasRuntime(runner)
    prepared = []
    widths = []
    for row in selected_rows:
        request_id = str(row["request_id"])
        inputs, metadata = runner.prepare_pil_image(
            images[request_id], image_source=request_id
        )
        expected = row["prefill"]["prep"]["processed_image_size"]
        if metadata["processed_image_size"] != expected:
            raise RuntimeError(
                f"processed-size mismatch for {request_id}: "
                f"{metadata['processed_image_size']} != {expected}"
            )
        width, height = (int(value) for value in expected)
        if height != args.bucket_height or width > args.bucket_width:
            raise RuntimeError(f"selected crop {request_id} does not fit the bucket")
        prepared.append(inputs["pixel_values"])
        widths.append(width)

    device = torch.device("npu:0")
    padded = torch.zeros(
        (args.batch_size, 3, args.bucket_height, args.bucket_width),
        dtype=runner.dtype,
        device=device,
    )
    for row, pixels in enumerate(prepared):
        padded[row : row + 1, :, :, : widths[row]].copy_(pixels)
    masks = _make_masks(
        widths,
        bucket_height=args.bucket_height,
        bucket_width=args.bucket_width,
        dtype=runner.dtype,
        device=device,
    )

    cache_compile, compile_api = import_torchair_cache_compile()
    from torch_npu.dynamo.torchair.configs.compiler_config import CompilerConfig

    config = CompilerConfig()
    config.mode.value = "max-autotune"
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    bucket_module = MaskedWidthBucketPrefix(runner).eval()
    bucket_cache_dir = cache_root / (
        f"vision_masked_width_bucket_b{args.batch_size}_"
        f"{args.bucket_width}x{args.bucket_height}_float16_src{source_hash}"
    )
    bucket_cache_dir.mkdir(parents=True, exist_ok=True)
    bucket_compiled = cache_compile(
        bucket_module.forward,
        config=config,
        dynamic=False,
        cache_dir=str(bucket_cache_dir),
        ge_cache=True,
        fullgraph=True,
    )

    exact_compiled: dict[int, Callable[[torch.Tensor], torch.Tensor]] = {}
    exact_cache_dirs: dict[int, str] = {}
    exact_source_hash = static_prefix_source_hash()
    for width in sorted(set(widths)):
        module = _new_static_prefix_module(
            runner,
            input_height=args.bucket_height,
            input_width=width,
        ).eval()
        exact_cache_dir = cache_root / (
            f"vision_static_prefix_{width}x{args.bucket_height}_"
            f"float16_src{exact_source_hash}"
        )
        exact_cache_dir.mkdir(parents=True, exist_ok=True)
        exact_compiled[width] = cache_compile(
            module.forward,
            config=config,
            dynamic=False,
            cache_dir=str(exact_cache_dir),
            ge_cache=True,
            fullgraph=True,
        )
        exact_cache_dirs[width] = str(exact_cache_dir)

    with torch.inference_mode():
        first_call_started = time.perf_counter()
        bucket_output = bucket_compiled(padded, *masks)
        synchronize_device("npu:0")
        bucket_first_call_wall_s = time.perf_counter() - first_call_started

        exact_first_call_wall_s = {}
        exact_eager_outputs = []
        for width, pixels in zip(widths, prepared):
            if str(width) not in exact_first_call_wall_s:
                started = time.perf_counter()
                exact_compiled[width](pixels)
                synchronize_device("npu:0")
                exact_first_call_wall_s[str(width)] = time.perf_counter() - started
            exact_eager_outputs.append(eager_runtime._run_prefix(pixels)[0])
        synchronize_device("npu:0")

        correctness_rows = []
        for row, (width, reference) in enumerate(zip(widths, exact_eager_outputs)):
            actual = _extract_valid_tokens(
                bucket_output,
                row=row,
                width=width,
                bucket_height=args.bucket_height,
                bucket_width=args.bucket_width,
            )
            difference = (actual - reference).abs()
            correctness_rows.append(
                {
                    "request_id": str(selected_rows[row]["request_id"]),
                    "processed_size": [width, args.bucket_height],
                    "allclose_atol_5e_2_rtol_5e_2": bool(
                        torch.allclose(actual, reference, atol=5e-2, rtol=5e-2)
                    ),
                    "max_abs": float(difference.max().item()),
                    "mean_abs": float(difference.mean().item()),
                }
            )

        def run_bucket() -> torch.Tensor:
            return bucket_compiled(padded, *masks)

        def run_exact_compiled() -> list[torch.Tensor]:
            return [
                exact_compiled[width](pixels)
                for width, pixels in zip(widths, prepared)
            ]

        def run_exact_eager() -> list[torch.Tensor]:
            return [eager_runtime._run_prefix(pixels)[0] for pixels in prepared]

        for _ in range(args.warmup):
            run_bucket()
            run_exact_compiled()
            run_exact_eager()
        synchronize_device("npu:0")

        samples = {"bucket": [], "exact_compiled": [], "exact_eager": []}
        lanes = (
            ("bucket", run_bucket),
            ("exact_compiled", run_exact_compiled),
            ("exact_eager", run_exact_eager),
        )
        for repeat_index in range(args.repeats):
            ordered = lanes if repeat_index % 2 == 0 else tuple(reversed(lanes))
            for name, fn in ordered:
                samples[name].append(_measure_ms(fn))

    timing_ms = {name: _summary_ms(values) for name, values in samples.items()}
    allclose = all(item["allclose_atol_5e_2_rtol_5e_2"] for item in correctness_rows)
    total_rows = len(all_rows)
    report = {
        "status": "ok" if allclose else "correctness_failed",
        "physical_devices": physical_devices,
        "worker_count": 1,
        "page_limit": args.page_limit,
        "compile_api": compile_api,
        "bucket": {
            "processed_size": [args.bucket_width, args.bucket_height],
            "batch_size": args.batch_size,
            "graph_count": 1,
            "cache_dir": str(bucket_cache_dir),
            "first_call_wall_s": bucket_first_call_wall_s,
            "selected_widths": widths,
            "distinct_selected_widths": sorted(set(widths)),
        },
        "coverage": {
            "crop_manifest_rows": total_rows,
            "fitting_crops": fitting_count,
            "fraction": fitting_count / total_rows,
            "width_counts": {str(width): width_counts[width] for width in sorted(width_counts)},
        },
        "correctness": {
            "all_rows_close": allclose,
            "max_abs": max(item["max_abs"] for item in correctness_rows),
            "mean_abs_max": max(item["mean_abs"] for item in correctness_rows),
            "rows": correctness_rows,
        },
        "warmup": args.warmup,
        "repeats": args.repeats,
        "timing_ms": timing_ms,
        "speed": {
            "bucket_crops_per_s": args.batch_size * 1000.0 / timing_ms["bucket"]["p50"],
            "bucket_vs_sequential_exact_compiled": (
                timing_ms["exact_compiled"]["p50"] / timing_ms["bucket"]["p50"]
            ),
            "bucket_vs_sequential_exact_eager": (
                timing_ms["exact_eager"]["p50"] / timing_ms["bucket"]["p50"]
            ),
            "bucket_per_crop_ms": timing_ms["bucket"]["p50"] / args.batch_size,
        },
        "exact_compiled": {
            "graph_count": len(exact_compiled),
            "cache_dirs": exact_cache_dirs,
            "first_call_wall_s": exact_first_call_wall_s,
        },
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("UNIREC_VISION_WIDTH_BUCKET_LAB " + json.dumps(report), flush=True)
    print(f"OUTPUT_JSON={output}", flush=True)
    if not allclose:
        raise RuntimeError("masked width bucket failed native eager validation")


if __name__ == "__main__":
    main()
