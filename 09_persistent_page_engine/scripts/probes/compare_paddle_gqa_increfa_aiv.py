#!/usr/bin/env python3
"""Correctness and latency probe for the independent GQA AIV operator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable, Sequence

import torch

from paddleocr_vl.model.compile_utils import import_torchair
from paddleocr_vl.model.gqa_increfa_aiv import (
    GE_OP_NAME,
    MIN_KV_LENGTH_FOR_48_CORES,
    PYTORCH_OP_NAME,
    gqa_incre_flash_attention_aiv,
    register_gqa_increfa_aiv_converter,
)

QUERY_HEADS = 16
KV_HEADS = 2
HEAD_DIM = 128
SCALE_VALUE = 1.0 / math.sqrt(HEAD_DIM)
FP16_ATOL = 5e-4
FP16_RTOL = 5e-3
REFERENCE_ATOL = 2e-3
REFERENCE_RTOL = 1e-2
EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXTENSION_ROOT = EXPERIMENT_ROOT / "custom_ops/paddle_gqa_increfa_aiv/pytorch_extension"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("eager", "torchair"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--extension-root", type=Path, default=DEFAULT_EXTENSION_ROOT)
    parser.add_argument("--kv-length", type=int, required=True)
    parser.add_argument("--valid-kv-length", type=int)
    parser.add_argument("--vector-core-count", type=int, required=True)
    parser.add_argument(
        "--experimental-grouped-serial-control",
        action="store_true",
        help=(
            "Allow the separately packaged two-core grouped-serial research "
            "variant. Eager-only; the supported 16-core resource attribute is "
            "retained while grouped tiling launches two AIV blocks. This does "
            "not change the production graph op."
        ),
    )
    parser.add_argument(
        "--experimental-grouped-half-control",
        action="store_true",
        help=(
            "Allow the separately packaged four-core half-group research "
            "variant. Eager-only; the supported 16-core resource attribute is "
            "retained while grouped tiling launches four AIV blocks. This does "
            "not change the production graph op."
        ),
    )
    parser.add_argument(
        "--experimental-split-k32-control",
        action="store_true",
        help=(
            "Identify the separately packaged forced two-way split-K research "
            "variant. It requires vector_core_count=32 and does not change the "
            "production graph operator package."
        ),
    )
    parser.add_argument(
        "--experimental-split-k32-pairwise-sync-control",
        action="store_true",
        help=(
            "Identify the separately packaged two-way split-K variant whose "
            "even worker reduces after waiting only for its paired producer. "
            "It requires vector_core_count=32."
        ),
    )
    parser.add_argument(
        "--experimental-split-k32-two-way-reduce-control",
        action="store_true",
        help=(
            "Identify the separately packaged two-way split-K variant with a "
            "specialized two-part softmax/output reduction. It requires "
            "vector_core_count=32."
        ),
    )
    parser.add_argument(
        "--experimental-split-k48-control",
        action="store_true",
        help=(
            "Identify the separately packaged forced three-way split-K "
            "research variant. It requires vector_core_count=48 and does "
            "not change the production graph operator package."
        ),
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--blocks", type=int, default=7)
    parser.add_argument("--repeats-per-block", type=int, default=200)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument(
        "--profile-metric",
        choices=("pipe", "memory", "l2", "memory_access"),
        default="pipe",
    )
    parser.add_argument("--profile-calls", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260809)
    args = parser.parse_args(argv)
    if args.kv_length <= 0:
        parser.error("--kv-length must be positive")
    if (
        args.valid_kv_length is not None
        and not 1 <= args.valid_kv_length <= args.kv_length
    ):
        parser.error("--valid-kv-length must be in [1, --kv-length]")
    if (
        int(args.experimental_grouped_serial_control)
        + int(args.experimental_grouped_half_control)
        + int(args.experimental_split_k32_control)
        + int(args.experimental_split_k32_pairwise_sync_control)
        + int(args.experimental_split_k32_two_way_reduce_control)
        + int(args.experimental_split_k48_control)
        > 1
    ):
        parser.error("select at most one experimental control")
    if (
        args.experimental_grouped_serial_control
        or args.experimental_grouped_half_control
    ):
        if args.backend != "eager" or args.vector_core_count != QUERY_HEADS:
            option = (
                "--experimental-grouped-serial-control"
                if args.experimental_grouped_serial_control
                else "--experimental-grouped-half-control"
            )
            parser.error(
                f"{option} requires --backend eager "
                "and --vector-core-count 16; grouped tiling, not this resource "
                "attribute, reduces the launched AIV block count"
            )
    elif not QUERY_HEADS <= args.vector_core_count <= 48:
        parser.error("--vector-core-count must be in [16, 48]")
    if args.experimental_split_k32_control and args.vector_core_count != 32:
        parser.error(
            "--experimental-split-k32-control requires --vector-core-count 32"
        )
    if (
        args.experimental_split_k32_pairwise_sync_control
        and args.vector_core_count != 32
    ):
        parser.error(
            "--experimental-split-k32-pairwise-sync-control requires "
            "--vector-core-count 32"
        )
    if (
        args.experimental_split_k32_two_way_reduce_control
        and args.vector_core_count != 32
    ):
        parser.error(
            "--experimental-split-k32-two-way-reduce-control requires "
            "--vector-core-count 32"
        )
    if args.experimental_split_k48_control and args.vector_core_count != 48:
        parser.error(
            "--experimental-split-k48-control requires --vector-core-count 48"
        )
    if (
        args.experimental_split_k48_control
        and args.kv_length < MIN_KV_LENGTH_FOR_48_CORES
    ):
        parser.error(
            "--experimental-split-k48-control requires --kv-length >= "
            f"{MIN_KV_LENGTH_FOR_48_CORES}; shorter three-way partitions "
            "stall the existing split-K kernel"
        )
    if args.warmup < 0 or args.blocks <= 0 or args.repeats_per_block <= 0:
        parser.error("invalid timing counts")
    if args.profile_calls <= 0:
        parser.error("--profile-calls must be positive")
    if args.profile_dir is not None and args.backend != "eager":
        parser.error("--profile-dir currently supports only --backend eager")
    if args.backend == "torchair" and args.cache_root is None:
        parser.error("--cache-root is required for TorchAir")
    return args


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return float(ordered[index])


def timing_summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "minimum": float(min(values)),
        "maximum": float(max(values)),
        "p90": percentile(values, 0.90),
    }


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def fp32_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Independent CPU FP32 BNSD GQA reference for the probe contract."""
    group_size = QUERY_HEADS // KV_HEADS
    q_fp32 = q.float().cpu()
    k_fp32 = k.float().cpu().repeat_interleave(group_size, dim=1)
    v_fp32 = v.float().cpu().repeat_interleave(group_size, dim=1)
    mask_cpu = mask.cpu().expand(
        q_fp32.shape[0], QUERY_HEADS, q_fp32.shape[2], k_fp32.shape[2]
    )
    scores = torch.matmul(q_fp32, k_fp32.transpose(-2, -1)) * SCALE_VALUE
    scores = scores.masked_fill(mask_cpu, torch.finfo(torch.float32).min)
    probabilities = torch.softmax(scores, dim=-1)
    return torch.matmul(probabilities, v_fp32).contiguous()


def time_step(
    step: Callable[..., torch.Tensor],
    inputs: tuple[torch.Tensor, ...],
    *,
    warmup: int,
    blocks: int,
    repeats_per_block: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    output = None
    for _ in range(warmup):
        output = step(*inputs)
    torch.npu.synchronize()
    event_us: list[float] = []
    host_us: list[float] = []
    for _ in range(blocks):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        host_start = time.perf_counter()
        start.record()
        for _ in range(repeats_per_block):
            output = step(*inputs)
        end.record()
        end.synchronize()
        event_us.append(float(start.elapsed_time(end)) * 1000.0 / repeats_per_block)
        host_us.append((time.perf_counter() - host_start) * 1e6 / repeats_per_block)
    assert output is not None
    return output, {
        "warmup_calls": warmup,
        "blocks": blocks,
        "repeats_per_block": repeats_per_block,
        "npu_event_us_per_call": event_us,
        "host_wall_us_per_call": host_us,
        "npu_event_us_per_call_summary": timing_summary(event_us),
        "host_wall_us_per_call_summary": timing_summary(host_us),
    }


def compile_step(module: torch.nn.Module, cache_dir: Path) -> Callable[..., torch.Tensor]:
    torchair, CompilerConfig = import_torchair()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return torchair.inference.cache_compile(
        module.forward,
        config=CompilerConfig(),
        dynamic=False,
        cache_dir=str(cache_dir),
        ge_cache=True,
    )


def profile_step(
    step: Callable[..., torch.Tensor],
    inputs: tuple[torch.Tensor, ...],
    *,
    output_dir: Path,
    metric: str,
    calls: int,
) -> dict[str, Any]:
    """Capture a bounded custom-op-only torch-npu profile."""
    import torch_npu.profiler as npu_prof

    metrics = {
        "pipe": npu_prof.AiCMetrics.PipeUtilization,
        "memory": npu_prof.AiCMetrics.Memory,
        "l2": npu_prof.AiCMetrics.L2Cache,
        "memory_access": npu_prof.AiCMetrics.MemoryAccess,
    }
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"profile directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    step(*inputs)
    torch.npu.synchronize()
    with npu_prof.profile(
        activities=[npu_prof.ProfilerActivity.CPU, npu_prof.ProfilerActivity.NPU],
        schedule=npu_prof.schedule(wait=0, warmup=0, active=1, repeat=1),
        experimental_config=npu_prof._ExperimentalConfig(
            profiler_level=npu_prof.ProfilerLevel.Level1,
            aic_metrics=metrics[metric],
            l2_cache=metric == "l2",
            export_type=npu_prof.ExportType.Text,
        ),
        on_trace_ready=npu_prof.tensorboard_trace_handler(
            str(output_dir), analyse_flag=True
        ),
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
    ) as profiler:
        for _ in range(calls):
            step(*inputs)
        torch.npu.synchronize()
        profiler.step()
    return {
        "path": str(output_dir),
        "metric": metric,
        "custom_calls": calls,
        "profiled_lane": "custom_only",
        "timing_interpretation": "profile_overhead_excluded_from_clean_timing",
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    import torch_npu

    if not torch.npu.is_available():
        raise RuntimeError("an Ascend NPU is required")
    torch.npu.set_compile_mode(jit_compile=False)
    device = torch.device("npu:0")

    def stock(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch_npu.npu_incre_flash_attention(
            q, k, v, atten_mask=mask, actual_seq_lengths=None,
            num_heads=QUERY_HEADS, num_key_value_heads=KV_HEADS,
            input_layout="BNSD", scale_value=SCALE_VALUE, inner_precise=1,
        )

    eager_identity: dict[str, Any] = {}
    if args.backend == "eager":
        sys.path.insert(0, str(args.extension_root.resolve()))
        from paddle_gqa_increfa_aiv_eager import (
            ACLNN_OP_NAME,
            EXTENSION_PATH,
            PYTORCH_OP_NAME as EAGER_OP_NAME,
            paddle_gqa_incre_flash_attention_aiv_eager,
        )

        def custom(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            return paddle_gqa_incre_flash_attention_aiv_eager(
                q, k, v, mask, num_heads=QUERY_HEADS,
                num_key_value_heads=KV_HEADS, scale_value=SCALE_VALUE,
                inner_precise=1, vector_core_count=args.vector_core_count,
            )

        eager_identity = {
            "pytorch_eager": EAGER_OP_NAME,
            "aclnn": ACLNN_OP_NAME,
            "extension_so": str(EXTENSION_PATH),
            "dispatch_table": torch._C._dispatch_dump_table(EAGER_OP_NAME),
        }
        stock_step: Callable[..., torch.Tensor] = stock
        custom_step: Callable[..., torch.Tensor] = custom
    else:
        register_gqa_increfa_aiv_converter()

        class Stock(torch.nn.Module):
            def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
                return stock(q, k, v, mask)

        class Custom(torch.nn.Module):
            def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
                return gqa_incre_flash_attention_aiv(
                    q, k, v, mask, num_heads=QUERY_HEADS,
                    num_key_value_heads=KV_HEADS, scale_value=SCALE_VALUE,
                    inner_precise=1, vector_core_count=args.vector_core_count,
                )

        assert args.cache_root is not None
        stock_step = compile_step(Stock(), args.cache_root / "stock")
        custom_step = compile_step(Custom(), args.cache_root / "custom")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    q = torch.randn((1, QUERY_HEADS, 1, HEAD_DIM), generator=generator, dtype=torch.float16).to(device)
    k = torch.randn((1, KV_HEADS, args.kv_length, HEAD_DIM), generator=generator, dtype=torch.float16).to(device)
    v = torch.randn((1, KV_HEADS, args.kv_length, HEAD_DIM), generator=generator, dtype=torch.float16).to(device)
    mask = torch.zeros((1, 1, 1, args.kv_length), dtype=torch.bool, device=device)
    if args.valid_kv_length is not None:
        mask[..., args.valid_kv_length:] = True
    inputs = (q, k, v, mask)

    first_call_s: dict[str, float] = {}
    outputs: dict[str, torch.Tensor] = {}
    timings: dict[str, Any] = {}
    with torch.inference_mode():
        # Run the independent operator first.  An earlier GQA kernel left most
        # output heads unwritten; running stock first let the NPU allocator reuse
        # stock-filled memory and briefly hid that bug in the eager lane.
        for name, step in (("custom", custom_step), ("stock", stock_step)):
            started = time.perf_counter()
            output = step(*inputs)
            torch.npu.synchronize()
            first_call_s[name] = time.perf_counter() - started
            output, timings[name] = time_step(
                step, inputs, warmup=args.warmup,
                blocks=args.blocks, repeats_per_block=args.repeats_per_block,
            )
            outputs[name] = output.float().cpu().contiguous()

    profile_metadata = None
    if args.profile_dir is not None:
        with torch.inference_mode():
            profile_metadata = profile_step(
                custom_step,
                inputs,
                output_dir=args.profile_dir,
                metric=args.profile_metric,
                calls=args.profile_calls,
            )

    reference = fp32_reference(q, k, v, mask)
    difference = (outputs["stock"] - outputs["custom"]).abs()
    stock_reference_difference = (outputs["stock"] - reference).abs()
    custom_reference_difference = (outputs["custom"] - reference).abs()
    exact = bool(torch.equal(outputs["stock"], outputs["custom"]))
    fp16_close = bool(
        torch.allclose(
            outputs["stock"], outputs["custom"], atol=FP16_ATOL, rtol=FP16_RTOL
        )
    )
    stock_reference_close = bool(
        torch.allclose(
            outputs["stock"], reference, atol=REFERENCE_ATOL, rtol=REFERENCE_RTOL
        )
    )
    custom_reference_close = bool(
        torch.allclose(
            outputs["custom"], reference, atol=REFERENCE_ATOL, rtol=REFERENCE_RTOL
        )
    )
    required_checks_passed = (
        fp16_close and stock_reference_close and custom_reference_close
    )
    per_head_max_abs = difference.amax(dim=(0, 2, 3)).tolist()
    per_head_mean_abs = difference.mean(dim=(0, 2, 3)).tolist()
    per_head_custom_absmax = outputs["custom"].abs().amax(dim=(0, 2, 3)).tolist()
    pairwise_head_mean_abs = (
        outputs["custom"].transpose(0, 1)[:, None]
        - outputs["stock"].transpose(0, 1)[None, :]
    ).abs().mean(dim=(2, 3, 4))
    pairwise_best_mean_abs, pairwise_best_stock_head = pairwise_head_mean_abs.min(dim=1)
    result = {
        "schema_version": 1,
        "kind": "separate_paddle_gqa_increfa_aiv_comparison",
        "operator_identity": {
            "pytorch_graph": PYTORCH_OP_NAME,
            "ge": GE_OP_NAME,
            "stock_reference": "torch_npu.npu_incre_flash_attention",
            "same_name_override": False,
            **eager_identity,
        },
        "backend": args.backend,
        "experiment_variant": (
            "grouped_serial_control"
            if args.experimental_grouped_serial_control
            else (
                "grouped_half_control"
                if args.experimental_grouped_half_control
                else (
                    "split_k32_control"
                    if args.experimental_split_k32_control
                    else (
                        "split_k32_pairwise_sync_control"
                        if args.experimental_split_k32_pairwise_sync_control
                        else (
                            "split_k32_two_way_reduce_control"
                            if args.experimental_split_k32_two_way_reduce_control
                            else (
                                "split_k48_control"
                                if args.experimental_split_k48_control
                                else "production"
                            )
                        )
                    )
                )
            )
        ),
        "contract": {
            "batch_size": 1, "query_heads": QUERY_HEADS,
            "key_value_heads": KV_HEADS, "head_dim": HEAD_DIM,
            "kv_length": args.kv_length, "vector_core_count": args.vector_core_count,
            "valid_kv_length": args.valid_kv_length,
            "dtype": "fp16", "layout": "BNSD", "masked": True,
            "actual_seq_lengths": None, "inner_precise": 1,
            "scale_value": SCALE_VALUE,
        },
        "environment": {
            "hostname": os.uname().nodename,
            "ascend_rt_visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            "ascend_custom_opp_path": os.environ.get("ASCEND_CUSTOM_OPP_PATH"),
            "device_name": torch.npu.get_device_name(0),
            "torch": torch.__version__,
            "torch_npu": getattr(torch_npu, "__version__", None),
        },
        "first_call_s": first_call_s,
        "timing": timings,
        "profile": profile_metadata,
        "output_sha256": {name: tensor_sha256(value) for name, value in outputs.items()},
        "parity": {
            "exact": exact,
            "allclose_atol_0_rtol_0": bool(torch.allclose(outputs["stock"], outputs["custom"], atol=0, rtol=0)),
            "allclose_fp16_tolerance": fp16_close,
            "fp16_tolerance": {"atol": FP16_ATOL, "rtol": FP16_RTOL},
            "max_abs": float(difference.max()),
            "mean_abs": float(difference.mean()),
            "per_query_head_max_abs": [float(value) for value in per_head_max_abs],
            "per_query_head_mean_abs": [float(value) for value in per_head_mean_abs],
            "per_query_head_custom_absmax": [float(value) for value in per_head_custom_absmax],
            "custom_head_best_stock_head": [int(value) for value in pairwise_best_stock_head.tolist()],
            "custom_head_best_stock_mean_abs": [float(value) for value in pairwise_best_mean_abs.tolist()],
            "custom_zero_fraction": float((outputs["custom"] == 0).float().mean()),
            "stock_sample": outputs["stock"].reshape(-1)[:16].tolist(),
            "custom_sample": outputs["custom"].reshape(-1)[:16].tolist(),
        },
        "independent_fp32_reference": {
            "implementation": "cpu_fp32_matmul_softmax_matmul_with_gqa_repeat",
            "tolerance": {"atol": REFERENCE_ATOL, "rtol": REFERENCE_RTOL},
            "stock_allclose": stock_reference_close,
            "custom_allclose": custom_reference_close,
            "stock_max_abs": float(stock_reference_difference.max()),
            "stock_mean_abs": float(stock_reference_difference.mean()),
            "custom_max_abs": float(custom_reference_difference.max()),
            "custom_mean_abs": float(custom_reference_difference.mean()),
            "reference_sample": reference.reshape(-1)[:16].tolist(),
        },
        "required_checks_passed": required_checks_passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if not required_checks_passed:
        raise SystemExit("GQA AIV output failed FP16 or independent-reference parity")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
