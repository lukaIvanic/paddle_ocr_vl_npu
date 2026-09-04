#!/usr/bin/env python3
"""Compare current two-call mixed attention against one padded PromptFA call."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Callable

import torch


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXPERIMENT_ROOT))


QUERY_HEADS = 16
KV_HEADS = 2
GROUPS = QUERY_HEADS // KV_HEADS
HEAD_DIM = 128
VERIFIER_Q = 8
VERIFIER_KV = 4096
DRAFT_BATCH = 8
DRAFT_Q = 1
DRAFT_KV = 768
PACKED_Q = VERIFIER_Q + (DRAFT_BATCH * DRAFT_Q)
PACKED_KV = VERIFIER_KV + (DRAFT_BATCH * DRAFT_KV)
SCALE = 1.0 / math.sqrt(HEAD_DIM)
FULL_ATTENTION_TOKENS = (1 << 31) - 1
VERIFIER_START = 1249
DRAFT_POSITIONS = (128, 155, 173, 189, 205, 225, 270, 382)
LANES = (
    "current_two_call",
    "packed_bsnd_promptfa",
    "padded_b9_promptfa",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=LANES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("eager", "torchair"), default="eager")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=EXPERIMENT_ROOT.parent
        / ".runtime_cache/09_persistent_page_engine_torchair/mixed_single_attention",
    )
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--profile-warmups", type=int, default=10)
    parser.add_argument("--profile-repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()
    for name in ("warmups", "repeats", "profile_warmups", "profile_repeats"):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def _write(path: Path, payload: dict[str, object]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _verifier_legal_mask(device: torch.device) -> torch.Tensor:
    query_positions = VERIFIER_START + torch.arange(
        VERIFIER_Q, device=device, dtype=torch.int64
    )
    kv_positions = torch.arange(VERIFIER_KV, device=device, dtype=torch.int64)
    future = kv_positions.view(1, 1, 1, 1, VERIFIER_KV) > (
        query_positions.view(1, 1, 1, VERIFIER_Q, 1)
    )
    return (
        future.expand(1, KV_HEADS, GROUPS, VERIFIER_Q, VERIFIER_KV)
        .reshape(1, 1, QUERY_HEADS * VERIFIER_Q, VERIFIER_KV)
        .contiguous()
    )


def _draft_mask(device: torch.device) -> torch.Tensor:
    positions = torch.tensor(DRAFT_POSITIONS, device=device, dtype=torch.int64)
    kv_positions = torch.arange(DRAFT_KV, device=device, dtype=torch.int64)
    return (kv_positions.view(1, DRAFT_KV) > positions.view(-1, 1)).view(
        DRAFT_BATCH, 1, 1, DRAFT_KV
    )


def _packed_mask(device: torch.device) -> torch.Tensor:
    query_indices = torch.arange(PACKED_Q, device=device, dtype=torch.int64)
    kv_indices = torch.arange(PACKED_KV, device=device, dtype=torch.int64)
    verifier_limits = VERIFIER_START + torch.arange(
        VERIFIER_Q, device=device, dtype=torch.int64
    )
    draft_positions = torch.tensor(
        DRAFT_POSITIONS, device=device, dtype=torch.int64
    )
    draft_offsets = VERIFIER_KV + (
        torch.arange(DRAFT_BATCH, device=device, dtype=torch.int64) * DRAFT_KV
    )
    segment_starts = torch.cat(
        (
            torch.zeros(VERIFIER_Q, device=device, dtype=torch.int64),
            draft_offsets,
        )
    )
    limits = torch.cat((verifier_limits, draft_offsets + draft_positions))
    allowed = (kv_indices.view(1, -1) >= segment_starts.view(-1, 1)) & (
        kv_indices.view(1, -1) <= limits.view(-1, 1)
    )
    if tuple(allowed.shape) != (PACKED_Q, PACKED_KV):
        raise AssertionError("packed mask has the wrong shape")
    del query_indices
    return (~allowed).view(1, 1, PACKED_Q, PACKED_KV).contiguous()


def _padded_b9_mask(device: torch.device) -> torch.Tensor:
    mask = torch.ones(
        (1 + DRAFT_BATCH, 1, VERIFIER_Q, VERIFIER_KV),
        dtype=torch.bool,
        device=device,
    )
    kv_positions = torch.arange(VERIFIER_KV, device=device, dtype=torch.int64)
    verifier_limits = VERIFIER_START + torch.arange(
        VERIFIER_Q, device=device, dtype=torch.int64
    )
    mask[0, 0] = kv_positions.view(1, -1) > verifier_limits.view(-1, 1)
    draft_positions = torch.tensor(
        DRAFT_POSITIONS, device=device, dtype=torch.int64
    )
    mask[1:, 0, 0] = kv_positions.view(1, -1) > draft_positions.view(-1, 1)
    return mask.contiguous()


def _manual_verifier(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    legal_mask: torch.Tensor,
) -> torch.Tensor:
    import torch_npu

    grouped_query = query.view(1, KV_HEADS, GROUPS, VERIFIER_Q, HEAD_DIM)
    grouped_query = grouped_query.reshape(KV_HEADS, GROUPS * VERIFIER_Q, HEAD_DIM)
    grouped_key = key.reshape(KV_HEADS, VERIFIER_KV, HEAD_DIM)
    grouped_value = value.reshape(KV_HEADS, VERIFIER_KV, HEAD_DIM)
    scores = torch.bmm(grouped_query, grouped_key.transpose(1, 2))
    probabilities = torch_npu.npu_scaled_masked_softmax(
        scores.reshape(1, 1, QUERY_HEADS * VERIFIER_Q, VERIFIER_KV),
        legal_mask,
        SCALE,
        False,
    ).view_as(scores)
    return torch.bmm(probabilities, grouped_value).view(
        1, QUERY_HEADS, VERIFIER_Q, HEAD_DIM
    )


def _current_two_call(
    query_bsnd: torch.Tensor,
    verifier_key_bnsd: torch.Tensor,
    verifier_value_bnsd: torch.Tensor,
    draft_key_bnsd: torch.Tensor,
    draft_value_bnsd: torch.Tensor,
    verifier_mask: torch.Tensor,
    draft_mask: torch.Tensor,
) -> torch.Tensor:
    import torch_npu

    verifier_query = query_bsnd[:, :VERIFIER_Q].transpose(1, 2).contiguous()
    draft_query = query_bsnd[:, VERIFIER_Q:].reshape(
        DRAFT_BATCH, DRAFT_Q, QUERY_HEADS, HEAD_DIM
    ).transpose(1, 2)
    verifier_output = _manual_verifier(
        verifier_query,
        verifier_key_bnsd,
        verifier_value_bnsd,
        verifier_mask,
    )
    draft_output = torch_npu.npu_incre_flash_attention(
        draft_query.contiguous(),
        draft_key_bnsd,
        draft_value_bnsd,
        atten_mask=draft_mask,
        num_heads=QUERY_HEADS,
        num_key_value_heads=KV_HEADS,
        input_layout="BNSD",
        scale_value=SCALE,
    )
    return torch.cat(
        (
            verifier_output.transpose(1, 2),
            draft_output.transpose(1, 2).reshape(
                1, DRAFT_BATCH, QUERY_HEADS, HEAD_DIM
            ),
        ),
        dim=1,
    ).contiguous()


def _promptfa_bsnd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    import torch_npu

    return torch_npu.npu_prompt_flash_attention(
        query,
        key,
        value,
        atten_mask=mask,
        num_heads=QUERY_HEADS,
        num_key_value_heads=KV_HEADS,
        input_layout="BSND",
        scale_value=SCALE,
        pre_tokens=FULL_ATTENTION_TOKENS,
        next_tokens=FULL_ATTENTION_TOKENS,
        sparse_mode=0,
    )


class CurrentTwoCallStage(torch.nn.Module):
    def forward(
        self,
        query_bsnd: torch.Tensor,
        verifier_key_bnsd: torch.Tensor,
        verifier_value_bnsd: torch.Tensor,
        draft_key_bnsd: torch.Tensor,
        draft_value_bnsd: torch.Tensor,
        verifier_mask: torch.Tensor,
        draft_mask: torch.Tensor,
    ) -> torch.Tensor:
        return _current_two_call(
            query_bsnd,
            verifier_key_bnsd,
            verifier_value_bnsd,
            draft_key_bnsd,
            draft_value_bnsd,
            verifier_mask,
            draft_mask,
        )


class PackedBsndPromptFaStage(torch.nn.Module):
    def forward(
        self,
        query_bsnd: torch.Tensor,
        packed_key_bsnd: torch.Tensor,
        packed_value_bsnd: torch.Tensor,
        packed_mask: torch.Tensor,
    ) -> torch.Tensor:
        return _promptfa_bsnd(
            query_bsnd,
            packed_key_bsnd,
            packed_value_bsnd,
            packed_mask,
        )


class PaddedB9PromptFaStage(torch.nn.Module):
    def forward(
        self,
        padded_query: torch.Tensor,
        padded_key: torch.Tensor,
        padded_value: torch.Tensor,
        padded_mask: torch.Tensor,
    ) -> torch.Tensor:
        output = _promptfa_bsnd(
            padded_query,
            padded_key,
            padded_value,
            padded_mask,
        )
        return torch.cat(
            (
                output[:1],
                output[1:, :1].reshape(
                    1, DRAFT_BATCH, QUERY_HEADS, HEAD_DIM
                ),
            ),
            dim=1,
        )


def _compiled_call(
    stage: torch.nn.Module,
    stage_args: tuple[torch.Tensor, ...],
    *,
    lane: str,
    cache_root: Path,
) -> tuple[Callable[[], torch.Tensor], dict[str, object]]:
    from paddleocr_vl.model.compile_utils import import_torchair
    from paddleocr_vl.model.text_spec_verify import (
        _register_scaled_masked_softmax_torchair_converter,
    )

    _register_scaled_masked_softmax_torchair_converter()
    source_hash = hashlib.sha1(Path(__file__).read_bytes()).hexdigest()[:12]
    cache_dir = cache_root.expanduser().resolve() / f"{lane}_{source_hash}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_was_warm = any(cache_dir.iterdir())
    torchair, CompilerConfig = import_torchair()
    compiled = torchair.inference.cache_compile(
        stage.forward,
        config=CompilerConfig(),
        dynamic=False,
        cache_dir=str(cache_dir),
        ge_cache=True,
    )
    return lambda: compiled(*stage_args), {
        "cache_dir": str(cache_dir),
        "cache_was_warm_before_setup": cache_was_warm,
    }


def _measure(
    call: Callable[[], torch.Tensor],
    *,
    warmups: int,
    repeats: int,
) -> tuple[dict[str, object], torch.Tensor]:
    import torch_npu

    output: torch.Tensor | None = None
    for _ in range(warmups):
        output = call()
    torch_npu.npu.synchronize()
    samples: list[float] = []
    wall_started = time.perf_counter()
    for _ in range(repeats):
        start = torch_npu.npu.Event(enable_timing=True)
        end = torch_npu.npu.Event(enable_timing=True)
        start.record()
        output = call()
        end.record()
        torch_npu.npu.synchronize()
        samples.append(float(start.elapsed_time(end)) * 1000.0)
    wall_s = time.perf_counter() - wall_started
    if output is None:
        raise AssertionError("measurement produced no output")
    return {
        "warmups": warmups,
        "repeats": repeats,
        "device_us": {
            "mean": statistics.mean(samples),
            "median": statistics.median(samples),
            "p95": _percentile(samples, 0.95),
            "min": min(samples),
            "max": max(samples),
        },
        "host_wall_s": wall_s,
    }, output.detach().clone()


def _profile(
    call: Callable[[], torch.Tensor],
    profile_dir: Path,
    *,
    warmups: int,
    repeats: int,
) -> dict[str, object]:
    import torch_npu
    import torch_npu.profiler as npu_prof

    resolved = profile_dir.expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=False)
    schedule = npu_prof.schedule(wait=0, warmup=warmups, active=repeats, repeat=1)
    started = time.perf_counter()
    with npu_prof.profile(
        activities=[npu_prof.ProfilerActivity.CPU, npu_prof.ProfilerActivity.NPU],
        schedule=schedule,
        on_trace_ready=npu_prof.tensorboard_trace_handler(
            str(resolved), analyse_flag=True
        ),
        record_shapes=True,
        profile_memory=False,
        with_stack=True,
        with_modules=False,
        with_flops=False,
        experimental_config=npu_prof._ExperimentalConfig(
            profiler_level=npu_prof.ProfilerLevel.Level1,
            aic_metrics=npu_prof.AiCMetrics.PipeUtilization,
            l2_cache=False,
            export_type=npu_prof.ExportType.Text,
            data_simplification=False,
        ),
    ) as profiler:
        for _ in range(warmups + repeats):
            call()
            torch_npu.npu.synchronize()
            profiler.step()
    torch_npu.npu.synchronize()
    return {
        "path": str(resolved),
        "warmups": warmups,
        "captured_calls": repeats,
        "wall_s": time.perf_counter() - started,
        "outside_clean_timing": True,
    }


def main() -> None:
    args = parse_args()
    import torch_npu

    if not torch_npu.npu.is_available():
        raise RuntimeError("this probe requires an Ascend NPU")
    torch.npu.config.allow_internal_format = True
    torch.npu.set_compile_mode(jit_compile=False)
    device = torch.device("npu:0")
    torch.manual_seed(args.seed)
    query_bsnd = torch.randn(
        (1, PACKED_Q, QUERY_HEADS, HEAD_DIM),
        device=device,
        dtype=torch.float16,
    )
    verifier_key_bnsd = torch.randn(
        (1, KV_HEADS, VERIFIER_KV, HEAD_DIM),
        device=device,
        dtype=torch.float16,
    )
    verifier_value_bnsd = torch.randn_like(verifier_key_bnsd)
    draft_key_bnsd = torch.randn(
        (DRAFT_BATCH, KV_HEADS, DRAFT_KV, HEAD_DIM),
        device=device,
        dtype=torch.float16,
    )
    draft_value_bnsd = torch.randn_like(draft_key_bnsd)
    verifier_mask = _verifier_legal_mask(device)
    draft_mask = _draft_mask(device)
    packed_mask = _packed_mask(device)

    packed_key_bsnd = torch.cat(
        (
            verifier_key_bnsd.transpose(1, 2),
            draft_key_bnsd.transpose(1, 2).reshape(
                1, DRAFT_BATCH * DRAFT_KV, KV_HEADS, HEAD_DIM
            ),
        ),
        dim=1,
    ).contiguous()
    packed_value_bsnd = torch.cat(
        (
            verifier_value_bnsd.transpose(1, 2),
            draft_value_bnsd.transpose(1, 2).reshape(
                1, DRAFT_BATCH * DRAFT_KV, KV_HEADS, HEAD_DIM
            ),
        ),
        dim=1,
    ).contiguous()

    padded_query = torch.zeros(
        (1 + DRAFT_BATCH, VERIFIER_Q, QUERY_HEADS, HEAD_DIM),
        device=device,
        dtype=torch.float16,
    )
    padded_query[0] = query_bsnd[0, :VERIFIER_Q]
    padded_query[1:, 0] = query_bsnd[0, VERIFIER_Q:]
    padded_key = torch.zeros(
        (1 + DRAFT_BATCH, VERIFIER_KV, KV_HEADS, HEAD_DIM),
        device=device,
        dtype=torch.float16,
    )
    padded_value = torch.zeros_like(padded_key)
    padded_key[0] = verifier_key_bnsd[0].transpose(0, 1)
    padded_value[0] = verifier_value_bnsd[0].transpose(0, 1)
    padded_key[1:, :DRAFT_KV] = draft_key_bnsd.transpose(1, 2)
    padded_value[1:, :DRAFT_KV] = draft_value_bnsd.transpose(1, 2)
    padded_mask = _padded_b9_mask(device)
    torch_npu.npu.synchronize()

    reference_call = lambda: _current_two_call(
        query_bsnd,
        verifier_key_bnsd,
        verifier_value_bnsd,
        draft_key_bnsd,
        draft_value_bnsd,
        verifier_mask,
        draft_mask,
    )
    reference = reference_call().detach()
    if args.lane == "current_two_call":
        stage = CurrentTwoCallStage().eval()
        stage_args = (
            query_bsnd,
            verifier_key_bnsd,
            verifier_value_bnsd,
            draft_key_bnsd,
            draft_value_bnsd,
            verifier_mask,
            draft_mask,
        )
    elif args.lane == "packed_bsnd_promptfa":
        stage = PackedBsndPromptFaStage().eval()
        stage_args = (
            query_bsnd,
            packed_key_bsnd,
            packed_value_bsnd,
            packed_mask,
        )
    else:
        stage = PaddedB9PromptFaStage().eval()
        stage_args = (padded_query, padded_key, padded_value, padded_mask)

    setup: dict[str, object] = {"backend": args.backend}
    if args.backend == "torchair":
        call, compile_metadata = _compiled_call(
            stage,
            stage_args,
            lane=args.lane,
            cache_root=args.cache_dir,
        )
        setup["compile"] = compile_metadata
    else:
        call = lambda: stage(*stage_args)

    timing, output = _measure(
        call,
        warmups=args.warmups,
        repeats=args.repeats,
    )
    delta = (output.float() - reference.float()).abs()
    profile = _profile(
        call,
        args.profile_dir,
        warmups=args.profile_warmups,
        repeats=args.profile_repeats,
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "kind": "mixed_single_attention_operator_probe",
        "status": "complete",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "lane": args.lane,
        "device": str(device),
        "seed": args.seed,
        "setup": setup,
        "contract": {
            "query_heads": QUERY_HEADS,
            "kv_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "verifier": {"q": VERIFIER_Q, "kv": VERIFIER_KV},
            "draft": {"batch": DRAFT_BATCH, "q": DRAFT_Q, "kv": DRAFT_KV},
            "packed": {"q": PACKED_Q, "kv": PACKED_KV},
            "cache_assembly_in_timed_region": False,
        },
        "timing": timing,
        "comparison_to_current": {
            "exact": bool(torch.equal(output, reference)),
            "allclose_atol_5e_2_rtol_5e_2": bool(
                torch.allclose(output, reference, atol=5e-2, rtol=5e-2)
            ),
            "max_abs": float(delta.max().item()),
            "mean_abs": float(delta.mean().item()),
        },
        "profile": profile,
    }
    _write(args.output, payload)
    print(
        "MIXED_SINGLE_ATTENTION_RESULT "
        f"lane={args.lane} "
        f"median_us={timing['device_us']['median']:.3f} "
        f"p95_us={timing['device_us']['p95']:.3f} "
        f"max_abs={payload['comparison_to_current']['max_abs']:.6f} "
        f"allclose={payload['comparison_to_current']['allclose_atol_5e_2_rtol_5e_2']} "
        f"output={args.output.expanduser().resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
