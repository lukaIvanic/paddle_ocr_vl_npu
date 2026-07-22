"""Pinned 910B2 vision routing profile and cached batched graph runtime."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from ..model.compile_utils import (
    TORCHAIR_EXECUTION_MODE,
    cache_key_part,
    import_torchair,
    short_file_hash,
    torch_npu_version_label,
    torchair_version_label,
)
from ..model.vision_prefill import (
    PreparedVisionPrefill,
    VisionPrefillStage,
    get_vision_prompt_fa_layout,
    get_vision_prompt_fa_mask_sparse_mode,
    unique_bucket_forward,
    vision_source_hash,
)
from utils.timing import synchronize


PINNED_910B2_PROFILE = {
    "measured_commit": "bbefb38e9217bfdd614ee72614cd8568bff8c324",
    "large_b1_measured_commit": "70a97e5",
    "device_name": "Ascend910B2",
    "model_config_hash": "6d2211febbe9",
    "torch": "2.10.0+cpu",
    "torch_npu": "2.10.0",
    "vision_source_hash": "a2cd9cd7ae53",
    "attention": "prompt_flash_attention",
    "layout": "bnsd",
    "sparse_mode": 1,
    "execution_mode": "inference",
    "timing_basis": "median of 10 NPU device-event samples after 2 warmups",
    "graphs": {
        (1, 32): {"median_ms": 8.620610, "raw_physical_tokens_per_s": 3712.0},
        (1, 64): {"median_ms": 9.979780, "raw_physical_tokens_per_s": 6413.0},
        (1, 96): {"median_ms": 10.892880, "raw_physical_tokens_per_s": 8813.1},
        (1, 128): {"median_ms": 11.215550, "raw_physical_tokens_per_s": 11412.7},
        (1, 160): {"median_ms": 11.640450, "raw_physical_tokens_per_s": 13745.2},
        (1, 192): {"median_ms": 12.035350, "raw_physical_tokens_per_s": 15953.0},
        (1, 224): {"median_ms": 12.469300, "raw_physical_tokens_per_s": 17964.1},
        (1, 256): {"median_ms": 12.718330, "raw_physical_tokens_per_s": 20128.4},
        (1, 288): {"median_ms": 14.218220, "raw_physical_tokens_per_s": 20255.7},
        (1, 320): {"median_ms": 14.706450, "raw_physical_tokens_per_s": 21759.2},
        (1, 352): {"median_ms": 15.144390, "raw_physical_tokens_per_s": 23242.9},
        (1, 384): {"median_ms": 14.996730, "raw_physical_tokens_per_s": 25605.6},
        (1, 416): {"median_ms": 15.976960, "raw_physical_tokens_per_s": 26037.5},
        (1, 448): {"median_ms": 16.098060, "raw_physical_tokens_per_s": 27829.4},
        (1, 480): {"median_ms": 16.096730, "raw_physical_tokens_per_s": 29819.7},
        (1, 512): {"median_ms": 16.255239, "raw_physical_tokens_per_s": 31497.5},
        (1, 576): {"median_ms": 17.389590, "raw_physical_tokens_per_s": 33123.3},
        (1, 640): {"median_ms": 18.319080, "raw_physical_tokens_per_s": 34936.3},
        (1, 704): {"median_ms": 19.980740, "raw_physical_tokens_per_s": 35233.9},
        (1, 768): {"median_ms": 19.846950, "raw_physical_tokens_per_s": 38696.1},
        (1, 832): {"median_ms": 20.338870, "raw_physical_tokens_per_s": 40906.9},
        (1, 896): {"median_ms": 20.862309, "raw_physical_tokens_per_s": 42948.3},
        (1, 960): {"median_ms": 21.347060, "raw_physical_tokens_per_s": 44971.1},
        (1, 1024): {"median_ms": 22.167390, "raw_physical_tokens_per_s": 46194.0},
        (1, 1152): {"median_ms": 25.000620, "raw_physical_tokens_per_s": 46078.9},
        (1, 1280): {"median_ms": 26.380490, "raw_physical_tokens_per_s": 48520.7},
        (1, 1408): {"median_ms": 27.151820, "raw_physical_tokens_per_s": 51856.6},
        (1, 1536): {"median_ms": 29.161800, "raw_physical_tokens_per_s": 52671.6},
        (1, 1664): {"median_ms": 30.104520, "raw_physical_tokens_per_s": 55274.1},
        (1, 1792): {"median_ms": 28.499870, "raw_physical_tokens_per_s": 62877.5},
        (1, 1920): {"median_ms": 29.164190, "raw_physical_tokens_per_s": 65834.2},
        (1, 2048): {"median_ms": 31.565830, "raw_physical_tokens_per_s": 64880.3},
        (1, 2304): {"median_ms": 37.365240, "raw_physical_tokens_per_s": 61661.6},
        (1, 2560): {"median_ms": 40.903900, "raw_physical_tokens_per_s": 62585.7},
        (1, 2816): {"median_ms": 44.992441, "raw_physical_tokens_per_s": 62588.3},
        (1, 3072): {"median_ms": 46.371449, "raw_physical_tokens_per_s": 66247.7},
        (1, 3584): {"median_ms": 57.649879, "raw_physical_tokens_per_s": 62168.4},
        (1, 4096): {"median_ms": 64.408413, "raw_physical_tokens_per_s": 63594.2},
        (1, 4608): {"median_ms": 74.239029, "raw_physical_tokens_per_s": 62069.8},
        (1, 5120): {"median_ms": 86.321079, "raw_physical_tokens_per_s": 59313.4},
        (2, 3072): {"median_ms": 80.734600, "raw_physical_tokens_per_s": 76101.2},
        (4, 1024): {"median_ms": 46.402910, "raw_physical_tokens_per_s": 88270.3},
    },
}

LARGE_B1_FALLBACK_SHAPES = (
    (1, 2304),
    (1, 2560),
    (1, 2816),
    (1, 3072),
    (1, 3584),
    (1, 4096),
    (1, 4608),
    (1, 5120),
)


@dataclass(frozen=True)
class ProfiledVisionRoute:
    batch_size: int
    sequence_length: int
    rows: tuple[tuple[int, ...], ...]
    real_tokens: int
    physical_tokens: int
    profiled_ms: float | None
    execution: str


def _candidate(
    lengths: Sequence[int],
    *,
    batch_size: int,
    sequence_length: int,
) -> ProfiledVisionRoute | None:
    if int(lengths[0]) > sequence_length:
        return None
    rows: list[list[int]] = [[] for _ in range(batch_size)]
    totals = [0] * batch_size
    rows[0].append(0)
    totals[0] = int(lengths[0])
    for item_index in sorted(
        range(1, len(lengths)),
        key=lambda index: (-int(lengths[index]), index),
    ):
        tokens = int(lengths[item_index])
        available = [
            row_index
            for row_index, used in enumerate(totals)
            if used + tokens <= sequence_length
        ]
        if not available:
            continue
        row_index = max(available, key=lambda index: totals[index])
        rows[row_index].append(item_index)
        totals[row_index] += tokens
    real_tokens = sum(totals)
    profile = PINNED_910B2_PROFILE["graphs"][(batch_size, sequence_length)]
    return ProfiledVisionRoute(
        batch_size=batch_size,
        sequence_length=sequence_length,
        rows=tuple(tuple(row) for row in rows),
        real_tokens=real_tokens,
        physical_tokens=batch_size * sequence_length,
        profiled_ms=float(profile["median_ms"]),
        execution="compiled",
    )


def select_profiled_vision_route(lengths: Sequence[int]) -> ProfiledVisionRoute:
    if not lengths or any(int(value) <= 0 for value in lengths):
        raise ValueError("profiled vision routing requires positive crop lengths")
    candidates = []
    for batch_size, sequence_length in PINNED_910B2_PROFILE["graphs"]:
        if (batch_size, sequence_length) in LARGE_B1_FALLBACK_SHAPES:
            continue
        route = _candidate(
            lengths,
            batch_size=int(batch_size),
            sequence_length=int(sequence_length),
        )
        if route is not None:
            candidates.append(route)
    if not candidates:
        for batch_size, sequence_length in LARGE_B1_FALLBACK_SHAPES:
            route = _candidate(
                lengths,
                batch_size=batch_size,
                sequence_length=sequence_length,
            )
            if route is not None:
                candidates.append(route)
    if not candidates:
        return ProfiledVisionRoute(
            batch_size=1,
            sequence_length=int(lengths[0]),
            rows=((0,),),
            real_tokens=int(lengths[0]),
            physical_tokens=int(lengths[0]),
            profiled_ms=None,
            execution="eager_overflow",
        )
    return max(
        candidates,
        key=lambda route: (
            route.real_tokens / float(route.profiled_ms),
            route.real_tokens,
            -float(route.profiled_ms),
            -route.physical_tokens,
        ),
    )


def batched_vision_cache_dir(
    *,
    batch_size: int,
    sequence_length: int,
    cache_root: Path,
    model_dir: Path,
    dtype: torch.dtype,
    device: torch.device,
) -> Path:
    key = "_".join(
        (
            "encoder_postln_promptfa",
            f"b{batch_size}",
            f"s{sequence_length}",
            f"dtype{cache_key_part(dtype)}",
            f"layout{cache_key_part(get_vision_prompt_fa_layout())}",
            f"sparse{get_vision_prompt_fa_mask_sparse_mode()}",
            f"mode{cache_key_part(TORCHAIR_EXECUTION_MODE)}",
            f"model{short_file_hash(model_dir / 'config.json')}",
            f"torch{cache_key_part(torch.__version__)}",
            f"torchnpu{torch_npu_version_label(device)}",
            f"torchair{torchair_version_label(device)}",
            f"src{vision_source_hash()}",
        )
    )
    return cache_root.expanduser().resolve() / key


class BatchedVisionGraphRuntime:
    SHAPES = ((2, 3072), (4, 1024))

    def __init__(
        self,
        model: Any,
        *,
        cache_root: Path,
        model_dir: Path,
        dtype: torch.dtype,
        device: torch.device,
    ):
        expected_source = str(PINNED_910B2_PROFILE["vision_source_hash"])
        if vision_source_hash() != expected_source:
            raise RuntimeError(
                "pinned vision profile source mismatch: "
                f"expected={expected_source} actual={vision_source_hash()}"
            )
        torchair, CompilerConfig = import_torchair()
        hidden_size = int(model.config.vision_config.hidden_size)
        head_dim = hidden_size // int(model.config.vision_config.num_attention_heads)
        self.compiled: dict[tuple[int, int], Callable[..., torch.Tensor]] = {}
        per_shape: dict[str, Any] = {}
        setup_started = time.perf_counter()
        for batch_size, sequence_length in self.SHAPES:
            cache_dir = batched_vision_cache_dir(
                batch_size=batch_size,
                sequence_length=sequence_length,
                cache_root=cache_root,
                model_dir=model_dir,
                dtype=dtype,
                device=device,
            )
            if not cache_dir.is_dir() or not any(cache_dir.rglob("*")):
                raise RuntimeError(
                    "profile-guided vision routing requires a warm batched graph: "
                    f"shape=b{batch_size}_s{sequence_length} cache={cache_dir}"
                )
            module = VisionPrefillStage(
                model,
                attention_impl="prompt_flash_attention",
            ).eval()
            entrypoint = unique_bucket_forward(module, sequence_length)
            synchronize(device)
            wrapper_started = time.perf_counter()
            compiled = torchair.inference.cache_compile(
                entrypoint,
                config=CompilerConfig(),
                dynamic=False,
                cache_dir=str(cache_dir),
                ge_cache=True,
            )
            synchronize(device)
            warm_prefix = torch.zeros(
                (batch_size, sequence_length, hidden_size),
                device=device,
                dtype=dtype,
            )
            warm_cos = torch.ones(
                (batch_size, sequence_length, head_dim),
                device=device,
                dtype=torch.float32,
            )
            warm_sin = torch.zeros_like(warm_cos)
            warm_mask = torch.zeros(
                (batch_size, 1, sequence_length, sequence_length),
                device=device,
                dtype=torch.bool,
            )
            warm_started = time.perf_counter()
            output = compiled(warm_prefix, warm_cos, warm_sin, warm_mask)
            synchronize(device)
            if tuple(output.shape[:2]) != (batch_size, sequence_length):
                raise RuntimeError(
                    "cached batched vision graph returned the wrong shape: "
                    f"expected={(batch_size, sequence_length)} got={tuple(output.shape)}"
                )
            self.compiled[(batch_size, sequence_length)] = compiled
            per_shape[f"b{batch_size}_s{sequence_length}"] = {
                "cache_dir": str(cache_dir),
                "wrapper_s": warm_started - wrapper_started,
                "first_call_s": time.perf_counter() - warm_started,
            }
            del output, warm_prefix, warm_cos, warm_sin, warm_mask
        self.metadata = {
            "profile": {
                key: value
                for key, value in PINNED_910B2_PROFILE.items()
                if key != "graphs"
            },
            "shapes": per_shape,
            "setup_s": time.perf_counter() - setup_started,
        }

    def run(
        self,
        *,
        batch_size: int,
        sequence_length: int,
        prepared_rows: Sequence[PreparedVisionPrefill],
    ) -> torch.Tensor:
        if len(prepared_rows) != batch_size:
            raise ValueError(
                f"expected {batch_size} prepared rows, got {len(prepared_rows)}"
            )
        return self.compiled[(batch_size, sequence_length)](
            torch.cat([row.prefix_hidden_states for row in prepared_rows], dim=0),
            torch.cat([row.rope_cos for row in prepared_rows], dim=0),
            torch.cat([row.rope_sin for row in prepared_rows], dim=0),
            torch.cat([row.attention_mask for row in prepared_rows], dim=0),
        )
