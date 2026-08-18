"""Static UniRec full-vision bucket presets and page-local call planning."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True, order=True)
class VisionBucketSpec:
    width: int
    height: int
    batch_size: int
    planning_cost_ms: float | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1 or self.batch_size < 1:
            raise ValueError(f"invalid UniRec vision bucket: {self}")
        if self.width % 32 or self.height % 32:
            raise ValueError(
                "UniRec vision bucket dimensions must be divisible by 32: "
                f"{self.width}x{self.height}"
            )
        if self.planning_cost_ms is not None and self.planning_cost_ms <= 0:
            raise ValueError("vision bucket planning cost must be positive")

    @property
    def key(self) -> str:
        return f"{self.width}x{self.height}_b{self.batch_size}"

    def accepts(self, width: int, height: int) -> bool:
        return width <= self.width and height <= self.height

    @property
    def final_stage_rows(self) -> int:
        """Flattened B*H*W rows entering the final focal-vision stage."""
        return self.batch_size * (self.width // 32) * (self.height // 32)

    @property
    def has_aligned_final_stage_rows(self) -> bool:
        """Whether the final-stage row count fills complete 16-row tiles."""
        return self.final_stage_rows % 16 == 0


# Five graphs cover 1,513/1,564 accepted crops in the first 32 hard pages.
DEFAULT_VISION_BUCKETS = (
    VisionBucketSpec(width=960, height=64, batch_size=16),
    VisionBucketSpec(width=512, height=256, batch_size=16),
    VisionBucketSpec(width=960, height=256, batch_size=4),
    VisionBucketSpec(width=512, height=512, batch_size=8),
    VisionBucketSpec(width=960, height=512, batch_size=4),
)


# CPU-optimized against the distribution-matched representative-128 trace with
# one-page vision lookahead and the measured Ascend 310P physical-pixel latency
# curve. Multiple variants of one canvas are deliberate: the runtime uses the
# estimated costs below to select B4/B2 combinations for each page-local row
# count. These estimates select the call mix only; NPU timing remains measured.
VISION_BUCKETS_310P_K10_L1 = (
    VisionBucketSpec(448, 64, 4, planning_cost_ms=6.167823529),
    VisionBucketSpec(448, 256, 2, planning_cost_ms=10.148571429),
    VisionBucketSpec(448, 384, 2, planning_cost_ms=14.655714286),
    VisionBucketSpec(512, 128, 4, planning_cost_ms=10.920000000),
    VisionBucketSpec(960, 64, 2, planning_cost_ms=6.388411765),
    VisionBucketSpec(960, 64, 4, planning_cost_ms=10.738500000),
    VisionBucketSpec(960, 128, 1, planning_cost_ms=6.388411765),
    VisionBucketSpec(960, 256, 1, planning_cost_ms=10.738500000),
    VisionBucketSpec(960, 384, 1, planning_cost_ms=15.776428571),
    VisionBucketSpec(960, 512, 1, planning_cost_ms=21.380000000),
)


# Jointly optimized against the same representative-128 workload and measured
# 310P latency curve, but with four-page crop pooling and all formerly eager
# fallback shapes included.  This keeps the runtime at ten static graph slots,
# covers every representative crop through height 1408, and permits layout B2.
VISION_BUCKETS_310P_K10_L4_ALL = (
    VisionBucketSpec(448, 192, 2, planning_cost_ms=8.083821429),
    VisionBucketSpec(448, 384, 2, planning_cost_ms=14.655714286),
    VisionBucketSpec(512, 64, 4, planning_cost_ms=6.609000000),
    VisionBucketSpec(960, 64, 4, planning_cost_ms=10.738500000),
    VisionBucketSpec(960, 128, 2, planning_cost_ms=10.738500000),
    VisionBucketSpec(960, 256, 1, planning_cost_ms=10.738500000),
    VisionBucketSpec(960, 448, 1, planning_cost_ms=18.578214286),
    VisionBucketSpec(960, 576, 1, planning_cost_ms=23.950375000),
    VisionBucketSpec(960, 896, 1, planning_cost_ms=41.847250000),
    VisionBucketSpec(960, 1408, 1, planning_cost_ms=70.597607143),
)


# Correctness-safe successor to K10/L4/all. Candidate canvases come from the
# aligned K=10 frontier: every final-stage B*H*W row count fills complete
# 16-row physical tiles. Synthetic 1024-wide canvases are deliberate; they are
# cheaper than padding the corresponding tall 960-wide crops in height or B.
VISION_BUCKETS_310P_K10_L4_ALIGNED = (
    VisionBucketSpec(448, 384, 2, planning_cost_ms=14.655714286),
    VisionBucketSpec(512, 64, 4, planning_cost_ms=6.609000000),
    VisionBucketSpec(512, 192, 2, planning_cost_ms=8.968714286),
    VisionBucketSpec(960, 64, 4, planning_cost_ms=10.738500000),
    VisionBucketSpec(960, 128, 2, planning_cost_ms=10.738500000),
    VisionBucketSpec(960, 256, 1, planning_cost_ms=10.738500000),
    VisionBucketSpec(960, 512, 1, planning_cost_ms=21.380000000),
    VisionBucketSpec(960, 1024, 1, planning_cost_ms=49.006000000),
    VisionBucketSpec(1024, 704, 1, planning_cost_ms=33.734000000),
    VisionBucketSpec(1024, 1408, 1, planning_cost_ms=76.407571429),
)


# Unrestricted K=20 frontier for the representative-128 workload with four-page
# lookahead. The final-stage tile constraint is intentionally absent: the 310P
# correctness failure was traced to the ambiguous two-stage global-context
# reduction, not to the flattened row count. Planning costs use the same
# measured 310P physical-pixel latency curve as the K10 presets.
VISION_BUCKETS_310P_K20_L4 = (
    VisionBucketSpec(128, 1408, 1, planning_cost_ms=8.378785714),
    VisionBucketSpec(192, 64, 4, planning_cost_ms=4.403117647),
    VisionBucketSpec(320, 320, 2, planning_cost_ms=9.263678571),
    VisionBucketSpec(448, 192, 2, planning_cost_ms=8.083821429),
    VisionBucketSpec(448, 384, 2, planning_cost_ms=14.655714286),
    VisionBucketSpec(448, 576, 1, planning_cost_ms=10.874625000),
    VisionBucketSpec(512, 64, 4, planning_cost_ms=6.609000000),
    VisionBucketSpec(512, 128, 4, planning_cost_ms=10.920000000),
    VisionBucketSpec(512, 768, 1, planning_cost_ms=16.897142857),
    VisionBucketSpec(576, 256, 2, planning_cost_ms=12.414285714),
    VisionBucketSpec(960, 64, 4, planning_cost_ms=10.738500000),
    VisionBucketSpec(960, 128, 2, planning_cost_ms=10.738500000),
    VisionBucketSpec(960, 192, 1, planning_cost_ms=8.526267857),
    VisionBucketSpec(960, 256, 1, planning_cost_ms=10.738500000),
    VisionBucketSpec(960, 384, 1, planning_cost_ms=15.776428571),
    VisionBucketSpec(960, 512, 1, planning_cost_ms=21.380000000),
    VisionBucketSpec(960, 704, 1, planning_cost_ms=31.109125000),
    VisionBucketSpec(960, 896, 1, planning_cost_ms=41.847250000),
    VisionBucketSpec(960, 1152, 1, planning_cost_ms=54.752250000),
    VisionBucketSpec(960, 1344, 1, planning_cost_ms=66.636267857),
)


VISION_BUCKET_PRESETS = {
    "production_v1": DEFAULT_VISION_BUCKETS,
    "310p_k10_l1": VISION_BUCKETS_310P_K10_L1,
    "310p_k10_l4_all": VISION_BUCKETS_310P_K10_L4_ALL,
    "310p_k10_l4_aligned": VISION_BUCKETS_310P_K10_L4_ALIGNED,
    "310p_k20_l4": VISION_BUCKETS_310P_K20_L4,
}
VISION_BUCKET_PRESET_CHOICES = tuple(VISION_BUCKET_PRESETS)


# TorchAir's persisted cache path includes the bound method name. Preserve the
# original K10/L4 slot for shared shapes so changing bucket sets does not turn a
# cache hit into a new `_forward_bucket_slot_N` specialization. New aligned
# shapes occupy slots unused by the aligned preset. Keep this table append-only.
VISION_BUCKET_CACHE_SLOT_PREFERENCES = {
    "128x1408_b1": 6,
    "192x64_b4": 7,
    "320x320_b2": 10,
    "448x192_b2": 0,
    "448x384_b2": 1,
    "448x576_b1": 11,
    "512x64_b4": 2,
    "512x128_b4": 12,
    "512x768_b1": 13,
    "576x256_b2": 14,
    "960x64_b4": 3,
    "960x128_b2": 4,
    "960x192_b1": 15,
    "960x256_b1": 5,
    "960x384_b1": 16,
    "960x448_b1": 6,
    "960x576_b1": 7,
    "960x896_b1": 8,
    "960x704_b1": 17,
    "960x1152_b1": 18,
    "960x1344_b1": 19,
    "960x1408_b1": 9,
    "960x512_b1": 9,
    "512x192_b2": 0,
    "960x1024_b1": 7,
    "1024x704_b1": 8,
    "1024x1408_b1": 6,
}


def assign_vision_bucket_cache_slots(
    specs: Sequence[VisionBucketSpec],
    *,
    slot_count: int = 10,
) -> tuple[int, ...]:
    """Assign distinct, cache-stable static method slots to one preset."""
    if len(specs) > slot_count:
        raise ValueError(
            f"{len(specs)} vision buckets exceed {slot_count} static slots"
        )
    used: set[int] = set()
    result = []
    for spec in specs:
        preferred = VISION_BUCKET_CACHE_SLOT_PREFERENCES.get(spec.key)
        if preferred is not None and 0 <= preferred < slot_count:
            candidates = (preferred, *range(slot_count))
        else:
            candidates = tuple(range(slot_count))
        selected = next((slot for slot in candidates if slot not in used), None)
        if selected is None:
            raise ValueError(f"no static cache slot remains for {spec.key}")
        used.add(selected)
        result.append(selected)
    return tuple(result)


def resolve_vision_bucket_specs(name: str) -> tuple[VisionBucketSpec, ...]:
    try:
        return VISION_BUCKET_PRESETS[str(name)]
    except KeyError as error:
        raise ValueError(
            f"unknown vision bucket preset {name!r}; expected one of "
            f"{VISION_BUCKET_PRESET_CHOICES}"
        ) from error


def plan_canvas_bucket_calls(
    specs: Sequence[VisionBucketSpec],
    real_rows: int,
) -> tuple[VisionBucketSpec, ...]:
    """Return the lowest estimated-cost fixed-batch calls for one canvas."""
    if real_rows < 1:
        return ()
    if not specs:
        raise ValueError("canvas call planning requires at least one graph")
    canvases = {(spec.width, spec.height) for spec in specs}
    if len(canvases) != 1:
        raise ValueError(f"canvas call planner received mixed shapes: {canvases}")
    if len(specs) == 1:
        spec = specs[0]
        return (spec,) * math.ceil(real_rows / spec.batch_size)
    if any(spec.planning_cost_ms is None for spec in specs):
        raise ValueError(
            "multiple batch variants of one canvas require planning costs"
        )

    # State is (estimated_ms, physical_rows, call_count, call tuple). Padding is
    # legal on the final call, so each transition clamps completed real rows.
    infinite: tuple[float, int, int, tuple[VisionBucketSpec, ...]] = (
        float("inf"),
        2**31 - 1,
        2**31 - 1,
        (),
    )
    dp = [infinite] * (real_rows + 1)
    dp[0] = (0.0, 0, 0, ())
    for completed in range(real_rows):
        if not math.isfinite(dp[completed][0]):
            continue
        for spec in specs:
            target = min(real_rows, completed + spec.batch_size)
            candidate = (
                dp[completed][0] + float(spec.planning_cost_ms),
                dp[completed][1] + spec.batch_size,
                dp[completed][2] + 1,
                (*dp[completed][3], spec),
            )
            if candidate[:3] < dp[target][:3]:
                dp[target] = candidate
    return dp[real_rows][3]
