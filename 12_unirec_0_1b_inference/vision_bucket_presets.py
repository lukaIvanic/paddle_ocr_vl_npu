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


VISION_BUCKET_PRESETS = {
    "production_v1": DEFAULT_VISION_BUCKETS,
    "310p_k10_l1": VISION_BUCKETS_310P_K10_L1,
}
VISION_BUCKET_PRESET_CHOICES = tuple(VISION_BUCKET_PRESETS)


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
