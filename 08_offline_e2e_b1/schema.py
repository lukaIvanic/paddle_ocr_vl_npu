"""Typed records shared by the Experiment 08 offline pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Box:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    def as_list(self) -> list[float]:
        return [float(self.x0), float(self.y0), float(self.x1), float(self.y1)]


@dataclass(frozen=True)
class LayoutRegion:
    order: int
    label_id: int
    label: str
    score: float
    box: Box
    polygon_points: list[list[float]] | None = None


@dataclass
class RecognitionResult:
    request_id: str
    decode_schedule_id: str
    decode_slot_index: int | None
    decode_slot_epoch: int | None
    layout_order: int
    label: str
    prompt: str
    box: Box
    crop_size: tuple[int, int]
    text: str
    token_ids: list[int]
    stop_reason: str
    input_tokens: int
    projected_image_tokens: int
    generated_tokens_including_eos: int
    decode_tokens_after_prefill_including_eos: int
    decode_calls_executed: int
    timing_s: dict[str, float]
    device_stage_s: dict[str, float]
    rates: dict[str, float | None]


@dataclass
class ContinuousDecodeResult:
    schedule_id: str
    batch_size: int
    requests: int
    ready_buffer_capacity: int
    ready_buffer_low_watermark: int
    max_ready_queue_depth: int
    ready_source_refill_count: int
    graph_calls: int
    initial_admissions: int
    hot_swap_admissions: int
    prefill_only_completions: int
    raw_decode_token_slots: int
    active_decode_token_slots: int
    effective_decode_tokens: int
    idle_decode_token_slots: int
    lookahead_decode_token_slots: int
    kv_prefix_bytes_copied: int
    initial_kv_prefix_bytes_copied: int
    hot_swap_kv_prefix_bytes_copied: int
    timing_s: dict[str, float]
    rates: dict[str, float | None]


@dataclass
class SkippedRegion:
    layout_order: int
    label: str
    reason: str
    box: Box


@dataclass
class PageResult:
    page_id: str
    image_path: Path
    image_size: tuple[int, int]
    layout_regions: list[LayoutRegion]
    recognized_regions: list[RecognitionResult]
    decode_schedule_id: str
    skipped_regions: list[SkippedRegion]
    reading_order_text: str
    timing_s: dict[str, float]
    rates: dict[str, float | None]
    partial: bool = False


@dataclass
class RunResult:
    experiment: str
    configuration: dict[str, Any]
    setup_timing_s: dict[str, Any]
    pages: list[PageResult]
    decode_schedule: ContinuousDecodeResult
    aggregate: dict[str, Any] = field(default_factory=dict)
    metric_definitions: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return json_ready(asdict(self))


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def per_second(count: int | float, seconds: float) -> float | None:
    if seconds <= 0:
        return None
    return float(count) / float(seconds)
