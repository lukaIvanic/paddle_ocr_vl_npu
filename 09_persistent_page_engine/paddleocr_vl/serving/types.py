"""Request, result, and schedule contracts for PaddleOCR-VL serving."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class RecognitionRequest:
    request_id: str
    crop: Image.Image
    prompt: str
    skip_special_tokens: bool = True
    min_pixels: int | None = None
    max_pixels: int | None = None


@dataclass
class RecognitionResult:
    request_id: str
    decode_schedule_id: str
    decode_slot_index: int | None
    decode_slot_epoch: int | None
    prompt: str
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
    vision: dict[str, Any] = field(default_factory=dict)
    text_prefill: dict[str, Any] = field(default_factory=dict)


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
