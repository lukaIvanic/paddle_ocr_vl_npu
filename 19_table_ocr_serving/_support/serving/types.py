"""Request, result, and schedule contracts for PaddleOCR-VL serving."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import io
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class RecognitionRequest:
    request_id: str
    crop: Image.Image | bytes
    prompt: str
    skip_special_tokens: bool = True
    source_crop_size: tuple[int, int] | None = None
    submitted_at: float | None = None

    def resolve_image(self) -> RecognitionRequest:
        """Decode an already-cropped HTTP image on the CPU preparation worker."""
        if not isinstance(self.crop, bytes):
            return self
        with Image.open(io.BytesIO(self.crop)) as opened:
            crop = opened.convert("RGB")
        return replace(self, crop=crop, source_crop_size=self.source_crop_size or crop.size)


@dataclass
class RequestTiming:
    """Wall-clock seconds for one request, as reported in every result's timing_s.

    Filled once, in ContinuousRecognizer._result_from_completion, from the
    values the four pipeline stages recorded on the way through.
    """

    # CPU preparation, on the request source's background thread.
    cpu_image_decode: float  # encoded bytes to an RGB image; 0 when the crop arrived decoded
    cpu_image_and_prompt_preprocess: float  # resize, patchify, prompt tokens
    cpu_mrope_index: float  # multimodal rotary positions
    cpu_pin_memory: float  # pin the input tensors for async H2D
    cpu_preprocess_background_queue_wait: float  # submitted -> CPU thread started
    cpu_preprocess_background_service: float  # CPU thread work
    # Handoff to the NPU.
    cpu_preprocess_background_consumer_wait: float  # prefill waited for the CPU thread
    cpu_preprocess_background_ready_wait: float  # CPU finished -> picked up for prefill
    prefill_h2d_submit_host: float  # host time to submit the H2D copies
    # Prefill.
    prefill_enqueue_host: float  # host time to enqueue the prefill chain
    recognizer_h2d: float  # NPU time of the H2D copies
    first_token_d2h: float  # wait for the first token to copy back
    prefill_resolve_wait: float  # host time resolving device timing
    vision_and_text_prefill_wall: float  # enqueue start -> first token resolved
    time_to_first_token: float  # request submitted -> first token
    prefill_request_total: float  # sum of the CPU and prefill stages
    # Decode and result.
    decode_ready_queue_wait: float  # prefill finished -> decode slot admitted
    decode_slot_residency: float  # admitted -> finished generating
    detokenize: float  # token IDs to text
    request_total: float  # request submitted -> text ready


@dataclass
class PrefillDeviceTiming:
    """NPU seconds of each prefill stage, in execution order, as reported in device_stage_s."""

    recognition_inputs_h2d: float
    vision_embeddings: float
    vision_prefill_input_prep: float
    vision_prefill: float
    adaptive_mlp_projector: float
    text_token_embedding: float
    image_embed_scatter: float
    static_cache_alloc: float
    text_prefill_input_prep: float
    text_prefill: float
    prefill_lm_head: float
    prefill_argmax: float
    text_kv_redistribute: float  # always 0; kept because the summary schema lists it


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
    timing_s: RequestTiming
    device_stage_s: PrefillDeviceTiming
    rates: dict[str, float | None]
    vision: dict[str, Any] = field(default_factory=dict)
    text_prefill: dict[str, Any] = field(default_factory=dict)
    input_fingerprints: dict[str, Any] = field(default_factory=dict)
    repetition: dict[str, Any] = field(default_factory=dict)
    scheduling_metrics: dict[str, Any] = field(default_factory=dict)


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
    timing_s: dict[str, float | None]
    vision_packing: dict[str, Any]
    text_packing: dict[str, Any]
    rates: dict[str, float | None]
