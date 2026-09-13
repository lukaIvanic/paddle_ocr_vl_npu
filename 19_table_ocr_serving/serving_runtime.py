"""Persistent PaddleOCR-VL recognizer: CPU preparation, NPU prefill, batched decode.

One ContinuousRecognizer owns the model for the life of the inference process.
serve() takes an open request source (serve.py's InferenceWorker) and returns
once the source is closed and every accepted request has produced a result.

Each request travels through four stages:

1. CPU preparation, on a background thread: decode the image, resize and
   patchify it (turn 14x14 pixel patches to tokens), build the prompt tokens and rotary positions.
2. Staging: copy the prepared tensors to the NPU on a dedicated transfer stream.
3. Prefill, on the compute stream: vision encoder, projector, text prefill into
   a private KV cache slot, and the first generated token.
4. Decode: the request takes a slot in a fixed-size decode arena shared by all
   in-flight requests and generates until EOS or MAX_NEW_TOKENS. The finished
   token list is detokenized into a RecognitionResult and handed to emit_result.

The decode arena and scheduler live in _support/serving/continuous_decode.py.
This file coordinates them and owns stages 1 to 3.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import sys
import time
from collections import Counter, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
from tokenizers import Tokenizer

from _support.serving.continuous_decode import (
    ContinuousDecodeScheduler,
    DecodeCompletion,
    DecodeArena,
    ReadyDecodeRequest,
)
from _support.serving.prefill_cache_pool import PrefillKVCacheLease, PrefillKVCachePool
from _support.serving.scheduling_metrics import RequestSchedulingMetrics
from _support.serving.types import (
    PrefillDeviceTiming,
    RequestTiming,
    ContinuousDecodeResult,
    RecognitionRequest,
    RecognitionResult,
)
from _support.utils.metrics import per_second
from _support.utils.timing import DeviceTimeline, synchronize
from crop_processing import (
    prepare_prompt_tokens,
    preprocess_pil_image,
    PATCH_SIZE, MERGE_SIZE, MIN_PIXELS, MAX_PIXELS,
)
from paddle_ocr_vl_1_6_modeling import (
    LocalPaddleOCRVLForConditionalGeneration,
    IMAGE_TOKEN_ID,
)
from text_prefill_and_decode import (
    LocalPaddleOCRVLStaticCache,
    TEXT_EOS_TOKEN_ID,
    cast_decode_linear_weights_to_nz,
    load_decode_vocab_token_ids,
    prepare_decode_compact_lm_head,
    prepare_decode_projections,
)
from vision_prefill import (
    VISION_SEQUENCE_ALIGNMENT,
    prepare_vision_linear_weight_format,
    prepare_vision_mlp_intermediate,
    prepare_vision_attention_weight_padding,
)


# Fixed serving settings. They are part of the validated configuration for the
# supported checkpoint and are deliberately not exposed as options.

DTYPE = torch.float16
# KV cache rows per request: prompt tokens plus generated tokens must fit.
CACHE_LENGTH = 4096
MAX_NEW_TOKENS = 4096
# Prefill KV cache slots beyond the ready buffer, so prefill never waits on decode.
PRIVATE_CACHE_STAGING_HEADROOM = 32
VISION_ATTENTION = "prompt_flash_attention"
# The default 60,416-row decode vocabulary (see presets/table_compact_vocab/README.md).
DECODE_VOCAB_TOKEN_IDS_PATH = (
    Path(__file__).resolve().parent / "presets/table_compact_vocab/native_han_core_60416.json"
)


# Per-request timings are the RequestTiming and PrefillDeviceTiming records in
# _support/serving/types.py. Each stage below records its own part; the result
# step assembles them.


class ContinuousRecognizer:
    """One persistent model: independent prefill per crop, continuous batched decode."""

    # Request flow: serve() is the entrypoint; the methods below it follow one
    # request in order. Setup (__init__) and configuration() come after.

    @torch.inference_mode()
    def serve(
        self,
        requests: Any,
        *,
        schedule_id: str,
        emit_result: Callable[[RecognitionResult], None],
        on_request_error: Callable[[str, BaseException], None],
        collect_scheduling_metrics: bool = False,
    ) -> ContinuousDecodeResult:
        """Serve an open request source until it closes; results go to emit_result.

        `requests` provides pull(block=...) and closed (serve.py's InferenceWorker).
        The source may be temporarily empty without ending the run. Preparation
        failures are reported through on_request_error instead of stopping decode.
        The request loop itself is _OpenPrefillSource.pull, after this class.
        """
        self._vision_prefill_stats = _VisionPrefillStats()
        self._text_prefill_stats = _TextPrefillStats()
        scheduling_metrics = (
            RequestSchedulingMetrics(self.batch_size) if collect_scheduling_metrics else None
        )
        ready_source = _OpenPrefillSource(
            self,
            requests,
            on_request_error=on_request_error,
            scheduling_metrics=scheduling_metrics,
        )
        try:
            return self._run_decode(
                ready_source,
                schedule_id=schedule_id,
                emit_result=emit_result,
                scheduling_metrics=scheduling_metrics,
            )
        finally:
            ready_source.close()

    def _run_decode(
        self,
        ready_source: Any,
        *,
        schedule_id: str,
        emit_result: Callable[[RecognitionResult], None],
        scheduling_metrics: RequestSchedulingMetrics | None = None,
    ) -> ContinuousDecodeResult:
        """Run the decode scheduler over prefilled requests and summarize the run."""

        def handle_completion(completion: DecodeCompletion) -> None:
            emit_result(self._result_from_completion(completion, schedule_id=schedule_id))

        decoded = self.decode_scheduler.run_stream(
            ready_source,
            on_completion=handle_completion,
            ready_buffer_capacity=self.ready_buffer_capacity,
            ready_buffer_low_watermark=self.ready_buffer_low_watermark,
            scheduling_metrics=scheduling_metrics,
        )
        decode_wall_s = decoded.timing_s["continuous_decode_wall"]
        private_cache_pool_stats = self.prefill_cache_pool.stats()
        if int(private_cache_pool_stats["active_slots"]) != 0:
            raise RuntimeError(
                "prefill KV cache arena still owns active request slots after decode: "
                f"{private_cache_pool_stats}"
            )

        def fraction(numerator: int) -> float | None:
            if decoded.raw_decode_token_slots <= 0:
                return None
            return float(numerator) / float(decoded.raw_decode_token_slots)

        return ContinuousDecodeResult(
            schedule_id=schedule_id,
            batch_size=self.batch_size,
            requests=decoded.submitted_requests,
            ready_buffer_capacity=decoded.ready_buffer_capacity,
            ready_buffer_low_watermark=decoded.ready_buffer_low_watermark,
            max_ready_queue_depth=decoded.max_ready_queue_depth,
            ready_source_refill_count=decoded.ready_source_refill_count,
            graph_calls=decoded.graph_calls,
            initial_admissions=decoded.initial_admissions,
            hot_swap_admissions=decoded.hot_swap_admissions,
            prefill_only_completions=decoded.prefill_only_completions,
            raw_decode_token_slots=decoded.raw_decode_token_slots,
            active_decode_token_slots=decoded.active_decode_token_slots,
            effective_decode_tokens=decoded.effective_decode_tokens,
            idle_decode_token_slots=decoded.idle_decode_token_slots,
            lookahead_decode_token_slots=decoded.lookahead_decode_token_slots,
            kv_prefix_bytes_copied=decoded.kv_prefix_bytes_copied,
            initial_kv_prefix_bytes_copied=decoded.initial_kv_prefix_bytes_copied,
            hot_swap_kv_prefix_bytes_copied=decoded.hot_swap_kv_prefix_bytes_copied,
            timing_s=dict(decoded.timing_s),
            vision_packing=self._vision_prefill_stats.summary(),
            text_packing={
                **self._text_prefill_stats.summary(),
                "private_cache_pool": private_cache_pool_stats,
            },
            rates={
                "raw_decode_tok_per_s": per_second(decoded.raw_decode_token_slots, decode_wall_s),
                "effective_decode_tok_per_s": per_second(decoded.effective_decode_tokens, decode_wall_s),
                "effective_fraction": fraction(decoded.effective_decode_tokens),
                "active_slot_fraction": fraction(decoded.active_decode_token_slots),
                "effective_device_tok_per_s": per_second(
                    decoded.effective_decode_tokens,
                    decoded.timing_s["decode_model_and_argmax_device"],
                ),
                "scheduler_effective_tok_per_s": per_second(
                    decoded.effective_decode_tokens,
                    decoded.timing_s["run_scoped_scheduler_wall"],
                ),
            },
        )

    # Stage 1: CPU preparation. Runs on the request source's background thread.

    @torch.inference_mode()
    def _prepare_cpu(self, request: RecognitionRequest, submitted_at: float) -> PreparedCrop:
        preparation_started = time.perf_counter()
        image_decode_s = 0.0
        if isinstance(request.crop, bytes):
            started = time.perf_counter()
            request = request.resolve_image()
            image_decode_s = time.perf_counter() - started
        crop_size = tuple(int(value) for value in request.crop.size)

        started = time.perf_counter()
        pixel_values, image_grid_thw = preprocess_pil_image(request.crop)
        input_ids, attention_mask = prepare_prompt_tokens(
            self.preprocessing_tokenizer, image_grid_thw, request.prompt,
        )
        preprocess_s = time.perf_counter() - started

        prompt_length = int(input_ids.shape[1])
        if prompt_length > CACHE_LENGTH:
            raise ValueError(
                f"request {request.request_id} has prompt_length={prompt_length}, "
                f"configured cache_length={CACHE_LENGTH}"
            )
        image_token_count = int((input_ids == IMAGE_TOKEN_ID).sum().item())

        started = time.perf_counter()
        position_ids, rope_deltas = self.model.get_rope_index(
            input_ids, image_grid_thw, attention_mask,
        )
        mrope_index_s = time.perf_counter() - started

        started = time.perf_counter()
        input_ids = _pin_memory_or_keep(input_ids)
        attention_mask = _pin_memory_or_keep(attention_mask)
        pixel_values = _pin_memory_or_keep(pixel_values)
        position_ids = _pin_memory_or_keep(position_ids)
        rope_deltas = _pin_memory_or_keep(rope_deltas)
        pin_memory_s = time.perf_counter() - started

        preparation_finished = time.perf_counter()
        cpu_timing = CpuTiming(
            cpu_image_decode=image_decode_s,
            cpu_image_and_prompt_preprocess=preprocess_s,
            cpu_mrope_index=mrope_index_s,
            cpu_pin_memory=pin_memory_s,
            cpu_preprocess_background_queue_wait=max(0.0, preparation_started - submitted_at),
            cpu_preprocess_background_service=preparation_finished - preparation_started,
        )
        return PreparedCrop(
            request_id=request.request_id,
            prompt=request.prompt,
            crop_size=crop_size,
            skip_special_tokens=bool(request.skip_special_tokens),
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            rope_deltas=rope_deltas,
            image_token_count=image_token_count,
            cpu_timing=cpu_timing,
            request_started=submitted_at,
            preparation_finished=preparation_finished,
        )

    # Stages 2 and 3: one prepared crop becomes a request the decode arena can admit.

    def _prefill_for_decode(self, prepared: PreparedCrop, consumer_wait_s: float) -> ReadyDecodeRequest:
        """Stage, prefill, and package one crop for the decode scheduler."""
        staged = self._stage_crop(prepared, consumer_wait_s)
        inflight = self._enqueue_crop(staged)
        prefilled = self._finalize_crop(inflight)
        cache, rope_deltas, cache_position, first_token_tensor, cache_release = (
            prefilled.take_device_state()
        )
        return ReadyDecodeRequest(
            request_id=prefilled.request_id,
            payload=prefilled,
            cache=cache,
            rope_deltas=rope_deltas,
            cache_position=cache_position,
            first_token_tensor=first_token_tensor,
            first_token=prefilled.first_token,
            prompt_length=prefilled.input_tokens,
            cache_release=cache_release,
        )

    @torch.inference_mode()
    def _stage_crop(self, prepared: PreparedCrop, consumer_wait_s: float) -> StagedCrop:
        """Copy the prepared tensors to the NPU on the transfer stream, without waiting."""
        import torch_npu  # imported here so CPU tests can substitute a fake module

        device_timeline = DeviceTimeline(self.device)
        submit_started = time.perf_counter()
        with torch_npu.npu.stream(self.prefill_transfer_stream):
            ready_wait_s = max(0.0, time.perf_counter() - prepared.preparation_finished)

            def move_inputs() -> tuple[torch.Tensor, ...]:
                pixels = prepared.pixel_values.to(device=self.device, non_blocking=True)
                return (
                    prepared.input_ids.to(self.device, non_blocking=True),
                    prepared.attention_mask.to(self.device, non_blocking=True),
                    pixels,
                    prepared.position_ids.to(self.device, non_blocking=True),
                    prepared.rope_deltas.to(self.device, non_blocking=True),
                )

            device_inputs = device_timeline.measure("recognition_inputs_h2d", move_inputs)
            h2d_ready_event = self.prefill_transfer_stream.record_event()
        return StagedCrop(
            prepared=prepared,
            device_timeline=device_timeline,
            h2d_ready_event=h2d_ready_event,
            device_inputs=device_inputs,
            cpu_preprocess_background_consumer_wait=float(consumer_wait_s),
            cpu_preprocess_background_ready_wait=ready_wait_s,
            prefill_h2d_submit_host=time.perf_counter() - submit_started,
        )

    @torch.inference_mode()
    def _enqueue_crop(self, staged: StagedCrop) -> InFlightCrop:
        """Enqueue the whole prefill forward pass on the compute stream.

        In order: normalize pixels, vision embeddings, vision transformer,
        projector, text token embeddings with image embeddings scattered in,
        text prefill into a private KV cache slot, LM head, first-token argmax.
        Nothing here waits for the NPU; _finalize_crop does.
        """
        import torch_npu  # imported here so CPU tests can substitute a fake module

        prepared = staged.prepared
        measure = staged.device_timeline.measure
        enqueue_started = time.perf_counter()
        torch_npu.npu.current_stream().wait_event(staged.h2d_ready_event)
        prefill_started = time.perf_counter()
        input_ids, attention_mask, pixels, position_ids, rope_deltas = staged.device_inputs

        def normalize_uint8() -> torch.Tensor:
            output = pixels.to(torch.float32)
            output.mul_(1.0 / 255.0)
            output.sub_(0.5)
            output.div_(0.5)
            return output.to(self.model.visual.dtype).contiguous()

        pixels = measure("vision_input_normalize", normalize_uint8)
        vision_model = self.model.visual.vision_model
        hidden = measure("vision_embeddings", lambda: vision_model.embeddings(
            pixels.unsqueeze(0), image_grid_thw=prepared.image_grid_thw,
        ))
        real_length = int(hidden.shape[0])
        vision_route = self.vision_prefill.route(real_length)
        prepared_vision = measure("vision_prefill_input_prep", lambda: self.vision_prefill.prepare(
            hidden, prepared.image_grid_thw, route=vision_route,
        ))
        features = measure("vision_prefill", lambda: self.vision_prefill.run_prepared(prepared_vision))
        self._vision_prefill_stats.record(vision_route)
        image_embeds = measure("adaptive_mlp_projector", lambda: self.model.mlp_AR(features, prepared.image_grid_thw))
        inputs_embeds = measure("text_token_embedding", lambda: self.model.model.embed_tokens(input_ids))

        def scatter_image_embeds() -> torch.Tensor:
            projected = image_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            image_mask = (input_ids == IMAGE_TOKEN_ID).unsqueeze(-1).expand_as(inputs_embeds)
            return inputs_embeds.masked_scatter(image_mask, projected)

        inputs_embeds = measure("image_embed_scatter", scatter_image_embeds)
        lease = measure("static_cache_alloc", self.prefill_cache_pool.acquire)
        self._text_prefill_stats.record()
        text_route = self.text_prefill.route(int(inputs_embeds.shape[1]))
        prepared_text = measure("text_prefill_input_prep", lambda: self.text_prefill.prepare(
            inputs_embeds, attention_mask, position_ids, route=text_route,
        ))
        last_hidden = measure("text_prefill", lambda: self.text_prefill.run_prepared(prepared_text, lease.cache))
        logits = measure("prefill_lm_head", lambda: self.model.lm_head(last_hidden))
        next_token = measure("prefill_argmax", lambda: torch.argmax(logits[:, -1, :].float(), dim=-1, keepdim=True))
        text_route = {
            **text_route,
            "private_cache_slot_index": int(lease.slot_index),
            "private_cache_generation": int(lease.generation),
        }
        next_position = torch.full((1,), int(input_ids.shape[1]), device=self.device, dtype=torch.int64)
        # A one-element copy of the first token, owned by this request, for the D2H copy in _finalize_crop.
        first_token_device = torch.cat([next_token.detach().reshape(-1)], dim=0).contiguous()
        prefill_ready_event = torch_npu.npu.current_stream().record_event()
        return InFlightCrop(
            staged=staged,
            cache=lease.cache,
            cache_lease=lease,
            rope_deltas=rope_deltas,
            next_cache_position=next_position,
            next_token=next_token,
            vision=vision_route,
            text_prefill=text_route,
            input_tokens=int(prepared.input_ids.shape[1]),
            projected_image_tokens=int(image_embeds.shape[0]),
            prefill_ready_event=prefill_ready_event,
            first_token_device=first_token_device,
            prefill_started=prefill_started,
            prefill_enqueue_host=time.perf_counter() - enqueue_started,
        )

    @torch.inference_mode()
    def _finalize_crop(self, inflight: InFlightCrop) -> PrefilledCrop:
        """Wait for the first token, read it back, and collect the prefill timings."""
        import torch_npu  # imported here so CPU tests can substitute a fake module

        resolve_started = time.perf_counter()
        spans = inflight.staged.device_timeline.resolve_spans()
        started = time.perf_counter()
        with torch_npu.npu.stream(self.prefill_transfer_stream):
            self.prefill_transfer_stream.wait_event(inflight.prefill_ready_event)
            self.prefill_host_tokens[:1].copy_(inflight.first_token_device, non_blocking=True)
            first_token_ready = self.prefill_transfer_stream.record_event()
        first_token_ready.synchronize()
        first_token = int(self.prefill_host_tokens[:1].tolist()[0])
        first_token_d2h_s = time.perf_counter() - started
        resolve_finished = time.perf_counter()

        staged = inflight.staged
        prepared = staged.prepared
        cpu = prepared.cpu_timing

        def seconds(stage: str) -> float:
            return float(spans[stage]["seconds"])

        device_timing = PrefillDeviceTiming(
            recognition_inputs_h2d=seconds("recognition_inputs_h2d"),
            vision_embeddings=seconds("vision_embeddings"),
            vision_prefill_input_prep=seconds("vision_prefill_input_prep"),
            vision_prefill=seconds("vision_prefill"),
            adaptive_mlp_projector=seconds("adaptive_mlp_projector"),
            text_token_embedding=seconds("text_token_embedding"),
            image_embed_scatter=seconds("image_embed_scatter"),
            static_cache_alloc=seconds("static_cache_alloc"),
            text_prefill_input_prep=seconds("text_prefill_input_prep"),
            text_prefill=seconds("text_prefill"),
            prefill_lm_head=seconds("prefill_lm_head"),
            prefill_argmax=seconds("prefill_argmax"),
            text_kv_redistribute=0.0,
        )
        # vision_input_normalize is measured too but has never been reported.
        prefill_wall_s = resolve_finished - inflight.prefill_started
        prefill_timing = PrefillTiming(
            cpu_preprocess_background_consumer_wait=staged.cpu_preprocess_background_consumer_wait,
            cpu_preprocess_background_ready_wait=staged.cpu_preprocess_background_ready_wait,
            prefill_h2d_submit_host=staged.prefill_h2d_submit_host,
            prefill_enqueue_host=inflight.prefill_enqueue_host,
            recognizer_h2d=device_timing.recognition_inputs_h2d,
            first_token_d2h=first_token_d2h_s,
            prefill_resolve_wait=resolve_finished - resolve_started,
            vision_and_text_prefill_wall=prefill_wall_s,
            time_to_first_token=resolve_finished - prepared.request_started,
            prefill_request_total=(
                cpu.cpu_image_and_prompt_preprocess + cpu.cpu_mrope_index + cpu.cpu_pin_memory
                + device_timing.recognition_inputs_h2d + prefill_wall_s + first_token_d2h_s
            ),
        )
        return PrefilledCrop(
            request_id=prepared.request_id,
            prompt=prepared.prompt,
            crop_size=prepared.crop_size,
            skip_special_tokens=prepared.skip_special_tokens,
            cache=inflight.cache,
            cache_release=inflight.cache_lease.release,
            rope_deltas=inflight.rope_deltas,
            next_cache_position=inflight.next_cache_position,
            next_token=inflight.next_token,
            first_token=first_token,
            input_tokens=inflight.input_tokens,
            projected_image_tokens=inflight.projected_image_tokens,
            vision=inflight.vision,
            text_prefill=inflight.text_prefill,
            cpu_timing=cpu,
            prefill_timing=prefill_timing,
            device_timing=device_timing,
            request_started=prepared.request_started,
            prefill_finished=resolve_finished,
        )

    # Stage 4 ends here: the decode scheduler reports a finished request.

    def _result_from_completion(
        self, completion: DecodeCompletion, *, schedule_id: str,
    ) -> RecognitionResult:
        """Detokenize one finished request and attach its timings."""
        prefilled: PrefilledCrop = completion.ready.payload
        token_ids = completion.token_ids
        started = time.perf_counter()
        text = self.tokenizer.decode(token_ids, skip_special_tokens=prefilled.skip_special_tokens)
        detokenize_s = time.perf_counter() - started

        generated_tokens = len(token_ids)
        admitted_at = completion.admitted_at
        cpu = prefilled.cpu_timing
        prefill = prefilled.prefill_timing
        timing = RequestTiming(
            cpu_image_decode=cpu.cpu_image_decode,
            cpu_image_and_prompt_preprocess=cpu.cpu_image_and_prompt_preprocess,
            cpu_mrope_index=cpu.cpu_mrope_index,
            cpu_pin_memory=cpu.cpu_pin_memory,
            cpu_preprocess_background_queue_wait=cpu.cpu_preprocess_background_queue_wait,
            cpu_preprocess_background_service=cpu.cpu_preprocess_background_service,
            cpu_preprocess_background_consumer_wait=prefill.cpu_preprocess_background_consumer_wait,
            cpu_preprocess_background_ready_wait=prefill.cpu_preprocess_background_ready_wait,
            prefill_h2d_submit_host=prefill.prefill_h2d_submit_host,
            prefill_enqueue_host=prefill.prefill_enqueue_host,
            recognizer_h2d=prefill.recognizer_h2d,
            first_token_d2h=prefill.first_token_d2h,
            prefill_resolve_wait=prefill.prefill_resolve_wait,
            vision_and_text_prefill_wall=prefill.vision_and_text_prefill_wall,
            time_to_first_token=prefill.time_to_first_token,
            prefill_request_total=prefill.prefill_request_total,
            decode_ready_queue_wait=(
                max(0.0, admitted_at - prefilled.prefill_finished) if admitted_at is not None else 0.0
            ),
            decode_slot_residency=(
                max(0.0, completion.completed_at - admitted_at) if admitted_at is not None else 0.0
            ),
            detokenize=float(detokenize_s),
            request_total=float(completion.completed_at - prefilled.request_started + detokenize_s),
        )
        return RecognitionResult(
            request_id=prefilled.request_id,
            decode_schedule_id=schedule_id,
            decode_slot_index=completion.slot_index,
            decode_slot_epoch=completion.slot_epoch,
            prompt=prefilled.prompt,
            crop_size=prefilled.crop_size,
            text=text,
            token_ids=token_ids,
            stop_reason=completion.stop_reason,
            input_tokens=prefilled.input_tokens,
            projected_image_tokens=prefilled.projected_image_tokens,
            generated_tokens_including_eos=generated_tokens,
            decode_tokens_after_prefill_including_eos=max(0, generated_tokens - 1),
            decode_calls_executed=completion.iterations_launched,
            timing_s=timing,
            device_stage_s=prefilled.device_timing,
            rates={
                "request_output_tok_per_s": per_second(generated_tokens, timing.request_total),
            },
            vision=dict(prefilled.vision),
            text_prefill=dict(prefilled.text_prefill),
            scheduling_metrics=dict(completion.scheduling_metrics),
            repetition=(
                dict(completion.repetition_evidence)
                if completion.repetition_evidence is not None
                else {}
            ),
        )

    # Setup: load the model once and prepare every stage for the process lifetime.

    @torch.inference_mode()
    def __init__(
        self,
        *,
        model: str,
        device: str = "npu:0",
        batch_size: int,
        torchair_cache_dir: Path,
        vision_torchair_cache_dir: Path,
        text_torchair_cache_dir: Path,
        full_decode_lm_head: bool = False,
        decode_device_timing: bool = True,
        eager: bool = False,
    ):
        runtime_started = time.perf_counter()
        import torch_npu  # imported here so CPU tests can substitute a fake module

        self.model_dir = Path(model).expanduser()
        self.device = torch.device(device)
        if self.device.type != "npu":
            raise ValueError("table serving requires an NPU device")
        if not torch.npu.is_available():
            raise RuntimeError("Table serving requires an available NPU")
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dtype = DTYPE
        self.eager = eager
        self.decode_backend = "raw_eager" if eager else "torchair"
        self.full_decode_lm_head = bool(full_decode_lm_head)
        self.decode_device_timing = bool(decode_device_timing)
        torch.npu.config.allow_internal_format = True  # allow NZ weight layouts
        torch.npu.set_compile_mode(jit_compile=False)
        self.setup_timing_s: dict[str, float] = {}

        with self._setup_stage("frontend"):
            # One tokenizer per thread: the CPU preparation thread builds prompts,
            # the decode thread detokenizes results.
            self.preprocessing_tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
            self.tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        with self._setup_stage("model_load"):
            self.model = LocalPaddleOCRVLForConditionalGeneration.from_pretrained(
                self.model_dir, dtype=self.dtype, device=self.device
            )
        with self._setup_stage("decode_lm_head"):
            decode_head_cache_key = self._prepare_decode_lm_head()
        with self._setup_stage("vision_mlp_padding"):
            self.vision_mlp = prepare_vision_mlp_intermediate(self.model)
        with self._setup_stage("vision_attention_weight_padding"):
            prepare_vision_attention_weight_padding(self.model)
        with self._setup_stage("vision_weight_format"):
            self.vision_weight_format = prepare_vision_linear_weight_format(self.model)
        with self._setup_stage("decode_optimization_setup"):
            prepare_decode_projections(self.model)
        with self._setup_stage("decode_weight_format"):
            self.weight_format = cast_decode_linear_weights_to_nz(self.model)

        # Compiled (or eager) vision prefill, text prefill, and text decode stages.
        self.stages = self.model.make_inference_stages(
            vision_cache_root=vision_torchair_cache_dir,
            text_cache_root=text_torchair_cache_dir,
            decode_cache_root=torchair_cache_dir / decode_head_cache_key,
            batch_size=self.batch_size,
            cache_length=CACHE_LENGTH,
            device=self.device,
            eager=self.eager,
            setup_progress=_emit_setup_progress,
        )
        self.setup_timing_s.update(self.stages.setup_timing_s)
        self.vision_prefill = self.stages.vision_prefill
        self.text_prefill = self.stages.text_prefill
        self.text_decode = self.stages.text_decode
        synchronize(self.device)

        # Request flow buffers. The CPU thread may run this many requests ahead;
        # at most batch_size prefilled requests wait for a decode slot.
        self.prefill_transfer_stream = torch_npu.npu.Stream(device=self.device)
        self.cpu_preprocess_max_pending = max(2, self.batch_size)
        self.ready_buffer_capacity = self.batch_size
        self.ready_buffer_low_watermark = max(1, self.ready_buffer_capacity // 2)
        # Pinned host buffer that receives each request's first token; only row 0 is used.
        self.prefill_host_tokens = torch.empty(
            (max(self.cpu_preprocess_max_pending + 1, PRIVATE_CACHE_STAGING_HEADROOM),),
            dtype=torch.int64,
            pin_memory=True,
        )
        self._vision_prefill_stats = _VisionPrefillStats()
        self._text_prefill_stats = _TextPrefillStats()

        with self._setup_stage("private_cache_pool"):
            # Every prefill writes into its own KV cache slot; decode copies the
            # prefix into the arena and releases the slot.
            private_cache_storage = self.model.allocate_static_cache(
                batch_size=self.ready_buffer_capacity + PRIVATE_CACHE_STAGING_HEADROOM,
                cache_length=CACHE_LENGTH,
                device=self.device,
                dtype=self.dtype,
            )
            self.prefill_cache_pool = PrefillKVCachePool(private_cache_storage, device=self.device)
        with self._setup_stage("decode_control"):
            self.decode_arena = DecodeArena(
                cache=self.text_decode.warm_cache,
                device=self.device,
                batch_size=self.batch_size,
                eos_token_id=int(TEXT_EOS_TOKEN_ID),
                decode_device_timing=self.decode_device_timing,
            )
            self.decode_scheduler = ContinuousDecodeScheduler(
                arena=self.decode_arena,
                decode_fn=self.text_decode.fn,
                max_new_tokens=MAX_NEW_TOKENS,
                stop_repetitions=True,
            )
        self.setup_timing_s["recognizer_runtime_total"] = time.perf_counter() - runtime_started
        _emit_setup_progress("recognizer_runtime", "done", self.setup_timing_s["recognizer_runtime_total"])

    @contextmanager
    def _setup_stage(self, name: str) -> Iterator[None]:
        """Log and time one setup stage; waits for the NPU so the time is real."""
        _emit_setup_progress(name, "start")
        started = time.perf_counter()
        yield
        synchronize(self.device)
        self.setup_timing_s[name] = time.perf_counter() - started
        _emit_setup_progress(name, "done", self.setup_timing_s[name])

    def _prepare_decode_lm_head(self) -> str:
        """Select the decode vocabulary and return the decode graph cache key."""
        full_vocab_size = int(self.model.lm_head.weight.shape[0])
        if self.full_decode_lm_head:
            # The checkpoint head emits native token IDs directly.
            self.decode_vocab = {
                "enabled": False,
                "path": None,
                "full_vocab_size": full_vocab_size,
                "selected_vocab_size": full_vocab_size,
                "token_ids_sha256": None,
            }
            return f"full_vocab_{full_vocab_size}"
        token_ids, self.decode_vocab = load_decode_vocab_token_ids(
            DECODE_VOCAB_TOKEN_IDS_PATH, full_vocab_size=full_vocab_size,
        )
        prepare_decode_compact_lm_head(self.model, token_ids)
        return (
            f"selected_vocab_{self.decode_vocab['selected_vocab_size']}_"
            f"{self.decode_vocab['token_ids_sha256'][:12]}"
        )

    def configuration(self) -> dict[str, Any]:
        """What was loaded and how it is scheduled; reported by /ready and the summary."""
        return {
            "recognizer_model": str(self.model_dir),
            "device": str(self.device),
            "dtype": str(self.dtype),
            "decode_backend": self.decode_backend,
            "decode_vocab": dict(self.decode_vocab),
            "full_decode_lm_head": self.full_decode_lm_head,
            "token_selection": "greedy",
            "cache_length": CACHE_LENGTH,
            "max_new_tokens": MAX_NEW_TOKENS,
            "batch_size": self.batch_size,
            "vision_prefill": self.vision_prefill.metadata,
            "vision_mlp": dict(self.vision_mlp),
            "vision_linear_weight_format": dict(self.vision_weight_format),
            "vision_backend": self.decode_backend,
            "vision_attention": VISION_ATTENTION,
            "decode_device_timing": self.decode_device_timing,
            "compact_decode_control": False,
            "vision_sequence_alignment": VISION_SEQUENCE_ALIGNMENT,
            "vision_packing": {
                "mode": "off",
                "target": 1920,
                "lookahead": 32,
                "grouping": "independent_crops",
                "oversized": "faithful_eager_single_crop_route",
                "batched_runtime": None,
            },
            "vision_prompt_fa_layout": "bnsd",
            "text_prefill": self.text_prefill.metadata,
            "text_backend": self.decode_backend,
            "text_packing": {
                "mode": "off",
                "buckets": [128, 256, 512, 1024],
                "max_members": PRIVATE_CACHE_STAGING_HEADROOM,
                "grouping": "independent_crops",
                "runtime": None,
            },
            "preprocessor": {
                "effective_min_pixels": MIN_PIXELS,
                "effective_max_pixels": MAX_PIXELS,
                "patch_size": PATCH_SIZE,
                "merge_size": MERGE_SIZE,
                "resize_factor": PATCH_SIZE * MERGE_SIZE,
                "nominal_minimum_projected_image_tokens": MIN_PIXELS // ((PATCH_SIZE * MERGE_SIZE) ** 2),
            },
            "cpu_preprocessing": {
                "execution": "background_thread",
                "workers": 1,
                "max_pending": self.cpu_preprocess_max_pending,
                "ordering": "fifo",
                "pin_recognition_inputs": "best_effort",
            },
            "prefill_production": "next_crop_h2d_staged_before_ready_yield",
            "prefill_transfer": "dedicated_stream_event_dependencies",
            "decode": f"{'eager' if self.eager else 'compiled'}_static_b{self.batch_size}",
            "decode_schedule": "run_scoped_persistent_slots_iteration_hot_swap",
            "ready_buffer_capacity": self.ready_buffer_capacity,
            "ready_buffer_low_watermark": self.ready_buffer_low_watermark,
            "private_cache_staging_headroom": PRIVATE_CACHE_STAGING_HEADROOM,
            "decode_completion_detection": "queue_depth_one_async_token_copy",
            "private_prefill_cache": self.prefill_cache_pool.stats(),
            "kv_admission": "full_prefill_cache_foreach_copy_into_fixed_slot",
            "text_decode": self.text_decode.metadata,
            "linear_weight_format": self.weight_format,
        }


# The request loop: CPU-prepare ahead, prefill only when a decode slot is free.


class _OpenPrefillSource:
    """The decode scheduler's view of the open request stream.

    Requests are pulled from the caller's source and CPU-prepared on one
    background thread, up to cpu_preprocess_max_pending ahead. NPU prefill
    happens only when the scheduler asks with a free decode slot, so prefilled
    KV never piles up waiting for decode capacity.
    """

    def __init__(
        self,
        recognizer: ContinuousRecognizer,
        requests: Any,
        *,
        on_request_error: Callable[[str, BaseException], None],
        scheduling_metrics: RequestSchedulingMetrics | None = None,
    ):
        self.recognizer = recognizer
        self.requests = requests
        self.on_request_error = on_request_error
        self.scheduling_metrics = scheduling_metrics
        # Requests handed to the CPU thread and not yet prefilled, oldest first.
        self.pending: deque[tuple[str, Future[PreparedCrop]]] = deque()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="paddleocr-vl-open-cpu-prepare")
        self._executor_closed = False

    def pull_for_decode_slots(self, *, block: bool, available_slots: int) -> ReadyDecodeRequest | None:
        """Called by the decode scheduler; prefill is allowed only with a free slot."""
        return self.pull(block=block, allow_prefill=available_slots > 0)

    def pull(self, *, block: bool, allow_prefill: bool = True) -> ReadyDecodeRequest | None:
        """Return the next prefilled request, or None when none is ready right now."""
        while True:
            pull_started = time.perf_counter() if self.scheduling_metrics is not None else 0.0
            self._submit_available(block_for_first=allow_prefill and block and not self.pending)
            if not allow_prefill:
                # Every decode slot is busy or reserved. CPU preparation above
                # keeps running ahead; NPU prefill waits for a free slot.
                return None
            if not self.pending:
                return None
            if self.scheduling_metrics is not None:
                self.scheduling_metrics.cpu_prefill_eligible(self.pending[0][0], block=block)
            if not block and not self.pending[0][1].done():
                # Live decoding must not stall on CPU work. Only an idle
                # scheduler (block=True) waits for the first prepared request.
                return None
            request_id, future = self.pending.popleft()
            wait_started = time.perf_counter()
            try:
                prepared = future.result()
            except BaseException as exc:
                if self.scheduling_metrics is not None:
                    self.scheduling_metrics.record_prefill(
                        request_id, pull_started, time.perf_counter(), status="error",
                    )
                self.on_request_error(request_id, exc)
                block = False
                continue
            consumer_wait_s = time.perf_counter() - wait_started
            if self.scheduling_metrics is not None:
                self.scheduling_metrics.cpu_prepared(
                    request_id, submitted_at=prepared.request_started,
                    queue_wait_s=prepared.cpu_timing.cpu_preprocess_background_queue_wait,
                    finished_at=prepared.preparation_finished, consumed_at=wait_started,
                )
            # Refill the CPU lane first so the next request's preparation overlaps this prefill.
            self._submit_available(block_for_first=False)
            ready = self.recognizer._prefill_for_decode(prepared, consumer_wait_s)
            if self.scheduling_metrics is not None:
                self.scheduling_metrics.record_prefill(request_id, pull_started, time.perf_counter())
            return ready

    def _submit_available(self, *, block_for_first: bool) -> None:
        """Hand new requests to the CPU thread until the lookahead limit is reached."""
        while len(self.pending) < self.recognizer.cpu_preprocess_max_pending:
            request = self.requests.pull(block=block_for_first and not self.pending)
            block_for_first = False
            if request is None:
                break
            submitted_at = time.perf_counter()
            if self.scheduling_metrics is not None:
                self.scheduling_metrics.register(
                    request.request_id,
                    submitted_at if request.submitted_at is None else request.submitted_at,
                )
            self.pending.append(
                (request.request_id, self.executor.submit(self.recognizer._prepare_cpu, request, submitted_at))
            )

    @property
    def closed(self) -> bool:
        return bool(self.requests.closed) and not self.pending

    def close(self) -> None:
        if self._executor_closed:
            return
        self._executor_closed = True
        self.executor.shutdown(wait=True, cancel_futures=True)


# One request's state as it moves through the stages.


# The RequestTiming fields each stage produces, named exactly as they appear there.


@dataclass
class CpuTiming:
    """The CPU preparation part of RequestTiming."""

    cpu_image_decode: float
    cpu_image_and_prompt_preprocess: float
    cpu_mrope_index: float
    cpu_pin_memory: float
    cpu_preprocess_background_queue_wait: float
    cpu_preprocess_background_service: float


@dataclass
class PrefillTiming:
    """The NPU handoff and prefill part of RequestTiming."""

    cpu_preprocess_background_consumer_wait: float
    cpu_preprocess_background_ready_wait: float
    prefill_h2d_submit_host: float
    prefill_enqueue_host: float
    recognizer_h2d: float
    first_token_d2h: float
    prefill_resolve_wait: float
    vision_and_text_prefill_wall: float
    time_to_first_token: float
    prefill_request_total: float


@dataclass
class PreparedCrop:
    """Output of CPU preparation: everything the NPU needs, still in host memory."""

    request_id: str
    prompt: str
    crop_size: tuple[int, int]
    skip_special_tokens: bool
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    rope_deltas: torch.Tensor
    image_token_count: int
    cpu_timing: CpuTiming
    request_started: float
    preparation_finished: float


@dataclass
class StagedCrop:
    """Inputs submitted to the NPU; h2d_ready_event fires when they have arrived."""

    prepared: PreparedCrop
    device_timeline: DeviceTimeline
    h2d_ready_event: Any
    # Kept referenced until prefill has finished reading them on the NPU.
    device_inputs: tuple[torch.Tensor, ...]
    cpu_preprocess_background_consumer_wait: float
    cpu_preprocess_background_ready_wait: float
    prefill_h2d_submit_host: float


@dataclass
class InFlightCrop:
    """Prefill enqueued; prefill_ready_event fires when the first token exists."""

    staged: StagedCrop
    cache: LocalPaddleOCRVLStaticCache
    cache_lease: PrefillKVCacheLease
    rope_deltas: torch.Tensor
    next_cache_position: torch.Tensor
    next_token: torch.Tensor
    vision: dict[str, Any]
    text_prefill: dict[str, Any]
    input_tokens: int
    projected_image_tokens: int
    prefill_ready_event: Any
    first_token_device: torch.Tensor
    prefill_started: float
    prefill_enqueue_host: float


@dataclass
class PrefilledCrop:
    """Prefill complete. Travels with the request through decode as its payload."""

    request_id: str
    prompt: str
    crop_size: tuple[int, int]
    skip_special_tokens: bool
    cache: LocalPaddleOCRVLStaticCache | None
    cache_release: Callable[[], None] | None
    rope_deltas: torch.Tensor | None
    next_cache_position: torch.Tensor | None
    next_token: torch.Tensor | None
    first_token: int
    input_tokens: int
    projected_image_tokens: int
    vision: dict[str, Any]
    text_prefill: dict[str, Any]
    cpu_timing: CpuTiming
    prefill_timing: PrefillTiming
    device_timing: PrefillDeviceTiming
    request_started: float
    prefill_finished: float

    def take_device_state(
        self,
    ) -> tuple[LocalPaddleOCRVLStaticCache, torch.Tensor, torch.Tensor, torch.Tensor, Callable[[], None] | None]:
        """Hand the NPU prefix to the decode arena; this record keeps only host data."""
        cache = self.cache
        rope_deltas = self.rope_deltas
        next_cache_position = self.next_cache_position
        next_token = self.next_token
        cache_release = self.cache_release
        if cache is None or rope_deltas is None or next_cache_position is None or next_token is None:
            raise RuntimeError(f"prefill device state already taken for {self.request_id}")
        self.cache = None
        self.cache_release = None
        self.rope_deltas = None
        self.next_cache_position = None
        self.next_token = None
        return cache, rope_deltas, next_cache_position, next_token, cache_release


# Helpers and run statistics.


def _pin_memory_or_keep(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.device.type != "cpu" or tensor.is_pinned():
        return tensor
    try:
        return tensor.pin_memory()
    except RuntimeError:
        # Pageable staging is slower to submit but has identical semantics.
        return tensor


def _emit_setup_progress(stage: str, status: str, elapsed_s: float | None = None) -> None:
    """One stderr line per setup stage, so a slow startup shows where it is."""
    record: dict[str, Any] = {"stage": str(stage), "status": str(status)}
    if elapsed_s is not None:
        record["elapsed_s"] = round(float(elapsed_s), 6)
    print("SETUP " + json.dumps(record, ensure_ascii=False, separators=(",", ":")), file=sys.stderr, flush=True)


@dataclass
class _VisionPrefillStats:
    calls: int = 0
    overflows: int = 0
    shapes: Counter = field(default_factory=Counter)

    def record(self, route: dict[str, Any]) -> None:
        self.calls += 1
        shape = (f"b1_s{route['physical_vision_tokens']}"
                 if route["execution"] == "compiled" else route["execution"])
        self.shapes[shape] += 1
        self.overflows += int(route["execution"] == "eager_overflow")

    def summary(self) -> dict[str, Any]:
        # The summary schema predates this runtime; the packing fields are constants here.
        return dict(mode="off", target=1920, lookahead=32, groups=self.calls,
            crops=self.calls, packed_groups=0, singleton_groups=self.calls,
            eager_overflow_groups=self.overflows,
            crops_per_group=1.0 if self.calls else None,
            group_size_histogram={"1": self.calls} if self.calls else {},
            graph_shape_histogram=dict(sorted(self.shapes.items())),
            ready_window_histogram={}, router_cpu_s=0.0, packed_real_vision_tokens=0,
            packed_physical_vision_tokens=0, packed_fill_fraction=None)


@dataclass
class _TextPrefillStats:
    calls: int = 0

    def record(self) -> None:
        self.calls += 1

    def summary(self) -> dict[str, Any]:
        # The summary schema predates this runtime; the packing fields are constants here.
        return dict(mode="off", buckets=[128, 256, 512, 1024], groups=self.calls,
            crops=self.calls, packs=0, packed_crops=0, fallback_crops=self.calls,
            calls=self.calls, call_reduction_fraction=0.0 if self.calls else None,
            pack_size_histogram={}, bucket_histogram={}, packed_real_text_tokens=0,
            packed_physical_text_tokens=0, packed_fill_fraction=None,
            redistributed_kv_bytes=0)
