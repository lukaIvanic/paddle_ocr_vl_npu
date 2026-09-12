"""Persistent PaddleOCR-VL runtime with pipelined prefill and batched decode."""

from __future__ import annotations

import hashlib
import json
import sys
import time
import threading
from collections import Counter, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

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
from paddle_ocr_vl_1_6_modeling import (
    LocalPaddleOCRVLForConditionalGeneration,
    IMAGE_TOKEN_ID,
)
from text_prefill_and_decode import (
    LocalPaddleOCRVLStaticCache,
    cast_decode_linear_weights_to_nz,
    load_decode_vocab_token_ids,
    prepare_decode_compact_lm_head,
    prepare_decode_projections,
)
from crop_processing import (
    prepare_prompt_tokens,
    preprocess_pil_image,
    PATCH_SIZE, MERGE_SIZE, MIN_PIXELS, MAX_PIXELS,
)
from text_prefill_and_decode import TEXT_PREFILL_BUCKETS, TEXT_EOS_TOKEN_ID
from _support.serving.types import (
    ContinuousDecodeResult,
    RecognitionRequest,
    RecognitionResult,
)
from _support.utils.timing import DeviceTimeline, synchronize
from _support.utils.timeline import TimelineRecorder
from _support.utils.metrics import per_second
from _support.utils.input_fingerprints import fingerprint_recognition_inputs
from vision_prefill import (
    VISION_BUCKETS,
    VISION_SEQUENCE_ALIGNMENT,
    prepare_vision_linear_weight_format,
    prepare_vision_mlp_intermediate,
    prepare_vision_attention_weight_padding,
)


# Persistent recognizer: request flow before setup details


class ContinuousRecognizer:
    """One persistent model with sequential prefill and continuous decode.

    Every real crop is prefilled independently. Vision prefill, text prefill,
    and text decode use the fixed serving implementations and compiled buckets.
    The inherited overflow handling still executes the same prefill stage.
    A fixed decode arena keeps its tensor shapes
    stable while ready KV prefixes replace finished requests between steps.
    """

    # Request entrypoints: open serving stream and finite request iterable.
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
        """Serve an open stream whose input can be temporarily empty.

        Each arrival remains one independent crop request. The model stages
        stay unchanged, while ready crops enter the same fixed decode arena
        and can hot-swap into free slots until the caller closes the source.
        """

        self._begin_decode_schedule()
        scheduling_metrics = (
            RequestSchedulingMetrics(self.batch_size)
            if collect_scheduling_metrics
            else None
        )
        ready_source = _OpenPrefillSource(
            self,
            requests,
            on_request_error=on_request_error,
            scheduling_metrics=scheduling_metrics,
        )
        try:
            return self._decode_ready_source(
                ready_source,
                schedule_id=schedule_id,
                emit_result=emit_result,
                scheduling_metrics=scheduling_metrics,
            )
        finally:
            ready_source.close()

    @torch.inference_mode()
    def run(
        self,
        requests: Iterable[RecognitionRequest],
        *,
        schedule_id: str,
        emit_result: Callable[[RecognitionResult], None],
    ) -> ContinuousDecodeResult:
        """Emit independent crop results as they complete.

        Request ordering and higher-level grouping belong to the caller. The
        return value contains only run-scoped scheduler metrics, which become
        final after the input stream is drained.
        """

        self._begin_decode_schedule()

        def ready_stream() -> Iterable[ReadyDecodeRequest]:
            current_staged: _StagedCrop | None = None
            drained_normally = False
            h2d_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="prefill-h2d",
            )
            self._emit_scheduler_progress("ready_stream_begin")
            try:
                crops = self._iter_prepared_crops(requests)
                crop_source = iter(crops)
                try:
                    self._emit_scheduler_progress(
                        "prefill_crop_source_next_begin",
                        phase="first",
                    )
                    first_crop = next(crop_source)
                except StopIteration:
                    self._emit_scheduler_progress(
                        "prefill_crop_source_exhausted",
                        phase="first",
                    )
                    drained_normally = True
                    return
                self._emit_scheduler_progress(
                    "prefill_crop_source_next_end",
                    phase="first",
                    crop_id=first_crop.crop_id,
                    crops=1,
                    real_vision_tokens=first_crop.real_vision_tokens,
                )
                self._emit_scheduler_progress(
                    "prefill_h2d_stage_begin",
                    crop_id=first_crop.crop_id,
                    crops=1,
                )
                current_staged = self._stage_crop(first_crop)
                self._emit_scheduler_progress(
                    "prefill_h2d_stage_end",
                    crop_id=first_crop.crop_id,
                    crops=1,
                )
                while current_staged is not None:
                    try:
                        self._emit_scheduler_progress(
                            "prefill_crop_source_next_begin",
                            phase="lookahead",
                            current_crop_id=current_staged.crop.crop_id,
                        )
                        next_crop = next(crop_source)
                    except StopIteration:
                        self._emit_scheduler_progress(
                            "prefill_crop_source_exhausted",
                            phase="lookahead",
                            current_crop_id=current_staged.crop.crop_id,
                        )
                        next_stage_future = None
                    else:
                        self._emit_scheduler_progress(
                            "prefill_crop_source_next_end",
                            phase="lookahead",
                            current_crop_id=current_staged.crop.crop_id,
                            crop_id=next_crop.crop_id,
                            crops=1,
                            real_vision_tokens=next_crop.real_vision_tokens,
                        )
                        # TorchAir occupies this thread for much of G's device
                        # work. Submit only G+1's H2D on a dedicated host
                        # worker while this thread invokes G's compute chain.
                        next_stage_future = h2d_executor.submit(
                            self._stage_crop,
                            next_crop,
                        )

                    self._emit_scheduler_progress(
                        "prefill_enqueue_begin",
                        crop_id=current_staged.crop.crop_id,
                        crops=1,
                        real_vision_tokens=current_staged.crop.real_vision_tokens,
                    )
                    final = self._enqueue_crop(current_staged)
                    self._emit_scheduler_progress(
                        "prefill_enqueue_end",
                        crop_id=final.crop_id,
                        crops=1,
                    )
                    current_staged = None
                    if next_stage_future is not None:
                        self._emit_scheduler_progress(
                            "prefill_lookahead_h2d_wait_begin",
                            current_crop_id=final.crop_id,
                        )
                    next_staged = (
                        None
                        if next_stage_future is None
                        else next_stage_future.result()
                    )
                    if next_stage_future is not None:
                        assert next_staged is not None
                        self._emit_scheduler_progress(
                            "prefill_lookahead_h2d_wait_end",
                            current_crop_id=final.crop_id,
                            next_crop_id=next_staged.crop.crop_id,
                        )
                    self._emit_scheduler_progress(
                        "prefill_finalize_begin",
                        crop_id=final.crop_id,
                        crops=1,
                    )
                    finalized = self._finalize_crop(final)
                    self._emit_scheduler_progress(
                        "prefill_finalize_end",
                        crop_id=final.crop_id,
                        crops=1,
                    )
                    self._emit_scheduler_progress(
                        "ready_state_yield", crop_id=final.crop_id,
                        request_id=finalized.request_id,
                    )
                    yield self._ready_from_prefilled(finalized)
                    self._emit_scheduler_progress(
                        "prefill_crop_yield_complete",
                        crop_id=final.crop_id,
                        crops=1,
                    )
                    if next_staged is None:
                        break
                    current_staged = next_staged
                drained_normally = True
            finally:
                self._emit_scheduler_progress(
                    "prefill_h2d_executor_shutdown_begin",
                    drained_normally=drained_normally,
                    has_current_staged=current_staged is not None,
                )
                h2d_executor.shutdown(wait=True, cancel_futures=True)
                self._emit_scheduler_progress(
                    "prefill_h2d_executor_shutdown_end",
                    drained_normally=drained_normally,
                    has_current_staged=current_staged is not None,
                )
                if drained_normally and current_staged is not None:
                    raise RuntimeError(
                        "ready stream drained with an unused staged prefill"
                    )
                self._emit_scheduler_progress(
                    "ready_stream_end",
                    drained_normally=drained_normally,
                )

        return self._decode_ready_source(
            ready_stream(),
            schedule_id=schedule_id,
            emit_result=emit_result,
        )

    # Decode coordination and ready-state handoff.
    def _begin_decode_schedule(self) -> None:
        self._prefill_sequence = 0
        self._vision_prefill_stats = _VisionPrefillStats()
        self._text_prefill_stats = _TextPrefillStats()

    def _decode_ready_source(
        self,
        ready_source: Any,
        *,
        schedule_id: str,
        emit_result: Callable[[RecognitionResult], None],
        scheduling_metrics: RequestSchedulingMetrics | None = None,
    ) -> ContinuousDecodeResult:
        def handle_completion(completion: DecodeCompletion) -> None:
            result = self._result_from_completion(
                completion,
                schedule_id=schedule_id,
            )
            emit_result(result)

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
                "raw_decode_tok_per_s": per_second(
                    decoded.raw_decode_token_slots,
                    decode_wall_s,
                ),
                "effective_decode_tok_per_s": per_second(
                    decoded.effective_decode_tokens,
                    decode_wall_s,
                ),
                "effective_fraction": (
                    float(decoded.effective_decode_tokens)
                    / float(decoded.raw_decode_token_slots)
                    if decoded.raw_decode_token_slots > 0
                    else None
                ),
                "active_slot_fraction": (
                    float(decoded.active_decode_token_slots)
                    / float(decoded.raw_decode_token_slots)
                    if decoded.raw_decode_token_slots > 0
                    else None
                ),
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

    def _ready_from_prefilled(
        self,
        state: PrefilledRecognition,
    ) -> ReadyDecodeRequest:
        cache, rope_deltas, cache_position, first_token_tensor, cache_release = (
            state.take_device_state()
        )
        return ReadyDecodeRequest(
            request_id=state.request_id,
            payload=state,
            cache=cache,
            rope_deltas=rope_deltas,
            cache_position=cache_position,
            first_token_tensor=first_token_tensor,
            first_token=state.first_token,
            prompt_length=state.input_tokens,
            cache_release=cache_release,
        )

    # CPU lookahead and preparation; this remains asynchronous with NPU work.
    def _iter_cpu_prepared(
        self,
        requests: Iterable[RecognitionRequest],
    ) -> Iterable[tuple[CpuPreparedRecognition, float]]:
        """Prepare requests on one background CPU lane with bounded FIFO state."""

        source = iter(requests)
        pending: deque[Future[CpuPreparedRecognition]] = deque()
        source_exhausted = False
        executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="paddleocr-vl-cpu-prepare",
        )

        def fill_cpu_pipeline() -> None:
            nonlocal source_exhausted
            while (
                not source_exhausted and len(pending) < self.cpu_preprocess_max_pending
            ):
                try:
                    request = next(source)
                except StopIteration:
                    source_exhausted = True
                    break
                submitted_at = time.perf_counter()
                if self.timeline is not None:
                    self.timeline.instant(
                        "CPU / queue wait",
                        "Crop submitted to CPU worker",
                        flow_id=request.request_id,
                        track="queue",
                        lane="cpu-prep",
                        args={"pending_before_submit": len(pending)},
                    )
                pending.append(
                    executor.submit(
                        self._prepare_cpu,
                        request,
                        submitted_at,
                    )
                )

        try:
            fill_cpu_pipeline()
            while pending:
                future = pending.popleft()
                wait_started = time.perf_counter()
                prepared = future.result()
                wait_finished = time.perf_counter()
                consumer_wait_s = wait_finished - wait_started
                if self.timeline is not None:
                    self.timeline.record_span_seconds(
                        "CPU / queue wait",
                        "Consumer waiting for prepared crop",
                        wait_started,
                        wait_finished,
                        flow_id=prepared.request_id,
                        event_type="wait",
                        args={"pending_after_pop": len(pending)},
                    )
                # Refill before yielding so the worker remains productive while
                # the consumer performs H2D and NPU prefill for this request.
                fill_cpu_pipeline()
                yield prepared, consumer_wait_s
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    @torch.inference_mode()
    def _prepare_cpu(
        self,
        request: RecognitionRequest,
        submitted_at: float,
    ) -> CpuPreparedRecognition:
        preparation_started = time.perf_counter()
        if self.timeline is not None:
            self.timeline.record_span_seconds(
                "CPU / queue wait",
                "Queued for CPU preprocessing",
                submitted_at,
                preparation_started,
                flow_id=request.request_id,
                event_type="wait",
                track="queue",
                lane="cpu-prep",
            )
        timing: dict[str, float] = {}
        if isinstance(request.crop, bytes):
            image_decode_started = time.perf_counter()
            request = request.resolve_image()
            timing["cpu_image_decode"] = time.perf_counter() - image_decode_started
        crop_size = tuple(int(value) for value in request.crop.size)
        started = time.perf_counter()
        pixel_values, image_grid_thw = preprocess_pil_image(
            request.crop,
        )
        input_ids, attention_mask = prepare_prompt_tokens(
            self.preprocessing_tokenizer,
            image_grid_thw,
            request.prompt,
        )
        timing["cpu_image_and_prompt_preprocess"] = time.perf_counter() - started
        if self.timeline is not None:
            self.timeline.record_span_seconds(
                "CPU preprocessing",
                "Image resize, patchify, and prompt construction",
                started,
                started + timing["cpu_image_and_prompt_preprocess"],
                flow_id=request.request_id,
                args={"crop_width": crop_size[0], "crop_height": crop_size[1]},
            )

        prompt_length = int(input_ids.shape[1])
        if prompt_length > self.cache_length:
            raise ValueError(
                f"request {request.request_id} has prompt_length={prompt_length}, "
                f"configured cache_length={self.cache_length}"
            )
        image_token_count = int(
            (input_ids == IMAGE_TOKEN_ID).sum().item()
        )

        started = time.perf_counter()
        position_ids_cpu, rope_deltas_cpu = self.model.get_rope_index(
            input_ids,
            image_grid_thw,
            attention_mask,
        )
        timing["cpu_mrope_index"] = time.perf_counter() - started
        if self.timeline is not None:
            self.timeline.record_span_seconds(
                "CPU MRoPE",
                "Build multimodal rotary positions",
                started,
                started + timing["cpu_mrope_index"],
                flow_id=request.request_id,
                args={"input_tokens": int(input_ids.shape[1])},
            )

        input_fingerprints: dict[str, Any] = {}
        if self.recognition_input_fingerprints:
            started = time.perf_counter()
            input_fingerprints = fingerprint_recognition_inputs(
                crop=request.crop,
                tensors={
                    "attention_mask": attention_mask,
                    "image_grid_thw": image_grid_thw,
                    "input_ids": input_ids,
                    "pixel_values": pixel_values,
                    "position_ids": position_ids_cpu,
                    "rope_deltas": rope_deltas_cpu,
                },
            )
            timing["cpu_input_fingerprints"] = time.perf_counter() - started

        started = time.perf_counter()
        input_ids = _pin_memory_or_keep(input_ids)
        attention_mask = _pin_memory_or_keep(attention_mask)
        pixel_values = _pin_memory_or_keep(pixel_values)
        position_ids_cpu = _pin_memory_or_keep(position_ids_cpu)
        rope_deltas_cpu = _pin_memory_or_keep(rope_deltas_cpu)
        timing["cpu_pin_memory"] = time.perf_counter() - started
        if self.timeline is not None:
            self.timeline.record_span_seconds(
                "CPU preprocessing",
                "Pin recognition input staging",
                started,
                started + timing["cpu_pin_memory"],
                flow_id=request.request_id,
                args={
                    "pinned_tensors": sum(
                        int(tensor.is_pinned())
                        for tensor in (
                            input_ids,
                            attention_mask,
                            pixel_values,
                            position_ids_cpu,
                            rope_deltas_cpu,
                        )
                    ),
                    "requested_tensors": 5,
                },
            )

        preparation_finished = time.perf_counter()
        timing["cpu_preprocess_background_queue_wait"] = max(
            0.0,
            preparation_started - submitted_at,
        )
        timing["cpu_preprocess_background_service"] = (
            preparation_finished - preparation_started
        )
        return CpuPreparedRecognition(
            request_id=request.request_id,
            prompt=request.prompt,
            crop_size=crop_size,
            skip_special_tokens=bool(request.skip_special_tokens),
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids_cpu,
            rope_deltas=rope_deltas_cpu,
            image_token_count=image_token_count,
            timing_s=timing,
            request_started=submitted_at,
            preparation_finished=preparation_finished,
            input_fingerprints=input_fingerprints,
        )

    def _prepared_crop(
        self, prepared: CpuPreparedRecognition, consumer_wait_s: float,
    ) -> _PreparedCrop:
        self._prefill_sequence += 1
        return _PreparedCrop(
            self._prefill_sequence, prepared, consumer_wait_s,
            int(prepared.pixel_values.shape[0]),
        )

    def _iter_prepared_crops(
        self, requests: Iterable[RecognitionRequest],
    ) -> Iterable[_PreparedCrop]:
        for prepared, wait_s in self._iter_cpu_prepared(requests):
            yield self._prepared_crop(prepared, wait_s)

    # Single-crop prefill: stage inputs, enqueue device work, then finalize.
    @torch.inference_mode()
    def prefill_one(self, request: RecognitionRequest) -> PrefilledRecognition:
        """Run the faithful crop frontend and prefill without entering decode.

        Specialized B1 target runtimes use this seam to consume the same
        prepared image, prompt, vision, projector, text-prefill, and private-KV
        result as the normal scheduler. The returned state owns one cache lease;
        its consumer must eventually call ``take_device_state`` and release it.
        """

        submitted_at = time.perf_counter()
        prepared = self._prepare_cpu(request, submitted_at)
        return self.prefill_prepared_one(prepared)

    @torch.inference_mode()
    def prefill_prepared_one(
        self,
        prepared: CpuPreparedRecognition,
    ) -> PrefilledRecognition:
        """Run one already-prepared crop through the normal NPU prefill path.

        This is the ownership-safe handoff for callers that prepare a request
        on a CPU worker while unrelated NPU work is running. The returned state
        owns one cache lease; its consumer must eventually call
        ``take_device_state`` and release it.
        """

        crop = self._prepared_crop(prepared, 0.0)
        staged = self._stage_crop(crop)
        inflight = self._enqueue_crop(staged)
        finalized = self._finalize_crop(inflight)
        return finalized

    @torch.inference_mode()
    def _stage_crop(self, crop: _PreparedCrop) -> _StagedCrop:
        import torch_npu

        prepared = crop.prepared
        device_timeline = DeviceTimeline(self.device)
        submit_started = time.perf_counter()
        with torch_npu.npu.stream(self.prefill_transfer_stream):
            timing = dict(prepared.timing_s)
            timing["cpu_preprocess_background_consumer_wait"] = float(crop.consumer_wait_s)
            ready_consumed_at = time.perf_counter()
            timing["cpu_preprocess_background_ready_wait"] = max(
                0.0, ready_consumed_at - prepared.preparation_finished,
            )
            if self.timeline is not None:
                self.timeline.record_span_seconds(
                    "CPU / queue wait", "Prepared crop waiting for NPU prefill",
                    prepared.preparation_finished, ready_consumed_at,
                    flow_id=prepared.request_id, event_type="wait", track="queue",
                    lane="prefill-ready",
                )

            def move_inputs() -> tuple[torch.Tensor, ...]:
                pixels = prepared.pixel_values.to(device=self.device, non_blocking=True)
                return (
                    prepared.input_ids.to(self.device, non_blocking=True),
                    prepared.attention_mask.to(self.device, non_blocking=True),
                    pixels,
                    prepared.position_ids.to(self.device, non_blocking=True),
                    prepared.rope_deltas.to(self.device, non_blocking=True),
                )

            moved = device_timeline.measure("recognition_inputs_h2d", move_inputs)
            h2d_ready_event = self.prefill_transfer_stream.record_event()
        submit_finished = time.perf_counter()
        timing["prefill_h2d_submit_host"] = submit_finished - submit_started
        if self.timeline is not None:
            self.timeline.record_span_seconds(
                "H2D / D2H transfer", "Submit crop H2D", submit_started, submit_finished,
                flow_id=prepared.request_id, event_type="io", track="host",
                lane="prefill-submit", args={"crop_id": crop.crop_id},
            )
        return _StagedCrop(crop, device_timeline, h2d_ready_event, moved, timing)

    @torch.inference_mode()
    def _enqueue_crop(self, staged: _StagedCrop) -> _InFlightCrop:
        import torch_npu

        crop = staged.crop
        prepared = crop.prepared
        device_timeline = staged.device_timeline
        moved = staged.moved
        timing = staged.timing_s
        enqueue_started = time.perf_counter()
        compute_stream = torch_npu.npu.current_stream()
        compute_stream.wait_event(staged.h2d_ready_event)
        prefill_started = time.perf_counter()
        vision_model = self.model.visual.vision_model
        pixels = moved[2]
        def normalize_uint8():
            output = pixels.to(torch.float32)
            output.mul_(1.0 / 255.0)
            output.sub_(0.5)
            output.div_(0.5)
            return output.to(self.model.visual.dtype).contiguous()
        pixels = device_timeline.measure("vision_input_normalize", normalize_uint8)
        hidden = device_timeline.measure("vision_embeddings", lambda: vision_model.embeddings(
            pixels.unsqueeze(0), image_grid_thw=prepared.image_grid_thw,
        ))
        real_length = int(hidden.shape[0])
        vision_route = self.vision_prefill.route(real_length)
        prepared_vision = device_timeline.measure("vision_prefill_input_prep", lambda: self.vision_prefill.prepare(
            hidden, prepared.image_grid_thw, route=vision_route,
        ))
        features = device_timeline.measure("vision_prefill", lambda: self.vision_prefill.run_prepared(prepared_vision))
        self._vision_prefill_stats.record(vision_route)
        input_ids, attention_mask, _pixels, position_ids, rope_deltas = moved
        next_position = torch.full((1,), int(input_ids.shape[1]), device=self.device, dtype=torch.int64)
        image_embeds = device_timeline.measure("adaptive_mlp_projector", lambda: self.model.mlp_AR(features, prepared.image_grid_thw))
        inputs_embeds = device_timeline.measure("text_token_embedding", lambda: self.model.model.embed_tokens(input_ids))

        def scatter_image_embeds():
            projected = image_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            image_mask = (input_ids == IMAGE_TOKEN_ID).unsqueeze(-1).expand_as(inputs_embeds)
            return inputs_embeds.masked_scatter(image_mask, projected)

        inputs_embeds = device_timeline.measure("image_embed_scatter", scatter_image_embeds)
        lease = device_timeline.measure("static_cache_alloc", self.prefill_cache_pool.acquire)
        self._text_prefill_stats.record()
        text_route = self.text_prefill.route(int(inputs_embeds.shape[1]))
        prepared_text = device_timeline.measure("text_prefill_input_prep", lambda: self.text_prefill.prepare(
            inputs_embeds, attention_mask, position_ids, route=text_route,
        ))
        last_hidden = device_timeline.measure("text_prefill", lambda: self.text_prefill.run_prepared(prepared_text, lease.cache))
        logits = device_timeline.measure("prefill_lm_head", lambda: self.model.lm_head(last_hidden))
        next_token = device_timeline.measure("prefill_argmax", lambda: torch.argmax(logits[:, -1, :].float(), dim=-1, keepdim=True))
        text_route = {
            **text_route,
            "private_cache_slot_index": int(lease.slot_index),
            "private_cache_generation": int(lease.generation),
        }
        # Retain the original one-element concat/copy operation in this structural
        # pass; eliminating it is a separate execution change.
        first_token_device = torch.cat([next_token.detach().reshape(-1)], dim=0).contiguous()
        prefill_ready_event = torch_npu.npu.current_stream().record_event()
        enqueue_finished = time.perf_counter()
        timing["prefill_enqueue_host"] = enqueue_finished - enqueue_started
        if self.timeline is not None:
            self.timeline.record_span_seconds(
                "Vision prefill", "Enqueue prefill chain", enqueue_started, enqueue_finished,
                flow_id=prepared.request_id, args={"crop_id": crop.crop_id,
                    "real_tokens": real_length, "physical_tokens": vision_route["physical_vision_tokens"]},
            )
        return _InFlightCrop(
            crop_id=crop.crop_id, prepared=prepared, cache=lease.cache, cache_lease=lease,
            rope_deltas=rope_deltas, next_cache_position=next_position, next_token=next_token,
            device_inputs=moved, vision=vision_route, text_prefill=text_route, timing_s=timing,
            input_tokens=int(prepared.input_ids.shape[1]), projected_image_tokens=int(image_embeds.shape[0]),
            device_timeline=device_timeline, h2d_ready_event=staged.h2d_ready_event,
            prefill_ready_event=prefill_ready_event, first_token_device=first_token_device,
            prefill_started=prefill_started,
        )

    @torch.inference_mode()
    def _finalize_crop(self, inflight: _InFlightCrop) -> PrefilledRecognition:
        import torch_npu

        resolve_started = time.perf_counter()
        spans = inflight.device_timeline.resolve_spans()
        started = time.perf_counter()
        with torch_npu.npu.stream(self.prefill_transfer_stream):
            self.prefill_transfer_stream.wait_event(inflight.prefill_ready_event)
            self.prefill_host_tokens[:1].copy_(inflight.first_token_device, non_blocking=True)
            first_token_ready = self.prefill_transfer_stream.record_event()
        first_token_ready.synchronize()
        first_token = int(self.prefill_host_tokens[:1].tolist()[0])
        self._diagnose_prefill_kv_finiteness(inflight)
        first_token_d2h_s = time.perf_counter() - started
        resolve_finished = time.perf_counter()
        prepared = inflight.prepared
        stages = {
            "recognition_inputs_h2d": ("H2D / D2H transfer", "Recognition inputs H2D"),
            "vision_embeddings": ("Vision prefill", "Patch and position embeddings"),
            "vision_prefill_input_prep": ("Vision prefill", "Vision bucket preparation"),
            "vision_prefill": ("Vision prefill", "Vision transformer"),
            "adaptive_mlp_projector": ("Vision prefill", "Adaptive MLP projector"),
            "text_token_embedding": ("Text prefill", "Text token embeddings"),
            "image_embed_scatter": ("Text prefill", "Scatter projected image embeddings"),
            "static_cache_alloc": ("Text prefill", "Acquire private KV cache slot"),
            "text_prefill_input_prep": ("Text prefill", "Text bucket preparation"),
            "text_prefill": ("Text prefill", "Text transformer prefill"),
            "prefill_lm_head": ("Text prefill", "Prefill LM head"),
            "prefill_argmax": ("Text prefill", "First-token argmax"),
        }
        device_stage_s = {stage: float(spans[stage]["seconds"]) for stage in stages}
        # Existing summary schema; no redistribution happens in this pipeline.
        device_stage_s["text_kv_redistribute"] = 0.0
        timing = inflight.timing_s
        timing["recognizer_h2d"] = device_stage_s["recognition_inputs_h2d"]
        timing["first_token_d2h"] = first_token_d2h_s
        timing["prefill_resolve_wait"] = resolve_finished - resolve_started
        timing["vision_and_text_prefill_wall"] = resolve_finished - inflight.prefill_started
        timing["time_to_first_token"] = resolve_finished - prepared.request_started
        timing["prefill_request_total"] = sum(timing[name] for name in (
            "cpu_image_and_prompt_preprocess", "cpu_mrope_index", "cpu_pin_memory",
            "recognizer_h2d", "vision_and_text_prefill_wall", "first_token_d2h",
        ))
        if self.timeline is not None:
            for stage, (category, label) in stages.items():
                span = spans[stage]
                self.timeline.record_span(
                    category, label, int(span["start_ns"]), int(span["end_ns"]),
                    flow_id=prepared.request_id, clock=str(span["clock"]), track="device",
                    lane="prefill", args={"crop_id": inflight.crop_id, "stage": stage},
                )
            self.timeline.record_span_seconds(
                "H2D / D2H transfer", "First token D2H", started, started + first_token_d2h_s,
                flow_id=prepared.request_id, event_type="io",
            )
            self.timeline.record_span_seconds(
                "Vision prefill", "Resolve prefill completion", resolve_started, resolve_finished,
                flow_id=prepared.request_id, event_type="wait",
            )
        return PrefilledRecognition(
            request_id=prepared.request_id, prompt=prepared.prompt, crop_size=prepared.crop_size,
            skip_special_tokens=prepared.skip_special_tokens, cache=inflight.cache,
            cache_release=inflight.cache_lease.release, rope_deltas=inflight.rope_deltas,
            next_cache_position=inflight.next_cache_position, next_token=inflight.next_token,
            first_token=first_token, input_tokens=inflight.input_tokens,
            projected_image_tokens=inflight.projected_image_tokens, vision=inflight.vision,
            text_prefill=inflight.text_prefill, timing_s=timing, device_stage_s=device_stage_s,
            request_started=prepared.request_started, prefill_finished=resolve_finished,
            input_fingerprints=dict(prepared.input_fingerprints),
        )

    # Materialize each completed request.
    def _result_from_completion(
        self,
        completion: DecodeCompletion,
        *,
        schedule_id: str,
    ) -> RecognitionResult:
        state: PrefilledRecognition = completion.ready.payload
        token_ids = completion.token_ids
        started = time.perf_counter()
        text = self.tokenizer.decode(
            token_ids,
            skip_special_tokens=state.skip_special_tokens,
        )
        detokenize_s = time.perf_counter() - started
        if self.timeline is not None:
            self.timeline.record_span_seconds(
                "Result assembly",
                "Detokenize crop result",
                started,
                started + detokenize_s,
                flow_id=state.request_id,
                args={"generated_tokens": len(token_ids)},
            )
            self.timeline.instant(
                "Result assembly",
                "Crop recognition completed",
                flow_id=state.request_id,
                args={
                    "stop_reason": completion.stop_reason,
                    "decode_slot": completion.slot_index,
                    "generated_tokens": len(token_ids),
                },
            )
        generated_tokens = len(token_ids)
        effective_decode_tokens = max(0, generated_tokens - 1)
        timing = dict(state.timing_s)
        timing.update(
            {
                "decode_ready_queue_wait": (
                    max(0.0, completion.admitted_at - state.prefill_finished)
                    if completion.admitted_at is not None
                    else 0.0
                ),
                "decode_slot_residency": (
                    max(0.0, completion.completed_at - completion.admitted_at)
                    if completion.admitted_at is not None
                    else 0.0
                ),
                "detokenize": float(detokenize_s),
                "request_total": float(
                    completion.completed_at - state.request_started + detokenize_s
                ),
            }
        )
        return RecognitionResult(
            request_id=state.request_id,
            decode_schedule_id=schedule_id,
            decode_slot_index=completion.slot_index,
            decode_slot_epoch=completion.slot_epoch,
            prompt=state.prompt,
            crop_size=state.crop_size,
            text=text,
            token_ids=token_ids,
            stop_reason=completion.stop_reason,
            input_tokens=state.input_tokens,
            projected_image_tokens=state.projected_image_tokens,
            generated_tokens_including_eos=generated_tokens,
            decode_tokens_after_prefill_including_eos=effective_decode_tokens,
            decode_calls_executed=completion.iterations_launched,
            timing_s=timing,
            device_stage_s=dict(state.device_stage_s),
            rates={
                "request_output_tok_per_s": per_second(
                    generated_tokens,
                    timing["request_total"],
                ),
            },
            vision=dict(state.vision),
            text_prefill=dict(state.text_prefill),
            input_fingerprints=dict(state.input_fingerprints),
            scheduling_metrics=dict(completion.scheduling_metrics),
            repetition=(
                dict(completion.repetition_evidence)
                if completion.repetition_evidence is not None
                else {}
            ),
        )

    # Persistent setup and configuration.
    @torch.inference_mode()
    def __init__(
        self,
        *,
        model: str,
        device: str = "npu:0",
        batch_size: int,
        torchair_cache_dir: Path,
        full_decode_lm_head: bool = False,
        decode_device_timing: bool = True,
        vision_torchair_cache_dir: Path | None = None,
        eager: bool = False,
        text_torchair_cache_dir: Path | None = None,
        timeline: TimelineRecorder | None = None,
        scheduler_progress: bool = False,
        scheduler_progress_events: Iterable[str] | None = None,
        diagnostic_decode_effective_length: int | None = None,
        diagnostic_decode_request_id: str | None = None,
        diagnostic_prefill_kv_request_ids: Iterable[str] | None = None,
        recognition_input_fingerprints: bool = False,
    ):
        runtime_started = time.perf_counter()
        _emit_setup_progress("frontend", "start")
        import torch_npu

        self.eager = eager
        self.full_decode_lm_head = bool(full_decode_lm_head)
        torch.npu.config.allow_internal_format = True
        self.model_dir = Path(model).expanduser()
        self.device = torch.device(device)
        if self.device.type != "npu":
            raise ValueError("table serving requires an NPU device")
        if not torch.npu.is_available():
            raise RuntimeError("Table serving requires an available NPU")
        self.dtype = torch.float16
        torch.npu.set_compile_mode(jit_compile=False)
        self.decode_backend = "raw_eager" if eager else "torchair"
        self.decode_vocab_token_ids_path = (
            Path(__file__).resolve().parent
            / "presets/table_compact_vocab"
            / "native_han_core_60416.json"
        )
        self.timeline = timeline
        self.scheduler_progress = bool(scheduler_progress)
        self.scheduler_progress_events = (
            None
            if scheduler_progress_events is None
            else frozenset((str(event) for event in scheduler_progress_events))
        )
        self.diagnostic_decode_effective_length = (
            None
            if diagnostic_decode_effective_length is None
            else int(diagnostic_decode_effective_length)
        )
        self.diagnostic_decode_request_id = (
            None
            if diagnostic_decode_request_id is None
            else str(diagnostic_decode_request_id)
        )
        self.diagnostic_prefill_kv_request_ids = frozenset(
            (str(request_id) for request_id in diagnostic_prefill_kv_request_ids or ())
        )
        self.batch_size = int(batch_size)
        self.cache_length = 4096
        self.max_new_tokens = 4096
        self.recognition_input_fingerprints = bool(recognition_input_fingerprints)
        self.vision_backend = self.decode_backend
        self.vision_attention = "prompt_flash_attention"
        self.decode_device_timing = bool(decode_device_timing)
        self.compact_decode_control = False
        self.vision_seq_alignment = VISION_SEQUENCE_ALIGNMENT
        self.vision_buckets = VISION_BUCKETS
        self.text_backend = self.decode_backend
        self.text_buckets = TEXT_PREFILL_BUCKETS
        self.private_cache_staging_headroom = 32
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        self.preprocessing_tokenizer = Tokenizer.from_file(
            str(self.model_dir / "tokenizer.json")
        )
        self.tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        frontend_setup_s = time.perf_counter() - runtime_started
        _emit_setup_progress("frontend", "done", frontend_setup_s)
        synchronize(self.device)
        started = time.perf_counter()
        _emit_setup_progress("model_load", "start")
        self.model = LocalPaddleOCRVLForConditionalGeneration.from_pretrained(
            self.model_dir, dtype=self.dtype, device=self.device
        )
        synchronize(self.device)
        model_load_s = time.perf_counter() - started
        _emit_setup_progress("model_load", "done", model_load_s)
        started = time.perf_counter()
        _emit_setup_progress("decode_lm_head", "start")
        full_vocab_size = int(self.model.lm_head.weight.shape[0])
        if self.full_decode_lm_head:
            # No selected-row head or ID map: the checkpoint head emits native IDs.
            self.decode_vocab = {
                "enabled": False,
                "path": None,
                "full_vocab_size": full_vocab_size,
                "selected_vocab_size": full_vocab_size,
                "token_ids_sha256": None,
            }
            decode_head_cache_key = f"full_vocab_{full_vocab_size}"
        else:
            (token_ids, self.decode_vocab) = load_decode_vocab_token_ids(
                self.decode_vocab_token_ids_path,
                full_vocab_size=full_vocab_size,
            )
            prepare_decode_compact_lm_head(self.model, token_ids)
            decode_head_cache_key = (
                f"selected_vocab_{self.decode_vocab['selected_vocab_size']}_"
                f"{self.decode_vocab['token_ids_sha256'][:12]}"
            )
        synchronize(self.device)
        _emit_setup_progress(
            "decode_lm_head", "done", time.perf_counter() - started
        )
        synchronize(self.device)
        started = time.perf_counter()
        _emit_setup_progress("vision_mlp_padding", "start")
        self.vision_mlp = prepare_vision_mlp_intermediate(
            self.model,
        )
        synchronize(self.device)
        vision_mlp_setup_s = time.perf_counter() - started
        _emit_setup_progress("vision_mlp_padding", "done", vision_mlp_setup_s)
        started = time.perf_counter()
        _emit_setup_progress("vision_attention_weight_padding", "start")
        prepare_vision_attention_weight_padding(self.model)
        synchronize(self.device)
        _emit_setup_progress(
            "vision_attention_weight_padding", "done", time.perf_counter() - started
        )
        synchronize(self.device)
        started = time.perf_counter()
        _emit_setup_progress("vision_weight_format", "start")
        self.vision_weight_format = prepare_vision_linear_weight_format(
            self.model
        )
        synchronize(self.device)
        vision_weight_format_s = time.perf_counter() - started
        _emit_setup_progress("vision_weight_format", "done", vision_weight_format_s)
        synchronize(self.device)
        started = time.perf_counter()
        _emit_setup_progress("decode_optimization_setup", "start")
        prepare_decode_projections(self.model)
        synchronize(self.device)
        decode_optimization_setup_s = time.perf_counter() - started
        _emit_setup_progress(
            "decode_optimization_setup", "done", decode_optimization_setup_s
        )
        synchronize(self.device)
        started = time.perf_counter()
        _emit_setup_progress("decode_weight_format", "start")
        self.weight_format = cast_decode_linear_weights_to_nz(self.model)
        synchronize(self.device)
        weight_format_s = time.perf_counter() - started
        _emit_setup_progress("decode_weight_format", "done", weight_format_s)
        self.stages = self.model.make_inference_stages(
            vision_cache_root=vision_torchair_cache_dir
            if vision_torchair_cache_dir is not None
            else torchair_cache_dir.parent / f"{torchair_cache_dir.name}_vision",
            text_cache_root=text_torchair_cache_dir
            if text_torchair_cache_dir is not None
            else torchair_cache_dir.parent / f"{torchair_cache_dir.name}_text",
            decode_cache_root=torchair_cache_dir / decode_head_cache_key,
            batch_size=self.batch_size,
            cache_length=self.cache_length,
            device=self.device,
            eager=self.eager,
            setup_progress=_emit_setup_progress,
        )
        self.vision_prefill = self.stages.vision_prefill
        self.text_prefill = self.stages.text_prefill
        self.text_decode = self.stages.text_decode
        self.decode_fn = self.text_decode.fn
        stage_setup_sync_started = time.perf_counter()
        synchronize(self.device)
        stage_setup_sync_s = time.perf_counter() - stage_setup_sync_started
        self.prefill_transfer_stream = torch_npu.npu.Stream(device=self.device)
        self.cpu_preprocess_max_pending = max(2, self.batch_size)
        self.ready_buffer_capacity = self.batch_size
        self.ready_buffer_low_watermark = max(
            1, self.ready_buffer_capacity // 2
        )
        self.prefill_host_tokens = torch.empty(
            (max(self.cpu_preprocess_max_pending + 1, self.private_cache_staging_headroom),),
            dtype=torch.int64,
            pin_memory=True,
        )
        self._prefill_sequence = 0
        self._vision_prefill_stats = _VisionPrefillStats()
        self._text_prefill_stats = _TextPrefillStats()
        private_cache_pool_started = time.perf_counter()
        _emit_setup_progress("private_cache_pool", "start")
        private_cache_capacity = (
            self.ready_buffer_capacity + self.private_cache_staging_headroom
        )
        private_cache_storage = self.model.allocate_static_cache(
            batch_size=private_cache_capacity,
            cache_length=self.cache_length,
            device=self.device,
            dtype=self.dtype,
        )
        synchronize(self.device)
        self.prefill_cache_pool = PrefillKVCachePool(
            private_cache_storage, device=self.device
        )
        private_cache_pool_setup_s = time.perf_counter() - private_cache_pool_started
        _emit_setup_progress("private_cache_pool", "done", private_cache_pool_setup_s)
        started = time.perf_counter()
        _emit_setup_progress("decode_control", "start")
        self.decode_arena = DecodeArena(
            cache=self.text_decode.warm_cache,
            device=self.device,
            batch_size=self.batch_size,
            eos_token_id=int(TEXT_EOS_TOKEN_ID),
            timeline=self.timeline,
            decode_device_timing=self.decode_device_timing,
        )
        self.decode_scheduler = ContinuousDecodeScheduler(
            arena=self.decode_arena,
            decode_fn=self.decode_fn,
            max_new_tokens=self.max_new_tokens,
            timeline=self.timeline,
            stop_repetitions=True,
            progress=self._emit_scheduler_progress,
            diagnostic_effective_length=self.diagnostic_decode_effective_length,
            diagnostic_request_id=self.diagnostic_decode_request_id,
        )
        decode_control_setup_s = time.perf_counter() - started
        _emit_setup_progress("decode_control", "done", decode_control_setup_s)
        self.setup_timing_s = {
            "recognizer_frontend_setup": float(frontend_setup_s),
            "recognizer_model_load": float(model_load_s),
            "vision_mlp_padding": float(vision_mlp_setup_s),
            "vision_weight_format": float(vision_weight_format_s),
            "decode_optimization_setup": float(decode_optimization_setup_s),
            "decode_weight_format": float(weight_format_s),
            **self.stages.setup_timing_s,
            "vision_router_setup": 0.0,
            "packed_text_runtime_setup": float(stage_setup_sync_s),
            "private_cache_pool_setup": float(private_cache_pool_setup_s),
            "decode_control_setup": float(decode_control_setup_s),
            "recognizer_runtime_total": float(time.perf_counter() - runtime_started),
        }
        _emit_setup_progress(
            "recognizer_runtime",
            "done",
            self.setup_timing_s["recognizer_runtime_total"],
        )

    def configuration(self) -> dict[str, Any]:
        decode_label = f"{'eager' if self.eager else 'compiled'}_static_b{self.batch_size}"
        patch_size = PATCH_SIZE
        merge_size = MERGE_SIZE
        min_pixels = MIN_PIXELS
        vision_attention = self.vision_attention
        return {
            "recognizer_model": str(self.model_dir),
            "device": str(self.device),
            "dtype": str(self.dtype),
            "decode_backend": self.decode_backend,
            "decode_vocab": dict(self.decode_vocab),
            "full_decode_lm_head": self.full_decode_lm_head,
            "token_selection": "greedy",
            "cache_length": self.cache_length,
            "max_new_tokens": self.max_new_tokens,
            "recognition_input_fingerprints": (self.recognition_input_fingerprints),
            "batch_size": self.batch_size,
            "diagnostic_decode_effective_length": (
                self.diagnostic_decode_effective_length
            ),
            "scheduler_progress_events": (
                None
                if self.scheduler_progress_events is None
                else sorted(self.scheduler_progress_events)
            ),
            "diagnostic_decode_request_id": self.diagnostic_decode_request_id,
            "diagnostic_prefill_kv_request_ids": sorted(
                self.diagnostic_prefill_kv_request_ids
            ),
            "vision_prefill": self.vision_prefill.metadata,
            "vision_mlp": dict(self.vision_mlp),
            "vision_linear_weight_format": dict(self.vision_weight_format),
            "vision_backend": self.vision_backend,
            "vision_attention": vision_attention,
            "decode_device_timing": self.decode_device_timing,
            "compact_decode_control": self.compact_decode_control,
            "vision_sequence_alignment": self.vision_seq_alignment,
            "vision_packing": {
                "mode": "off",
                "target": 1920,
                "lookahead": 32,
                "grouping": "independent_crops",
                "oversized": "faithful_eager_single_crop_route",
                "batched_runtime": (None),
            },
            "vision_prompt_fa_layout": "bnsd",
            "text_prefill": self.text_prefill.metadata,
            "text_backend": self.text_backend,
            "text_packing": {
                "mode": "off",
                "buckets": [128, 256, 512, 1024],
                "max_members": self.private_cache_staging_headroom,
                "grouping": "independent_crops",
                "runtime": (None),
            },
            "preprocessor": {
                "effective_min_pixels": min_pixels,
                "effective_max_pixels": MAX_PIXELS,
                "patch_size": patch_size,
                "merge_size": merge_size,
                "resize_factor": patch_size * merge_size,
                "nominal_minimum_projected_image_tokens": (
                    min_pixels // ((patch_size * merge_size) ** 2)
                ),
            },
            "cpu_preprocessing": {
                "execution": "background_thread",
                "workers": 1,
                "max_pending": self.cpu_preprocess_max_pending,
                "ordering": "fifo",
                "pin_recognition_inputs": "best_effort",
            },
            "prefill_production": ("next_crop_h2d_staged_before_ready_yield"),
            "prefill_transfer": "dedicated_stream_event_dependencies",
            "decode": decode_label,
            "decode_schedule": "run_scoped_persistent_slots_iteration_hot_swap",
            "ready_buffer_capacity": self.ready_buffer_capacity,
            "ready_buffer_low_watermark": self.ready_buffer_low_watermark,
            "private_cache_staging_headroom": (self.private_cache_staging_headroom),
            "decode_completion_detection": "queue_depth_one_async_token_copy",
            "private_prefill_cache": self.prefill_cache_pool.stats(),
            "kv_admission": "full_prefill_cache_foreach_copy_into_fixed_slot",
            "text_decode": self.text_decode.metadata,
            "linear_weight_format": self.weight_format,
        }

    # Progress reporting and retained diagnostics.
    def _emit_scheduler_progress(self, event: str, **fields: Any) -> None:
        if not self.scheduler_progress:
            return
        if (
            self.scheduler_progress_events is not None
            and event not in self.scheduler_progress_events
        ):
            return
        record = {
            "event": str(event),
            "monotonic_s": round(time.perf_counter(), 6),
            "thread": threading.current_thread().name,
            **fields,
        }
        print(
            "EXP09_SCHEDULER "
            + json.dumps(record, ensure_ascii=False, separators=(",", ":")),
            file=sys.stderr,
            flush=True,
        )

    @torch.inference_mode()
    def _diagnose_prefill_kv_finiteness(
        self,
        member: _InFlightCrop,
    ) -> None:
        """Synchronously inspect one explicitly targeted private KV row."""

        request_id = member.prepared.request_id
        if request_id not in self.diagnostic_prefill_kv_request_ids:
            return
        prefix_length = int(member.input_tokens)
        cache_length = int(member.cache.cache_length)
        pending: list[tuple[str, int, str, str, torch.Tensor]] = []
        for kind, tensors in (
            ("key", member.cache.key_caches),
            ("value", member.cache.value_caches),
        ):
            for layer, tensor in enumerate(tensors):
                for scope, view in (
                    ("prefix", tensor[..., :prefix_length, :]),
                    ("tail", tensor[..., prefix_length:, :]),
                ):
                    pending.append(
                        (
                            kind,
                            layer,
                            scope,
                            "nan",
                            torch.count_nonzero(torch.isnan(view)),
                        )
                    )
                    pending.append(
                        (
                            kind,
                            layer,
                            scope,
                            "inf",
                            torch.count_nonzero(torch.isinf(view)),
                        )
                    )
        counts = torch.stack([item[-1] for item in pending]).cpu().tolist()
        totals = {
            "prefix": {"nan": 0, "inf": 0},
            "tail": {"nan": 0, "inf": 0},
        }
        nonfinite_layers: list[dict[str, int | str]] = []
        for (kind, layer, scope, value_kind, _tensor), count in zip(pending, counts):
            count = int(count)
            totals[scope][value_kind] += count
            if count:
                nonfinite_layers.append(
                    {
                        "kind": kind,
                        "layer": layer,
                        "scope": scope,
                        "value_kind": value_kind,
                        "count": count,
                    }
                )
        digest_limit = min(cache_length, 1152)
        digest_chunk_size = 128
        chunk_hashes = [
            hashlib.sha256() for _start in range(0, digest_limit, digest_chunk_size)
        ]
        for kind, tensors in (
            ("key", member.cache.key_caches),
            ("value", member.cache.value_caches),
        ):
            for layer, tensor in enumerate(tensors):
                cpu = tensor[..., :digest_limit, :].detach().contiguous().cpu()
                for chunk_index, start in enumerate(
                    range(0, digest_limit, digest_chunk_size)
                ):
                    end = min(start + digest_chunk_size, digest_limit)
                    digest = chunk_hashes[chunk_index]
                    digest.update(f"{kind}:{layer}:{start}:{end}|".encode())
                    digest.update(cpu[..., start:end, :].contiguous().numpy().tobytes())
        print(
            "EXP09_PREFILL_KV_DIAGNOSTIC "
            + json.dumps(
                {
                    "request_id": request_id,
                    "input_tokens": prefix_length,
                    "cache_length": cache_length,
                    "private_cache_slot": member.cache_lease.slot_index,
                    "private_cache_generation": member.cache_lease.generation,
                    "totals": totals,
                    "nonfinite_layers": nonfinite_layers,
                    "sha256_by_absolute_token_chunk": [
                        {
                            "start": start,
                            "end": min(
                                start + digest_chunk_size,
                                digest_limit,
                            ),
                            "sha256": chunk_hashes[index].hexdigest(),
                        }
                        for index, start in enumerate(
                            range(0, digest_limit, digest_chunk_size)
                        )
                    ],
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            file=sys.stderr,
            flush=True,
        )


# Open request stream and free-slot prefill admission


class _OpenPrefillSource:
    """Prepare CPU inputs ahead, but prefill NPU KV only for free decode slots."""

    def pull_for_decode_slots(
        self,
        *,
        block: bool,
        available_slots: int,
    ) -> ReadyDecodeRequest | None:
        """CPU may run ahead; NPU prefill needs an unreserved decode slot."""
        return self.pull(block=block, allow_prefill=available_slots > 0)

    def pull(
        self, *, block: bool, allow_prefill: bool = True
    ) -> ReadyDecodeRequest | None:
        while True:
            pull_started = (
                time.perf_counter() if self.scheduling_metrics is not None else 0.0
            )
            self._submit_available(
                block_for_first=allow_prefill and block and not self.pending,
            )
            if not allow_prefill:
                # Do not consume/wait for a future or allocate/prefill NPU KV
                # while every decode slot is active or already reserved by a
                # ready request. CPU preparation above remains bounded/ahead.
                return None
            if not self.pending:
                return None
            if self.scheduling_metrics is not None:
                self.scheduling_metrics.cpu_prefill_eligible(self.pending[0][0], block=block)
            if not block and not self.pending[0][1].done():
                # CPU preparation is background work, not a reason to stall
                # live decoding. Keep ownership in pending until a later poll;
                # only an idle scheduler may wait for the first ready request.
                return None
            request_id, future = self.pending.popleft()
            wait_started = time.perf_counter()
            try:
                prepared = future.result()
            except BaseException as exc:
                if self.scheduling_metrics is not None:
                    self.scheduling_metrics.record_prefill(
                        request_id,
                        pull_started,
                        time.perf_counter(),
                        status="error",
                    )
                self.on_request_error(request_id, exc)
                block = False
                continue
            consumer_wait_s = time.perf_counter() - wait_started
            if self.scheduling_metrics is not None:
                self.scheduling_metrics.cpu_prepared(
                    request_id, submitted_at=prepared.request_started,
                    queue_wait_s=prepared.timing_s["cpu_preprocess_background_queue_wait"],
                    finished_at=prepared.preparation_finished, consumed_at=wait_started,
                )
            # Refill the CPU lane before NPU prefill so host preparation for
            # later HTTP requests overlaps the current crop's device work.
            self._submit_available(block_for_first=False)
            crop = self.recognizer._prepared_crop(prepared, consumer_wait_s)
            staged = self.recognizer._stage_crop(crop)
            inflight = self.recognizer._enqueue_crop(staged)
            finalized = self.recognizer._finalize_crop(inflight)
            if self.scheduling_metrics is not None:
                self.scheduling_metrics.record_prefill(
                    request_id,
                    pull_started,
                    time.perf_counter(),
                )
            return self.recognizer._ready_from_prefilled(finalized)

    def _submit_available(self, *, block_for_first: bool) -> None:
        while len(self.pending) < self.recognizer.cpu_preprocess_max_pending:
            request = self.requests.pull(
                block=block_for_first and not self.pending,
            )
            block_for_first = False
            if request is None:
                break
            submitted_at = time.perf_counter()
            if self.scheduling_metrics is not None:
                self.scheduling_metrics.register(
                    request.request_id,
                    submitted_at
                    if request.submitted_at is None
                    else request.submitted_at,
                )
            self.pending.append(
                (
                    request.request_id,
                    self.executor.submit(
                        self.recognizer._prepare_cpu,
                        request,
                        submitted_at,
                    ),
                )
            )

    @property
    def closed(self) -> bool:
        return bool(self.requests.closed) and not self.pending

    def close(self) -> None:
        if self._executor_closed:
            return
        self._executor_closed = True
        self.executor.shutdown(wait=True, cancel_futures=True)

    def __init__(
        self,
        recognizer: Any,
        requests: Any,
        *,
        on_request_error: Callable[[str, BaseException], None],
        scheduling_metrics: RequestSchedulingMetrics | None = None,
    ):
        self.recognizer = recognizer
        self.requests = requests
        self.on_request_error = on_request_error
        self.scheduling_metrics = scheduling_metrics
        self.pending: deque[tuple[str, Future[CpuPreparedRecognition]]] = deque()
        self.executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="paddleocr-vl-open-cpu-prepare",
        )
        self._executor_closed = False


# Crop state through CPU preparation, transfer, and prefill


@dataclass
class CpuPreparedRecognition:
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
    timing_s: dict[str, float]
    request_started: float
    preparation_finished: float
    input_fingerprints: dict[str, Any] = field(default_factory=dict)


@dataclass
class _PreparedCrop:
    crop_id: int
    prepared: CpuPreparedRecognition
    consumer_wait_s: float
    real_vision_tokens: int


@dataclass
class _StagedCrop:
    crop: _PreparedCrop
    device_timeline: DeviceTimeline
    h2d_ready_event: Any
    moved: tuple[torch.Tensor, ...]
    timing_s: dict[str, float]


@dataclass
class _InFlightCrop:
    crop_id: int
    prepared: CpuPreparedRecognition
    cache: LocalPaddleOCRVLStaticCache
    cache_lease: PrefillKVCacheLease
    rope_deltas: torch.Tensor
    next_cache_position: torch.Tensor
    next_token: torch.Tensor
    device_inputs: tuple[torch.Tensor, ...]
    vision: dict[str, Any]
    text_prefill: dict[str, Any]
    timing_s: dict[str, float]
    input_tokens: int
    projected_image_tokens: int
    device_timeline: DeviceTimeline
    h2d_ready_event: Any
    prefill_ready_event: Any
    first_token_device: torch.Tensor
    prefill_started: float


@dataclass
class PrefilledRecognition:
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
    timing_s: dict[str, float]
    device_stage_s: dict[str, float]
    request_started: float
    prefill_finished: float
    input_fingerprints: dict[str, Any] = field(default_factory=dict)

    def take_device_state(
        self,
    ) -> tuple[
        LocalPaddleOCRVLStaticCache,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Callable[[], None] | None,
    ]:
        """Move the pending NPU prefix out of the long-lived result payload."""

        cache = self.cache
        rope_deltas = self.rope_deltas
        next_cache_position = self.next_cache_position
        next_token = self.next_token
        cache_release = self.cache_release
        if (
            cache is None
            or rope_deltas is None
            or next_cache_position is None
            or next_token is None
        ):
            raise RuntimeError(
                f"prefill device state already taken for {self.request_id}"
            )
        self.cache = None
        self.cache_release = None
        self.rope_deltas = None
        self.next_cache_position = None
        self.next_token = None
        return (
            cache,
            rope_deltas,
            next_cache_position,
            next_token,
            cache_release,
        )


# Memory staging and setup reporting


def _pin_memory_or_keep(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.device.type != "cpu" or tensor.is_pinned():
        return tensor
    try:
        return tensor.pin_memory()
    except RuntimeError:
        # Pageable staging is slower to submit but has identical semantics.
        return tensor


def _emit_setup_progress(
    stage: str,
    status: str,
    elapsed_s: float | None = None,
) -> None:
    record: dict[str, Any] = {
        "stage": str(stage),
        "status": str(status),
    }
    if elapsed_s is not None:
        record["elapsed_s"] = round(float(elapsed_s), 6)
    print(
        "EXP09_SETUP " + json.dumps(record, ensure_ascii=False, separators=(",", ":")),
        file=sys.stderr,
        flush=True,
    )


# Prefill statistics


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
        # Preserve the existing external metrics schema until its separate review.
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
        # Compatibility fields are constants, not implemented packing alternatives.
        return dict(mode="off", buckets=[128, 256, 512, 1024], groups=self.calls,
            crops=self.calls, packs=0, packed_crops=0, fallback_crops=self.calls,
            calls=self.calls, call_reduction_fraction=0.0 if self.calls else None,
            pack_size_histogram={}, bucket_histogram={}, packed_real_text_tokens=0,
            packed_physical_text_tokens=0, packed_fill_fraction=None,
            redistributed_kv_bytes=0)
