"""Persistent PaddleOCR-VL recognizer: CPU preparation, NPU prefill, batched decode.

One ContinuousRecognizer owns the model for the life of the inference process.
serve() takes an open request source (p01_serve.py's InferenceWorker) and returns
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

This file owns the full request lifecycle, including the decode arena,
scheduler, prefill cache ownership and scheduling measurements.
"""

from __future__ import annotations

from contextlib import contextmanager
import io
import json
import sys
import time
from bisect import bisect_right
from collections import Counter, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import torch
import torch_npu
from PIL import Image
from tokenizers import Tokenizer

from p03_crop_processing import (
    prepare_prompt_tokens,
    preprocess_pil_image,
    PATCH_SIZE, MERGE_SIZE, MIN_PIXELS, MAX_PIXELS,
)
from p04_paddle_ocr_vl_1_6_modeling import (
    LocalPaddleOCRVLForConditionalGeneration,
    IMAGE_TOKEN_ID,
)
from p06_text_prefill_and_decode import (
    LocalPaddleOCRVLStaticCache,
    TEXT_EOS_TOKEN_ID,
    cast_decode_linear_weights_to_nz,
    load_decode_vocab_token_ids,
    prepare_decode_compact_lm_head,
    prepare_decode_projections,
)
from p05_vision_prefill import (
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


# Each stage records its own part of the request timing; the result step
# assembles the RequestTiming record defined below.


class ContinuousRecognizer:
    """Own the model and request-processing resources for the inference process.

    Construction performs setup once. serve() then stays active across requests,
    returning each result through a callback while other requests keep running.
    """

    # One-time setup comes first; serving and per-request operations follow.

    @torch.inference_mode()
    def __init__(
        self,
        *,
        model: str,
        device: str = "npu:0",
        batch_size: int,
        graph_cache_directory: Path,
        full_decode_lm_head: bool = False,
        setup_progress: Callable[[str, str, float | None], None] | None = None,
        eager: bool = False,
    ):
        """Load the model and create the streams, caches and scheduler once."""
        runtime_started = time.perf_counter()

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
        self.setup_progress = setup_progress or _emit_setup_progress
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
            graph_cache_directory=graph_cache_directory,
            decode_head_cache_key=decode_head_cache_key,
            batch_size=self.batch_size,
            cache_length=CACHE_LENGTH,
            device=self.device,
            eager=self.eager,
            setup_progress=self.setup_progress,
        )
        self.setup_timing_s.update(self.stages.setup_timing_s)
        self.vision_prefill = self.stages.vision_prefill
        self.text_prefill = self.stages.text_prefill
        self.text_decode = self.stages.text_decode
        torch_npu.npu.synchronize(self.device)

        # CPU preparation may work ahead while decode slots are occupied.
        # NPU prefill still requires available decode capacity; these buffers
        # do not authorize prefilling extra requests while every slot is busy.
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
            )
            # Decode's asynchronous token-copy resources are created here,
            # after the active cache slots, just as in the original setup.
            self.max_new_tokens = MAX_NEW_TOKENS
            self.eos_token_id = self.decode_arena.eos_token_id
            self.completion_policy = None
            self.progress = None
            self.diagnostic_effective_length = None
            self.diagnostic_request_id = None
            self.copy_stream = None
            self.host_token_ring = None
            if self.device.type == "npu":
                self.copy_stream = torch_npu.npu.Stream(device=self.device)
                self.host_token_ring = torch.empty(
                    (2, self.batch_size), dtype=torch.int64, pin_memory=True,
                )
        self.setup_timing_s["recognizer_runtime_total"] = time.perf_counter() - runtime_started
        self.setup_progress("recognizer_runtime", "done", self.setup_timing_s["recognizer_runtime_total"])


    @torch.inference_mode()
    def serve(
        self,
        requests: Any,
        *,
        schedule_id: str,
        emit_result: Callable[[RecognitionResult], None],
        on_request_error: Callable[[str, BaseException], None],
        collect_scheduling_metrics: bool = False,
        report_status: Callable[[], None] | None = None,
    ) -> ServingSummary:
        """Keep serving crops until shutdown, finishing any crops already accepted.

        The inference worker in p01_serve.py calls this once and stays here
        while the server runs. A temporary gap between images does not stop it.
        Each finished crop is sent to emit_result without waiting for shutdown.

        `requests` reads images from the HTTP-to-inference queue. During graceful
        shutdown, the HTTP side puts None in that queue to signal that no more
        images will arrive. Once remaining crops finish, this method returns
        totals for the whole serving period, before the inference process exits.

        CPU preparation runs ahead on one background thread. The decoding
        loop starts NPU prefill only when decode capacity is available.
        Preparation failures go to on_request_error so other crops can continue.
        """
        self._vision_prefill_stats = _VisionPrefillStats()
        self._text_prefill_stats = _TextPrefillStats()
        scheduling_metrics = (
            RequestSchedulingMetrics(self.batch_size) if collect_scheduling_metrics else None
        )
        self.requests = requests
        self.on_request_error = on_request_error
        self.scheduling_metrics = scheduling_metrics
        self.report_status = report_status
        self.output_tokens = 0
        # Prepared crops wait here in arrival order; only CPU work runs ahead.
        self.crops_awaiting_prefill: deque[tuple[str, Future[PreparedCrop]]] = deque()
        self.cpu_preparation_worker = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="paddleocr-vl-open-cpu-prepare",
        )
        try:
            return self._run_continuous_decoding_loop_until_pipeline_shutdown(
                schedule_id=schedule_id,
                emit_result=emit_result,
                scheduling_metrics=scheduling_metrics,
            )
        finally:
            self.cpu_preparation_worker.shutdown(wait=True, cancel_futures=True)


    def _run_continuous_decoding_loop_until_pipeline_shutdown(
        self,
        *,
        schedule_id: str,
        emit_result: Callable[[RecognitionResult], None],
        scheduling_metrics: RequestSchedulingMetrics | None = None,
    ) -> ServingSummary:
        """Keep generating and sending OCR results until shutdown work is finished.

        Quiet periods do not end this loop. It waits for another image when
        idle, and finishes accepted crops after the HTTP worker signals shutdown.
        The summary is returned only then, not after each crop or decode step.
        """
        self.decode_arena.begin_run()
        buffer_capacity = self.ready_buffer_capacity
        low_watermark = self.ready_buffer_low_watermark
        ready_queue: deque[DecodeRequest] = deque()
        self.ready_queue = ready_queue
        source_exhausted = (bool(self.requests.closed) and not self.crops_awaiting_prefill)
        submitted_count = 0
        unfinished_request_ids: set[str] = set()
        max_ready_queue_depth = 0
        ready_source_refill_count = 0
        completed_count = 0
        effective_tokens = 0
        graph_calls = 0
        initial_admissions = 0
        hot_swap_admissions = 0
        prefill_only_completions = 0
        active_decode_slots = 0
        initial_kv_bytes = 0
        hot_swap_kv_bytes = 0
        d2h_wait_wall_s = 0.0
        retire_and_refill_wall_s = 0.0
        hot_swap_safety_sync_wall_s = 0.0
        ready_source_wall_s = 0.0
        completion_callback_wall_s = 0.0
        refill_sequence = 0

        # The helpers below share this run's queue and counters. They report
        # progress, deliver completions, obtain prefilled requests, fill free
        # slots, and consume copied tokens. Their definitions do not run them;
        # the initial fill and repeated decode loop follow these definitions.

        def progress(event: str, **fields: Any) -> None:
            if self.progress is None:
                return
            active = [
                {
                    "slot": index,
                    "request_id": state.ready.request_id,
                    "tokens": len(state.token_ids),
                    "prompt_length": state.ready.prompt_length,
                }
                for index, state in enumerate(self.decode_arena.slots)
                if state is not None
            ]
            self._progress(
                event,
                active_count=len(active),
                active=active,
                ready_depth=len(ready_queue),
                source_exhausted=source_exhausted,
                submitted=submitted_count,
                completed=completed_count,
                **fields,
            )

        # Send a finished request immediately; the final run summary comes later.
        def record_and_report_finished_crop(completion: DecodeCompletion) -> None:
            nonlocal completion_callback_wall_s, completed_count, effective_tokens
            if scheduling_metrics is not None:
                completion.scheduling_metrics = scheduling_metrics.finish(
                    completion.ready.request_id,
                    completion.completed_at,
                )
            # The shutdown summary needs totals, not a history of token lists.
            # Result delivery keeps its own data alive for as long as it needs it.
            completed_count += 1
            effective_tokens += max(0, len(completion.token_ids) - 1)
            unfinished_request_ids.remove(completion.ready.request_id)
            started = time.perf_counter()
            emit_result(self._build_recognition_result(completion, schedule_id=schedule_id))
            finished = time.perf_counter()
            completion_callback_wall_s += finished - started

        # Ask the source for requests, respecting the available decode capacity.
        def refill_ready_queue(
            *,
            reason: str,
            block_if_idle: bool = False,
        ) -> None:
            nonlocal source_exhausted, ready_source_wall_s
            nonlocal max_ready_queue_depth, ready_source_refill_count
            nonlocal refill_sequence, submitted_count
            refill_sequence += 1
            refill_id = refill_sequence
            pulled = 0
            progress(
                "refill_begin",
                refill_id=refill_id,
                reason=reason,
                target_depth=buffer_capacity,
            )
            while not source_exhausted and len(ready_queue) < buffer_capacity:
                started = time.perf_counter()
                progress(
                    "ready_source_next_begin",
                    refill_id=refill_id,
                    reason=reason,
                    pull_index=pulled,
                )
                should_block = (
                    block_if_idle
                    and pulled == 0
                    and not ready_queue
                    and self.decode_arena.num_active == 0
                )
                try:
                    # Ready requests already reserve otherwise free slots.
                    # With no free slot, this still submits CPU preparation,
                    # but does not start another NPU prefill.
                    ready = self._prepare_next_crop_for_decode(
                        block=should_block,
                        available_slots=(
                            self.batch_size - self.decode_arena.num_active - len(ready_queue)
                        ),
                    )
                except BaseException as exc:
                    progress(
                        "ready_source_next_error",
                        refill_id=refill_id,
                        reason=reason,
                        pull_index=pulled,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                    raise
                finished = time.perf_counter()
                ready_source_wall_s += finished - started
                if ready is None:
                    source_exhausted = (bool(self.requests.closed) and not self.crops_awaiting_prefill)
                    progress(
                        (
                            "ready_source_exhausted"
                            if source_exhausted
                            else "ready_source_temporarily_empty"
                        ),
                        refill_id=refill_id,
                        reason=reason,
                        pull_index=pulled,
                        wait_s=finished - started,
                    )
                    break
                progress(
                    "ready_source_next_end",
                    refill_id=refill_id,
                    reason=reason,
                    pull_index=pulled,
                    request_id=ready.request_id,
                    wait_s=finished - started,
                )
                if ready.request_id in unfinished_request_ids:
                    raise ValueError(f"duplicate decode request id: {ready.request_id}")
                unfinished_request_ids.add(ready.request_id)
                submitted_count += 1
                ready_queue.append(ready)
                pulled += 1
                max_ready_queue_depth = max(max_ready_queue_depth, len(ready_queue))
            if pulled:
                ready_source_refill_count += 1
            progress(
                "refill_end",
                refill_id=refill_id,
                reason=reason,
                pulled=pulled,
            )

        # Move ready requests into free cache slots so decoding can include them.
        def fill_free_slots(*, hot_swap: bool) -> None:
            nonlocal initial_admissions, hot_swap_admissions
            nonlocal prefill_only_completions, initial_kv_bytes, hot_swap_kv_bytes
            for slot_index in self.decode_arena.free_slot_indices():
                while True:
                    if not ready_queue and not source_exhausted:
                        refill_ready_queue(reason="free_slot_empty_queue")
                    if not ready_queue:
                        break
                    ready = ready_queue.popleft()
                    self.output_tokens += 1  # Prefill's first token, including EOS.
                    progress(
                        "admission_begin",
                        hot_swap=hot_swap,
                        slot=slot_index,
                        request_id=ready.request_id,
                    )
                    prefill_stop_reason = None
                    if ready.first_token == self.eos_token_id:
                        prefill_stop_reason = "eos"
                    elif ready.prompt_length >= int(self.decode_arena.cache.cache_length):
                        prefill_stop_reason = "kv_cache_full"
                    elif self.max_new_tokens == 1:
                        prefill_stop_reason = "length"
                    if prefill_stop_reason is not None:
                        ready.release_device_state()
                        record_and_report_finished_crop(
                            DecodeCompletion(
                                ready=ready,
                                token_ids=[int(ready.first_token)],
                                stop_reason=prefill_stop_reason,
                                slot_index=None,
                                slot_epoch=None,
                                admitted_at=None,
                                first_decode_launched_at=None,
                                completed_at=time.perf_counter(),
                                iterations_launched=0,
                            )
                        )
                        prefill_only_completions += 1
                        continue
                    _state, copied_bytes = self.decode_arena.admit(slot_index, ready)
                    progress(
                        "admission_end",
                        hot_swap=hot_swap,
                        slot=slot_index,
                        request_id=ready.request_id,
                        useful_prefix_bytes=copied_bytes,
                    )
                    if hot_swap:
                        hot_swap_admissions += 1
                        hot_swap_kv_bytes += copied_bytes
                    else:
                        initial_admissions += 1
                        initial_kv_bytes += copied_bytes
                    break

        # Consume a completed token copy, finish requests, and reuse freed slots.
        def process_copied_tokens_and_refill_slots(
            pending_copy: PendingTokenCopy,
            *,
            iteration: int,
            refill_reason: str,
        ) -> None:
            nonlocal d2h_wait_wall_s, retire_and_refill_wall_s
            nonlocal hot_swap_safety_sync_wall_s
            progress(
                "pending_token_wait_begin",
                iteration=iteration,
                pending_iteration=pending_copy.iteration,
            )
            diagnostic_slots = self._diagnostic_slots(pending_copy)
            if diagnostic_slots:
                progress(
                    "diagnostic_pending_state",
                    iteration=iteration,
                    pending_iteration=pending_copy.iteration,
                    diagnostic_slots=diagnostic_slots,
                )
                if pending_copy.diagnostic_compute_event is None:
                    raise RuntimeError(
                        "targeted decode diagnostic lost its compute event"
                    )
                progress(
                    "diagnostic_compute_sync_begin",
                    iteration=iteration,
                    pending_iteration=pending_copy.iteration,
                    diagnostic_slots=diagnostic_slots,
                )
                compute_sync_started = time.perf_counter()
                try:
                    pending_copy.diagnostic_compute_event.synchronize()
                except BaseException as exc:
                    progress(
                        "diagnostic_compute_sync_error",
                        iteration=iteration,
                        pending_iteration=pending_copy.iteration,
                        diagnostic_slots=diagnostic_slots,
                        wait_s=time.perf_counter() - compute_sync_started,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                    raise
                progress(
                    "diagnostic_compute_sync_end",
                    iteration=iteration,
                    pending_iteration=pending_copy.iteration,
                    diagnostic_slots=diagnostic_slots,
                    wait_s=time.perf_counter() - compute_sync_started,
                )
                progress(
                    "diagnostic_d2h_sync_begin",
                    iteration=iteration,
                    pending_iteration=pending_copy.iteration,
                    diagnostic_slots=diagnostic_slots,
                )
            try:
                host_tokens, wait_s = self._wait_for_copied_tokens(pending_copy)
            except BaseException as exc:
                if diagnostic_slots:
                    progress(
                        "diagnostic_d2h_sync_error",
                        iteration=iteration,
                        pending_iteration=pending_copy.iteration,
                        diagnostic_slots=diagnostic_slots,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                raise
            if diagnostic_slots:
                progress(
                    "diagnostic_d2h_sync_end",
                    iteration=iteration,
                    pending_iteration=pending_copy.iteration,
                    diagnostic_slots=diagnostic_slots,
                    wait_s=wait_s,
                )
            progress(
                "pending_token_wait_end",
                iteration=iteration,
                pending_iteration=pending_copy.iteration,
                wait_s=wait_s,
            )
            d2h_wait_wall_s += wait_s
            started = time.perf_counter()
            completed_before = completed_count
            if scheduling_metrics is not None:
                scheduling_metrics.consume(
                    state.ready.request_id
                    for slot_index, was_active in enumerate(pending_copy.active_slots)
                    if was_active
                    and (state := self.decode_arena.slots[slot_index]) is not None
                    and state.epoch == pending_copy.slot_epochs[slot_index]
                )
            for slot_index, was_active in enumerate(pending_copy.active_slots):
                if not was_active:
                    continue
                state = self.decode_arena.slots[slot_index]
                expected_epoch = pending_copy.slot_epochs[slot_index]
                if state is None or state.epoch != expected_epoch:
                    continue
                token_id = int(host_tokens[slot_index])
                state.token_ids.append(token_id)
                self.output_tokens += 1  # Only retained tokens; stale copies were skipped above.
                stop_reason = self._completion_reason(state, token_id)
                if stop_reason is not None:
                    released = self.decode_arena.release(slot_index)
                    record_and_report_finished_crop(
                        DecodeCompletion(
                            ready=released.ready,
                            token_ids=list(released.token_ids),
                            stop_reason=stop_reason,
                            slot_index=slot_index,
                            slot_epoch=released.epoch,
                            admitted_at=released.admitted_at,
                            first_decode_launched_at=released.first_decode_launched_at,
                            completed_at=time.perf_counter(),
                            iterations_launched=released.iterations_launched,
                        )
                    )
            progress(
                "retire_end",
                iteration=iteration,
                pending_iteration=pending_copy.iteration,
                newly_completed=completed_count - completed_before,
            )
            newly_completed = completed_count - completed_before
            if newly_completed and (ready_queue or not source_exhausted):
                # The next decode graph is submitted before the previous
                # sampled tokens are retired so its D2H can overlap compute.
                # A slot that just completed therefore still participates in
                # that speculative graph.  TorchAir's in-place KV writes are
                # not reliably protected from an immediately following
                # hot-swap copy by enqueue order alone on every Ascend target.
                # Resolve only the compute stream at an actual replacement
                # boundary; iterations without a hot swap remain pipelined.
                progress(
                    "hot_swap_safety_sync_begin",
                    iteration=iteration,
                    newly_completed=newly_completed,
                )
                safety_started = time.perf_counter()
                torch.npu.current_stream(self.device).synchronize()
                safety_wait_s = time.perf_counter() - safety_started
                hot_swap_safety_sync_wall_s += safety_wait_s
                progress(
                    "hot_swap_safety_sync_end",
                    iteration=iteration,
                    newly_completed=newly_completed,
                    wait_s=safety_wait_s,
                )
            progress("hot_swap_admission_begin", iteration=iteration)
            fill_free_slots(hot_swap=True)
            progress("hot_swap_admission_end", iteration=iteration)
            if len(ready_queue) < low_watermark:
                refill_ready_queue(reason=refill_reason)
            finished = time.perf_counter()
            retire_and_refill_wall_s += finished - started

        # Begin execution: obtain the first requests and fill available slots.
        progress("scheduler_device_sync_begin", phase="before_initial_fill")
        torch_npu.npu.synchronize(self.device)
        progress("scheduler_device_sync_end", phase="before_initial_fill")
        scheduler_started = time.perf_counter()
        progress(
            "scheduler_run_begin",
            buffer_capacity=buffer_capacity,
            low_watermark=low_watermark,
        )
        if len(ready_queue) < low_watermark:
            refill_ready_queue(
                reason="initial_low_watermark",
                block_if_idle=True,
            )
        progress("initial_admission_begin")
        fill_free_slots(hot_swap=False)
        progress("initial_admission_end")
        refill_ready_queue(reason="initial_top_up")
        previous_token_copy: PendingTokenCopy | None = None
        iteration = 0

        # Repeatedly decode active requests, consume the previous token copy,
        # and admit new requests as slots become free.
        while True:
            if self.report_status is not None:
                self.report_status()
            if self.decode_arena.num_active == 0:
                if not ready_queue and not source_exhausted:
                    refill_ready_queue(
                        reason="idle_wait_for_request",
                        block_if_idle=True,
                    )
                if ready_queue:
                    progress("idle_admission_begin", iteration=iteration)
                    fill_free_slots(hot_swap=graph_calls > 0)
                    progress("idle_admission_end", iteration=iteration)
                    refill_ready_queue(reason="idle_top_up")
                if self.decode_arena.num_active == 0:
                    if source_exhausted:
                        break
                    continue
            progress(
                "iteration_begin",
                iteration=iteration,
                pending_iteration=(None if previous_token_copy is None else previous_token_copy.iteration),
            )
            boundary_slots = [
                index
                for index, state in enumerate(self.decode_arena.slots)
                if state is not None
                and int(state.ready.prompt_length) + int(state.iterations_launched)
                >= int(self.decode_arena.cache.cache_length)
            ]
            if previous_token_copy is not None and boundary_slots:
                progress(
                    "kv_cache_boundary_drain_begin",
                    iteration=iteration,
                    pending_iteration=previous_token_copy.iteration,
                    slots=boundary_slots,
                )
                process_copied_tokens_and_refill_slots(
                    previous_token_copy,
                    iteration=iteration,
                    refill_reason="kv_cache_boundary",
                )
                progress(
                    "kv_cache_boundary_drain_end",
                    iteration=iteration,
                    slots=boundary_slots,
                )
                previous_token_copy = None
                continue
            progress("decode_step_begin", iteration=iteration)
            if scheduling_metrics is not None:
                scheduling_metrics.step(
                    (
                        state.ready.request_id
                        for state in self.decode_arena.slots
                        if state is not None
                    ),
                    time.perf_counter(),
                )
            step = self.decode_arena.step(self.text_decode.fn)
            progress("decode_step_end", iteration=iteration)
            graph_calls += 1
            active_decode_slots += sum(step.active_slots)
            progress("token_copy_schedule_begin", iteration=iteration)
            new_token_copy = self._start_copying_tokens_to_cpu(step, iteration)
            progress("token_copy_schedule_end", iteration=iteration)

            if previous_token_copy is not None:
                process_copied_tokens_and_refill_slots(
                    previous_token_copy,
                    iteration=iteration,
                    refill_reason="steady_low_watermark",
                )

            previous_token_copy = new_token_copy
            iteration += 1
            progress(
                "iteration_end",
                iteration=iteration - 1,
                next_iteration=iteration,
            )
            if self.decode_arena.num_active == 0:
                progress(
                    "final_token_drain_begin",
                    iteration=iteration,
                    pending_iteration=previous_token_copy.iteration,
                )
                _ignored, wait_s = self._wait_for_copied_tokens(previous_token_copy)
                progress(
                    "final_token_drain_end",
                    iteration=iteration,
                    pending_iteration=previous_token_copy.iteration,
                    wait_s=wait_s,
                )
                d2h_wait_wall_s += wait_s
                previous_token_copy = None
                continue

        progress("scheduler_device_sync_begin", phase="after_decode_loop")
        torch_npu.npu.synchronize(self.device)
        progress("scheduler_device_sync_end", phase="after_decode_loop")
        scheduler_wall_s = time.perf_counter() - scheduler_started
        decode_host_exclusive_wall_s = max(
            0.0,
            scheduler_wall_s - ready_source_wall_s - completion_callback_wall_s,
        )

        if ready_queue or not source_exhausted:
            raise AssertionError(
                f"continuous decode stopped with {len(ready_queue)} ready requests"
            )
        if completed_count != submitted_count:
            raise AssertionError(
                f"continuous decode completed {completed_count} of {submitted_count} requests"
            )

        raw_slots = graph_calls * self.batch_size
        idle_slots = raw_slots - active_decode_slots
        lookahead_slots = active_decode_slots - effective_tokens
        if idle_slots < 0 or lookahead_slots < 0:
            raise AssertionError(
                "continuous decode accounting went negative: "
                f"raw={raw_slots} active={active_decode_slots} effective={effective_tokens}"
            )
        if raw_slots != effective_tokens + idle_slots + lookahead_slots:
            raise AssertionError("continuous decode slot accounting does not balance")

        private_cache_pool_stats = self.prefill_cache_pool.stats()
        if int(private_cache_pool_stats["active_slots"]) != 0:
            raise RuntimeError(
                "prefill KV cache arena still owns active request slots after decode: "
                f"{private_cache_pool_stats}"
            )

        def fraction_of_decode_token_slots(numerator: int) -> float | None:
            if raw_slots <= 0:
                return None
            return float(numerator) / float(raw_slots)

        # One summary for the serving lifetime. Per-crop results have already
        # been delivered; these are the existing totals, not a live log.
        return ServingSummary(
            schedule_id=schedule_id,
            batch_size=self.batch_size,
            requests=submitted_count,
            ready_buffer_capacity=buffer_capacity,
            ready_buffer_low_watermark=low_watermark,
            max_ready_queue_depth=max_ready_queue_depth,
            ready_source_refill_count=ready_source_refill_count,
            graph_calls=graph_calls,
            initial_admissions=initial_admissions,
            hot_swap_admissions=hot_swap_admissions,
            prefill_only_completions=prefill_only_completions,
            raw_decode_token_slots=raw_slots,
            active_decode_token_slots=active_decode_slots,
            effective_decode_tokens=effective_tokens,
            idle_decode_token_slots=idle_slots,
            lookahead_decode_token_slots=lookahead_slots,
            kv_prefix_bytes_copied=self.decode_arena.kv_prefix_bytes_copied,
            initial_kv_prefix_bytes_copied=initial_kv_bytes,
            hot_swap_kv_prefix_bytes_copied=hot_swap_kv_bytes,
            timing_s={
                "decode_host_exclusive_wall": float(decode_host_exclusive_wall_s),
                "run_scoped_scheduler_wall": float(scheduler_wall_s),
                "ready_source_wall": float(ready_source_wall_s),
                "completion_callback_wall": float(completion_callback_wall_s),
                "slot_admission_enqueue_wall": float(
                    self.decode_arena.admission_enqueue_wall_s
                ),
                "d2h_wait_wall": float(d2h_wait_wall_s),
                "retire_and_refill_host_wall": float(retire_and_refill_wall_s),
                "hot_swap_safety_sync_wall": float(hot_swap_safety_sync_wall_s),
            },
            vision_packing=self._vision_prefill_stats.summary(),
            text_packing={
                **self._text_prefill_stats.summary(),
                "private_cache_pool": private_cache_pool_stats,
            },
            rates={
                "output_tok_per_s": per_second(self.output_tokens, float(scheduler_wall_s)),
                "effective_fraction": fraction_of_decode_token_slots(effective_tokens),
                "active_slot_fraction": fraction_of_decode_token_slots(active_decode_slots),
                "scheduler_effective_tok_per_s": per_second(effective_tokens, float(scheduler_wall_s)),
            },
        )


    # Preparing upcoming crops: CPU lookahead, then prefill for a free decode slot.


    def _prepare_next_crop_for_decode(self, *, block: bool, available_slots: int) -> DecodeRequest | None:
        """Return the next prefilled request, or None when none is ready right now."""
        allow_prefill = available_slots > 0
        while True:
            pull_started = time.perf_counter() if self.scheduling_metrics is not None else 0.0
            self._submit_available_crops_for_cpu_preparation(block_for_first=allow_prefill and block and not self.crops_awaiting_prefill)
            if not allow_prefill:
                # Every decode slot is busy or reserved. CPU preparation above
                # keeps running ahead; NPU prefill waits for a free slot.
                return None
            if not self.crops_awaiting_prefill:
                return None
            if self.scheduling_metrics is not None:
                self.scheduling_metrics.cpu_prefill_eligible(self.crops_awaiting_prefill[0][0], block=block)
            if not block and not self.crops_awaiting_prefill[0][1].done():
                # Live decoding must not stall on CPU work. Only an idle
                # scheduler (block=True) waits for the first prepared request.
                return None
            request_id, future = self.crops_awaiting_prefill.popleft()
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
            self._submit_available_crops_for_cpu_preparation(block_for_first=False)
            ready = self._prefill_for_decode(prepared, consumer_wait_s)
            if self.scheduling_metrics is not None:
                self.scheduling_metrics.record_prefill(request_id, pull_started, time.perf_counter())
            return ready


    def _submit_available_crops_for_cpu_preparation(self, *, block_for_first: bool) -> None:
        """Hand new requests to the CPU thread until the lookahead limit is reached."""
        while len(self.crops_awaiting_prefill) < self.cpu_preprocess_max_pending:
            request = self.requests.pull(block=block_for_first and not self.crops_awaiting_prefill)
            block_for_first = False
            if request is None:
                break
            submitted_at = time.perf_counter()
            if self.scheduling_metrics is not None:
                self.scheduling_metrics.register(
                    request.request_id,
                    submitted_at if request.submitted_at is None else request.submitted_at,
                )
            self.crops_awaiting_prefill.append(
                (request.request_id, self.cpu_preparation_worker.submit(self._prepare_cpu, request, submitted_at))
            )


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


    @torch.inference_mode()
    def _prefill_for_decode(self, prepared_crop: PreparedCrop, consumer_wait_s: float) -> DecodeRequest:
        """Copy one crop to the NPU, run vision/text prefill, and obtain its first token.

        Called only when decode has room. The request is not finished: its
        prompt KV cache and first token will enter a shared decode slot next.
        """
        # Copy inputs on the transfer stream; the compute stream waits on its event.
        submit_started = time.perf_counter()
        with torch_npu.npu.stream(self.prefill_transfer_stream):
            ready_wait_s = max(0.0, time.perf_counter() - prepared_crop.preparation_finished)

            def copy_input_tensors() -> tuple[torch.Tensor, ...]:
                pixels = prepared_crop.pixel_values.to(device=self.device, non_blocking=True)
                return (
                    prepared_crop.input_ids.to(self.device, non_blocking=True),
                    prepared_crop.attention_mask.to(self.device, non_blocking=True),
                    pixels,
                    prepared_crop.position_ids.to(self.device, non_blocking=True),
                    prepared_crop.rope_deltas.to(self.device, non_blocking=True),
                )

            device_inputs = copy_input_tensors()
            h2d_ready_event = self.prefill_transfer_stream.record_event()
        consumer_wait_s = float(consumer_wait_s)
        prefill_h2d_submit_host = time.perf_counter() - submit_started

        # Run normalization, vision, projector, text prefill and first-token selection.
        enqueue_started = time.perf_counter()
        torch_npu.npu.current_stream().wait_event(h2d_ready_event)
        prefill_started = time.perf_counter()
        input_ids, attention_mask, pixels, position_ids, rope_deltas = device_inputs

        def normalize_uint8() -> torch.Tensor:
            output = pixels.to(torch.float32)
            output.mul_(1.0 / 255.0)
            output.sub_(0.5)
            output.div_(0.5)
            return output.to(self.model.visual.dtype).contiguous()

        pixels = normalize_uint8()
        vision_model = self.model.visual.vision_model
        hidden = vision_model.embeddings(
            pixels.unsqueeze(0), image_grid_thw=prepared_crop.image_grid_thw,
        )
        real_length = int(hidden.shape[0])
        vision_route = self.vision_prefill.route(real_length)
        prepared_vision = self.vision_prefill.prepare(
            hidden, prepared_crop.image_grid_thw, route=vision_route,
        )
        features = self.vision_prefill.run_prepared(prepared_vision)
        self._vision_prefill_stats.record(vision_route)
        next_position = torch.full((1,), int(input_ids.shape[1]), device=self.device, dtype=torch.int64)
        image_embeds = self.model.mlp_AR(features, prepared_crop.image_grid_thw)
        inputs_embeds = self.model.model.embed_tokens(input_ids)

        def scatter_image_embeds() -> torch.Tensor:
            projected = image_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            image_mask = (input_ids == IMAGE_TOKEN_ID).unsqueeze(-1).expand_as(inputs_embeds)
            return inputs_embeds.masked_scatter(image_mask, projected)

        inputs_embeds = scatter_image_embeds()
        lease = self.prefill_cache_pool.acquire()
        self._text_prefill_stats.record()
        text_route = self.text_prefill.route(int(inputs_embeds.shape[1]))
        prepared_text = self.text_prefill.prepare(
            inputs_embeds, attention_mask, position_ids, route=text_route,
        )
        last_hidden = self.text_prefill.run_prepared(prepared_text, lease.cache)
        logits = self.model.lm_head(last_hidden)
        next_token = torch.argmax(logits[:, -1, :].float(), dim=-1, keepdim=True)
        text_route = {
            **text_route,
            "private_cache_slot_index": int(lease.slot_index),
            "private_cache_generation": int(lease.generation),
        }
        # A one-element copy of the first token, owned by this request, for the transfer to CPU below.
        first_token_device = torch.cat([next_token.detach().reshape(-1)], dim=0).contiguous()
        prefill_ready_event = torch_npu.npu.current_stream().record_event()
        prompt_length = int(prepared_crop.input_ids.shape[1])
        projected_image_tokens = int(image_embeds.shape[0])
        prefill_enqueue_host = time.perf_counter() - enqueue_started

        # These temporaries previously left scope when submission returned.
        # Keep that lifetime boundary: only the inputs, cache and first-token
        # state must remain referenced while we wait for prefill to finish.
        del normalize_uint8, scatter_image_embeds
        del pixels, hidden, prepared_vision, features, image_embeds, inputs_embeds
        del prepared_text, last_hidden, logits

        # Wait for the first token on CPU before admitting this crop to decode.
        resolve_started = time.perf_counter()
        # Preserve the prefill completion wait before copying its first token.
        # This existing dependency event is not a profiling event.
        prefill_ready_event.synchronize()
        started = time.perf_counter()
        with torch_npu.npu.stream(self.prefill_transfer_stream):
            self.prefill_transfer_stream.wait_event(prefill_ready_event)
            self.prefill_host_tokens[:1].copy_(first_token_device, non_blocking=True)
            first_token_ready = self.prefill_transfer_stream.record_event()
        first_token_ready.synchronize()
        first_token = int(self.prefill_host_tokens[:1].tolist()[0])
        first_token_d2h_s = time.perf_counter() - started
        resolve_finished = time.perf_counter()

        cpu_timing = prepared_crop.cpu_timing

        prefill_wall_s = resolve_finished - prefill_started
        prefill_timing = PrefillTiming(
            cpu_preprocess_background_consumer_wait=consumer_wait_s,
            cpu_preprocess_background_ready_wait=ready_wait_s,
            prefill_h2d_submit_host=prefill_h2d_submit_host,
            prefill_enqueue_host=prefill_enqueue_host,
            first_token_d2h=first_token_d2h_s,
            prefill_resolve_wait=resolve_finished - resolve_started,
            vision_and_text_prefill_wall=prefill_wall_s,
            time_to_first_token=resolve_finished - prepared_crop.request_started,
            prefill_request_total=(
                cpu_timing.cpu_image_and_prompt_preprocess + cpu_timing.cpu_mrope_index + cpu_timing.cpu_pin_memory
                + prefill_h2d_submit_host + prefill_wall_s
            ),
        )
        return DecodeRequest(
            request_id=prepared_crop.request_id,
            prompt=prepared_crop.prompt,
            crop_size=prepared_crop.crop_size,
            skip_special_tokens=prepared_crop.skip_special_tokens,
            cache=lease.cache,
            cache_lease=lease,
            rope_deltas=rope_deltas,
            cache_position=next_position,
            first_token_tensor=next_token,
            first_token=first_token,
            prompt_length=prompt_length,
            projected_image_tokens=projected_image_tokens,
            vision=vision_route,
            text_prefill=text_route,
            cpu_timing=cpu_timing,
            prefill_timing=prefill_timing,
            request_started=prepared_crop.request_started,
            prefill_finished=resolve_finished,
        )


    def _build_recognition_result(
        self, completed_crop: DecodeCompletion, *, schedule_id: str,
    ) -> RecognitionResult:
        """Detokenize one finished request and attach its timings."""
        prefill_result = completed_crop.ready
        token_ids = completed_crop.token_ids
        started = time.perf_counter()
        text = self.tokenizer.decode(token_ids, skip_special_tokens=prefill_result.skip_special_tokens)
        detokenize_s = time.perf_counter() - started

        generated_tokens = len(token_ids)
        admitted_at = completed_crop.admitted_at
        cpu_timing = prefill_result.cpu_timing
        prefill_timing = prefill_result.prefill_timing
        request_timing = RequestTiming(
            cpu_image_decode=cpu_timing.cpu_image_decode,
            cpu_image_and_prompt_preprocess=cpu_timing.cpu_image_and_prompt_preprocess,
            cpu_mrope_index=cpu_timing.cpu_mrope_index,
            cpu_pin_memory=cpu_timing.cpu_pin_memory,
            cpu_preprocess_background_queue_wait=cpu_timing.cpu_preprocess_background_queue_wait,
            cpu_preprocess_background_service=cpu_timing.cpu_preprocess_background_service,
            cpu_preprocess_background_consumer_wait=prefill_timing.cpu_preprocess_background_consumer_wait,
            cpu_preprocess_background_ready_wait=prefill_timing.cpu_preprocess_background_ready_wait,
            prefill_h2d_submit_host=prefill_timing.prefill_h2d_submit_host,
            prefill_enqueue_host=prefill_timing.prefill_enqueue_host,
            first_token_d2h=prefill_timing.first_token_d2h,
            prefill_resolve_wait=prefill_timing.prefill_resolve_wait,
            vision_and_text_prefill_wall=prefill_timing.vision_and_text_prefill_wall,
            time_to_first_token=prefill_timing.time_to_first_token,
            prefill_request_total=prefill_timing.prefill_request_total,
            decode_ready_queue_wait=(
                max(0.0, admitted_at - prefill_result.prefill_finished) if admitted_at is not None else 0.0
            ),
            decode_slot_residency=(
                max(0.0, completed_crop.completed_at - admitted_at) if admitted_at is not None else 0.0
            ),
            detokenize=float(detokenize_s),
            request_total=float(completed_crop.completed_at - prefill_result.request_started + detokenize_s),
        )
        return RecognitionResult(
            request_id=prefill_result.request_id,
            decode_schedule_id=schedule_id,
            decode_slot_index=completed_crop.slot_index,
            decode_slot_epoch=completed_crop.slot_epoch,
            prompt=prefill_result.prompt,
            crop_size=prefill_result.crop_size,
            text=text,
            token_ids=token_ids,
            stop_reason=completed_crop.stop_reason,
            input_tokens=prefill_result.prompt_length,
            projected_image_tokens=prefill_result.projected_image_tokens,
            generated_tokens_including_eos=generated_tokens,
            decode_tokens_after_prefill_including_eos=max(0, generated_tokens - 1),
            decode_calls_executed=completed_crop.iterations_launched,
            timing_s=request_timing,
            rates={
                "request_output_tok_per_s": per_second(generated_tokens, request_timing.request_total),
            },
            vision=dict(prefill_result.vision),
            text_prefill=dict(prefill_result.text_prefill),
            scheduling_metrics=dict(completed_crop.scheduling_metrics),
        )


    # Token copies and completion checks used by the continuous loop above.


    def _completion_reason(
        self,
        state: DecodeSlotState,
        token_id: int,
    ) -> str | None:
        generated_tokens = len(state.token_ids)
        cache_is_full = int(state.ready.prompt_length) + generated_tokens - 1 >= int(
            self.decode_arena.cache.cache_length
        )
        if self.completion_policy is not None:
            if cache_is_full:
                return "kv_cache_full"
            reason = self.completion_policy(state, token_id)
            if reason is not None and not reason:
                raise ValueError("completion policy returned an empty stop reason")
            return reason
        if token_id == self.eos_token_id:
            return "eos"
        if cache_is_full:
            return "kv_cache_full"
        if generated_tokens >= self.max_new_tokens:
            return "length"
        return None


    def _start_copying_tokens_to_cpu(
        self,
        step: DecodeStep,
        iteration: int,
    ) -> PendingTokenCopy:
        if self.device.type == "npu":
            import torch_npu

            assert self.copy_stream is not None
            assert self.host_token_ring is not None
            ring_index = iteration % 2
            diagnostic_compute_event = None
            if self._diagnostic_slots(step):
                # This is separate from the copy stream's dependency event.
                # Retaining it does not change the lifetime of the production
                # ready_event; it gives the diagnostic path a precise compute
                # completion boundary to synchronize.
                diagnostic_compute_event = torch_npu.npu.current_stream().record_event()
            ready_event = torch_npu.npu.current_stream().record_event()
            done_event = torch_npu.npu.Event()
            with torch_npu.npu.stream(self.copy_stream):
                self.copy_stream.wait_event(ready_event)
                self.host_token_ring[ring_index].copy_(
                    step.sampled.reshape(-1),
                    non_blocking=True,
                )
                done_event.record(self.copy_stream)
            return PendingTokenCopy(
                iteration=iteration,
                active_slots=step.active_slots,
                slot_epochs=step.slot_epochs,
                slot_request_ids=step.slot_request_ids,
                cache_positions=step.cache_positions,
                generated_token_counts=step.generated_token_counts,
                ring_index=ring_index,
                done_event=done_event,
                diagnostic_compute_event=diagnostic_compute_event,
                host_tokens=None,
            )
        return PendingTokenCopy(
            iteration=iteration,
            active_slots=step.active_slots,
            slot_epochs=step.slot_epochs,
            slot_request_ids=step.slot_request_ids,
            cache_positions=step.cache_positions,
            generated_token_counts=step.generated_token_counts,
            ring_index=None,
            done_event=None,
            diagnostic_compute_event=None,
            host_tokens=[
                int(value) for value in step.sampled.detach().cpu().reshape(-1).tolist()
            ],
        )


    def _wait_for_copied_tokens(self, previous_token_copy: PendingTokenCopy) -> tuple[list[int], float]:
        started = time.perf_counter()
        if previous_token_copy.done_event is not None:
            previous_token_copy.done_event.synchronize()
            assert self.host_token_ring is not None
            assert previous_token_copy.ring_index is not None
            tokens = [
                int(value)
                for value in self.host_token_ring[previous_token_copy.ring_index].tolist()
            ]
        else:
            assert previous_token_copy.host_tokens is not None
            tokens = previous_token_copy.host_tokens
        return tokens, time.perf_counter() - started


    def _diagnostic_slots(
        self,
        step: DecodeStep | PendingTokenCopy,
    ) -> list[dict[str, int | str]]:
        target_length = self.diagnostic_effective_length
        if target_length is None:
            return []
        matches: list[dict[str, int | str]] = []
        for slot, (request_id, cache_position, generated_tokens) in enumerate(
            zip(
                step.slot_request_ids,
                step.cache_positions,
                step.generated_token_counts,
            )
        ):
            if request_id is None or cache_position is None:
                continue
            if (
                self.diagnostic_request_id is not None
                and request_id != self.diagnostic_request_id
            ):
                continue
            effective_length = int(cache_position) + 1
            if effective_length != target_length:
                continue
            matches.append(
                {
                    "slot": slot,
                    "request_id": request_id,
                    "cache_position": int(cache_position),
                    "effective_length": effective_length,
                    "generated_tokens": int(generated_tokens or 0),
                }
            )
        return matches


    def _progress(self, event: str, **fields: Any) -> None:
        if self.progress is not None:
            self.progress(event, **fields)


    # Model setup and configuration reporting.


    @contextmanager
    def _setup_stage(self, name: str) -> Iterator[None]:
        """Log and time one setup stage; waits for the NPU so the time is real."""
        self.setup_progress(name, "start")
        started = time.perf_counter()
        yield
        torch_npu.npu.synchronize(self.device)
        self.setup_timing_s[name] = time.perf_counter() - started
        self.setup_progress(name, "done", self.setup_timing_s[name])


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
            "decode_device_timing": False,
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


# The scheduler above decides when to admit, step and release requests.
# DecodeArena below performs those operations on the shared NPU cache slots.


class DecodeArena:
    """Own the tensors whose shapes and identities remain stable across steps."""

    def __init__(
        self,
        *,
        cache: LocalPaddleOCRVLStaticCache,
        device: torch.device,
        batch_size: int,
        eos_token_id: int,
    ) -> None:
        self.cache = cache
        self.device = device
        self.batch_size = int(batch_size)
        self.eos_token_id = int(eos_token_id)
        self.next_token = torch.full(
            (self.batch_size, 1),
            self.eos_token_id,
            device=self.device,
            dtype=torch.int64,
        )
        self.cache_position = torch.zeros(
            (self.batch_size,),
            device=self.device,
            dtype=torch.int64,
        )
        self.rope_deltas = torch.zeros(
            (self.batch_size, 1),
            device=self.device,
            dtype=torch.int64,
        )
        self.active_mask = torch.zeros(
            (self.batch_size,),
            device=self.device,
            dtype=torch.bool,
        )
        self.active_increment = torch.zeros_like(self.cache_position)
        self.slots: list[DecodeSlotState | None] = [None] * self.batch_size
        self._epochs = [0] * self.batch_size
        self.admission_enqueue_wall_s = 0.0
        self.kv_prefix_bytes_copied = 0

    def begin_run(self) -> None:
        if any(slot is not None for slot in self.slots):
            raise RuntimeError("decode arena still contains active slots")
        self.admission_enqueue_wall_s = 0.0
        self.kv_prefix_bytes_copied = 0
        self.next_token.fill_(self.eos_token_id)
        self.cache_position.zero_()
        self.rope_deltas.zero_()
        self.active_mask.zero_()
        self.active_increment.zero_()

    def admit(
        self,
        slot_index: int,
        ready: DecodeRequest,
    ) -> tuple[DecodeSlotState, int]:
        if self.slots[slot_index] is not None:
            raise RuntimeError(f"decode slot {slot_index} is not free")
        source_cache = ready.cache
        source_rope_deltas = ready.rope_deltas
        source_cache_position = ready.cache_position
        source_first_token = ready.first_token_tensor
        if (
            source_cache is None
            or source_rope_deltas is None
            or source_cache_position is None
            or source_first_token is None
        ):
            raise RuntimeError(
                f"request {ready.request_id} no longer owns prefill device state"
            )
        prompt_length = int(ready.prompt_length)
        if prompt_length <= 0 or prompt_length > int(self.cache.cache_length):
            raise ValueError(
                f"request {ready.request_id} has invalid prompt length {prompt_length}"
            )
        if len(source_cache.key_caches) != len(self.cache.key_caches):
            raise ValueError("ready cache and decode arena have different layer counts")
        if int(source_cache.cache_length) != int(self.cache.cache_length):
            raise ValueError(
                "ready cache and decode arena have different cache lengths"
            )

        source_tensors = source_cache.flat_tensors()
        destination_tensors = tuple(
            destination[slot_index : slot_index + 1]
            for destination in self.cache.flat_tensors()
        )
        source_heads = int(source_tensors[0].shape[1])
        destination_heads = int(destination_tensors[0].shape[1])
        if destination_heads % source_heads != 0:
            raise ValueError(
                "decode arena KV heads must equal or be an integer multiple "
                f"of prefill KV heads: source={source_heads}, "
                f"destination={destination_heads}"
            )
        cache_head_expansion = destination_heads // source_heads
        useful_prefix_bytes = sum(
            int(source[:, :, :prompt_length, :].numel()) * source.element_size()
            for source in source_tensors
        )

        def copy_state() -> None:
            if cache_head_expansion == 1:
                torch._foreach_copy_(destination_tensors, source_tensors)
            else:
                for destination, source in zip(
                    destination_tensors,
                    source_tensors,
                    strict=True,
                ):
                    batch_size, kv_heads, cache_length, head_dim = source.shape
                    expanded = source[:, :, None, :, :].expand(
                        batch_size,
                        kv_heads,
                        cache_head_expansion,
                        cache_length,
                        head_dim,
                    )
                    destination.view_as(expanded).copy_(expanded)
            self.rope_deltas[slot_index : slot_index + 1].copy_(source_rope_deltas)
            self.cache_position[slot_index : slot_index + 1].copy_(
                source_cache_position.reshape(1)
            )
            self.next_token[slot_index : slot_index + 1].copy_(source_first_token)
            self.active_mask[slot_index].fill_(True)
            self.active_increment[slot_index].fill_(1)

        started = time.perf_counter()
        copy_state()
        self.admission_enqueue_wall_s += time.perf_counter() - started
        self.kv_prefix_bytes_copied += useful_prefix_bytes
        self._epochs[slot_index] += 1
        state = DecodeSlotState(
            slot_index=slot_index,
            epoch=self._epochs[slot_index],
            ready=ready,
            token_ids=[int(ready.first_token)],
            admitted_at=time.perf_counter(),
        )
        self.slots[slot_index] = state
        # DecodeSlotState and DecodeCompletion intentionally retain request
        # metadata, but the copied prefill cache must not survive admission.
        # Large producer streams may contain hundreds of crops; retaining one
        # cache per completed crop grows HBM until the outer call returns.
        ready.release_device_state()
        return state, useful_prefix_bytes

    def step(
        self,
        decode_fn: Callable[..., torch.Tensor],
    ) -> DecodeStep:
        active_slots = tuple(slot is not None for slot in self.slots)
        slot_epochs = tuple(
            slot.epoch if slot is not None else None for slot in self.slots
        )
        slot_request_ids = tuple(
            slot.ready.request_id if slot is not None else None for slot in self.slots
        )
        cache_positions = tuple(
            (
                int(slot.ready.prompt_length) + int(slot.iterations_launched)
                if slot is not None
                else None
            )
            for slot in self.slots
        )
        generated_token_counts = tuple(
            len(slot.token_ids) if slot is not None else None for slot in self.slots
        )
        launched_at = time.perf_counter()
        for slot in self.slots:
            if slot is not None:
                slot.iterations_launched += 1
                if slot.first_decode_launched_at is None:
                    slot.first_decode_launched_at = launched_at

        def execute() -> torch.Tensor:
            decode_output = decode_fn(
                self.next_token,
                self.cache_position,
                self.rope_deltas,
                *self.cache.flat_tensors(),
            )
            return decode_output.reshape(-1, 1)

        sampled = execute()
        self.next_token = torch.where(
            self.active_mask.view(-1, 1),
            sampled,
            torch.full_like(sampled, self.eos_token_id),
        )
        self.cache_position = torch.where(
            self.active_mask,
            self.cache_position + 1,
            torch.zeros_like(self.cache_position),
        )
        return DecodeStep(
            sampled=sampled,
            active_slots=active_slots,
            slot_epochs=slot_epochs,
            slot_request_ids=slot_request_ids,
            cache_positions=cache_positions,
            generated_token_counts=generated_token_counts,
        )

    def release(self, slot_index: int) -> DecodeSlotState:
        state = self.slots[slot_index]
        if state is None:
            raise RuntimeError(f"decode slot {slot_index} is already free")
        self.slots[slot_index] = None
        self.active_mask[slot_index].fill_(False)
        self.active_increment[slot_index].zero_()
        self.next_token[slot_index].fill_(self.eos_token_id)
        self.cache_position[slot_index].zero_()
        self.rope_deltas[slot_index].zero_()
        return state

    @property
    def num_active(self) -> int:
        return sum(slot is not None for slot in self.slots)

    def free_slot_indices(self) -> list[int]:
        return [index for index, state in enumerate(self.slots) if state is None]






# Requests accepted by the runtime, individual OCR results, and run summaries.


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
    rates: dict[str, float | None]
    vision: dict[str, Any] = field(default_factory=dict)
    text_prefill: dict[str, Any] = field(default_factory=dict)
    input_fingerprints: dict[str, Any] = field(default_factory=dict)
    scheduling_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class ServingSummary:
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


# Crop metadata and the asynchronous state used by the decoding loop.


@dataclass
class DecodeRequest:
    """One crop after prefill, with the metadata needed to finish and report OCR.

    NPU state is cleared after admission copies it into a shared decode slot.
    The remaining CPU metadata accompanies the crop until its result is sent.
    """

    request_id: str
    prompt: str
    crop_size: tuple[int, int]
    skip_special_tokens: bool
    cache: LocalPaddleOCRVLStaticCache | None
    cache_lease: PrefillKVCacheLease | None
    rope_deltas: torch.Tensor | None
    cache_position: torch.Tensor | None
    first_token_tensor: torch.Tensor | None
    first_token: int
    prompt_length: int
    projected_image_tokens: int
    vision: dict[str, Any]
    text_prefill: dict[str, Any]
    cpu_timing: CpuTiming
    prefill_timing: PrefillTiming
    request_started: float
    prefill_finished: float

    def release_device_state(self) -> None:
        """Drop the per-request NPU prefix after it enters the decode arena."""

        cache_lease = self.cache_lease
        self.cache_lease = None
        self.cache = None
        self.rope_deltas = None
        self.cache_position = None
        self.first_token_tensor = None
        if cache_lease is not None:
            cache_lease.release()


@dataclass
class DecodeSlotState:
    slot_index: int
    epoch: int
    ready: DecodeRequest
    token_ids: list[int]
    admitted_at: float
    first_decode_launched_at: float | None = None
    iterations_launched: int = 0


@dataclass
class DecodeCompletion:
    ready: DecodeRequest
    token_ids: list[int]
    stop_reason: str
    slot_index: int | None
    slot_epoch: int | None
    admitted_at: float | None
    first_decode_launched_at: float | None
    completed_at: float
    iterations_launched: int
    scheduling_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class PendingTokenCopy:
    iteration: int
    active_slots: tuple[bool, ...]
    slot_epochs: tuple[int | None, ...]
    slot_request_ids: tuple[str | None, ...]
    cache_positions: tuple[int | None, ...]
    generated_token_counts: tuple[int | None, ...]
    ring_index: int | None
    done_event: Any | None
    diagnostic_compute_event: Any | None
    host_tokens: list[int] | None


@dataclass
class DecodeStep:
    sampled: torch.Tensor
    active_slots: tuple[bool, ...]
    slot_epochs: tuple[int | None, ...]
    slot_request_ids: tuple[str | None, ...]
    cache_positions: tuple[int | None, ...]
    generated_token_counts: tuple[int | None, ...]




# One request's state as it moves through the stages.


# The RequestTiming fields each stage produces, named exactly as they appear there.


@dataclass
class RequestTiming:
    """Wall-clock seconds for one request, as reported in every result's timing_s.

    Filled once, in ContinuousRecognizer._build_recognition_result, from the
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


# Reusable prefill KV storage and ownership until decode admission.


def _cache_nbytes(cache: LocalPaddleOCRVLStaticCache) -> int:
    return sum(
        int(tensor.numel()) * int(tensor.element_size())
        for tensor in cache.flat_tensors()
    )


@dataclass
class _FreeSlot:
    slot_index: int
    ready_event: Any | None = None


class PrefillKVCacheLease:
    """Exclusive ownership of one B=1 row in the prefill KV arena."""

    def __init__(
        self,
        pool: PrefillKVCachePool,
        *,
        slot_index: int,
        generation: int,
        cache: LocalPaddleOCRVLStaticCache,
    ) -> None:
        self._pool = pool
        self.slot_index = int(slot_index)
        self.generation = int(generation)
        self.cache = cache
        self._released = False

    @property
    def released(self) -> bool:
        return self._released

    def release(self) -> None:
        if self._released:
            raise RuntimeError("prefill KV cache lease was released twice")
        self._released = True
        self._pool._release(self)


class PrefillKVCachePool:
    """Fixed-capacity, zero-once arena of private request KV caches."""

    def __init__(
        self,
        cache: LocalPaddleOCRVLStaticCache,
        *,
        device: torch.device,
    ) -> None:
        if not cache.key_caches or not cache.value_caches:
            raise ValueError("prefill KV cache arena must contain K and V tensors")
        capacity = int(cache.key_caches[0].shape[0])
        if capacity <= 0:
            raise ValueError("prefill KV cache arena capacity must be positive")
        if any(
            int(tensor.shape[0]) != capacity for tensor in cache.flat_tensors()
        ):
            raise ValueError("prefill KV cache arena tensors disagree on capacity")

        self.cache = cache
        self.device = torch.device(device)
        self.capacity = capacity
        self.nbytes = _cache_nbytes(cache)
        self._free = deque(_FreeSlot(index) for index in range(capacity))
        self._active: dict[int, PrefillKVCacheLease] = {}
        self._generations = [0] * capacity
        self.acquisitions = 0
        self.reuses = 0
        self.releases = 0
        self.high_water_active = 0

    def _cache_view(self, slot_index: int) -> LocalPaddleOCRVLStaticCache:
        start = int(slot_index)
        end = start + 1
        return LocalPaddleOCRVLStaticCache(
            key_caches=tuple(
                tensor[start:end] for tensor in self.cache.key_caches
            ),
            value_caches=tuple(
                tensor[start:end] for tensor in self.cache.value_caches
            ),
            cache_length=int(self.cache.cache_length),
        )

    def acquire(self) -> PrefillKVCacheLease:
        if not self._free:
            raise RuntimeError(
                "prefill KV cache arena exhausted: "
                f"capacity={self.capacity} active={len(self._active)}"
            )
        free = self._free.popleft()
        if free.ready_event is not None:
            import torch_npu

            torch_npu.npu.current_stream().wait_event(free.ready_event)
            self.reuses += 1
        self._generations[free.slot_index] += 1
        lease = PrefillKVCacheLease(
            self,
            slot_index=free.slot_index,
            generation=self._generations[free.slot_index],
            cache=self._cache_view(free.slot_index),
        )
        if free.slot_index in self._active:
            raise RuntimeError("prefill KV cache arena handed out an active slot")
        self._active[free.slot_index] = lease
        self.acquisitions += 1
        self.high_water_active = max(self.high_water_active, len(self._active))
        return lease

    def _release(self, lease: PrefillKVCacheLease) -> None:
        active = self._active.get(lease.slot_index)
        if active is not lease:
            raise RuntimeError("prefill KV cache arena received a stale lease")
        import torch_npu

        ready_event = torch_npu.npu.current_stream().record_event()
        del self._active[lease.slot_index]
        self._free.append(_FreeSlot(lease.slot_index, ready_event))
        self.releases += 1

    def stats(self) -> dict[str, int | str]:
        return {
            "storage": "zero_once_fixed_arena",
            "capacity": self.capacity,
            "allocated_bytes": self.nbytes,
            "acquisitions": self.acquisitions,
            "reuses": self.reuses,
            "releases": self.releases,
            "active_slots": len(self._active),
            "free_slots": len(self._free),
            "high_water_active_slots": self.high_water_active,
        }


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


# Request rates.


def per_second(count: int | float, seconds: float | None) -> float | None:
    if seconds is None or seconds <= 0:
        return None
    return float(count) / float(seconds)


# Scheduling measurements: observed CPU readiness, prefill intervals and slot occupancy.
# These do not schedule work or introduce device synchronization. An already
# submitted decode may overlap a recorded host interval; these are not exclusive
# NPU stall times and must not be added to device or residency durations.


@dataclass
class _Request:
    started_at: float
    ready_at: float | None = None
    first_decode_at: float | None = None
    launched: Counter[int] = field(default_factory=Counter)
    consumed: Counter[int] = field(default_factory=Counter)
    cpu_eligible_at: float | None = None
    cpu_idle_eligible_at: float | None = None
    cpu_readiness: dict[str, Any] = field(default_factory=dict)


class RequestSchedulingMetrics:
    """One recorder per serving run; called only on the scheduler thread."""

    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.requests: dict[str, _Request] = {}
        # One span per prepared request, not per iteration. Late arrivals can
        # reach the worker after a prefill that delayed them, requiring history.
        self.prefills: list[tuple[str, float, float, str]] = []
        self.prefill_ends: list[float] = []

    def register(self, request_id: str, started_at: float) -> None:
        if request_id in self.requests:
            raise ValueError(f"duplicate scheduling metrics request: {request_id}")
        self.requests[request_id] = _Request(started_at=started_at)

    def record_prefill(
        self, request_id: str, started_at: float, finished_at: float,
        *, status: str = "ok",
    ) -> None:
        # Exclude idle blocking before this request arrived.
        started_at = max(started_at, self.requests[request_id].started_at)
        if finished_at < started_at:
            raise ValueError("prefill span ends before it starts")
        if self.prefill_ends and started_at < self.prefill_ends[-1]:
            raise ValueError("ready-source spans must not overlap")
        self.prefills.append((request_id, started_at, finished_at, status))
        self.prefill_ends.append(finished_at)
        if status == "ok":
            self.requests[request_id].ready_at = finished_at
        else:
            self.requests.pop(request_id)

    def cpu_prefill_eligible(self, request_id: str, *, block: bool) -> None:
        """FIFO head has an unreserved slot; CPU may still be unfinished.

        Once eligible, this head cannot lose its slot to a later request:
        prefill is FIFO on the scheduler thread. Read the clock only at the
        first eligibility observation or transition into an idle blocking pull.
        """
        request = self.requests[request_id]
        if request.cpu_eligible_at is None:
            request.cpu_eligible_at = time.perf_counter()
        if block and request.cpu_idle_eligible_at is None:
            request.cpu_idle_eligible_at = time.perf_counter()

    def cpu_prepared(
        self, request_id: str, *, submitted_at: float, queue_wait_s: float,
        finished_at: float, consumed_at: float,
    ) -> None:
        request = self.requests[request_id]
        eligible = request.cpu_eligible_at
        assert eligible is not None
        started = submitted_at + queue_wait_s
        blocked = max(0.0, finished_at - eligible)
        queue = max(0.0, min(started, finished_at) - eligible)
        service = max(0.0, finished_at - max(started, eligible))
        idle = (0.0 if request.cpu_idle_eligible_at is None else
                max(0.0, finished_at - request.cpu_idle_eligible_at))
        request.cpu_readiness = {
            "prefill_blocked_s": blocked,
            "blocked_cpu_queue_s": queue,
            "blocked_cpu_service_s": service,
            "scheduler_idle_blocked_s": idle,
            "ready_to_consumer_poll_s": max(0.0, consumed_at - max(eligible, finished_at)),
            "eligible_offset_s": eligible - request.started_at,
            "cpu_finished_offset_s": finished_at - request.started_at,
            "semantics": (
                "Observed FIFO-head/free-slot eligibility until CPU preparation "
                "finishes; zero if already ready. Includes CPU queue and service, "
                "excludes full-slot and earlier-request prefill waits. Idle subset "
                "means no active/ready decode request at a blocking pull, not "
                "device-traced NPU idle. Poll-boundary measurement, not a "
                "counterfactual E2E saving; do not subtract from latency."
            ),
        }

    def step(self, request_ids: Iterable[str], started_at: float) -> None:
        ids = tuple(request_ids)
        for request_id in ids:
            request = self.requests[request_id]
            if request.first_decode_at is None:
                request.first_decode_at = started_at
            request.launched[len(ids)] += 1

    def consume(self, request_ids: Iterable[str]) -> None:
        """Count output-bearing slots after stale epoch/look-ahead filtering."""
        ids = tuple(request_ids)
        for request_id in ids:
            self.requests[request_id].consumed[len(ids)] += 1

    def finish(self, request_id: str, completed_at: float) -> dict[str, Any]:
        request = self.requests.pop(request_id)
        split = request.first_decode_at
        if split is None:
            split = completed_at
        spans: list[dict[str, Any]] = []
        counts: Counter[str] = Counter()
        seconds: Counter[str] = Counter()
        first = bisect_right(self.prefill_ends, request.started_at)
        for owner, started, ended, status in self.prefills[first:]:
            if started >= completed_at:
                break
            if owner == request_id:
                continue
            for phase, left, right in (
                ("before_first_decode", request.started_at, split),
                ("during_decode", split, completed_at),
            ):
                begin, end = max(started, left), min(ended, right)
                if end <= begin:
                    continue
                counts[phase] += 1
                seconds[phase] += end - begin
                spans.append({
                    "other_request_id": owner,
                    "other_request_status": status,
                    "phase": phase,
                    "start_offset_s": begin - request.started_at,
                    "end_offset_s": end - request.started_at,
                    "host_pause_s": end - begin,
                })
        return {
            "format": "request_scheduling_metrics_v1",
            "clock": "host_monotonic",
            "scope": "request_submission_to_completion_detection",
            "pause_semantics": (
                "Other-request CPU preparation waits and prefill in the ready "
                "source. No new decode is submitted within these spans; an "
                "in-flight decode can overlap them. Not exclusive device "
                "stall time; do not add to device-stage or residency timings."
            ),
            "batch_size": self.batch_size,
            "cpu_readiness": dict(request.cpu_readiness),
            "first_decode_offset_s": (
                None if request.first_decode_at is None
                else request.first_decode_at - request.started_at
            ),
            "own_prefill_ready_offset_s": (
                None if request.ready_at is None
                else request.ready_at - request.started_at
            ),
            "own_prefill_ready_to_first_decode_s": (
                None if request.ready_at is None or request.first_decode_at is None
                else max(0.0, request.first_decode_at - request.ready_at)
            ),
            "before_first_decode_other_prefill_count": counts["before_first_decode"],
            "before_first_decode_other_prefill_host_s": seconds["before_first_decode"],
            "decode_other_prefill_count": counts["during_decode"],
            "decode_other_prefill_host_s": seconds["during_decode"],
            "other_prefill_spans": spans,
            "launched_decode_iterations_by_active_slots": dict(request.launched),
            "consumed_decode_iterations_by_useful_slots": dict(request.consumed),
            "occupancy_semantics": (
                "Launched counts include completion look-ahead. Consumed counts "
                "exclude stale slot epochs. Prefill's first token is not a decode iteration."
            ),
        }


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
