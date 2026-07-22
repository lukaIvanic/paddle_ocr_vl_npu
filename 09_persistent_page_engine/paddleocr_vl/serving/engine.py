"""Persistent PaddleOCR-VL runtime with pipelined prefill and batched decode."""

from __future__ import annotations

import time
import queue
import threading
from collections import Counter, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from tokenizers import Tokenizer

from .continuous_decode import (
    ContinuousDecodeScheduler,
    DecodeCompletion,
    DecodeArena,
    ReadyDecodeRequest,
)
from ..model.modeling import (
    LocalPaddleOCRVLForConditionalGeneration,
    _resolve_model_dir,
)
from ..model.text_decode import (
    DECODE_ATTENTION,
    DECODE_CACHE_UPDATE,
    LocalPaddleOCRVLStaticCache,
    cast_decode_linear_weights_to_nz,
)
from ..model.preprocessing import (
    apply_min_pixels_override,
    build_inputs,
    load_preprocessor_config,
    preprocess_pil_image,
)
from .runtime_defaults import (
    DECODE_BACKEND_CHOICES,
    DEFAULT_VISION_PACKING,
    DEFAULT_VISION_PACK_TARGET,
    DEFAULT_VISION_ROUTER_LOOKAHEAD,
    DEFAULT_TEXT_BACKEND,
    DEFAULT_VISION_BACKEND,
    OPTIMIZED_TEXT_BUCKETS,
    OPTIMIZED_VISION_BUCKETS,
    READY_BUFFER_BATCH_MULTIPLIER,
    VISION_PACKING_CHOICES,
)
from ..model.text_prefill import parse_text_buckets
from .types import ContinuousDecodeResult, RecognitionRequest, RecognitionResult
from utils.timing import DeviceTimeline, synchronize
from utils.timeline import TimelineRecorder
from utils.metrics import per_second
from ..model.vision_prefill import (
    PreparedVisionPrefill,
    VISION_ATTENTION_CHOICES,
    get_vision_prompt_fa_layout,
    parse_vision_buckets,
)
from .vision_router import BatchedVisionGraphRuntime, select_profiled_vision_route


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


@dataclass
class _InFlightPrefillMember:
    prepared: CpuPreparedRecognition
    cache: LocalPaddleOCRVLStaticCache
    rope_deltas: torch.Tensor
    next_cache_position: torch.Tensor
    next_token: torch.Tensor
    device_inputs: tuple[torch.Tensor, ...]
    vision: dict[str, Any]
    text_prefill: dict[str, Any]
    timing_s: dict[str, float]
    input_tokens: int
    projected_image_tokens: int


@dataclass
class _InFlightPrefillGroup:
    group_id: int
    members: list[_InFlightPrefillMember]
    device_timeline: DeviceTimeline
    h2d_ready_event: Any
    prefill_ready_event: Any
    packed_next_tokens: torch.Tensor
    prefill_started: float
    pack_route: dict[str, Any]


@dataclass
class _PreparedPrefillGroup:
    group_id: int
    members: list[tuple[CpuPreparedRecognition, float]]
    real_vision_tokens: int
    row_sizes: tuple[int, ...]
    profiled_route: dict[str, Any] | None = None


@dataclass
class _StagedPrefillGroup:
    group: _PreparedPrefillGroup
    device_timeline: DeviceTimeline
    h2d_ready_event: Any
    moved_members: list[tuple[torch.Tensor, ...]]
    timings: list[dict[str, float]]


@dataclass
class _VisionPackingRunStats:
    mode: str
    target: int
    lookahead: int
    groups: int = 0
    crops: int = 0
    packed_groups: int = 0
    singleton_groups: int = 0
    packed_real_tokens: int = 0
    packed_physical_tokens: int = 0
    eager_overflow_groups: int = 0
    group_size_histogram: Counter[int] | None = None
    graph_shape_histogram: Counter[str] | None = None
    ready_window_histogram: Counter[int] | None = None
    router_cpu_s: float = 0.0

    def __post_init__(self) -> None:
        if self.group_size_histogram is None:
            self.group_size_histogram = Counter()
        if self.graph_shape_histogram is None:
            self.graph_shape_histogram = Counter()
        if self.ready_window_histogram is None:
            self.ready_window_histogram = Counter()

    def record(self, *, crops: int, route: dict[str, Any]) -> None:
        self.groups += 1
        self.crops += int(crops)
        self.group_size_histogram[int(crops)] += 1
        sequence_length = int(
            route.get("sequence_length")
            or route.get("bucket")
            or route["physical_vision_tokens"]
        )
        shape = (
            f"b{int(route.get('batch_size', 1))}_s"
            f"{sequence_length}"
            if route.get("execution") == "compiled"
            else str(route.get("execution", "unknown"))
        )
        self.graph_shape_histogram[shape] += 1
        if route.get("visible_window_size") is not None:
            self.ready_window_histogram[int(route["visible_window_size"])] += 1
        self.router_cpu_s += float(route.get("router_cpu_s", 0.0))
        if crops > 1:
            self.packed_groups += 1
            self.packed_real_tokens += int(route["real_vision_tokens"])
            self.packed_physical_tokens += int(route["physical_vision_tokens"])
        else:
            self.singleton_groups += 1
        if route.get("execution") == "eager_overflow":
            self.eager_overflow_groups += 1

    def summary(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "target": self.target,
            "lookahead": self.lookahead,
            "groups": self.groups,
            "crops": self.crops,
            "packed_groups": self.packed_groups,
            "singleton_groups": self.singleton_groups,
            "eager_overflow_groups": self.eager_overflow_groups,
            "crops_per_group": (
                float(self.crops) / float(self.groups) if self.groups else None
            ),
            "group_size_histogram": {
                str(size): count
                for size, count in sorted(self.group_size_histogram.items())
            },
            "graph_shape_histogram": dict(sorted(self.graph_shape_histogram.items())),
            "ready_window_histogram": {
                str(size): count
                for size, count in sorted(self.ready_window_histogram.items())
            },
            "router_cpu_s": self.router_cpu_s,
            "packed_real_vision_tokens": self.packed_real_tokens,
            "packed_physical_vision_tokens": self.packed_physical_tokens,
            "packed_fill_fraction": (
                float(self.packed_real_tokens) / float(self.packed_physical_tokens)
                if self.packed_physical_tokens
                else None
            ),
        }


def _pin_memory_or_keep(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.device.type != "cpu" or tensor.is_pinned():
        return tensor
    try:
        return tensor.pin_memory()
    except RuntimeError:
        # Pageable staging is slower to submit but has identical semantics.
        return tensor


@dataclass
class PrefilledRecognition:
    request_id: str
    prompt: str
    crop_size: tuple[int, int]
    skip_special_tokens: bool
    cache: LocalPaddleOCRVLStaticCache | None
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

    def take_device_state(
        self,
    ) -> tuple[
        LocalPaddleOCRVLStaticCache,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Move the pending NPU prefix out of the long-lived result payload."""

        cache = self.cache
        rope_deltas = self.rope_deltas
        next_cache_position = self.next_cache_position
        next_token = self.next_token
        if (
            cache is None
            or rope_deltas is None
            or next_cache_position is None
            or next_token is None
        ):
            raise RuntimeError(f"prefill device state already taken for {self.request_id}")
        self.cache = None
        self.rope_deltas = None
        self.next_cache_position = None
        self.next_token = None
        return cache, rope_deltas, next_cache_position, next_token


class ContinuousRecognizer:
    """One persistent model with sequential prefill and continuous decode.

    Every real crop is prefilled independently. Vision prefill, text prefill,
    and text decode each have one model path; an execution wrapper selects
    eager or compiled execution around that path. Prefill padding is configured
    separately from compilation. A fixed decode arena keeps its tensor shapes
    stable while ready KV prefixes replace finished requests between steps.
    """

    @torch.inference_mode()
    def __init__(
        self,
        *,
        model: str,
        dtype: str,
        decode_backend: str,
        batch_size: int,
        cache_length: int,
        max_new_tokens: int,
        torchair_cache_dir: Path,
        vision_backend: str = DEFAULT_VISION_BACKEND,
        vision_attention: str = "manual",
        vision_buckets: str | Iterable[int] = OPTIMIZED_VISION_BUCKETS,
        vision_torchair_cache_dir: Path | None = None,
        vision_padding: str = "auto",
        vision_packing: str = DEFAULT_VISION_PACKING,
        vision_pack_target: int = DEFAULT_VISION_PACK_TARGET,
        vision_router_lookahead: int = DEFAULT_VISION_ROUTER_LOOKAHEAD,
        vision_batched_cache_dir: Path | None = None,
        text_backend: str = DEFAULT_TEXT_BACKEND,
        text_buckets: str | Iterable[int] = OPTIMIZED_TEXT_BUCKETS,
        text_torchair_cache_dir: Path | None = None,
        text_padding: str = "auto",
        preprocessor_min_pixels: int | None = None,
        timeline: TimelineRecorder | None = None,
    ):
        # TorchAir guards tensor dispatch-key sets. Build and warm every graph
        # under the same inference-mode contract used by run(),
        # otherwise the first real request invalidates the persistent cache
        # and recompiles the vision, text-prefill, and decode boundaries.
        runtime_started = time.perf_counter()
        import torch_npu

        self.model_dir = _resolve_model_dir(model)
        self.device = torch.device("npu:0")
        if not torch.npu.is_available():
            raise RuntimeError("Experiment 09 requires an available NPU")
        if dtype in {"fp16", "float16"}:
            self.dtype = torch.float16
        elif dtype in {"bf16", "bfloat16"}:
            self.dtype = torch.bfloat16
        else:
            raise ValueError(f"unsupported dtype: {dtype}")
        torch.npu.set_compile_mode(jit_compile=False)
        self.decode_backend = decode_backend
        self.timeline = timeline
        self.batch_size = int(batch_size)
        self.cache_length = int(cache_length)
        self.max_new_tokens = int(max_new_tokens)
        self.vision_backend = str(vision_backend)
        self.vision_attention = str(vision_attention)
        if self.vision_attention not in VISION_ATTENTION_CHOICES:
            raise ValueError(
                "vision attention must be one of "
                f"{VISION_ATTENTION_CHOICES}, got {vision_attention!r}"
            )
        self.vision_buckets = parse_vision_buckets(vision_buckets)
        self.vision_padding = str(vision_padding)
        self.vision_packing = str(vision_packing)
        self.vision_pack_target = int(vision_pack_target)
        self.vision_router_lookahead = int(vision_router_lookahead)
        if self.vision_packing not in VISION_PACKING_CHOICES:
            raise ValueError(
                "vision_packing must be one of "
                f"{VISION_PACKING_CHOICES}, got {vision_packing!r}"
            )
        if self.vision_pack_target not in self.vision_buckets:
            raise ValueError(
                "vision_pack_target must be one of the configured vision buckets: "
                f"target={self.vision_pack_target} buckets={self.vision_buckets}"
            )
        if self.vision_router_lookahead <= 0:
            raise ValueError("vision_router_lookahead must be positive")
        if self.vision_packing == "profile_guided":
            if self.vision_backend != "torchair":
                raise ValueError("profile_guided vision routing requires TorchAir")
            if self.vision_attention != "prompt_flash_attention":
                raise ValueError(
                    "profile_guided vision routing requires prompt_flash_attention"
                )
            if vision_batched_cache_dir is None:
                raise ValueError(
                    "profile_guided vision routing requires vision_batched_cache_dir"
                )
        self.text_backend = str(text_backend)
        self.text_buckets = parse_text_buckets(text_buckets)
        self.text_padding = str(text_padding)
        if self.decode_backend not in DECODE_BACKEND_CHOICES:
            raise ValueError(
                f"decode_backend must be one of {DECODE_BACKEND_CHOICES}, "
                f"got {self.decode_backend!r}"
            )
        if self.batch_size <= 0 or self.batch_size & (self.batch_size - 1):
            raise ValueError("batch_size must be a positive power of two")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.cache_length <= self.max_new_tokens:
            raise ValueError("cache_length must leave room for both prompt and generated tokens")

        model_preprocessor_config = load_preprocessor_config(self.model_dir)
        self.model_preprocessor_min_pixels = int(model_preprocessor_config["min_pixels"])
        self.preprocessor_min_pixels_override = (
            None if preprocessor_min_pixels is None else int(preprocessor_min_pixels)
        )
        self.preprocessor_config = apply_min_pixels_override(
            model_preprocessor_config,
            self.preprocessor_min_pixels_override,
        )
        # Encoding runs continuously on the CPU preparation thread while this
        # tokenizer remains owned by the NPU thread for result decoding.
        self.preprocessing_tokenizer = Tokenizer.from_file(
            str(self.model_dir / "tokenizer.json")
        )
        self.tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        frontend_setup_s = time.perf_counter() - runtime_started

        synchronize(self.device)
        started = time.perf_counter()
        self.model = LocalPaddleOCRVLForConditionalGeneration.from_pretrained(
            self.model_dir,
            dtype=self.dtype,
            device=self.device,
        )
        synchronize(self.device)
        model_load_s = time.perf_counter() - started

        synchronize(self.device)
        started = time.perf_counter()
        self.weight_format = cast_decode_linear_weights_to_nz(self.model)
        synchronize(self.device)
        weight_format_s = time.perf_counter() - started

        self.stages = self.model.make_inference_stages(
            vision_backend=self.vision_backend,
            vision_attention=self.vision_attention,
            vision_buckets=self.vision_buckets,
            vision_cache_root=(
                vision_torchair_cache_dir
                if vision_torchair_cache_dir is not None
                else torchair_cache_dir.parent / f"{torchair_cache_dir.name}_vision"
            ),
            vision_padding=self.vision_padding,
            text_backend=self.text_backend,
            text_buckets=self.text_buckets,
            text_cache_root=(
                text_torchair_cache_dir
                if text_torchair_cache_dir is not None
                else torchair_cache_dir.parent / f"{torchair_cache_dir.name}_text"
            ),
            text_padding=self.text_padding,
            decode_backend=self.decode_backend,
            decode_cache_root=torchair_cache_dir,
            batch_size=self.batch_size,
            cache_length=self.cache_length,
            device=self.device,
            dtype=self.dtype,
            model_dir=self.model_dir,
            linear_weight_format=str(self.weight_format["effective_mode"]),
        )
        self.vision_prefill = self.stages.vision_prefill
        self.text_prefill = self.stages.text_prefill
        self.text_decode = self.stages.text_decode
        self.decode_fn = self.text_decode.fn
        self.batched_vision = (
            BatchedVisionGraphRuntime(
                self.model,
                cache_root=vision_batched_cache_dir,
                model_dir=self.model_dir,
                dtype=self.dtype,
                device=self.device,
            )
            if self.vision_packing == "profile_guided"
            else None
        )
        self.prefill_transfer_stream = torch_npu.npu.Stream(device=self.device)
        # Keep one complete decode cohort in CPU preparation without coupling
        # correctness to the relative speed of CPU and NPU stages. B=1 still
        # needs one item being consumed and one item prepared in the background.
        self.cpu_preprocess_max_pending = max(
            2,
            self.batch_size,
            (
                self.vision_router_lookahead
                if self.vision_packing == "profile_guided"
                else 0
            ),
        )
        self.prefill_host_tokens = torch.empty(
            (self.cpu_preprocess_max_pending + 1,),
            dtype=torch.int64,
            pin_memory=True,
        )
        self._vision_pack_sequence = 0
        self._vision_packing_stats = _VisionPackingRunStats(
            self.vision_packing,
            self.vision_pack_target,
            self.vision_router_lookahead,
        )

        started = time.perf_counter()
        self.decode_arena = DecodeArena(
            cache=self.text_decode.warm_cache,
            device=self.device,
            batch_size=self.batch_size,
            eos_token_id=int(self.model.config.eos_token_id),
            timeline=self.timeline,
        )
        self.decode_scheduler = ContinuousDecodeScheduler(
            arena=self.decode_arena,
            decode_fn=self.decode_fn,
            max_new_tokens=self.max_new_tokens,
            timeline=self.timeline,
        )
        decode_control_setup_s = time.perf_counter() - started

        self.setup_timing_s = {
            "recognizer_frontend_setup": float(frontend_setup_s),
            "recognizer_model_load": float(model_load_s),
            "decode_weight_format": float(weight_format_s),
            **self.stages.setup_timing_s,
            "vision_router_setup": (
                float(self.batched_vision.metadata["setup_s"])
                if self.batched_vision is not None
                else 0.0
            ),
            "decode_control_setup": float(decode_control_setup_s),
            "recognizer_runtime_total": float(time.perf_counter() - runtime_started),
        }

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

        self._vision_pack_sequence = 0
        self._vision_packing_stats = _VisionPackingRunStats(
            self.vision_packing,
            self.vision_pack_target,
            self.vision_router_lookahead,
        )

        def ready_from(state: PrefilledRecognition) -> ReadyDecodeRequest:
            cache, rope_deltas, cache_position, first_token_tensor = (
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
            )

        def ready_stream() -> Iterable[ReadyDecodeRequest]:
            current_staged: _StagedPrefillGroup | None = None
            drained_normally = False
            try:
                if self.vision_packing == "off":
                    groups = self._iter_single_prefill_groups(requests)
                elif self.vision_packing == "greedy":
                    groups = self._iter_packed_prefill_groups(requests)
                else:
                    groups = self._iter_profiled_prefill_groups(requests)
                group_source = iter(groups)
                try:
                    first_group = next(group_source)
                except StopIteration:
                    drained_normally = True
                    return
                current_staged = self._stage_prefill_group(first_group)
                while current_staged is not None:
                    try:
                        next_group = next(group_source)
                    except StopIteration:
                        next_staged = None
                    else:
                        # Submit G+1 before invoking G's packed graph. TorchAir
                        # keeps the host occupied for much of that graph call,
                        # so staging after it returns is too late to overlap.
                        next_staged = self._stage_prefill_group(next_group)

                    final = self._enqueue_staged_prefill_group(current_staged)
                    current_staged = None
                    for state in self._finalize_prefill_group(final):
                        yield ready_from(state)
                    if next_staged is None:
                        break
                    current_staged = next_staged
                drained_normally = True
            finally:
                if drained_normally and current_staged is not None:
                    raise RuntimeError(
                        "ready stream drained with an unused staged prefill"
                    )

        def handle_completion(completion: DecodeCompletion) -> None:
            result = self._result_from_completion(
                completion,
                schedule_id=schedule_id,
            )
            emit_result(result)

        decoded = self.decode_scheduler.run_stream(
            ready_stream(),
            on_completion=handle_completion,
            ready_buffer_capacity=READY_BUFFER_BATCH_MULTIPLIER * self.batch_size,
            ready_buffer_low_watermark=self.batch_size,
        )
        decode_wall_s = decoded.timing_s["continuous_decode_wall"]

        schedule_result = ContinuousDecodeResult(
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
            vision_packing=self._vision_packing_stats.summary(),
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
        return schedule_result

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
                not source_exhausted
                and len(pending) < self.cpu_preprocess_max_pending
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

    def _prepared_group(
        self,
        members: list[tuple[CpuPreparedRecognition, float]],
        *,
        row_sizes: tuple[int, ...] | None = None,
        profiled_route: dict[str, Any] | None = None,
    ) -> _PreparedPrefillGroup:
        if not members:
            raise ValueError("prefill group must contain at least one crop")
        if row_sizes is None:
            row_sizes = (len(members),)
        if sum(row_sizes) != len(members) or any(size < 0 for size in row_sizes):
            raise ValueError(
                f"invalid vision row sizes {row_sizes} for {len(members)} crops"
            )
        self._vision_pack_sequence += 1
        return _PreparedPrefillGroup(
            group_id=self._vision_pack_sequence,
            members=members,
            real_vision_tokens=sum(
                int(prepared.pixel_values.shape[0]) for prepared, _wait_s in members
            ),
            row_sizes=tuple(int(size) for size in row_sizes),
            profiled_route=profiled_route,
        )

    def _iter_single_prefill_groups(
        self,
        requests: Iterable[RecognitionRequest],
    ) -> Iterable[_PreparedPrefillGroup]:
        for item in self._iter_cpu_prepared(requests):
            yield self._prepared_group([item])

    def _iter_packed_prefill_groups(
        self,
        requests: Iterable[RecognitionRequest],
    ) -> Iterable[_PreparedPrefillGroup]:
        """Form impatient arrival-order packs from currently prepared crops."""

        prepared_queue: queue.Queue[object] = queue.Queue(
            maxsize=self.cpu_preprocess_max_pending
        )
        sentinel = object()
        stop = threading.Event()
        errors: list[BaseException] = []

        def put(item: object) -> bool:
            while not stop.is_set():
                try:
                    prepared_queue.put(item, timeout=0.1)
                    return True
                except queue.Full:
                    continue
            return False

        def produce() -> None:
            try:
                for item in self._iter_cpu_prepared(requests):
                    if not put(item):
                        return
            except BaseException as exc:
                errors.append(exc)
            finally:
                put(sentinel)

        producer = threading.Thread(
            target=produce,
            name="paddleocr-vl-packed-prefill-source",
            daemon=True,
        )
        producer.start()
        carry: tuple[CpuPreparedRecognition, float] | None = None
        exhausted = False
        try:
            while not exhausted or carry is not None:
                form_started = time.perf_counter()
                if carry is None:
                    item = prepared_queue.get()
                    if item is sentinel:
                        exhausted = True
                        if errors:
                            raise errors[0]
                        break
                    if not isinstance(item, tuple) or len(item) != 2:
                        raise TypeError(f"unexpected prepared crop item: {type(item)!r}")
                    first = item
                else:
                    first = carry
                    carry = None

                members = [first]
                total = int(first[0].pixel_values.shape[0])
                if total <= self.vision_pack_target:
                    while True:
                        try:
                            item = prepared_queue.get_nowait()
                        except queue.Empty:
                            break
                        if item is sentinel:
                            exhausted = True
                            break
                        if not isinstance(item, tuple) or len(item) != 2:
                            raise TypeError(
                                f"unexpected prepared crop item: {type(item)!r}"
                            )
                        candidate = item
                        candidate_tokens = int(candidate[0].pixel_values.shape[0])
                        if total + candidate_tokens > self.vision_pack_target:
                            carry = candidate
                            break
                        members.append(candidate)
                        total += candidate_tokens

                group = self._prepared_group(members)
                if self.timeline is not None:
                    self.timeline.record_span_seconds(
                        "Vision prefill",
                        "Form vision pack",
                        form_started,
                        time.perf_counter(),
                        flow_id=members[0][0].request_id,
                        flow_ids=[member.request_id for member, _wait_s in members],
                        args={
                            "group_id": group.group_id,
                            "crops": len(members),
                            "real_tokens": group.real_vision_tokens,
                            "target": self.vision_pack_target,
                        },
                    )
                yield group
                if exhausted and carry is None:
                    if errors:
                        raise errors[0]
                    break
        finally:
            stop.set()
            producer.join(timeout=30.0)
            if producer.is_alive():
                raise RuntimeError("packed prefill source did not stop within 30 seconds")

    def _iter_profiled_prefill_groups(
        self,
        requests: Iterable[RecognitionRequest],
    ) -> Iterable[_PreparedPrefillGroup]:
        """Route up to the currently ready lookahead without waiting to fill it."""

        prepared_queue: queue.Queue[object] = queue.Queue(
            maxsize=self.cpu_preprocess_max_pending
        )
        sentinel = object()
        stop = threading.Event()
        errors: list[BaseException] = []

        def put(item: object) -> bool:
            while not stop.is_set():
                try:
                    prepared_queue.put(item, timeout=0.1)
                    return True
                except queue.Full:
                    continue
            return False

        def produce() -> None:
            try:
                for item in self._iter_cpu_prepared(requests):
                    if not put(item):
                        return
            except BaseException as exc:
                errors.append(exc)
            finally:
                put(sentinel)

        producer = threading.Thread(
            target=produce,
            name="paddleocr-vl-profiled-prefill-source",
            daemon=True,
        )
        producer.start()
        visible: list[tuple[CpuPreparedRecognition, float]] = []
        exhausted = False
        try:
            while visible or not exhausted:
                if not visible:
                    item = prepared_queue.get()
                    if item is sentinel:
                        exhausted = True
                        if errors:
                            raise errors[0]
                        break
                    if not isinstance(item, tuple) or len(item) != 2:
                        raise TypeError(f"unexpected prepared crop item: {type(item)!r}")
                    visible.append(item)

                while (
                    not exhausted
                    and len(visible) < self.vision_router_lookahead
                ):
                    try:
                        item = prepared_queue.get_nowait()
                    except queue.Empty:
                        break
                    if item is sentinel:
                        exhausted = True
                        break
                    if not isinstance(item, tuple) or len(item) != 2:
                        raise TypeError(f"unexpected prepared crop item: {type(item)!r}")
                    visible.append(item)

                form_started = time.perf_counter()
                visible_window_size = len(visible)
                route = select_profiled_vision_route(
                    [int(item[0].pixel_values.shape[0]) for item in visible]
                )
                selected_indices = [index for row in route.rows for index in row]
                if not selected_indices or selected_indices[0] != 0:
                    raise RuntimeError("profiled vision route did not select the oldest crop")
                if len(set(selected_indices)) != len(selected_indices):
                    raise RuntimeError("profiled vision route selected a crop twice")
                selected = [visible[index] for index in selected_indices]
                selected_set = set(selected_indices)
                visible = [
                    item for index, item in enumerate(visible) if index not in selected_set
                ]
                form_finished = time.perf_counter()
                router_cpu_s = form_finished - form_started
                route_dict = {
                    "execution": route.execution,
                    "real_vision_tokens": route.real_tokens,
                    "physical_vision_tokens": route.physical_tokens,
                    "padding_vision_tokens": route.physical_tokens - route.real_tokens,
                    "useful_token_fraction": (
                        float(route.real_tokens) / float(route.physical_tokens)
                    ),
                    "bucket": (
                        route.sequence_length if route.batch_size == 1 else None
                    ),
                    "batch_size": route.batch_size,
                    "sequence_length": route.sequence_length,
                    "row_sizes": [len(row) for row in route.rows],
                    "profiled_ms": route.profiled_ms,
                    "visible_window_size": visible_window_size,
                    "router_cpu_s": router_cpu_s,
                }
                group = self._prepared_group(
                    selected,
                    row_sizes=tuple(len(row) for row in route.rows),
                    profiled_route=route_dict,
                )
                if group.real_vision_tokens != route.real_tokens:
                    raise RuntimeError(
                        "profiled route token accounting mismatch: "
                        f"group={group.real_vision_tokens} route={route.real_tokens}"
                    )
                if self.timeline is not None:
                    self.timeline.record_span_seconds(
                        "Vision prefill",
                        "Route prepared vision crops",
                        form_started,
                        form_finished,
                        flow_id=selected[0][0].request_id,
                        flow_ids=[member.request_id for member, _wait_s in selected],
                        args={
                            "group_id": group.group_id,
                            "visible_crops": visible_window_size,
                            "selected_crops": len(selected),
                            "batch_size": route.batch_size,
                            "sequence_length": route.sequence_length,
                            "real_tokens": route.real_tokens,
                            "physical_tokens": route.physical_tokens,
                        },
                    )
                yield group
                if exhausted and not visible:
                    if errors:
                        raise errors[0]
                    break
        finally:
            stop.set()
            producer.join(timeout=30.0)
            if producer.is_alive():
                raise RuntimeError(
                    "profiled prefill source did not stop within 30 seconds"
                )

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
        )

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
        crop_size = tuple(int(value) for value in request.crop.size)
        preprocessor_config = self.preprocessor_config
        if request.min_pixels is not None or request.max_pixels is not None:
            if request.min_pixels is None or request.max_pixels is None:
                raise ValueError(
                    "request-specific min_pixels and max_pixels must be provided together"
                )
            if request.min_pixels <= 0 or request.max_pixels < request.min_pixels:
                raise ValueError(
                    "invalid request-specific pixel profile: "
                    f"min={request.min_pixels} max={request.max_pixels}"
                )
            preprocessor_config = dict(preprocessor_config)
            preprocessor_config["min_pixels"] = int(request.min_pixels)
            preprocessor_config["max_pixels"] = int(request.max_pixels)

        started = time.perf_counter()
        pixel_values, image_grid_thw = preprocess_pil_image(
            request.crop,
            preprocessor_config,
        )
        input_ids, attention_mask = build_inputs(
            self.preprocessing_tokenizer,
            image_grid_thw,
            request.prompt,
            merge_size=int(preprocessor_config["merge_size"]),
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

        min_cache_length = int(input_ids.shape[1]) + max(0, self.max_new_tokens - 1)
        if self.cache_length < min_cache_length:
            raise ValueError(
                f"request {request.request_id} needs cache_length>={min_cache_length}, "
                f"configured cache_length={self.cache_length}"
            )
        image_token_count = int(
            (input_ids == self.model.config.image_token_id).sum().item()
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
        )

    @staticmethod
    def _group_stage_key(member_index: int, stage: str) -> str:
        return f"member:{int(member_index)}:{stage}"

    @torch.inference_mode()
    def _stage_prefill_group(
        self,
        group: _PreparedPrefillGroup,
    ) -> _StagedPrefillGroup:
        import torch_npu

        if not group.members:
            raise ValueError("cannot stage an empty prefill group")
        device_timeline = DeviceTimeline(self.device)
        submit_started = time.perf_counter()
        moved_members: list[tuple[torch.Tensor, ...]] = []
        timings: list[dict[str, float]] = []

        with torch_npu.npu.stream(self.prefill_transfer_stream):
            for index, (prepared, consumer_wait_s) in enumerate(group.members):
                timing = dict(prepared.timing_s)
                timing["cpu_preprocess_background_consumer_wait"] = float(
                    consumer_wait_s
                )
                ready_consumed_at = time.perf_counter()
                timing["cpu_preprocess_background_ready_wait"] = max(
                    0.0,
                    ready_consumed_at - prepared.preparation_finished,
                )
                timings.append(timing)
                if self.timeline is not None:
                    self.timeline.record_span_seconds(
                        "CPU / queue wait",
                        "Prepared crop waiting for NPU prefill",
                        prepared.preparation_finished,
                        ready_consumed_at,
                        flow_id=prepared.request_id,
                        event_type="wait",
                        track="queue",
                        lane="prefill-ready",
                    )

                def move_inputs(
                    prepared: CpuPreparedRecognition = prepared,
                ) -> tuple[torch.Tensor, ...]:
                    return (
                        prepared.input_ids.to(self.device, non_blocking=True),
                        prepared.attention_mask.to(self.device, non_blocking=True),
                        prepared.pixel_values.to(
                            device=self.device,
                            dtype=self.model.visual.dtype,
                            non_blocking=True,
                        ),
                        prepared.position_ids.to(self.device, non_blocking=True),
                        prepared.rope_deltas.to(self.device, non_blocking=True),
                    )

                moved_members.append(
                    device_timeline.measure(
                        self._group_stage_key(index, "recognition_inputs_h2d"),
                        move_inputs,
                    )
                )
            h2d_ready_event = self.prefill_transfer_stream.record_event()
        submit_finished = time.perf_counter()
        for timing in timings:
            timing["prefill_h2d_submit_host"] = submit_finished - submit_started
        if self.timeline is not None:
            self.timeline.record_span_seconds(
                "H2D / D2H transfer",
                "Submit next prefill group H2D",
                submit_started,
                submit_finished,
                flow_id=group.members[0][0].request_id,
                flow_ids=[
                    prepared.request_id for prepared, _wait_s in group.members
                ],
                event_type="io",
                track="host",
                lane="prefill-submit",
                args={"group_id": group.group_id, "crops": len(group.members)},
            )
        return _StagedPrefillGroup(
            group=group,
            device_timeline=device_timeline,
            h2d_ready_event=h2d_ready_event,
            moved_members=moved_members,
            timings=timings,
        )

    @torch.inference_mode()
    def _enqueue_staged_prefill_group(
        self,
        staged: _StagedPrefillGroup,
    ) -> _InFlightPrefillGroup:
        import torch_npu

        group = staged.group
        device_timeline = staged.device_timeline
        h2d_ready_event = staged.h2d_ready_event
        moved_members = staged.moved_members
        timings = staged.timings
        enqueue_started = time.perf_counter()

        compute_stream = torch_npu.npu.current_stream()
        compute_stream.wait_event(h2d_ready_event)
        prefill_started = time.perf_counter()
        vision_model = self.model.visual.vision_model
        hidden_states: list[torch.Tensor] = []
        for index, ((prepared, _consumer_wait_s), moved) in enumerate(
            zip(group.members, moved_members)
        ):
            pixel_values_device = moved[2]
            hidden_states.append(
                device_timeline.measure(
                    self._group_stage_key(index, "vision_embeddings"),
                    lambda prepared=prepared, pixels=pixel_values_device: (
                        vision_model.embeddings(
                            pixels.unsqueeze(0),
                            image_grid_thw=prepared.image_grid_thw,
                        )
                    ),
                )
            )

        real_lengths = [int(hidden.shape[0]) for hidden in hidden_states]
        real_vision_tokens = sum(real_lengths)
        pack_route = (
            dict(group.profiled_route)
            if group.profiled_route is not None
            else self.vision_prefill.route(real_vision_tokens)
        )
        batch_size = int(pack_route.get("batch_size", 1))
        sequence_length = int(
            pack_route.get("sequence_length", pack_route["physical_vision_tokens"])
        )
        if int(pack_route["real_vision_tokens"]) != real_vision_tokens:
            raise RuntimeError(
                "vision route real-token mismatch: "
                f"route={pack_route['real_vision_tokens']} actual={real_vision_tokens}"
            )
        if batch_size == 1 and len(group.members) == 1:
            prepared_vision = device_timeline.measure(
                "group:vision_prefill_input_prep",
                lambda: self.vision_prefill.prepare(
                    hidden_states[0],
                    group.members[0][0].image_grid_thw,
                    route=pack_route,
                ),
            )
            image_features = [
                device_timeline.measure(
                    "group:vision_prefill",
                    lambda: self.vision_prefill.run_prepared(prepared_vision),
                )
            ]
        elif batch_size == 1:
            prepared_packed = device_timeline.measure(
                "group:vision_prefill_input_prep",
                lambda: self.vision_prefill.prepare_packed(
                    hidden_states,
                    [prepared.image_grid_thw for prepared, _wait_s in group.members],
                    route=pack_route,
                ),
            )
            packed_features = device_timeline.measure(
                "group:vision_prefill",
                lambda: self.vision_prefill.run_prepared(
                    prepared_packed.prepared
                ),
            )
            image_features = list(
                torch.split(
                    packed_features,
                    prepared_packed.segment_lengths,
                    dim=0,
                )
            )
        else:
            if self.batched_vision is None:
                raise RuntimeError("batched vision route selected without a runtime")

            def prepare_rows() -> tuple[
                list[PreparedVisionPrefill],
                list[tuple[int, ...]],
            ]:
                rows: list[PreparedVisionPrefill] = []
                row_segments: list[tuple[int, ...]] = []
                offset = 0
                row_route = {
                    "execution": "compiled",
                    "physical_vision_tokens": sequence_length,
                }
                for row_size in group.row_sizes:
                    end = offset + row_size
                    if row_size:
                        packed = self.vision_prefill.prepare_packed(
                            hidden_states[offset:end],
                            [
                                prepared.image_grid_thw
                                for prepared, _wait_s in group.members[offset:end]
                            ],
                            route=row_route,
                        )
                        rows.append(packed.prepared)
                        row_segments.append(packed.segment_lengths)
                    offset = end
                if not rows:
                    raise RuntimeError("batched vision route had no non-empty rows")
                while len(rows) < batch_size:
                    template = rows[0]
                    rows.append(
                        PreparedVisionPrefill(
                            prefix_hidden_states=torch.zeros_like(
                                template.prefix_hidden_states
                            ),
                            rope_cos=torch.ones_like(template.rope_cos),
                            rope_sin=torch.zeros_like(template.rope_sin),
                            attention_mask=torch.zeros_like(
                                template.attention_mask
                            ),
                            real_seq_len=0,
                            physical_seq_len=sequence_length,
                            execution="compiled",
                        )
                    )
                    row_segments.append(())
                if len(rows) != batch_size or offset != len(group.members):
                    raise RuntimeError("batched vision row accounting mismatch")
                return rows, row_segments

            prepared_rows, row_segments = device_timeline.measure(
                "group:vision_prefill_input_prep",
                prepare_rows,
            )
            batched_features = device_timeline.measure(
                "group:vision_prefill",
                lambda: self.batched_vision.run(
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                    prepared_rows=prepared_rows,
                ),
            )
            image_features = []
            for row_index, segment_lengths in enumerate(row_segments):
                if not segment_lengths:
                    continue
                row_real_tokens = sum(segment_lengths)
                image_features.extend(
                    torch.split(
                        batched_features[row_index, :row_real_tokens].contiguous(),
                        segment_lengths,
                        dim=0,
                    )
                )
            if len(image_features) != len(group.members):
                raise RuntimeError(
                    "batched vision output count mismatch: "
                    f"features={len(image_features)} members={len(group.members)}"
                )

        self._vision_packing_stats.record(
            crops=len(group.members),
            route=pack_route,
        )
        group_padding = int(pack_route["physical_vision_tokens"]) - real_vision_tokens
        members: list[_InFlightPrefillMember] = []
        next_tokens: list[torch.Tensor] = []
        for index, (
            ((prepared, _consumer_wait_s), moved, image_feature, real_length, timing)
        ) in enumerate(
            zip(
                group.members,
                moved_members,
                image_features,
                real_lengths,
                timings,
            )
        ):
            (
                input_ids_device,
                attention_mask_device,
                _pixel_values_device,
                position_ids,
                rope_deltas,
            ) = moved
            next_cache_position = torch.full(
                (1,),
                int(input_ids_device.shape[1]),
                device=self.device,
                dtype=torch.int64,
            )
            image_embeds = device_timeline.measure(
                self._group_stage_key(index, "adaptive_mlp_projector"),
                lambda feature=image_feature, prepared=prepared: self.model.mlp_AR(
                    feature,
                    prepared.image_grid_thw,
                ),
            )
            inputs_embeds = device_timeline.measure(
                self._group_stage_key(index, "text_token_embedding"),
                lambda input_ids_device=input_ids_device: self.model.model.embed_tokens(
                    input_ids_device
                ),
            )

            def scatter_image_embeds(
                image_embeds: torch.Tensor = image_embeds,
                inputs_embeds: torch.Tensor = inputs_embeds,
                input_ids_device: torch.Tensor = input_ids_device,
                prepared: CpuPreparedRecognition = prepared,
            ) -> torch.Tensor:
                projected = image_embeds.to(
                    device=inputs_embeds.device,
                    dtype=inputs_embeds.dtype,
                )
                image_mask = (
                    (input_ids_device == self.model.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                )
                expected_values = prepared.image_token_count * int(
                    inputs_embeds.shape[-1]
                )
                if expected_values != projected.numel():
                    raise ValueError(
                        "image features and image tokens do not match: "
                        f"tokens={prepared.image_token_count} "
                        f"features={int(projected.shape[0])}"
                    )
                return inputs_embeds.masked_scatter(image_mask, projected)

            inputs_embeds = device_timeline.measure(
                self._group_stage_key(index, "image_embed_scatter"),
                scatter_image_embeds,
            )
            cache = device_timeline.measure(
                self._group_stage_key(index, "static_cache_alloc"),
                lambda inputs_embeds=inputs_embeds: self.model.allocate_static_cache(
                    batch_size=1,
                    cache_length=self.cache_length,
                    device=self.device,
                    dtype=inputs_embeds.dtype,
                    init_mode="empty",
                ),
            )
            text_route = self.text_prefill.route(int(inputs_embeds.shape[1]))
            prepared_text = device_timeline.measure(
                self._group_stage_key(index, "text_prefill_input_prep"),
                lambda inputs_embeds=inputs_embeds,
                attention_mask_device=attention_mask_device,
                position_ids=position_ids,
                text_route=text_route: self.text_prefill.prepare(
                    inputs_embeds,
                    attention_mask_device,
                    position_ids,
                    route=text_route,
                ),
            )
            last_hidden_state = device_timeline.measure(
                self._group_stage_key(index, "text_prefill"),
                lambda prepared_text=prepared_text, cache=cache: (
                    self.text_prefill.run_prepared(prepared_text, cache)
                ),
            )
            logits = device_timeline.measure(
                self._group_stage_key(index, "prefill_lm_head"),
                lambda last_hidden_state=last_hidden_state: self.model.lm_head(
                    last_hidden_state
                ),
            )
            next_token = device_timeline.measure(
                self._group_stage_key(index, "prefill_argmax"),
                lambda logits=logits: torch.argmax(
                    logits[:, -1, :].float(),
                    dim=-1,
                    keepdim=True,
                ),
            )
            next_tokens.append(next_token)
            member_padding = group_padding if index == len(group.members) - 1 else 0
            vision_route = {
                **pack_route,
                "real_vision_tokens": real_length,
                "physical_vision_tokens": real_length + member_padding,
                "padding_vision_tokens": member_padding,
                "useful_token_fraction": (
                    float(real_length) / float(real_length + member_padding)
                ),
                "packing": "packed" if len(group.members) > 1 else "single",
                "pack_group_id": group.group_id,
                "pack_crops": len(group.members),
                "pack_real_vision_tokens": real_vision_tokens,
                "pack_physical_vision_tokens": int(
                    pack_route["physical_vision_tokens"]
                ),
                "pack_batch_size": batch_size,
                "pack_sequence_length": sequence_length,
                "pack_row_sizes": list(group.row_sizes),
                "router_visible_crops": pack_route.get("visible_window_size"),
            }
            members.append(
                _InFlightPrefillMember(
                    prepared=prepared,
                    cache=cache,
                    rope_deltas=rope_deltas,
                    next_cache_position=next_cache_position,
                    next_token=next_token,
                    device_inputs=moved,
                    vision=vision_route,
                    text_prefill=text_route,
                    timing_s=timing,
                    input_tokens=int(prepared.input_ids.shape[1]),
                    projected_image_tokens=int(image_embeds.shape[0]),
                )
            )

        packed_next_tokens = torch.cat(
            [token.detach().reshape(-1) for token in next_tokens],
            dim=0,
        ).contiguous()
        prefill_ready_event = torch_npu.npu.current_stream().record_event()
        enqueue_finished = time.perf_counter()
        for member in members:
            member.timing_s["prefill_enqueue_host"] = (
                enqueue_finished - enqueue_started
            )
        if self.timeline is not None:
            self.timeline.record_span_seconds(
                "Vision prefill",
                "Enqueue prefill chain",
                enqueue_started,
                enqueue_finished,
                flow_id=members[0].prepared.request_id,
                flow_ids=[member.prepared.request_id for member in members],
                args={
                    "group_id": group.group_id,
                    "crops": len(members),
                    "real_tokens": real_vision_tokens,
                    "physical_tokens": int(pack_route["physical_vision_tokens"]),
                },
            )
        return _InFlightPrefillGroup(
            group_id=group.group_id,
            members=members,
            device_timeline=device_timeline,
            h2d_ready_event=h2d_ready_event,
            prefill_ready_event=prefill_ready_event,
            packed_next_tokens=packed_next_tokens,
            prefill_started=prefill_started,
            pack_route=pack_route,
        )

    @torch.inference_mode()
    def _finalize_prefill_group(
        self,
        inflight: _InFlightPrefillGroup,
    ) -> list[PrefilledRecognition]:
        import torch_npu

        resolve_started = time.perf_counter()
        device_spans = inflight.device_timeline.resolve_spans()
        started = time.perf_counter()
        count = len(inflight.members)
        with torch_npu.npu.stream(self.prefill_transfer_stream):
            self.prefill_transfer_stream.wait_event(inflight.prefill_ready_event)
            self.prefill_host_tokens[:count].copy_(
                inflight.packed_next_tokens,
                non_blocking=True,
            )
            first_tokens_ready = self.prefill_transfer_stream.record_event()
        first_tokens_ready.synchronize()
        first_tokens = [int(value) for value in self.prefill_host_tokens[:count].tolist()]
        first_token_d2h_s = time.perf_counter() - started
        resolve_finished = time.perf_counter()
        request_ids = [member.prepared.request_id for member in inflight.members]

        if self.timeline is not None:
            shared_prep = device_spans["group:vision_prefill_input_prep"]
            shared_tower = device_spans["group:vision_prefill"]
            shared_args = {
                "group_id": inflight.group_id,
                "crops": count,
                "execution": inflight.pack_route.get("execution"),
                "bucket": inflight.pack_route.get("bucket"),
                "real_tokens": inflight.pack_route.get("real_vision_tokens"),
                "physical_tokens": inflight.pack_route.get(
                    "physical_vision_tokens"
                ),
            }
            self.timeline.record_span(
                "Vision prefill",
                "Vision bucket preparation",
                int(shared_prep["start_ns"]),
                int(shared_prep["end_ns"]),
                flow_id=request_ids[0],
                flow_ids=request_ids,
                clock=str(shared_prep["clock"]),
                track="device",
                lane="prefill",
                args=shared_args,
            )
            self.timeline.record_span(
                "Vision prefill",
                (
                    "Packed vision transformer"
                    if count > 1
                    else "Vision transformer"
                ),
                int(shared_tower["start_ns"]),
                int(shared_tower["end_ns"]),
                flow_id=request_ids[0],
                flow_ids=request_ids,
                clock=str(shared_tower["clock"]),
                track="device",
                lane="prefill",
                args=shared_args,
            )
            self.timeline.record_span_seconds(
                "H2D / D2H transfer",
                "First tokens D2H",
                started,
                started + first_token_d2h_s,
                flow_id=request_ids[0],
                flow_ids=request_ids,
                event_type="io",
                args={"group_id": inflight.group_id, "crops": count},
            )
            self.timeline.record_span_seconds(
                "Vision prefill",
                "Resolve prefill completion",
                resolve_started,
                resolve_finished,
                flow_id=request_ids[0],
                flow_ids=request_ids,
                event_type="wait",
                args={"group_id": inflight.group_id, "crops": count},
            )

        shared_device_stage_s = {
            "vision_prefill_input_prep": float(
                device_spans["group:vision_prefill_input_prep"]["seconds"]
            ),
            "vision_prefill": float(device_spans["group:vision_prefill"]["seconds"]),
        }
        vision_stages = {
            "vision_embeddings": "Patch and position embeddings",
            "adaptive_mlp_projector": "Adaptive MLP projector",
        }
        text_stages = {
            "text_token_embedding": "Text token embeddings",
            "image_embed_scatter": "Scatter projected image embeddings",
            "static_cache_alloc": "Allocate request KV cache",
            "text_prefill_input_prep": "Text bucket preparation",
            "text_prefill": "Text transformer prefill",
            "prefill_lm_head": "Prefill LM head",
            "prefill_argmax": "First-token argmax",
        }
        results: list[PrefilledRecognition] = []
        for index, (member, first_token) in enumerate(
            zip(inflight.members, first_tokens)
        ):
            prepared = member.prepared
            device_stage_s: dict[str, float] = {}
            for stage in (
                "recognition_inputs_h2d",
                *vision_stages,
                *text_stages,
            ):
                key = self._group_stage_key(index, stage)
                device_stage_s[stage] = float(device_spans[key]["seconds"])
            device_stage_s.update(
                shared_device_stage_s
                if index == 0
                else {name: 0.0 for name in shared_device_stage_s}
            )
            timing = member.timing_s
            timing["recognizer_h2d"] = device_stage_s["recognition_inputs_h2d"]
            timing["first_token_d2h"] = first_token_d2h_s
            timing["prefill_resolve_wait"] = resolve_finished - resolve_started
            timing["vision_and_text_prefill_wall"] = (
                resolve_finished - inflight.prefill_started
            )
            timing["time_to_first_token"] = (
                resolve_finished - prepared.request_started
            )
            timing["prefill_request_total"] = sum(
                timing[name]
                for name in (
                    "cpu_image_and_prompt_preprocess",
                    "cpu_mrope_index",
                    "cpu_pin_memory",
                    "recognizer_h2d",
                    "vision_and_text_prefill_wall",
                    "first_token_d2h",
                )
            )

            if self.timeline is not None:
                h2d_span = device_spans[
                    self._group_stage_key(index, "recognition_inputs_h2d")
                ]
                self.timeline.record_span(
                    "H2D / D2H transfer",
                    "Recognition inputs H2D",
                    int(h2d_span["start_ns"]),
                    int(h2d_span["end_ns"]),
                    flow_id=prepared.request_id,
                    event_type="io",
                    clock=str(h2d_span["clock"]),
                    track="device",
                    lane="prefill",
                    args={"input_tokens": member.input_tokens},
                )
                for stage, label in (*vision_stages.items(), *text_stages.items()):
                    span = device_spans[self._group_stage_key(index, stage)]
                    route = (
                        member.vision if stage in vision_stages else member.text_prefill
                    )
                    self.timeline.record_span(
                        (
                            "Vision prefill"
                            if stage in vision_stages
                            else "Text prefill"
                        ),
                        label,
                        int(span["start_ns"]),
                        int(span["end_ns"]),
                        flow_id=prepared.request_id,
                        clock=str(span["clock"]),
                        track="device",
                        lane="prefill",
                        args={
                            "stage": stage,
                            "input_tokens": member.input_tokens,
                            "projected_image_tokens": member.projected_image_tokens,
                            "execution": route.get("execution"),
                            "bucket": route.get("bucket"),
                            "real_tokens": route.get(
                                "real_vision_tokens",
                                route.get("real_text_tokens"),
                            ),
                            "physical_tokens": route.get(
                                "physical_vision_tokens",
                                route.get("physical_text_tokens"),
                            ),
                            "pack_group_id": inflight.group_id,
                        },
                    )

            results.append(
                PrefilledRecognition(
                    request_id=prepared.request_id,
                    prompt=prepared.prompt,
                    crop_size=prepared.crop_size,
                    skip_special_tokens=prepared.skip_special_tokens,
                    cache=member.cache,
                    rope_deltas=member.rope_deltas,
                    next_cache_position=member.next_cache_position,
                    next_token=member.next_token,
                    first_token=first_token,
                    input_tokens=member.input_tokens,
                    projected_image_tokens=member.projected_image_tokens,
                    vision=member.vision,
                    text_prefill=member.text_prefill,
                    timing_s=timing,
                    device_stage_s=device_stage_s,
                    request_started=prepared.request_started,
                    prefill_finished=resolve_finished,
                )
            )
        return results

    def configuration(self) -> dict[str, Any]:
        decode_label = (
            f"compiled_static_b{self.batch_size}"
            if self.decode_backend != "raw_eager"
            else f"eager_static_b{self.batch_size}"
        )
        patch_size = int(self.preprocessor_config["patch_size"])
        merge_size = int(self.preprocessor_config["merge_size"])
        min_pixels = int(self.preprocessor_config["min_pixels"])
        vision_attention = self.vision_attention
        return {
            "recognizer_model": str(self.model_dir),
            "device": str(self.device),
            "dtype": str(self.dtype),
            "decode_backend": self.decode_backend,
            "decode_attention": DECODE_ATTENTION if self.device.type == "npu" else "manual",
            "decode_cache_update": DECODE_CACHE_UPDATE if self.device.type == "npu" else "per_row_copy",
            "cache_length": self.cache_length,
            "max_new_tokens": self.max_new_tokens,
            "batch_size": self.batch_size,
            "vision_prefill": self.vision_prefill.metadata,
            "vision_backend": self.vision_backend,
            "vision_attention": vision_attention,
            "vision_packing": {
                "mode": self.vision_packing,
                "target": self.vision_pack_target,
                "lookahead": self.vision_router_lookahead,
                "grouping": (
                    "profiled_oldest_anchored_best_fit"
                    if self.vision_packing == "profile_guided"
                    else "impatient_arrival_order_greedy"
                ),
                "oversized": "faithful_eager_single_crop_route",
                "batched_runtime": (
                    self.batched_vision.metadata
                    if self.batched_vision is not None
                    else None
                ),
            },
            "vision_prompt_fa_layout": (
                get_vision_prompt_fa_layout()
                if vision_attention == "prompt_flash_attention"
                else None
            ),
            "text_prefill": self.text_prefill.metadata,
            "text_backend": self.text_backend,
            "preprocessor": {
                "model_default_min_pixels": self.model_preprocessor_min_pixels,
                "min_pixels_override": self.preprocessor_min_pixels_override,
                "effective_min_pixels": min_pixels,
                "effective_max_pixels": int(self.preprocessor_config["max_pixels"]),
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
            "prefill_production": (
                "next_group_h2d_staged_before_ready_yield"
                if self.vision_packing != "off"
                else "next_crop_h2d_staged_before_ready_yield"
            ),
            "prefill_transfer": "dedicated_stream_event_dependencies",
            "decode": decode_label,
            "decode_schedule": "run_scoped_persistent_slots_iteration_hot_swap",
            "ready_buffer_capacity": READY_BUFFER_BATCH_MULTIPLIER * self.batch_size,
            "ready_buffer_low_watermark": self.batch_size,
            "decode_completion_detection": "queue_depth_one_async_token_copy",
            "kv_admission": "copy_valid_prefill_prefix_into_fixed_slot",
            "text_decode": self.text_decode.metadata,
            "linear_weight_format": self.weight_format,
        }
