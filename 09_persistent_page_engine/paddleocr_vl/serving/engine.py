"""Persistent PaddleOCR-VL runtime with pipelined prefill and batched decode."""

from __future__ import annotations

import hashlib
import json
import sys
import time
import queue
import threading
from collections import Counter, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
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
from .prefill_cache_pool import PrefillKVCacheLease, PrefillKVCachePool
from ..model.modeling import (
    LocalPaddleOCRVLForConditionalGeneration,
    _resolve_model_dir,
)
from ..model.text_decode import (
    DECODE_ATTENTION,
    DECODE_CACHE_UPDATE,
    LocalPaddleOCRVLStaticCache,
    cast_decode_linear_weights_to_nz,
    load_decode_vocab_token_ids,
    prepare_decode_compact_lm_head,
    prepare_decode_optimization_modules,
)
from ..model.token_selection import (
    TOKEN_SELECTION_CHOICES,
    TOKEN_SELECTION_GREEDY,
    TOKEN_SELECTION_SUPPRESS_MATH_OPEN_AND_SLASH_GREEDY,
    TOKEN_SELECTION_SUPPRESS_MATH_OPEN_GREEDY,
    TOKEN_SELECTION_PREFER_MATH_OPEN_PROBABILITY_NEAR_TOP,
    TOKEN_SELECTION_PREFER_MATH_OPEN_TOP2_FIRST_OVERRIDE,
    TOKEN_SELECTION_PREFER_MATH_OPEN_TOP2_NON_NESTED,
    TOKEN_SELECTION_PREFER_MATH_OPEN_VARIANTS_TOP2_P10,
    TOKEN_SELECTION_PREFER_MATH_OPEN_ADJUSTERS_COMBINED,
    select_token_ids,
)
from ..model.preprocessing import (
    apply_pixel_overrides,
    build_inputs,
    load_preprocessor_config,
    preprocess_pil_image,
)
from .runtime_defaults import (
    DECODE_BACKEND_CHOICES,
    DEFAULT_DECODE_OPTIMIZATION,
    DEFAULT_VISION_PACKING,
    DEFAULT_VISION_PACK_TARGET,
    DEFAULT_VISION_ROUTER_LOOKAHEAD,
    DEFAULT_TEXT_BACKEND,
    DEFAULT_TEXT_PACK_BUCKETS,
    DEFAULT_TEXT_PACK_MAX_MEMBERS,
    DEFAULT_TEXT_PACKING,
    DEFAULT_VISION_BACKEND,
    OPTIMIZED_TEXT_BUCKETS,
    OPTIMIZED_VISION_BUCKETS,
    READY_BUFFER_BATCH_MULTIPLIER,
    READY_BUFFER_LOW_WATERMARK_DIVISOR,
    TEXT_PACKING_CHOICES,
    VISION_PACKING_CHOICES,
)
from ..model.text_packed_prefill import PackedTextPrefillRuntime
from ..model.text_prefill import parse_text_buckets
from .types import ContinuousDecodeResult, RecognitionRequest, RecognitionResult
from utils.timing import DeviceTimeline, synchronize
from utils.timeline import TimelineRecorder
from utils.metrics import per_second
from utils.input_fingerprints import fingerprint_recognition_inputs
from ..model.vision_prefill import (
    PreparedVisionPrefill,
    VISION_ATTENTION_CHOICES,
    VISION_LINEAR_WEIGHT_FORMAT_CHOICES,
    VISION_PROMPT_FA_310P_SEQ_ALIGNMENT,
    align_vision_buckets,
    align_vision_seq_len,
    get_vision_prompt_fa_layout,
    prepare_vision_linear_weight_format,
    prepare_vision_mlp_intermediate,
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
    input_fingerprints: dict[str, Any] = field(default_factory=dict)


@dataclass
class _InFlightPrefillMember:
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


@dataclass
class _TextPrefillInputMember:
    prepared: CpuPreparedRecognition
    moved: tuple[torch.Tensor, ...]
    cache: LocalPaddleOCRVLStaticCache
    cache_lease: PrefillKVCacheLease
    rope_deltas: torch.Tensor
    next_cache_position: torch.Tensor
    inputs_embeds: torch.Tensor
    vision: dict[str, Any]
    timing_s: dict[str, float]
    projected_image_tokens: int


@dataclass(frozen=True)
class _TextPackTrace:
    member_indices: tuple[int, ...]
    route: dict[str, Any]
    stage_keys: dict[str, str]


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
    text_packs: list[_TextPackTrace]


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


@dataclass
class _TextPackingRunStats:
    mode: str
    buckets: tuple[int, ...]
    groups: int = 0
    crops: int = 0
    packs: int = 0
    packed_crops: int = 0
    fallback_crops: int = 0
    packed_real_tokens: int = 0
    packed_physical_tokens: int = 0
    redistributed_kv_bytes: int = 0
    pack_size_histogram: Counter[int] | None = None
    bucket_histogram: Counter[int] | None = None

    def __post_init__(self) -> None:
        if self.pack_size_histogram is None:
            self.pack_size_histogram = Counter()
        if self.bucket_histogram is None:
            self.bucket_histogram = Counter()

    def record_pack(
        self,
        *,
        members: int,
        real_tokens: int,
        physical_tokens: int,
        redistributed_kv_bytes: int,
    ) -> None:
        self.packs += 1
        self.packed_crops += int(members)
        self.packed_real_tokens += int(real_tokens)
        self.packed_physical_tokens += int(physical_tokens)
        self.redistributed_kv_bytes += int(redistributed_kv_bytes)
        self.pack_size_histogram[int(members)] += 1
        self.bucket_histogram[int(physical_tokens)] += 1

    def summary(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "buckets": list(self.buckets),
            "groups": self.groups,
            "crops": self.crops,
            "packs": self.packs,
            "packed_crops": self.packed_crops,
            "fallback_crops": self.fallback_crops,
            "calls": self.packs + self.fallback_crops,
            "call_reduction_fraction": (
                1.0 - (self.packs + self.fallback_crops) / self.crops
                if self.crops
                else None
            ),
            "pack_size_histogram": {
                str(size): count
                for size, count in sorted(self.pack_size_histogram.items())
            },
            "bucket_histogram": {
                str(bucket): count
                for bucket, count in sorted(self.bucket_histogram.items())
            },
            "packed_real_text_tokens": self.packed_real_tokens,
            "packed_physical_text_tokens": self.packed_physical_tokens,
            "packed_fill_fraction": (
                self.packed_real_tokens / self.packed_physical_tokens
                if self.packed_physical_tokens
                else None
            ),
            "redistributed_kv_bytes": self.redistributed_kv_bytes,
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
            raise RuntimeError(f"prefill device state already taken for {self.request_id}")
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


class _OpenPrefillSource:
    """Turn an open crop source into ready decode requests on demand."""

    def __init__(
        self,
        recognizer: Any,
        requests: Any,
        *,
        on_request_error: Callable[[str, BaseException], None],
    ):
        self.recognizer = recognizer
        self.requests = requests
        self.on_request_error = on_request_error
        self.pending: deque[tuple[str, Future[CpuPreparedRecognition]]] = deque()
        self.executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="paddleocr-vl-open-cpu-prepare",
        )
        self._executor_closed = False

    @property
    def closed(self) -> bool:
        return bool(self.requests.closed) and not self.pending

    def _submit_available(self, *, block_for_first: bool) -> None:
        while len(self.pending) < self.recognizer.cpu_preprocess_max_pending:
            request = self.requests.pull(
                block=block_for_first and not self.pending,
            )
            block_for_first = False
            if request is None:
                break
            submitted_at = time.perf_counter()
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

    def pull(self, *, block: bool) -> ReadyDecodeRequest | None:
        while True:
            self._submit_available(block_for_first=block and not self.pending)
            if not self.pending:
                return None
            request_id, future = self.pending.popleft()
            wait_started = time.perf_counter()
            try:
                prepared = future.result()
            except BaseException as exc:
                self.on_request_error(request_id, exc)
                block = False
                continue
            consumer_wait_s = time.perf_counter() - wait_started
            # Refill the CPU lane before NPU prefill so host preparation for
            # later HTTP requests overlaps the current crop's device work.
            self._submit_available(block_for_first=False)
            group = self.recognizer._prepared_group(
                [(prepared, consumer_wait_s)]
            )
            staged = self.recognizer._stage_prefill_group(group)
            inflight = self.recognizer._enqueue_staged_prefill_group(staged)
            finalized = self.recognizer._finalize_prefill_group(inflight)
            if len(finalized) != 1:
                raise RuntimeError(
                    "open single-crop prefill produced "
                    f"{len(finalized)} ready states"
                )
            return self.recognizer._ready_from_prefilled(finalized[0])

    def close(self) -> None:
        if self._executor_closed:
            return
        self._executor_closed = True
        self.executor.shutdown(wait=True, cancel_futures=True)


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
        vision_promptfa_align_128: bool = False,
        vision_mlp_intermediate_size: int | None = None,
        vision_linear_weight_format: str = "native",
        vision_packing: str = DEFAULT_VISION_PACKING,
        vision_pack_target: int = DEFAULT_VISION_PACK_TARGET,
        vision_router_lookahead: int = DEFAULT_VISION_ROUTER_LOOKAHEAD,
        vision_batched_cache_dir: Path | None = None,
        text_backend: str = DEFAULT_TEXT_BACKEND,
        text_buckets: str | Iterable[int] = OPTIMIZED_TEXT_BUCKETS,
        text_torchair_cache_dir: Path | None = None,
        text_padding: str = "auto",
        text_packing: str = DEFAULT_TEXT_PACKING,
        text_pack_buckets: str | Iterable[int] = DEFAULT_TEXT_PACK_BUCKETS,
        text_pack_max_members: int = DEFAULT_TEXT_PACK_MAX_MEMBERS,
        text_packed_cache_dir: Path | None = None,
        preprocessor_min_pixels: int | None = None,
        preprocessor_max_pixels: int | None = None,
        vision_route_plan: dict[str, Any] | None = None,
        timeline: TimelineRecorder | None = None,
        scheduler_progress: bool = False,
        scheduler_progress_events: Iterable[str] | None = None,
        diagnostic_decode_effective_length: int | None = None,
        diagnostic_decode_request_id: str | None = None,
        diagnostic_prefill_kv_request_ids: Iterable[str] | None = None,
        decode_optimization: str = DEFAULT_DECODE_OPTIMIZATION,
        decode_vocab_token_ids: Path | None = None,
        recognition_input_fingerprints: bool = False,
        compact_uint8_preprocess: bool = False,
        token_selection: str = TOKEN_SELECTION_GREEDY,
    ):
        # TorchAir guards tensor dispatch-key sets. Build and warm every graph
        # under the same inference-mode contract used by run(),
        # otherwise the first real request invalidates the persistent cache
        # and recompiles the vision, text-prefill, and decode boundaries.
        runtime_started = time.perf_counter()
        import torch_npu

        self.vision_linear_weight_format_requested = str(
            vision_linear_weight_format
        )
        if (
            self.vision_linear_weight_format_requested
            not in VISION_LINEAR_WEIGHT_FORMAT_CHOICES
        ):
            raise ValueError(
                "vision_linear_weight_format must be one of "
                f"{VISION_LINEAR_WEIGHT_FORMAT_CHOICES}, "
                f"got {self.vision_linear_weight_format_requested!r}"
            )
        if self.vision_linear_weight_format_requested == "fractal_nz":
            # torch-npu 2.10 keeps npu_format_cast in ND unless this is set
            # before the first NPU allocation in the process. The production
            # runner sets it before layout setup; this makes direct recognizer
            # construction obey the same contract.
            torch.npu.config.allow_internal_format = True

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
        self.decode_optimization = str(decode_optimization)
        self.decode_vocab_token_ids_path = (
            None
            if decode_vocab_token_ids is None
            else Path(decode_vocab_token_ids).expanduser().resolve()
        )
        self.token_selection = str(token_selection)
        if self.token_selection not in TOKEN_SELECTION_CHOICES:
            raise ValueError(
                "token_selection must be one of "
                f"{TOKEN_SELECTION_CHOICES}, got {self.token_selection!r}"
            )
        self.timeline = timeline
        self.scheduler_progress = bool(scheduler_progress)
        self.scheduler_progress_events = (
            None
            if scheduler_progress_events is None
            else frozenset(str(event) for event in scheduler_progress_events)
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
            str(request_id)
            for request_id in (diagnostic_prefill_kv_request_ids or ())
        )
        self.batch_size = int(batch_size)
        self.cache_length = int(cache_length)
        self.max_new_tokens = int(max_new_tokens)
        self.recognition_input_fingerprints = bool(
            recognition_input_fingerprints
        )
        self.compact_uint8_preprocess = bool(compact_uint8_preprocess)
        self.vision_backend = str(vision_backend)
        self.vision_mlp_intermediate_size_requested = (
            None
            if vision_mlp_intermediate_size is None
            else int(vision_mlp_intermediate_size)
        )
        self.vision_attention = str(vision_attention)
        if self.vision_attention not in VISION_ATTENTION_CHOICES:
            raise ValueError(
                "vision attention must be one of "
                f"{VISION_ATTENTION_CHOICES}, got {vision_attention!r}"
            )
        self.vision_promptfa_align_128 = bool(vision_promptfa_align_128)
        if (
            self.vision_promptfa_align_128
            and self.vision_attention != "prompt_flash_attention"
        ):
            raise ValueError(
                "vision_promptfa_align_128 requires "
                "vision_attention='prompt_flash_attention'"
            )
        self.vision_seq_alignment = (
            VISION_PROMPT_FA_310P_SEQ_ALIGNMENT
            if self.vision_promptfa_align_128
            else 1
        )
        self.vision_buckets = align_vision_buckets(
            vision_buckets,
            self.vision_seq_alignment,
        )
        self.vision_padding = str(vision_padding)
        self.vision_packing = str(vision_packing)
        self.vision_pack_target = align_vision_seq_len(
            int(vision_pack_target),
            self.vision_seq_alignment,
        )
        self.vision_router_lookahead = int(vision_router_lookahead)
        if self.vision_packing not in VISION_PACKING_CHOICES:
            raise ValueError(
                "vision_packing must be one of "
                f"{VISION_PACKING_CHOICES}, got {vision_packing!r}"
            )
        if (
            self.vision_packing != "off"
            and self.vision_pack_target not in self.vision_buckets
        ):
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
        if vision_route_plan is not None:
            if self.vision_packing != "profile_guided":
                raise ValueError(
                    "a vision route plan requires profile_guided vision packing"
                )
            if int(vision_route_plan.get("schema_version", 0)) != 1:
                raise ValueError("unsupported vision route plan schema")
            groups = vision_route_plan.get("groups")
            if not isinstance(groups, list) or not groups:
                raise ValueError("vision route plan must contain non-empty groups")
            self._vision_route_replay_groups = [dict(group) for group in groups]
        else:
            self._vision_route_replay_groups = None
        self.text_backend = str(text_backend)
        self.text_buckets = parse_text_buckets(text_buckets)
        self.text_padding = str(text_padding)
        self.text_packing = str(text_packing)
        self.text_pack_buckets = parse_text_buckets(text_pack_buckets)
        self.text_pack_max_members = int(text_pack_max_members)
        if self.text_packing not in TEXT_PACKING_CHOICES:
            raise ValueError(
                "text_packing must be one of "
                f"{TEXT_PACKING_CHOICES}, got {text_packing!r}"
            )
        if self.text_pack_max_members <= 0:
            raise ValueError("text_pack_max_members must be positive")
        if self.text_packing != "off" and self.text_backend != "torchair":
            raise ValueError("packed text prefill requires TorchAir text prefill")
        if self.decode_backend not in DECODE_BACKEND_CHOICES:
            raise ValueError(
                f"decode_backend must be one of {DECODE_BACKEND_CHOICES}, "
                f"got {self.decode_backend!r}"
            )
        if self.batch_size <= 0 or self.batch_size & (self.batch_size - 1):
            raise ValueError("batch_size must be a positive power of two")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")

        model_preprocessor_config = load_preprocessor_config(self.model_dir)
        self.model_preprocessor_min_pixels = int(model_preprocessor_config["min_pixels"])
        self.model_preprocessor_max_pixels = int(model_preprocessor_config["max_pixels"])
        self.preprocessor_min_pixels_override = (
            None if preprocessor_min_pixels is None else int(preprocessor_min_pixels)
        )
        self.preprocessor_max_pixels_override = (
            None if preprocessor_max_pixels is None else int(preprocessor_max_pixels)
        )
        self.preprocessor_config = apply_pixel_overrides(
            model_preprocessor_config,
            min_pixels=self.preprocessor_min_pixels_override,
            max_pixels=self.preprocessor_max_pixels_override,
        )
        if self.compact_uint8_preprocess:
            mean = tuple(
                float(value) for value in self.preprocessor_config["image_mean"]
            )
            std = tuple(
                float(value) for value in self.preprocessor_config["image_std"]
            )
            if (
                not self.preprocessor_config["do_rescale"]
                or not self.preprocessor_config["do_normalize"]
                or len(set(mean)) != 1
                or len(set(std)) != 1
            ):
                raise ValueError(
                    "compact_uint8_preprocess currently requires scalar RGB "
                    "rescale and normalization parameters"
                )
            self.compact_rescale_factor = float(
                self.preprocessor_config["rescale_factor"]
            )
            self.compact_image_mean = mean[0]
            self.compact_image_std = std[0]
        # Encoding runs continuously on the CPU preparation thread while this
        # tokenizer remains owned by the NPU thread for result decoding.
        self.preprocessing_tokenizer = Tokenizer.from_file(
            str(self.model_dir / "tokenizer.json")
        )
        self.tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        math_open_token_id = self.tokenizer.token_to_id(r"\(")
        if math_open_token_id is None:
            raise ValueError("recognizer tokenizer does not contain the exact \\( token")
        self.math_open_token_id = int(math_open_token_id)
        math_slash_token_id = self.tokenizer.token_to_id("\\")
        if math_slash_token_id is None:
            raise ValueError("recognizer tokenizer does not contain the exact \\ token")
        self.math_slash_token_id = int(math_slash_token_id)
        math_close_token_id = self.tokenizer.token_to_id(r"\)")
        if math_close_token_id is None:
            raise ValueError("recognizer tokenizer does not contain the exact \\) token")
        self.math_close_token_id = int(math_close_token_id)
        table_cell_pieces = ("<fcel>", "<ecel>", "<lcel>", "<ucel>", "<xcel>")
        table_cell_token_ids = [
            self.tokenizer.token_to_id(piece) for piece in table_cell_pieces
        ]
        if any(token_id is None for token_id in table_cell_token_ids):
            raise ValueError("recognizer tokenizer is missing a table cell token")
        self.table_cell_token_ids = tuple(
            int(token_id) for token_id in table_cell_token_ids if token_id is not None
        )
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

        self.decode_vocab: dict[str, Any] = {"enabled": False}
        if self.decode_vocab_token_ids_path is not None:
            if self.token_selection != TOKEN_SELECTION_GREEDY:
                raise ValueError(
                    "--decode-vocab-token-ids currently requires "
                    "--token-selection greedy"
                )
            token_ids, self.decode_vocab = load_decode_vocab_token_ids(
                self.decode_vocab_token_ids_path,
                full_vocab_size=int(self.model.lm_head.weight.shape[0]),
            )
            prepare_decode_compact_lm_head(self.model, token_ids)
            synchronize(self.device)

        synchronize(self.device)
        started = time.perf_counter()
        self.vision_mlp = prepare_vision_mlp_intermediate(
            self.model,
            target_intermediate_size=(
                self.vision_mlp_intermediate_size_requested
            ),
        )
        synchronize(self.device)
        vision_mlp_setup_s = time.perf_counter() - started

        synchronize(self.device)
        started = time.perf_counter()
        self.vision_weight_format = prepare_vision_linear_weight_format(
            self.model,
            requested=self.vision_linear_weight_format_requested,
        )
        synchronize(self.device)
        vision_weight_format_s = time.perf_counter() - started

        synchronize(self.device)
        started = time.perf_counter()
        decode_optimization_config = prepare_decode_optimization_modules(
            self.model,
            self.decode_optimization,
        )
        synchronize(self.device)
        decode_optimization_setup_s = time.perf_counter() - started

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
            vision_seq_alignment=self.vision_seq_alignment,
            vision_mlp_intermediate_size=int(
                self.vision_mlp["target_intermediate_size"]
            ),
            vision_linear_weight_format=str(
                self.vision_weight_format["effective_mode"]
            ),
            text_backend=self.text_backend,
            text_buckets=self.text_buckets,
            text_cache_root=(
                text_torchair_cache_dir
                if text_torchair_cache_dir is not None
                else torchair_cache_dir.parent / f"{torchair_cache_dir.name}_text"
            ),
            text_padding=self.text_padding,
            decode_backend=self.decode_backend,
            decode_optimization=decode_optimization_config.name,
            decode_cache_root=(
                torchair_cache_dir
                if not self.decode_vocab["enabled"]
                else torchair_cache_dir
                / (
                    f"selected_vocab_{self.decode_vocab['selected_vocab_size']}_"
                    f"{self.decode_vocab['token_ids_sha256'][:12]}"
                )
            ),
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
        packed_text_started = time.perf_counter()
        self.packed_text_prefill = (
            PackedTextPrefillRuntime(
                self.model,
                buckets=self.text_pack_buckets,
                max_members=self.text_pack_max_members,
                cache_root=(
                    text_packed_cache_dir
                    if text_packed_cache_dir is not None
                    else torchair_cache_dir.parent
                    / f"{torchair_cache_dir.name}_text_packed"
                ),
                destination_cache_length=self.cache_length,
                device=self.device,
                dtype=self.dtype,
                model_dir=self.model_dir,
                linear_weight_format=str(self.weight_format["effective_mode"]),
            )
            if self.text_packing != "off"
            else None
        )
        synchronize(self.device)
        packed_text_setup_s = time.perf_counter() - packed_text_started
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
        self.ready_buffer_capacity = (
            READY_BUFFER_BATCH_MULTIPLIER * self.batch_size
        )
        self.ready_buffer_low_watermark = max(
            1,
            self.ready_buffer_capacity // READY_BUFFER_LOW_WATERMARK_DIVISOR,
        )
        # A refill may suspend ready_stream() partway through yielding one
        # already-prefilled production group.  Keep one maximum text-pack
        # group's worth of leases beyond the bounded ready reservoir so that
        # those not-yet-yielded members cannot exhaust the arena.
        self.private_cache_staging_headroom = self.text_pack_max_members
        self.prefill_host_tokens = torch.empty(
            (
                max(
                    self.cpu_preprocess_max_pending + 1,
                    self.text_pack_max_members,
                ),
            ),
            dtype=torch.int64,
            pin_memory=True,
        )
        self._vision_pack_sequence = 0
        self._captured_vision_route_groups: list[dict[str, Any]] = []
        self._vision_packing_stats = _VisionPackingRunStats(
            self.vision_packing,
            self.vision_pack_target,
            self.vision_router_lookahead,
        )
        self._text_packing_stats = _TextPackingRunStats(
            self.text_packing,
            self.text_pack_buckets,
        )

        private_cache_pool_started = time.perf_counter()
        private_cache_capacity = (
            self.ready_buffer_capacity + self.private_cache_staging_headroom
        )
        private_cache_storage = self.model.allocate_static_cache(
            batch_size=private_cache_capacity,
            cache_length=self.cache_length,
            device=self.device,
            dtype=self.dtype,
            init_mode="zeros",
        )
        synchronize(self.device)
        self.prefill_cache_pool = PrefillKVCachePool(
            private_cache_storage,
            device=self.device,
        )
        private_cache_pool_setup_s = (
            time.perf_counter() - private_cache_pool_started
        )

        started = time.perf_counter()
        self.decode_arena = DecodeArena(
            cache=self.text_decode.warm_cache,
            device=self.device,
            batch_size=self.batch_size,
            eos_token_id=int(self.model.config.eos_token_id),
            token_selection=self.token_selection,
            preferred_token_id=self.math_open_token_id,
            alternate_preferred_token_id=self.math_slash_token_id,
            cell_start_token_ids=self.table_cell_token_ids,
            math_close_token_id=self.math_close_token_id,
            decode_token_id_map=getattr(
                self.model,
                "decode_token_id_map",
                None,
            ),
            timeline=self.timeline,
        )
        self.decode_scheduler = ContinuousDecodeScheduler(
            arena=self.decode_arena,
            decode_fn=self.decode_fn,
            max_new_tokens=self.max_new_tokens,
            timeline=self.timeline,
            stop_repetitions=True,
            progress=self._emit_scheduler_progress,
            diagnostic_effective_length=(
                self.diagnostic_decode_effective_length
            ),
            diagnostic_request_id=self.diagnostic_decode_request_id,
        )
        decode_control_setup_s = time.perf_counter() - started

        self.setup_timing_s = {
            "recognizer_frontend_setup": float(frontend_setup_s),
            "recognizer_model_load": float(model_load_s),
            "vision_mlp_padding": float(vision_mlp_setup_s),
            "vision_weight_format": float(vision_weight_format_s),
            "decode_optimization_setup": float(decode_optimization_setup_s),
            "decode_weight_format": float(weight_format_s),
            **self.stages.setup_timing_s,
            "vision_router_setup": (
                float(self.batched_vision.metadata["setup_s"])
                if self.batched_vision is not None
                else 0.0
            ),
            "packed_text_runtime_setup": float(packed_text_setup_s),
            "private_cache_pool_setup": float(private_cache_pool_setup_s),
            "decode_control_setup": float(decode_control_setup_s),
            "recognizer_runtime_total": float(time.perf_counter() - runtime_started),
        }

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

    def _begin_decode_schedule(self) -> None:
        self._vision_pack_sequence = 0
        self._captured_vision_route_groups = []
        self._vision_packing_stats = _VisionPackingRunStats(
            self.vision_packing,
            self.vision_pack_target,
            self.vision_router_lookahead,
        )
        self._text_packing_stats = _TextPackingRunStats(
            self.text_packing,
            self.text_pack_buckets,
        )

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

        group = self._prepared_group([(prepared, 0.0)])
        staged = self._stage_prefill_group(group)
        inflight = self._enqueue_staged_prefill_group(staged)
        finalized = self._finalize_prefill_group(inflight)
        if len(finalized) != 1:
            raise RuntimeError(
                f"single-crop prefill produced {len(finalized)} states"
            )
        return finalized[0]

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
            token_selection_policy_active=(
                self.token_selection != TOKEN_SELECTION_GREEDY
                and self._is_table_prompt(state.prompt)
            ),
            cache_release=cache_release,
        )

    def _decode_ready_source(
        self,
        ready_source: Any,
        *,
        schedule_id: str,
        emit_result: Callable[[RecognitionResult], None],
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
            vision_packing=self._vision_packing_stats.summary(),
            text_packing={
                **self._text_packing_stats.summary(),
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
            current_staged: _StagedPrefillGroup | None = None
            drained_normally = False
            h2d_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="prefill-h2d",
            )
            self._emit_scheduler_progress("ready_stream_begin")
            try:
                if self.vision_packing == "off":
                    groups = self._iter_single_prefill_groups(requests)
                elif self.vision_packing == "greedy":
                    groups = self._iter_packed_prefill_groups(requests)
                elif self.vision_packing == "cohort":
                    groups = self._iter_cohort_prefill_groups(requests)
                else:
                    groups = self._iter_profiled_prefill_groups(requests)
                group_source = iter(groups)
                try:
                    self._emit_scheduler_progress(
                        "prefill_group_source_next_begin",
                        phase="first",
                    )
                    first_group = next(group_source)
                except StopIteration:
                    self._emit_scheduler_progress(
                        "prefill_group_source_exhausted",
                        phase="first",
                    )
                    drained_normally = True
                    return
                self._emit_scheduler_progress(
                    "prefill_group_source_next_end",
                    phase="first",
                    group_id=first_group.group_id,
                    crops=len(first_group.members),
                    real_vision_tokens=first_group.real_vision_tokens,
                )
                self._emit_scheduler_progress(
                    "prefill_h2d_stage_begin",
                    group_id=first_group.group_id,
                    crops=len(first_group.members),
                )
                current_staged = self._stage_prefill_group(first_group)
                self._emit_scheduler_progress(
                    "prefill_h2d_stage_end",
                    group_id=first_group.group_id,
                    crops=len(first_group.members),
                )
                while current_staged is not None:
                    try:
                        self._emit_scheduler_progress(
                            "prefill_group_source_next_begin",
                            phase="lookahead",
                            current_group_id=current_staged.group.group_id,
                        )
                        next_group = next(group_source)
                    except StopIteration:
                        self._emit_scheduler_progress(
                            "prefill_group_source_exhausted",
                            phase="lookahead",
                            current_group_id=current_staged.group.group_id,
                        )
                        next_stage_future = None
                    else:
                        self._emit_scheduler_progress(
                            "prefill_group_source_next_end",
                            phase="lookahead",
                            current_group_id=current_staged.group.group_id,
                            group_id=next_group.group_id,
                            crops=len(next_group.members),
                            real_vision_tokens=next_group.real_vision_tokens,
                        )
                        # TorchAir occupies this thread for much of G's device
                        # work. Submit only G+1's H2D on a dedicated host
                        # worker while this thread invokes G's compute chain.
                        next_stage_future = h2d_executor.submit(
                            self._stage_prefill_group,
                            next_group,
                        )

                    self._emit_scheduler_progress(
                        "prefill_enqueue_begin",
                        group_id=current_staged.group.group_id,
                        crops=len(current_staged.group.members),
                        real_vision_tokens=current_staged.group.real_vision_tokens,
                    )
                    final = self._enqueue_staged_prefill_group(current_staged)
                    self._emit_scheduler_progress(
                        "prefill_enqueue_end",
                        group_id=final.group_id,
                        crops=len(final.members),
                    )
                    current_staged = None
                    if next_stage_future is not None:
                        self._emit_scheduler_progress(
                            "prefill_lookahead_h2d_wait_begin",
                            current_group_id=final.group_id,
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
                            current_group_id=final.group_id,
                            next_group_id=next_staged.group.group_id,
                        )
                    self._emit_scheduler_progress(
                        "prefill_finalize_begin",
                        group_id=final.group_id,
                        crops=len(final.members),
                    )
                    finalized = self._finalize_prefill_group(final)
                    self._emit_scheduler_progress(
                        "prefill_finalize_end",
                        group_id=final.group_id,
                        crops=len(finalized),
                    )
                    for state in finalized:
                        self._emit_scheduler_progress(
                            "ready_state_yield",
                            group_id=final.group_id,
                            request_id=state.request_id,
                        )
                        yield self._ready_from_prefilled(state)
                    self._emit_scheduler_progress(
                        "prefill_group_yield_complete",
                        group_id=final.group_id,
                        crops=len(finalized),
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

    @torch.inference_mode()
    def serve(
        self,
        requests: Any,
        *,
        schedule_id: str,
        emit_result: Callable[[RecognitionResult], None],
        on_request_error: Callable[[str, BaseException], None],
    ) -> ContinuousDecodeResult:
        """Serve an open stream whose input can be temporarily empty.

        Each arrival remains one independent crop request. The model stages
        stay unchanged, while ready crops enter the same fixed decode arena
        and can hot-swap into free slots until the caller closes the source.
        """

        if self.vision_packing != "off" or self.text_packing != "off":
            raise ValueError(
                "open crop serving currently requires independent vision and "
                "text prefill (vision_packing='off', text_packing='off')"
            )
        self._begin_decode_schedule()
        ready_source = _OpenPrefillSource(
            self,
            requests,
            on_request_error=on_request_error,
        )
        try:
            return self._decode_ready_source(
                ready_source,
                schedule_id=schedule_id,
                emit_result=emit_result,
            )
        finally:
            ready_source.close()

    def vision_route_plan(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "vision_packing": self.vision_packing,
            "groups": list(self._captured_vision_route_groups),
        }

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

    def _iter_cohort_prefill_groups(
        self,
        requests: Iterable[RecognitionRequest],
    ) -> Iterable[_PreparedPrefillGroup]:
        """Pack each bounded request cohort before launching vision prefill."""

        members: list[tuple[CpuPreparedRecognition, float]] = []
        total = 0
        for item in self._iter_cpu_prepared(requests):
            tokens = int(item[0].pixel_values.shape[0])
            if members and total + tokens > self.vision_pack_target:
                yield self._prepared_group(members)
                members = []
                total = 0
            members.append(item)
            total += tokens
        if members:
            yield self._prepared_group(members)

    def _iter_profiled_prefill_groups(
        self,
        requests: Iterable[RecognitionRequest],
    ) -> Iterable[_PreparedPrefillGroup]:
        """Route up to the currently ready lookahead without waiting to fill it."""

        if self._vision_route_replay_groups is not None:
            yield from self._iter_replayed_profiled_prefill_groups(requests)
            return

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
                self._capture_profiled_vision_group(group)
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

    def _capture_profiled_vision_group(
        self,
        group: _PreparedPrefillGroup,
    ) -> None:
        if group.profiled_route is None:
            raise ValueError("profiled vision group is missing its route")
        route = {
            key: value
            for key, value in group.profiled_route.items()
            if key != "router_cpu_s"
        }
        self._captured_vision_route_groups.append(
            {
                "request_ids": [
                    prepared.request_id for prepared, _wait_s in group.members
                ],
                "row_sizes": list(group.row_sizes),
                "route": route,
            }
        )

    def _iter_replayed_profiled_prefill_groups(
        self,
        requests: Iterable[RecognitionRequest],
    ) -> Iterable[_PreparedPrefillGroup]:
        prepared_source = iter(self._iter_cpu_prepared(requests))
        waiting: dict[str, tuple[CpuPreparedRecognition, float]] = {}
        consumed_request_ids: set[str] = set()

        for plan_index, entry in enumerate(self._vision_route_replay_groups or ()):
            request_ids = entry.get("request_ids")
            row_sizes = entry.get("row_sizes")
            planned_route = entry.get("route")
            if (
                not isinstance(request_ids, list)
                or not request_ids
                or len(set(request_ids)) != len(request_ids)
                or not isinstance(row_sizes, list)
                or not isinstance(planned_route, dict)
            ):
                raise ValueError(f"invalid vision route plan group {plan_index}")

            missing = set(str(request_id) for request_id in request_ids)
            while not missing.issubset(waiting):
                try:
                    item = next(prepared_source)
                except StopIteration as exc:
                    raise RuntimeError(
                        "vision route plan references requests absent from the "
                        f"input stream: {sorted(missing - set(waiting))[:5]}"
                    ) from exc
                request_id = item[0].request_id
                if request_id in waiting or request_id in consumed_request_ids:
                    raise RuntimeError(
                        f"duplicate prepared request while replaying route: {request_id}"
                    )
                waiting[request_id] = item

            selected = [waiting.pop(str(request_id)) for request_id in request_ids]
            consumed_request_ids.update(str(request_id) for request_id in request_ids)
            route_started = time.perf_counter()
            route = dict(planned_route)
            route["router_cpu_s"] = time.perf_counter() - route_started
            group = self._prepared_group(
                selected,
                row_sizes=tuple(int(size) for size in row_sizes),
                profiled_route=route,
            )
            self._capture_profiled_vision_group(group)
            yield group

        try:
            extra = next(prepared_source)
        except StopIteration:
            extra = None
        if extra is not None or waiting:
            extra_ids = list(waiting)
            if extra is not None:
                extra_ids.append(extra[0].request_id)
            raise RuntimeError(
                "vision route plan did not consume the full request stream: "
                f"{extra_ids[:5]}"
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
            input_fingerprints=dict(state.input_fingerprints),
            repetition=(
                dict(completion.repetition_evidence)
                if completion.repetition_evidence is not None
                else {}
            ),
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
            defer_normalization=self.compact_uint8_preprocess,
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

        prompt_length = int(input_ids.shape[1])
        if prompt_length > self.cache_length:
            raise ValueError(
                f"request {request.request_id} has prompt_length={prompt_length}, "
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

    @staticmethod
    def _group_stage_key(member_index: int, stage: str) -> str:
        return f"member:{int(member_index)}:{stage}"

    def _form_text_pack_indices(
        self,
        lengths: list[int],
    ) -> tuple[list[tuple[int, ...]], tuple[int, ...]]:
        """Best-fit decreasing inside one already-selected vision group."""

        if self.packed_text_prefill is None:
            return [], tuple(range(len(lengths)))
        capacity = self.text_pack_buckets[-1]
        eligible = [
            index for index, length in enumerate(lengths) if length <= capacity
        ]
        fallback = tuple(
            index for index, length in enumerate(lengths) if length > capacity
        )
        packs: list[list[int]] = []
        totals: list[int] = []
        for member_index in sorted(
            eligible,
            key=lambda index: (-lengths[index], index),
        ):
            length = lengths[member_index]
            choices = [
                index
                for index, pack in enumerate(packs)
                if len(pack) < self.text_pack_max_members
                and totals[index] + length <= capacity
            ]
            if choices:
                selected = max(choices, key=lambda index: totals[index])
                packs[selected].append(member_index)
                totals[selected] += length
            else:
                packs.append([member_index])
                totals.append(length)
        if sum(len(pack) for pack in packs) + len(fallback) != len(lengths):
            raise AssertionError("text pack formation lost prefill members")
        return [tuple(pack) for pack in packs], fallback

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
                    pixel_values = (
                        prepared.pixel_values.to(
                            device=self.device,
                            non_blocking=True,
                        )
                        if prepared.pixel_values.dtype == torch.uint8
                        else prepared.pixel_values.to(
                            device=self.device,
                            dtype=self.model.visual.dtype,
                            non_blocking=True,
                        )
                    )
                    return (
                        prepared.input_ids.to(self.device, non_blocking=True),
                        prepared.attention_mask.to(self.device, non_blocking=True),
                        pixel_values,
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

    @staticmethod
    def _is_table_prompt(prompt: str) -> bool:
        return str(prompt).strip() == "Table Recognition:"

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
            if pixel_values_device.dtype == torch.uint8:
                def normalize_uint8(
                    pixels: torch.Tensor = pixel_values_device,
                ) -> torch.Tensor:
                    output = pixels.to(torch.float32)
                    output.mul_(self.compact_rescale_factor)
                    output.sub_(self.compact_image_mean)
                    output.div_(self.compact_image_std)
                    return output.to(self.model.visual.dtype).contiguous()

                pixel_values_device = device_timeline.measure(
                    self._group_stage_key(index, "vision_input_normalize"),
                    normalize_uint8,
                )
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
        text_inputs: list[_TextPrefillInputMember] = []
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
            cache_lease = device_timeline.measure(
                self._group_stage_key(index, "static_cache_alloc"),
                self.prefill_cache_pool.acquire,
            )
            cache = cache_lease.cache
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
            text_inputs.append(
                _TextPrefillInputMember(
                    prepared=prepared,
                    moved=moved,
                    cache=cache,
                    cache_lease=cache_lease,
                    rope_deltas=rope_deltas,
                    next_cache_position=next_cache_position,
                    inputs_embeds=inputs_embeds,
                    vision=vision_route,
                    timing_s=timing,
                    projected_image_tokens=int(image_embeds.shape[0]),
                )
            )

        self._text_packing_stats.groups += 1
        self._text_packing_stats.crops += len(text_inputs)
        lengths = [int(item.inputs_embeds.shape[1]) for item in text_inputs]
        pack_indices, fallback_indices = self._form_text_pack_indices(lengths)
        next_tokens: list[torch.Tensor | None] = [None] * len(text_inputs)
        text_routes: list[dict[str, Any] | None] = [None] * len(text_inputs)
        text_packs: list[_TextPackTrace] = []

        if self.packed_text_prefill is not None:
            for pack_index, indices in enumerate(pack_indices):
                pack_lengths = [lengths[index] for index in indices]
                text_route = self.packed_text_prefill.route(pack_lengths)
                prefix = f"group:text_pack:{pack_index}"
                prepared_text = device_timeline.measure(
                    f"{prefix}:text_prefill_input_prep",
                    lambda indices=indices, text_route=text_route: (
                        self.packed_text_prefill.prepare(
                            [text_inputs[index].inputs_embeds for index in indices],
                            [text_inputs[index].moved[3] for index in indices],
                            route=text_route,
                        )
                    ),
                )
                packed_hidden = device_timeline.measure(
                    f"{prefix}:text_prefill",
                    lambda prepared_text=prepared_text: (
                        self.packed_text_prefill.run_prepared(prepared_text)
                    ),
                )
                redistributed_bytes = device_timeline.measure(
                    f"{prefix}:text_kv_redistribute",
                    lambda prepared_text=prepared_text, indices=indices: (
                        self.packed_text_prefill.redistribute_cache(
                            prepared_text,
                            [text_inputs[index].cache for index in indices],
                        )
                    ),
                )
                valid_hidden = packed_hidden[:, : len(indices)]
                logits = device_timeline.measure(
                    f"{prefix}:prefill_lm_head",
                    lambda valid_hidden=valid_hidden: self.model.lm_head(valid_hidden),
                )
                packed_policy_mask = torch.tensor(
                    [
                        self._is_table_prompt(text_inputs[index].prepared.prompt)
                        for index in indices
                    ],
                    device=logits.device,
                    dtype=torch.bool,
                ).unsqueeze(0)
                packed_tokens = device_timeline.measure(
                    f"{prefix}:prefill_argmax",
                    lambda logits=logits, packed_policy_mask=packed_policy_mask: (
                        select_token_ids(
                            logits.float(),
                            mode=self.token_selection,
                            preferred_token_id=self.math_open_token_id,
                            alternate_preferred_token_id=self.math_slash_token_id,
                            policy_mask=(
                                torch.zeros_like(packed_policy_mask)
                                if self.token_selection
                                == TOKEN_SELECTION_PREFER_MATH_OPEN_ADJUSTERS_COMBINED
                                else packed_policy_mask
                            ),
                            legacy_policy_mask=packed_policy_mask,
                        )
                        if self.token_selection in (
                            TOKEN_SELECTION_SUPPRESS_MATH_OPEN_GREEDY,
                            TOKEN_SELECTION_SUPPRESS_MATH_OPEN_AND_SLASH_GREEDY,
                            TOKEN_SELECTION_PREFER_MATH_OPEN_PROBABILITY_NEAR_TOP,
                            TOKEN_SELECTION_PREFER_MATH_OPEN_ADJUSTERS_COMBINED,
                        )
                        else torch.argmax(logits.float(), dim=-1)
                    ),
                )
                pack_padding = (
                    int(text_route["physical_text_tokens"])
                    - int(text_route["real_text_tokens"])
                )
                for position, member_index in enumerate(indices):
                    member_item = text_inputs[member_index]
                    member_padding = (
                        pack_padding if position == len(indices) - 1 else 0
                    )
                    real_tokens = lengths[member_index]
                    text_routes[member_index] = {
                        **text_route,
                        "real_text_tokens": real_tokens,
                        "physical_text_tokens": real_tokens + member_padding,
                        "padding_text_tokens": member_padding,
                        "useful_token_fraction": (
                            real_tokens / (real_tokens + member_padding)
                        ),
                        "packing": "production_group",
                        "pack_group_id": group.group_id,
                        "text_pack_index": pack_index,
                        "pack_real_text_tokens": int(
                            text_route["real_text_tokens"]
                        ),
                        "pack_physical_text_tokens": int(
                            text_route["physical_text_tokens"]
                        ),
                        "private_cache_slot_index": int(
                            member_item.cache_lease.slot_index
                        ),
                        "private_cache_generation": int(
                            member_item.cache_lease.generation
                        ),
                    }
                    next_tokens[member_index] = packed_tokens[
                        :, position : position + 1
                    ]
                stage_keys = {
                    stage: f"{prefix}:{stage}"
                    for stage in (
                        "text_prefill_input_prep",
                        "text_prefill",
                        "text_kv_redistribute",
                        "prefill_lm_head",
                        "prefill_argmax",
                    )
                }
                text_packs.append(
                    _TextPackTrace(
                        member_indices=indices,
                        route=dict(text_route),
                        stage_keys=stage_keys,
                    )
                )
                self._text_packing_stats.record_pack(
                    members=len(indices),
                    real_tokens=int(text_route["real_text_tokens"]),
                    physical_tokens=int(text_route["physical_text_tokens"]),
                    redistributed_kv_bytes=int(redistributed_bytes),
                )

        self._text_packing_stats.fallback_crops += len(fallback_indices)
        for member_index in fallback_indices:
            item = text_inputs[member_index]
            attention_mask_device = item.moved[1]
            position_ids = item.moved[3]
            text_route = self.text_prefill.route(lengths[member_index])
            prepared_text = device_timeline.measure(
                self._group_stage_key(
                    member_index,
                    "text_prefill_input_prep",
                ),
                lambda item=item,
                attention_mask_device=attention_mask_device,
                position_ids=position_ids,
                text_route=text_route: self.text_prefill.prepare(
                    item.inputs_embeds,
                    attention_mask_device,
                    position_ids,
                    route=text_route,
                ),
            )
            last_hidden_state = device_timeline.measure(
                self._group_stage_key(member_index, "text_prefill"),
                lambda prepared_text=prepared_text, cache=item.cache: (
                    self.text_prefill.run_prepared(prepared_text, cache)
                ),
            )
            logits = device_timeline.measure(
                self._group_stage_key(member_index, "prefill_lm_head"),
                lambda last_hidden_state=last_hidden_state: self.model.lm_head(
                    last_hidden_state
                ),
            )
            prefill_policy_mask = torch.tensor(
                [self._is_table_prompt(item.prepared.prompt)],
                device=logits.device,
                dtype=torch.bool,
            )
            next_token = device_timeline.measure(
                self._group_stage_key(member_index, "prefill_argmax"),
                lambda logits=logits, prefill_policy_mask=prefill_policy_mask: (
                    select_token_ids(
                        logits[:, -1, :].float(),
                        mode=self.token_selection,
                        preferred_token_id=self.math_open_token_id,
                        alternate_preferred_token_id=self.math_slash_token_id,
                        policy_mask=(
                            torch.zeros_like(prefill_policy_mask)
                            if self.token_selection
                            == TOKEN_SELECTION_PREFER_MATH_OPEN_ADJUSTERS_COMBINED
                            else prefill_policy_mask
                        ),
                        legacy_policy_mask=prefill_policy_mask,
                    ).unsqueeze(-1)
                    if self.token_selection in (
                        TOKEN_SELECTION_SUPPRESS_MATH_OPEN_GREEDY,
                        TOKEN_SELECTION_SUPPRESS_MATH_OPEN_AND_SLASH_GREEDY,
                        TOKEN_SELECTION_PREFER_MATH_OPEN_PROBABILITY_NEAR_TOP,
                        TOKEN_SELECTION_PREFER_MATH_OPEN_ADJUSTERS_COMBINED,
                    )
                    else torch.argmax(
                        logits[:, -1, :].float(),
                        dim=-1,
                        keepdim=True,
                    )
                ),
            )
            text_route = {
                **text_route,
                "packing": "fallback_single" if pack_indices else "off",
                "pack_group_id": group.group_id,
                "text_pack_index": None,
                "pack_real_text_tokens": int(text_route["real_text_tokens"]),
                "pack_physical_text_tokens": int(
                    text_route["physical_text_tokens"]
                ),
                "private_cache_slot_index": int(
                    item.cache_lease.slot_index
                ),
                "private_cache_generation": int(
                    item.cache_lease.generation
                ),
            }
            next_tokens[member_index] = next_token
            text_routes[member_index] = text_route
            text_packs.append(
                _TextPackTrace(
                    member_indices=(member_index,),
                    route=dict(text_route),
                    stage_keys={
                        stage: self._group_stage_key(member_index, stage)
                        for stage in (
                            "text_prefill_input_prep",
                            "text_prefill",
                            "prefill_lm_head",
                            "prefill_argmax",
                        )
                    },
                )
            )

        if any(token is None for token in next_tokens) or any(
            route is None for route in text_routes
        ):
            raise AssertionError("text prefill did not produce every group member")
        members = [
            _InFlightPrefillMember(
                prepared=item.prepared,
                cache=item.cache,
                cache_lease=item.cache_lease,
                rope_deltas=item.rope_deltas,
                next_cache_position=item.next_cache_position,
                next_token=next_tokens[index],
                device_inputs=item.moved,
                vision=item.vision,
                text_prefill=text_routes[index],
                timing_s=item.timing_s,
                input_tokens=int(item.prepared.input_ids.shape[1]),
                projected_image_tokens=item.projected_image_tokens,
            )
            for index, item in enumerate(text_inputs)
        ]

        packed_next_tokens = torch.cat(
            [token.detach().reshape(-1) for token in next_tokens if token is not None],
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
            text_packs=text_packs,
        )

    @torch.inference_mode()
    def _diagnose_prefill_kv_finiteness(
        self,
        member: _InFlightPrefillMember,
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
        for (kind, layer, scope, value_kind, _tensor), count in zip(
            pending, counts
        ):
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
            hashlib.sha256()
            for _start in range(0, digest_limit, digest_chunk_size)
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
                    digest.update(
                        cpu[..., start:end, :].contiguous().numpy().tobytes()
                    )
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
        for member in inflight.members:
            self._diagnose_prefill_kv_finiteness(member)
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
        member_text_stages = {
            "text_token_embedding": "Text token embeddings",
            "image_embed_scatter": "Scatter projected image embeddings",
            "static_cache_alloc": "Acquire private KV cache slot",
        }
        shared_text_stages = {
            "text_prefill_input_prep": "Text bucket preparation",
            "text_prefill": "Text transformer prefill",
            "text_kv_redistribute": "Redistribute packed text KV prefixes",
            "prefill_lm_head": "Prefill LM head",
            "prefill_argmax": "First-token argmax",
        }
        text_device_by_member = [
            {stage: 0.0 for stage in shared_text_stages}
            for _member in inflight.members
        ]
        for pack_index, text_pack in enumerate(inflight.text_packs):
            owner_index = text_pack.member_indices[0]
            pack_request_ids = [
                request_ids[index] for index in text_pack.member_indices
            ]
            for stage, key in text_pack.stage_keys.items():
                span = device_spans[key]
                text_device_by_member[owner_index][stage] += float(
                    span["seconds"]
                )
                if self.timeline is not None:
                    self.timeline.record_span(
                        "Text prefill",
                        shared_text_stages[stage],
                        int(span["start_ns"]),
                        int(span["end_ns"]),
                        flow_id=pack_request_ids[0],
                        flow_ids=pack_request_ids,
                        clock=str(span["clock"]),
                        track="device",
                        lane="prefill",
                        args={
                            "stage": stage,
                            "vision_group_id": inflight.group_id,
                            "text_pack_index": pack_index,
                            "members": len(text_pack.member_indices),
                            "execution": text_pack.route.get("execution"),
                            "bucket": text_pack.route.get("bucket"),
                            "real_tokens": text_pack.route.get(
                                "real_text_tokens"
                            ),
                            "physical_tokens": text_pack.route.get(
                                "physical_text_tokens"
                            ),
                        },
                    )
        results: list[PrefilledRecognition] = []
        for index, (member, first_token) in enumerate(
            zip(inflight.members, first_tokens)
        ):
            prepared = member.prepared
            device_stage_s: dict[str, float] = {}
            for stage in (
                "recognition_inputs_h2d",
                *vision_stages,
                *member_text_stages,
            ):
                key = self._group_stage_key(index, stage)
                device_stage_s[stage] = float(device_spans[key]["seconds"])
            device_stage_s.update(text_device_by_member[index])
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
                for stage, label in (
                    *vision_stages.items(),
                    *member_text_stages.items(),
                ):
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
                    cache_release=member.cache_lease.release,
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
                    input_fingerprints=dict(prepared.input_fingerprints),
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
            "decode_optimization": self.decode_optimization,
            "decode_vocab": dict(self.decode_vocab),
            "token_selection": {
                "mode": self.token_selection,
                "scope": "table_prompt_only",
                "preferred_token_id": self.math_open_token_id,
                "preferred_token_piece": r"\(",
                "alternate_preferred_token_id": self.math_slash_token_id,
                "alternate_preferred_token_piece": "\\",
                "math_close_token_id": self.math_close_token_id,
                "rule": {
                    TOKEN_SELECTION_GREEDY: "ordinary_argmax",
                    TOKEN_SELECTION_SUPPRESS_MATH_OPEN_GREEDY: (
                        "table_prompt_argmax_except_math_open_token_id"
                    ),
                    TOKEN_SELECTION_SUPPRESS_MATH_OPEN_AND_SLASH_GREEDY: (
                        "table_prompt_argmax_except_math_open_and_backslash_token_ids"
                    ),
                    TOKEN_SELECTION_PREFER_MATH_OPEN_TOP2_NON_NESTED: (
                        "prefer_rank2_math_open_only_outside_open_math_region"
                    ),
                    TOKEN_SELECTION_PREFER_MATH_OPEN_TOP2_FIRST_OVERRIDE: (
                        "prefer_rank2_math_open_on_first_override_only"
                    ),
                    TOKEN_SELECTION_PREFER_MATH_OPEN_PROBABILITY_NEAR_TOP: (
                        "prefer_math_open_when_probability_gt_0.10_and_at_least_0.3_top1"
                    ),
                    TOKEN_SELECTION_PREFER_MATH_OPEN_VARIANTS_TOP2_P10: (
                        "at_cell_start_prefer_backslash_or_math_open_in_top2_when_probability_gt_0.10"
                    ),
                    TOKEN_SELECTION_PREFER_MATH_OPEN_ADJUSTERS_COMBINED: (
                        "cell_start_union_of_probability30_math_open_and_top2_p10_backslash_variants"
                    ),
                }[self.token_selection],
            },
            "decode_attention": DECODE_ATTENTION if self.device.type == "npu" else "manual",
            "decode_cache_update": DECODE_CACHE_UPDATE if self.device.type == "npu" else "per_row_copy",
            "cache_length": self.cache_length,
            "max_new_tokens": self.max_new_tokens,
            "recognition_input_fingerprints": (
                self.recognition_input_fingerprints
            ),
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
            "vision_promptfa_align_128": self.vision_promptfa_align_128,
            "vision_sequence_alignment": self.vision_seq_alignment,
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
            "text_packing": {
                "mode": self.text_packing,
                "buckets": list(self.text_pack_buckets),
                "max_members": self.text_pack_max_members,
                "grouping": "within_vision_production_group_best_fit_decreasing",
                "runtime": (
                    self.packed_text_prefill.metadata
                    if self.packed_text_prefill is not None
                    else None
                ),
            },
            "preprocessor": {
                "model_default_min_pixels": self.model_preprocessor_min_pixels,
                "model_default_max_pixels": self.model_preprocessor_max_pixels,
                "min_pixels_override": self.preprocessor_min_pixels_override,
                "max_pixels_override": self.preprocessor_max_pixels_override,
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
            "ready_buffer_capacity": self.ready_buffer_capacity,
            "ready_buffer_low_watermark": self.ready_buffer_low_watermark,
            "private_cache_staging_headroom": (
                self.private_cache_staging_headroom
            ),
            "decode_completion_detection": "queue_depth_one_async_token_copy",
            "private_prefill_cache": self.prefill_cache_pool.stats(),
            "kv_admission": "full_prefill_cache_foreach_copy_into_fixed_slot",
            "text_decode": self.text_decode.metadata,
            "linear_weight_format": self.weight_format,
        }
