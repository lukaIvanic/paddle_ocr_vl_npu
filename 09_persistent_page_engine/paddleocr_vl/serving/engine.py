"""Persistent PaddleOCR-VL runtime with sequential prefill and batched decode."""

from __future__ import annotations

import time
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
    DEFAULT_TEXT_BACKEND,
    DEFAULT_VISION_BACKEND,
    OPTIMIZED_TEXT_BUCKETS,
    OPTIMIZED_VISION_BUCKETS,
    READY_BUFFER_BATCH_MULTIPLIER,
)
from ..model.text_prefill import parse_text_buckets
from .types import ContinuousDecodeResult, RecognitionRequest, RecognitionResult
from utils.timing import DeviceTimeline, synchronize, timed_wall
from utils.metrics import per_second
from ..model.vision_prefill import (
    get_vision_attention_impl,
    get_vision_prompt_fa_layout,
    parse_vision_buckets,
)


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
        vision_buckets: str | Iterable[int] = OPTIMIZED_VISION_BUCKETS,
        vision_torchair_cache_dir: Path | None = None,
        vision_padding: str = "auto",
        text_backend: str = DEFAULT_TEXT_BACKEND,
        text_buckets: str | Iterable[int] = OPTIMIZED_TEXT_BUCKETS,
        text_torchair_cache_dir: Path | None = None,
        text_padding: str = "auto",
        preprocessor_min_pixels: int | None = None,
    ):
        # TorchAir guards tensor dispatch-key sets. Build and warm every graph
        # under the same inference-mode contract used by recognize_stream(),
        # otherwise the first real request invalidates the persistent cache
        # and recompiles the vision, text-prefill, and decode boundaries.
        runtime_started = time.perf_counter()
        import torch_npu  # noqa: F401

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
        self.batch_size = int(batch_size)
        self.cache_length = int(cache_length)
        self.max_new_tokens = int(max_new_tokens)
        self.vision_backend = str(vision_backend)
        self.vision_buckets = parse_vision_buckets(vision_buckets)
        self.vision_padding = str(vision_padding)
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

        started = time.perf_counter()
        self.decode_arena = DecodeArena(
            cache=self.text_decode.warm_cache,
            device=self.device,
            batch_size=self.batch_size,
            eos_token_id=int(self.model.config.eos_token_id),
        )
        self.decode_scheduler = ContinuousDecodeScheduler(
            arena=self.decode_arena,
            decode_fn=self.decode_fn,
            max_new_tokens=self.max_new_tokens,
        )
        decode_control_setup_s = time.perf_counter() - started

        self.setup_timing_s = {
            "recognizer_frontend_setup": float(frontend_setup_s),
            "recognizer_model_load": float(model_load_s),
            "decode_weight_format": float(weight_format_s),
            **self.stages.setup_timing_s,
            "decode_control_setup": float(decode_control_setup_s),
            "recognizer_runtime_total": float(time.perf_counter() - runtime_started),
        }

    @torch.inference_mode()
    def recognize_stream(
        self,
        requests: Iterable[RecognitionRequest],
        *,
        schedule_id: str,
        on_result: Callable[[RecognitionResult], None] | None = None,
    ) -> tuple[list[RecognitionResult], ContinuousDecodeResult]:
        """Prefill and decode a bounded stream of independent crop requests."""

        def ready_stream() -> Iterable[ReadyDecodeRequest]:
            for request in requests:
                state = self._prefill(request)
                cache, rope_deltas, cache_position, first_token_tensor = (
                    state.take_device_state()
                )
                ready = ReadyDecodeRequest(
                    request_id=state.request_id,
                    payload=state,
                    cache=cache,
                    rope_deltas=rope_deltas,
                    cache_position=cache_position,
                    first_token_tensor=first_token_tensor,
                    first_token=state.first_token,
                    prompt_length=state.input_tokens,
                )
                del request
                yield ready

        completion_results: list[RecognitionResult] = []

        def handle_completion(completion: DecodeCompletion) -> None:
            result = self._result_from_completion(
                completion,
                schedule_id=schedule_id,
            )
            completion_results.append(result)
            if on_result is not None:
                on_result(result)

        decoded = self.decode_scheduler.run_stream(
            ready_stream(),
            on_completion=handle_completion,
            ready_buffer_capacity=READY_BUFFER_BATCH_MULTIPLIER * self.batch_size,
            ready_buffer_low_watermark=self.batch_size,
        )
        decode_wall_s = decoded.timing_s["continuous_decode_wall"]

        result_by_id = {result.request_id: result for result in completion_results}
        results = [
            result_by_id[completion.ready.request_id]
            for completion in decoded.completions
        ]
        for result in results:
            result.timing_s["continuous_decode_wall_shared"] = float(decode_wall_s)
            result.rates["decode_effective_token_contribution_per_s"] = per_second(
                result.decode_tokens_after_prefill_including_eos,
                decode_wall_s,
            )

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
        return results, schedule_result

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
                "decode_effective_token_contribution_per_s": None,
                "request_output_tok_per_s": per_second(
                    generated_tokens,
                    timing["request_total"],
                ),
            },
            vision=dict(state.vision),
            text_prefill=dict(state.text_prefill),
        )

    @torch.inference_mode()
    def _prefill(self, request: RecognitionRequest) -> PrefilledRecognition:
        request_started = time.perf_counter()
        timing: dict[str, float] = {}
        crop_size = tuple(int(value) for value in request.crop.size)

        started = time.perf_counter()
        pixel_values, image_grid_thw = preprocess_pil_image(
            request.crop,
            self.preprocessor_config,
        )
        input_ids, attention_mask = build_inputs(
            self.tokenizer,
            image_grid_thw,
            request.prompt,
            merge_size=int(self.preprocessor_config["merge_size"]),
        )
        timing["cpu_image_and_prompt_preprocess"] = time.perf_counter() - started

        min_cache_length = int(input_ids.shape[1]) + max(0, self.max_new_tokens - 1)
        if self.cache_length < min_cache_length:
            raise ValueError(
                f"request {request.request_id} needs cache_length>={min_cache_length}, "
                f"configured cache_length={self.cache_length}"
            )

        started = time.perf_counter()
        position_ids_cpu, rope_deltas_cpu = self.model.get_rope_index(
            input_ids,
            image_grid_thw,
            attention_mask,
        )
        timing["cpu_mrope_index"] = time.perf_counter() - started

        moved, transfer_s = timed_wall(
            self.device,
            lambda: (
                input_ids.to(self.device),
                attention_mask.to(self.device),
                pixel_values.to(device=self.device, dtype=self.model.visual.dtype),
                position_ids_cpu.to(self.device),
                rope_deltas_cpu.to(self.device),
            ),
        )
        input_ids_device, attention_mask_device, pixel_values_device, position_ids, rope_deltas = moved
        timing["recognizer_h2d"] = transfer_s

        timeline = DeviceTimeline(self.device)
        prefill_started = time.perf_counter()
        vision_model = self.model.visual.vision_model
        vision_hidden = timeline.measure(
            "vision_embeddings",
            lambda: vision_model.embeddings(
                pixel_values_device.unsqueeze(0),
                image_grid_thw=image_grid_thw,
            ),
        )
        vision_route = self.vision_prefill.route(int(vision_hidden.shape[0]))
        prepared_vision = timeline.measure(
            "vision_prefill_input_prep",
            lambda: self.vision_prefill.prepare(
                vision_hidden,
                image_grid_thw,
                route=vision_route,
            ),
        )
        image_features = timeline.measure(
            "vision_prefill",
            lambda: self.vision_prefill.run_prepared(prepared_vision),
        )
        image_embeds = timeline.measure(
            "adaptive_mlp_projector",
            lambda: self.model.mlp_AR(image_features, image_grid_thw),
        )
        inputs_embeds = timeline.measure(
            "text_token_embedding",
            lambda: self.model.model.embed_tokens(input_ids_device),
        )

        def scatter_image_embeds() -> torch.Tensor:
            projected = image_embeds.to(
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )
            image_mask = (
                (input_ids_device == self.model.config.image_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
            )
            if inputs_embeds[image_mask].numel() != projected.numel():
                raise ValueError(
                    "image features and image tokens do not match: "
                    f"tokens={int((input_ids_device == self.model.config.image_token_id).sum().item())} "
                    f"features={int(projected.shape[0])}"
                )
            return inputs_embeds.masked_scatter(image_mask, projected)

        inputs_embeds = timeline.measure("image_embed_scatter", scatter_image_embeds)
        cache = timeline.measure(
            "static_cache_alloc",
            lambda: self.model.allocate_static_cache(
                batch_size=1,
                cache_length=self.cache_length,
                device=self.device,
                dtype=inputs_embeds.dtype,
                # Prefill overwrites every real prompt row. Decode admission
                # copies only that valid prefix, so clearing the much larger
                # unused generation tail (notably at cache_length=8192) would
                # be pure memory-bandwidth overhead.
                init_mode="empty",
            ),
        )
        text_route = self.text_prefill.route(int(inputs_embeds.shape[1]))
        prepared_text = timeline.measure(
            "text_prefill_input_prep",
            lambda: self.text_prefill.prepare(
                inputs_embeds,
                attention_mask_device,
                position_ids,
                route=text_route,
            ),
        )
        last_hidden_state = timeline.measure(
            "text_prefill",
            lambda: self.text_prefill.run_prepared(prepared_text, cache),
        )
        logits = timeline.measure(
            "prefill_lm_head",
            lambda: self.model.lm_head(last_hidden_state),
        )
        next_token = timeline.measure(
            "prefill_argmax",
            lambda: torch.argmax(
                logits[:, -1, :].float(),
                dim=-1,
                keepdim=True,
            ),
        )
        device_stage_s = timeline.resolve()
        timing["vision_and_text_prefill_wall"] = time.perf_counter() - prefill_started

        started = time.perf_counter()
        first_token = int(next_token.detach().cpu().item())
        timing["first_token_d2h"] = time.perf_counter() - started
        prefill_finished = time.perf_counter()
        timing["time_to_first_token"] = prefill_finished - request_started
        timing["prefill_request_total"] = prefill_finished - request_started
        return PrefilledRecognition(
            request_id=request.request_id,
            prompt=request.prompt,
            crop_size=crop_size,
            skip_special_tokens=bool(request.skip_special_tokens),
            cache=cache,
            rope_deltas=rope_deltas,
            next_cache_position=torch.full(
                (1,),
                int(input_ids_device.shape[1]),
                device=self.device,
                dtype=torch.int64,
            ),
            next_token=next_token,
            first_token=first_token,
            input_tokens=int(input_ids.shape[1]),
            projected_image_tokens=int(image_embeds.shape[0]),
            vision=vision_route,
            text_prefill=text_route,
            timing_s=timing,
            device_stage_s=device_stage_s,
            request_started=request_started,
            prefill_finished=prefill_finished,
        )

    def configuration(self) -> dict[str, Any]:
        decode_label = (
            f"compiled_static_b{self.batch_size}"
            if self.decode_backend != "raw_eager"
            else f"eager_static_b{self.batch_size}"
        )
        patch_size = int(self.preprocessor_config["patch_size"])
        merge_size = int(self.preprocessor_config["merge_size"])
        min_pixels = int(self.preprocessor_config["min_pixels"])
        vision_attention = get_vision_attention_impl()
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
            "decode": decode_label,
            "decode_schedule": "run_scoped_persistent_slots_iteration_hot_swap",
            "ready_buffer_capacity": READY_BUFFER_BATCH_MULTIPLIER * self.batch_size,
            "ready_buffer_low_watermark": self.batch_size,
            "decode_completion_detection": "queue_depth_one_async_token_copy",
            "kv_admission": "copy_valid_prefill_prefix_into_fixed_slot",
            "text_decode": self.text_decode.metadata,
            "linear_weight_format": self.weight_format,
        }
