"""Persistent PaddleOCR-VL runtime with sequential prefill and batched decode."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from tokenizers import Tokenizer

from continuous_decode import (
    ContinuousDecodeScheduler,
    DecodeArena,
    ReadyDecodeRequest,
)
from local_modeling_paddleocr_vl import (
    DECODE_ATTENTION,
    DECODE_CACHE_UPDATE,
    LocalPaddleOCRVLForConditionalGeneration,
    LocalPaddleOCRVLStaticCache,
    _resolve_model_dir,
    cast_decode_linear_weights_to_nz,
)
from probe_static_compile import compile_decode_module
from run_local_recognition import (
    build_inputs,
    configure_npu_jit_compile,
    load_preprocessor_config,
    parse_dtype,
    preprocess_pil_image,
    resolve_device,
)
from schema import Box, ContinuousDecodeResult, RecognitionResult, per_second
from timing import DeviceTimeline, synchronize, timed_wall


@dataclass(frozen=True)
class RecognitionInput:
    request_id: str
    layout_order: int
    label: str
    prompt: str
    box: Box
    crop: Image.Image


@dataclass
class ReadyRecognition:
    request: RecognitionInput
    cache: LocalPaddleOCRVLStaticCache
    rope_deltas: torch.Tensor
    next_cache_position: torch.Tensor
    next_token: torch.Tensor
    first_token: int
    input_tokens: int
    projected_image_tokens: int
    timing_s: dict[str, float]
    device_stage_s: dict[str, float]
    request_started: float
    prefill_finished: float


class ContinuousRecognizer:
    """One persistent model with sequential eager prefill and continuous decode.

    Every real crop is prefilled independently without image or prompt padding.
    A fixed compiled decode arena keeps its tensor shapes stable while ready KV
    prefixes replace finished requests between autoregressive iterations.
    """

    def __init__(
        self,
        *,
        model: str,
        device: str,
        dtype: str,
        decode_backend: str,
        batch_size: int,
        cache_length: int,
        max_new_tokens: int,
        torchair_cache_dir: Path,
        npu_jit_compile: str = "off",
    ):
        runtime_started = time.perf_counter()
        self.model_dir = _resolve_model_dir(model)
        self.device = resolve_device(device)
        self.dtype = parse_dtype(dtype, self.device)
        self.decode_backend = decode_backend
        self.batch_size = int(batch_size)
        self.cache_length = int(cache_length)
        self.max_new_tokens = int(max_new_tokens)
        if self.batch_size <= 0 or self.batch_size & (self.batch_size - 1):
            raise ValueError("batch_size must be a positive power of two")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.cache_length <= self.max_new_tokens:
            raise ValueError("cache_length must leave room for both prompt and generated tokens")

        configure_npu_jit_compile(npu_jit_compile, self.device)
        self.preprocessor_config = load_preprocessor_config(self.model_dir)
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

        flat_decode = self.model.make_flat_static_decode_module().eval()
        synchronize(self.device)
        started = time.perf_counter()
        self.decode_fn, self.compile_metadata = compile_decode_module(
            flat_decode,
            backend_name=self.decode_backend,
            device=self.device,
            cache_root=torchair_cache_dir,
            batch_size=self.batch_size,
            cache_length=self.cache_length,
            dtype=self.dtype,
            model_dir=self.model_dir,
            linear_weight_format=str(self.weight_format["effective_mode"]),
        )
        synchronize(self.device)
        compile_wrapper_s = time.perf_counter() - started

        warm_cache = self.model.allocate_static_cache(
            batch_size=self.batch_size,
            cache_length=self.cache_length,
            device=self.device,
            dtype=self.dtype,
            init_mode="zeros",
        )
        warm_input = torch.zeros((self.batch_size, 1), device=self.device, dtype=torch.int64)
        warm_position = torch.ones((self.batch_size,), device=self.device, dtype=torch.int64)
        warm_rope = torch.zeros((self.batch_size, 1), device=self.device, dtype=torch.int64)
        synchronize(self.device)
        started = time.perf_counter()
        self.decode_fn(warm_input, warm_position, warm_rope, *warm_cache.flat_tensors())
        synchronize(self.device)
        compile_first_call_s = time.perf_counter() - started
        del warm_input, warm_position, warm_rope

        started = time.perf_counter()
        self.decode_arena = DecodeArena(
            cache=warm_cache,
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
            "compile_wrapper": float(compile_wrapper_s),
            "compile_first_call": float(compile_first_call_s),
            "decode_control_setup": float(decode_control_setup_s),
            "recognizer_runtime_total": float(time.perf_counter() - runtime_started),
        }

    @torch.inference_mode()
    def recognize_many(
        self,
        requests: list[RecognitionInput],
        *,
        schedule_id: str,
    ) -> tuple[list[RecognitionResult], ContinuousDecodeResult]:
        if not requests:
            raise ValueError("recognize_many requires at least one request")

        ready_states = [self._prefill(request) for request in requests]
        decode_ready = [
            ReadyDecodeRequest(
                request_id=state.request.request_id,
                payload=state,
                cache=state.cache,
                rope_deltas=state.rope_deltas,
                cache_position=state.next_cache_position,
                first_token_tensor=state.next_token,
                first_token=state.first_token,
                prompt_length=state.input_tokens,
            )
            for state in ready_states
        ]
        decoded = self.decode_scheduler.run(decode_ready)
        decode_wall_s = decoded.timing_s["continuous_decode_wall"]

        results: list[RecognitionResult] = []
        for completion in decoded.completions:
            state: ReadyRecognition = completion.ready.payload
            token_ids = completion.token_ids
            started = time.perf_counter()
            text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
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
                    "continuous_decode_wall_shared": float(decode_wall_s),
                    "detokenize": float(detokenize_s),
                    "request_total": float(
                        completion.completed_at - state.request_started + detokenize_s
                    ),
                }
            )
            results.append(
                RecognitionResult(
                    request_id=state.request.request_id,
                    decode_schedule_id=schedule_id,
                    decode_slot_index=completion.slot_index,
                    decode_slot_epoch=completion.slot_epoch,
                    layout_order=int(state.request.layout_order),
                    label=state.request.label,
                    prompt=state.request.prompt,
                    box=state.request.box,
                    crop_size=tuple(int(value) for value in state.request.crop.size),
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
                        "decode_effective_token_contribution_per_s": per_second(
                            effective_decode_tokens,
                            decode_wall_s,
                        ),
                        "request_output_tok_per_s": per_second(
                            generated_tokens,
                            timing["request_total"],
                        ),
                    },
                )
            )

        results.sort(key=lambda result: result.layout_order)
        schedule_result = ContinuousDecodeResult(
            schedule_id=schedule_id,
            batch_size=self.batch_size,
            requests=len(requests),
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
            },
        )
        return results, schedule_result

    @torch.inference_mode()
    def _prefill(self, request: RecognitionInput) -> ReadyRecognition:
        request_started = time.perf_counter()
        timing: dict[str, float] = {}

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
        cu_seqlens = F.pad(
            torch.repeat_interleave(
                image_grid_thw[:, 1] * image_grid_thw[:, 2],
                image_grid_thw[:, 0],
            ).cumsum(dim=0, dtype=torch.int32),
            (1, 0),
            value=0,
        )
        vision_model = self.model.visual.vision_model
        vision_hidden = timeline.measure(
            "vision_embeddings",
            lambda: vision_model.embeddings(
                pixel_values_device.unsqueeze(0),
                image_grid_thw=image_grid_thw,
            ),
        )
        vision_hidden = timeline.measure(
            "vision_encoder",
            lambda: vision_model.encoder(
                vision_hidden,
                cu_seqlens=cu_seqlens,
                image_grid_thw=image_grid_thw,
            ),
        )
        image_features = timeline.measure(
            "vision_post_layernorm",
            lambda: vision_model.post_layernorm(vision_hidden),
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
                init_mode="zeros",
            ),
        )
        hidden_states = timeline.measure(
            "text_prefill",
            lambda: self.model.model.forward_prefill_static(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask_device,
                position_ids=position_ids,
                cache=cache,
            ),
        )
        logits = timeline.measure(
            "prefill_lm_head",
            lambda: self.model.lm_head(hidden_states[:, -1:, :]),
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
        timing["eager_vision_and_text_prefill_wall"] = time.perf_counter() - prefill_started

        started = time.perf_counter()
        first_token = int(next_token.detach().cpu().item())
        timing["first_token_d2h"] = time.perf_counter() - started
        prefill_finished = time.perf_counter()
        timing["time_to_first_token"] = prefill_finished - request_started
        timing["prefill_request_total"] = prefill_finished - request_started
        return ReadyRecognition(
            request=request,
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
            "vision_prefill": "eager_sequential_no_padding",
            "text_prefill": "eager_sequential_no_padding",
            "decode": decode_label,
            "decode_schedule": "persistent_slots_iteration_hot_swap",
            "decode_completion_detection": "queue_depth_one_async_token_copy",
            "kv_admission": "copy_valid_prefill_prefix_into_fixed_slot",
            "compile": self.compile_metadata,
            "linear_weight_format": self.weight_format,
        }
