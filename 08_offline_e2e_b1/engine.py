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
from schema import Box, DecodeBatchResult, RecognitionResult, per_second
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


@dataclass
class PaddedDecodeCohort:
    ready: list[ReadyRecognition]
    cache: LocalPaddleOCRVLStaticCache
    rope_deltas: torch.Tensor
    next_cache_position: torch.Tensor
    next_token: torch.Tensor
    initial_finished: torch.Tensor
    padded_items: int


@dataclass
class BatchDecodeOutput:
    token_ids_by_real_item: list[list[int]]
    decode_calls: int
    decode_wall_s: float
    d2h_and_trim_s: float


class SequentialRecognizer:
    """One persistent model with sequential eager prefill and fixed-size decode.

    Every real crop is prefilled independently without image or prompt padding.
    Ready KV states are concatenated into fixed decode cohorts. Short sequences
    are EOS-filled, and the final incomplete cohort receives dummy EOS rows.
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
        del warm_cache, warm_input, warm_position, warm_rope

        started = time.perf_counter()
        self.decode_copy_stream = None
        self.decode_finished_flags = None
        if self.device.type == "npu" and self.max_new_tokens > 1:
            import torch_npu

            self.decode_copy_stream = torch_npu.npu.Stream(device=self.device)
            self.decode_finished_flags = torch.zeros(
                (self.max_new_tokens - 1, self.batch_size),
                dtype=torch.bool,
                pin_memory=True,
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
    def recognize_batch(
        self,
        requests: list[RecognitionInput],
        *,
        batch_id: str,
    ) -> tuple[list[RecognitionResult], DecodeBatchResult]:
        if not requests:
            raise ValueError("recognize_batch requires at least one request")
        if len(requests) > self.batch_size:
            raise ValueError(
                f"batch {batch_id} has {len(requests)} requests, configured batch_size={self.batch_size}"
            )

        ready = [self._prefill(request) for request in requests]
        cohort, cohort_assembly_s = timed_wall(
            self.device,
            lambda: self._make_padded_cohort(ready),
        )
        decode_started = time.perf_counter()
        decoded = self._decode_batch(cohort)
        decode_finished = time.perf_counter()

        results: list[RecognitionResult] = []
        for row_idx, state in enumerate(ready):
            token_ids = decoded.token_ids_by_real_item[row_idx]
            started = time.perf_counter()
            text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            detokenize_s = time.perf_counter() - started
            generated_tokens = len(token_ids)
            effective_decode_tokens = max(0, generated_tokens - 1)
            timing = dict(state.timing_s)
            timing.update(
                {
                    "decode_batch_queue_wait": max(0.0, decode_started - state.prefill_finished),
                    "decode_cohort_assembly_wall_shared": float(cohort_assembly_s),
                    "compiled_decode_batch_wall_shared": float(decoded.decode_wall_s),
                    "decode_batch_d2h_and_trim_shared": float(decoded.d2h_and_trim_s),
                    "detokenize": float(detokenize_s),
                    "request_total": float(decode_finished - state.request_started + detokenize_s),
                }
            )
            results.append(
                RecognitionResult(
                    request_id=state.request.request_id,
                    decode_batch_id=batch_id,
                    layout_order=int(state.request.layout_order),
                    label=state.request.label,
                    prompt=state.request.prompt,
                    box=state.request.box,
                    crop_size=tuple(int(value) for value in state.request.crop.size),
                    text=text,
                    token_ids=token_ids,
                    stop_reason=(
                        "eos"
                        if int(self.model.config.eos_token_id) in token_ids
                        else "length"
                    ),
                    input_tokens=state.input_tokens,
                    projected_image_tokens=state.projected_image_tokens,
                    generated_tokens_including_eos=generated_tokens,
                    decode_tokens_after_prefill_including_eos=effective_decode_tokens,
                    decode_calls_executed=decoded.decode_calls,
                    timing_s=timing,
                    device_stage_s=dict(state.device_stage_s),
                    rates={
                        "decode_effective_token_contribution_per_s": per_second(
                            effective_decode_tokens,
                            decoded.decode_wall_s,
                        ),
                        "request_output_tok_per_s": per_second(
                            generated_tokens,
                            timing["request_total"],
                        ),
                    },
                )
            )

        effective_tokens = sum(
            result.decode_tokens_after_prefill_including_eos
            for result in results
        )
        raw_slots = int(decoded.decode_calls) * self.batch_size
        padded_slots = raw_slots - effective_tokens
        final_padding_slots = cohort.padded_items * int(decoded.decode_calls)
        finished_padding_slots = padded_slots - final_padding_slots
        if finished_padding_slots < 0:
            raise AssertionError(
                "decode padding accounting went negative: "
                f"raw={raw_slots} effective={effective_tokens} final={final_padding_slots}"
            )
        batch_result = DecodeBatchResult(
            batch_id=batch_id,
            batch_size=self.batch_size,
            real_items=len(ready),
            padded_items=cohort.padded_items,
            decode_calls=int(decoded.decode_calls),
            raw_decode_token_slots=raw_slots,
            effective_decode_tokens=effective_tokens,
            padded_decode_token_slots=padded_slots,
            final_cohort_padding_token_slots=final_padding_slots,
            finished_sequence_padding_token_slots=finished_padding_slots,
            timing_s={
                "cohort_assembly_wall": float(cohort_assembly_s),
                "compiled_decode_wall": float(decoded.decode_wall_s),
                "d2h_and_trim": float(decoded.d2h_and_trim_s),
            },
            rates={
                "raw_decode_tok_per_s": per_second(raw_slots, decoded.decode_wall_s),
                "effective_decode_tok_per_s": per_second(effective_tokens, decoded.decode_wall_s),
                "effective_fraction": (
                    float(effective_tokens) / float(raw_slots)
                    if raw_slots > 0
                    else None
                ),
            },
        )
        return results, batch_result

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

    def _make_padded_cohort(
        self,
        ready: list[ReadyRecognition],
    ) -> PaddedDecodeCohort:
        padded_items = self.batch_size - len(ready)
        num_layers = len(ready[0].cache.key_caches)
        dummy_cache = None
        if padded_items:
            dummy_cache = self.model.allocate_static_cache(
                batch_size=padded_items,
                cache_length=self.cache_length,
                device=self.device,
                dtype=self.dtype,
                init_mode="zeros",
            )

        def rows_for(layer_idx: int, *, key: bool) -> list[torch.Tensor]:
            rows = [
                (item.cache.key_caches if key else item.cache.value_caches)[layer_idx]
                for item in ready
            ]
            if dummy_cache is not None:
                rows.append(
                    (dummy_cache.key_caches if key else dummy_cache.value_caches)[layer_idx]
                )
            return rows

        key_caches = tuple(
            torch.cat(rows_for(layer_idx, key=True), dim=0).contiguous()
            for layer_idx in range(num_layers)
        )
        value_caches = tuple(
            torch.cat(rows_for(layer_idx, key=False), dim=0).contiguous()
            for layer_idx in range(num_layers)
        )
        rope_rows = [item.rope_deltas for item in ready]
        position_rows = [item.next_cache_position.reshape(1) for item in ready]
        token_rows = [item.next_token for item in ready]
        if padded_items:
            rope_rows.append(
                torch.zeros(
                    (padded_items, *ready[0].rope_deltas.shape[1:]),
                    device=self.device,
                    dtype=ready[0].rope_deltas.dtype,
                )
            )
            position_rows.append(
                torch.zeros((padded_items,), device=self.device, dtype=torch.int64)
            )
            token_rows.append(
                torch.full(
                    (padded_items, 1),
                    int(self.model.config.eos_token_id),
                    device=self.device,
                    dtype=ready[0].next_token.dtype,
                )
            )
        initial_finished = torch.tensor(
            [
                item.first_token == int(self.model.config.eos_token_id)
                for item in ready
            ]
            + [True] * padded_items,
            device=self.device,
            dtype=torch.bool,
        )
        return PaddedDecodeCohort(
            ready=ready,
            cache=LocalPaddleOCRVLStaticCache(
                key_caches,
                value_caches,
                self.cache_length,
            ),
            rope_deltas=torch.cat(rope_rows, dim=0).contiguous(),
            next_cache_position=torch.cat(position_rows, dim=0).contiguous(),
            next_token=torch.cat(token_rows, dim=0).contiguous(),
            initial_finished=initial_finished,
            padded_items=padded_items,
        )

    def _decode_batch(self, cohort: PaddedDecodeCohort) -> BatchDecodeOutput:
        eos_token_id = int(self.model.config.eos_token_id)
        next_token = cohort.next_token
        cache_position = cohort.next_cache_position
        generated = [next_token]
        finished = cohort.initial_finished.clone()
        decode_calls = 0
        flat_cache = cohort.cache.flat_tensors()

        synchronize(self.device)
        started = time.perf_counter()
        if not bool(finished.detach().cpu().all().item()):
            pending_event = None
            pending_step = None
            for step in range(max(0, self.max_new_tokens - 1)):
                logits = self.decode_fn(
                    next_token,
                    cache_position,
                    cohort.rope_deltas,
                    *flat_cache,
                )
                sampled = torch.argmax(
                    logits[:, -1, :].float(),
                    dim=-1,
                    keepdim=True,
                )
                active_before = ~finished
                next_token = torch.where(
                    active_before.view(-1, 1),
                    sampled,
                    torch.full_like(sampled, eos_token_id),
                )
                new_hits = (next_token.reshape(-1) == eos_token_id) & active_before
                finished = finished | new_hits
                generated.append(next_token)
                cache_position = cache_position + 1
                decode_calls += 1

                if self.device.type == "npu":
                    import torch_npu

                    assert self.decode_finished_flags is not None
                    assert self.decode_copy_stream is not None
                    ready_event = torch_npu.npu.current_stream().record_event()
                    done_event = torch_npu.npu.Event()
                    with torch_npu.npu.stream(self.decode_copy_stream):
                        self.decode_copy_stream.wait_event(ready_event)
                        self.decode_finished_flags[step].copy_(
                            finished,
                            non_blocking=True,
                        )
                        done_event.record(self.decode_copy_stream)
                    if pending_event is not None and pending_step is not None:
                        pending_event.synchronize()
                        if bool(self.decode_finished_flags[pending_step].all().item()):
                            break
                    pending_event = done_event
                    pending_step = step
                elif bool(finished.detach().cpu().all().item()):
                    break
            if pending_event is not None:
                pending_event.synchronize()
        synchronize(self.device)
        decode_wall_s = time.perf_counter() - started

        started = time.perf_counter()
        rows = torch.cat(generated, dim=1).detach().cpu().tolist()
        token_ids_by_real_item: list[list[int]] = []
        for row in rows[: len(cohort.ready)]:
            token_ids = [int(value) for value in row]
            if eos_token_id in token_ids:
                token_ids = token_ids[: token_ids.index(eos_token_id) + 1]
            token_ids_by_real_item.append(token_ids)
        d2h_and_trim_s = time.perf_counter() - started
        return BatchDecodeOutput(
            token_ids_by_real_item=token_ids_by_real_item,
            decode_calls=decode_calls,
            decode_wall_s=float(decode_wall_s),
            d2h_and_trim_s=float(d2h_and_trim_s),
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
            "decode_schedule": "fixed_cohort_eos_padding",
            "final_cohort_padding": "dummy_eos_rows",
            "compile": self.compile_metadata,
            "linear_weight_format": self.weight_format,
        }
