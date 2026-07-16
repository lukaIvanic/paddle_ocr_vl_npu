"""Persistent B=1 PaddleOCR-VL recognizer with compiled static decode."""

from __future__ import annotations

import time
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
from schema import Box, RecognitionResult, per_second
from timing import DeviceTimeline, synchronize, timed_wall


class SequentialRecognizer:
    """One persistent model and one compiled B=1 decode graph.

    Vision and text prefill intentionally remain eager. Only the single-token
    static-cache decode module is compiled.
    """

    def __init__(
        self,
        *,
        model: str,
        device: str,
        dtype: str,
        decode_backend: str,
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
        self.cache_length = int(cache_length)
        self.max_new_tokens = int(max_new_tokens)
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
            batch_size=1,
            cache_length=self.cache_length,
            dtype=self.dtype,
            model_dir=self.model_dir,
            linear_weight_format=str(self.weight_format["effective_mode"]),
        )
        synchronize(self.device)
        compile_wrapper_s = time.perf_counter() - started

        warm_cache = self.model.allocate_static_cache(
            batch_size=1,
            cache_length=self.cache_length,
            device=self.device,
            dtype=self.dtype,
            init_mode="zeros",
        )
        warm_input = torch.zeros((1, 1), device=self.device, dtype=torch.int64)
        warm_position = torch.ones((1,), device=self.device, dtype=torch.int64)
        warm_rope = torch.zeros((1, 1), device=self.device, dtype=torch.int64)
        synchronize(self.device)
        started = time.perf_counter()
        self.decode_fn(warm_input, warm_position, warm_rope, *warm_cache.flat_tensors())
        synchronize(self.device)
        compile_first_call_s = time.perf_counter() - started
        del warm_cache, warm_input, warm_position, warm_rope

        self.setup_timing_s = {
            "recognizer_frontend_setup": float(frontend_setup_s),
            "recognizer_model_load": float(model_load_s),
            "decode_weight_format": float(weight_format_s),
            "compile_wrapper": float(compile_wrapper_s),
            "compile_first_call": float(compile_first_call_s),
            "recognizer_runtime_total": float(time.perf_counter() - runtime_started),
        }

    @torch.inference_mode()
    def recognize(
        self,
        *,
        request_id: str,
        layout_order: int,
        label: str,
        prompt: str,
        box: Box,
        crop: Image.Image,
    ) -> RecognitionResult:
        request_started = time.perf_counter()
        timing: dict[str, float] = {}

        started = time.perf_counter()
        pixel_values, image_grid_thw = preprocess_pil_image(crop, self.preprocessor_config)
        input_ids, attention_mask = build_inputs(
            self.tokenizer,
            image_grid_thw,
            prompt,
            merge_size=int(self.preprocessor_config["merge_size"]),
        )
        timing["cpu_image_and_prompt_preprocess"] = time.perf_counter() - started

        min_cache_length = int(input_ids.shape[1]) + max(0, self.max_new_tokens - 1)
        if self.cache_length < min_cache_length:
            raise ValueError(
                f"request {request_id} needs cache_length>={min_cache_length}, "
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
            projected = image_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            image_mask = (input_ids_device == self.model.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
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
            lambda: torch.argmax(logits[:, -1, :].float(), dim=-1, keepdim=True),
        )
        device_stage_s = timeline.resolve()
        timing["eager_vision_and_text_prefill_wall"] = time.perf_counter() - prefill_started

        started = time.perf_counter()
        first_token = int(next_token.detach().cpu().item())
        timing["first_token_d2h"] = time.perf_counter() - started
        timing["time_to_first_token"] = time.perf_counter() - request_started

        decode = self._decode_to_eos(
            next_token=next_token,
            cache_position=torch.full(
                (1,),
                int(input_ids_device.shape[1]),
                device=self.device,
                dtype=torch.int64,
            ),
            rope_deltas=rope_deltas,
            flat_cache=cache.flat_tensors(),
            first_token=first_token,
        )
        timing.update(decode["timing_s"])

        started = time.perf_counter()
        text = self.tokenizer.decode(decode["token_ids"], skip_special_tokens=True)
        timing["detokenize"] = time.perf_counter() - started
        timing["request_total"] = time.perf_counter() - request_started

        generated_tokens = len(decode["token_ids"])
        decode_tokens = max(0, generated_tokens - 1)
        return RecognitionResult(
            request_id=request_id,
            layout_order=int(layout_order),
            label=label,
            prompt=prompt,
            box=box,
            crop_size=tuple(int(value) for value in crop.size),
            text=text,
            token_ids=decode["token_ids"],
            stop_reason=decode["stop_reason"],
            input_tokens=int(input_ids.shape[1]),
            projected_image_tokens=int(image_embeds.shape[0]),
            generated_tokens_including_eos=generated_tokens,
            decode_tokens_after_prefill_including_eos=decode_tokens,
            decode_calls_executed=int(decode["decode_calls"]),
            timing_s={key: float(value) for key, value in timing.items()},
            device_stage_s={key: float(value) for key, value in device_stage_s.items()},
            rates={
                "decode_effective_tok_per_s": per_second(decode_tokens, timing["compiled_decode_wall"]),
                "decode_executed_calls_per_s": per_second(decode["decode_calls"], timing["compiled_decode_wall"]),
                "request_output_tok_per_s": per_second(generated_tokens, timing["request_total"]),
            },
        )

    def _decode_to_eos(
        self,
        *,
        next_token: torch.Tensor,
        cache_position: torch.Tensor,
        rope_deltas: torch.Tensor,
        flat_cache: tuple[torch.Tensor, ...],
        first_token: int,
    ) -> dict[str, Any]:
        eos_token_id = int(self.model.config.eos_token_id)
        generated = [next_token]
        decode_calls = 0
        eos_detected = first_token == eos_token_id

        synchronize(self.device)
        started = time.perf_counter()
        if not eos_detected and self.device.type == "npu":
            import torch_npu

            max_calls = max(0, self.max_new_tokens - 1)
            flags = torch.zeros((max_calls,), dtype=torch.bool, pin_memory=True)
            copy_stream = torch_npu.npu.Stream(device=self.device)
            pending_event = None
            pending_step = None
            for step in range(max_calls):
                logits = self.decode_fn(next_token, cache_position, rope_deltas, *flat_cache)
                next_token = torch.argmax(logits[:, -1, :].float(), dim=-1, keepdim=True)
                generated.append(next_token)
                cache_position = cache_position + 1
                decode_calls += 1

                hit_eos = (next_token.reshape(-1) == eos_token_id).all()
                ready = torch_npu.npu.current_stream().record_event()
                done = torch_npu.npu.Event()
                with torch_npu.npu.stream(copy_stream):
                    copy_stream.wait_event(ready)
                    flags[step : step + 1].copy_(hit_eos.reshape(1), non_blocking=True)
                    done.record(copy_stream)
                if pending_event is not None and pending_step is not None:
                    pending_event.synchronize()
                    if bool(flags[pending_step].item()):
                        eos_detected = True
                        break
                pending_event = done
                pending_step = step
            if not eos_detected and pending_event is not None and pending_step is not None:
                pending_event.synchronize()
                eos_detected = bool(flags[pending_step].item())
        elif not eos_detected:
            for _ in range(max(0, self.max_new_tokens - 1)):
                logits = self.decode_fn(next_token, cache_position, rope_deltas, *flat_cache)
                next_token = torch.argmax(logits[:, -1, :].float(), dim=-1, keepdim=True)
                generated.append(next_token)
                cache_position = cache_position + 1
                decode_calls += 1
                if int(next_token.detach().cpu().item()) == eos_token_id:
                    eos_detected = True
                    break
        synchronize(self.device)
        decode_wall_s = time.perf_counter() - started

        started = time.perf_counter()
        all_ids = [int(value) for value in torch.cat(generated, dim=1)[0].detach().cpu().tolist()]
        if eos_token_id in all_ids:
            all_ids = all_ids[: all_ids.index(eos_token_id) + 1]
            eos_detected = True
        d2h_trim_s = time.perf_counter() - started
        return {
            "token_ids": all_ids,
            "decode_calls": int(decode_calls),
            "stop_reason": "eos" if eos_detected else "length",
            "timing_s": {
                "compiled_decode_wall": float(decode_wall_s),
                "decode_tokens_d2h_and_trim": float(d2h_trim_s),
            },
        }

    def configuration(self) -> dict[str, Any]:
        return {
            "recognizer_model": str(self.model_dir),
            "device": str(self.device),
            "dtype": str(self.dtype),
            "decode_backend": self.decode_backend,
            "decode_attention": DECODE_ATTENTION if self.device.type == "npu" else "manual",
            "decode_cache_update": DECODE_CACHE_UPDATE if self.device.type == "npu" else "per_row_copy",
            "cache_length": self.cache_length,
            "max_new_tokens": self.max_new_tokens,
            "batch_size": 1,
            "vision_prefill": "eager",
            "text_prefill": "eager",
            "decode": "compiled_static_b1" if self.decode_backend != "raw_eager" else "eager_static_b1",
            "compile": self.compile_metadata,
            "linear_weight_format": self.weight_format,
        }
