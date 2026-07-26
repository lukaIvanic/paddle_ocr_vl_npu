#!/usr/bin/env python3
"""Top-level PaddleOCR-VL composition, checkpoint loading, and generation.

Vision prefill, text prefill, and text decode own their model math and runtime
policy. This module only connects those stages into one conditional-generation
model and provides the small offline-reference generation surface.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from .config import PaddleOCRVLConfig
from .text_decode import (
    LocalPaddleOCRVLStaticCache,
    TextDecodeRuntime,
    run_text_decode_transformer,
)
from .text_prefill import (
    PaddleOCRRotaryEmbedding,
    PaddleOCRTextModel,
    TextPrefillRuntime,
)
from .vision_prefill import (
    PaddleOCRProjector,
    PaddleOCRVisionModel,
    PaddleOCRVisionRotaryEmbedding,
    VisionPrefillRuntime,
)


def _resolve_model_dir(model_id_or_path: str | Path) -> Path:
    path = Path(model_id_or_path).expanduser()
    if path.exists():
        return path
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover - dependency/environment guard
        raise RuntimeError("Pass a local model directory or install huggingface_hub.") from exc
    return Path(
        snapshot_download(
            str(model_id_or_path),
            allow_patterns=[
                "config.json",
                "model.safetensors",
                "tokenizer.json",
                "tokenizer.model",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "added_tokens.json",
                "preprocessor_config.json",
                "processor_config.json",
                "chat_template.jinja",
                "generation_config.json",
            ],
        )
    )


@dataclass
class LocalModelOutput:
    logits: torch.Tensor
    past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None
    rope_deltas: torch.Tensor | None = None


@dataclass
class LocalStaticModelOutput:
    logits: torch.Tensor
    cache: "LocalPaddleOCRVLStaticCache"
    rope_deltas: torch.Tensor
    next_cache_position: torch.Tensor


@dataclass(frozen=True)
class PaddleOCRVLInferenceStages:
    """The three persistent model-stage runtimes used by the page engine."""

    vision_prefill: VisionPrefillRuntime
    text_prefill: TextPrefillRuntime
    text_decode: TextDecodeRuntime
    setup_timing_s: dict[str, float]


class LocalPaddleOCRVLForConditionalGeneration(nn.Module):
    def __init__(self, config: PaddleOCRVLConfig):
        super().__init__()
        self.config = config
        self.visual = PaddleOCRVisionModel(config.vision_config)
        self.mlp_AR = PaddleOCRProjector(config)
        self.model = PaddleOCRTextModel(config.text_config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.rope_deltas: torch.Tensor | None = None

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str | Path = "PaddlePaddle/PaddleOCR-VL-1.6",
        *,
        dtype: torch.dtype | None = torch.float16,
        device: str | torch.device | None = None,
    ) -> "LocalPaddleOCRVLForConditionalGeneration":
        model_dir = _resolve_model_dir(model_id_or_path)
        config = PaddleOCRVLConfig.from_model_dir(model_dir)
        model = cls(config)
        if dtype is not None:
            model = model.to(dtype=dtype)
        if device is not None:
            model = model.to(device)
        from safetensors.torch import load_file

        state_dict = load_file(model_dir / "model.safetensors", device=str(device or "cpu"))
        ignored = {"visual.vision_model.embeddings.packing_position_embedding.weight"}
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if key not in ignored and not key.startswith("visual.vision_model.head.")
        }
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected checkpoint keys: {unexpected}")
        if missing:
            raise RuntimeError(f"missing checkpoint keys: {missing}")
        model._reset_rope_buffers()
        return model.eval()

    def _reset_rope_buffers(self) -> None:
        for module in self.modules():
            if isinstance(module, (PaddleOCRRotaryEmbedding, PaddleOCRVisionRotaryEmbedding)):
                module.reset_inv_freq(device=module.inv_freq.device)

    def get_image_features(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        pixel_values = pixel_values.type(self.visual.dtype).unsqueeze(0)
        cu_seqlens = torch.repeat_interleave(
            image_grid_thw[:, 1] * image_grid_thw[:, 2],
            image_grid_thw[:, 0],
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        image_embeds = self.visual(pixel_values=pixel_values, image_grid_thw=image_grid_thw, cu_seqlens=cu_seqlens)
        return self.mlp_AR(image_embeds, image_grid_thw)

    def get_rope_index(
        self,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        vision_start_token_id = self.config.vision_start_token_id
        if image_grid_thw is not None:
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            position_ids = torch.ones(3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device)
            mrope_position_deltas = []
            image_index = 0
            for batch_idx, sample_input_ids in enumerate(input_ids):
                visible_input_ids = sample_input_ids[attention_mask[batch_idx].to(sample_input_ids.device) == 1]
                vision_start_indices = torch.argwhere(visible_input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = visible_input_ids[vision_start_indices + 1]
                image_nums = int((vision_tokens == image_token_id).sum().item())
                input_tokens = visible_input_ids.tolist()
                llm_pos_ids_list = []
                st = 0
                remain_images = image_nums
                for _ in range(image_nums):
                    ed = input_tokens.index(image_token_id, st) if remain_images > 0 else len(input_tokens) + 1
                    t, h, w = image_grid_thw[image_index]
                    image_index += 1
                    remain_images -= 1
                    llm_grid_t = int(t.item())
                    llm_grid_h = int(h.item()) // spatial_merge_size
                    llm_grid_w = int(w.item()) // spatial_merge_size
                    text_len = ed - st
                    st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w
                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[:, batch_idx, attention_mask[batch_idx] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(input_ids[batch_idx]))
            return position_ids, torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0)[0].max(-1, keepdim=True)[0]
            return position_ids, max_position_ids + 1 - attention_mask.shape[-1]
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).view(1, 1, -1).expand(3, input_ids.shape[0], -1)
        return position_ids, torch.zeros([input_ids.shape[0], 1], device=input_ids.device, dtype=input_ids.dtype)

    def build_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
    ) -> torch.Tensor:
        inputs_embeds = self.model.embed_tokens(input_ids)
        if pixel_values is None:
            return inputs_embeds
        if image_grid_thw is None:
            raise ValueError("image_grid_thw is required when pixel_values is provided")
        image_embeds = self.get_image_features(pixel_values, image_grid_thw)
        image_embeds = image_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        if inputs_embeds[image_mask].numel() != image_embeds.numel():
            raise ValueError(
                "image features and image tokens do not match: "
                f"tokens={int((input_ids == self.config.image_token_id).sum().item())} "
                f"features={int(image_embeds.shape[0])}"
            )
        return inputs_embeds.masked_scatter(image_mask, image_embeds)

    def allocate_static_cache(
        self,
        *,
        batch_size: int,
        cache_length: int,
        device: torch.device,
        dtype: torch.dtype,
        init_mode: str = "zeros",
    ) -> LocalPaddleOCRVLStaticCache:
        return LocalPaddleOCRVLStaticCache.allocate(
            self.config.text_config,
            batch_size=batch_size,
            cache_length=cache_length,
            device=device,
            dtype=dtype,
            init_mode=init_mode,
        )

    def forward_static_prefill(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        *,
        cache_length: int,
        cache: LocalPaddleOCRVLStaticCache | None = None,
        cache_init_mode: str = "zeros",
        logits_to_keep: int = 0,
    ) -> LocalStaticModelOutput:
        inputs_embeds = self.build_inputs_embeds(input_ids, pixel_values, image_grid_thw)
        batch_size, sequence_length, _hidden = inputs_embeds.shape
        if int(sequence_length) > int(cache_length):
            raise ValueError(f"prefill sequence length {sequence_length} exceeds static cache length {cache_length}")
        position_ids, rope_deltas = self.get_rope_index(input_ids, image_grid_thw, attention_mask)
        if cache is None:
            cache = self.allocate_static_cache(
                batch_size=int(batch_size),
                cache_length=int(cache_length),
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
                init_mode=cache_init_mode,
            )
        hidden_states = self.model.forward_prefill_static(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) and logits_to_keep > 0 else slice(None)
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        next_cache_position = torch.full((int(batch_size),), int(sequence_length), device=inputs_embeds.device, dtype=torch.int64)
        self.rope_deltas = rope_deltas
        return LocalStaticModelOutput(
            logits=logits,
            cache=cache,
            rope_deltas=rope_deltas,
            next_cache_position=next_cache_position,
        )

    def forward_static_decode(
        self,
        input_ids: torch.Tensor,
        cache: LocalPaddleOCRVLStaticCache,
        cache_position: torch.Tensor,
        rope_deltas: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        logits_to_keep: int = 0,
    ) -> LocalModelOutput:
        inputs_embeds = self.model.embed_tokens(input_ids)
        hidden_states = run_text_decode_transformer(
            self.model,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            rope_deltas=rope_deltas,
            key_caches=cache.key_caches,
            value_caches=cache.value_caches,
            cache_length=cache.cache_length,
            attention_mask=attention_mask,
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) and logits_to_keep > 0 else slice(None)
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        return LocalModelOutput(logits=logits, rope_deltas=rope_deltas)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
        rope_deltas: torch.Tensor | None = None,
        logits_to_keep: int = 0,
    ) -> LocalModelOutput:
        inputs_embeds = self.build_inputs_embeds(input_ids, pixel_values, image_grid_thw)
        if position_ids is None:
            if past_key_values is None:
                position_ids, rope_deltas = self.get_rope_index(input_ids, image_grid_thw, attention_mask)
                self.rope_deltas = rope_deltas
            else:
                past_length = int(past_key_values[0][0].shape[2])
                batch_size, seq_length, _hidden = inputs_embeds.shape
                delta = rope_deltas if rope_deltas is not None else self.rope_deltas
                if delta is None:
                    raise ValueError("rope_deltas are required for cached decode")
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
                position_ids = position_ids + (past_length + delta.to(inputs_embeds.device)).view(1, batch_size, 1)
        hidden_states, past = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) and logits_to_keep > 0 else slice(None)
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        return LocalModelOutput(logits=logits, past_key_values=past, rope_deltas=rope_deltas if rope_deltas is not None else self.rope_deltas)

    @torch.inference_mode()
    def generate_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
        *,
        max_new_tokens: int = 128,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        eos_token_id = int(self.config.eos_token_id if eos_token_id is None else eos_token_id)
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            use_cache=True,
            logits_to_keep=1,
        )
        past = outputs.past_key_values
        rope_deltas = outputs.rope_deltas
        next_token = torch.argmax(outputs.logits[:, -1, :].float(), dim=-1, keepdim=True)
        generated = [next_token]
        finished = next_token.squeeze(1) == eos_token_id
        current_attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
        for _ in range(max(0, int(max_new_tokens) - 1)):
            if bool(finished.all().item()):
                break
            outputs = self.forward(
                input_ids=next_token,
                attention_mask=current_attention_mask,
                pixel_values=None,
                image_grid_thw=None,
                past_key_values=past,
                use_cache=True,
                rope_deltas=rope_deltas,
                logits_to_keep=1,
            )
            past = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :].float(), dim=-1, keepdim=True)
            next_token = torch.where(finished.view(-1, 1), torch.full_like(next_token, eos_token_id), next_token)
            generated.append(next_token)
            finished |= next_token.squeeze(1) == eos_token_id
            current_attention_mask = torch.cat([current_attention_mask, torch.ones_like(next_token)], dim=1)
        return torch.cat(generated, dim=1)

    @torch.inference_mode()
    def generate_ids_static(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
        *,
        max_new_tokens: int = 128,
        cache_length: int | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        eos_token_id = int(self.config.eos_token_id if eos_token_id is None else eos_token_id)
        prompt_length = int(input_ids.shape[1])
        min_cache_length = prompt_length + max(0, int(max_new_tokens) - 1)
        cache_length = int(
            cache_length
            if cache_length is not None
            else (prompt_length + int(max_new_tokens))
        )
        if cache_length < min_cache_length:
            raise ValueError(
                f"cache_length={cache_length} is too small for prompt length {prompt_length} "
                f"and max_new_tokens={max_new_tokens}; need at least {min_cache_length}"
            )
        outputs = self.forward_static_prefill(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            cache_length=cache_length,
            logits_to_keep=1,
        )
        cache = outputs.cache
        rope_deltas = outputs.rope_deltas
        cache_position = outputs.next_cache_position
        next_token = torch.argmax(outputs.logits[:, -1, :].float(), dim=-1, keepdim=True)
        generated = [next_token]
        finished = next_token.squeeze(1) == eos_token_id
        for _ in range(max(0, int(max_new_tokens) - 1)):
            if bool(finished.all().item()):
                break
            outputs_decode = self.forward_static_decode(
                input_ids=next_token,
                cache=cache,
                cache_position=cache_position,
                rope_deltas=rope_deltas,
                logits_to_keep=1,
            )
            next_token = torch.argmax(outputs_decode.logits[:, -1, :].float(), dim=-1, keepdim=True)
            next_token = torch.where(finished.view(-1, 1), torch.full_like(next_token, eos_token_id), next_token)
            generated.append(next_token)
            finished |= next_token.squeeze(1) == eos_token_id
            cache_position = cache_position + 1
        return torch.cat(generated, dim=1)

    def make_inference_stages(
        self,
        *,
        vision_backend: str,
        vision_attention: str,
        vision_buckets: str | Iterable[int],
        vision_cache_root: Path,
        vision_padding: str,
        vision_seq_alignment: int,
        text_backend: str,
        text_buckets: str | Iterable[int],
        text_cache_root: Path,
        text_padding: str,
        decode_backend: str,
        decode_optimization: str,
        decode_cache_root: Path,
        batch_size: int,
        cache_length: int,
        device: torch.device,
        dtype: torch.dtype,
        model_dir: Path,
        linear_weight_format: str,
    ) -> PaddleOCRVLInferenceStages:
        """Assemble vision prefill, text prefill, and text decode runtimes.

        Stage modules own the model math and their eager/compiled execution
        policy. This connector only establishes the model-level ordering and
        shared runtime configuration.
        """

        from utils.timing import synchronize

        setup_timing_s: dict[str, float] = {}

        synchronize(device)
        started = time.perf_counter()
        vision_prefill = VisionPrefillRuntime(
            self,
            backend=vision_backend,
            attention_impl=vision_attention,
            buckets=vision_buckets,
            cache_root=vision_cache_root,
            device=device,
            dtype=dtype,
            model_dir=model_dir,
            padding=vision_padding,
            seq_alignment=vision_seq_alignment,
        )
        synchronize(device)
        setup_timing_s["vision_runtime_setup"] = time.perf_counter() - started

        synchronize(device)
        started = time.perf_counter()
        text_prefill = TextPrefillRuntime(
            self,
            backend=text_backend,
            buckets=text_buckets,
            cache_root=text_cache_root,
            cache_length=cache_length,
            device=device,
            dtype=dtype,
            model_dir=model_dir,
            linear_weight_format=linear_weight_format,
            padding=text_padding,
        )
        synchronize(device)
        setup_timing_s["text_runtime_setup"] = time.perf_counter() - started

        started = time.perf_counter()
        text_decode = TextDecodeRuntime(
            self,
            backend=decode_backend,
            optimization=decode_optimization,
            device=device,
            cache_root=decode_cache_root,
            batch_size=batch_size,
            cache_length=cache_length,
            dtype=dtype,
            model_dir=model_dir,
            linear_weight_format=linear_weight_format,
        )
        setup_timing_s.update(text_decode.setup_timing_s)
        return PaddleOCRVLInferenceStages(
            vision_prefill=vision_prefill,
            text_prefill=text_prefill,
            text_decode=text_decode,
            setup_timing_s=setup_timing_s,
        )
