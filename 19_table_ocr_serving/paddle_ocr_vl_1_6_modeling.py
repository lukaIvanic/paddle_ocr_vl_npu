#!/usr/bin/env python3
"""PaddleOCR-VL checkpoint loading and persistent serving-stage composition.

Vision prefill, text prefill, and text decode own their model math and runtime
policy. This module only connects those stages into one conditional-generation
model. HTTP and a future Python interface share these same stages.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch import nn

from _support.model.config import PaddleOCRVLConfig
from text_prefill_and_decode import LocalPaddleOCRVLStaticCache, TextDecodeRuntime
from text_prefill_and_decode import PaddleOCRRotaryEmbedding, PaddleOCRTextModel, TextPrefillRuntime
from vision_prefill import PaddleOCRProjector, PaddleOCRVisionModel, PaddleOCRVisionRotaryEmbedding, VisionPrefillRuntime


# Model composition, loading, and stage assembly


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
        model_dir: str | Path,
        *,
        dtype: torch.dtype | None = torch.float16,
        device: str | torch.device | None = None,
    ) -> "LocalPaddleOCRVLForConditionalGeneration":
        model_dir = Path(model_dir).expanduser()
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

    def make_inference_stages(
        self,
        *,
        vision_cache_root: Path,
        text_cache_root: Path,
        decode_cache_root: Path,
        batch_size: int,
        cache_length: int,
        device: torch.device,
        model_dir: Path,
        eager: bool = False,
        setup_progress: Callable[[str, str, float | None], None] | None = None,
    ) -> PaddleOCRVLInferenceStages:
        """Assemble vision prefill, text prefill, and text decode runtimes.

        Stage modules own the model math and their eager/compiled execution
        policy. This connector only establishes the model-level ordering and
        shared runtime configuration.
        """

        from _support.utils.timing import synchronize

        setup_timing_s: dict[str, float] = {}

        def progress(stage: str, status: str, elapsed_s: float | None = None) -> None:
            if setup_progress is not None:
                setup_progress(stage, status, elapsed_s)

        synchronize(device)
        started = time.perf_counter()
        progress("vision_runtime", "start")
        vision_prefill = VisionPrefillRuntime(
            self,
            cache_root=vision_cache_root,
            device=device,
            model_dir=model_dir,
            eager=eager,
        )
        synchronize(device)
        setup_timing_s["vision_runtime_setup"] = time.perf_counter() - started
        progress(
            "vision_runtime",
            "done",
            setup_timing_s["vision_runtime_setup"],
        )

        synchronize(device)
        started = time.perf_counter()
        progress("text_prefill_runtime", "start")
        text_prefill = TextPrefillRuntime(
            self,
            cache_root=text_cache_root,
            cache_length=cache_length,
            device=device,
            model_dir=model_dir,
            eager=eager,
        )
        synchronize(device)
        setup_timing_s["text_runtime_setup"] = time.perf_counter() - started
        progress(
            "text_prefill_runtime",
            "done",
            setup_timing_s["text_runtime_setup"],
        )

        started = time.perf_counter()
        progress("text_decode_runtime", "start")
        text_decode = TextDecodeRuntime(
            self,
            device=device,
            cache_root=decode_cache_root,
            batch_size=batch_size,
            cache_length=cache_length,
            model_dir=model_dir,
            eager=eager,
        )
        setup_timing_s.update(text_decode.setup_timing_s)
        progress(
            "text_decode_runtime",
            "done",
            setup_timing_s["compile_wrapper"]
            + setup_timing_s["compile_first_call"],
        )
        return PaddleOCRVLInferenceStages(
            vision_prefill=vision_prefill,
            text_prefill=text_prefill,
            text_decode=text_decode,
            setup_timing_s=setup_timing_s,
        )

    def allocate_static_cache(
        self,
        *,
        batch_size: int,
        cache_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> LocalPaddleOCRVLStaticCache:
        return LocalPaddleOCRVLStaticCache.allocate(
            self.config.text_config,
            batch_size=batch_size,
            cache_length=cache_length,
            device=device,
            dtype=dtype,
        )

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
                for _ in range(image_nums):
                    ed = input_tokens.index(image_token_id, st)
                    t, h, w = image_grid_thw[image_index]
                    image_index += 1
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

    def _reset_rope_buffers(self) -> None:
        for module in self.modules():
            if isinstance(module, (PaddleOCRRotaryEmbedding, PaddleOCRVisionRotaryEmbedding)):
                module.reset_inv_freq(device=module.inv_freq.device)


@dataclass(frozen=True)
class PaddleOCRVLInferenceStages:
    """The three persistent model-stage runtimes used by the page engine."""

    vision_prefill: VisionPrefillRuntime
    text_prefill: TextPrefillRuntime
    text_decode: TextDecodeRuntime
    setup_timing_s: dict[str, float]
