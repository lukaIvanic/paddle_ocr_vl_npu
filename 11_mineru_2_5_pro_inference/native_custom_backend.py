"""Official MinerU client adapters for the local MinerU model implementation.

The correctness adapter deliberately exposes the Hugging Face ``generate``
contract so the unmodified official ``TransformersVlmClient`` owns request
rendering, batching, decoding, and output filtering.  The eager client is the
first custom serving boundary: it preserves the same official client protocol
but invokes ``generate_ids`` directly.
"""

from __future__ import annotations

from typing import Any

import torch


class LocalMinerUGenerateAdapter:
    """Expose the local model through the subset of the HF generation API used by MinerU."""

    def __init__(self, model: Any) -> None:
        self.local_model = model
        self.config = model.config

    @property
    def device(self) -> torch.device:
        return self.local_model.device

    @property
    def dtype(self) -> torch.dtype:
        return self.local_model.dtype

    @torch.inference_mode()
    def generate(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        max_new_tokens: int | None = None,
        max_length: int | None = None,
        **_: Any,
    ) -> torch.Tensor:
        if max_new_tokens is None:
            if max_length is None:
                raise ValueError("generate requires max_new_tokens or max_length")
            max_new_tokens = max(1, int(max_length) - int(input_ids.shape[1]))
        generated = self.local_model.generate_ids(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            max_new_tokens=int(max_new_tokens),
            eos_token_id=int(self.config.eos_token_id),
            pad_token_id=int(self.config.pad_token_id),
        )
        return torch.cat((input_ids, generated), dim=1)


def make_local_eager_vlm_client(
    model: Any,
    processor: Any,
    *,
    batch_size: int,
    system_prompt: str,
    allow_truncated_content: bool,
):
    """Build a direct eager client while retaining the official client protocol."""

    from mineru_vl_utils.vlm_client.transformers_client import TransformersVlmClient

    class LocalEagerMinerUVlmClient(TransformersVlmClient):
        def _predict_one_batch(
            self,
            image_objs,
            chat_prompts,
            sampling_params,
            **kwargs,
        ):
            # The first custom lane is intentionally B1.  This keeps the
            # generation state contract exact while the official page client
            # still owns layout/crop orchestration and post-processing.
            if len(chat_prompts) != 1:
                raise ValueError("local eager MinerU client currently requires batch_size=1")
            actual_images = [image for image in image_objs if image is not None]
            inputs = self.processor(
                text=chat_prompts,
                images=actual_images or None,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(device=model.device, dtype=model.dtype)
            params = self.build_sampling_params(sampling_params)
            max_new_tokens = params.max_new_tokens
            if max_new_tokens is None:
                max_new_tokens = max(1, int(self.model_max_length) - int(inputs.input_ids.shape[1]))
            generated = model.generate_ids(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pixel_values=getattr(inputs, "pixel_values", None),
                image_grid_thw=getattr(inputs, "image_grid_thw", None),
                max_new_tokens=int(max_new_tokens),
                eos_token_id=int(model.config.eos_token_id),
                pad_token_id=int(model.config.pad_token_id),
            )
            token_ids = generated.cpu().tolist()
            token_ids = [
                [token_id for token_id in row if token_id not in self.skip_token_ids]
                for row in token_ids
            ]
            return self.processor.batch_decode(
                token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

    adapter = LocalMinerUGenerateAdapter(model)
    return LocalEagerMinerUVlmClient(
        model=adapter,
        processor=processor,
        system_prompt=system_prompt,
        allow_truncated_content=allow_truncated_content,
        batch_size=int(batch_size),
        use_tqdm=False,
    )
