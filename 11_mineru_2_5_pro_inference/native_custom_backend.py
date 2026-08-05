"""Official MinerU client adapters for the local MinerU model implementation.

The correctness adapter deliberately exposes the Hugging Face ``generate``
contract so the unmodified official ``TransformersVlmClient`` owns request
rendering, batching, decoding, and output filtering.  The eager client is the
first custom serving boundary: it preserves the same official client protocol
but invokes ``generate_ids`` directly.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from io import BytesIO
from typing import Any

import torch
from PIL import Image


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


def make_local_compiled_vlm_client(
    model: Any,
    processor: Any,
    compiled_decoder: Any,
    *,
    batch_size: int,
    system_prompt: str,
    allow_truncated_content: bool,
):
    """Build a B1 client with eager prefill and TorchAir static-cache decode."""

    from mineru_vl_utils.vlm_client.transformers_client import TransformersVlmClient

    class LocalCompiledMinerUVlmClient(TransformersVlmClient):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.generation_metrics: list[dict[str, Any]] = []

        def _predict_one_batch(
            self,
            image_objs,
            chat_prompts,
            sampling_params,
            **kwargs,
        ):
            if len(chat_prompts) != 1:
                raise ValueError("local compiled MinerU client currently requires batch_size=1")
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
            if compiled_decoder.cache_length is not None:
                available = int(compiled_decoder.cache_length) - int(inputs.input_ids.shape[1])
                if available <= 0:
                    raise ValueError(
                        "local compiled cache is too short for the prepared prompt: "
                        f"cache_length={compiled_decoder.cache_length} "
                        f"input_tokens={int(inputs.input_ids.shape[1])}"
                    )
                max_new_tokens = min(int(max_new_tokens), available)

            started = time.perf_counter()
            generated, metrics = compiled_decoder.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pixel_values=getattr(inputs, "pixel_values", None),
                image_grid_thw=getattr(inputs, "image_grid_thw", None),
                max_new_tokens=int(max_new_tokens),
                eos_token_id=int(model.config.eos_token_id),
                pad_token_id=int(model.config.pad_token_id),
            )
            record = {
                **metrics,
                "generation_wall_s": float(time.perf_counter() - started),
                "input_tokens": int(inputs.input_ids.shape[1]),
            }
            self.generation_metrics.append(record)

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
    return LocalCompiledMinerUVlmClient(
        model=adapter,
        processor=processor,
        system_prompt=system_prompt,
        allow_truncated_content=allow_truncated_content,
        batch_size=int(batch_size),
        use_tqdm=False,
    )


def make_local_fixed_batch_vlm_client(
    model: Any,
    processor: Any,
    engine: Any,
    *,
    batch_size: int,
    continuous_refill: bool = False,
    system_prompt: str,
    allow_truncated_content: bool,
):
    """Build the compatibility wrapper around the request-owned KV engine."""

    from fixed_batch_engine import PreparedGeneration
    from mineru_vl_utils.vlm_client.base_client import SingleImageType, UnsupportedError
    from mineru_vl_utils.vlm_client.transformers_client import TransformersVlmClient
    from mineru_vl_utils.vlm_client.utils import get_rgb_image, load_resource

    class LocalFixedBatchMinerUVlmClient(TransformersVlmClient):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.generation_metrics: list[dict[str, Any]] = []

        def _prepare_generation(
            self,
            image,
            chat_prompt,
            sampling_param,
        ) -> PreparedGeneration:
            params = self.build_sampling_params(sampling_param)
            inputs = self.processor(
                text=[chat_prompt],
                images=[image] if image is not None else None,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(device=model.device, dtype=model.dtype)
            max_new_tokens = params.max_new_tokens
            if max_new_tokens is None:
                max_new_tokens = max(
                    1,
                    int(self.model_max_length) - int(inputs.input_ids.shape[1]),
                )
            max_new_tokens = min(
                int(max_new_tokens),
                engine.cache_length - int(inputs.input_ids.shape[1]),
            )
            if max_new_tokens <= 0:
                raise ValueError("prepared request leaves no room in the static KV cache")
            return PreparedGeneration(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pixel_values=getattr(inputs, "pixel_values", None),
                image_grid_thw=getattr(inputs, "image_grid_thw", None),
                max_new_tokens=max_new_tokens,
            )

        def _decode_outputs(self, generated, metrics):
            self.generation_metrics.append(metrics)
            rows = [tensor[0].detach().cpu().tolist() for tensor in generated]
            rows = [
                [token_id for token_id in row if token_id not in self.skip_token_ids]
                for row in rows
            ]
            return self.processor.batch_decode(
                rows,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

        def _predict_one_batch(
            self,
            image_objs,
            chat_prompts,
            sampling_params,
            **kwargs,
        ):
            if isinstance(sampling_params, Sequence):
                request_params = list(sampling_params)
            else:
                request_params = [sampling_params] * len(chat_prompts)
            if len(request_params) != len(chat_prompts):
                raise ValueError("sampling-parameter count must match request count")
            prepared = [
                self._prepare_generation(image, chat_prompt, sampling_param)
                for image, chat_prompt, sampling_param in zip(
                    image_objs,
                    chat_prompts,
                    request_params,
                )
            ]

            generated, metrics = engine.generate_many(prepared)
            return self._decode_outputs(generated, metrics)

        def batch_predict(
            self,
            images,
            prompts="",
            sampling_params=None,
            priority=None,
            **kwargs,
        ):
            if not continuous_refill:
                return super().batch_predict(
                    images,
                    prompts=prompts,
                    sampling_params=sampling_params,
                    priority=priority,
                    **kwargs,
                )

            if not isinstance(prompts, str) and len(prompts) != len(images):
                raise ValueError("prompt count must match image count")
            if (
                isinstance(sampling_params, Sequence)
                and len(sampling_params) != len(images)
            ):
                raise ValueError("sampling-parameter count must match image count")
            if isinstance(priority, Sequence) and len(priority) != len(images):
                raise ValueError("priority count must match image count")

            image_objs: list[Image.Image | None] = []
            for image in images:
                if image is None:
                    image_objs.append(None)
                    continue
                if not isinstance(image, SingleImageType):
                    raise UnsupportedError(
                        "continuous MinerU client requires single-image requests"
                    )
                if isinstance(image, str):
                    image = load_resource(image)
                if not isinstance(image, Image.Image):
                    image = Image.open(BytesIO(image))
                image_objs.append(get_rgb_image(image))

            if isinstance(prompts, str):
                chat_prompts = [
                    self.processor.apply_chat_template(
                        self.build_messages(
                            prompts,
                            has_image=image_obj is not None,
                        ),
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for image_obj in image_objs
                ]
            else:
                chat_prompts = [
                    self.processor.apply_chat_template(
                        self.build_messages(
                            prompt,
                            has_image=image_obj is not None,
                        ),
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for prompt, image_obj in zip(prompts, image_objs)
                ]

            if not isinstance(sampling_params, Sequence):
                sampling_params = [sampling_params] * len(images)
            generated, metrics = engine.generate_lazy(
                len(image_objs),
                lambda index: self._prepare_generation(
                    image_objs[index],
                    chat_prompts[index],
                    sampling_params[index],
                ),
            )
            return self._decode_outputs(generated, metrics)

    adapter = LocalMinerUGenerateAdapter(model)
    return LocalFixedBatchMinerUVlmClient(
        model=adapter,
        processor=processor,
        system_prompt=system_prompt,
        allow_truncated_content=allow_truncated_content,
        batch_size=int(batch_size),
        use_tqdm=False,
    )
