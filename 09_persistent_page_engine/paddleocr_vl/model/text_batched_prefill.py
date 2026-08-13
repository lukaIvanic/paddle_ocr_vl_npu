"""Fixed-shape batched text prefill for independent recognition requests."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence

import torch
import torch.nn.functional as F

from .compile_utils import (
    TORCHAIR_EXECUTION_MODE,
    cache_key_part,
    import_torchair,
    short_file_hash,
    torch_npu_version_label,
    torchair_version_label,
)
from .text_decode import LocalPaddleOCRVLStaticCache
from .text_prefill import (
    TextPrefillStage,
    build_causal_mask,
    get_text_softmax_dtype_mode,
    unique_bucket_forward,
)
from utils.timing import synchronize

if TYPE_CHECKING:
    from .modeling import LocalPaddleOCRVLForConditionalGeneration


def batched_text_source_hash() -> str:
    return short_file_hash(Path(__file__).resolve())


class BatchedTextPrefillStage(TextPrefillStage):
    """Text prefill for B independent padded rows with private scratch KV."""

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        last_token_indices: torch.Tensor,
        *flat_cache_tensors: torch.Tensor,
    ) -> torch.Tensor:
        key_caches = tuple(flat_cache_tensors[: self.num_layers])
        value_caches = tuple(flat_cache_tensors[self.num_layers :])
        cache_position = torch.arange(
            inputs_embeds.shape[1],
            device=inputs_embeds.device,
            dtype=torch.int64,
        )
        causal_mask = build_causal_mask(
            inputs_embeds,
            attention_mask,
            cache_position,
        )
        position_embeddings = self.text_model.rotary_emb(
            inputs_embeds,
            position_ids,
        )
        hidden_states = inputs_embeds
        for layer_idx, layer in enumerate(self.text_model.layers):
            residual = hidden_states
            attention_input = layer.input_layernorm(hidden_states)
            attention_output = self._attention(
                layer.self_attn,
                attention_input,
                causal_mask,
                position_embeddings,
                key_caches[layer_idx],
                value_caches[layer_idx],
            )
            hidden_states = layer.apply_blocks(residual, attention_output)
        hidden_states = self.text_model.norm(hidden_states)
        gather_indices = last_token_indices.reshape(-1, 1, 1).expand(
            -1,
            1,
            hidden_states.shape[-1],
        )
        return torch.gather(hidden_states, 1, gather_indices)


@dataclass(frozen=True)
class PreparedBatchedTextPrefill:
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    last_token_indices: torch.Tensor
    real_seq_lengths: tuple[int, ...]
    batch_size: int
    physical_seq_len: int


def batched_text_cache_dir(
    cache_root: Path,
    *,
    batch_size: int,
    sequence_length: int,
    destination_cache_length: int,
    dtype: torch.dtype,
    device: torch.device,
    model_dir: Path,
    linear_weight_format: str,
) -> Path:
    key = "_".join(
        (
            "text_batched_causal",
            f"mode{cache_key_part(TORCHAIR_EXECUTION_MODE)}",
            f"softmax{cache_key_part(get_text_softmax_dtype_mode())}",
            f"bs{int(batch_size)}",
            f"seq{int(sequence_length)}",
            f"destcache{int(destination_cache_length)}",
            f"weights{cache_key_part(linear_weight_format)}",
            f"dtype{cache_key_part(dtype)}",
            f"model{short_file_hash(model_dir / 'config.json')}",
            f"torch{cache_key_part(torch.__version__)}",
            f"torchnpu{torch_npu_version_label(device)}",
            f"torchair{torchair_version_label(device)}",
            f"src{batched_text_source_hash()}",
        )
    )
    return cache_root.expanduser().resolve() / key


class BatchedTextPrefillRuntime:
    """One compiled BxS text graph plus scratch-to-private KV redistribution."""

    def __init__(
        self,
        model: LocalPaddleOCRVLForConditionalGeneration,
        *,
        batch_size: int,
        sequence_length: int,
        cache_root: Path,
        destination_cache_length: int,
        device: torch.device,
        dtype: torch.dtype,
        model_dir: Path,
        linear_weight_format: str,
        require_warm_cache: bool = True,
    ):
        if device.type != "npu":
            raise ValueError("batched text prefill requires an NPU device")
        self.model = model
        self.batch_size = int(batch_size)
        self.sequence_length = int(sequence_length)
        self.destination_cache_length = int(destination_cache_length)
        self.device = device
        self.dtype = dtype
        if self.batch_size <= 0 or self.sequence_length <= 0:
            raise ValueError(
                "batched text dimensions must be positive: "
                f"B={self.batch_size} S={self.sequence_length}"
            )
        if self.sequence_length > self.destination_cache_length:
            raise ValueError(
                f"text bucket {self.sequence_length} exceeds destination cache "
                f"{self.destination_cache_length}"
            )

        cache_dir = batched_text_cache_dir(
            cache_root,
            batch_size=self.batch_size,
            sequence_length=self.sequence_length,
            destination_cache_length=self.destination_cache_length,
            dtype=dtype,
            device=device,
            model_dir=model_dir,
            linear_weight_format=linear_weight_format,
        )
        cache_was_warm = cache_dir.is_dir() and any(cache_dir.rglob("*"))
        if require_warm_cache and not cache_was_warm:
            raise RuntimeError(
                "batched text prefill requires a warm graph: "
                f"shape=b{self.batch_size}_s{self.sequence_length} "
                f"cache={cache_dir}"
            )
        cache_dir.mkdir(parents=True, exist_ok=True)

        torchair, CompilerConfig = import_torchair()
        module = BatchedTextPrefillStage(model).eval()
        entrypoint = unique_bucket_forward(module, self.sequence_length)
        synchronize(device)
        wrapper_started = time.perf_counter()
        self.compiled: Callable[..., torch.Tensor] = (
            torchair.inference.cache_compile(
                entrypoint,
                config=CompilerConfig(),
                dynamic=False,
                cache_dir=str(cache_dir),
                ge_cache=True,
            )
        )
        synchronize(device)
        wrapper_s = time.perf_counter() - wrapper_started
        self.scratch_cache = model.allocate_static_cache(
            batch_size=self.batch_size,
            cache_length=self.sequence_length,
            device=device,
            dtype=dtype,
            init_mode="empty",
        )
        hidden_size = int(model.config.text_config.hidden_size)
        warm_inputs = torch.zeros(
            (self.batch_size, self.sequence_length, hidden_size),
            device=device,
            dtype=dtype,
        )
        warm_mask = torch.ones(
            (self.batch_size, self.sequence_length),
            device=device,
            dtype=torch.int64,
        )
        warm_positions = torch.zeros(
            (3, self.batch_size, self.sequence_length),
            device=device,
            dtype=torch.int64,
        )
        warm_last = torch.full(
            (self.batch_size,),
            self.sequence_length - 1,
            device=device,
            dtype=torch.int64,
        )
        synchronize(device)
        first_call_started = time.perf_counter()
        warm_output = self.compiled(
            warm_inputs,
            warm_mask,
            warm_positions,
            warm_last,
            *self.scratch_cache.flat_tensors(),
        )
        synchronize(device)
        first_call_s = time.perf_counter() - first_call_started
        expected = (self.batch_size, 1, hidden_size)
        if tuple(warm_output.shape) != expected:
            raise RuntimeError(
                "batched text graph returned the wrong shape: "
                f"expected={expected} got={tuple(warm_output.shape)}"
            )
        self.metadata: dict[str, Any] = {
            "enabled": True,
            "boundary": "batched_text_transformer_plus_scratch_kv_writes",
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "destination_cache_length": self.destination_cache_length,
            "cache_dir": str(cache_dir),
            "cache_was_warm": bool(cache_was_warm),
            "compile_wrapper_s": wrapper_s,
            "compile_first_call_s": first_call_s,
        }
        del warm_output, warm_inputs, warm_mask, warm_positions, warm_last

    def route(self, lengths: Sequence[int]) -> dict[str, Any]:
        real_lengths = tuple(int(length) for length in lengths)
        if len(real_lengths) != self.batch_size:
            raise ValueError(
                f"batched text expects {self.batch_size} rows, got "
                f"{len(real_lengths)}"
            )
        if any(length <= 0 or length > self.sequence_length for length in real_lengths):
            raise ValueError(
                f"text lengths do not fit B{self.batch_size}x{self.sequence_length}: "
                f"{real_lengths}"
            )
        real_tokens = sum(real_lengths)
        physical_tokens = self.batch_size * self.sequence_length
        return {
            "execution": "compiled_batched",
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "real_text_tokens": real_tokens,
            "physical_text_tokens": physical_tokens,
            "padding_text_tokens": physical_tokens - real_tokens,
            "useful_token_fraction": real_tokens / physical_tokens,
            "segment_lengths": list(real_lengths),
        }

    def prepare(
        self,
        inputs_embeds: Sequence[torch.Tensor],
        attention_masks: Sequence[torch.Tensor],
        position_ids: Sequence[torch.Tensor],
        *,
        route: dict[str, Any],
    ) -> PreparedBatchedTextPrefill:
        if not (
            len(inputs_embeds)
            == len(attention_masks)
            == len(position_ids)
            == self.batch_size
        ):
            raise ValueError("batched text input member counts do not align")
        lengths = tuple(int(tensor.shape[1]) for tensor in inputs_embeds)
        if lengths != tuple(int(value) for value in route["segment_lengths"]):
            raise ValueError("batched text route lengths do not match inputs")
        padded_embeds = []
        padded_masks = []
        padded_positions = []
        for index, (embeds, mask, positions, length) in enumerate(
            zip(inputs_embeds, attention_masks, position_ids, lengths)
        ):
            if tuple(embeds.shape[:2]) != (1, length) or embeds.ndim != 3:
                raise ValueError(
                    f"batched text member {index} embeddings: {tuple(embeds.shape)}"
                )
            if tuple(mask.shape) != (1, length):
                raise ValueError(
                    f"batched text member {index} mask: {tuple(mask.shape)}"
                )
            if tuple(positions.shape) != (3, 1, length):
                raise ValueError(
                    f"batched text member {index} positions: "
                    f"{tuple(positions.shape)}"
                )
            pad = self.sequence_length - length
            padded_embeds.append(F.pad(embeds, (0, 0, 0, pad)))
            padded_masks.append(F.pad(mask, (0, pad), value=0))
            padded_positions.append(F.pad(positions, (0, pad), value=1))
        return PreparedBatchedTextPrefill(
            inputs_embeds=torch.cat(padded_embeds, dim=0).contiguous(),
            attention_mask=torch.cat(padded_masks, dim=0).contiguous(),
            position_ids=torch.cat(padded_positions, dim=1).contiguous(),
            last_token_indices=torch.tensor(
                [length - 1 for length in lengths],
                device=inputs_embeds[0].device,
                dtype=torch.int64,
            ),
            real_seq_lengths=lengths,
            batch_size=self.batch_size,
            physical_seq_len=self.sequence_length,
        )

    def run_prepared(
        self,
        prepared: PreparedBatchedTextPrefill,
    ) -> torch.Tensor:
        return self.compiled(
            prepared.inputs_embeds,
            prepared.attention_mask,
            prepared.position_ids,
            prepared.last_token_indices,
            *self.scratch_cache.flat_tensors(),
        )

    def redistribute_cache(
        self,
        prepared: PreparedBatchedTextPrefill,
        destinations: Sequence[LocalPaddleOCRVLStaticCache],
    ) -> int:
        if len(destinations) != self.batch_size:
            raise ValueError("batched text cache destinations do not align")
        copied_bytes = 0
        for row, (destination, length) in enumerate(
            zip(destinations, prepared.real_seq_lengths)
        ):
            for source_tensor, destination_tensor in zip(
                self.scratch_cache.flat_tensors(),
                destination.flat_tensors(),
            ):
                source = source_tensor[row : row + 1, :, :length, :]
                destination_tensor[:, :, :length, :].copy_(source)
                copied_bytes += source.numel() * source.element_size()
        return copied_bytes
