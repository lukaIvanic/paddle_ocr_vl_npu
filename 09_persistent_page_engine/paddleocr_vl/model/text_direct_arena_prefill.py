"""Compiled packed text prefill that writes KV directly into decode slots."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable

import torch

from .compile_utils import (
    TORCHAIR_EXECUTION_MODE,
    cache_key_part,
    import_torchair,
    short_file_hash,
    torch_npu_version_label,
    torchair_version_label,
)
from .text_decode import LocalPaddleOCRVLStaticCache
from .text_packed_prefill import PreparedPackedTextPrefill
from .text_prefill import (
    TextPrefillStage,
    _linear_tokenwise,
    attention_softmax,
    get_text_softmax_dtype_mode,
    parse_text_buckets,
    repeat_kv,
    unique_bucket_forward,
)
from utils.timing import synchronize

if TYPE_CHECKING:
    from .modeling import LocalPaddleOCRVLForConditionalGeneration


def direct_arena_source_hash() -> str:
    return short_file_hash(Path(__file__))


class DirectArenaPackedTextPrefillStage(TextPrefillStage):
    """One packed B=1 transformer whose KV writes target arena rows."""

    def __init__(
        self,
        model: LocalPaddleOCRVLForConditionalGeneration,
        *,
        arena_cache_length: int,
    ) -> None:
        super().__init__(model)
        self.arena_cache_length = int(arena_cache_length)

    @staticmethod
    def _scatter_kv(
        cache: torch.Tensor,
        states: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        import torch_npu

        updates = states[0].transpose(0, 1).contiguous().reshape(
            -1,
            states.shape[-1],
        )
        torch_npu.npu_scatter_nd_update_(cache, indices, updates)

    def _direct_attention(
        self,
        attention: torch.nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        scatter_indices: torch.Tensor,
    ) -> torch.Tensor:
        query_states, key_states, value_states = attention.project_qkv(hidden_states)
        query_states, key_states = attention.apply_rotary(
            query_states,
            key_states,
            position_embeddings,
        )
        self._scatter_kv(key_cache, key_states, scatter_indices)
        self._scatter_kv(value_cache, value_states, scatter_indices)

        key_for_attn = repeat_kv(
            key_states,
            int(attention.num_key_value_groups),
        )
        value_for_attn = repeat_kv(
            value_states,
            int(attention.num_key_value_groups),
        )
        batch, num_heads, seq_length, head_dim = query_states.shape
        query_bh = query_states.reshape(batch * num_heads, seq_length, head_dim)
        key_bh = key_for_attn.reshape(batch * num_heads, seq_length, head_dim)
        value_bh = value_for_attn.reshape(batch * num_heads, seq_length, head_dim)
        scores = torch.bmm(query_bh, key_bh.transpose(1, 2)).view(
            batch,
            num_heads,
            seq_length,
            seq_length,
        ) * attention.scaling
        probabilities = attention_softmax(
            scores + attention_mask,
            dim=-1,
            output_dtype=query_states.dtype,
            mode=self.softmax_dtype_mode,
        )
        attention_output = torch.bmm(
            probabilities.reshape(batch * num_heads, seq_length, seq_length),
            value_bh,
        ).view(batch, num_heads, seq_length, head_dim)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch,
            seq_length,
            num_heads * head_dim,
        )
        return _linear_tokenwise(attention.o_proj, attention_output)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        local_positions: torch.Tensor,
        last_token_indices: torch.Tensor,
        segment_slots: torch.Tensor,
        *flat_arena_cache_tensors: torch.Tensor,
    ) -> torch.Tensor:
        key_caches = tuple(flat_arena_cache_tensors[: self.num_layers])
        value_caches = tuple(flat_arena_cache_tensors[self.num_layers :])
        valid = segment_ids >= 0
        allowed = (
            valid[:, None]
            & valid[None, :]
            & (segment_ids[:, None] == segment_ids[None, :])
            & (local_positions[:, None] >= local_positions[None, :])
        )
        attention_mask = torch.zeros(
            (1, 1, inputs_embeds.shape[1], inputs_embeds.shape[1]),
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        ).masked_fill(
            ~allowed[None, None],
            torch.finfo(inputs_embeds.dtype).min,
        )

        physical_positions = torch.arange(
            inputs_embeds.shape[1],
            device=inputs_embeds.device,
            dtype=torch.int64,
        )
        selected_slots = torch.index_select(
            segment_slots,
            0,
            segment_ids.clamp_min(0),
        )
        selected_slots = torch.where(valid, selected_slots, segment_slots[0])
        tail_positions = (
            self.arena_cache_length
            - inputs_embeds.shape[1]
            + physical_positions
        )
        selected_positions = torch.where(
            valid,
            local_positions,
            tail_positions,
        )
        num_key_value_heads = int(key_caches[0].shape[1])
        heads = torch.arange(
            num_key_value_heads,
            device=inputs_embeds.device,
            dtype=torch.int64,
        ).view(1, -1).expand(inputs_embeds.shape[1], -1)
        scatter_indices = torch.stack(
            (
                selected_slots.view(-1, 1).expand(-1, num_key_value_heads),
                heads,
                selected_positions.view(-1, 1).expand(
                    -1,
                    num_key_value_heads,
                ),
            ),
            dim=-1,
        ).reshape(-1, 3).contiguous()

        position_embeddings = self.text_model.rotary_emb(
            inputs_embeds,
            position_ids,
        )
        hidden_states = inputs_embeds
        for layer_idx, layer in enumerate(self.text_model.layers):
            residual = hidden_states
            attention_input = layer.input_layernorm(hidden_states)
            attention_output = self._direct_attention(
                layer.self_attn,
                attention_input,
                attention_mask,
                position_embeddings,
                key_caches[layer_idx],
                value_caches[layer_idx],
                scatter_indices,
            )
            hidden_states = layer.apply_blocks(residual, attention_output)
        hidden_states = self.text_model.norm(hidden_states)
        return torch.index_select(hidden_states, 1, last_token_indices)


def direct_arena_cache_dir_for_bucket(
    cache_root: Path,
    *,
    bucket: int,
    max_members: int,
    arena_batch_size: int,
    arena_cache_length: int,
    dtype: torch.dtype,
    device: torch.device,
    model_dir: Path,
    linear_weight_format: str,
) -> Path:
    key = "_".join(
        [
            "text_packed_direct_arena_scatter_nd",
            f"mode{cache_key_part(TORCHAIR_EXECUTION_MODE)}",
            f"softmax{cache_key_part(get_text_softmax_dtype_mode())}",
            f"seq{int(bucket)}",
            f"members{int(max_members)}",
            f"arena_bs{int(arena_batch_size)}",
            f"arena_cache{int(arena_cache_length)}",
            f"weights{cache_key_part(linear_weight_format)}",
            f"dtype{cache_key_part(dtype)}",
            f"model{short_file_hash(model_dir / 'config.json')}",
            f"torch{cache_key_part(torch.__version__)}",
            f"torchnpu{torch_npu_version_label(device)}",
            f"torchair{torchair_version_label(device)}",
            f"src{direct_arena_source_hash()}",
        ]
    )
    return cache_root.expanduser().resolve() / key


class DirectArenaPackedTextPrefillRuntime:
    """Static direct-arena graphs sharing one real decode-shaped cache."""

    def __init__(
        self,
        model: LocalPaddleOCRVLForConditionalGeneration,
        *,
        buckets: str | Iterable[int],
        max_members: int,
        arena_batch_size: int,
        arena_cache_length: int,
        cache_root: Path,
        device: torch.device,
        dtype: torch.dtype,
        model_dir: Path,
        linear_weight_format: str,
        allow_compile: bool,
    ) -> None:
        if device.type != "npu":
            raise ValueError("direct-arena text prefill requires an NPU")
        self.buckets = parse_text_buckets(buckets)
        self.max_members = int(max_members)
        self.arena_batch_size = int(arena_batch_size)
        self.arena_cache_length = int(arena_cache_length)
        if self.max_members < self.arena_batch_size:
            raise ValueError("max_members must cover every decode-arena slot")
        if self.buckets[-1] >= self.arena_cache_length:
            raise ValueError("arena cache needs a disjoint padding tail")

        self.arena_cache = model.allocate_static_cache(
            batch_size=self.arena_batch_size,
            cache_length=self.arena_cache_length,
            device=device,
            dtype=dtype,
            init_mode="empty",
        )
        torchair, CompilerConfig = import_torchair()
        hidden_size = int(model.config.text_config.hidden_size)
        self.compiled: dict[int, Callable[..., torch.Tensor]] = {}
        per_bucket: dict[str, Any] = {}
        for bucket in self.buckets:
            module = DirectArenaPackedTextPrefillStage(
                model,
                arena_cache_length=self.arena_cache_length,
            ).eval()
            entrypoint = unique_bucket_forward(module, bucket)
            cache_dir = direct_arena_cache_dir_for_bucket(
                cache_root,
                bucket=bucket,
                max_members=self.max_members,
                arena_batch_size=self.arena_batch_size,
                arena_cache_length=self.arena_cache_length,
                dtype=dtype,
                device=device,
                model_dir=model_dir,
                linear_weight_format=linear_weight_format,
            )
            cache_was_warm = cache_dir.is_dir() and any(cache_dir.iterdir())
            if not cache_was_warm and not allow_compile:
                raise RuntimeError(
                    "missing direct-arena graph; pass --allow-compile: "
                    f"{cache_dir}"
                )
            cache_dir.mkdir(parents=True, exist_ok=True)
            synchronize(device)
            wrapper_started = time.perf_counter()
            compiled = torchair.inference.cache_compile(
                entrypoint,
                config=CompilerConfig(),
                dynamic=False,
                cache_dir=str(cache_dir),
                ge_cache=True,
            )
            synchronize(device)
            wrapper_s = time.perf_counter() - wrapper_started

            warm_slots = torch.zeros(
                (self.max_members,),
                device=device,
                dtype=torch.int64,
            )
            warm_slots[: self.arena_batch_size] = torch.arange(
                self.arena_batch_size,
                device=device,
                dtype=torch.int64,
            )
            warm_inputs = (
                torch.zeros((1, bucket, hidden_size), device=device, dtype=dtype),
                torch.zeros((3, 1, bucket), device=device, dtype=torch.int64),
                torch.zeros((bucket,), device=device, dtype=torch.int64),
                torch.arange(bucket, device=device, dtype=torch.int64),
                torch.zeros((self.max_members,), device=device, dtype=torch.int64),
                warm_slots,
            )
            synchronize(device)
            first_call_started = time.perf_counter()
            warm_output = compiled(
                *warm_inputs,
                *self.arena_cache.flat_tensors(),
            )
            synchronize(device)
            first_call_s = time.perf_counter() - first_call_started
            expected = (1, self.max_members, hidden_size)
            if tuple(warm_output.shape) != expected:
                raise RuntimeError(
                    f"direct-arena graph returned {tuple(warm_output.shape)}, "
                    f"expected {expected}"
                )
            self.compiled[bucket] = compiled
            per_bucket[str(bucket)] = {
                "cache_dir": str(cache_dir),
                "cache_was_warm": cache_was_warm,
                "new_graph_compiled": not cache_was_warm,
                "wrapper_s": wrapper_s,
                "first_call_s": first_call_s,
            }
            del warm_output, warm_inputs, warm_slots
        self.metadata = {
            "buckets": list(self.buckets),
            "max_members": self.max_members,
            "arena_batch_size": self.arena_batch_size,
            "arena_cache_length": self.arena_cache_length,
            "cache_write": "npu_scatter_nd_update_in_graph",
            "padding_sink": "high_masked_cache_positions",
            "per_bucket": per_bucket,
        }

    def run_prepared(
        self,
        prepared: PreparedPackedTextPrefill,
        segment_slots: torch.Tensor,
    ) -> torch.Tensor:
        if tuple(segment_slots.shape) != (self.max_members,):
            raise ValueError(
                f"segment slots must have shape {(self.max_members,)}, "
                f"got {tuple(segment_slots.shape)}"
            )
        return self.compiled[prepared.physical_seq_len](
            prepared.inputs_embeds,
            prepared.position_ids,
            prepared.segment_ids,
            prepared.local_positions,
            prepared.last_token_indices,
            segment_slots,
            *self.arena_cache.flat_tensors(),
        )
