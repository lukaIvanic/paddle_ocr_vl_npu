"""Request-owned KV slots and fixed-batch compiled decode for MinerU.

This is the first scheduler-shaped inference engine.  Each request is
prefilled independently into a view of its own row in a shared KV arena, so
prompt padding never enters decode.  Full groups use one compiled graph;
incomplete tails deliberately reuse the established B1 path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from local_modeling_mineru import LocalMinerUStaticCache
from run_local_model_two_step_extract import maybe_sync_device


@dataclass
class PreparedGeneration:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    pixel_values: torch.Tensor | None
    image_grid_thw: torch.Tensor | None
    max_new_tokens: int


class FixedBatchDecodeEngine:
    """B1 prefill into request slots followed by lockstep compiled decode."""

    def __init__(
        self,
        model: Any,
        compiled_decoder: Any,
        *,
        batch_size: int,
        cache_length: int,
        eos_token_id: int,
        pad_token_id: int,
    ) -> None:
        if int(batch_size) <= 1:
            raise ValueError("fixed batch engine requires batch_size > 1")
        self.model = model
        self.compiled_decoder = compiled_decoder
        self.batch_size = int(batch_size)
        self.cache_length = int(cache_length)
        self.eos_token_id = int(eos_token_id)
        self.pad_token_id = int(pad_token_id)
        self._arena: LocalMinerUStaticCache | None = None

    def _arena_for_batch(self) -> LocalMinerUStaticCache:
        if self._arena is None:
            self._arena = self.model.allocate_static_cache(
                batch_size=self.batch_size,
                cache_length=self.cache_length,
                device=self.model.device,
                dtype=self.model.dtype,
                init_mode="empty",
            )
        return self._arena

    @staticmethod
    def _slot_view(arena: LocalMinerUStaticCache, slot: int) -> LocalMinerUStaticCache:
        return LocalMinerUStaticCache(
            key_caches=tuple(cache[slot : slot + 1] for cache in arena.key_caches),
            value_caches=tuple(cache[slot : slot + 1] for cache in arena.value_caches),
            cache_length=arena.cache_length,
        )

    @torch.inference_mode()
    def generate_many(
        self,
        requests: Sequence[PreparedGeneration],
    ) -> tuple[list[torch.Tensor], dict[str, Any]]:
        if not requests:
            return [], {
                "enabled": True,
                "batch_size": 0,
                "decode_calls": 0,
                "raw_decode_token_slots": 0,
                "decode_s": 0.0,
                "prefill_s": 0.0,
            }
        if len(requests) != self.batch_size:
            return self._generate_tail_b1(requests)
        return self._generate_full_batch(requests)

    @torch.inference_mode()
    def _generate_tail_b1(
        self,
        requests: Sequence[PreparedGeneration],
    ) -> tuple[list[torch.Tensor], dict[str, Any]]:
        started = time.perf_counter()
        outputs: list[torch.Tensor] = []
        records: list[dict[str, Any]] = []
        for request in requests:
            generated, metrics = self.compiled_decoder.generate(
                input_ids=request.input_ids,
                attention_mask=request.attention_mask,
                pixel_values=request.pixel_values,
                image_grid_thw=request.image_grid_thw,
                max_new_tokens=request.max_new_tokens,
                eos_token_id=self.eos_token_id,
                pad_token_id=self.pad_token_id,
            )
            outputs.append(generated)
            records.append(metrics)
        decode_calls = sum(int(item["decode_calls"]) for item in records)
        decode_s = sum(float(item["decode_s"]) for item in records)
        return outputs, {
            "enabled": True,
            "mode": "b1_tail",
            "batch_size": len(requests),
            "decode_calls": decode_calls,
            "raw_decode_token_slots": decode_calls,
            "decode_s": decode_s,
            "prefill_s": sum(float(item["prefill_s"]) for item in records),
            "generation_wall_s": float(time.perf_counter() - started),
            "compile_wrapper_s": sum(
                float(item["compile_wrapper_s"])
                for item in records
                if item.get("compile_warmup", {}).get("ran_this_call")
            ),
            "compiled_first_call_s": sum(
                float(item["compiled_first_call_s"])
                for item in records
                if item.get("compile_warmup", {}).get("ran_this_call")
            ),
        }

    @torch.inference_mode()
    def _generate_full_batch(
        self,
        requests: Sequence[PreparedGeneration],
    ) -> tuple[list[torch.Tensor], dict[str, Any]]:
        started = time.perf_counter()
        arena = self._arena_for_batch()
        next_tokens: list[torch.Tensor] = []
        rope_deltas: list[torch.Tensor] = []
        cache_positions: list[torch.Tensor] = []
        generated_steps: list[torch.Tensor] = []

        maybe_sync_device(self.model.device)
        prefill_started = time.perf_counter()
        for slot, request in enumerate(requests):
            if request.input_ids.shape[0] != 1:
                raise ValueError("each fixed-batch request must be prepared at B1")
            if request.input_ids.shape[1] + request.max_new_tokens > self.cache_length:
                raise ValueError(
                    "request exceeds static cache capacity: "
                    f"input={int(request.input_ids.shape[1])} "
                    f"max_new={request.max_new_tokens} cache={self.cache_length}"
                )
            prefill = self.model.forward_static_prefill(
                input_ids=request.input_ids,
                attention_mask=request.attention_mask,
                pixel_values=request.pixel_values,
                image_grid_thw=request.image_grid_thw,
                cache_length=self.cache_length,
                cache=self._slot_view(arena, slot),
                logits_to_keep=1,
            )
            token = torch.argmax(prefill.logits[:, -1, :].float(), dim=-1, keepdim=True)
            next_tokens.append(token)
            rope_deltas.append(prefill.rope_deltas)
            cache_positions.append(prefill.next_cache_position)
        maybe_sync_device(self.model.device)
        prefill_s = time.perf_counter() - prefill_started

        next_token = torch.cat(next_tokens, dim=0)
        generated_steps.append(next_token)
        rope_delta = torch.cat(rope_deltas, dim=0)
        cache_position = torch.cat(cache_positions, dim=0)
        generated_counts = torch.ones(
            self.batch_size,
            device=self.model.device,
            dtype=torch.int64,
        )
        max_new = torch.tensor(
            [request.max_new_tokens for request in requests],
            device=self.model.device,
            dtype=torch.int64,
        )
        finished = (next_token.squeeze(1) == self.eos_token_id) | (generated_counts >= max_new)

        compile_started = time.perf_counter()
        compiled_decode, compile_meta = self.compiled_decoder.compiled_decode_for(
            batch_size=self.batch_size,
            cache_length=self.cache_length,
        )
        compile_wrapper_s = time.perf_counter() - compile_started

        graph_calls = 0
        first_call_s = 0.0
        maybe_sync_device(self.model.device)
        decode_started = time.perf_counter()
        while not bool(finished.all().item()):
            call_started = time.perf_counter()
            logits = compiled_decode(
                next_token,
                cache_position,
                rope_delta,
                *arena.flat_tensors(),
            )
            if graph_calls == 0:
                maybe_sync_device(self.model.device)
                first_call_s = time.perf_counter() - call_started
            graph_calls += 1
            candidate = torch.argmax(logits[:, -1, :].float(), dim=-1, keepdim=True)
            active = ~finished
            emitted = torch.where(
                active.view(-1, 1),
                candidate,
                torch.full_like(candidate, self.pad_token_id),
            )
            generated_steps.append(emitted)
            generated_counts.add_(active.to(dtype=torch.int64))
            finished = finished | (candidate.squeeze(1) == self.eos_token_id) | (generated_counts >= max_new)
            next_token = torch.where(
                finished.view(-1, 1),
                torch.full_like(candidate, self.pad_token_id),
                candidate,
            )
            cache_position.add_(active.to(dtype=torch.int64))
        maybe_sync_device(self.model.device)
        decode_s = time.perf_counter() - decode_started

        generated_lengths = [int(value) for value in generated_counts.cpu().tolist()]
        all_generated = torch.cat(generated_steps, dim=1)
        outputs = [
            all_generated[slot : slot + 1, :length]
            for slot, length in enumerate(generated_lengths)
        ]
        effective_decode_tokens = sum(generated_lengths) - self.batch_size
        return outputs, {
            "enabled": True,
            "mode": "fixed_batch",
            "batch_size": self.batch_size,
            "cache_length": self.cache_length,
            "graph_calls": graph_calls,
            "decode_calls": effective_decode_tokens,
            "raw_decode_token_slots": graph_calls * self.batch_size,
            "decode_s": float(decode_s),
            "prefill_s": float(prefill_s),
            "generation_wall_s": float(time.perf_counter() - started),
            "compile_wrapper_s": float(compile_wrapper_s),
            "compiled_first_call_s": float(first_call_s),
            "compile": dict(compile_meta),
        }
