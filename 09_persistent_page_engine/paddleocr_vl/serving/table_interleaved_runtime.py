"""Real two-table reference runtime; no cross-phase fused graph or trace replay.

Each request owns stable target and eight draft slots. Different K graphs read
the same target storage: changing K never copies the historical KV cache.
Single-request and paired graphs use contiguous views of those stable slots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any

import torch

from ..model.text_decode import TextDecodeRuntime
from ..model.text_spec_verify import TextSpecVerifyRuntime
from .repetition import ExactCycleTracker
from .table_phase_scheduler import PhaseWork, accept_native_proposal


@dataclass
class _Sequence:
    prefilled: Any
    slot: int
    position: int
    cache_length: int
    limit: int
    eos: int
    tokens: list[int]
    tracker: ExactCycleTracker = field(default_factory=ExactCycleTracker)
    stop: str | None = None
    calls: int = 0

    def append(self, values: list[int]) -> None:
        for token in values:
            self.tokens.append(int(token))
            if token == self.eos:
                self.stop = "eos"
                break
            evidence = self.tracker.update(token)
            if evidence is not None:
                del self.tokens[evidence.trim_length:]
                self.stop = "repetition"
                break
            if self.prefilled.input_tokens + len(self.tokens) - 1 >= self.cache_length:
                self.stop = "kv_cache_full"
                break
            if len(self.tokens) >= self.limit:
                self.stop = "length"
                break


class _Arena:
    def __init__(self, recognizer: Any, capacity: int) -> None:
        self.recognizer = recognizer
        self.capacity = capacity
        self.cache = recognizer.model.allocate_static_cache(
            batch_size=capacity, cache_length=recognizer.cache_length,
            device=recognizer.device, dtype=recognizer.dtype, init_mode="zeros",
        )
        self.rope = torch.zeros((capacity, 1), device=recognizer.device, dtype=torch.int64)
        self.states: dict[int, _Sequence] = {}
        self.views: dict[tuple[int, int], tuple[Any, ...]] = {}

    def tensors(self, first: int, batch: int) -> tuple[Any, ...]:
        key = first, batch
        if key not in self.views:
            self.views[key] = tuple(tensor[first:first + batch] for tensor in self.cache.flat_tensors())
            if not all(tensor.is_contiguous() for tensor in self.views[key]):
                raise RuntimeError("interleaved cache views must be contiguous")
        return self.views[key]

    def admit(self, prefilled: Any, slot: int, limit: int) -> _Sequence:
        if slot in self.states:
            raise RuntimeError(f"occupied KV slot {slot}")
        cache, rope, position, _, release = prefilled.take_device_state()
        try:
            if cache.packed_kv_caches is not None:
                raise RuntimeError("reference runtime requires ordinary two-head KV storage")
            torch._foreach_copy_(
                tuple(t[slot:slot + 1] for t in self.cache.logical_tensors()),
                cache.logical_tensors(),
            )
            self.rope[slot:slot + 1].copy_(rope)
            state = _Sequence(
                prefilled=prefilled, slot=slot,
                position=int(position.detach().cpu().item()),
                cache_length=int(self.cache.cache_length), limit=limit,
                eos=int(self.recognizer.model.config.eos_token_id),
                tokens=[int(prefilled.first_token)],
            )
            state.tracker.update(state.tokens[0])
            if state.tokens[0] == state.eos:
                state.stop = "eos"
            elif state.position >= state.cache_length:
                state.stop = "kv_cache_full"
            elif limit <= 1:
                state.stop = "length"
            self.states[slot] = state
            return state
        finally:
            if release is not None:
                release()


class _Call:
    def __init__(self, runtime: Any, batch: int, query: int, device: Any) -> None:
        self.runtime = runtime
        self.host_ids = torch.empty((batch, query), dtype=torch.int64, pin_memory=True)
        self.ids = self.host_ids.numpy()
        self.host_pos = torch.empty(batch, dtype=torch.int64, pin_memory=True)
        self.positions = self.host_pos.numpy()
        self.device_ids = torch.empty((batch, query), dtype=torch.int64, device=device)
        self.device_pos = torch.empty(batch, dtype=torch.int64, device=device)
        self.host_result = torch.empty_like(self.host_ids)
        self.begin = torch.npu.Event(enable_timing=True)
        self.end = torch.npu.Event(enable_timing=True)

    def run(self, arena: _Arena, first: int, sequences: list[_Sequence], proposals: dict[int, Any]) -> tuple[list[list[int]], float]:
        batch, query = self.ids.shape
        self.ids.fill(int(arena.recognizer.model.config.eos_token_id))
        self.positions.fill(0)
        for state in sequences:
            index = state.slot - first
            self.ids[index, 0] = state.tokens[-1]
            self.positions[index] = state.position
            proposal = proposals.get(state.slot)
            if proposal is not None:
                self.ids[index, 1:1 + len(proposal.tokens)] = proposal.tokens
        self.device_ids.copy_(self.host_ids, non_blocking=True)
        self.device_pos.copy_(self.host_pos, non_blocking=True)
        self.begin.record()
        output = self.runtime.fn(
            self.device_ids, self.device_pos, arena.rope[first:first + batch],
            *arena.tensors(first, batch),
        )
        self.end.record()
        # These locked optimized graphs map compact argmax back to native IDs.
        if output.dtype != torch.int64 or output.numel() != batch * query:
            raise RuntimeError("expected native compact-head argmax IDs")
        self.host_result.copy_(output.reshape(batch, query), non_blocking=True)
        torch.npu.current_stream(arena.recognizer.device).synchronize()
        return self.host_result.numpy().tolist(), float(self.begin.elapsed_time(self.end)) / 1000.0


class InterleavedTableRuntime:
    """Step API used by the existing HTTP worker, with C1 or C2 capacity."""

    def __init__(self, b1: Any, draft: Any, args: Any, *, capacity: int) -> None:
        if capacity not in (1, 2):
            raise ValueError("reference supports one or two active tables")
        if not hasattr(b1.model, "decode_token_id_map") or not hasattr(draft.model, "decode_token_id_map"):
            raise ValueError("reference requires the production compact vocabulary")
        self.capacity, self.b1, self.draft, self.args = capacity, b1, draft, args
        self.k_values = tuple(sorted(int(k) for k in args.k_values.split(",")))
        if self.k_values != (7, 15, 31, 63) or args.initial_k not in self.k_values:
            raise ValueError("reference preserves the production adaptive K policy")
        self.target_arena = _Arena(b1, capacity)
        self.draft_arena = _Arena(draft, 8 * capacity)
        self.jobs: dict[str, dict[str, Any]] = {}
        self.calls: dict[tuple[str, int, int], _Call] = {}
        self.metadata: dict[str, Any] = {}
        for batch in range(1, capacity + 1):
            for kind, recognizer, physical_batch in (("target", b1, batch), ("draft", draft, 8 * batch)):
                print(f"TABLE_PHASE setup=decode kind={kind} B={physical_batch} KV={recognizer.cache_length}", flush=True)
                runtime = recognizer.text_decode if physical_batch == recognizer.batch_size else TextDecodeRuntime(
                    recognizer.model, backend="torchair", device=recognizer.device,
                    cache_root=(args.decode_cache_dir / (
                        f"selected_vocab_{recognizer.decode_vocab['selected_vocab_size']}_"
                        f"{recognizer.decode_vocab['token_ids_sha256'][:12]}"
                    )), batch_size=physical_batch,
                    cache_length=recognizer.cache_length, dtype=recognizer.dtype,
                    model_dir=recognizer.model_dir,
                    linear_weight_format=str(recognizer.weight_format["effective_mode"]),
                    optimization=recognizer.decode_optimization,
                )
                self._add_call(kind, physical_batch, 1, runtime, recognizer.device)
            for k in self.k_values:
                print(f"TABLE_PHASE setup=verifier B={batch} Q={k + 1} KV={b1.cache_length}", flush=True)
                runtime = TextSpecVerifyRuntime(
                    b1.model, batch_size=batch, device=b1.device,
                    cache_root=args.decode_cache_dir, draft_length=k,
                    cache_length=b1.cache_length, dtype=b1.dtype, model_dir=b1.model_dir,
                    linear_weight_format=str(b1.weight_format["effective_mode"]),
                    optimization=args.verifier_optimization,
                    preferred_token_id=b1.math_open_token_id,
                    alternate_preferred_token_id=b1.math_slash_token_id,
                    cell_start_token_ids=b1.table_cell_token_ids,
                )
                self._add_call("target", batch, k + 1, runtime, b1.device)

    def _add_call(self, kind: str, batch: int, query: int, runtime: Any, device: Any) -> None:
        self.calls[kind, batch, query] = _Call(runtime, batch, query, device)
        self.metadata[f"{kind}_b{batch}q{query}"] = dict(runtime.metadata)

    def add(self, request_id: str, *, route: str, payload: Any, target_prepared: Any, row_groups: Any = None) -> None:
        if request_id in self.jobs or len(self.jobs) >= self.capacity:
            raise RuntimeError("invalid table admission")
        used = {job["slot"] for job in self.jobs.values()}
        slot = next(slot for slot in range(self.capacity) if slot not in used)
        self.jobs[request_id] = {
            "slot": slot, "route": route, "payload": payload,
            "phase": "draft_prefill" if route == "spec" else "target_prefill",
            "target_prepared": target_prepared, "row_groups": row_groups,
            "draft_states": [], "target": None, "matcher": None,
            "policy_k": int(self.args.initial_k), "proposal": None,
            "proposal_ready": False, "query": 1, "trace": [],
            "prefill_records": [],
        }

    def work(self) -> list[PhaseWork]:
        result = []
        for request_id, job in self.jobs.items():
            if job["phase"] == "verify" and not job["proposal_ready"]:
                state = job["target"]
                usable = [k for k in self.k_values if k <= job["policy_k"] and state.position + k + 1 <= state.cache_length]
                k = max(usable) if usable else None
                if k is not None:
                    job["matcher"].block_size = k
                proposal = job["matcher"].propose(state.tokens)
                job["proposal"] = proposal if k is not None and proposal is not None and proposal.tokens else None
                job["query"] = k + 1 if job["proposal"] is not None else 1
                job["proposal_ready"] = True
            if job["phase"] != "done":
                result.append(PhaseWork(request_id, job["phase"], job["query"] if job["phase"] == "verify" else 1))
        return result

    def prefill(self, request_id: str) -> None:
        job = self.jobs[request_id]
        if job["phase"] == "draft_prefill":
            # Use the established impatient vision-pack former and exact prefill
            # stages. Only the consumer is changed to expose independent steps.
            group = next(job["row_groups"])
            inflight = self.draft._enqueue_staged_prefill_group(self.draft._stage_prefill_group(group))
            rows = self.draft._finalize_prefill_group(inflight)
            for row in rows:
                row_index = int(row.request_id.rsplit("_", 1)[-1])
                state = self.draft_arena.admit(row, 8 * job["slot"] + row_index, int(self.args.draft_cache_length))
                job["draft_states"].append(state)
                job["prefill_records"].append({"kind": "draft", "row": row_index, "timing_s": dict(row.timing_s), "device_stage_s": dict(row.device_stage_s)})
            if len(job["draft_states"]) == 8:
                job["row_groups"].close()
                job["row_groups"] = None
                job["phase"] = "draft"
            elif len(job["draft_states"]) > 8:
                raise RuntimeError("more than eight draft rows admitted")
        elif job["phase"] == "target_prefill":
            prefilled = self.b1.prefill_prepared_one(job["target_prepared"])
            job["target_prepared"] = None
            job["target"] = self.target_arena.admit(prefilled, job["slot"], self.args.b1_max_new_tokens)
            job["prefill_records"].append({"kind": "target", "timing_s": dict(prefilled.timing_s), "device_stage_s": dict(prefilled.device_stage_s)})
            if job["matcher"] is not None:
                job["matcher"].start(job["target"].tokens[0])
            job["phase"] = "verify" if job["route"] == "spec" else "ordinary"
        else:
            raise RuntimeError("request is not ready for prefill")

    def decode_step(self, work: list[PhaseWork]) -> tuple[str, float]:
        selected = [self.jobs[item.request_id] for item in work]
        phase, query = work[0].key
        if any(item.key != work[0].key for item in work):
            raise ValueError("only identical shapes can share a reference call")
        if phase == "draft":
            arena, batch = self.draft_arena, 8 * len(selected)
            first = 0 if len(selected) == 2 else 8 * selected[0]["slot"]
            states = [state for job in selected for state in job["draft_states"] if state.stop is None]
            proposals = {}
            kind = "draft"
        else:
            arena, batch = self.target_arena, len(selected)
            first = 0 if batch == 2 else selected[0]["slot"]
            states = [job["target"] for job in selected]
            proposals = {job["slot"]: job["proposal"] for job in selected if query > 1}
            kind = "target"
        output, device_s = self.calls[kind, batch, query].run(arena, first, states, proposals)
        for state in states:
            predictions = output[state.slot - first]
            state.calls += 1
            if phase == "draft":
                state.append(predictions[:1])
                state.position += 1
                continue
            job = next(job for job in selected if job["slot"] == state.slot)
            proposal = job["proposal"] if query > 1 else None
            emitted, accepted = accept_native_proposal(proposal.tokens if proposal is not None else [], predictions)
            if job["matcher"] is not None:
                job["matcher"].commit(proposal, accepted_draft_tokens=accepted, emitted_tokens=emitted)
            if proposal is not None:
                k = query - 1
                full = accepted == len(proposal.tokens)
                index = self.k_values.index(k)
                next_k = self.k_values[min(len(self.k_values) - 1, index + 1) if full else max(0, index - 1)]
                job["trace"].append({"position": state.position, "k": k, "proposed": len(proposal.tokens), "accepted": accepted, "next_k": next_k})
                job["policy_k"] = next_k
            state.append(emitted)
            state.position += len(emitted)
            job["proposal_ready"] = False
        return f"{phase}_b{batch}q{query}", device_s

    def transitions(self, make_matcher: Any, request_id: str | None = None) -> list[str]:
        done = []
        entries = self.jobs.items() if request_id is None else [(request_id, self.jobs[request_id])]
        for request_id, job in entries:
            if job["phase"] == "draft" and all(state.stop is not None for state in job["draft_states"]):
                rows = [{"row_index": state.slot % 8, "token_ids": state.tokens} for state in job["draft_states"]]
                job["matcher"] = make_matcher({"request_id": request_id, "rows": rows})
                for state in job["draft_states"]:
                    del self.draft_arena.states[state.slot]
                job["phase"] = "target_prefill"
            if job["target"] is not None and job["target"].stop is not None:
                job["phase"] = "done"
                done.append(request_id)
        return done

    def retire(self, request_id: str) -> dict[str, Any]:
        job = self.jobs.pop(request_id)
        del self.target_arena.states[job["slot"]]
        return job

    def close(self) -> None:
        for job in self.jobs.values():
            groups = job.get("row_groups")
            if groups is not None:
                groups.close()
        self.jobs.clear()
        self.draft_arena.states.clear()
        self.target_arena.states.clear()
