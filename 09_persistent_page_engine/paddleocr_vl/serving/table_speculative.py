"""CPU draft matching and B1 target-side speculative table decode."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import time
from typing import Any, Iterable

import torch

from ..model.text_spec_verify import TextSpecVerifyRuntime
from .engine import PrefilledRecognition
from .repetition import ExactCycleTracker


@dataclass(frozen=True)
class DraftPosition:
    lane: int
    logical_row: int
    column: int
    row_width: int


@dataclass(frozen=True)
class DraftProposal:
    start: int
    tokens: tuple[int, ...]
    anchor_tokens: int


@dataclass
class TargetStructure:
    row: int = 0
    column: int = -1
    width_counts: Counter[int] = field(default_factory=Counter)
    modal_width: int | None = None
    modal_width_count: int = 0

    @property
    def width(self) -> int | None:
        return self.modal_width


def _token_id(tokenizer: Any, token: str) -> int:
    if hasattr(tokenizer, "token_to_id"):
        value = tokenizer.token_to_id(token)
    else:
        value = tokenizer.convert_tokens_to_ids(token)
    if value is None:
        raise ValueError(f"tokenizer does not contain {token!r}")
    return int(value)


def _preceding_match(
    prefix: list[int],
    draft: list[int],
    continuation: int,
    maximum: int,
) -> int:
    matched = 0
    while (
        matched < maximum
        and matched < len(prefix)
        and matched < continuation
        and prefix[-matched - 1] == draft[continuation - matched - 1]
    ):
        matched += 1
    return matched


def _continuation_index(
    draft: list[int],
    maximum_anchor: int,
) -> dict[int, dict[tuple[int, ...], list[int]]]:
    lengths: list[int] = []
    length = 1
    while length <= maximum_anchor:
        lengths.append(length)
        length *= 2
    result: dict[int, dict[tuple[int, ...], list[int]]] = {}
    for anchor_length in lengths:
        by_anchor: defaultdict[tuple[int, ...], list[int]] = defaultdict(list)
        for continuation in range(anchor_length, len(draft)):
            key = tuple(draft[continuation - anchor_length : continuation])
            by_anchor[key].append(continuation)
        result[anchor_length] = dict(by_anchor)
    return result


def _flatten_rows(
    record: dict[str, Any],
    *,
    eos_token_id: int,
    cell_tokens: set[int],
    newline_token: int,
) -> tuple[list[int], list[DraftPosition]]:
    flat: list[int] = []
    metadata: list[DraftPosition] = []
    rows = sorted(record.get("rows") or [], key=lambda item: item["row_index"])
    for lane, row_record in enumerate(rows):
        tokens = [int(value) for value in row_record.get("token_ids") or ()]
        if tokens and tokens[-1] == eos_token_id:
            tokens.pop()
        logical_rows: list[list[int]] = [[]]
        for token in tokens:
            logical_rows[-1].append(token)
            if token == newline_token:
                logical_rows.append([])
        for logical_row, logical_tokens in enumerate(
            row for row in logical_rows if row
        ):
            width = sum(token in cell_tokens for token in logical_tokens)
            column = -1
            for token in logical_tokens:
                if token in cell_tokens:
                    column += 1
                flat.append(token)
                metadata.append(
                    DraftPosition(lane, logical_row, column, width)
                )
    return flat, metadata


class TableDraftMatcher:
    """Best validated legal matcher from the offline 665-table analysis.

    Exact target-prefix length remains authoritative. OTSL column position and
    the bounded width patch break ties. The cursor follows the most recently
    accepted draft location and can move in either direction.
    """

    def __init__(
        self,
        record: dict[str, Any],
        tokenizer: Any,
        *,
        eos_token_id: int,
        block_size: int = 16,
        maximum_anchor: int = 64,
        column_weight: float = 0.25,
    ) -> None:
        self.block_size = int(block_size)
        self.maximum_anchor = int(maximum_anchor)
        self.column_weight = float(column_weight)
        self.cell_tokens = {
            _token_id(tokenizer, token)
            for token in ("<fcel>", "<ecel>", "<lcel>", "<ucel>", "<xcel>")
        }
        self.newline_token = _token_id(tokenizer, "<nl>")
        self.ecel_token = _token_id(tokenizer, "<ecel>")
        self.fcel_token = _token_id(tokenizer, "<fcel>")
        self.draft, self.metadata = _flatten_rows(
            record,
            eos_token_id=int(eos_token_id),
            cell_tokens=self.cell_tokens,
            newline_token=self.newline_token,
        )
        self.index = _continuation_index(self.draft, self.maximum_anchor)
        self.cursor = 0
        self.structure = TargetStructure()
        self._started = False

    def _observe(self, token: int) -> None:
        if token in self.cell_tokens:
            self.structure.column += 1
        if token == self.newline_token:
            width = self.structure.column + 1
            self.structure.width_counts[width] += 1
            count = self.structure.width_counts[width]
            if count > self.structure.modal_width_count:
                self.structure.modal_width = width
                self.structure.modal_width_count = count
            self.structure.row += 1
            self.structure.column = -1

    def start(self, first_token: int) -> None:
        if self._started:
            raise RuntimeError("table matcher was already started")
        self._started = True
        self._observe(int(first_token))

    def _column_score(self, next_token: int, meta: DraftPosition) -> float:
        expected = (
            self.structure.column + 1
            if next_token in self.cell_tokens
            else self.structure.column
        )
        weight = self.column_weight
        score = weight if expected == meta.column else -weight
        target_width = self.structure.width
        if target_width is None:
            return score
        width_delta = meta.row_width - target_width
        if width_delta == 0:
            return score + weight
        if abs(width_delta) > 2:
            return score - weight
        lower = max(-1, meta.column - max(0, width_delta))
        upper = meta.column + max(0, -width_delta)
        if lower <= expected <= upper:
            return score + weight * (0.75 - 0.25 * abs(width_delta))
        return score - weight * 0.5

    def propose(self, prefix: list[int]) -> DraftProposal | None:
        if not self._started:
            raise RuntimeError("call matcher.start(first_token) before propose")
        if not prefix or not self.draft:
            return None
        if (
            len(prefix) == 1
            and prefix[0] == self.ecel_token
            and self.draft[0] == self.fcel_token
        ):
            return DraftProposal(
                0,
                tuple(self.draft[: self.block_size]),
                0,
            )

        usable = [length for length in self.index if length <= len(prefix)]
        for indexed_anchor in sorted(usable, reverse=True):
            continuations = self.index[indexed_anchor].get(
                tuple(prefix[-indexed_anchor:]),
                (),
            )
            best: tuple[tuple[float, float, int, int], DraftProposal] | None = None
            for continuation in continuations:
                if continuation >= len(self.draft):
                    continue
                tokens = tuple(
                    self.draft[continuation : continuation + self.block_size]
                )
                if not tokens:
                    continue
                anchor = _preceding_match(
                    prefix,
                    self.draft,
                    continuation,
                    self.maximum_anchor,
                )
                score = (
                    float(anchor),
                    self._column_score(tokens[0], self.metadata[continuation]),
                    int(continuation >= self.cursor),
                    -abs(continuation - self.cursor),
                )
                proposal = DraftProposal(continuation, tokens, anchor)
                if best is None or score > best[0]:
                    best = (score, proposal)
            if best is not None:
                return best[1]
        return None

    def commit(
        self,
        proposal: DraftProposal | None,
        *,
        accepted_draft_tokens: int,
        emitted_tokens: Iterable[int],
    ) -> None:
        emitted = [int(token) for token in emitted_tokens]
        if proposal is not None and accepted_draft_tokens > 0:
            self.cursor = proposal.start + int(accepted_draft_tokens)
        if (
            proposal is not None
            and emitted
            and proposal.start + int(accepted_draft_tokens) < len(self.draft)
            and emitted[-1]
            == self.draft[proposal.start + int(accepted_draft_tokens)]
        ):
            self.cursor = proposal.start + int(accepted_draft_tokens) + 1
        for token in emitted:
            self._observe(token)


@dataclass
class TableSpecDecodeResult:
    token_ids: list[int]
    text: str
    stop_reason: str
    target_calls: int
    speculative_calls: int
    fully_accepted_speculative_calls: int
    rejected_speculative_calls: int
    fallback_calls: int
    proposed_draft_tokens: int
    accepted_draft_tokens: int
    verifier_device_s: float
    fallback_device_s: float
    wall_s: float
    repetition: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["accepted_fraction_of_proposed"] = (
            self.accepted_draft_tokens / self.proposed_draft_tokens
            if self.proposed_draft_tokens
            else None
        )
        result["effective_target_tok_per_s"] = (
            max(0, len(self.token_ids) - 1) / self.wall_s
            if self.wall_s > 0
            else None
        )
        return result


class TableSpeculativeDecodeRuntime:
    """Run fixed-D B1 speculative verification over a real prefill cache."""

    def __init__(
        self,
        recognizer: Any,
        *,
        draft_length: int = 16,
        cache_root: Any,
    ) -> None:
        if int(recognizer.batch_size) != 1:
            raise ValueError("table speculative decode currently requires B1")
        self.recognizer = recognizer
        self.device = recognizer.device
        self.draft_length = int(draft_length)
        self.query_length = self.draft_length + 1
        self.cache_length = int(recognizer.cache_length)
        self.eos_token_id = int(recognizer.model.config.eos_token_id)
        self.verify = TextSpecVerifyRuntime(
            recognizer.model,
            device=recognizer.device,
            cache_root=cache_root,
            draft_length=self.draft_length,
            cache_length=self.cache_length,
            dtype=recognizer.dtype,
            model_dir=recognizer.model_dir,
            linear_weight_format=str(recognizer.weight_format["effective_mode"]),
            optimization="combined_apply",
        )
        self.host_input = torch.empty(
            (1, self.query_length), dtype=torch.int64, pin_memory=True
        )
        self.device_input = torch.empty(
            (1, self.query_length), device=self.device, dtype=torch.int64
        )
        self.host_targets = torch.empty(
            (1, self.query_length), dtype=torch.int64, pin_memory=True
        )
        self.decode_input = torch.empty((1, 1), device=self.device, dtype=torch.int64)
        self.host_decode_target = torch.empty((1, 1), dtype=torch.int64, pin_memory=True)

    def _event(self) -> Any:
        import torch_npu

        return torch_npu.npu.Event(enable_timing=True)

    def _verify_call(
        self,
        current_token: int,
        proposal: tuple[int, ...],
        cache_position: torch.Tensor,
        rope_deltas: torch.Tensor,
        flat_cache: tuple[torch.Tensor, ...],
    ) -> tuple[list[int], float]:
        self.host_input.fill_(self.eos_token_id)
        self.host_input[0, 0] = int(current_token)
        if proposal:
            self.host_input[0, 1 : len(proposal) + 1] = torch.tensor(
                proposal,
                dtype=torch.int64,
            )
        self.device_input.copy_(self.host_input, non_blocking=True)
        start = self._event()
        end = self._event()
        start.record()
        targets = self.verify.fn(
            self.device_input,
            cache_position,
            rope_deltas,
            *flat_cache,
        )
        end.record()
        self.host_targets.copy_(targets, non_blocking=True)
        done = self._event()
        done.record()
        done.synchronize()
        return (
            [int(value) for value in self.host_targets[0].tolist()],
            float(start.elapsed_time(end)) / 1000.0,
        )

    def _decode_call(
        self,
        current_token: int,
        cache_position: torch.Tensor,
        rope_deltas: torch.Tensor,
        flat_cache: tuple[torch.Tensor, ...],
    ) -> tuple[int, float]:
        self.decode_input.fill_(int(current_token))
        start = self._event()
        end = self._event()
        start.record()
        logits = self.recognizer.decode_fn(
            self.decode_input,
            cache_position,
            rope_deltas,
            *flat_cache,
        )
        sampled = torch.argmax(logits[:, -1, :].float(), dim=-1, keepdim=True)
        end.record()
        self.host_decode_target.copy_(sampled, non_blocking=True)
        done = self._event()
        done.record()
        done.synchronize()
        return int(self.host_decode_target[0, 0]), float(start.elapsed_time(end)) / 1000.0

    @torch.inference_mode()
    def decode(
        self,
        prefilled: PrefilledRecognition,
        matcher: TableDraftMatcher,
        *,
        max_new_tokens: int | None = None,
    ) -> TableSpecDecodeResult:
        (
            cache,
            rope_deltas,
            cache_position,
            _first_token_tensor,
            cache_release,
        ) = prefilled.take_device_state()
        started = time.perf_counter()
        token_ids = [int(prefilled.first_token)]
        matcher.start(token_ids[0])
        tracker = ExactCycleTracker()
        tracker.update(token_ids[0])
        position = int(cache_position.detach().cpu().item())
        target_calls = 0
        speculative_calls = 0
        fully_accepted_speculative_calls = 0
        rejected_speculative_calls = 0
        fallback_calls = 0
        proposed = 0
        accepted = 0
        verify_device_s = 0.0
        fallback_device_s = 0.0
        stop_reason: str | None = (
            "eos" if token_ids[0] == self.eos_token_id else None
        )
        repetition: dict[str, Any] = {}
        limit = int(max_new_tokens or self.cache_length)
        flat_cache = cache.flat_tensors()

        def append_tokens(values: Iterable[int]) -> bool:
            nonlocal stop_reason, repetition
            for value in values:
                token = int(value)
                token_ids.append(token)
                if token == self.eos_token_id:
                    stop_reason = "eos"
                    return True
                evidence = tracker.update(token)
                if evidence is not None:
                    repetition = evidence.to_dict()
                    del token_ids[evidence.trim_length :]
                    stop_reason = "repetition"
                    return True
                if prefilled.input_tokens + len(token_ids) - 1 >= self.cache_length:
                    stop_reason = "kv_cache_full"
                    return True
                if len(token_ids) >= limit:
                    stop_reason = "length"
                    return True
            return False

        try:
            while stop_reason is None:
                proposal = matcher.propose(token_ids)
                can_verify = (
                    proposal is not None
                    and bool(proposal.tokens)
                    and position + self.query_length <= self.cache_length
                )
                cache_position.fill_(position)
                if not can_verify:
                    next_token, device_s = self._decode_call(
                        token_ids[-1],
                        cache_position,
                        rope_deltas,
                        flat_cache,
                    )
                    target_calls += 1
                    fallback_calls += 1
                    fallback_device_s += device_s
                    matcher.commit(
                        None,
                        accepted_draft_tokens=0,
                        emitted_tokens=(next_token,),
                    )
                    append_tokens((next_token,))
                    position += 1
                    continue

                assert proposal is not None
                proposal_tokens = proposal.tokens
                targets, device_s = self._verify_call(
                    token_ids[-1],
                    proposal_tokens,
                    cache_position,
                    rope_deltas,
                    flat_cache,
                )
                target_calls += 1
                speculative_calls += 1
                verify_device_s += device_s
                proposed += len(proposal_tokens)
                accepted_here = 0
                for draft_token, target_token in zip(proposal_tokens, targets):
                    if draft_token != target_token:
                        break
                    accepted_here += 1
                accepted += accepted_here
                if accepted_here == len(proposal_tokens):
                    fully_accepted_speculative_calls += 1
                else:
                    rejected_speculative_calls += 1
                emitted = list(proposal_tokens[:accepted_here])
                emitted.append(int(targets[accepted_here]))
                matcher.commit(
                    proposal,
                    accepted_draft_tokens=accepted_here,
                    emitted_tokens=emitted,
                )
                append_tokens(emitted)
                position += accepted_here + 1
        finally:
            if cache_release is not None:
                cache_release()

        text = self.recognizer.tokenizer.decode(
            token_ids,
            skip_special_tokens=prefilled.skip_special_tokens,
        )
        return TableSpecDecodeResult(
            token_ids=token_ids,
            text=text,
            stop_reason=str(stop_reason),
            target_calls=target_calls,
            speculative_calls=speculative_calls,
            fully_accepted_speculative_calls=fully_accepted_speculative_calls,
            rejected_speculative_calls=rejected_speculative_calls,
            fallback_calls=fallback_calls,
            proposed_draft_tokens=proposed,
            accepted_draft_tokens=accepted,
            verifier_device_s=verify_device_s,
            fallback_device_s=fallback_device_s,
            wall_s=time.perf_counter() - started,
            repetition=repetition,
        )
