"""CPU draft matching and B1 target-side speculative table decode."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import re
import time
from typing import Any, Iterable

import torch

from ..model.text_spec_verify import TextSpecVerifyRuntime
from ..model.token_selection import (
    TOKEN_SELECTION_GREEDY,
    TOKEN_SELECTION_SUPPRESS_MATH_OPEN_AND_SLASH_GREEDY,
    TOKEN_SELECTION_SUPPRESS_MATH_OPEN_GREEDY,
    TOKEN_SELECTION_PREFER_MATH_OPEN_VARIANTS_TOP2_P10,
    TOKEN_SELECTION_PREFER_MATH_OPEN_ADJUSTERS_COMBINED,
    select_token_ids,
)
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


@dataclass(frozen=True)
class WrapperRescueCandidate:
    """One draft-led cell-opening override that passed strict alignment guards."""

    draft_index: int
    draft_token: int
    target_row: int
    target_column: int
    target_width: int | None
    draft_lane: int
    draft_logical_row: int
    draft_column: int
    draft_row_width: int
    previous_cell_match: str
    previous_target_cell: tuple[int, ...]
    previous_draft_cell: tuple[int, ...]
    draft_prefix_text: str


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


def _previous_cell(
    tokens: list[int],
    *,
    boundary_index: int,
    cell_tokens: set[int],
) -> tuple[int, ...] | None:
    """Return content IDs before a cell marker at ``boundary_index``."""

    if boundary_index <= 0 or tokens[boundary_index] not in cell_tokens:
        return None
    for index in range(boundary_index - 1, -1, -1):
        if tokens[index] in cell_tokens:
            return tuple(tokens[index + 1 : boundary_index])
    return None


def _outer_math_wrapper_key(tokenizer: Any, tokens: tuple[int, ...]) -> str:
    """Decode IDs once and remove only a complete outer ``\\(...\\)`` pair."""

    text = str(tokenizer.decode(list(tokens), skip_special_tokens=False))
    if text.startswith(r"\(") and text.endswith(r"\)"):
        return text[2:-2]
    return text


def _formula_content_key(text: str) -> str:
    """Remove only common formula presentation syntax for a decoded-ID guard."""

    value = text.replace(r"\(", "").replace(r"\)", "").replace("$", "")
    value = re.sub(r"\^\{\{(\*+)\}\}", r"\1", value)
    value = re.sub(r"\^\{(\*+)\}", r"\1", value)
    value = value.replace("{", "").replace("}", "")
    return "".join(value.split())


def _target_position(
    tokens: list[int],
    *,
    cell_tokens: set[int],
    newline_token: int,
) -> tuple[int, int, int | None]:
    """Compute the current logical row, column, and modal completed-row width."""

    row = 0
    column = -1
    widths: Counter[int] = Counter()
    for token in tokens:
        if token in cell_tokens:
            column += 1
        if token == newline_token:
            widths[column + 1] += 1
            row += 1
            column = -1
    width = widths.most_common(1)[0][0] if widths else None
    return row, column, width


def wrapper_rescue_candidate(
    token_ids: list[int],
    matcher: "TableDraftMatcher",
    proposal: DraftProposal,
    *,
    accepted_before_rejection: int,
    tokenizer: Any,
    formula_previous_only: bool = False,
) -> WrapperRescueCandidate | None:
    """Return a strict previous-cell-aligned draft math-open candidate.

    Generated IDs are decoded only for two narrow prefix/wrapper checks. They
    are never encoded back to IDs.
    """

    accepted = int(accepted_before_rejection)
    draft_index = int(proposal.start) + accepted
    if accepted < 0 or accepted >= len(proposal.tokens):
        return None
    if draft_index <= 0 or draft_index >= len(matcher.draft):
        return None

    prospective_prefix = [
        *[int(token) for token in token_ids],
        *[int(token) for token in proposal.tokens[:accepted]],
    ]
    if not prospective_prefix:
        return None
    target_boundary = len(prospective_prefix) - 1
    draft_boundary = draft_index - 1
    target_marker = prospective_prefix[target_boundary]
    if target_marker not in matcher.cell_tokens:
        return None
    if matcher.draft[draft_boundary] != target_marker:
        return None

    previous_target = _previous_cell(
        prospective_prefix,
        boundary_index=target_boundary,
        cell_tokens=matcher.cell_tokens,
    )
    previous_draft = _previous_cell(
        matcher.draft,
        boundary_index=draft_boundary,
        cell_tokens=matcher.cell_tokens,
    )
    if previous_target is None or previous_draft is None:
        return None
    previous_target_text = str(
        tokenizer.decode(list(previous_target), skip_special_tokens=False)
    )
    previous_draft_text = str(
        tokenizer.decode(list(previous_draft), skip_special_tokens=False)
    )
    if formula_previous_only:
        if not any(marker in previous_draft_text for marker in (r"\(", "$", "^")):
            return None
        if _formula_content_key(previous_target_text) != _formula_content_key(
            previous_draft_text
        ):
            return None
        previous_cell_match = "formula_content"
    elif previous_target == previous_draft:
        previous_cell_match = "exact_ids"
    elif _outer_math_wrapper_key(
        tokenizer, previous_target
    ) == _outer_math_wrapper_key(tokenizer, previous_draft):
        previous_cell_match = "outer_math_wrapper_only"
    else:
        return None

    target_row, target_column, target_width = _target_position(
        prospective_prefix,
        cell_tokens=matcher.cell_tokens,
        newline_token=matcher.newline_token,
    )
    draft_meta = matcher.metadata[draft_index]
    if target_column <= 0 or draft_meta.column <= 0:
        return None
    if target_column != draft_meta.column:
        return None
    if target_width is not None and target_width != draft_meta.row_width:
        return None

    draft_probe = matcher.draft[draft_index : draft_index + 2]
    draft_prefix_text = str(
        tokenizer.decode(draft_probe, skip_special_tokens=False)
    )
    if not draft_prefix_text.startswith(r"\("):
        return None
    draft_token = int(matcher.draft[draft_index])
    if draft_token in matcher.cell_tokens or draft_token == matcher.newline_token:
        return None
    return WrapperRescueCandidate(
        draft_index=draft_index,
        draft_token=draft_token,
        target_row=target_row,
        target_column=target_column,
        target_width=target_width,
        draft_lane=draft_meta.lane,
        draft_logical_row=draft_meta.logical_row,
        draft_column=draft_meta.column,
        draft_row_width=draft_meta.row_width,
        previous_cell_match=previous_cell_match,
        previous_target_cell=previous_target,
        previous_draft_cell=previous_draft,
        draft_prefix_text=draft_prefix_text,
    )


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
    wrapper_rescue_probe_device_s: float
    wall_s: float
    wrapper_rescue: dict[str, Any]
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
        wrapper_rescue: bool = False,
        wrapper_rescue_top_k: int = 3,
        wrapper_rescue_formula_previous: bool = False,
    ) -> None:
        if int(recognizer.batch_size) != 1:
            raise ValueError("table speculative decode currently requires B1")
        if recognizer.token_selection not in (
            TOKEN_SELECTION_GREEDY,
            TOKEN_SELECTION_SUPPRESS_MATH_OPEN_GREEDY,
            TOKEN_SELECTION_SUPPRESS_MATH_OPEN_AND_SLASH_GREEDY,
            TOKEN_SELECTION_PREFER_MATH_OPEN_VARIANTS_TOP2_P10,
            TOKEN_SELECTION_PREFER_MATH_OPEN_ADJUSTERS_COMBINED,
        ):
            raise NotImplementedError(
                "non-greedy token selection is generation-only until the "
                "speculative verifier implements the identical policy"
            )
        self.recognizer = recognizer
        self.device = recognizer.device
        self.draft_length = int(draft_length)
        self.query_length = self.draft_length + 1
        self.cache_length = int(recognizer.cache_length)
        self.eos_token_id = int(recognizer.model.config.eos_token_id)
        self.wrapper_rescue = bool(wrapper_rescue)
        self.wrapper_rescue_top_k = int(wrapper_rescue_top_k)
        self.wrapper_rescue_formula_previous = bool(wrapper_rescue_formula_previous)
        if self.wrapper_rescue_top_k <= 0:
            raise ValueError("wrapper_rescue_top_k must be positive")
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
            token_selection=recognizer.token_selection,
            preferred_token_id=recognizer.math_open_token_id,
            alternate_preferred_token_id=recognizer.math_slash_token_id,
            cell_start_token_ids=recognizer.table_cell_token_ids,
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
        self.host_probe_ids = torch.empty(
            (1, self.wrapper_rescue_top_k), dtype=torch.int64, pin_memory=True
        )
        self.host_probe_logits = torch.empty(
            (1, self.wrapper_rescue_top_k), dtype=torch.float32, pin_memory=True
        )
        self.host_probe_probabilities = torch.empty(
            (1, self.wrapper_rescue_top_k), dtype=torch.float32, pin_memory=True
        )

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
        policy_mask = torch.tensor(
            [
                self.recognizer.token_selection in (
                    TOKEN_SELECTION_SUPPRESS_MATH_OPEN_GREEDY,
                    TOKEN_SELECTION_SUPPRESS_MATH_OPEN_AND_SLASH_GREEDY,
                )
                or int(current_token) in self.recognizer.table_cell_token_ids
            ],
            device=logits.device,
            dtype=torch.bool,
        )
        sampled = select_token_ids(
            logits[:, -1, :].float(),
            mode=self.recognizer.token_selection,
            preferred_token_id=self.recognizer.math_open_token_id,
            alternate_preferred_token_id=self.recognizer.math_slash_token_id,
            policy_mask=policy_mask,
            legacy_policy_mask=torch.ones(
                (1,), device=logits.device, dtype=torch.bool
            ),
        ).view(-1, 1)
        end.record()
        self.host_decode_target.copy_(sampled, non_blocking=True)
        done = self._event()
        done.record()
        done.synchronize()
        return int(self.host_decode_target[0, 0]), float(start.elapsed_time(end)) / 1000.0

    def _probe_topk_call(
        self,
        current_token: int,
        cache_position: torch.Tensor,
        rope_deltas: torch.Tensor,
        flat_cache: tuple[torch.Tensor, ...],
    ) -> tuple[list[dict[str, float | int]], float]:
        """Return live full-vocabulary softmax top-k for one decode position."""

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
        scores = logits[:, -1, :].float()
        top_logits, top_ids = torch.topk(
            scores,
            k=self.wrapper_rescue_top_k,
            dim=-1,
        )
        probabilities = torch.softmax(scores, dim=-1).gather(-1, top_ids)
        end.record()
        self.host_probe_ids.copy_(top_ids, non_blocking=True)
        self.host_probe_logits.copy_(top_logits, non_blocking=True)
        self.host_probe_probabilities.copy_(probabilities, non_blocking=True)
        done = self._event()
        done.record()
        done.synchronize()
        rows = [
            {
                "rank": rank + 1,
                "token_id": int(self.host_probe_ids[0, rank]),
                "logit": float(self.host_probe_logits[0, rank]),
                "probability": float(self.host_probe_probabilities[0, rank]),
            }
            for rank in range(self.wrapper_rescue_top_k)
        ]
        return rows, float(start.elapsed_time(end)) / 1000.0

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
        wrapper_rescue_probe_device_s = 0.0
        wrapper_rescue_used = False
        wrapper_rescue_probes = 0
        wrapper_rescue_events: list[dict[str, Any]] = []
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
                rescue_candidate = None
                if (
                    self.wrapper_rescue
                    and not wrapper_rescue_used
                    and accepted_here < len(proposal_tokens)
                ):
                    rescue_candidate = wrapper_rescue_candidate(
                        token_ids,
                        matcher,
                        proposal,
                        accepted_before_rejection=accepted_here,
                        tokenizer=self.recognizer.tokenizer,
                        formula_previous_only=self.wrapper_rescue_formula_previous,
                    )
                if rescue_candidate is not None:
                    probe_position = position + accepted_here
                    cache_position.fill_(probe_position)
                    probe_current_token = (
                        int(proposal_tokens[accepted_here - 1])
                        if accepted_here > 0
                        else int(token_ids[-1])
                    )
                    topk, probe_device_s = self._probe_topk_call(
                        probe_current_token,
                        cache_position,
                        rope_deltas,
                        flat_cache,
                    )
                    target_calls += 1
                    wrapper_rescue_probes += 1
                    wrapper_rescue_probe_device_s += probe_device_s
                    selected_rank = next(
                        (
                            int(row["rank"])
                            for row in topk
                            if int(row["token_id"])
                            == rescue_candidate.draft_token
                        ),
                        None,
                    )
                    event = {
                        **asdict(rescue_candidate),
                        "proposal_start": int(proposal.start),
                        "accepted_before_rejection": accepted_here,
                        "base_rejected_token": int(targets[accepted_here]),
                        "probe_current_token": probe_current_token,
                        "probe_cache_position": probe_position,
                        "top_k": self.wrapper_rescue_top_k,
                        "topk": topk,
                        "selected_rank": selected_rank,
                        "applied": selected_rank is not None,
                    }
                    wrapper_rescue_events.append(event)
                    if selected_rank is not None:
                        wrapper_rescue_used = True
                        accepted += accepted_here
                        rejected_speculative_calls += 1
                        emitted = list(proposal_tokens[: accepted_here + 1])
                        matcher.commit(
                            proposal,
                            accepted_draft_tokens=accepted_here + 1,
                            emitted_tokens=emitted,
                        )
                        append_tokens(emitted)
                        position += accepted_here + 1
                        continue
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
            wrapper_rescue_probe_device_s=wrapper_rescue_probe_device_s,
            wall_s=time.perf_counter() - started,
            wrapper_rescue={
                "enabled": self.wrapper_rescue,
                "top_k": self.wrapper_rescue_top_k,
                "formula_previous_only": self.wrapper_rescue_formula_previous,
                "used": wrapper_rescue_used,
                "probe_calls": wrapper_rescue_probes,
                "forced_tokens": int(wrapper_rescue_used),
                "events": wrapper_rescue_events,
            },
            repetition=repetition,
        )
