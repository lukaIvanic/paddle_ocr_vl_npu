"""Token-ID selection policies applied directly to model logits."""

from __future__ import annotations

import torch


TOKEN_SELECTION_GREEDY = "greedy"
TOKEN_SELECTION_SUPPRESS_MATH_OPEN_GREEDY = "suppress_math_open_greedy"
TOKEN_SELECTION_SUPPRESS_MATH_OPEN_AND_SLASH_GREEDY = (
    "suppress_math_open_and_slash_greedy"
)
TOKEN_SELECTION_PREFER_MATH_OPEN_TOP2_NON_NESTED = (
    "prefer_math_open_top2_non_nested"
)
TOKEN_SELECTION_PREFER_MATH_OPEN_TOP2_FIRST_OVERRIDE = (
    "prefer_math_open_top2_first_override"
)
TOKEN_SELECTION_PREFER_MATH_OPEN_PROBABILITY_NEAR_TOP = (
    "prefer_math_open_probability_near_top"
)
TOKEN_SELECTION_PREFER_MATH_OPEN_VARIANTS_TOP2_P10 = (
    "prefer_math_open_variants_top2_p10"
)
TOKEN_SELECTION_PREFER_MATH_OPEN_ADJUSTERS_COMBINED = (
    "prefer_math_open_adjusters_combined"
)
TOKEN_SELECTION_CHOICES = (
    TOKEN_SELECTION_GREEDY,
    TOKEN_SELECTION_SUPPRESS_MATH_OPEN_GREEDY,
    TOKEN_SELECTION_SUPPRESS_MATH_OPEN_AND_SLASH_GREEDY,
    TOKEN_SELECTION_PREFER_MATH_OPEN_TOP2_NON_NESTED,
    TOKEN_SELECTION_PREFER_MATH_OPEN_TOP2_FIRST_OVERRIDE,
    TOKEN_SELECTION_PREFER_MATH_OPEN_PROBABILITY_NEAR_TOP,
    TOKEN_SELECTION_PREFER_MATH_OPEN_VARIANTS_TOP2_P10,
    TOKEN_SELECTION_PREFER_MATH_OPEN_ADJUSTERS_COMBINED,
)


def select_token_ids(
    logits: torch.Tensor,
    *,
    mode: str = TOKEN_SELECTION_GREEDY,
    preferred_token_id: int | None = None,
    alternate_preferred_token_id: int | None = None,
    policy_mask: torch.Tensor | None = None,
    legacy_policy_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select one token ID per logits row without decoding or re-encoding text.

    The result has ``logits.shape[:-1]``. ``policy_mask`` must broadcast to
    that shape. False rows retain ordinary greedy selection.
    """

    mode = str(mode)
    if mode not in TOKEN_SELECTION_CHOICES:
        raise ValueError(
            f"token selection must be one of {TOKEN_SELECTION_CHOICES}, got {mode!r}"
        )
    scores = logits.float()
    if mode == TOKEN_SELECTION_GREEDY:
        return torch.argmax(scores, dim=-1)
    if preferred_token_id is None:
        raise ValueError(f"{mode} requires preferred_token_id")
    if int(scores.shape[-1]) < 2:
        raise ValueError("top-2 token selection requires a vocabulary of at least 2")
    if not 0 <= int(preferred_token_id) < int(scores.shape[-1]):
        raise ValueError(
            f"preferred_token_id {preferred_token_id} is outside vocabulary "
            f"size {scores.shape[-1]}"
        )

    greedy = torch.argmax(scores, dim=-1)
    if mode == TOKEN_SELECTION_SUPPRESS_MATH_OPEN_GREEDY:
        top2 = torch.topk(scores, k=2, dim=-1).indices
        replacement = torch.where(
            top2[..., 0] == int(preferred_token_id),
            top2[..., 1],
            top2[..., 0],
        )
        selected = torch.where(
            greedy == int(preferred_token_id),
            replacement,
            greedy,
        )
        if policy_mask is not None:
            selected = torch.where(
                policy_mask.to(device=selected.device, dtype=torch.bool),
                selected,
                greedy,
            )
        return selected
    if mode == TOKEN_SELECTION_SUPPRESS_MATH_OPEN_AND_SLASH_GREEDY:
        if alternate_preferred_token_id is None:
            raise ValueError(f"{mode} requires alternate_preferred_token_id")
        alternate = int(alternate_preferred_token_id)
        if not 0 <= alternate < int(scores.shape[-1]):
            raise ValueError(
                f"alternate_preferred_token_id {alternate} is outside vocabulary "
                f"size {scores.shape[-1]}"
            )
        top3 = torch.topk(scores, k=3, dim=-1).indices
        allowed = (
            (top3 != int(preferred_token_id))
            & (top3 != alternate)
        )
        first_allowed = torch.argmax(allowed.to(dtype=torch.int64), dim=-1)
        replacement = top3.gather(-1, first_allowed.unsqueeze(-1)).squeeze(-1)
        selected = torch.where(
            (greedy == int(preferred_token_id)) | (greedy == alternate),
            replacement,
            greedy,
        )
        if policy_mask is not None:
            selected = torch.where(
                policy_mask.to(device=selected.device, dtype=torch.bool),
                selected,
                greedy,
            )
        return selected

    preferred = torch.full_like(greedy, int(preferred_token_id))
    if mode == TOKEN_SELECTION_PREFER_MATH_OPEN_PROBABILITY_NEAR_TOP:
        probabilities = torch.softmax(scores, dim=-1)
        top_probability = probabilities.gather(
            -1, greedy.unsqueeze(-1)
        ).squeeze(-1)
        preferred_probability = probabilities[..., int(preferred_token_id)]
        eligible = (
            (preferred_probability >= 0.3 * top_probability)
            & (preferred_probability > 0.10)
        )
    elif mode in (
        TOKEN_SELECTION_PREFER_MATH_OPEN_VARIANTS_TOP2_P10,
        TOKEN_SELECTION_PREFER_MATH_OPEN_ADJUSTERS_COMBINED,
    ):
        if alternate_preferred_token_id is None:
            raise ValueError(f"{mode} requires alternate_preferred_token_id")
        probabilities = torch.softmax(scores, dim=-1)
        top2 = torch.topk(scores, k=2, dim=-1).indices
        primary_eligible = (
            (top2 == int(preferred_token_id)).any(dim=-1)
            & (probabilities[..., int(preferred_token_id)] > 0.10)
        )
        alternate_eligible = (
            (top2 == int(alternate_preferred_token_id)).any(dim=-1)
            & (probabilities[..., int(alternate_preferred_token_id)] > 0.10)
        )
        use_alternate = alternate_eligible & (
            ~primary_eligible
            | (
                scores[..., int(alternate_preferred_token_id)]
                > scores[..., int(preferred_token_id)]
            )
        )
        selected = torch.where(
            use_alternate,
            torch.full_like(greedy, int(alternate_preferred_token_id)),
            torch.where(primary_eligible, preferred, greedy),
        )
        if mode == TOKEN_SELECTION_PREFER_MATH_OPEN_ADJUSTERS_COMBINED:
            top_probability = probabilities.gather(
                -1, greedy.unsqueeze(-1)
            ).squeeze(-1)
            legacy_primary = (
                (probabilities[..., int(preferred_token_id)] >= 0.3 * top_probability)
                & (probabilities[..., int(preferred_token_id)] > 0.10)
            )
            variant_selected = selected
            if policy_mask is not None:
                variant_selected = torch.where(
                    policy_mask.to(device=greedy.device, dtype=torch.bool),
                    variant_selected,
                    greedy,
                )
            legacy_mask = (
                torch.ones_like(greedy, dtype=torch.bool)
                if legacy_policy_mask is None
                else legacy_policy_mask.to(device=greedy.device, dtype=torch.bool)
            )
            return torch.where(
                legacy_mask & legacy_primary,
                preferred,
                variant_selected,
            )
        if policy_mask is not None:
            selected = torch.where(
                policy_mask.to(device=selected.device, dtype=torch.bool),
                selected,
                greedy,
            )
        return selected
    else:
        top2 = torch.topk(scores, k=2, dim=-1).indices
        eligible = (top2 == int(preferred_token_id)).any(dim=-1)
    selected = torch.where(eligible, preferred, greedy)
    if policy_mask is not None:
        selected = torch.where(
            policy_mask.to(device=selected.device, dtype=torch.bool),
            selected,
            greedy,
        )
    return selected
