"""Token-ID selection policies applied directly to model logits."""

from __future__ import annotations

import torch


TOKEN_SELECTION_GREEDY = "greedy"
TOKEN_SELECTION_PREFER_MATH_OPEN_TOP2_NON_NESTED = (
    "prefer_math_open_top2_non_nested"
)
TOKEN_SELECTION_PREFER_MATH_OPEN_TOP2_FIRST_OVERRIDE = (
    "prefer_math_open_top2_first_override"
)
TOKEN_SELECTION_CHOICES = (
    TOKEN_SELECTION_GREEDY,
    TOKEN_SELECTION_PREFER_MATH_OPEN_TOP2_NON_NESTED,
    TOKEN_SELECTION_PREFER_MATH_OPEN_TOP2_FIRST_OVERRIDE,
)


def select_token_ids(
    logits: torch.Tensor,
    *,
    mode: str = TOKEN_SELECTION_GREEDY,
    preferred_token_id: int | None = None,
    policy_mask: torch.Tensor | None = None,
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

    top2 = torch.topk(scores, k=2, dim=-1).indices
    greedy = top2[..., 0]
    preferred = torch.full_like(greedy, int(preferred_token_id))
    selected = torch.where(
        (top2 == int(preferred_token_id)).any(dim=-1),
        preferred,
        greedy,
    )
    if policy_mask is not None:
        selected = torch.where(
            policy_mask.to(device=selected.device, dtype=torch.bool),
            selected,
            greedy,
        )
    return selected
