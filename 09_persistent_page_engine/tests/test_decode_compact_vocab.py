from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from paddleocr_vl.model.text_decode import (  # noqa: E402
    LocalPaddleOCRVLStaticCache,
    load_decode_vocab_token_ids,
)
from paddleocr_vl.serving.continuous_decode import DecodeArena  # noqa: E402


def _digest(token_ids: list[int]) -> str:
    raw = json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def test_load_decode_vocab_uses_native_ids_without_a_tokenizer(tmp_path: Path) -> None:
    path = tmp_path / "decode_vocab.json"
    path.write_text(
        json.dumps(
            {
                "token_ids": [101309, 4, 93937],
                "token_ids_sha256": _digest([101309, 4, 93937]),
                "source": "native_generation_ids.jsonl",
            }
        ),
        encoding="utf-8",
    )

    token_ids, metadata = load_decode_vocab_token_ids(
        path,
        full_vocab_size=103424,
    )

    assert token_ids == (101309, 4, 93937)
    assert metadata["selected_vocab_size"] == 3
    assert metadata["source"] == "native_generation_ids.jsonl"


@pytest.mark.parametrize(
    "token_ids",
    ([1, 1], [-1, 2], [1, 3]),
)
def test_load_decode_vocab_rejects_invalid_ids(
    tmp_path: Path,
    token_ids: list[int],
) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"token_ids": token_ids}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_decode_vocab_token_ids(path, full_vocab_size=3)


def test_decode_arena_maps_compact_argmax_to_original_token_id() -> None:
    cache = LocalPaddleOCRVLStaticCache(
        key_caches=(torch.zeros((1, 1, 4, 1)),),
        value_caches=(torch.zeros((1, 1, 4, 1)),),
        cache_length=4,
    )
    arena = DecodeArena(
        cache=cache,
        device=torch.device("cpu"),
        batch_size=1,
        eos_token_id=2,
        decode_token_id_map=torch.tensor([101309, 4, 93937]),
    )

    step = arena.step(
        lambda *unused: torch.tensor([[[0.0, 1.0, 3.0]]]),
        iteration=0,
    )

    assert step.sampled.tolist() == [[93937]]
