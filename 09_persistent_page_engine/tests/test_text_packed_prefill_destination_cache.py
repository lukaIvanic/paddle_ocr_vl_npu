from __future__ import annotations

import pytest

from paddleocr_vl.model.text_packed_prefill import PackedTextPrefillRuntime


def _runtime(*, buckets: tuple[int, ...], destination_cache_length: int):
    runtime = object.__new__(PackedTextPrefillRuntime)
    runtime.buckets = buckets
    runtime.max_members = 32
    runtime.destination_cache_length = destination_cache_length
    return runtime


def test_aggregate_pack_can_exceed_each_destination_cache() -> None:
    runtime = _runtime(buckets=(1536,), destination_cache_length=128)

    route = runtime.route((69,) * 20)

    assert route["real_text_tokens"] == 1380
    assert route["physical_text_tokens"] == 1536
    assert route["pack_members"] == 20


def test_each_segment_must_fit_its_destination_cache() -> None:
    runtime = _runtime(buckets=(1536,), destination_cache_length=128)

    with pytest.raises(ValueError, match="segment exceeds destination cache"):
        runtime.route((129, 64))
