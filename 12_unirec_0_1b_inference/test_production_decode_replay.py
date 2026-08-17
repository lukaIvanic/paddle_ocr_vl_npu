#!/usr/bin/env python3
"""CPU-only tests for production UniRec decode artifact replay."""

from __future__ import annotations

import json
import tempfile
import unittest
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from production_decode_replay import compare_completions, load_artifact


def write_artifact(root: Path, lengths: list[int]) -> None:
    arrays = [
        np.arange(12 * 1 * 6 * length * 128, dtype=np.float16).reshape(
            12, 1, 6, length, 128
        )
        for length in lengths
    ]
    offsets: list[int] = []
    with (root / "cross_kv.bin").open("wb") as handle:
        for array in arrays:
            padding = (-handle.tell()) % 64
            handle.write(b"\0" * padding)
            offsets.append(handle.tell())
            handle.write(memoryview(array).cast("B"))
    rows = []
    for index, (array, length, offset) in enumerate(
        zip(arrays, lengths, offsets)
    ):
        checksum = zlib.crc32(memoryview(array).cast("B")) & 0xFFFFFFFF
        rows.append(
            {
                "format": "unirec_cross_kv_v1",
                "request_id": f"page_000000_crop_{index:04d}",
                "page_index": 0,
                "crop_index": index,
                "label": "text_01",
                "cross_kv": {
                    "file": "cross_kv.bin",
                    "offset": offset,
                    "nbytes": array.nbytes,
                    "shape": list(array.shape),
                    "dtype": array.dtype.str,
                    "crc32": f"{checksum:08x}",
                    "source_length": length,
                },
                "prefill": {
                    "prep": {"image": "/tmp/crop.png", "prepare_total_s": 0.1},
                    "prefill_s": 0.2,
                    "actual_cross_attention_length": length,
                    "text_prefill_real_source_tokens": length,
                    "text_prefill_physical_source_tokens": 32,
                },
            }
        )
    (root / "crops.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (root / "summary.json").write_text(
        json.dumps(
            {
                "format": "unirec_cross_kv_v1",
                "status": "ok",
                "artifact": {"crop_count": len(rows)},
            }
        ),
        encoding="utf-8",
    )


class ProductionDecodeReplayTest(unittest.TestCase):
    def test_loads_writable_contiguous_real_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_artifact(root, [3, 5])
            artifact = load_artifact(
                root,
                cross_cache_length=8,
                verify_crc=True,
                prefault=True,
            )
            self.assertEqual([crop.source_length for crop in artifact.crops], [3, 5])
            self.assertEqual(len(artifact.skipped_rows), 0)
            self.assertTrue(artifact.crops[0].packed_cross_kv.flags.writeable)
            self.assertTrue(artifact.crops[0].packed_cross_kv.flags.c_contiguous)

    def test_lower_capacity_requires_explicit_skip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_artifact(root, [3, 5])
            with self.assertRaisesRegex(ValueError, "exceed cross-KV capacity"):
                load_artifact(root, cross_cache_length=4, prefault=False)
            artifact = load_artifact(
                root,
                cross_cache_length=4,
                over_capacity="skip",
                prefault=False,
            )
            self.assertEqual([crop.source_length for crop in artifact.crops], [3])
            self.assertEqual(len(artifact.skipped_rows), 1)

    def test_reference_token_comparison(self) -> None:
        completed = [
            SimpleNamespace(
                request_id="a",
                result={"generated_ids": [1, 2, 3]},
            ),
            SimpleNamespace(
                request_id="b",
                result={"generated_ids": [4, 5]},
            ),
        ]
        comparison = compare_completions(
            completed,
            {
                "a": {"token_ids": [1, 2, 3]},
                "b": {"token_ids": [4, 6]},
            },
        )
        self.assertEqual(comparison["compared_rows"], 2)
        self.assertEqual(comparison["length_exact_count"], 2)
        self.assertEqual(comparison["token_exact_count"], 1)
        self.assertEqual(comparison["first_mismatches"][0]["request_id"], "b")


if __name__ == "__main__":
    unittest.main()
