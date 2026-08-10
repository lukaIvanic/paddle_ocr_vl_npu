#!/usr/bin/env python3
"""CPU-only round-trip checks for the UniRec cross-KV artifact."""

from __future__ import annotations

import tempfile
import unittest
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import numpy as np

from prefill_artifact import CrossKvArtifactWriter, read_crop_array, read_jsonl


class CrossKvArtifactTest(unittest.TestCase):
    def test_shared_page_round_trip(self) -> None:
        expected = np.arange(12 * 1 * 6 * 7 * 64, dtype=np.float16).reshape(
            12, 1, 6, 7, 64
        )
        image = np.zeros((4, 5, 3), dtype=np.uint8)
        crop = np.zeros((2, 3, 3), dtype=np.uint8)
        image_offset = 0
        crop_offset = image.nbytes
        kv_offset = (crop_offset + crop.nbytes + 63) // 64 * 64
        storage = SharedMemory(create=True, size=kv_offset + expected.nbytes)
        np.ndarray(image.shape, image.dtype, storage.buf, image_offset)[:] = image
        np.ndarray(crop.shape, crop.dtype, storage.buf, crop_offset)[:] = crop
        np.ndarray(expected.shape, expected.dtype, storage.buf, kv_offset)[:] = expected
        payload = {
            "page_index": 3,
            "image_path": "/tmp/page.png",
            "shared_memory": {"name": storage.name, "nbytes": storage.size},
            "image_bgr_descriptor": {
                "offset": image_offset,
                "shape": list(image.shape),
                "dtype": image.dtype.str,
                "nbytes": image.nbytes,
            },
            "width": 5,
            "height": 4,
            "layout_results": {"boxes": []},
            "blocks": [],
            "vlm_block_ids": [0],
            "drop_figures_set": [],
            "frontend_timing_s": {"layout_s": 0.1},
            "cross_capacity_rejected_crops": 0,
            "crops": [
                {
                    "crop_index": 0,
                    "label": "text_01",
                    "figure_token_map": {},
                    "image_rgb_descriptor": {
                        "offset": crop_offset,
                        "shape": list(crop.shape),
                        "dtype": crop.dtype.str,
                        "nbytes": crop.nbytes,
                    },
                    "worker_cross_kv_descriptor": {
                        "offset": kv_offset,
                        "shape": list(expected.shape),
                        "dtype": expected.dtype.str,
                        "nbytes": expected.nbytes,
                    },
                    "worker_prefill_metadata": {
                        "prep": {"processed_image_size": [3, 2]},
                        "prefill_s": 0.2,
                        "cache_d2h_s": 0.01,
                        "prefill_device_stage_s": None,
                        "text_prefill_execution": "compiled_packed_s1024",
                        "text_prefill_real_source_tokens": 7,
                        "text_prefill_physical_source_tokens": 1024,
                        "actual_cross_attention_length": 7,
                    },
                }
            ],
        }
        storage.close()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                output_dir = Path(temporary) / "artifact"
                writer = CrossKvArtifactWriter(output_dir)
                writer.add_page(payload)
                summary = writer.finish({"status": "ok"})
                rows = read_jsonl(output_dir / "crops.jsonl")
                actual = read_crop_array(output_dir, rows[0], verify_crc=True)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(summary["artifact"]["page_count"], 1)
                self.assertEqual(summary["artifact"]["crop_count"], 1)
                self.assertEqual(summary["artifact"]["real_source_tokens"], 7)
                self.assertEqual(
                    summary["artifact"]["cross_kv_payload_bytes"], expected.nbytes
                )
        finally:
            try:
                storage.unlink()
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    unittest.main()
