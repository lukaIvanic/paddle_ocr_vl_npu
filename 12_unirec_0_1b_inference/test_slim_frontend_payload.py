#!/usr/bin/env python3
"""CPU-only checks for image-free worker-prefill payloads."""

from __future__ import annotations

import unittest

import numpy as np

from layout_process_pool import _pack_frontend_payload_shared
from run_opendoc_batched_unirec import (
    page_request_from_process_payload,
    release_page_frontend_storage,
)


class SlimFrontendPayloadTest(unittest.TestCase):
    def test_cross_kv_round_trip_without_page_or_crop_images(self) -> None:
        page_image = np.zeros((8, 12, 3), dtype=np.uint8)
        crop_image = np.zeros((4, 5, 3), dtype=np.uint8)
        cross_kv = np.arange(12 * 1 * 6 * 7 * 64, dtype=np.float16).reshape(
            12, 1, 6, 7, 64
        )
        payload = {
            "page_index": 3,
            "image_path": "/tmp/page.png",
            "image_bgr": page_image,
            "width": 12,
            "height": 8,
            "layout_results": {"boxes": []},
            "blocks": [],
            "vlm_block_ids": [0],
            "drop_figures_set": [],
            "started_at": 1.0,
            "frontend_timing_s": {"layout_s": 0.1},
            "cross_capacity_rejected_crops": 0,
            "crops": [
                {
                    "crop_index": 0,
                    "label": "text_01",
                    "figure_token_map": {},
                    "image_rgb": crop_image,
                    "worker_cross_kv": cross_kv,
                    "worker_prefill_metadata": {
                        "actual_cross_attention_length": 7,
                    },
                }
            ],
        }

        packed, _pack_s, retained_bytes = _pack_frontend_payload_shared(
            payload,
            retain_images=False,
        )
        self.assertEqual(retained_bytes, cross_kv.nbytes)
        self.assertNotIn("image_bgr_descriptor", packed)
        self.assertNotIn("image_rgb_descriptor", packed["crops"][0])
        self.assertEqual(packed["crops"][0]["source_image_size"], [5, 4])

        page = page_request_from_process_payload(packed, measured_layout_s=0.1)
        try:
            self.assertIsNone(page.image)
            self.assertIsNone(page.crops[0].image)
            self.assertEqual(page.crops[0].image_size, (5, 4))
            np.testing.assert_array_equal(page.crops[0].worker_cross_kv, cross_kv)
        finally:
            release_page_frontend_storage(page)


if __name__ == "__main__":
    unittest.main()
