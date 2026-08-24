#!/usr/bin/env python3
"""CPU-only checks for image-free worker-prefill payloads."""

from __future__ import annotations

import unittest
import multiprocessing as mp
import threading
import time

import numpy as np

from layout_process_pool import _pack_frontend_payload_shared
from run_opendoc_batched_unirec import (
    PageCrossKvAdmissionTracker,
    page_request_from_process_payload,
    release_page_frontend_storage,
)
from continuous_unirec import ContinuousReadyItem, ContinuousWorkerPrefilledItem
from shared_byte_budget import SharedByteBudget


class SlimFrontendPayloadTest(unittest.TestCase):
    def test_shared_byte_budget_blocks_before_oversubscription(self) -> None:
        budget = SharedByteBudget(mp.get_context("spawn"), 100)
        budget.reserve(80)
        acquired = threading.Event()

        def reserve_tail() -> None:
            budget.reserve(30)
            acquired.set()

        thread = threading.Thread(target=reserve_tail)
        thread.start()
        time.sleep(0.02)
        self.assertFalse(acquired.is_set())
        self.assertEqual(budget.snapshot()["peak_bytes"], 80)
        budget.release(80)
        thread.join(timeout=1.0)
        self.assertTrue(acquired.is_set())
        self.assertLessEqual(budget.snapshot()["peak_bytes"], 100)
        budget.release(30)
        self.assertEqual(budget.snapshot()["live_bytes"], 0)

    def test_ready_item_releases_source_once(self) -> None:
        calls: list[str] = []
        item = ContinuousWorkerPrefilledItem(
            packed_cross_kv=np.zeros((2, 1, 1, 1, 1), dtype=np.float16),
            prep={},
            prefill_s=0.0,
            actual_cross_attention_length=1,
        )
        ready = ContinuousReadyItem(
            request_id="test",
            payload=None,
            prefilled=item,
            on_admitted=lambda: calls.append("released"),
        )
        ready.release_source_after_admission()
        ready.release_source_after_admission()
        self.assertEqual(calls, ["released"])

    def test_zero_crop_page_needs_no_shared_arena(self) -> None:
        payload = {
            "page_index": 0,
            "image_path": "/tmp/empty.png",
            "image_bgr": np.zeros((2, 3, 3), dtype=np.uint8),
            "width": 3,
            "height": 2,
            "layout_results": {"boxes": []},
            "blocks": [],
            "vlm_block_ids": [],
            "drop_figures_set": [],
            "started_at": 1.0,
            "frontend_timing_s": {"layout_s": 0.1},
            "cross_capacity_rejected_crops": 0,
            "crops": [],
        }
        packed, _pack_s, retained_bytes = _pack_frontend_payload_shared(
            payload,
            retain_images=False,
        )
        self.assertEqual(retained_bytes, 0)
        self.assertIsNone(packed["shared_memory"])
        page = page_request_from_process_payload(packed, measured_layout_s=0.1)
        self.assertIsNone(page.image)
        self.assertEqual(page.crops, [])
        self.assertTrue(page.is_ready())

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

    def test_page_budget_releases_on_last_arena_admission(self) -> None:
        cross_kv = np.zeros((12, 1, 6, 7, 64), dtype=np.float16)
        payload = {
            "page_index": 4,
            "image_path": "/tmp/page.png",
            "image_bgr": np.zeros((8, 12, 3), dtype=np.uint8),
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
                    "image_rgb": np.zeros((4, 5, 3), dtype=np.uint8),
                    "worker_cross_kv": cross_kv,
                    "worker_prefill_metadata": {
                        "actual_cross_attention_length": 7,
                    },
                }
            ],
        }
        budget = SharedByteBudget(mp.get_context("spawn"), cross_kv.nbytes)
        packed, _pack_s, retained_bytes = _pack_frontend_payload_shared(
            payload,
            retain_images=False,
            byte_budget=budget,
        )
        self.assertEqual(budget.snapshot()["live_bytes"], retained_bytes)
        page = page_request_from_process_payload(
            packed,
            measured_layout_s=0.1,
            shared_byte_budget=budget,
        )
        tracker = PageCrossKvAdmissionTracker(page)
        item = ContinuousWorkerPrefilledItem(
            packed_cross_kv=page.crops[0].worker_cross_kv,
            prep={},
            prefill_s=0.0,
            actual_cross_attention_length=7,
        )
        tracker.release_crop(page.crops[0], item)
        self.assertIsNone(item.packed_cross_kv)
        self.assertIsNone(page.crops[0].worker_cross_kv)
        self.assertIsNone(page.frontend_storage_lease)
        snapshot = budget.snapshot()
        self.assertEqual(snapshot["live_bytes"], 0)
        self.assertEqual(snapshot["peak_bytes"], retained_bytes)
        self.assertEqual(snapshot["reservation_count"], 1)
        self.assertEqual(snapshot["release_count"], 1)


if __name__ == "__main__":
    unittest.main()
