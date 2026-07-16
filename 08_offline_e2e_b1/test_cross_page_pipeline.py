"""CPU tests for run-scoped page routing and immediate page emission."""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from PIL import Image

from pipeline import OfflinePagePipeline
from schema import (
    Box,
    ContinuousDecodeResult,
    LayoutRegion,
    RecognitionResult,
)


class FakeLayout:
    def predict(self, image: Image.Image):
        count = 2 if image.width == 20 else 1
        regions = [
            LayoutRegion(
                order=index,
                label_id=0,
                label="text",
                score=1.0,
                box=Box(0.0, 0.0, float(image.width), float(image.height)),
            )
            for index in range(count)
        ]
        return regions, {"layout_inference": 0.001, "layout_total": 0.001}


class FakeRecognizer:
    batch_size = 2

    def recognize_stream(self, requests, *, schedule_id, on_result):
        submitted = list(requests)
        by_id = {}
        for request in submitted:
            result = RecognitionResult(
                request_id=request.request_id,
                decode_schedule_id=schedule_id,
                decode_slot_index=0,
                decode_slot_epoch=1,
                layout_order=request.layout_order,
                label=request.label,
                prompt=request.prompt,
                box=request.box,
                crop_size=request.crop.size,
                text=f"text-{request.layout_order}",
                token_ids=[10, 2],
                stop_reason="eos",
                input_tokens=4,
                projected_image_tokens=1,
                generated_tokens_including_eos=2,
                decode_tokens_after_prefill_including_eos=1,
                decode_calls_executed=1,
                timing_s={"prefill_request_total": 0.01},
                device_stage_s={},
                rates={
                    "decode_effective_token_contribution_per_s": 1.0,
                    "request_output_tok_per_s": 2.0,
                },
            )
            by_id[request.request_id] = result

        completion_order = [submitted[-1], submitted[1], submitted[0]]
        for request in completion_order:
            on_result(by_id[request.request_id])

        request_count = len(submitted)
        schedule = ContinuousDecodeResult(
            schedule_id=schedule_id,
            batch_size=2,
            requests=request_count,
            ready_buffer_capacity=2,
            ready_buffer_low_watermark=1,
            max_ready_queue_depth=2,
            ready_source_refill_count=1,
            graph_calls=request_count,
            initial_admissions=2,
            hot_swap_admissions=max(0, request_count - 2),
            prefill_only_completions=0,
            raw_decode_token_slots=request_count * 2,
            active_decode_token_slots=request_count * 2,
            effective_decode_tokens=request_count,
            idle_decode_token_slots=0,
            lookahead_decode_token_slots=request_count,
            kv_prefix_bytes_copied=0,
            initial_kv_prefix_bytes_copied=0,
            hot_swap_kv_prefix_bytes_copied=0,
            timing_s={
                "continuous_decode_wall": 0.01,
                "decode_host_exclusive_wall": 0.01,
                "run_scoped_scheduler_wall": 0.02,
                "ready_source_wall": 0.01,
                "completion_callback_wall": 0.0,
                "decode_model_and_argmax_device": 0.005,
                "slot_admission_device": 0.0,
                "slot_admission_enqueue_wall": 0.0,
                "d2h_wait_wall": 0.0,
                "retire_and_refill_host_wall": 0.0,
            },
            rates={
                "raw_decode_tok_per_s": float(request_count * 200),
                "effective_decode_tok_per_s": float(request_count * 100),
                "effective_fraction": 0.5,
                "active_slot_fraction": 1.0,
                "effective_device_tok_per_s": float(request_count * 200),
                "scheduler_effective_tok_per_s": float(request_count * 50),
            },
        )
        return [by_id[item.request_id] for item in submitted], schedule


class CrossPagePipelineTest(unittest.TestCase):
    def test_page_crops_are_created_lazily_and_page_image_is_released(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "page.png"
            Image.new("RGB", (20, 10), "white").save(image_path)
            pipeline = OfflinePagePipeline(
                layout=FakeLayout(),
                recognizer=FakeRecognizer(),
                save_annotated=False,
            )

            work = pipeline._prepare_page(
                image_path,
                0,
                run_started=time.perf_counter(),
            )
            self.assertEqual(len(work.prepared_regions), 2)
            requests = list(pipeline._iter_page_requests(work))

        self.assertEqual(len(requests), 2)
        self.assertEqual([request.crop.size for request in requests], [(20, 10)] * 2)
        self.assertEqual(work.prepared_regions, [])
        self.assertIsNone(work.image)

    def test_emits_completed_page_without_changing_return_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.png"
            second = root / "second.png"
            Image.new("RGB", (20, 10), "white").save(first)
            Image.new("RGB", (10, 10), "white").save(second)

            emitted: list[str] = []
            pipeline = OfflinePagePipeline(
                layout=FakeLayout(),
                recognizer=FakeRecognizer(),
                save_annotated=False,
            )
            result = pipeline.run_pages(
                [first, second],
                on_page_completed=lambda page: emitted.append(page.page_id),
            )

        self.assertEqual(
            [page.page_id for page in result.pages],
            ["page_0000_first", "page_0001_second"],
        )
        self.assertEqual(
            emitted,
            ["page_0001_second", "page_0000_first"],
        )
        self.assertEqual(result.completion_order, emitted)
        self.assertEqual(
            result.pages[0].reading_order_text,
            "text-0\n\ntext-1",
        )
        self.assertEqual(result.decode_schedule.requests, 3)


if __name__ == "__main__":
    unittest.main()
