"""Tests for the open-loop table request load simulator."""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/table_request_load_simulator.py"
)
SPEC = importlib.util.spec_from_file_location("table_request_load_simulator", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TableRequestLoadSimulatorTest(unittest.TestCase):
    def test_freeze_tail_cohort_excludes_first_record(self) -> None:
        records = [
            {"request_id": f"table_{index}", "worker_wall_s": float(index)}
            for index in range(101)
        ]
        cohort = MODULE.freeze_tail_cohort(records, "p90")
        self.assertEqual(len(cohort), 10)
        self.assertEqual(cohort[0]["request_id"], "table_100")
        self.assertEqual(cohort[-1]["request_id"], "table_91")
        self.assertNotIn("table_0", {row["request_id"] for row in cohort})

    def test_poisson_schedule_is_deterministic_and_open_loop(self) -> None:
        cohort = [{"request_id": f"table_{index}"} for index in range(4)]
        first = MODULE.make_schedule(cohort, qps=10.0, duration_s=2.0, seed=7)
        second = MODULE.make_schedule(cohort, qps=10.0, duration_s=2.0, seed=7)
        self.assertEqual(MODULE.schedule_rows(first), MODULE.schedule_rows(second))
        offsets = [item.scheduled_offset_s for item in first]
        self.assertEqual(offsets, sorted(offsets))
        self.assertTrue(all(offset < 2.0 for offset in offsets))

    def test_poisson_schedule_can_use_a_fixed_request_count(self) -> None:
        cohort = [{"request_id": f"table_{index}"} for index in range(4)]
        schedule = MODULE.make_schedule(
            cohort,
            qps=2.0,
            duration_s=0.01,
            seed=7,
            max_requests=50,
        )
        self.assertEqual(len(schedule), 50)
        self.assertEqual(
            [item.sequence for item in schedule],
            list(range(1, 51)),
        )
        offsets = [item.scheduled_offset_s for item in schedule]
        self.assertEqual(offsets, sorted(offsets))
        self.assertGreater(offsets[-1], 0.01)

    def test_requests_overlap_and_each_result_is_written(self) -> None:
        schedule = [
            MODULE.ScheduledRequest(
                sequence=index + 1,
                scheduled_offset_s=index * 0.01,
                table={
                    "request_id": f"table_{index}",
                    "has_latex_markup": False,
                },
            )
            for index in range(3)
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                results, stats = asyncio.run(
                    MODULE.run_schedule(
                        schedule,
                        ocr_time_s=0.05,
                        result_handle=handle,
                        print_events=False,
                    )
                )
            self.assertEqual(len(results), 3)
            self.assertEqual(stats["max_active"], 3)
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 3)
            self.assertTrue(all(row["latency_s"] >= 0.045 for row in results))

    def test_async_http_client_posts_table_crop(self) -> None:
        payload = bytes(
            '{"worker_wall_s":1.25,"http_wall_s":1.3,'
            '"output_tokens":123,"stop_reason":"eos"}',
            "utf-8",
        )

        class FakeReader:
            def __init__(self) -> None:
                self.lines = [
                    b"HTTP/1.1 200 OK\r\n",
                    b"Content-Type: application/json\r\n",
                    f"Content-Length: {len(payload)}\r\n".encode("ascii"),
                    b"\r\n",
                ]

            async def readline(self) -> bytes:
                return self.lines.pop(0)

            async def readexactly(self, length: int) -> bytes:
                self.assert_length = length
                return payload

            async def read(self) -> bytes:
                return payload

        class FakeWriter:
            def __init__(self) -> None:
                self.writes: list[bytes] = []

            def write(self, value: bytes) -> None:
                self.writes.append(value)

            async def drain(self) -> None:
                return None

            def close(self) -> None:
                return None

            async def wait_closed(self) -> None:
                return None

        reader = FakeReader()
        writer = FakeWriter()
        open_connection = AsyncMock(return_value=(reader, writer))
        with patch.object(MODULE.asyncio, "open_connection", open_connection):
            result = asyncio.run(
                MODULE.post_table_ocr(
                    "http://127.0.0.1:8767/v1/ocr",
                    "table_1",
                    b"png-bytes",
                    timeout_s=2.0,
                    source_request_id="source_table_1",
                )
            )
        self.assertEqual(result["http_status"], 200)
        self.assertEqual(result["worker_wall_s"], 1.25)
        self.assertEqual(result["output_tokens"], 123)
        self.assertIn(b"crop_type=table", writer.writes[0])
        self.assertIn(b"source_request_id=source_table_1", writer.writes[0])
        self.assertEqual(writer.writes[1], b"png-bytes")

    def test_http_runtime_metrics_aggregate_prefill_and_decode(self) -> None:
        response = {
            "timing_s": {
                "cpu_preprocess_background_service": 0.1,
                "prefill_request_total": 0.2,
                "decode_ready_queue_wait": 0.3,
                "decode_slot_residency": 0.5,
            },
            "device_stage_s": {
                "vision_prefill": 0.04,
                "text_prefill": 0.01,
            },
            "vision": {
                "real_vision_tokens": 900,
                "physical_vision_tokens": 1024,
            },
            "text_prefill": {
                "real_text_tokens": 200,
                "physical_text_tokens": 256,
            },
            "decode_calls_executed": 400,
            "decode_tokens_after_prefill_including_eos": 399,
        }
        metrics = MODULE._http_runtime_metrics(
            [
                {
                    "status": "ok",
                    "service_result": {"response": response},
                }
            ],
            api_configuration={"batch_size": 8},
            run_wall_s=1.0,
        )
        ordinary = metrics["ordinary"]
        self.assertEqual(
            ordinary["stages"]["vision_prefill"]["real_tok_per_s"],
            22500.0,
        )
        self.assertEqual(
            ordinary["stages"]["text_prefill"]["physical_tok_per_s"],
            25600.0,
        )
        self.assertEqual(
            ordinary["decode"]["slot_residency_fraction_over_run_wall"],
            0.0625,
        )
        self.assertEqual(
            ordinary["decode"]["active_decode_token_iterations"],
            400,
        )

    def test_http_runtime_metrics_aggregate_speculative_stages(self) -> None:
        stage = {
            "timing_s": {"prefill_request_total": 0.1},
            "device_stage_s": {
                "vision_prefill": 0.04,
                "text_prefill": 0.01,
            },
            "vision": {
                "real_vision_tokens": 900,
                "physical_vision_tokens": 1024,
            },
            "text_prefill": {
                "real_text_tokens": 200,
                "physical_text_tokens": 256,
            },
        }
        response = {
            "route_lane": "spec",
            "runtime_metrics": {
                "draft": {
                    "rows": [stage],
                    "schedule": {
                        "batch_size": 8,
                        "requests": 8,
                        "graph_calls": 10,
                        "raw_decode_token_slots": 80,
                        "active_decode_token_slots": 60,
                        "effective_decode_tokens": 59,
                        "idle_decode_token_slots": 20,
                        "lookahead_decode_token_slots": 1,
                        "timing_s": {
                            "continuous_decode_wall": 0.02,
                            "decode_model_and_argmax_device": 0.019,
                        },
                    },
                },
                "target_prefill": stage,
                "verifier": {
                    "target_calls": 2,
                    "speculative_calls": 1,
                    "fallback_calls": 1,
                    "proposed_draft_tokens": 7,
                    "accepted_draft_tokens": 6,
                    "output_tokens_after_prefill": 7,
                    "verifier_device_s": 0.0,
                    "wall_s": 0.01,
                    "per_k": {"7": {"calls": 1}},
                },
            },
        }
        metrics = MODULE._http_runtime_metrics(
            [
                {
                    "status": "ok",
                    "service_result": {"response": response},
                }
            ],
            api_configuration={"lane": "height_routed_u8_adaptive_k"},
            run_wall_s=1.0,
        )
        speculative = metrics["speculative"]
        self.assertEqual(
            speculative["draft_decode"]["rates"]["active_slot_fraction"],
            0.75,
        )
        self.assertEqual(
            speculative["verifier"]["physical_verifier_tokens"], 9
        )
        self.assertEqual(
            speculative["verifier"]["accepted_fraction_of_proposed"],
            6 / 7,
        )


if __name__ == "__main__":
    unittest.main()
