"""Client-only concurrency limits, response-driven refill, and durable output."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "table_closed_loop_api_client", SCRIPTS / "table_closed_loop_api_client.py",
)
assert SPEC is not None and SPEC.loader is not None
CLIENT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLIENT)


def tables(count: int) -> list[dict]:
    return [{
        "request_id": str(i), "tail_rank": i + 1,
        "baseline_b1_latency_s": 1.0, "has_latex_markup": False,
    } for i in range(count)]


def args(limit: int) -> argparse.Namespace:
    return argparse.Namespace(
        api_url="http://unused/v1/ocr", client_label="test", set="a",
        request_timeout_s=1.0, start_at_epoch_s=None, max_in_flight=limit,
    )


class ClosedLoopClientTest(unittest.TestCase):
    def test_four_requests_refill_while_three_partners_wait(self) -> None:
        async def scenario(path):
            release_partners = asyncio.Event()
            active, peak = 0, 0

            async def post(*unused, source_request_id, **kwargs):
                nonlocal active, peak
                active += 1
                peak = max(peak, active)
                if int(source_request_id) < 3:
                    await asyncio.wait_for(release_partners.wait(), 2)
                elif source_request_id == "6":
                    saved = [json.loads(line) for line in path.read_text().splitlines()]
                    self.assertEqual([r["request_id"] for r in saved], ["3", "4", "5"])
                    release_partners.set()
                active -= 1
                return {"response": {"token_ids": [1, 2]}}

            with patch.object(CLIENT.load, "post_table_ocr", post):
                result = await CLIENT.run_closed_loop(
                    args(4), tables(7), {str(i): b"image" for i in range(7)}, path,
                )
            self.assertEqual(peak, 4)
            return result

        with tempfile.TemporaryDirectory() as directory:
            rows, _, _, stats = asyncio.run(scenario(Path(directory) / "results.jsonl"))
        self.assertEqual([r["request_id"] for r in rows[:4]], ["3", "4", "5", "6"])
        self.assertEqual(stats["observed_max_in_flight"], 4)
        self.assertEqual(stats["unsent_request_count"], 0)

    def test_two_requests_refill_without_waiting_for_slow_partner(self) -> None:
        async def scenario(path):
            release_first = asyncio.Event()
            active, peak = 0, 0
            dispatched = []

            async def post(*unused, source_request_id, **kwargs):
                nonlocal active, peak
                dispatched.append(source_request_id)
                active += 1
                peak = max(peak, active)
                if source_request_id == "0":
                    await asyncio.wait_for(release_first.wait(), 2)
                elif source_request_id == "3":
                    # Two earlier responses are durable while request 0 waits.
                    saved = [json.loads(line) for line in path.read_text().splitlines()]
                    self.assertEqual([row["request_id"] for row in saved], ["1", "2"])
                    release_first.set()
                active -= 1
                return {"response": {"token_ids": [1, 2]}}

            with patch.object(CLIENT.load, "post_table_ocr", post):
                output = await CLIENT.run_closed_loop(
                    args(2), tables(4), {str(i): b"image" for i in range(4)}, path,
                )
            self.assertEqual(dispatched, ["0", "1", "2", "3"])
            self.assertEqual(peak, 2)
            return output

        with tempfile.TemporaryDirectory() as directory:
            result, _, _, stats = asyncio.run(scenario(Path(directory) / "results.jsonl"))
        self.assertEqual([row["request_id"] for row in result], ["1", "2", "3", "0"])
        self.assertEqual(stats["observed_max_in_flight"], 2)
        self.assertEqual(stats["unsent_request_count"], 0)

    def test_one_request_is_strictly_sequential(self) -> None:
        async def post(*unused, **kwargs):
            await asyncio.sleep(0)
            return {"response": {}}

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(CLIENT.load, "post_table_ocr", post):
                results, _, _, stats = asyncio.run(CLIENT.run_closed_loop(
                    args(1), tables(4), {str(i): b"image" for i in range(4)},
                    Path(directory) / "results.jsonl",
                ))
        self.assertEqual([row["sequence"] for row in results], [1, 2, 3, 4])
        self.assertEqual(stats["observed_max_in_flight"], 1)
        self.assertTrue(all(row["active_after_response"] == 0 for row in results))

    def test_timeout_stops_new_submissions_and_drains_partner(self) -> None:
        async def post(*unused, source_request_id, **kwargs):
            await asyncio.sleep(0)
            if source_request_id == "0":
                raise TimeoutError("server may still be working")
            return {"response": {}}

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(CLIENT.load, "post_table_ocr", post):
                results, _, _, stats = asyncio.run(CLIENT.run_closed_loop(
                    args(2), tables(5), {str(i): b"image" for i in range(5)},
                    Path(directory) / "results.jsonl",
                ))
        self.assertEqual(len(results), 2)
        self.assertEqual(sum(row["status"] == "error" for row in results), 1)
        self.assertTrue(stats["stopped_sending_after_error"])
        self.assertEqual(stats["unsent_request_count"], 3)

    def test_limit_does_not_change_fixed_selection(self) -> None:
        records = [{"request_id": str(i), "worker_wall_s": float(i)} for i in range(666)]
        with patch.object(CLIENT.load, "read_jsonl", return_value=records):
            one = args(1)
            one.source_jsonl, one.count = Path("unused"), 32
            two = args(2)
            two.source_jsonl, two.count = Path("unused"), 32
            self.assertEqual(CLIENT.select_tables(one), CLIENT.select_tables(two))
            four = args(4)
            four.source_jsonl, four.count = Path("unused"), 32
            self.assertEqual(CLIENT.select_tables(one), CLIENT.select_tables(four))

    def test_invalid_limit_rejected(self) -> None:
        with self.assertRaises(ValueError):
            asyncio.run(CLIENT.run_closed_loop(args(0), [], {}, Path("unused")))


if __name__ == "__main__":
    unittest.main()
