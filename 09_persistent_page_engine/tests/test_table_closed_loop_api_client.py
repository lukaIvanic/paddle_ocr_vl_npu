"""Client-only concurrency limits, response-driven refill, and durable output."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
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
TEST_LIMITS = (1, 2, 3, 4, 5, 8, 16)


def tables(count: int) -> list[dict]:
    return [{
        "request_id": str(i), "tail_rank": i + 1,
        "baseline_b1_latency_s": 1.0, "has_latex_markup": False,
    } for i in range(count)]


def args(limit: int) -> argparse.Namespace:
    return argparse.Namespace(
        api_url="http://unused/v1/ocr", client_label="test", set="a",
        request_timeout_s=1.0, start_at_epoch_s=None, max_in_flight=limit,
        shuffle_seed=None,
    )


class ClosedLoopClientTest(unittest.TestCase):
    def test_goal_gate_manifests_are_frozen_and_reproduce_existing_selection(self) -> None:
        manifest = json.loads((SCRIPTS.parent / "presets/table_serving_goal_validation.json").read_text())
        source = SCRIPTS.parent.parent / manifest["source"]
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), manifest["source_sha256"])
        for gate in manifest["gates"]:
            selected_args = argparse.Namespace(source_jsonl=source, set="random",
                                               count=gate["count"], shuffle_seed=gate["seed"])
            rows = CLIENT.select_tables(selected_args)
            serialized = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                                 for row in rows).encode()
            self.assertEqual(hashlib.sha256(serialized).hexdigest(), gate["tables_jsonl_sha256"])
            self.assertEqual([row["request_id"] for row in rows], gate["dispatch_request_ids"])
        final_ids = manifest["gates"][-1]["dispatch_request_ids"]
        self.assertEqual(len(final_ids), 1000)
        self.assertEqual(len(set(final_ids[:665])), 665)
        self.assertEqual(len(set(final_ids[665:])), 335)
        self.assertLessEqual(set(final_ids[665:]), set(final_ids[:665]))

    def test_vllm_body_requests_native_ids_and_remaining_context(self) -> None:
        body = json.loads(CLIENT.vllm_payload(b"png", "PaddleOCR-VL-1.6"))
        self.assertTrue(body["return_token_ids"])
        self.assertEqual(body["temperature"], 0)
        self.assertNotIn("max_tokens", body)
        self.assertEqual(body["messages"][0]["content"][1]["text"], "Table Recognition:")
        self.assertEqual(CLIENT.vllm_endpoint("http://localhost:18081/v1/chat/completions", "/metrics"),
                         "http://localhost:18081/metrics")

    def test_vllm_uses_same_closed_loop_limit_and_saves_native_ids(self) -> None:
        async def post(*unused):
            await asyncio.sleep(0)
            return {"response": {"token_ids": [93980, 2]}}

        selected_args = args(2)
        selected_args.api_kind = "vllm"
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(CLIENT, "post_vllm", post):
                rows, _, _, stats = asyncio.run(CLIENT.run_closed_loop(
                    selected_args, tables(4), {str(i): b"json" for i in range(4)},
                    Path(directory) / "results.jsonl",
                ))
        self.assertEqual(stats["observed_max_in_flight"], 2)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["service_result"]["response"]["token_ids"] == [93980, 2] for row in rows))

    def test_large_limits_refill_while_other_partners_wait(self) -> None:
        async def scenario(path, limit):
            release_partners = asyncio.Event()
            active, peak = 0, 0

            async def post(*unused, source_request_id, **kwargs):
                nonlocal active, peak
                active += 1
                peak = max(peak, active)
                if int(source_request_id) < limit - 1:
                    await asyncio.wait_for(release_partners.wait(), 2)
                elif int(source_request_id) == limit + 2:
                    saved = [json.loads(line) for line in path.read_text().splitlines()]
                    self.assertEqual(
                        [r["request_id"] for r in saved],
                        [str(i) for i in range(limit - 1, limit + 2)],
                    )
                    release_partners.set()
                active -= 1
                return {"response": {"token_ids": [1, 2]}}

            with patch.object(CLIENT.load, "post_table_ocr", post):
                result = await CLIENT.run_closed_loop(
                    args(limit), tables(limit + 3),
                    {str(i): b"image" for i in range(limit + 3)}, path,
                )
            self.assertEqual(peak, limit)
            return result

        for limit in (3, 4, 5, 8, 16):
            with self.subTest(limit=limit), tempfile.TemporaryDirectory() as directory:
                rows, _, _, stats = asyncio.run(scenario(Path(directory) / "results.jsonl", limit))
                self.assertEqual(
                    [r["request_id"] for r in rows[:4]],
                    [str(i) for i in range(limit - 1, limit + 3)],
                )
                self.assertEqual(stats["observed_max_in_flight"], limit)
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
        for limit in (0, -1):
            with self.assertRaises(ValueError):
                asyncio.run(CLIENT.run_closed_loop(args(limit), [], {}, Path("unused")))

    def test_cli_admission_limit_is_independent_of_decode_shape(self) -> None:
        argv = ["client", "--api-url", "http://unused", "--output-dir", "unused",
                "--set", "random", "--shuffle-seed", "1", "--max-in-flight", "5"]
        with patch.object(sys, "argv", argv):
            self.assertEqual(CLIENT.parse_args().max_in_flight, 5)

    def test_fifty_table_set_is_fixed_and_disjoint_from_warmup(self) -> None:
        records = [{"request_id": str(i), "worker_wall_s": float(i)} for i in range(666)]
        with patch.object(CLIENT.load, "read_jsonl", return_value=records):
            selections = []
            for limit in TEST_LIMITS:
                selected_args = args(limit)
                selected_args.set = "p90"
                selected_args.source_jsonl, selected_args.count = Path("unused"), 50
                selections.append(CLIENT.select_tables(selected_args))
            self.assertTrue(all(s == selections[0] for s in selections))
            self.assertEqual([r["tail_rank"] for r in selections[0]], list(range(1, 51)))
            warm_args = args(1)
            warm_args.set = "warm"
            warm_args.source_jsonl, warm_args.count = Path("unused"), 1
            self.assertNotIn(CLIENT.select_tables(warm_args)[0]["request_id"],
                             {r["request_id"] for r in selections[0]})

    def test_shuffle_preserves_top_fifty_and_is_identical_across_limits(self) -> None:
        records = [{"request_id": str(i), "worker_wall_s": float(i)} for i in range(666)]
        with patch.object(CLIENT.load, "read_jsonl", return_value=records):
            shuffled = []
            for limit in TEST_LIMITS:
                selected_args = args(limit)
                selected_args.set, selected_args.count = "p90", 50
                selected_args.source_jsonl = Path("unused")
                selected_args.shuffle_seed = 1
                shuffled.append(CLIENT.select_tables(selected_args))
            self.assertTrue(all(rows == shuffled[0] for rows in shuffled))
            ranks = [row["tail_rank"] for row in shuffled[0]]
            self.assertEqual(sorted(ranks), list(range(1, 51)))
            self.assertNotEqual(ranks, sorted(ranks))
            self.assertEqual(CLIENT.select_tables(selected_args), shuffled[0])
            selected_args.shuffle_seed = 2
            self.assertNotEqual(CLIENT.select_tables(selected_args), shuffled[0])

    def test_random_sample_uses_all_tables_and_is_identical_across_limits(self) -> None:
        records = [{"request_id": str(i), "worker_wall_s": float(i)} for i in range(665)]
        with patch.object(CLIENT.load, "read_jsonl", return_value=records):
            selections = []
            for limit in TEST_LIMITS:
                selected_args = args(limit)
                selected_args.set, selected_args.count = "random", 100
                selected_args.source_jsonl = Path("unused")
                selected_args.shuffle_seed = 1
                selections.append(CLIENT.select_tables(selected_args))
            self.assertTrue(all(rows == selections[0] for rows in selections))
            ids = [row["request_id"] for row in selections[0]]
            self.assertEqual(len(set(ids)), 100)
            self.assertEqual(ids, [row["request_id"] for row in CLIENT.random.Random(1).sample(records, 100)])
            selected_args.count = 665
            self.assertEqual({row["request_id"] for row in CLIENT.select_tables(selected_args)},
                             {str(i) for i in range(665)})
            selected_args.count, selected_args.shuffle_seed = 100, None
            with self.assertRaisesRegex(ValueError, "requires --shuffle-seed"):
                CLIENT.select_tables(selected_args)

    def test_thousand_requests_cover_full_pool_before_repeating(self) -> None:
        records = [{"request_id": str(i), "worker_wall_s": float(i)} for i in range(665)]
        selected_args = args(2)
        selected_args.set, selected_args.count = "random", 1000
        selected_args.source_jsonl, selected_args.shuffle_seed = Path("unused"), 1
        with patch.object(CLIENT.load, "read_jsonl", return_value=records):
            selected = CLIENT.select_tables(selected_args)
            self.assertEqual(selected, CLIENT.select_tables(selected_args))
        ids = [row["request_id"] for row in selected]
        self.assertEqual(len(ids), 1000)
        self.assertEqual(set(ids[:665]), {str(i) for i in range(665)})
        self.assertEqual(len(set(ids[665:])), 335)
        self.assertTrue(all(ids.count(value) in (1, 2) for value in set(ids)))
        rng = CLIENT.random.Random(1)
        expected = rng.sample(records, 665) + rng.sample(records, 335)
        self.assertEqual(ids, [row["request_id"] for row in expected])
        with patch.object(CLIENT.load, "read_jsonl", return_value=[]):
            with self.assertRaisesRegex(ValueError, "no tables"):
                CLIENT.select_tables(selected_args)

    def test_repeated_table_requests_have_unique_api_ids(self) -> None:
        sent_ids = []

        async def post(url, request_id, *unused, **kwargs):
            sent_ids.append(request_id)
            await asyncio.sleep(0)
            return {"response": {"token_ids": [1, 2]}}

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(CLIENT.load, "post_table_ocr", post):
                rows, _, _, stats = asyncio.run(CLIENT.run_closed_loop(
                    args(2), tables(1) * 3, {"0": b"image"}, Path(directory) / "results.jsonl",
                ))
        self.assertEqual(len(set(sent_ids)), 3)
        self.assertEqual(len(rows), 3)
        self.assertEqual(stats["observed_max_in_flight"], 2)

    def test_random_sample_rejects_duplicate_source_ids(self) -> None:
        selected_args = args(1)
        selected_args.set, selected_args.count = "random", 1
        selected_args.source_jsonl, selected_args.shuffle_seed = Path("unused"), 1
        with patch.object(CLIENT.load, "read_jsonl", return_value=[
            {"request_id": "same", "worker_wall_s": 1.0},
            {"request_id": "same", "worker_wall_s": 2.0},
        ]):
            with self.assertRaisesRegex(ValueError, "duplicate"):
                CLIENT.select_tables(selected_args)


if __name__ == "__main__":
    unittest.main()
