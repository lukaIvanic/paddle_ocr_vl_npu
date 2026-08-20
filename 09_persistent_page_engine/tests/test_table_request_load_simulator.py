"""Tests for the open-loop table request load simulator."""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


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

    def test_send_and_receive_for_one_request_use_the_same_identifier(self) -> None:
        for style in MODULE.EVENT_STYLE_CHOICES:
            send = MODULE.style_event_line("SEND", 17, style, enabled=True)
            receive = MODULE.style_event_line("RECV", 17, style, enabled=True)
            self.assertIn("17", send)
            self.assertIn("17", receive)
            self.assertEqual(
                send.split("m", 1)[0],
                receive.split("m", 1)[0],
            )

    def test_first_48_request_tags_are_unique(self) -> None:
        for style in MODULE.EVENT_STYLE_CHOICES:
            tags = {
                MODULE.event_tag(sequence, style, enabled=True)
                for sequence in range(1, 49)
            }
            self.assertEqual(len(tags), 48)

    def test_same_background_family_has_four_distinct_patterns(self) -> None:
        rendered = [
            MODULE.style_event_line(
                "SEND table=preview",
                sequence,
                "background-pattern",
                enabled=True,
                line_width=80,
            )
            for sequence in (1, 13, 25, 37)
        ]
        self.assertEqual(len(set(rendered)), 4)
        first_background_codes = [line.split("m", 1)[0] for line in rendered]
        self.assertEqual(len(set(first_background_codes)), 1)

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


if __name__ == "__main__":
    unittest.main()
