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
        marker = MODULE.event_marker(17, enabled=True)
        send = MODULE.format_event_line("SEND", 17, enabled=True)
        receive = MODULE.format_event_line("RECV", 17, enabled=True)
        self.assertTrue(send.startswith(marker))
        self.assertTrue(receive.startswith(marker))

    def test_first_48_request_markers_are_unique(self) -> None:
        markers = {
            MODULE.event_marker(sequence, enabled=True)
            for sequence in range(1, 49)
        }
        self.assertEqual(len(markers), 48)

    def test_a_b_c_d_lanes_use_doubled_offsets(self) -> None:
        identities = [MODULE.event_identity(sequence) for sequence in (1, 13, 25, 37)]
        lanes = [identity[0] for identity in identities]
        leading_spaces = [identity[1] for identity in identities]
        backgrounds = [identity[2] for identity in identities]
        self.assertEqual(lanes, ["A", "B", "C", "D"])
        self.assertEqual(leading_spaces, [0, 16, 32, 48])
        self.assertEqual(len(set(backgrounds)), 1)

    def test_only_marker_has_background_color(self) -> None:
        rendered = MODULE.format_event_line("SEND table=preview", 1, enabled=True)
        self.assertEqual(rendered.count("\033["), 2)
        reset_end = rendered.index(MODULE.ANSI_RESET) + len(MODULE.ANSI_RESET)
        self.assertEqual(rendered[reset_end:], " SEND table=preview")

    def test_plain_logs_keep_lane_marker_and_spacing(self) -> None:
        rendered = [
            MODULE.format_event_line("SEND", sequence, enabled=False)
            for sequence in (1, 13, 25, 37)
        ]
        leading_spaces = [len(line) - len(line.lstrip(" ")) for line in rendered]
        self.assertEqual(leading_spaces, [0, 16, 32, 48])
        self.assertEqual(
            [line.strip().split(" ", 1)[0] for line in rendered],
            ["[A]", "[B]", "[C]", "[D]"],
        )

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
