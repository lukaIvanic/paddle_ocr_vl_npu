from __future__ import annotations

import unittest

from dual_lane_decode_policy import (
    DecodeLaneSpec,
    DecodeLaneStatus,
    choose_lane,
    route_lane,
)


def status(
    name: str,
    *,
    active: int,
    capacity: int = 128,
    queued: int = 1,
    step_ms: float | None = None,
    skipped: int = 0,
) -> DecodeLaneStatus:
    return DecodeLaneStatus(
        name=name,
        capacity=capacity,
        active_slots=active,
        queued_items=queued,
        mean_step_ms=step_ms,
        skipped_quanta=skipped,
    )


class DualLaneDecodePolicyTest(unittest.TestCase):
    def test_lane_spec_rejects_oversized_max_length(self) -> None:
        with self.assertRaises(ValueError):
            DecodeLaneSpec("a", 128, 256, 256, 257)

    def test_cross_boundary_routes_exactly(self) -> None:
        self.assertEqual(route_lane(cross_length=384, a_cross_capacity=384), "a")
        self.assertEqual(route_lane(cross_length=385, a_cross_capacity=384), "b")

    def test_full_lanes_use_round_robin(self) -> None:
        a = status("a", active=128, step_ms=4.0)
        b = status("b", active=128, step_ms=6.0)
        self.assertEqual(
            choose_lane(
                a,
                b,
                round_robin_next="b",
                max_skipped_quanta=8,
            ),
            "b",
        )

    def test_partial_lanes_compare_useful_tokens_per_ms(self) -> None:
        a = status("a", active=64, step_ms=4.0)
        b = status("b", active=96, step_ms=8.0)
        self.assertEqual(
            choose_lane(
                a,
                b,
                round_robin_next="b",
                max_skipped_quanta=8,
            ),
            "a",
        )

    def test_starvation_bound_overrides_rate(self) -> None:
        a = status("a", active=128, step_ms=4.0)
        b = status("b", active=4, step_ms=8.0, skipped=8)
        self.assertEqual(
            choose_lane(
                a,
                b,
                round_robin_next="a",
                max_skipped_quanta=8,
            ),
            "b",
        )

    def test_empty_lane_never_selected(self) -> None:
        a = status("a", active=0, queued=0)
        b = status("b", active=3, queued=0, step_ms=8.0)
        self.assertEqual(
            choose_lane(
                a,
                b,
                round_robin_next="a",
                max_skipped_quanta=8,
            ),
            "b",
        )


if __name__ == "__main__":
    unittest.main()
