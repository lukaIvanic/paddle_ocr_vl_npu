import unittest
from oracle_api import DecodeCadence, protection


class PolicyTest(unittest.TestCase):
    def test_cadence_excludes_prefill_and_idle(self):
        clock = DecodeCadence()
        for i in range(33):
            clock.step(i*.001)
        self.assertAlmostEqual(clock.step_s, .001)
        clock.interrupted = True
        clock.step(1.0)
        self.assertEqual(clock.count, 32)
        self.assertLess(max(clock.samples), .002)
        clock.step(1.001)
        self.assertEqual(clock.count, 33)

    def test_decision_uses_observed_cadence(self):
        self.assertEqual(protection(100, .01, [("a", .21, 1351)], .00130), ["a"])
        self.assertEqual(protection(100, .01, [("a", .21, 1351)], .00135), [])

    def test_threatened_running_request_with_short_newcomer(self):
        self.assertEqual(protection(100, .01, [("a", 1.8, 100)]), ["a"])

    def test_cannot_save_already_late_request(self):
        self.assertEqual(protection(100, .01, [("a", 2.1, 100)]), [])

    def test_newcomer_deadline_expires(self):
        self.assertEqual(protection(100, 1.8, [("a", 1.8, 100)]), [])

    def test_exhausted_or_missing_running_estimate_falls_back(self):
        for remaining in (0, -1, None):
            self.assertEqual(protection(100, .01, [("a", 1.8, remaining)]), [])

    def test_missing_or_long_newcomer_falls_back(self):
        for tokens in (None, 256, 300):
            self.assertEqual(protection(tokens, .01, [("a", 1.8, 100)]), [])

    def test_no_active_request_cannot_deadlock_initial_fill(self):
        self.assertEqual(protection(100, .01, []), [])


if __name__ == "__main__":
    unittest.main()
