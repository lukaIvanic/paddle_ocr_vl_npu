import unittest
from oracle_api import protection


class PolicyTest(unittest.TestCase):
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
