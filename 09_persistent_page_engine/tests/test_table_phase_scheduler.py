"""No torch/NPU needed for admission, fairness, and tail-accounting checks."""

import importlib.util
from pathlib import Path
import sys
import unittest
from itertools import combinations

PATH = Path(__file__).resolve().parents[1] / "paddleocr_vl/serving/table_phase_scheduler.py"
SPEC = importlib.util.spec_from_file_location("table_phase_policy_tested", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
PhaseWork = MODULE.PhaseWork
TablePhasePolicy = MODULE.TablePhasePolicy
PhaseLedger = MODULE.PhaseLedger


class PolicyTests(unittest.TestCase):
    def test_every_c4_slot_subset_has_correct_cover_and_holes(self):
        for size in range(1, 5):
            for slots in combinations(range(4), size):
                first, batch, holes = MODULE.covering_batch(slots, 4)
                self.assertIn(batch, (1, 2, 4))
                self.assertTrue(0 <= first <= 4-batch)
                self.assertEqual(set(slots) | set(holes), set(range(first, first+batch)))
                self.assertFalse(set(slots) & set(holes))
        self.assertEqual(MODULE.covering_batch([0,3],4), (0,4,(1,2)))
        self.assertEqual(MODULE.covering_batch([1,2],4), (1,2,()))
        self.assertEqual(MODULE.covering_batch([0,1],4), (0,2,()))

    def test_one_request_never_waits_for_a_partner(self):
        policy = TablePhasePolicy()
        for phase, q in (("draft", 1), ("verify", 8), ("ordinary", 1)):
            work = PhaseWork("a", phase, q)
            self.assertEqual(policy.choose([work]), [work])

    def test_matching_shapes_batch_without_shared_acceptance(self):
        policy = TablePhasePolicy()
        work = [PhaseWork("a", "verify", 8), PhaseWork("b", "verify", 8)]
        self.assertEqual(policy.choose(work), work)
        # A changes K, B does not. Neither is forced to follow the other.
        changed = [PhaseWork("a", "verify", 16), work[1]]
        self.assertEqual(len(policy.choose(changed)), 1)
        self.assertEqual(policy.choose(changed), [work[1]])

    def test_mixed_phases_alternate(self):
        policy = TablePhasePolicy()
        work = [PhaseWork("a", "draft"), PhaseWork("b", "verify", 8)]
        self.assertEqual([policy.choose(work)[0].request_id for _ in range(6)], list("ababab"))

    def test_new_prefill_is_immediate_then_decode_continues(self):
        policy = TablePhasePolicy()
        a = PhaseWork("a", "verify", 8)
        policy.choose([a])
        b = PhaseWork("b", "draft_prefill")
        self.assertEqual(policy.choose([a, b]), [b])
        self.assertEqual(policy.choose([a, PhaseWork("b", "draft")]), [a])

    def test_completion_does_not_wait_for_partner(self):
        policy = TablePhasePolicy()
        policy.choose([PhaseWork("a", "draft"), PhaseWork("b", "draft")])
        policy.retire("a")
        self.assertNotIn("a", policy.last_served)
        c = PhaseWork("c", "target_prefill")
        self.assertEqual(policy.choose([PhaseWork("b", "draft"), c]), [c])

    def test_duplicate_ready_states_rejected(self):
        with self.assertRaises(ValueError):
            TablePhasePolicy().choose([PhaseWork("a", "draft"), PhaseWork("a", "verify", 8)])


class LedgerTests(unittest.TestCase):
    def test_shared_work_counted_once_globally(self):
        ledger = PhaseLedger()
        for key in ("a", "b"):
            ledger.admit(key)
        ledger.record("verify_b2q8", owners=["a", "b"], phases={"a": "verify", "b": "verify"}, wall_s=.002, device_s=.0015, decode=True)
        self.assertEqual(ledger.summary()["action_host_wall_s"]["verify_b2q8"], .002)
        for key in ("a", "b"):
            row = ledger.retire(key)
            self.assertEqual(row["own_action_wall_s"]["verify_b2q8"], .002)
            self.assertEqual(row["decode_phase_combination_wall_s"]["verify+verify"], .002)

    def test_foreign_prefill_is_tail_wait_not_own_work(self):
        ledger = PhaseLedger()
        for key in ("a", "b"):
            ledger.admit(key)
        ledger.record("draft_prefill", owners=["b"], phases={"a": "verify", "b": "draft_prefill"}, wall_s=.1)
        self.assertEqual(ledger.retire("a")["other_action_wait_s"], {"draft_prefill": .1})
        self.assertEqual(ledger.retire("b")["own_action_wall_s"], {"draft_prefill": .1})


class NativeAcceptanceTests(unittest.TestCase):
    def test_rejection_commits_only_authoritative_prefix(self):
        self.assertEqual(MODULE.accept_native_proposal([1, 2, 3], [1, 9, 3, 4]), ([1, 9], 1))

    def test_full_acceptance_includes_bonus(self):
        self.assertEqual(MODULE.accept_native_proposal([1, 2], [1, 2, 9]), ([1, 2, 9], 2))

    def test_fallback_commits_one_native_id(self):
        self.assertEqual(MODULE.accept_native_proposal([], [101309]), ([101309], 0))

    def test_rows_advance_independently(self):
        a = MODULE.accept_native_proposal([1, 2], [1, 2, 7])
        b = MODULE.accept_native_proposal([3, 4], [9, 4, 8])
        self.assertEqual((len(a[0]), len(b[0])), (3, 1))

    def test_missing_bonus_rejected(self):
        with self.assertRaises(ValueError):
            MODULE.accept_native_proposal([1], [1])


if __name__ == "__main__":
    unittest.main()
