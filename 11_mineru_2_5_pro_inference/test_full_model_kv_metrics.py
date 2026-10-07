"""Token-accounting checks; these are not inference or speed validation."""
import unittest
from bench_full_model_kv import token_metrics


class AccountingTests(unittest.TestCase):
    def test_eos_and_inactive_slots_are_not_useful_tokens(self):
        result = token_metrics([10,99], [[11,99],[99,99],[55,99]], 99, 8)
        self.assertEqual(result["token_ids"], [[10,11,99],[99]])
        self.assertEqual(result["output_tokens_excluding_eos"], 2)
        self.assertEqual(result["decode_tokens_excluding_prefill_and_eos"], 1)
        self.assertEqual(result["length_cap_hit_count"], 0)

    def test_capped_row_counts_only_generated_decode_outputs(self):
        result = token_metrics([10,20], [[11,99],[12,99]], 99, 3)
        self.assertEqual(result["decode_tokens_excluding_prefill_and_eos"], 2)
        self.assertEqual(result["output_tokens_excluding_eos"], 4)
        self.assertEqual(result["length_cap_hit_count"], 1)
        self.assertEqual(result["eos_count"], 1)


if __name__ == "__main__":
    unittest.main()
