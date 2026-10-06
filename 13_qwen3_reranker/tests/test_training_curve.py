import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from curve_metrics import ndcg10
from mixture_data import clean_row, quotas, select_split, text_hash
from run_query_first_training_curve import plans


class TrainingCurveTest(unittest.TestCase):
    def test_touché_block_and_family_guard(self):
        row = {"query": "  SAME   Question ", "pos": ["answer"], "neg": ["negative"]}
        blocked = {"query_hashes": {text_hash("same question")}, "document_hashes": set()}
        self.assertIsNone(clean_row(row, "pubmed_qa_labeled_len-0-500", 0, blocked))
        with self.assertRaises(ValueError):
            clean_row(row, "msmarco_len-0-500", 0, blocked)

    def test_disjoint_mixture_and_capped_quotas(self):
        counts = {"a_len-0-500": 20, "b_len-0-500": 20}
        pool = []
        for s in ("a", "b"):
            for i in range(20):
                pool.append({"id": f"{s}/{i}", "config": s + "_len-0-500", "source": s,
                             "query": f"q{s}{i}", "positives": [f"p{s}{i}"],
                             "negatives": [f"n{s}{i}", "shared negative"]})
        split = select_split(pool, counts, 16, 8, 1)
        docs = lambda groups: {text_hash(d) for g in groups for d in g["documents"]}
        self.assertFalse(docs(split["train"]) & docs(split["validation"]))
        self.assertEqual(len(split["train"]), 16)
        self.assertEqual(quotas(8, {"a": 100, "b": 1}, {"a": 3, "b": 8}), {"a": 3, "b": 5})

    def test_ndcg_uses_full_qrels_and_linear_gains(self):
        value = ndcg10({"q": {"b": 2, "a": 1}}, {"q": {"a": 2, "b": 0, "missing": 1}})
        expected = (2 / math.log2(3)) / (2 + 1 / math.log2(3))
        self.assertAlmostEqual(value["ndcg10"], expected)

    def test_batch_budget_preserves_every_record(self):
        records = [{"index": i, "ids": [1] * n} for i, n in enumerate((100, 8000, 2000, 4000, 300))]
        batches = list(plans(records, 4, 8192))
        self.assertEqual(sorted(r["index"] for b in batches for r in b), list(range(5)))
        self.assertTrue(all(len(b) * math.ceil(max(len(r["ids"]) for r in b)/128)*128 <= 8192 for b in batches))


if __name__ == "__main__":
    unittest.main()
