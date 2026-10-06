import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_smoke_data import ALLOWED_SOURCE, body, normalize, ranking_metrics, select_groups


class TrainingSmokeDataTest(unittest.TestCase):
    def source(self):
        return {"provenance": {"source_allowlist": [ALLOWED_SOURCE]}, "rows": [
            {"source_config": ALLOWED_SOURCE, "source_row_index": i,
             "query": f"question {i}", "pos": [f"positive {i}"],
             "neg": [f"negative {i}/{j}" for j in range(4)]} for i in range(12)]}

    def test_disjoint_split_and_normalized_duplicates(self):
        source = self.source()
        duplicate = dict(source["rows"][0], query=" QUESTION   0 ", source_row_index=99)
        source["rows"].append(duplicate)
        source["rows"][1]["neg"].append(source["rows"][0]["pos"][0])
        split = select_groups(source, lambda q, d: True, train_count=5, eval_count=3)
        self.assertEqual(len(split["train"]), 5)
        tq = {normalize(r["query"]) for r in split["train"]}
        eq = {normalize(r["query"]) for r in split["eval"]}
        td = {normalize(d) for r in split["train"] for d in r["documents"]}
        ed = {normalize(d) for r in split["eval"] for d in r["documents"]}
        self.assertFalse(tq & eq)
        self.assertFalse(td & ed)
        self.assertEqual(split, select_groups(source, lambda q, d: True, train_count=5, eval_count=3))

    def test_source_guard(self):
        source = self.source()
        source["rows"][0]["source_config"] = "hotpotqa"
        with self.assertRaises(ValueError):
            select_groups(source, lambda q, d: True, train_count=5, eval_count=3)

    def test_prompt_roles_are_preserved(self):
        self.assertEqual(body("task", "question", "passage", "document_first"),
                         "<Instruct>: task\n<Document>: passage\n<Query>: question")
        self.assertEqual(body("task", "question", "passage", "query_first"),
                         "<Instruct>: task\n<Query>: question\n<Document>: passage")

    def test_ranking_uses_logit_differences(self):
        groups = [{"documents": ["positive", "negative", "negative"]}]
        logits = [[100, 102], [-100, -99], [3, 6]]
        metrics = ranking_metrics(groups, logits)
        self.assertEqual(metrics["correct_orderings"], 1)
        self.assertEqual(metrics["top1_accuracy"], 0)
        self.assertEqual(metrics["mrr"], .5)
        shifted = [[no + 50, yes + 50] for no, yes in logits]
        self.assertEqual(metrics, ranking_metrics(groups, shifted))

    def test_ties_do_not_favour_positive_first_index(self):
        metrics = ranking_metrics([{"documents": ["positive", "negative"]}],
                                  [[0, 1], [2, 3]])
        self.assertEqual(metrics["ties"], 1)
        self.assertEqual(metrics["ordering_accuracy"], 0)
        self.assertEqual(metrics["top1_accuracy"], 0)
        self.assertEqual(metrics["mrr"], .5)


if __name__ == "__main__":
    unittest.main()
