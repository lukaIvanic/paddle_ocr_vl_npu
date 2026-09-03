import json
from pathlib import Path
import tempfile
import unittest

from compare_generation_traces import compare, canonical_table_placeholders, table_placeholder_equivalent


class ComparisonTests(unittest.TestCase):
    def test_placeholder_renaming_preserves_repeated_references(self):
        self.assertEqual(canonical_table_placeholders("[AC23] [KH45] [AC23]"),
                         canonical_table_placeholders("[ZY87] [DT46] [ZY87]"))
        self.assertNotEqual(canonical_table_placeholders("[AC23] [KH45] [AC23]"),
                            canonical_table_placeholders("[ZY87] [DT46] [DT46]"))

    def test_placeholder_allowance_requires_identical_final_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            left, right = Path(directory) / "left", Path(directory) / "right"
            for root in (left, right):
                for folder, suffix in (("predictions", ".md"), ("content_lists", ".json")):
                    (root / folder).mkdir(parents=True)
                    (root / folder / ("p" + suffix)).write_text("same")
            a = {"page": "p.png", "block_type": "table", "stop_reason": "eos", "raw_text": "<fcel>[AC23]<nl>"}
            b = dict(a, raw_text="<fcel>[KH45]<nl>")
            self.assertTrue(table_placeholder_equivalent(a, b, left, right))
            (right / "content_lists/p.json").write_text("changed")
            self.assertFalse(table_placeholder_equivalent(a, b, left, right))

    def test_exact_and_missing_crop_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            left, right = Path(directory) / "left", Path(directory) / "right"
            layout = {"request_id": "p:layout", "page": "p.png", "phase": "layout",
                      "generated_token_ids": [1, 9], "raw_text": "layout", "stop_reason": "eos"}
            crop = {"request_id": "p:recognition:0", "page": "p.png", "phase": "recognition",
                    "generated_token_ids": [2, 9], "raw_text": "text", "stop_reason": "eos"}
            for root in (left, right):
                (root / "predictions").mkdir(parents=True)
                (root / "predictions/p.md").write_text("text")
                (root / "generation_trace.jsonl").write_text("\n".join(map(json.dumps, [layout, crop])))
            result = compare(left, right)
            self.assertEqual(result["counts"]["recognition_token_exact"], 1)
            self.assertFalse(result["differences"])
            (right / "generation_trace.jsonl").write_text(json.dumps(layout))
            result = compare(left, right)
            self.assertEqual(result["missing_requests_with_unchanged_layout"], ["p:recognition:0"])

    def test_changed_layout_input_is_hard_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            left, right = Path(directory) / "left", Path(directory) / "right"
            record = {"request_id": "p:layout", "page": "p.png", "phase": "layout",
                      "generated_token_ids": [1, 9], "raw_text": "layout", "stop_reason": "eos"}
            for root, prompt in ((left, [1]), (right, [2])):
                root.mkdir()
                (root / "generation_trace.jsonl").write_text(json.dumps(dict(record, prompt_token_ids=prompt)))
            self.assertEqual(compare(left, right)["unexpected_input_changes"], ["p:layout"])


if __name__ == "__main__":
    unittest.main()
