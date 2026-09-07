"""CPU-only tests of report accounting; these do not validate NPU inference."""
import json
from pathlib import Path
import tempfile
import unittest

from report_live_pipeline import compare_saved


class ReportParityTests(unittest.TestCase):
    def test_ordered_inputs_ignore_sparse_block_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            live, saved = Path(tmp) / "live", Path(tmp) / "saved"
            base = dict(phase="recognition", page="page.jpg", block_index=0,
                        block_type="text", bbox=[0, 0, 1, 1], image_sha256="pixels",
                        chat_prompt="OCR:", prompt_token_ids=[1], generated_token_ids=[2])
            for root, index in [(live, 3), (saved, 0)]:
                (root / "output/predictions").mkdir(parents=True)
                (root / "output/predictions/page.md").write_text("same")
                (root / "output/generation_trace.jsonl").write_text(
                    json.dumps({**base, "block_index": index}) + "\n")
            result = compare_saved(live, saved)
            self.assertEqual(result["markdown_exact_pages"], 1)
            self.assertEqual(result["input_exact_recognition_pages"], 1)
            self.assertEqual(result["generated_ids_exact_requests_on_input_exact_pages"], 1)
            # A different crop must not be counted as a numerical generation difference.
            (live / "output/generation_trace.jsonl").write_text(
                json.dumps({**base, "image_sha256": "different"}) + "\n")
            result = compare_saved(live, saved)
            self.assertEqual(result["input_exact_recognition_pages"], 0)
            self.assertEqual(len(result["input_mismatch_pages"]), 1)
            self.assertEqual(result["requests_on_input_exact_pages"], 0)

    def test_duplicate_trace_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "output/predictions").mkdir(parents=True)
            row = json.dumps(dict(phase="recognition", page="p", block_index=0)) + "\n"
            (root / "output/generation_trace.jsonl").write_text(row * 2)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                compare_saved(root, root)


if __name__ == "__main__":
    unittest.main()
