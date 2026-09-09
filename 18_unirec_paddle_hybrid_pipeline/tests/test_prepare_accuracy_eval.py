import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_accuracy_eval import prepare


class PrepareAccuracyEvalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "output"
        self.output.mkdir()
        (self.output / "run_summary.json").write_text(json.dumps(dict(pages=1, engines={})))
        self.original = 'Text <img src="x.png"> $x$\n<table><tr><td>1</td></tr></table>'
        (self.output / "page.md").write_text(self.original)
        self.dataset = self.root / "dataset.json"
        self.dataset.write_text(json.dumps([dict(page_info=dict(image_path="page.png"))]))
        self.evaluation = self.root / "evaluation"

    def test_copy_only_image_tags_and_preserve_original(self):
        result = prepare(self.output, self.dataset, self.evaluation, 1)
        self.assertEqual(result["removed_image_tags"], 1)
        self.assertEqual((self.output / "page.md").read_text(), self.original)
        self.assertEqual((self.evaluation / "predictions/page.md").read_text(),
                         self.original.replace('<img src="x.png">', ''))
        self.assertIn("metric: [TEDS, Edit_dist]", (self.evaluation / "work/config.yaml").read_text())
        with self.assertRaises(FileExistsError):
            prepare(self.output, self.dataset, self.evaluation, 1)

    def test_missing_extra_and_duplicate_predictions_fail(self):
        (self.output / "page.md").rename(self.output / "wrong.md")
        with self.assertRaises(ValueError):
            prepare(self.output, self.dataset, self.evaluation, 1)
        (self.output / "wrong.md").rename(self.output / "page.md")
        (self.output / "duplicate").mkdir()
        (self.output / "duplicate/page.md").write_text("duplicate")
        with self.assertRaises(ValueError):
            prepare(self.output, self.dataset, self.evaluation, 1)
        self.assertFalse(self.evaluation.exists())

    def test_incomplete_and_nested_eval_fail(self):
        with self.assertRaises(ValueError):
            prepare(self.output, self.dataset, self.evaluation, 2)
        with self.assertRaises(ValueError):
            prepare(self.output, self.dataset, self.output / "evaluation", 1)


if __name__ == "__main__":
    unittest.main()
