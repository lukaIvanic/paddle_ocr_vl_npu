import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from prepare_serving_eval import prepare


class PrepareServingEvalTest(unittest.TestCase):
    def fixture(self, root):
        dataset = root / "dataset.json"
        dataset.write_text(json.dumps([{"page_info": {"image_path": n}, "layout_dets": []}
                                       for n in ("a.png", "b.jpg")]))
        output = root / "output"
        for directory in ("predictions", "content_lists", "progress", "failures"):
            (output / directory).mkdir(parents=True)
        rows = []
        for index, (name, text) in enumerate((("a.png", "alpha <img src='x'>"), ("b.jpg", ""))):
            stem = Path(name).stem
            row = {"status": "completed", "image": name, "dataset_index": index,
                   "block_count": 0, "markdown_chars": len(text)}
            rows.append(row)
            (output / "predictions" / f"{stem}.md").write_text(text)
            (output / "content_lists" / f"{stem}.json").write_text("[]")
            (output / "progress" / f"{stem}.json").write_text(json.dumps(row))
        # Streaming completion order need not equal dataset order.
        (output / "progress_shard_00.jsonl").write_text("\n".join(json.dumps(r) for r in reversed(rows)))
        summary = {"completed": 2, "failed": 0, "skipped": 0, "shard_count": 1,
                   "shard_index": 0, "selected_pages": 2, "shard_pages": 2, "offset": 0,
                   "model_hashes": {"dataset_json": hashlib.sha256(dataset.read_bytes()).hexdigest()}}
        (output / "run_summary_shard_00.json").write_text(json.dumps(summary))
        return output, dataset

    def test_exact_unmodified_membership_including_empty_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output, dataset = self.fixture(root)
            result = prepare(output, dataset, root / "eval", 2, None)
            self.assertEqual(result["prediction_transform"], "none")
            self.assertEqual(result["page_count"], 2)
            for stem in ("a", "b"):
                target = root / "eval/predictions" / f"{stem}.md"
                self.assertTrue(target.is_symlink())
                self.assertEqual(target.read_bytes(), (output / "predictions" / f"{stem}.md").read_bytes())

    def test_rejects_incomplete_or_modified_outputs(self):
        for mutation in ("missing", "extra", "changed", "duplicate", "hash", "failed", "wrong_index"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                output, dataset = self.fixture(root)
                if mutation == "missing":
                    (output / "content_lists/b.json").unlink()
                elif mutation == "extra":
                    (output / "predictions/c.md").write_text("extra")
                elif mutation == "changed":
                    (output / "predictions/a.md").write_text("modified")
                elif mutation == "duplicate":
                    path = output / "progress_shard_00.jsonl"
                    path.write_text(path.read_text() + "\n" + path.read_text().splitlines()[0])
                elif mutation == "hash":
                    dataset.write_text(dataset.read_text() + "\n")
                elif mutation == "failed":
                    (output / "failures/page.json").write_text("{}")
                else:
                    path = output / "progress_shard_00.jsonl"
                    path.write_text(path.read_text().replace('"dataset_index": 0', '"dataset_index": 10'))
                with self.assertRaises(ValueError):
                    prepare(output, dataset, root / "eval", 2, None)
                self.assertFalse((root / "eval").exists())


if __name__ == "__main__":
    unittest.main()
