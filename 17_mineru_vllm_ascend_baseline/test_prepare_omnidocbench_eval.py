#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("prepare_omnidocbench_eval.py")
SPEC = importlib.util.spec_from_file_location("experiment17_eval_prep", MODULE_PATH)
assert SPEC and SPEC.loader
PREP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREP)


class PrepareOmniDocBenchEvalTest(unittest.TestCase):
    def make_completed_run(self, root: Path) -> tuple[Path, Path]:
        dataset_path = root / "OmniDocBench.json"
        dataset = [
            {"page_info": {"image_path": "a.jpg"}, "layout_dets": []},
            {"page_info": {"image_path": "b.jpg"}, "layout_dets": []},
        ]
        dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
        output = root / "output"
        predictions = output / "predictions"
        predictions.mkdir(parents=True)
        (predictions / "a.md").write_text("alpha", encoding="utf-8")
        (predictions / "b.md").write_text("beta", encoding="utf-8")
        (output / "run_summary.json").write_text(
            json.dumps(
                {
                    "completed": 2,
                    "failed": 0,
                    "selected_pages": 2,
                    "git_commit": "abc123",
                }
            ),
            encoding="utf-8",
        )
        (output / "input_manifest.json").write_text(
            json.dumps(
                {
                    "count": 2,
                    "pages": [{"image": "a.jpg"}, {"image": "b.jpg"}],
                }
            ),
            encoding="utf-8",
        )
        return dataset_path, output

    def test_prepares_exact_prediction_membership_and_cdm_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_path, output = self.make_completed_run(root)
            evaluation = root / "evaluation"
            result = PREP.prepare_evaluation(
                dataset_json=dataset_path,
                run_output=output,
                evaluation_root=evaluation,
                expected_pages=2,
                match_workers=4,
                teds_workers=3,
                cdm_workers=2,
                evaluator_root=None,
            )
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["page_count"], 2)
            self.assertTrue((evaluation / "predictions" / "a.md").is_symlink())
            config = (evaluation / "work" / "config.yaml").read_text()
            self.assertIn("metric: [Edit_dist, CDM]", config)
            self.assertIn("cdm_workers: 2", config)

    def test_rejects_missing_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_path, output = self.make_completed_run(root)
            (output / "predictions" / "b.md").unlink()
            with self.assertRaisesRegex(RuntimeError, "expected 2 unique"):
                PREP.prepare_evaluation(
                    dataset_json=dataset_path,
                    run_output=output,
                    evaluation_root=root / "evaluation",
                    expected_pages=2,
                    match_workers=4,
                    teds_workers=3,
                    cdm_workers=2,
                    evaluator_root=None,
                )


if __name__ == "__main__":
    unittest.main()
