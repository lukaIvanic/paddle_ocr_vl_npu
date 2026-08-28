#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("assemble_recovered_output.py")
SPEC = importlib.util.spec_from_file_location("experiment17_recovery", MODULE_PATH)
assert SPEC and SPEC.loader
RECOVERY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECOVERY)


class AssembleRecoveredOutputTest(unittest.TestCase):
    def test_assembles_exact_prefix_and_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            primary = root / "primary"
            recovery = root / "recovery"
            names = ["a.png", "b.png", "c.png", "d.png"]
            primary.mkdir()
            recovery.mkdir()
            (primary / "predictions").mkdir()
            (primary / "content_lists").mkdir()
            (recovery / "predictions").mkdir()
            (recovery / "content_lists").mkdir()
            manifest = {
                "count": 4,
                "pages": [
                    {"dataset_index": index, "image": name}
                    for index, name in enumerate(names)
                ],
            }
            (primary / "input_manifest.json").write_text(json.dumps(manifest))
            (primary / "failure.json").write_text(
                json.dumps({"experiment": "17", "selected_pages": 4})
            )
            for name in names[:2]:
                stem = Path(name).stem
                markdown = "" if name == "b.png" else stem
                (primary / "predictions" / f"{stem}.md").write_text(markdown)
                (primary / "content_lists" / f"{stem}.json").write_text("[]")
            suffix_manifest = {"count": 2, "pages": manifest["pages"][2:]}
            (recovery / "input_manifest.json").write_text(
                json.dumps(suffix_manifest)
            )
            (recovery / "run_summary.json").write_text(
                json.dumps(
                    {
                        "offset": 2,
                        "selected_pages": 2,
                        "completed": 2,
                        "failed": 0,
                        "git_commit": "abc123",
                    }
                )
            )
            for name in names[2:]:
                stem = Path(name).stem
                (recovery / "predictions" / f"{stem}.md").write_text(stem)
                (recovery / "content_lists" / f"{stem}.json").write_text("[]")

            combined = root / "combined"
            summary = RECOVERY.assemble_recovered_output(
                primary_output=primary,
                recovery_output=recovery,
                recovery_offset=2,
                combined_output=combined,
            )
            self.assertEqual(summary["completed"], 4)
            self.assertTrue(summary["accuracy_only"])
            self.assertFalse(summary["throughput_comparable"])
            self.assertEqual((combined / "predictions" / "b.md").stat().st_size, 0)
            for name in names:
                stem = Path(name).stem
                self.assertTrue((combined / "predictions" / f"{stem}.md").is_symlink())
                self.assertTrue((combined / "content_lists" / f"{stem}.json").is_symlink())


if __name__ == "__main__":
    unittest.main()
