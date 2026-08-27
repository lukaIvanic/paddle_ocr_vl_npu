#!/usr/bin/env python3
"""Regression test for the 310P low-memory report-only retry."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
RUNNER = SCRIPT_DIR / "run_310p_lowmem_full1651_hbm_background.sh"


class ReportOnlyTest(unittest.TestCase):
    def test_accepts_both_trace_token_keys_and_persists_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output = root / "output"
            output.mkdir()

            memory = {
                "exit_code": 0,
                "wall_s": 2.0,
                "sample_count": 1,
                "peak": {
                    "total_pss_bytes": 100,
                    "total_rss_bytes": 200,
                    "elapsed_s": 1.0,
                    "processes": [{"pid": 1, "pss_bytes": 100}],
                },
                "npu_hbm": {
                    "physical_npu": 0,
                    "baseline": {"used_mb": 10},
                    "peak": {
                        "used_mb": 20,
                        "total_mb": 21527,
                        "elapsed_s": 1.0,
                        "raw_npu_smi": "| 0 test row",
                    },
                    "peak_increase_from_baseline_mb": 10,
                    "sample_count": 1,
                    "errors": [],
                },
            }
            run_summary = {
                "status": "pass",
                "commit": "test",
                "chip": "Ascend310P",
                "page_count": 1651,
                "crop_count": 2,
                "process_wall_s": 1.0,
                "pages_per_s": 1651.0,
                "frontend_wall_s": 0.1,
                "vision_phase_wall_s": 0.2,
                "decode_wall_s": 0.3,
                "settings": {
                    "workers": 4,
                    "recognition_threads": 8,
                    "layout_batch_size": 2,
                    "vision_bucket_preset": "310p_k20_l4",
                    "decode_batch_size": 128,
                    "cross_cache_length": 1320,
                    "self_cache_length": 2048,
                },
                "layout": {"owner_wall_s": 0.1},
                "vision": {"wall_s": 0.2},
                "text_prefill": {"wall_s": 0.1},
                "decode": {
                    "decode_iterations": 3,
                    "raw_decode_token_slots": 384,
                    "effective_decode_tokens": 4,
                    "decode_s": 0.3,
                    "raw_decode_tokens_per_s": 1280.0,
                    "effective_decode_tokens_per_s": 13.333333,
                },
            }
            canonical_rows = [
                {"request_id": "b", "text": "second", "token_ids": [3, 4]},
                {"request_id": "a", "text": "first", "token_ids": [1, 2]},
            ]
            candidate_rows = [
                {"request_id": "a", "text": "first", "generated_ids": [1, 2]},
                {"request_id": "b", "text": "second", "generated_ids": [3, 4]},
            ]

            (root / "process_tree_and_hbm.json").write_text(
                json.dumps(memory) + "\n", encoding="utf-8"
            )
            (output / "run_summary.json").write_text(
                json.dumps(run_summary) + "\n", encoding="utf-8"
            )
            (output / "recognition_trace.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in candidate_rows),
                encoding="utf-8",
            )
            canonical_trace = root / "canonical_trace.jsonl"
            canonical_trace.write_text(
                "".join(json.dumps(row) + "\n" for row in canonical_rows),
                encoding="utf-8",
            )
            (root / "exit_code.txt").write_text("1\n", encoding="utf-8")

            completed = subprocess.run(
                ["bash", str(RUNNER), "--report-only", str(root)],
                env={
                    "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
                    "PYTHON_BIN": sys.executable,
                    "CANONICAL_TRACE": str(canonical_trace),
                },
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("UNIREC_310P_LOWMEM_FULL1651_HBM: PASS", completed.stdout)
            self.assertIn("UNIREC_310P_LOWMEM_REPORT_ONLY: PASS", completed.stdout)
            self.assertEqual((root / "exit_code.txt").read_text(), "1\n")
            self.assertEqual((root / "report_only_exit_code.txt").read_text(), "0\n")
            self.assertIn(
                "UNIREC_310P_LOWMEM_FULL1651_HBM: PASS",
                (root / "final_report.txt").read_text(),
            )
            report = json.loads((root / "final_report.json").read_text())
            self.assertEqual(report["crops"], 2)
            self.assertTrue(report["trace_request_ids_match"])
            self.assertEqual(report["trace_mismatch_count"], 0)
            self.assertEqual(
                report["trace_reference_sha256"], report["trace_candidate_sha256"]
            )


if __name__ == "__main__":
    unittest.main()
