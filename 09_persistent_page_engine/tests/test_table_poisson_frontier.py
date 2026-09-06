"""CPU tests for selection and ownership guards; not an NPU performance test."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / "scripts/table_poisson_frontier.py"
SPEC = importlib.util.spec_from_file_location("frontier", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FrontierTest(unittest.TestCase):
    def test_frontier_keeps_tradeoffs_and_excludes_failed_run(self):
        a = dict(valid=True, target_qps=3, completed_qps=2.9, p95_s=2)
        b = dict(valid=True, target_qps=3, completed_qps=2.8, p95_s=3)
        c = dict(valid=True, target_qps=4, completed_qps=3.9, p95_s=4)
        bad = dict(valid=False, target_qps=10, completed_qps=10, p95_s=.1)
        self.assertEqual(MODULE.frontier([a,b,c,bad]), [a,c])
        self.assertEqual(MODULE.frontier([a,b,c,bad], offered=True), [a,c])

    def test_ownership_does_not_adopt_other_container_work(self):
        sweep = MODULE.Sweep.__new__(MODULE.Sweep)
        sweep.marker = "tmp/my-run/b2/service.json"
        processes = """PID PPID COMMAND
100 1 python serve_crop_ocr_api.py --service-summary-output tmp/my-run/b2/service.json
101 100 python -c spawn_main()
102 101 helper
200 1 python serve_crop_ocr_api.py --service-summary-output tmp/another/service.json
201 200 python -c spawn_main()
"""
        with patch.object(MODULE, "output", return_value=processes):
            self.assertEqual(sweep.owned(), ({100,101,102},100))

    def test_device_guard_rejects_unknown_pid_and_unknown_output(self):
        import io
        from types import SimpleNamespace
        sweep = MODULE.Sweep.__new__(MODULE.Sweep)
        sweep.args = SimpleNamespace(npu=6)
        sweep.ownership = io.StringIO()
        sweep.server_pid = 100
        with patch.object(sweep, "owned", return_value=({100,101},100)):
            with patch.object(MODULE, "output", return_value="Process id:201"):
                with self.assertRaisesRegex(RuntimeError, "contamination"):
                    sweep.check_device()
            with patch.object(MODULE, "output", return_value="driver unavailable"):
                with self.assertRaisesRegex(RuntimeError, "Cannot establish"):
                    sweep.check_device()
            with patch.object(MODULE, "output", return_value="Process id:101"):
                sweep.check_device()

    def test_setup_subshells_are_one_owned_process_tree(self):
        sweep = MODULE.Sweep.__new__(MODULE.Sweep)
        sweep.marker = "tmp/my-run/b1/service.json"
        processes = """PID PPID COMMAND
100 1 bash -c source npu-setup; exec python serve_crop_ocr_api.py tmp/my-run/b1/service.json
101 100 bash -c source npu-setup; exec python serve_crop_ocr_api.py tmp/my-run/b1/service.json
102 101 npu-status
200 1 unrelated-job
"""
        with patch.object(MODULE, "output", return_value=processes):
            self.assertEqual(sweep.owned(), ({100,101,102},100))
        with patch.object(MODULE, "output", return_value=processes +
                          "300 1 python serve_crop_ocr_api.py tmp/my-run/b1/service.json\n"):
            with self.assertRaisesRegex(RuntimeError, "Ambiguous"):
                sweep.owned()

    def test_stop_refuses_recycled_pid(self):
        sweep = MODULE.Sweep.__new__(MODULE.Sweep)
        sweep.server = object()
        sweep.server_pid = 100
        sweep.server_start = "first"
        with patch.object(sweep, "owned", return_value=({100},100)), \
             patch.object(sweep, "start_identity", return_value="second"), \
             patch.object(MODULE.os, "kill") as kill:
            with self.assertRaisesRegex(RuntimeError, "identity changed"):
                sweep.stop()
            kill.assert_not_called()


if __name__ == "__main__":
    unittest.main()
