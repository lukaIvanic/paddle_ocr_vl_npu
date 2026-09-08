"""CPU-only checks: commands/defaults/schedule, never model execution."""
import json
from pathlib import Path
import shlex
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify


def historical_module(commit, path, name):
    source = subprocess.check_output(["git", "-C", str(verify.ROOT), "show", f"{commit}:{path}"], text=True)
    module = types.ModuleType(name)
    module.__file__ = str(verify.ROOT / path)
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


class LockTest(unittest.TestCase):
    def setUp(self):
        self.lock = verify.load()

    def test_evidence_and_historical_sources(self):
        self.assertEqual(verify.check(self.lock), [])

    def test_explicit_arguments_equal_historical_defaults_for_all_chart_batches(self):
        module = historical_module(self.lock["optimized"]["commit"],
                   "09_persistent_page_engine/scripts/serve_crop_ocr_api.py", "locked_server")
        for batch in self.lock["optimized"]["chart_batch_by_qps"].values():
            old = (verify.ROOT / self.lock["optimized"]["artifact_root"] / f"b{batch}/server_command.txt").read_text()
            script = shlex.split(old)[-1]
            original = shlex.split(script.split("; exec ",1)[1])
            with patch.object(sys,"argv",original[2:]):
                expected = vars(module.parse_args())
            output = expected["service_summary_output"]
            explicit = [x.format(batch=batch,output="tmp/check") for x in self.lock["optimized"]["expanded_server_argv"]]
            explicit[-1] = str(output)
            with patch.object(sys,"argv",explicit[2:]):
                actual = vars(module.parse_args())
            for values in (expected,actual):
                for key,value in values.items():
                    if isinstance(value,Path):
                        # Host-local paths versus the same container-repo paths.
                        values[key] = str(value).replace(str(verify.ROOT),"/workspace/repos/paddle_ocr_vl_npu")
                        if key == "decode_vocab_token_ids" and not value.is_absolute():
                            values[key] = "/workspace/repos/paddle_ocr_vl_npu/"+str(value)
            self.assertEqual(expected,actual)

    def test_original_seed_regenerates_exact_1000_schedule(self):
        module = historical_module(self.lock["optimized"]["commit"],
                 "09_persistent_page_engine/scripts/table_request_load_simulator.py", "locked_client")
        records = module.read_jsonl(verify.ROOT/self.lock["source_jsonl"])
        cohort = module.freeze_tail_cohort(records,"all")
        generated = module.schedule_rows(module.make_schedule(cohort,1,10,1,max_requests=1000,shuffle_all=True))
        saved = module.read_jsonl(verify.ROOT/self.lock["schedule"])
        self.assertEqual(generated,saved)

    def test_command_counts_and_vllm_lock(self):
        for lane,count in (("optimized",3),("vllm",4)):
            cmds = verify.commands(self.lock,lane,6,4,"tmp/reproduction_unit_test_unique")
            self.assertEqual(len(cmds),count)
            self.assertIn("--max-requests",cmds[-1][1])
            self.assertNotIn("--max-in-flight",cmds[-1][1])
        vllm = verify.commands(self.lock,"vllm",6,4,"tmp/reproduction_unit_test_unique")
        self.assertIn("TABLE_VLLM_MAX_SEQS=64",vllm[0][1])
        self.assertIn("TABLE_VLLM_TOKEN_BUDGET=16384",vllm[0][1])
        self.assertIn("--schedule-source-qps",vllm[-1][1])

    def test_no_unanchored_or_overwriting_command(self):
        with self.assertRaises(ValueError):
            verify.commands(self.lock,"optimized",7,4,"tmp/new-repro")
        with self.assertRaises(ValueError):
            verify.commands(self.lock,"optimized",1,4,"/tmp/new-repro")
        with self.assertRaises(ValueError):
            verify.commands(self.lock,"optimized",1,4,self.lock["optimized"]["artifact_root"])


if __name__ == "__main__":
    unittest.main()
