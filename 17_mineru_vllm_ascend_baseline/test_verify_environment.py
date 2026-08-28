#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("verify_environment.py")
SPEC = importlib.util.spec_from_file_location("experiment17_environment", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
VERIFY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VERIFY
SPEC.loader.exec_module(VERIFY)


class VerifyEnvironmentTest(unittest.TestCase):
    def test_310p_policy_records_vllm_version_drift(self) -> None:
        actual = dict(VERIFY.EXPECTED)
        actual["vllm"] = "310p-work-version"
        actual["vllm-ascend"] = "310p-work-plugin-version"
        self.assertEqual(
            VERIFY.version_mismatches(
                actual,
                allow_vllm_version_drift=True,
            ),
            {},
        )

    def test_310p_policy_keeps_other_packages_strict(self) -> None:
        actual = dict(VERIFY.EXPECTED)
        actual["torch-npu"] = "wrong"
        self.assertEqual(
            VERIFY.version_mismatches(
                actual,
                allow_vllm_version_drift=True,
            ),
            {
                "torch-npu": {
                    "expected": VERIFY.EXPECTED["torch-npu"],
                    "actual": "wrong",
                }
            },
        )


if __name__ == "__main__":
    unittest.main()
