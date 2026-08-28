#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("verify_310p_artifacts.py")
SPEC = importlib.util.spec_from_file_location("experiment17_310p_artifacts", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
VERIFY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VERIFY
SPEC.loader.exec_module(VERIFY)


class Verify310PArtifactsTest(unittest.TestCase):
    def test_reference_bundle_matches_pinned_identities(self) -> None:
        VERIFY.verify_reference_bundle()

    def test_path_independent_model_manifest_digest(self) -> None:
        self.assertEqual(
            VERIFY.digest_rows(VERIFY.expected_model_rows()),
            VERIFY.EXPECTED_MODEL_MANIFEST_SHA256,
        )

    def test_path_independent_image_manifest_digest(self) -> None:
        self.assertEqual(
            VERIFY.digest_rows(VERIFY.expected_image_rows()),
            VERIFY.EXPECTED_IMAGE_MANIFEST_SHA256,
        )


if __name__ == "__main__":
    unittest.main()
