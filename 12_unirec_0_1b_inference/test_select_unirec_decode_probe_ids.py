#!/usr/bin/env python3

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).with_name("select_unirec_decode_probe_ids.py")
SPEC = importlib.util.spec_from_file_location("select_unirec_decode_probe_ids", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class GeneratedTokenCountTests(unittest.TestCase):
    def test_decode_replay_schema(self) -> None:
        self.assertEqual(MODULE.generated_token_count({"generated_token_count": 17}), 17)

    def test_production_page_trace_schema(self) -> None:
        self.assertEqual(MODULE.generated_token_count({"token_count": 23}), 23)

    def test_token_ids_fallback_excludes_initial_token(self) -> None:
        self.assertEqual(MODULE.generated_token_count({"token_ids": [0, 9, 2]}), 2)

    def test_missing_length_fails_explicitly(self) -> None:
        with self.assertRaisesRegex(ValueError, "no generated_token_count"):
            MODULE.generated_token_count({"request_id": "page_0_crop_0"})


if __name__ == "__main__":
    unittest.main()
