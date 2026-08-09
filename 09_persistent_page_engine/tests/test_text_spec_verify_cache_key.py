"""Tests for bounded TorchAir verifier cache path components."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from paddleocr_vl.model.text_spec_verify import bounded_spec_cache_component


class SpecCacheKeyTest(unittest.TestCase):
    def test_short_key_is_preserved(self) -> None:
        self.assertEqual(bounded_spec_cache_component("short_key"), "short_key")

    def test_long_key_is_short_deterministic_and_sensitive(self) -> None:
        first = bounded_spec_cache_component("a" * 300)
        repeated = bounded_spec_cache_component("a" * 300)
        second = bounded_spec_cache_component("a" * 299 + "b")
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, second)
        self.assertLessEqual(len(first.encode("utf-8")), 240)


if __name__ == "__main__":
    unittest.main()
