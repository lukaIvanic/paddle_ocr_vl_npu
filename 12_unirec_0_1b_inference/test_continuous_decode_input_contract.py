#!/usr/bin/env python3
"""CPU check for the production UniRec decode input guard contract."""

from __future__ import annotations

import sys
import types
import unittest

import torch

# Keep this contract test CPU-only and independent of the Transformers model
# package that is intentionally absent from the local authoring environment.
modeling_stub = types.ModuleType("modeling_optimized_unirec")
modeling_stub.LOCAL_UNIREC_STATIC_CACHE_LEN = 2048
modeling_stub.LocalUniRecStaticCache = type("LocalUniRecStaticCache", (), {})
modeling_stub.OptimizedUniRecRunner = type("OptimizedUniRecRunner", (), {})
modeling_stub.UniRecPrefilledItem = type("UniRecPrefilledItem", (), {})
sys.modules.setdefault("modeling_optimized_unirec", modeling_stub)

from continuous_unirec import ContinuousUniRecDecoder


class ContinuousDecodeInputContractTest(unittest.TestCase):
    def test_decode_device_inputs_are_static_inference_tensors(self) -> None:
        next_token, cache_position = (
            ContinuousUniRecDecoder._allocate_decode_device_inputs(7, "cpu")
        )
        self.assertEqual(next_token.shape, (7, 1))
        self.assertEqual(cache_position.shape, (7,))
        self.assertEqual(next_token.dtype, torch.long)
        self.assertEqual(cache_position.dtype, torch.int64)
        self.assertTrue(next_token.is_contiguous())
        self.assertTrue(cache_position.is_contiguous())
        self.assertTrue(next_token.is_inference())
        self.assertTrue(cache_position.is_inference())


if __name__ == "__main__":
    unittest.main()
