from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

from local_reranker_w8a8 import W8A8Linear  # noqa: E402


class W8A8ContractTest(unittest.TestCase):
    def test_inference_formatted_weight_transposes_inside_forward(self) -> None:
        captured: dict[str, torch.Tensor] = {}

        def fake_quant_matmul(x1, x2, **kwargs):
            captured["x1"] = x1
            captured["x2"] = x2
            return torch.zeros(x1.shape[0], x2.shape[1], dtype=torch.float16)

        fake_torch_npu = SimpleNamespace(
            npu_trans_quant_param=lambda scale, _offset: scale,
            npu_quant_matmul=fake_quant_matmul,
        )
        linear = W8A8Linear(4, 6, out_dtype=torch.float16)
        linear.weight_is_matmul_ready = True
        linear.weight_requires_graph_transpose = True
        x_q = torch.zeros(3, 4, dtype=torch.int8)

        with patch.dict(sys.modules, {"torch_npu": fake_torch_npu}):
            output = linear.quant_matmul_from_quantized(x_q, torch.tensor(0.1))

        self.assertEqual(linear.weight_q.shape, (6, 4))
        self.assertEqual(captured["x2"].shape, (4, 6))
        self.assertEqual(output.shape, (3, 6))


if __name__ == "__main__":
    unittest.main()
