"""CPU structural/unit checks only; real model parity requires the NPU runner."""
import ast
from dataclasses import asdict
from pathlib import Path
import unittest

import torch

from config import ColQwenConfig
from local_modeling_colqwen3 import image_positions, attention, PatchMerger, LocalColQwen3


class LocalContracts(unittest.TestCase):
    def test_config_rejects_architecture_changes(self):
        raw = asdict(ColQwenConfig())
        raw['model_type'] = 'ops_colqwen3'
        raw['text_config']['rope_scaling'] = {'rope_type': 'default', 'mrope_interleaved': True,
                                             'mrope_section': [24, 20, 20]}
        self.assertEqual(ColQwenConfig.from_dict(raw), ColQwenConfig())
        raw['text_config']['head_dim'] = 80
        with self.assertRaises(ValueError):
            ColQwenConfig.from_dict(raw)

    def test_no_transformers_model_imports(self):
        for name in ('config.py', 'local_modeling_colqwen3.py'):
            tree = ast.parse(Path(__file__).with_name(name).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    self.assertFalse(any(a.name.startswith('transformers') for a in node.names))
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse((node.module or '').startswith('transformers'))

    def test_strict_model_key_contract(self):
        with torch.device('meta'):
            model = LocalColQwen3(ColQwenConfig())
        self.assertEqual(len(model.state_dict()), 715)
        self.assertEqual(model.language_model.layers[0].self_attn.q_proj.weight.shape, (4096, 2560))
        self.assertEqual(model.visual.blocks[0].attn.qkv.weight.shape, (3072, 1024))

    def test_text_positions_with_padding(self):
        mask = torch.tensor([[0, 1, 1], [1, 1, 0]])
        pos = image_positions(torch.zeros_like(mask), mask, [], ColQwenConfig())
        self.assertTrue(torch.equal(pos[0], torch.tensor([[1, 0, 1], [0, 1, 1]])))

    def test_image_positions_and_tail(self):
        c = ColQwenConfig()
        ids = torch.tensor([[0, 9, c.vision_start_token_id, *([c.image_token_id]*4), 10]])
        mask = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 1]])
        pos = image_positions(ids, mask, [[1, 4, 4]], c)
        expected = torch.tensor([[1, 0, 1, 2, 2, 2, 2, 4],
                                 [1, 0, 1, 2, 2, 3, 3, 4],
                                 [1, 0, 1, 2, 3, 2, 3, 4]])
        self.assertTrue(torch.equal(pos[:, 0], expected))

    def test_merger_norm_shapes(self):
        c = ColQwenConfig().vision_config
        with torch.device('meta'):
            self.assertEqual(PatchMerger(c).norm.weight.shape, (1024,))
            self.assertEqual(PatchMerger(c, True).norm.weight.shape, (4096,))

    def test_grouped_attention_matches_repeated_kv(self):
        torch.manual_seed(21)
        q, k, v = torch.randn(2, 4, 3, 8), torch.randn(2, 2, 3, 8), torch.randn(2, 2, 3, 8)
        expected = torch.softmax(q @ k.repeat_interleave(2, 1).transpose(2, 3) / 8**0.5, -1)
        expected = (expected @ v.repeat_interleave(2, 1)).transpose(1, 2)
        torch.testing.assert_close(attention(q, k, v, 8**-0.5), expected)


if __name__ == '__main__':
    unittest.main()
