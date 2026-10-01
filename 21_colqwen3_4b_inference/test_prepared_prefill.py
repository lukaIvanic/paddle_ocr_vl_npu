"""Tiny CPU tests for graph boundaries; not NPU validation."""
import ast
from dataclasses import replace
from pathlib import Path
import unittest
import torch

from config import ColQwenConfig, VisionConfig, TextConfig
from local_modeling_colqwen3 import LocalColQwen3
from prepared_prefill import (PreparedVisionStage, PreparedTextStage, prepare_inputs,
                              prepare_text, finish_embeddings, unique_forward)


def tiny_model():
    vision = replace(VisionConfig(), depth=3, hidden_size=32, intermediate_size=64,
                     num_heads=2, out_hidden_size=32, num_position_embeddings=16,
                     deepstack_visual_indexes=(0,1,2), patch_size=2)
    text = replace(TextConfig(), hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                   num_attention_heads=2, num_key_value_heads=1, head_dim=16, vocab_size=32)
    config = ColQwenConfig(vision, text, dims=32, image_token_id=28, video_token_id=29,
                          vision_start_token_id=27, mrope_section=(4,2,2))
    return LocalColQwen3(config).eval()


class PreparedContracts(unittest.TestCase):
    def test_prepared_image_and_query(self):
        torch.manual_seed(21)
        model = tiny_model()
        for image in (False, True):
            inputs = {'input_ids': torch.tensor([[1,27,28,28,28,28,2] if image else [1,2,3]]),
                      'attention_mask': torch.ones(1,7 if image else 3,dtype=torch.long)}
            if image:
                inputs.update(pixel_values=torch.randn(1,16,24), image_grid_thw=torch.tensor([[1,4,4]]))
            with torch.inference_mode():
                expected = model(**inputs)
                prepared = prepare_inputs(model, inputs)
                outputs = PreparedVisionStage(model)(*prepared.vision_args) if image else None
                text_args = prepare_text(model, prepared, outputs)
                actual = finish_embeddings(model, prepared, PreparedTextStage(model)(*text_args))
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
            for deep in text_args[-3:]:
                self.assertTrue((deep[inputs['input_ids'] != 28] == 0).all())

    def test_no_tensor_to_host_in_stage_forwards(self):
        tree = ast.parse(Path(__file__).with_name('prepared_prefill.py').read_text())
        for cls in (n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in ('PreparedVisionStage','PreparedTextStage')):
            for call in (n for n in ast.walk(cls) if isinstance(n, ast.Call)):
                if isinstance(call.func, ast.Attribute):
                    self.assertNotIn(call.func.attr, ('cpu','item','tolist','numpy'))

    def test_reject_padding_and_batched_input(self):
        model = tiny_model()
        for ids,mask in [(torch.ones(2,3,dtype=torch.long),torch.ones(2,3,dtype=torch.long)),
                         (torch.ones(1,3,dtype=torch.long),torch.tensor([[0,1,1]]))]:
            with self.assertRaises(ValueError):
                prepare_inputs(model, {'input_ids':ids,'attention_mask':mask})

    def test_distinct_entry_code_objects(self):
        model = tiny_model()
        stage = PreparedTextStage(model)
        a, b = unique_forward(stage, 'a'), unique_forward(stage, 'b')
        self.assertIsNot(a.__func__.__code__, b.__func__.__code__)

    def test_diagnostic_outputs_preserve_production_vision(self):
        from diagnose_compiled_vision import DiagnosticVisionStage
        model = tiny_model()
        args = (torch.randn(16,32), torch.randn(16,16), torch.randn(16,16),
                torch.zeros(1,1,16,16))
        with torch.inference_mode():
            expected = PreparedVisionStage(model)(*args)
            actual = DiagnosticVisionStage(model)(*args)
        self.assertEqual(len(actual), len(DiagnosticVisionStage.labels))
        for a,b in zip(actual[:4], expected):
            self.assertTrue(torch.equal(a,b))


if __name__ == '__main__':
    unittest.main()
