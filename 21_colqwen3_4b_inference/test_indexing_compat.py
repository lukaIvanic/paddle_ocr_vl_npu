"""CPU semantics and ATen dispatch gates; 310P kernels still need target validation."""
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from local_modeling_colqwen3 import (add_image_features, dense_image_features,
                                   interleave_mrope, causal_attention_bias, image_positions)
from prepared_prefill import prepare_inputs, prepare_text, PreparedVisionStage, PreparedTextStage
from test_prepared_prefill import tiny_model
from bench_optimized_prefill import validity


class NoAdvancedIndex(TorchDispatchMode):
    """Catch the actual ATen operations, not just spelling in Python source."""
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = func._schema.name
        if name in {'aten::index', 'aten::index_put', 'aten::index_put_',
                    'aten::_index_put_impl_', 'aten::nonzero', 'aten::masked_select',
                    'aten::select_scatter', 'aten::slice_scatter'}:
            raise AssertionError(f'Unsafe indexing dispatched: {func}')
        return func(*args, **(kwargs or {}))


class IndexingCompatibility(unittest.TestCase):
    def test_masked_rows_preserve_values_order_and_unselected_bits(self):
        torch.manual_seed(310)
        for dtype in (torch.float16, torch.float32):
            for mask in (torch.zeros(2,5,dtype=torch.bool), torch.ones(2,5,dtype=torch.bool),
                         torch.tensor([[0,1,0,1,0],[1,0,0,1,1]],dtype=torch.bool)):
                hidden = torch.randn(2,5,8,dtype=dtype)
                hidden[0,0,0] = -0.0
                # Noncontiguous source, multiple rows, empty/full/interleaved masks.
                source = torch.randn(8,int(mask.sum()),dtype=dtype).t()
                dense = torch.zeros_like(hidden)
                dense[mask] = source
                expected = hidden.clone()
                expected[mask] = expected[mask].clone()+source
                with NoAdvancedIndex():
                    actual_dense = dense_image_features(hidden,mask,source)
                    actual = add_image_features(hidden,mask,source)
                self.assertTrue(torch.equal(actual_dense,dense))
                self.assertTrue(torch.equal(actual,expected))
                self.assertTrue(torch.equal(torch.signbit(actual),torch.signbit(expected)))

    def test_rotary_matches_legacy_strided_writes(self):
        for length, sections in ((64,(24,20,20)),(8,(4,2,2)),(17,(3,0,9))):
            freqs = torch.randn(3,2,7,length)
            expected = freqs[0].clone()
            for axis in (1,2):
                expected[...,axis:sections[axis]*3:3] = freqs[axis,...,axis:sections[axis]*3:3]
            with NoAdvancedIndex():
                actual = interleave_mrope(freqs,sections)
            self.assertTrue(torch.equal(actual,expected))

    def test_embedding_lookup_matches_multidimensional_table_index(self):
        table = torch.randn(84,32)
        coordinates = torch.tensor([[0,83],[60,7],[3,3]])
        expected = table[coordinates]
        with NoAdvancedIndex():
            actual = F.embedding(coordinates,table)
        self.assertTrue(torch.equal(actual,expected))

    def test_causal_bias_preserves_left_and_right_padding(self):
        valid = torch.tensor([[0,1,1,1],[1,1,0,0]])
        seq = torch.arange(4)
        allowed = (seq[:,None]>=seq[None,:])[None,None] & valid[:,None,None,:].bool()
        for dtype in (torch.float16,torch.float32):
            expected = torch.where(allowed,0.,torch.finfo(dtype).min).to(dtype)
            with NoAdvancedIndex():
                actual = causal_attention_bias(valid,dtype)
            self.assertTrue(torch.equal(actual,expected))

    def test_full_owned_and_prepared_paths_dispatch_no_advanced_index(self):
        model = tiny_model()
        for image in (False,True):
            ids = torch.tensor([[1,27,28,28,28,28,2] if image else [1,2,3]])
            inputs = dict(input_ids=ids,attention_mask=torch.ones_like(ids))
            grids = [[1,4,4]] if image else []
            if image:
                inputs.update(pixel_values=torch.randn(1,16,24),image_grid_thw=torch.tensor(grids))
            # Production computes this metadata on CPU intentionally. Test the
            # accelerator path separately, without banning legitimate CPU indexing.
            positions = image_positions(ids,inputs['attention_mask'],grids,model.config)
            with torch.inference_mode(), NoAdvancedIndex(), \
                 patch('local_modeling_colqwen3.image_positions',return_value=positions), \
                 patch('prepared_prefill.image_positions',return_value=positions):
                reference = model(**inputs)
                prepared = prepare_inputs(model,inputs)
                visual = PreparedVisionStage(model)(*prepared.vision_args) if image else None
                args = prepare_text(model,prepared,visual)
                result = PreparedTextStage(model)(*args)
                self.assertTrue(torch.isfinite(reference).all())
                self.assertTrue(torch.isfinite(result).all())

    def test_validity_diagnostics_do_not_index_on_device(self):
        reference = torch.tensor([[[1.,0.],[0.,0.]]])
        with NoAdvancedIndex():
            self.assertTrue(validity(reference,reference)['passed'])
            corrupt = torch.tensor([[[1.,0.],[1.,0.]]])
            self.assertFalse(validity(corrupt,reference)['passed'])
            self.assertFalse(validity(torch.zeros_like(reference),torch.zeros_like(reference))['passed'])


if __name__ == '__main__':
    unittest.main()
