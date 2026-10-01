"""CPU patch-layout algebra; NPU measurements are separate."""
import unittest
from types import SimpleNamespace
import torch
from torch import nn
from patch_embedding import LinearPatchEmbed,prepare_linear_patch_inputs
from prepared_prefill import prepare_inputs
from test_prepared_prefill import tiny_model


class PatchEmbeddingTests(unittest.TestCase):
    def test_bias_and_flatten_order(self):
        torch.manual_seed(3)
        for bias in (True,False):
            conv=nn.Conv3d(3,7,(2,4,4),stride=(2,4,4),bias=bias)
            if bias:
                with torch.no_grad():conv.bias.fill_(.75)
            original=conv.weight.detach().clone()
            linear=LinearPatchEmbed(SimpleNamespace(proj=conv))
            patches=torch.randn(5,96)
            expected=conv(patches.reshape(5,3,2,4,4)).reshape(5,7)
            torch.testing.assert_close(linear(patches),expected,atol=1e-6,rtol=1e-6)
            self.assertTrue(torch.equal(conv.weight,original))
            with self.assertRaises(ValueError):linear(patches[:,:-1])

    def test_reject_overlapping_kernel(self):
        with self.assertRaises(ValueError):
            LinearPatchEmbed(SimpleNamespace(proj=nn.Conv3d(3,7,2,stride=1)))

    def test_preparation_does_not_mutate_reference(self):
        model=tiny_model()
        original=model.visual.patch_embed
        inputs={'input_ids':torch.tensor([[1,27,28,28,28,28,2]]),
                'attention_mask':torch.ones(1,7,dtype=torch.long),
                'pixel_values':torch.randn(1,16,24),'image_grid_thw':torch.tensor([[1,4,4]])}
        with torch.inference_mode():
            reference=prepare_inputs(model,inputs)
            actual=prepare_linear_patch_inputs(model,inputs,LinearPatchEmbed(original))
        self.assertIs(model.visual.patch_embed,original)
        for a,b in zip(actual.vision_args,reference.vision_args):
            torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)
        self.assertTrue(torch.equal(actual.positions,reference.positions))


if __name__=='__main__':unittest.main()
