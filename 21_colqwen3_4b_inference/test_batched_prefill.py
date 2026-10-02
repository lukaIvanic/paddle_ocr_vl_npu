"""CPU algebra tests with mocked PromptFA; NPU parity is checked separately."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from batched_prefill import BatchedVisionStage, prepare_batched_images, pad_page_batch
from config import ColQwenConfig, VisionConfig, TextConfig
from local_modeling_colqwen3 import LocalColQwen3
from optimized_prefill import Options, OptimizedVisionStage, OptimizedTextStage, text_args_for_promptfa
from patch_embedding import LinearPatchEmbed, prepare_linear_patch_inputs
from prepared_prefill import prepare_text, finish_embeddings
from test_optimized_prefill import fake_promptfa


class BatchContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        vc=replace(VisionConfig(),depth=3,hidden_size=128,intermediate_size=256,num_heads=2,
            out_hidden_size=128,num_position_embeddings=16,deepstack_visual_indexes=(0,1,2),patch_size=2)
        tc=replace(TextConfig(),hidden_size=128,intermediate_size=256,num_hidden_layers=3,
            num_attention_heads=2,num_key_value_heads=1,head_dim=64,vocab_size=32)
        self.model=LocalColQwen3(ColQwenConfig(vc,tc,dims=128,image_token_id=28,video_token_id=29,
            vision_start_token_id=27,mrope_section=(8,8,8))).half().eval()
        self.inputs=dict(input_ids=torch.tensor([[1,27]+[28]*8+[2]]*3),
            attention_mask=torch.ones(3,11,dtype=torch.long),
            pixel_values=torch.randn(3,32,24).half(),
            image_grid_thw=torch.tensor([[1,4,8],[1,8,4],[1,4,8]]))
        self.patch=LinearPatchEmbed(self.model.visual.patch_embed)

    def test_batched_images_match_independent_pages_and_do_not_mix(self):
        with torch.inference_mode(),patch('optimized_prefill._promptfa',side_effect=fake_promptfa):
            vision=BatchedVisionStage(self.model,Options())
            single_vision=OptimizedVisionStage(self.model,Options())
            text=OptimizedTextStage(self.model,Options())
            def batched(inputs):
                p=prepare_batched_images(self.model,inputs,self.patch)
                vo=vision(*p.vision_args)
                return finish_embeddings(self.model,p,text(*text_args_for_promptfa(prepare_text(self.model,p,vo))))
            actual=batched(self.inputs)
            padded=batched(pad_page_batch(self.inputs,4))
            torch.testing.assert_close(padded[:3],actual,atol=.003,rtol=.003)
            self.assertTrue(torch.equal(padded[2],padded[3]))
            for i in range(3):
                one={k:v[i:i+1] for k,v in self.inputs.items()}
                p=prepare_linear_patch_inputs(self.model,one,self.patch)
                vo=single_vision(*p.vision_args[:3])
                expected=finish_embeddings(self.model,p,text(*text_args_for_promptfa(prepare_text(self.model,p,vo))))
                torch.testing.assert_close(actual[i:i+1],expected,atol=.003,rtol=.003)
            altered={k:v.clone() for k,v in self.inputs.items()}
            altered['pixel_values'][1].mul_(-3)
            changed=batched(altered)
            self.assertTrue(torch.equal(actual[0],changed[0]))
            self.assertTrue(torch.equal(actual[2],changed[2]))
            self.assertFalse(torch.equal(actual[1],changed[1]))

    def test_slot_padding_preserves_valid_inputs_and_valid_attention(self):
        padded=pad_page_batch(self.inputs,4)
        for key,value in self.inputs.items():
            self.assertEqual(padded[key].shape[0],4)
            self.assertTrue(torch.equal(padded[key][:3],value))
            self.assertTrue(torch.equal(padded[key][3],value[-1]))
        self.assertTrue(bool((padded['attention_mask']==1).all()))
        self.assertIs(pad_page_batch(self.inputs,3),self.inputs)
        with self.assertRaises(ValueError):
            pad_page_batch(self.inputs,2)

    def test_padding_and_unequal_grids_are_rejected(self):
        inputs={k:v.clone() for k,v in self.inputs.items()}
        inputs['attention_mask'][0,0]=0
        with self.assertRaisesRegex(ValueError,'unpadded'):
            prepare_batched_images(self.model,inputs,self.patch)
        inputs={k:v.clone() for k,v in self.inputs.items()}
        inputs['image_grid_thw'][0]=torch.tensor([1,4,4])
        with self.assertRaisesRegex(ValueError,'patch lengths'):
            prepare_batched_images(self.model,inputs,self.patch)
        inputs={k:v.clone() for k,v in self.inputs.items()}
        inputs['input_ids'][0,2]=2
        with self.assertRaisesRegex(ValueError,'placeholder'):
            prepare_batched_images(self.model,inputs,self.patch)

if __name__=='__main__':
    unittest.main()
