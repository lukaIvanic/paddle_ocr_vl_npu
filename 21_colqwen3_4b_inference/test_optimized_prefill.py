"""CPU algebra/call-contract tests only; PromptFA is mocked, not NPU validated."""
from dataclasses import replace
import unittest
from unittest.mock import patch
import torch
from torch import nn

from config import ColQwenConfig,VisionConfig,TextConfig
from local_modeling_colqwen3 import LocalColQwen3
from prepared_prefill import prepare_inputs,prepare_text,finish_embeddings,bmm_attention
from optimized_prefill import (Linear,Options,OptimizedVisionStage,OptimizedTextStage,
    prompt_attention,text_args_for_promptfa)
from bench_optimized_prefill import score_smoke


def fake_promptfa(q,k,v,**kw):
    assert 'actual_seq_lengths' not in kw and 'actual_seq_lengths_kv' not in kw
    assert kw['input_layout']=='BNSD' and kw['sparse_mode']==0
    assert kw['pre_tokens']==kw['next_tokens']==2147483647
    mask=kw.get('atten_mask')
    if mask is None:
        mask=q.new_zeros(1,1,q.shape[2],k.shape[2])
    else:
        assert mask.dtype==torch.bool
        mask=torch.where(mask,torch.finfo(q.dtype).min,0.).to(q.dtype)
    return bmm_attention(q,k,v,kw['scale_value'],mask).transpose(1,2).contiguous()


class OptimizedContracts(unittest.TestCase):
    def test_fused_linear_does_not_modify_reference(self):
        torch.manual_seed(21)
        layers=[nn.Linear(8,n) for n in (16,4,4)]
        weights=[m.weight.clone() for m in layers]
        x=torch.randn(2,3,8)
        combined=Linear(layers,'native')
        torch.testing.assert_close(combined(x),torch.cat([m(x) for m in layers],-1))
        for m,w in zip(layers,weights):
            self.assertTrue(torch.equal(m.weight,w))
        with self.assertRaises(ValueError):
            Linear(layers,'fractal_nz')

    def test_native_repeat_gqa_and_causal_mask(self):
        q=torch.randn(1,4,5,64).half();k=torch.randn(1,2,5,64).half();v=torch.randn_like(k)
        mask=torch.ones(1,1,5,5,dtype=torch.bool).triu(1)
        with patch('optimized_prefill._promptfa',side_effect=fake_promptfa) as op:
            native=prompt_attention(q,k,v,64**-.5,mask,'native')
            self.assertEqual(op.call_args.kwargs['num_key_value_heads'],2)
            repeated=prompt_attention(q,k,v,64**-.5,mask,'repeat')
            self.assertNotIn('num_key_value_heads',op.call_args.kwargs)
        self.assertTrue(torch.equal(native,repeated))
        # The first causal output must be V at position zero.
        torch.testing.assert_close(native[:,0,0],v[:,0,0])
        args=[q,q,q,torch.full((1,1,5,5),-100.),q,q,q]
        with self.assertRaises(ValueError):
            text_args_for_promptfa(args)

    def test_full_candidate_algebra(self):
        vc=replace(VisionConfig(),depth=3,hidden_size=128,intermediate_size=256,num_heads=2,
            out_hidden_size=128,num_position_embeddings=16,deepstack_visual_indexes=(0,1,2),patch_size=2)
        tc=replace(TextConfig(),hidden_size=128,intermediate_size=256,num_hidden_layers=3,
            num_attention_heads=2,num_key_value_heads=1,head_dim=64,vocab_size=32)
        model=LocalColQwen3(ColQwenConfig(vc,tc,dims=128,image_token_id=28,video_token_id=29,
            vision_start_token_id=27,mrope_section=(8,8,8))).half().eval()
        with torch.inference_mode(),patch('optimized_prefill._promptfa',side_effect=fake_promptfa):
            for image in (False,True):
                inputs={'input_ids':torch.tensor([[1,27,28,28,28,28,2] if image else [1,2,3]]),
                        'attention_mask':torch.ones(1,7 if image else 3,dtype=torch.long)}
                if image:
                    inputs.update(pixel_values=torch.randn(1,16,24).half(),image_grid_thw=torch.tensor([[1,4,4]]))
                expected=model(**inputs)
                prepared=prepare_inputs(model,inputs)
                for fused in (False,True):
                    options=Options(fused_projections=fused)
                    vision=OptimizedVisionStage(model,options)
                    text=OptimizedTextStage(model,options)
                    vo=vision(*prepared.vision_args[:3]) if image else None
                    ta=text_args_for_promptfa(prepare_text(model,prepared,vo))
                    actual=finish_embeddings(model,prepared,text(*ta))
                    torch.testing.assert_close(actual,expected,atol=.003,rtol=.003)

    def test_score_gate_records_rankings(self):
        q=torch.tensor([[[1.,0.]]]);d=torch.tensor([[[.5,.5]]])
        rows=[{'name':'q','image':False,'reference':q,'candidate':q},
              {'name':'d','image':True,'reference':d,'candidate':d}]
        self.assertTrue(score_smoke(rows)['passed'])
        rows[1]['candidate']=d+.1
        self.assertFalse(score_smoke(rows)['passed'])


if __name__=='__main__':
    unittest.main()
