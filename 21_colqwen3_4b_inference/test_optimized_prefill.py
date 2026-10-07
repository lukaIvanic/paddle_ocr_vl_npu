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
    prompt_attention,text_args_for_promptfa,format_code,prepare_310p_text_inputs)
from bench_optimized_prefill import score_smoke
from text_forward_variants import prepare_measured_text


def fake_promptfa(q,k,v,**kw):
    assert 'actual_seq_lengths' not in kw and 'actual_seq_lengths_kv' not in kw
    assert kw['input_layout'] in ('BNSD','BSND') and kw['sparse_mode']==0
    bsnd=kw['input_layout']=='BSND'
    if bsnd:
        q,k,v=(a.transpose(1,2) for a in (q,k,v))
    assert kw['pre_tokens']==kw['next_tokens']==2147483647
    mask=kw.get('atten_mask')
    if mask is None:
        mask=q.new_zeros(1,1,q.shape[2],k.shape[2])
    else:
        assert mask.dtype==torch.bool
        mask=torch.where(mask,torch.finfo(q.dtype).min,0.).to(q.dtype)
    result=bmm_attention(q,k,v,kw['scale_value'],mask)
    return result if bsnd else result.transpose(1,2).contiguous()


class OptimizedContracts(unittest.TestCase):
    def test_batched_pages_are_independent_and_match_single_forward(self):
        from bench_page_batches import BatchVision, BatchForward
        from patch_embedding import LinearPatchEmbed
        from profile_warm_forward import Forward
        torch.manual_seed(214)
        vc=replace(VisionConfig(),depth=3,hidden_size=128,intermediate_size=256,num_heads=2,
            out_hidden_size=128,num_position_embeddings=16,deepstack_visual_indexes=(0,1,2),patch_size=2)
        tc=replace(TextConfig(),hidden_size=128,intermediate_size=256,num_hidden_layers=3,
            num_attention_heads=2,num_key_value_heads=1,head_dim=64,vocab_size=32)
        model=LocalColQwen3(ColQwenConfig(vc,tc,dims=128,image_token_id=28,video_token_id=29,
            vision_start_token_id=27,mrope_section=(8,8,8))).half().eval()
        rows=[dict(input_ids=torch.tensor([[1,27,28,28,28,28,2]]),
                   attention_mask=torch.ones(1,7,dtype=torch.long),
                   pixel_values=torch.randn(1,16,24).half(),image_grid_thw=torch.tensor([[1,4,4]]))
              for _ in range(4)]
        patcher=LinearPatchEmbed(model.visual.patch_embed)
        vision=OptimizedVisionStage(model,Options()); text=OptimizedTextStage(model,Options())
        with torch.inference_mode(),patch('optimized_prefill._promptfa',side_effect=fake_promptfa):
            expected=torch.cat([Forward(model,row,patcher,vision,text)() for row in rows])
            for size in (1,2,4):
                actual=BatchForward(model,rows[:size],patcher,BatchVision(vision),text)()
                torch.testing.assert_close(actual,expected[:size],atol=.003,rtol=.003)
            fn=BatchForward(model,rows,patcher,BatchVision(vision),text)
            before=fn()
            rows[3]['pixel_values'].mul_(5)
            after=fn()
            self.assertTrue(torch.equal(before[:3],after[:3]))
            self.assertFalse(torch.equal(before[3],after[3]))

    def test_cross_source_snapshot_keeps_checkpoint_and_input_guards(self):
        from profile_warm_text import validate_snapshot_identity
        saved=dict(model='model',config_sha256='config',weights={'a':[1,2]},
                   anchor_sha256='anchor',options={'gqa':'native'},source={'a':'old'})
        current={**saved,'options':{'gqa':'repeat'},'source':{'a':'new'}}
        with self.assertRaises(RuntimeError):
            validate_snapshot_identity(saved,current)
        self.assertEqual(validate_snapshot_identity(saved,current,reference_only=True),
                         ['options','source'])
        for key in ('model','config_sha256','weights','anchor_sha256'):
            with self.assertRaises(RuntimeError):
                validate_snapshot_identity(saved,{**current,key:None},reference_only=True)

    def test_portable_mask_alignment_and_real_outputs(self):
        torch.manual_seed(310)
        for length in (1,127,128,129):
            q=torch.randn(2,4,length,64).half()
            k=torch.randn(2,2,length,64).half();v=torch.randn_like(k)
            mask=torch.ones(2,1,length,length,dtype=torch.bool).triu(1)
            # Extra blocked keys exercise preservation of a supplied mask.
            if length>1:
                mask[:,:,:,1]=True
            hidden=torch.randn(2,length,256).half()
            cos=torch.randn(2,length,64).half();sin=torch.randn_like(cos)
            deep=tuple(torch.randn_like(hidden) for _ in range(3))
            prepared=prepare_310p_text_inputs(hidden,cos,sin,mask,*deep)
            physical=((length+127)//128)*128
            for original,padded in zip((hidden,cos,sin,*deep),(prepared[0],prepared[1],prepared[2],*prepared[4:])):
                self.assertEqual(padded.shape[1],physical)
                self.assertTrue(torch.equal(padded[:,:length],original))
            self.assertTrue(bool((prepared[1][:,length:]==1).all()))
            for padded in (prepared[0],prepared[2],*prepared[4:]):
                self.assertTrue(bool((padded[:,length:]==0).all()))
            with patch('optimized_prefill._promptfa',side_effect=fake_promptfa) as op:
                reference=prompt_attention(q,k,v,64**-.5,mask,'native')
                for layout in ('BNSD','BSND'):
                    padded_qkv=tuple(torch.nn.functional.pad(a,(0,0,0,physical-length)) for a in (q,k,v))
                    inputs=padded_qkv if layout=='BNSD' else tuple(a.transpose(1,2) for a in padded_qkv)
                    actual=prompt_attention(*inputs,64**-.5,prepared[3],'repeat',layout=layout)[:,:length]
                    call=op.call_args
                    self.assertNotIn('num_key_value_heads',call.kwargs)
                    self.assertEqual(call.args[0].shape,call.args[1].shape)
                    padded=call.kwargs['atten_mask']
                    self.assertEqual(padded.shape,(2,1,physical,physical))
                    self.assertTrue(torch.equal(padded[:,:,:length,:length],mask))
                    self.assertTrue(bool(padded[:,:,:length,length:].all()))
                    self.assertTrue(bool((~padded).any(-1).all()))
                    torch.testing.assert_close(actual,reference,atol=.002,rtol=.002)
                    self.assertEqual(actual.shape,(2,length,4,64))

    def test_strict_format_codes(self):
        for value in (29,'29','FRACTAL_NZ'):
            self.assertEqual(format_code(value),29)
        self.assertEqual(format_code('ND'),2)
        with self.assertRaises(ValueError):
            format_code('UNKNOWN')

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
                    hidden=text(*ta)
                    measured,aligned=prepare_measured_text(text,ta,'baseline')
                    with patch('optimized_prefill.prepare_310p_text_inputs',
                               side_effect=AssertionError('Padding entered measured forward')):
                        physical_hidden=measured(*aligned)
                    self.assertEqual(physical_hidden.shape[1]%128,0)
                    self.assertTrue(torch.equal(physical_hidden[:,:hidden.shape[1]],hidden))
                    actual=finish_embeddings(model,prepared,hidden)
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
