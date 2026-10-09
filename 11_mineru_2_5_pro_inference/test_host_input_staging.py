"""CPU semantic tests; actual NPU pinning and encoder parity are Level 2 checks."""
import unittest
from unittest.mock import patch, Mock
from contextlib import nullcontext
from types import SimpleNamespace
import sys
import torch
from transformers.feature_extraction_utils import BatchFeature
from host_input_staging import pin_processor_outputs,move_pinned_inputs,stage_pinned_inputs


class StagingTests(unittest.TestCase):
    def test_values_dtypes_and_retained_sources_match_original(self):
        source={'pixel_values':torch.arange(1176,dtype=torch.float32).reshape(1,-1)/255,
                'input_ids':torch.tensor([[1,2,3]]),'attention_mask':torch.ones(1,3,dtype=torch.int64),
                'image_grid_thw':torch.tensor([[1,2,2]])}
        positions=torch.arange(9).reshape(3,1,3);deltas=torch.tensor([[0]])
        ref=BatchFeature({k:v.clone() for k,v in source.items()}).to(device='cpu',dtype=torch.float16)
        inputs=BatchFeature({k:v.clone() for k,v in source.items()})
        pinned=set()
        def pin(t):
            out=t.clone();pinned.add(id(out));return out
        with patch.object(torch.Tensor,'pin_memory',pin),patch.object(torch.Tensor,'is_pinned',lambda t:id(t) in pinned):
            pos,delta=pin_processor_outputs(inputs,positions,deltas)
            moved,new_pos,new_delta,keep=move_pinned_inputs(inputs,pos,delta,device='cpu',dtype=torch.float16)
            for k in ref:
                self.assertEqual(moved[k].dtype,ref[k].dtype)
                self.assertTrue(torch.equal(moved[k].view(torch.uint8),ref[k].view(torch.uint8)))
            self.assertTrue(torch.equal(new_pos,positions));self.assertTrue(torch.equal(new_delta,deltas))
            self.assertEqual(keep[0].dtype,torch.float32)
            self.assertTrue(torch.equal(keep[0],source['pixel_values']))
            self.assertEqual(len(keep),len(source)+2)
            self.assertIs(keep[-2],pos);self.assertIs(keep[-1],delta)
    def test_staging_preserves_cpu_mapping_grid_and_sources(self):
        source=BatchFeature({'pixel_values':torch.arange(24,dtype=torch.float32).reshape(2,12),
                             'input_ids':torch.tensor([[1,2]]),'image_grid_thw':torch.tensor([[1,2,2]])})
        pos=torch.arange(6).reshape(3,1,2);delta=torch.tensor([[0]])
        originals=dict(source);event=object();stream=Mock();stream.record_event.return_value=event
        fake=SimpleNamespace(npu=SimpleNamespace(stream=lambda s:nullcontext()))
        with patch.dict(sys.modules,{'torch_npu':fake}),patch.object(torch.Tensor,'is_pinned',return_value=True):
            staged=stage_pinned_inputs(source,pos,delta,stream=stream,device='cpu',dtype=torch.float16,keep_grid_on_cpu=True)
        self.assertEqual(set(source),set(originals))
        for key,value in originals.items():self.assertIs(source[key],value)
        self.assertIs(staged.cpu_grid,originals['image_grid_thw'])
        self.assertNotIn('image_grid_thw',staged.inputs)
        self.assertIs(staged.ready_event,event)
        self.assertTrue(torch.equal(staged.inputs.pixel_values,originals['pixel_values'].half()))
        self.assertIs(staged.sources[0],originals['pixel_values'])
        self.assertEqual(staged.sources[0].dtype,torch.float32)
        stream.record_event.assert_called_once_with()

    def test_no_silent_unpinned_fallback(self):
        with self.assertRaisesRegex(RuntimeError,'unpinned'):
            move_pinned_inputs(BatchFeature({'pixel_values':torch.zeros(1,2)}),torch.zeros(1),torch.zeros(1),device='cpu',dtype=torch.float16)
    def test_pin_failure_propagates(self):
        with patch.object(torch.Tensor,'pin_memory',side_effect=RuntimeError('pin unavailable')):
            with self.assertRaisesRegex(RuntimeError,'pin unavailable'):
                pin_processor_outputs(BatchFeature({'pixel_values':torch.zeros(1,2)}),torch.zeros(1),torch.zeros(1))


if __name__=='__main__':unittest.main()
