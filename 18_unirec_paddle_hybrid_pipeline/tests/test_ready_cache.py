"""Exercise Paddle's real admission method on CPU or a selected NPU."""
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'09_persistent_page_engine'))
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'torch is not installed in the local authoring interpreter')
class ReadyCacheTests(unittest.TestCase):
    def setUp(self):
        from paddleocr_vl.model.text_decode import LocalPaddleOCRVLStaticCache
        from paddleocr_vl.serving.continuous_decode import DecodeArena, ReadyDecodeRequest
        self.Ready = ReadyDecodeRequest
        self.Cache = LocalPaddleOCRVLStaticCache
        self.device = torch.device(os.environ.get('READY_CACHE_TEST_DEVICE', 'cpu'))
        if self.device.type == 'npu':
            import torch_npu
            torch.npu.set_device(self.device)
        self.config = SimpleNamespace(num_hidden_layers=2, num_key_value_heads=2, head_dim=128)
        self.arena = DecodeArena(cache=self.cache(4096, 2), device=self.device,
                                 batch_size=2, eos_token_id=2)

    def cache(self, length, batch=1):
        return self.Cache.allocate(self.config, batch_size=batch, cache_length=length,
                                   device=self.device, dtype=torch.float16)

    def ready(self, length, prompt):
        cache=self.cache(length)
        for t in cache.logical_tensors():
            t.fill_(3)
        releases=[]
        request=self.Ready(request_id='test', payload=None, cache=cache,
            rope_deltas=torch.zeros((1,1),dtype=torch.long,device=self.device),
            cache_position=torch.tensor([prompt],device=self.device),
            first_token_tensor=torch.tensor([[7]],device=self.device), first_token=7,
            prompt_length=prompt,cache_release=lambda:releases.append(True))
        return request,releases

    def test_short_row_reuse_preserves_suffix_and_other_slots(self):
        for length in (1536,1536):
            for t in self.arena.cache.logical_tensors():
                t.fill_(9)
            ready,releases=self.ready(length,1036)
            self.arena.admit(0,ready,hot_swap=True)
            for t in self.arena.cache.logical_tensors():
                self.assertTrue(bool((t[0,:,:length,:]==3).all().cpu()))
                self.assertTrue(bool((t[0,:,length:,:]==9).all().cpu()))
                self.assertTrue(bool((t[1]==9).all().cpu()))
            self.assertEqual(releases,[True])
            self.assertIsNone(ready.cache)
            self.assertEqual(int(self.arena.cache_position[0].cpu()),1036)
            self.arena.release(0)

    def test_existing_full_row_admission(self):
        ready,releases=self.ready(4096,1036)
        self.arena.admit(0,ready,hot_swap=False)
        self.assertEqual(releases,[True])
        for t in self.arena.cache.logical_tensors():
            self.assertTrue(bool((t[0]==3).all().cpu()))

    def test_reject_prompt_larger_than_source(self):
        ready,releases=self.ready(1536,1537)
        with self.assertRaisesRegex(ValueError,'ready cache must hold'):
            self.arena.admit(0,ready,hot_swap=False)
        self.assertEqual(releases,[])

    def test_reject_source_larger_than_arena(self):
        ready,releases=self.ready(4097,1036)
        with self.assertRaisesRegex(ValueError,'ready cache must hold'):
            self.arena.admit(0,ready,hot_swap=False)
        self.assertEqual(releases,[])
