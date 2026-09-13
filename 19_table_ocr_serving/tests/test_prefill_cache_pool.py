"""Prefill pool relocation and reuse events, using CPU storage and fake NPU events."""
from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

import torch
from test_text_simplification import ROOT, EXPERIMENT
import serving_runtime as runtime


class PrefillCachePoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = subprocess.check_output(['git','-C',str(ROOT),'show',
            '564da03f:19_table_ocr_serving/_support/serving/prefill_cache_pool.py'],text=True)
        cls.old = types.ModuleType('historical_prefill_cache_pool')
        sys.modules[cls.old.__name__] = cls.old
        exec(compile(cls.source,'historical_prefill_cache_pool','exec'),cls.old.__dict__)

    def test_definitions_unchanged(self):
        def definitions(source):
            return {n.name:ast.dump(n) for n in ast.parse(source).body
                    if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
        old = definitions(self.source)
        moved = definitions(Path(runtime.__file__).read_text())
        self.assertEqual({name:moved[name] for name in old},old)
        self.assertFalse((EXPERIMENT/'_support/serving/prefill_cache_pool.py').exists())

    def test_pool_ownership_reuse_and_events_unchanged(self):
        def exercise(module):
            events = []
            class Stream:
                def record_event(self):
                    event = len(events)
                    events.append(('record',event))
                    return event
                def wait_event(self,event):
                    events.append(('wait',event))
            stream = Stream()
            fake = types.SimpleNamespace(npu=types.SimpleNamespace(current_stream=lambda:stream))
            cache = runtime.LocalPaddleOCRVLStaticCache(
                (torch.zeros(2,1,8,2),),(torch.zeros(2,1,8,2),),8)
            with patch.dict(sys.modules,{'torch_npu':fake}):
                pool = module.PrefillKVCachePool(cache,device=torch.device('cpu'))
                first,second = pool.acquire(),pool.acquire()
                self.assertEqual(events,[])
                self.assertEqual((first.slot_index,second.slot_index),(0,1))
                first.cache.key_caches[0].fill_(5)
                self.assertTrue(torch.equal(cache.key_caches[0][0],torch.full((1,8,2),5.)))
                with self.assertRaises(RuntimeError): pool.acquire()
                first.release()
                self.assertTrue(first.released)
                again = pool.acquire()
                self.assertEqual(events,[('record',0),('wait',0)])
                self.assertEqual(again.generation,2)
                self.assertEqual(again.cache.key_caches[0].data_ptr(),first.cache.key_caches[0].data_ptr())
                with self.assertRaises(RuntimeError): first.release()
                with self.assertRaises(RuntimeError): pool._release(first)
                second.release(); again.release()
                self.assertEqual(pool.stats()['active_slots'],0)
                self.assertEqual(pool.stats()['free_slots'],2)
                self.assertEqual(pool.stats()['reuses'],1)
                return pool.stats(),events
        self.assertEqual(exercise(self.old),exercise(runtime))


if __name__=='__main__':
    unittest.main()
