"""CPU bookkeeping checks with fake events; never CPU model inference."""
from contextlib import nullcontext
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'25_clef_inference'))
from profiling import Journal, PipelineProfiler, observe_model, summarize


class Event:
    created = []
    def __init__(self, **kwargs):
        self.ready = False
        self.records = 0
        self.created.append(self)
    def record(self):
        self.records += 1
    def query(self):
        return self.ready
    def elapsed_time(self, other):
        assert other.ready
        return 2.5
    def synchronize(self):
        raise AssertionError('Timing inserted a synchronization')


def forbidden(*args, **kwargs):
    raise AssertionError('Timing inserted a synchronization')


class ObserverTests(unittest.TestCase):
    def setUp(self):
        Event.created = []
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        torch = SimpleNamespace(npu=SimpleNamespace(Event=Event,synchronize=forbidden),
                                profiler=SimpleNamespace(record_function=lambda name:nullcontext()))
        self.patch = patch.dict(sys.modules, {'torch':torch})
        self.patch.start()
        self.stdout = patch('sys.stdout',new=io.StringIO())
        self.stdout.start()
    def tearDown(self):
        self.stdout.stop()
        self.patch.stop()
        self.temp.cleanup()
    def test_pending_events_not_reused_and_later_resolved(self):
        j=Journal(self.root)
        try:
            for index in range(2):
                with j.item({'phase':'cached','index':index,'first_use':index==0}):
                    with j.section('work',device=True): pass
            self.assertEqual(len(Event.created),4)
            self.assertEqual(len(j.pending),2)
            for e in Event.created: e.ready=True
            j.resolve()
            self.assertEqual(len(j.pending),0)
            with j.item({'phase':'cached','index':2,'first_use':False}):
                with j.section('work',device=True): pass
            self.assertEqual(len(Event.created),4)
            self.assertEqual(j.rows[0]['sections'][0]['device_interval_ms'],2.5)
        finally: j.close()
        self.assertEqual(len((self.root/'items.jsonl').read_text().splitlines()),3)
    def test_nested_sections_are_not_double_counted(self):
        j=Journal(self.root)
        try:
            with j.item({'phase':'uncached','index':0,'first_use':True}):
                with j.section('forward'):
                    with j.section('backbone'): pass
            row=j.rows[0]
            outer=next(s for s in row['sections'] if s['name']=='forward')
            self.assertAlmostEqual(row['unattributed_host_s'],row['wall_s']-outer['host_s'])
            self.assertEqual(summarize(j.rows)['phases']['uncached']['wall_s']['count'],1)
        finally: j.close()
    def test_wrappers_preserve_return_identity_and_restore_on_error(self):
        class Layer:
            def forward(self,x): return x
        class Model:
            backbone=Layer()
            head=Layer()
            def named_modules(self): return []
        def scan(x): return x
        modeling=SimpleNamespace(GatedDeltaNet=Layer,FullAttention=Layer,MLP=Layer,
                                 EvidenceRoutingLayer=Layer,chunk_gated_delta_rule=scan)
        model=Model(); value=object(); j=Journal(self.root,profile=True)
        try:
            with patch.dict(sys.modules,{'local_modeling_clef':modeling}):
                with self.assertRaises(ValueError):
                    with observe_model(model,j):
                        self.assertIs(model.backbone.forward(value),value)
                        self.assertIs(modeling.chunk_gated_delta_rule(value),value)
                        raise ValueError('intentional')
            self.assertNotIn('forward',vars(model.backbone))
            self.assertNotIn('forward',vars(model.head))
            self.assertIs(modeling.chunk_gated_delta_rule,scan)
        finally: j.close()
    def test_profiler_captures_real_item_in_each_phase(self):
        captured={}
        class Capture:
            def start(self): pass
            def stop(self): pass
        def profile(**kwargs): captured.update(kwargs); return Capture()
        profiler=SimpleNamespace(ProfilerAction=SimpleNamespace(WARMUP='warm',RECORD_AND_SAVE='record',NONE='none'),
          ProfilerActivity=SimpleNamespace(CPU='cpu',NPU='npu'),profile=profile,
          tensorboard_trace_handler=lambda *a,**kw:None)
        with patch.dict(sys.modules,{'torch_npu':SimpleNamespace(profiler=profiler),'torch_npu.profiler':profiler}):
            p=PipelineProfiler(self.root,3,3,1,True)
            self.assertEqual([captured['schedule'](i) for i in range(9)],['warm','record','none']*3)
            p.close()


if __name__=='__main__': unittest.main()
