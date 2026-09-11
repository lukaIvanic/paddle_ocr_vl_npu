"""Control/CPU tests; these are not accelerator inference validation."""
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parent
sys.path[:0] = [str(EXP), str(ROOT / '18_unirec_paddle_hybrid_pipeline'),
               str(ROOT / '11_mineru_2_5_pro_inference')]
from hybrid_routing import Routing
from mineru_adapter import ReadySource
from coordinator import Coordinator
try:
    import torch
except ImportError:
    torch = None


class RoutingTests(unittest.TestCase):
    def test_defaults_and_single_engine(self):
        self.assertEqual(Routing().models, ('unirec', 'mineru'))
        self.assertEqual(Routing().model_for('Table Recognition:'), 'mineru')
        self.assertEqual(Routing('mineru', 'mineru', 'mineru').models, ('mineru',))
        with self.assertRaises(KeyError):
            Routing().model_for('unknown')

    def test_runner_does_not_modify_experiment18_defaults(self):
        spec = importlib.util.spec_from_file_location('experiment20_runner', EXP / 'run_pipeline.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        args = module.build_parser().parse_args(['--input', 'images', '--output-dir', 'new'])
        self.assertEqual(args.table_model, 'mineru')
        self.assertEqual(args.mineru_max_pixels, 602112)
        self.assertFalse(hasattr(args, 'paddle_model_path'))
        old = module.shared.build_parser().parse_args(['--input', 'images', '--output-dir', 'old'])
        self.assertEqual(old.table_model, 'paddle')

    def test_ready_source_idle_is_not_closed(self):
        source = ReadySource(lambda *_: None)
        self.assertIsNone(source.pull())
        self.assertFalse(source.closed)
        source.upstream = False
        self.assertTrue(source.upstream_exhausted)
        self.assertFalse(source.closed)
        source.input_closed = True
        source.items.append((1, 'ready'))
        self.assertFalse(source.closed)
        self.assertEqual(source.pull(), (1, 'ready'))
        self.assertTrue(source.closed)
        with self.assertRaises(RuntimeError):
            source.pull(block=True)

    def test_full_priority_and_alternation_unchanged(self):
        engines = {name: SimpleNamespace(done=False, occupied=32, capacity=32)
                   for name in ('unirec', 'mineru')}
        pages = SimpleNamespace(set_wakeup=lambda f: None)
        c = Coordinator(engines, pages)
        self.assertEqual(c.action(), ('unirec', 'decode'))
        c.last_model = 'unirec'
        self.assertEqual(c.action(), ('mineru', 'decode'))


@unittest.skipIf(torch is None, 'CPU torch required')
class CooperativeTests(unittest.TestCase):
    def test_timing_modules_coexist(self):
        import subprocess
        subprocess.run([sys.executable, '-c', '''
import sys, torch
sys.path[:0] = ['12_unirec_0_1b_inference', '11_mineru_2_5_pro_inference']
import prefill_timing
from mineru_prefill_timing import PrefillDeviceTimeline
from local_modeling_mineru import PrefillDeviceTimeline as model_timeline
assert PrefillDeviceTimeline is model_timeline
assert PrefillDeviceTimeline is not prefill_timing.PrefillDeviceTimeline
assert PrefillDeviceTimeline(torch.device('cpu'), []).resolve() == {}
'''], cwd=ROOT, check=True)

    def fixture(self):
        from test_streaming_decode import FakeEngine, FakeSource
        from streaming_decode import iter_decode_stream
        return FakeEngine, FakeSource, iter_decode_stream

    def test_pause_resume_matches_blocking_original(self):
        FakeEngine, FakeSource, iterator = self.fixture()
        from streaming_decode import run_decode_stream
        requests = [(i, 10 * (i + 1) + 1, 10) for i in range(12)]
        old, new = FakeSource(requests), FakeSource(requests)
        module = SimpleNamespace(maybe_sync_device=lambda d: None)
        with patch.dict(sys.modules, {'run_local_model_two_step_extract': module}), patch('streaming_decode.log_phase'):
            expected = run_decode_stream(FakeEngine(), old)
            steps = iterator(FakeEngine(), new, cooperative=True)
            self.assertEqual(next(steps), {'active': 0, 'graph_calls': 0})
            yielded = []
            while True:
                try:
                    yielded.append(next(steps))
                except StopIteration as end:
                    result = end.value
                    break
        self.assertEqual(old.results, new.results)
        self.assertEqual(result['graph_calls'], expected['graph_calls'])
        self.assertEqual(result['raw_decode_token_slots'], expected['raw_decode_token_slots'])
        self.assertTrue(yielded)

    def test_underfilled_wait_yields_then_fills_then_drains(self):
        FakeEngine, FakeSource, iterator = self.fixture()
        class OpenSource(FakeSource):
            upstream = True
            @property
            def upstream_exhausted(self):
                return not self.upstream and not self.waiting
            @property
            def closed(self):
                return not self.upstream and not self.waiting and not self.active
            def pull(self, *, block):
                if block:
                    raise AssertionError('cooperative source blocked')
                return super().pull(block=False)
        source, engine = OpenSource([(0, 11, 10)]), FakeEngine()
        module = SimpleNamespace(maybe_sync_device=lambda d: None)
        with patch.dict(sys.modules, {'run_local_model_two_step_extract': module}), patch('streaming_decode.log_phase'):
            steps = iterator(engine, source, cooperative=True)
            next(steps)
            state = next(steps)
            self.assertEqual(state, {'active': 1, 'graph_calls': 0})
            self.assertEqual(engine.decode_inputs, [])
            source.waiting.extend([(1, 21, 10), (2, 31, 10)])
            self.assertEqual(next(steps)['graph_calls'], 1)
            source.upstream = False
            for _ in steps:
                pass
        self.assertEqual(len(source.results), 3)

    @unittest.skipIf(sys.version_info < (3, 10), 'production requires Python >=3.10')
    def test_ready_admission_preserves_dirty_suffix_and_other_rows(self):
        from fixed_batch_engine import ContinuousBatchDecodeEngine, PrefilledGeneration
        from local_modeling_mineru import LocalMinerUStaticCache
        device_name = os.environ.get('READY_CACHE_TEST_DEVICE', 'cpu')
        if device_name.startswith('npu'):
            import torch_npu
        device = torch.device(device_name)
        destination = LocalMinerUStaticCache((torch.full((3, 2, 8, 4), -7., device=device),),
                                             (torch.full((3, 2, 8, 4), -9., device=device),), 8)
        ready = LocalMinerUStaticCache((torch.full((1, 2, 8, 4), 3., device=device),),
                                      (torch.full((1, 2, 8, 4), 5., device=device),), 8)
        released = []
        item = PrefilledGeneration(ready, {'token_id': 11}, 3, 5, lambda: released.append(True))
        engine = SimpleNamespace(model=SimpleNamespace(device=device))
        states, _, metrics = ContinuousBatchDecodeEngine.admit_prefilled_slots(engine, destination, [(1, 42, item)])
        self.assertEqual(states[1]['request_index'], 42)
        self.assertEqual(released, [True])
        self.assertEqual(metrics['ready_kv_admission_bytes'], 2 * 2 * 3 * 4 * 4)
        for dst, value, sentinel in zip(destination.flat_tensors(), (3., 5.), (-7., -9.)):
            self.assertTrue(torch.all(dst[1, :, :3] == value))
            self.assertTrue(torch.all(dst[1, :, 3:] == sentinel))
            self.assertTrue(torch.all(dst[0] == sentinel))
            self.assertTrue(torch.all(dst[2] == sentinel))


if __name__ == '__main__':
    unittest.main()
