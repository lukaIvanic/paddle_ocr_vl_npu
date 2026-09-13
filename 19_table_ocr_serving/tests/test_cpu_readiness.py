"""CPU readiness attribution: no timing subtraction, synchronization or NPU."""
from collections import deque
from concurrent.futures import Future
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from _support.serving.scheduling_metrics import RequestSchedulingMetrics

class CPUReadinessTests(unittest.TestCase):
    def test_queue_service_idle_and_poll_split(self):
        m=RequestSchedulingMetrics(2); m.register('r',0)
        with patch('time.perf_counter',return_value=10): m.cpu_prefill_eligible('r',block=False)
        with patch('time.perf_counter',return_value=15): m.cpu_prefill_eligible('r',block=True)
        with patch('time.perf_counter',side_effect=AssertionError('repeated polls must not read the clock')):
            m.cpu_prefill_eligible('r',block=True)
        m.cpu_prepared('r',submitted_at=1,queue_wait_s=12,finished_at=20,consumed_at=21)
        c=m.finish('r',30)['cpu_readiness']
        self.assertEqual(c['prefill_blocked_s'],10)
        self.assertEqual(c['blocked_cpu_queue_s'],3)
        self.assertEqual(c['blocked_cpu_service_s'],7)
        self.assertEqual(c['scheduler_idle_blocked_s'],5)
        self.assertEqual(c['ready_to_consumer_poll_s'],1)

    def test_cpu_already_ready_before_free_slot(self):
        m=RequestSchedulingMetrics(8); m.register('r',0)
        with patch('time.perf_counter',return_value=10): m.cpu_prefill_eligible('r',block=False)
        m.cpu_prepared('r',submitted_at=1,queue_wait_s=2,finished_at=8,consumed_at=10.25)
        c=m.finish('r',30)['cpu_readiness']
        for key in ('prefill_blocked_s','blocked_cpu_queue_s','blocked_cpu_service_s','scheduler_idle_blocked_s'):
            self.assertEqual(c[key],0)
        self.assertEqual(c['ready_to_consumer_poll_s'],.25)

    def test_open_source_does_not_wait_or_count_while_slots_full(self):
        sys.modules.setdefault('torch_npu',types.ModuleType('torch_npu'))
        kornia_image=types.ModuleType('kornia_rs.image'); kornia_image.Image=object
        sys.modules.setdefault('kornia_rs.image',kornia_image)
        import serving_runtime as runtime
        now=[10.0]
        m=RequestSchedulingMetrics(2); m.register('r',0)
        source=runtime._OpenPrefillSource.__new__(runtime._OpenPrefillSource)
        future=Future(); source.pending=deque([('r',future)])
        source.scheduling_metrics=m
        source._submit_available=lambda **kwargs:None
        source.on_request_error=lambda *args:self.fail(str(args))
        source.recognizer=types.SimpleNamespace(_prefill_for_decode=lambda prepared,wait_s:prepared)
        with patch('time.perf_counter',side_effect=lambda:now[0]):
            self.assertIsNone(source.pull_for_decode_slots(block=False,available_slots=0))
            self.assertIsNone(m.requests['r'].cpu_eligible_at)
            self.assertIsNone(source.pull_for_decode_slots(block=False,available_slots=1))
            self.assertEqual(m.requests['r'].cpu_eligible_at,10)
            self.assertEqual(len(source.pending),1)
            prepared=types.SimpleNamespace(request_started=1.0,preparation_finished=18.0,
                cpu_timing=types.SimpleNamespace(cpu_preprocess_background_queue_wait=2.0))
            future.set_result(prepared); now[0]=20
            self.assertIs(source.pull_for_decode_slots(block=False,available_slots=1),prepared)
        self.assertEqual(m.requests['r'].cpu_readiness['prefill_blocked_s'],8)
        self.assertEqual(m.requests['r'].cpu_readiness['ready_to_consumer_poll_s'],2)
        self.assertEqual(len(source.pending),0)

if __name__=='__main__': unittest.main()
