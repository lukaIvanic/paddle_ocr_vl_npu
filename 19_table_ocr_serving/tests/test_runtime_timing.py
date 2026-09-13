"""Device-event timing relocation checked with fake NPU events, not hardware."""
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

from test_text_simplification import ROOT
import serving_runtime


class RuntimeTimingTests(unittest.TestCase):
    def test_npu_event_order_and_report_match_previous_implementation(self):
        source = subprocess.check_output(['git', '-C', str(ROOT), 'show',
            '564da03f:19_table_ocr_serving/_support/utils/timing.py'], text=True)
        old = types.ModuleType('historical_timing')
        exec(compile(source, 'historical_timing', 'exec'), old.__dict__)
        device = types.SimpleNamespace(type='npu')

        def exercise(cls):
            trace = []
            counter = 0
            class Event:
                def __init__(self, *, enable_timing):
                    nonlocal counter
                    self.index = counter
                    counter += 1
                    trace.append(('event', self.index, enable_timing))
                def record(self):
                    trace.append(('record', self.index))
                def synchronize(self):
                    trace.append(('wait_event', self.index))
                def elapsed_time(self, end):
                    trace.append(('elapsed', self.index, end.index))
                    return (end.index - self.index) * .25
            fake = types.SimpleNamespace(npu=types.SimpleNamespace(Event=Event,
                synchronize=lambda device:trace.append(('wait_device', device.type))))
            with patch.dict(sys.modules, {'torch_npu':fake}), \
                 patch.object(serving_runtime, 'torch_npu', fake), \
                 patch('time.perf_counter_ns', side_effect=[1000,2000]):
                timeline = cls(device)
                self.assertEqual(timeline.measure('first', lambda:trace.append(('work',1)) or 11),11)
                self.assertEqual(timeline.measure('second', lambda:trace.append(('work',2)) or 22),22)
                self.assertFalse(any(event[0].startswith('wait') for event in trace))
                report = timeline.resolve_spans()
                self.assertEqual([e for e in trace if e[0].startswith('wait')], [('wait_event',3)])
                empty = cls(device).resolve_spans()
                self.assertEqual(empty,{})
                self.assertEqual(trace[-1],('wait_device','npu'))
            return trace, report

        self.assertEqual(exercise(old.DeviceTimeline), exercise(serving_runtime.DeviceTimeline))

    def test_rate_semantics_and_removed_package(self):
        self.assertEqual(serving_runtime.per_second(10,2),5.)
        for duration in (None,0,-1):
            self.assertIsNone(serving_runtime.per_second(10,duration))
        self.assertEqual(serving_runtime.per_second(0,2),0.)
        folder = ROOT/'19_table_ocr_serving/_support/utils'
        for filename in ('timing.py','metrics.py','__init__.py'):
            self.assertFalse((folder/filename).exists())


if __name__ == '__main__':
    unittest.main()
