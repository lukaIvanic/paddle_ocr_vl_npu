"""Optional profiling is removed; mandatory dependency events and rate semantics remain."""
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

from test_text_simplification import ROOT
import p02_serving_runtime as serving_runtime


class RuntimeTimingTests(unittest.TestCase):
    def test_profiling_removed_but_dependency_events_retained(self):
        source = Path(serving_runtime.__file__).read_text()
        self.assertNotIn('enable_timing=True', source)
        self.assertFalse(hasattr(serving_runtime, 'DeviceTimeline'))
        self.assertFalse(hasattr(serving_runtime, 'PrefillDeviceTiming'))
        self.assertIn('prefill_ready_event.synchronize()', source)
        self.assertIn('first_token_ready.synchronize()', source)
        self.assertIn('done_event = torch_npu.npu.Event()', source)

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
