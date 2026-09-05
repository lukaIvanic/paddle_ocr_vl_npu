import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace
from prefill_timing import PrefillDeviceTimeline
from vision_timing_report import summarize_vision_samples


class StatisticsTests(unittest.TestCase):
    def test_tagged_events_reuse_timing_and_resolve_only_once(self):
        samples = []
        timeline = PrefillDeviceTimeline(SimpleNamespace(type='npu'), samples)
        start, end = Mock(), Mock()
        start.elapsed_time.return_value = 12.5
        with patch.object(timeline, '_event', side_effect=[start, end]):
            self.assertEqual(timeline.measure('vision_transformer_blocks', lambda: 42,
                                             tags={'route': 'packed_768'}), 42)
        start.record.assert_called_once()
        end.record.assert_called_once()
        end.synchronize.assert_not_called()
        self.assertEqual(timeline.resolve(), {'vision_transformer_blocks': .0125})
        end.synchronize.assert_called_once()
        self.assertEqual(samples[0]['device_s'], .0125)
        self.assertEqual(timeline.resolve(), {})
        self.assertEqual(len(samples), 1)

    def test_weighted_rates_and_exact_overflow_shapes(self):
        rows = [dict(route=route, real_tokens=real, physical_tokens=physical,
                     members=1, device_s=duration)
                for route, real, physical, duration in [
                    ('bucket_768', 640, 768, .01),
                    ('bucket_768', 700, 768, .03),
                    ('packed_768', 720, 768, .02),
                    ('eager_overflow', 6000, 6000, .1),
                    ('eager_overflow', 7000, 7000, .2)]]
        report = summarize_vision_samples(rows)
        direct = report['by_route']['bucket_768']
        self.assertAlmostEqual(direct['real_tok_s'], 1340/.04)
        self.assertAlmostEqual(direct['latency_ms']['p50'], 20)
        self.assertAlmostEqual(direct['latency_ms']['p99'], 29.8)
        self.assertEqual(report['by_exact_shape']['eager_overflow:S7000']['calls'], 1)
        self.assertEqual(report['by_route']['packed_768']['calls'], 1)
        self.assertEqual(report['slowest_calls'][0]['physical_tokens'], 7000)


if __name__ == '__main__':
    unittest.main()
