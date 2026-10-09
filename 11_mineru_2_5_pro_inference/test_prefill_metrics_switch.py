import unittest
from unittest.mock import Mock,patch
import torch
from mineru_prefill_timing import PrefillDeviceTimeline


class TimelineSwitchTests(unittest.TestCase):
    def test_production_cli_can_disable_default_metrics(self):
        from run_page_pipeline import pipeline_args
        args=['--dataset-json','/tmp/unused.json','--output-dir','/tmp/unused']
        self.assertTrue(pipeline_args(args).local_prefill_metrics)
        self.assertFalse(pipeline_args(args+['--no-local-prefill-metrics']).local_prefill_metrics)

    def test_disabled_does_not_create_events_or_sync(self):
        timeline=PrefillDeviceTimeline(torch.device('cpu'),enabled=False)
        value=object();fn=Mock(return_value=value)
        with patch.object(timeline,'_event',side_effect=AssertionError('event created')):
            self.assertIs(timeline.measure('stage',fn),value)
            self.assertEqual(timeline.resolve(),{})
        fn.assert_called_once_with()
    def test_enabled_keeps_existing_timing(self):
        timeline=PrefillDeviceTimeline(torch.device('cpu'))
        start=Mock();end=Mock();start.elapsed_time.return_value=123.0
        with patch.object(timeline,'_event',side_effect=[start,end]):
            self.assertEqual(timeline.measure('stage',lambda:7),7)
        self.assertEqual(timeline.resolve(),{'stage':0.123})
        start.record.assert_called_once();end.record.assert_called_once();end.synchronize.assert_called_once()
    def test_disabled_preserves_exceptions(self):
        timeline=PrefillDeviceTimeline(torch.device('cpu'),enabled=False)
        def fail():raise ValueError('real operation failed')
        with self.assertRaisesRegex(ValueError,'real operation'):timeline.measure('stage',fail)


if __name__=='__main__':unittest.main()
