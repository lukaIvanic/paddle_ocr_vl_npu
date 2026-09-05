import unittest
from unittest.mock import patch
from types import SimpleNamespace
from run_official_transformers_omnidocbench import parse_args, apply_processor_pixel_limits


class ServingDefaultTests(unittest.TestCase):
    def test_pixel_limits_update_current_and_legacy_processor_fields(self):
        processor = SimpleNamespace(size={'shortest_edge': 50176, 'longest_edge': 1605632},
                                    patch_size=14, merge_size=2)
        apply_processor_pixel_limits(processor)
        self.assertEqual(processor.size['longest_edge'], 1605632)
        apply_processor_pixel_limits(processor, min_pixels=25088, max_pixels=1103872)
        self.assertEqual(processor.max_pixels, processor.size['longest_edge'])
        self.assertEqual(processor.min_pixels, processor.size['shortest_edge'])
        self.assertEqual(processor.max_pixels // 14**2, 5632)
        for lower, upper in [(0, 100), (25088, 10), (1, 196), (25088, -1)]:
            with self.assertRaises(ValueError):
                apply_processor_pixel_limits(processor, min_pixels=lower, max_pixels=upper)

    def test_pixel_limit_switches(self):
        self.assertIsNone(self.parse().processor_max_pixels)
        args = self.parse('--processor-min-pixels', '25088', '--processor-max-pixels', '1103872')
        self.assertEqual(args.processor_min_pixels, 25088)
        self.assertEqual(args.processor_max_pixels, 1103872)

    def parse(self, *args):
        with patch("sys.argv", ["mineru", "--output-dir", "/tmp/unused-mineru-test", *args]):
            return parse_args()

    def test_continuous_defaults_to_serving(self):
        args = self.parse("--backend", "local-continuous-client")
        self.assertTrue(args.streaming_pages)
        self.assertEqual(args.streaming_page_window, 32)
        self.assertEqual(args.local_decode_diagnostic_steps, 0)
        self.assertFalse(args.local_decode_diagnostic_sync)
        self.assertEqual(args.local_decode_diagnostic_boundary_period, 1408)
        self.assertEqual(args.local_decode_increfa_length_mode, "none")
        self.assertEqual(args.local_decode_filler_control, "retain")

    def test_decode_diagnostic_controls_parse(self):
        args = self.parse(
            "--backend", "local-continuous-client",
            "--local-decode-diagnostic-steps", "17",
            "--local-decode-diagnostic-sync",
            "--local-decode-diagnostic-boundary-period", "896",
            "--local-decode-increfa-length-mode", "pse_sentinel_310p",
            "--local-decode-filler-control", "advance",
        )
        self.assertEqual(args.local_decode_diagnostic_steps, 17)
        self.assertTrue(args.local_decode_diagnostic_sync)
        self.assertEqual(args.local_decode_diagnostic_boundary_period, 896)
        self.assertEqual(args.local_decode_increfa_length_mode, "pse_sentinel_310p")
        self.assertEqual(args.local_decode_filler_control, "advance")

    def test_legacy_remains_explicit(self):
        args = self.parse("--backend", "local-continuous-client", "--no-streaming-pages")
        self.assertFalse(args.streaming_pages)

    def test_other_backends_do_not_change(self):
        self.assertFalse(self.parse().streaming_pages)
        self.assertFalse(self.parse("--backend", "vllm-engine").streaming_pages)


if __name__ == "__main__":
    unittest.main()
