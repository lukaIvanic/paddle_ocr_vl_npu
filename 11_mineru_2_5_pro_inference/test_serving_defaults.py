import unittest
from unittest.mock import patch
from run_official_transformers_omnidocbench import parse_args


class ServingDefaultTests(unittest.TestCase):
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
