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

    def test_legacy_remains_explicit(self):
        args = self.parse("--backend", "local-continuous-client", "--no-streaming-pages")
        self.assertFalse(args.streaming_pages)

    def test_other_backends_do_not_change(self):
        self.assertFalse(self.parse().streaming_pages)
        self.assertFalse(self.parse("--backend", "vllm-engine").streaming_pages)


if __name__ == "__main__":
    unittest.main()
