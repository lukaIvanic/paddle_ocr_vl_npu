import tempfile
import unittest
from pathlib import Path

from run_hf_baseline import model_manifest, parse_args, sha256


class Contracts(unittest.TestCase):
    def test_defaults(self):
        args = parse_args(['--model', '/model', '--output-dir', '/out'])
        self.assertEqual(args.dtype, 'fp16')
        self.assertEqual(args.device, 'npu:0')
        self.assertEqual(len(args.images), 2)
        self.assertTrue(all(p.is_file() for p in args.images))

    def test_no_cpu_fallback(self):
        with self.assertRaises(SystemExit):
            parse_args(['--model', '/model', '--output-dir', '/out', '--device', 'cpu'])

    def test_repeat_required(self):
        with self.assertRaises(SystemExit):
            parse_args(['--model', '/model', '--output-dir', '/out', '--repeats', '0'])

    def test_hash_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'config.json').write_text('{}')
            (path / 'model.safetensors').write_bytes(b'weights')
            manifest = model_manifest(path, False)
            self.assertEqual(manifest['config.json']['sha256'], sha256(path / 'config.json'))
            self.assertIsNone(manifest['model.safetensors']['sha256'])
            self.assertIsNotNone(model_manifest(path, True)['model.safetensors']['sha256'])


if __name__ == '__main__':
    unittest.main()
