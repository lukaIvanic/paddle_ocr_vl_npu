"""Exercise real HTTP streaming, resume and stall handling without HF or an NPU."""
import contextlib
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest

from download_hr_reference import download_file

DATA = b'benchmark payload\n' * 1000
DIGEST = hashlib.sha256(DATA).hexdigest()


class Handler(BaseHTTPRequestHandler):
    requests = []

    def log_message(self, *args):
        pass

    def do_GET(self):
        mode = self.path.split('/')[1]
        start = int(self.headers.get('Range', 'bytes=0-').split('=')[1].split('-')[0])
        self.requests.append((mode, start))
        if mode == 'stall':
            time.sleep(2)
            return
        if mode == 'ignore':
            start = 0
        payload = (b'x' * len(DATA) if mode == 'corrupt' else DATA)[start:]
        self.send_response(206 if start else 200)
        if start:
            self.send_header('Content-Range', f'bytes {start}-{len(DATA)-1}/{len(DATA)}')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class DownloadChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.args = SimpleNamespace(socket_timeout=3, stall_seconds=0.7,
                                    attempt_seconds=5, progress_seconds=0.1, attempts=1)
        Handler.requests = []

    def run_download(self, modes):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            download_file(self.root, 'file.parquet', DIGEST, len(DATA),
                          [self.base+'/'+mode for mode in modes], self.args)
        return [json.loads(line) for line in output.getvalue().splitlines()]

    def test_range_resume_and_verified_file_reuse(self):
        (self.root/'file.parquet.part').write_bytes(DATA[:123])
        self.run_download(['range'])
        self.assertEqual(Handler.requests, [('range', 123)])
        self.assertEqual((self.root/'file.parquet').read_bytes(), DATA)
        self.assertFalse((self.root/'file.parquet.part').exists())
        rows = self.run_download(['range'])
        self.assertEqual(len(Handler.requests), 1)
        self.assertTrue(rows[-1]['reused'])

    def test_ignored_range_restarts_without_appending(self):
        (self.root/'file.parquet.part').write_bytes(DATA[:123])
        self.run_download(['ignore'])
        self.assertEqual((self.root/'file.parquet').read_bytes(), DATA)

    def test_stall_prints_heartbeats_then_uses_fallback(self):
        rows = self.run_download(['stall', 'range'])
        progress = [r for r in rows if r['event'] == 'progress' and r['endpoint'].endswith('/stall')]
        self.assertGreaterEqual(len(progress), 3)
        self.assertGreater(progress[-1]['no_data_s'], 0.2)
        failure = next(r for r in rows if r['event'] == 'attempt_failed')
        self.assertIn('No new bytes', failure['error'])
        self.assertEqual((self.root/'file.parquet').read_bytes(), DATA)

    def test_corrupt_download_is_not_published_and_fallback_recovers(self):
        rows = self.run_download(['corrupt', 'range'])
        self.assertIn('SHA256 mismatch', next(r['error'] for r in rows if r['event'] == 'attempt_failed'))
        self.assertEqual(Handler.requests, [('corrupt', 0), ('range', 0)])
        self.assertEqual((self.root/'file.parquet').read_bytes(), DATA)

    def test_all_endpoints_fail_without_publishing_file(self):
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
            download_file(self.root, 'file.parquet', DIGEST, len(DATA),
                          [self.base+'/corrupt'], self.args)
        self.assertFalse((self.root/'file.parquet').exists())

    def test_existing_mismatched_file_is_preserved(self):
        (self.root/'file.parquet').write_bytes(b'other dataset')
        with self.assertRaises(ValueError), contextlib.redirect_stdout(io.StringIO()):
            self.run_download(['range'])
        self.assertEqual((self.root/'file.parquet').read_bytes(), b'other dataset')
        self.assertEqual(Handler.requests, [])


if __name__ == '__main__':
    unittest.main()
