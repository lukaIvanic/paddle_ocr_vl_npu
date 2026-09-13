"""The product client runs without Torch/NPU or research artifacts."""
import asyncio
from collections import Counter
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from urllib.parse import parse_qs, urlsplit

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import benchmark_tables as benchmark


class BenchmarkDataTests(unittest.TestCase):
    def test_dataset_crops_and_ignored_regions(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'images').mkdir()
            Image.new('RGB', (40, 30), 'white').save(root / 'images/page.png')
            region = dict(category_type='table', anno_id='one', poly=[1.2, 2.4, 20.8, 2.4, 20.8, 15.2, 1.2, 15.2])
            page = dict(page_info=dict(image_path='page.png', width=40, height=30),
                        layout_dets=[region, dict(region, ignore=True), dict(region, category_type='table_mask')])
            (root / 'OmniDocBench.json').write_text(json.dumps([page]))
            tables = benchmark.read_table_annotations(root)
            self.assertEqual(len(tables), 1)
            self.assertEqual(tables[0]['bbox'], [1, 2, 21, 16])
            payloads = benchmark.prepare_table_pngs(root, tables * 3)
            self.assertEqual(len(payloads), 1)
            with Image.open(io.BytesIO(next(iter(payloads.values())))) as crop:
                self.assertEqual(crop.size, (20, 14))

    def test_balanced_global_shuffle_and_reproducibility(self):
        tables = [dict(table_id=str(i)) for i in range(5)]
        schedule = benchmark.create_request_schedule(tables, 13, 6, 1)
        self.assertEqual(schedule, benchmark.create_request_schedule(tables, 13, 6, 1))
        counts = Counter(r['table_id'] for r in schedule)
        self.assertEqual(sorted(counts.values()), [2, 2, 3, 3, 3])
        self.assertEqual([r['sequence'] for r in schedule], list(range(1, 14)))
        self.assertTrue(all(a['scheduled_offset_s'] < b['scheduled_offset_s'] for a, b in zip(schedule, schedule[1:])))

    def test_percentiles(self):
        self.assertEqual(benchmark.percentile([1, 2, 3, 4], .5), 2.5)
        self.assertAlmostEqual(benchmark.percentile([1, 2, 3, 4], .95), 3.85)
        self.assertIsNone(benchmark.percentile([], .95))


class BenchmarkHttpTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_requests_failure_and_warmup_accounting(self):
        received = []
        async def handler(reader, writer):
            header = await reader.readuntil(b'\r\n\r\n')
            target = header.split(b' ')[1].decode()
            request_id = parse_qs(urlsplit(target).query)['request_id'][0]
            length = next(int(line.split(b':', 1)[1]) for line in header.split(b'\r\n') if line.lower().startswith(b'content-length:'))
            self.assertEqual(await reader.readexactly(length), b'image')
            received.append(request_id)
            await asyncio.sleep(.03)
            status = 503 if request_id == 'benchmark-2' else 200
            body = json.dumps({'error': 'full'} if status == 503 else {'text': '结果', 'token_ids': [10, 2]}).encode()
            writer.write(f'HTTP/1.0 {status} response\r\nContent-Length: {len(body)}\r\n\r\n'.encode() + body)
            await writer.drain()
            writer.close()
            await writer.wait_closed()
        server = await asyncio.start_server(handler, '127.0.0.1', 0)
        async with server:
            port = server.sockets[0].getsockname()[1]
            args = SimpleNamespace(api_url=f'http://127.0.0.1:{port}/v1/ocr', timeout_s=1, qps=1000)
            schedule = benchmark.create_request_schedule([{'table_id': 'one'}], 5, args.qps, 1)
            output = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()):
                summary = await benchmark.run_benchmark(args, schedule, {'one': b'image'}, output)
        self.assertEqual(len(received), 6)  # One warmup plus five measured requests.
        self.assertEqual(summary['completed'], 5)
        self.assertEqual(summary['succeeded'], 4)
        self.assertEqual(summary['failed'], 1)
        self.assertGreater(summary['max_active'], 1)
        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(len(rows), 5)
        self.assertEqual(sum(r['status'] == 'error' for r in rows), 1)


if __name__ == '__main__':
    unittest.main()
