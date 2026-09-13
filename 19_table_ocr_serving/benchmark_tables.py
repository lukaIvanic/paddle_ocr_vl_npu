#!/usr/bin/env python3
"""Send OmniDocBench table crops to a running server and print live latency results.

Only Pillow and Python's standard library are needed on this client machine.
The server owns inference; this script does not import the model or use an NPU.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
import io
import json
import math
from pathlib import Path
import random
import statistics
import time
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import urlopen

from PIL import Image


def main() -> None:
    args = parse_args()
    ready_url = urlunsplit(urlsplit(args.api_url)._replace(path='/ready', query=''))
    with urlopen(ready_url, timeout=args.timeout_s) as response:
        ready = json.load(response)
    if not ready.get('ready'):
        raise RuntimeError('The OCR server is not ready yet.')

    tables = read_tables(args.omnidocbench)
    schedule = make_schedule(tables, args.requests, args.qps, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Existing empty directories are fine; previous run files are never overwritten.
    with (args.output_dir / 'results.jsonl').open('x', encoding='utf-8') as results:
        with (args.output_dir / 'schedule.jsonl').open('x', encoding='utf-8') as saved_schedule:
            for request in schedule:
                saved_schedule.write(json.dumps(request) + '\n')
        print(f'Found {len(tables)} tables. Preparing crops before the measured run...', flush=True)
        images = prepare_images(args.omnidocbench, schedule)
        summary = asyncio.run(run_requests(args, schedule, images, results))
    summary.update(dataset=str(args.omnidocbench.resolve()), available_tables=len(tables),
                   requests=args.requests, target_qps=args.qps, seed=args.seed,
                   server_configuration=ready.get('configuration'),
                   workload='Balanced table repetitions, globally shuffled; seeded Poisson arrivals.')
    with (args.output_dir / 'summary.json').open('x', encoding='utf-8') as output:
        json.dump(summary, output, indent=2)
        output.write('\n')
    print(f'Summary: {args.output_dir / "summary.json"}', flush=True)
    if summary['failed']:
        raise SystemExit(1)


def read_tables(dataset: Path) -> list[dict]:
    """Use the annotated table rectangles, not layout detection or full-page OCR."""
    pages = json.loads((dataset / 'OmniDocBench.json').read_text(encoding='utf-8'))
    tables = []
    for page_index, page in enumerate(pages):
        info = page['page_info']
        for region in page['layout_dets']:
            if region['category_type'] != 'table' or region.get('ignore'):
                continue
            x, y = region['poly'][0::2], region['poly'][1::2]
            tables.append({
                'table_id': f"page_{page_index:06d}_table_{region['anno_id']}",
                'image': info['image_path'],
                'bbox': [max(0, math.floor(min(x))), max(0, math.floor(min(y))),
                         min(info['width'], math.ceil(max(x))), min(info['height'], math.ceil(max(y)))],
            })
    if not tables:
        raise ValueError('No non-ignored tables found in OmniDocBench.json.')
    return tables


def make_schedule(tables: list[dict], count: int, qps: float, seed: int) -> list[dict]:
    """Include every table in each full set, randomly fill the remainder, then shuffle all."""
    selection = random.Random(seed)
    chosen = tables * (count // len(tables)) + selection.sample(tables, count % len(tables))
    selection.shuffle(chosen)
    arrivals = random.Random(seed)
    offset = 0.0
    schedule = []
    for sequence, table in enumerate(chosen, 1):
        offset += arrivals.expovariate(qps)
        schedule.append({'sequence': sequence, 'scheduled_offset_s': offset, **table})
    return schedule


def prepare_images(dataset: Path, schedule: list[dict]) -> dict[str, bytes]:
    """Read each needed page once and keep encoded crops ready before starting the clock."""
    pages = defaultdict(dict)
    for request in schedule:
        pages[request['image']][request['table_id']] = request['bbox']
    images = {}
    for page_name, crops in pages.items():
        with Image.open(dataset / 'images' / page_name) as page:
            for table_id, box in crops.items():
                with page.crop(box).convert('RGB') as crop:
                    encoded = io.BytesIO()
                    crop.save(encoded, format='PNG')
                    images[table_id] = encoded.getvalue()
    return images


async def run_requests(args, schedule, images, results) -> dict:
    # One real request warms the serving path, outside the measured arrivals.
    print('WARMUP: one table (not counted below)', flush=True)
    await asyncio.wait_for(post_crop(args.api_url, 'benchmark-warmup', images[schedule[0]['table_id']]), args.timeout_s)
    print(f'SENDING {len(schedule)} requests at {args.qps:g} QPS (Poisson arrivals)', flush=True)
    started = time.perf_counter()
    latencies, lags = [], []
    active = maximum_active = completed = failed = 0

    async def recognize(request):
        nonlocal active, maximum_active, completed, failed
        sent = time.perf_counter()
        lag = max(0.0, sent - started - request['scheduled_offset_s'])
        active += 1
        maximum_active = max(maximum_active, active)
        print(f'SEND #{request["sequence"]:04d} active={active} lag={lag*1000:.1f}ms', flush=True)
        payload, error = None, None
        try:
            payload = await asyncio.wait_for(post_crop(args.api_url,
                f'benchmark-{request["sequence"]}', images[request['table_id']]), args.timeout_s)
        except Exception as exception:
            error = f'{type(exception).__name__}: {exception}'
            failed += 1
        elapsed = time.perf_counter() - sent
        active -= 1
        completed += 1
        lags.append(lag)
        if error is None:
            latencies.append(elapsed)
        row = dict(request, request_latency_s=elapsed, dispatch_lag_s=lag,
                   status='error' if error else 'ok', error=error, response=payload)
        results.write(json.dumps(row, ensure_ascii=False) + '\n')
        results.flush()
        print(f'RECV #{request["sequence"]:04d} {elapsed:.3f}s '
              f'completed={completed}/{len(schedule)} active={active} {error or "OK"}', flush=True)

    tasks = []
    for request in schedule:
        await asyncio.sleep(max(0.0, started + request['scheduled_offset_s'] - time.perf_counter()))
        tasks.append(asyncio.create_task(recognize(request)))
    await asyncio.gather(*tasks)  # Return only after every individual request finishes.
    wall = time.perf_counter() - started
    summary = dict(completed=completed, succeeded=len(latencies), failed=failed,
                   max_active=maximum_active, wall_s=wall, successful_requests_per_s=len(latencies)/wall,
                   mean_s=statistics.mean(latencies) if latencies else None,
                   p50_s=percentile(latencies, .50), p95_s=percentile(latencies, .95),
                   max_s=max(latencies) if latencies else None, max_dispatch_lag_s=max(lags),
                   latency_definition='Client request attempt to parsed response; successful requests only.')
    formatted = ' '.join(f'{key}={summary[key]:.3f}s' if summary[key] is not None else f'{key}=n/a'
                         for key in ('mean_s', 'p50_s', 'p95_s', 'max_s'))
    print(f'DONE completed={completed} failed={failed} max_active={maximum_active} {formatted}', flush=True)
    return summary


async def post_crop(api_url: str, request_id: str, image: bytes) -> dict:
    """One HTTP request; no thread pool or hidden limit on concurrent requests."""
    url = urlsplit(api_url)
    query = dict(parse_qsl(url.query))
    query.update(crop_type='table', request_id=request_id)
    target = (url.path or '/v1/ocr') + '?' + urlencode(query)
    reader, writer = await asyncio.open_connection(url.hostname, url.port or 80)
    try:
        headers = (f'POST {target} HTTP/1.1\r\nHost: {url.netloc}\r\n'
                   f'Content-Type: image/png\r\nContent-Length: {len(image)}\r\nConnection: close\r\n\r\n')
        writer.write(headers.encode('ascii'))
        writer.write(image)
        await writer.drain()
        # p01_serve returns Content-Length JSON responses and closes the connection.
        response_headers, body = (await reader.read()).split(b'\r\n\r\n', 1)
        status = int(response_headers.split(b'\r\n', 1)[0].split()[1])
        payload = json.loads(body)
        if status != 200:
            raise RuntimeError(f'HTTP {status}: {payload.get("error", payload)}')
        return payload
    finally:
        writer.close()
        await writer.wait_closed()


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--omnidocbench', type=Path, required=True, help='Folder containing OmniDocBench.json and images/.')
    parser.add_argument('--api-url', default='http://127.0.0.1:8765/v1/ocr')
    parser.add_argument('--requests', type=int, default=1000)
    parser.add_argument('--qps', type=float, default=6.0, help='Average arrival rate; requests arrive independently of completions.')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--timeout-s', type=float, default=900.0, help='Client deadline for each request, including warmup.')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.requests < 1 or not math.isfinite(args.qps) or args.qps <= 0:
        parser.error('--requests must be positive; --qps must be finite and positive.')
    if not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        parser.error('--timeout-s must be finite and positive.')
    if urlsplit(args.api_url).scheme != 'http' or not urlsplit(args.api_url).hostname:
        parser.error('--api-url must be an http:// endpoint for p01_serve.')
    return args


if __name__ == '__main__':
    main()
