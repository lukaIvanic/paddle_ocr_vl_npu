#!/usr/bin/env python3
"""Send OmniDocBench table, text or formula crops and print live latency results.

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
from typing import TextIO
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import urlopen

from PIL import Image


# These are separate annotated regions, not inline formulas inside text blocks.
CROP_CATEGORIES = {
    'table': 'table',
    'text': 'text_block',
    'formula': 'equation_isolated',
}


def main() -> None:
    args = parse_args()

    # Check that the server can accept OCR requests before preparing any images.
    ready_url = urlunsplit(urlsplit(args.api_url)._replace(path='/ready', query=''))
    with urlopen(ready_url, timeout=args.timeout_s) as response:
        server_status = json.load(response)
    if not server_status.get('ready'):
        raise RuntimeError('The OCR server is not ready yet.')

    # Choose which crops to send and when to send them. Save that exact schedule
    # so each measured response can be traced back to its crop and arrival time.
    crops = read_crop_annotations(args.omnidocbench, args.crop_type)
    request_count = len(crops) if args.requests == 'all' else args.requests
    schedule = create_request_schedule(crops, request_count, args.qps, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Existing empty directories are fine; previous run files are never overwritten.
    with (args.output_dir / 'results.jsonl').open('x', encoding='utf-8') as results_file:
        with (args.output_dir / 'schedule.jsonl').open('x', encoding='utf-8') as schedule_file:
            for request in schedule:
                schedule_file.write(json.dumps(request) + '\n')

        # Crop and encode images before measuring. The benchmark then sends one
        # warmup request, followed by all scheduled requests, and waits for replies.
        print(
            f'Found {len(crops)} {args.crop_type} crops. '
            'Preparing crops before the measured run...',
            flush=True,
        )
        crop_pngs = prepare_crop_pngs(args.omnidocbench, schedule)
        summary = asyncio.run(run_benchmark(args, schedule, crop_pngs, results_file))

    # Save the measured totals together with the settings needed to repeat the run.
    summary.update(
        dataset=str(args.omnidocbench.resolve()),
        crop_type=args.crop_type,
        annotation_category=CROP_CATEGORIES[args.crop_type],
        available_crops=len(crops),
        requests=request_count,
        target_qps=args.qps,
        seed=args.seed,
        server_configuration=server_status.get('configuration'),
        workload='Balanced crop repetitions, globally shuffled; seeded Poisson arrivals.',
    )
    with (args.output_dir / 'summary.json').open('x', encoding='utf-8') as summary_file:
        json.dump(summary, summary_file, indent=2)
        summary_file.write('\n')
    print(f'Summary: {args.output_dir / "summary.json"}', flush=True)
    if summary['failed']:
        raise SystemExit(1)


# Prepare the workload: crop locations, request order/times, then encoded images.


def read_crop_annotations(dataset_directory: Path, crop_type: str) -> list[dict]:
    """Find the selected type of crop in the annotations; no layout model runs here."""
    annotation_path = dataset_directory / 'OmniDocBench.json'
    pages = json.loads(annotation_path.read_text(encoding='utf-8'))
    annotation_category = CROP_CATEGORIES[crop_type]
    crops = []
    for page_index, page in enumerate(pages):
        page_info = page['page_info']
        for region in page['layout_dets']:
            if region['category_type'] != annotation_category or region.get('ignore'):
                continue

            # The annotation lists alternating x/y coordinates around the crop.
            # Enclose them in a rectangle, rounded outward and clipped to the page.
            x_coordinates = region['poly'][0::2]
            y_coordinates = region['poly'][1::2]
            left = max(0, math.floor(min(x_coordinates)))
            top = max(0, math.floor(min(y_coordinates)))
            right = min(page_info['width'], math.ceil(max(x_coordinates)))
            bottom = min(page_info['height'], math.ceil(max(y_coordinates)))
            crops.append({
                'crop_id': f"page_{page_index:06d}_{crop_type}_{region['anno_id']}",
                'crop_type': crop_type,
                'image': page_info['image_path'],
                'bbox': [left, top, right, bottom],
            })
    if not crops:
        raise ValueError(f'No non-ignored {crop_type} crops found in OmniDocBench.json.')
    return crops


def create_request_schedule(
    crops: list[dict], request_count: int, qps: float, seed: int,
) -> list[dict]:
    """Choose the requested number of crops and assign each a planned arrival time."""
    selection_random = random.Random(seed)

    # For 1,000 requests and 665 tables: include all 665, pick another 335,
    # then shuffle the complete list. Smaller runs select only the needed subset.
    complete_dataset_repetitions = request_count // len(crops)
    remaining_request_count = request_count % len(crops)
    selected_crops = crops * complete_dataset_repetitions
    selected_crops += selection_random.sample(crops, remaining_request_count)
    selection_random.shuffle(selected_crops)

    # Poisson arrivals use random gaps rather than evenly spaced requests.
    # A separate generator keeps arrival times independent of the crop selection.
    arrival_random = random.Random(seed)
    arrival_offset_s = 0.0
    schedule = []
    for sequence, crop in enumerate(selected_crops, 1):
        arrival_offset_s += arrival_random.expovariate(qps)
        schedule.append({
            'sequence': sequence,
            'scheduled_offset_s': arrival_offset_s,
            **crop,
        })
    return schedule


def prepare_crop_pngs(dataset_directory: Path, schedule: list[dict]) -> dict[str, bytes]:
    """Prepare PNG bytes for each selected crop before request timing starts."""
    # Group crops by page so each page is opened once. Repeated requests for the
    # same crop reuse its encoded PNG instead of cropping and encoding it again.
    crop_boxes_by_page = defaultdict(dict)
    for request in schedule:
        crop_boxes_by_page[request['image']][request['crop_id']] = request['bbox']

    crop_pngs = {}
    for page_name, crop_boxes in crop_boxes_by_page.items():
        with Image.open(dataset_directory / 'images' / page_name) as page:
            for crop_id, box in crop_boxes.items():
                with page.crop(box).convert('RGB') as crop:
                    png_buffer = io.BytesIO()
                    crop.save(png_buffer, format='PNG')
                    crop_pngs[crop_id] = png_buffer.getvalue()
    return crop_pngs


# Run the requests and measure what a client experiences, including waiting for OCR.


async def run_benchmark(
    args, schedule: list[dict], crop_pngs: dict[str, bytes], results_file: TextIO,
) -> dict:
    """Warm up once, send the scheduled requests, and summarize after all replies."""
    # One real request warms the serving path, outside the measured arrivals.
    print(f'WARMUP: one {args.crop_type} crop (not counted below)', flush=True)
    await asyncio.wait_for(
        send_crop_request(
            args.api_url,
            'benchmark-warmup',
            crop_pngs[schedule[0]['crop_id']],
            args.crop_type,
        ),
        args.timeout_s,
    )
    print(f'SENDING {len(schedule)} requests at {args.qps:g} QPS (Poisson arrivals)', flush=True)
    benchmark_started_s = time.perf_counter()
    successful_latencies_s = []
    dispatch_delays_s = []
    active_requests = 0
    maximum_active_requests = 0
    completed_requests = 0
    failed_requests = 0

    # Each scheduled request runs this function independently. It sends one crop,
    # records the reply or failure, and updates the counters for this benchmark.
    async def send_and_record_request(request: dict) -> None:
        nonlocal active_requests, maximum_active_requests, completed_requests, failed_requests

        # Latency starts at the actual attempt, not the planned arrival. Record
        # any delay in dispatch separately so a slow client is visible in results.
        request_started_s = time.perf_counter()
        dispatch_delay_s = max(
            0.0,
            request_started_s - benchmark_started_s - request['scheduled_offset_s'],
        )
        active_requests += 1
        maximum_active_requests = max(maximum_active_requests, active_requests)
        print(
            f'SEND #{request["sequence"]:04d} active={active_requests} '
            f'lag={dispatch_delay_s * 1000:.1f}ms',
            flush=True,
        )
        response = None
        error = None
        try:
            response = await asyncio.wait_for(
                send_crop_request(
                    args.api_url,
                    f'benchmark-{request["sequence"]}',
                    crop_pngs[request['crop_id']],
                    args.crop_type,
                ),
                args.timeout_s,
            )
        except Exception as exception:
            error = f'{type(exception).__name__}: {exception}'
            failed_requests += 1
        request_latency_s = time.perf_counter() - request_started_s
        active_requests -= 1
        completed_requests += 1
        dispatch_delays_s.append(dispatch_delay_s)
        if error is None:
            successful_latencies_s.append(request_latency_s)

        result_record = dict(
            request,
            request_latency_s=request_latency_s,
            dispatch_lag_s=dispatch_delay_s,
            status='error' if error else 'ok',
            error=error,
            response=response,
        )
        results_file.write(json.dumps(result_record, ensure_ascii=False) + '\n')
        results_file.flush()
        print(
            f'RECV #{request["sequence"]:04d} {request_latency_s:.3f}s '
            f'completed={completed_requests}/{len(schedule)} '
            f'active={active_requests} {error or "OK"}',
            flush=True,
        )

    # Wait for each planned arrival, not for the previous request to finish.
    # Creating a task allows earlier requests to keep waiting for OCR in parallel.
    request_tasks = []
    for request in schedule:
        await asyncio.sleep(max(
            0.0,
            benchmark_started_s + request['scheduled_offset_s'] - time.perf_counter(),
        ))
        request_tasks.append(asyncio.create_task(send_and_record_request(request)))

    # The run ends only after all requests have returned or failed. Failed requests
    # count toward run duration and failures, but not successful-request latency.
    await asyncio.gather(*request_tasks)
    benchmark_duration_s = time.perf_counter() - benchmark_started_s
    summary = dict(
        completed=completed_requests,
        succeeded=len(successful_latencies_s),
        failed=failed_requests,
        max_active=maximum_active_requests,
        wall_s=benchmark_duration_s,
        successful_requests_per_s=len(successful_latencies_s) / benchmark_duration_s,
        mean_s=statistics.mean(successful_latencies_s) if successful_latencies_s else None,
        p50_s=percentile(successful_latencies_s, .50),
        p95_s=percentile(successful_latencies_s, .95),
        max_s=max(successful_latencies_s) if successful_latencies_s else None,
        max_dispatch_lag_s=max(dispatch_delays_s),
        latency_definition='Client request attempt to parsed response; successful requests only.',
    )

    latency_fields = []
    for field_name in ('mean_s', 'p50_s', 'p95_s', 'max_s'):
        latency_s = summary[field_name]
        if latency_s is None:
            latency_fields.append(f'{field_name}=n/a')
        else:
            latency_fields.append(f'{field_name}={latency_s:.3f}s')
    formatted_latencies = ' '.join(latency_fields)
    print(
        f'DONE completed={completed_requests} failed={failed_requests} '
        f'max_active={maximum_active_requests} {formatted_latencies}',
        flush=True,
    )
    return summary


# HTTP transport for one crop; scheduling and timing belong to the benchmark above.


async def send_crop_request(
    api_url: str, request_id: str, png_bytes: bytes, crop_type: str,
) -> dict:
    """Send one PNG to p01_serve and return its JSON response; no concurrency limit here."""
    url = urlsplit(api_url)
    query_parameters = dict(parse_qsl(url.query))
    query_parameters.update(crop_type=crop_type, request_id=request_id)
    request_target = (url.path or '/v1/ocr') + '?' + urlencode(query_parameters)
    reader, writer = await asyncio.open_connection(url.hostname, url.port or 80)
    try:
        request_headers = (
            f'POST {request_target} HTTP/1.1\r\n'
            f'Host: {url.netloc}\r\n'
            f'Content-Type: image/png\r\n'
            f'Content-Length: {len(png_bytes)}\r\n'
            'Connection: close\r\n\r\n'
        )
        writer.write(request_headers.encode('ascii'))
        writer.write(png_bytes)
        await writer.drain()

        # p01_serve returns Content-Length JSON responses and closes the connection.
        response_headers, body = (await reader.read()).split(b'\r\n\r\n', 1)
        status_code = int(response_headers.split(b'\r\n', 1)[0].split()[1])
        response = json.loads(body)
        if status_code != 200:
            raise RuntimeError(f'HTTP {status_code}: {response.get("error", response)}')
        return response
    finally:
        writer.close()
        await writer.wait_closed()


def percentile(values: list[float], fraction: float) -> float | None:
    """Interpolate between sorted measurements; for example, fraction=0.95 gives P95."""
    if not values:
        return None
    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * fraction
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    return (
        sorted_values[lower_index]
        + (sorted_values[upper_index] - sorted_values[lower_index]) * (position - lower_index)
    )


# Command-line settings; the benchmark flow starts in main() at the top.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--omnidocbench', type=Path, required=True,
        help='Folder containing OmniDocBench.json and images/.',
    )
    parser.add_argument('--api-url', default='http://127.0.0.1:8765/v1/ocr')
    parser.add_argument(
        '--crop-type', choices=CROP_CATEGORIES, default='table',
        help='Select tables, text blocks, or isolated formulas; default: table.',
    )
    parser.add_argument(
        '--requests', default='1000', metavar='COUNT|all',
        help='Number of measured requests, or all for every selected crop once; default: 1000.',
    )
    parser.add_argument(
        '--qps', type=float, default=6.0,
        help='Average arrival rate; requests arrive independently of completions.',
    )
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument(
        '--timeout-s', type=float, default=900.0,
        help='Client deadline for each request, including warmup.',
    )
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.requests != 'all':
        try:
            args.requests = int(args.requests)
        except ValueError:
            parser.error('--requests must be a positive integer or all.')
        if args.requests < 1:
            parser.error('--requests must be a positive integer or all.')
    if not math.isfinite(args.qps) or args.qps <= 0:
        parser.error('--qps must be finite and positive.')
    if not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        parser.error('--timeout-s must be finite and positive.')
    if urlsplit(args.api_url).scheme != 'http' or not urlsplit(args.api_url).hostname:
        parser.error('--api-url must be an http:// endpoint for p01_serve.')
    return args


if __name__ == '__main__':
    main()
