"""Compare the saved Poisson100 control and current run, without inference."""
import hashlib
import json
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
REFERENCE = BASE.parent / 'preprocess_poisson100_20260911/b8_both_measured'
RELATIVE = Path('b8/measured/results')


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main():
    old, new = REFERENCE / RELATIVE, BASE / RELATIVE
    a = {row['sequence']: row for row in read_rows(old / 'results.jsonl')}
    b = {row['sequence']: row for row in read_rows(new / 'results.jsonl')}
    assert a.keys() == b.keys() and len(b) == 100
    assert read_rows(old / 'schedule.jsonl') == read_rows(new / 'schedule.jsonl')
    fields = ('token_ids', 'raw_text', 'text', 'stop_reason', 'crop_size',
              'input_tokens', 'projected_image_tokens')
    mismatches = {field: [] for field in fields}
    for sequence in sorted(a):
        assert a[sequence]['request_id'] == b[sequence]['request_id']
        assert a[sequence]['status'] == b[sequence]['status'] == 'ok'
        left, right = (row['service_result']['response'] for row in (a[sequence], b[sequence]))
        for field in fields:
            if left[field] != right[field]:
                mismatches[field].append(a[sequence]['request_id'])
    summaries = [json.loads((directory / 'summary.json').read_text()) for directory in (old, new)]
    performance = {}
    for label, summary in zip(('historical', 'current'), summaries):
        performance[label] = dict(latency_s=summary['request_latency_s'],
            scheduled_latency_s=summary['scheduled_latency_s'],
            completion_qps=summary['completed_request_count'] / summary['run_wall_s'],
            drain_s=summary['drain_after_last_arrival_s'],
            max_outstanding=summary['max_active_requests'],
            errors=summary['failed_request_count'])
    ownership = read_rows(BASE / 'ownership.jsonl')
    report = dict(schedule_identical=True, requests=100, mismatches=mismatches,
        performance=performance,
        current_stops=dict(Counter(row['service_result']['response']['stop_reason'] for row in b.values())),
        ownership=dict(checks=len(ownership),
            foreign_checks=sum(bool(set(row['pids']) - set(row['owned'])) for row in ownership),
            final_device_pids=ownership[-1]['pids']),
        result_sha256={label: hashlib.sha256((directory / 'results.jsonl').read_bytes()).hexdigest()
                       for label, directory in zip(('historical', 'current'), (old, new))})
    (BASE / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
