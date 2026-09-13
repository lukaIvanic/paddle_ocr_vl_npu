"""Summarize external memory samples and compare saved native OCR outputs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics


def read_rows(path):
    return [json.loads(line) for line in path.open()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--measurement', type=Path, required=True)
    parser.add_argument('--manual-results', type=Path, required=True)
    args = parser.parse_args()
    reference = args.repo / 'tmp/19_table_ocr_serving/integrated_runtime_20260913/poisson1000/cached/b8/qps6/measured'
    previous = {r['sequence']: r for r in read_rows(reference / 'results.jsonl')}
    schedule = read_rows(reference / 'schedule.jsonl')
    metrics = ('worker_Rss_KiB', 'http_Rss_KiB', 'server_tree_Pss_KiB',
               'device_hbm_MB', 'npu_process_memory_MB')
    samples = read_rows(args.measurement / 'memory.jsonl')
    idle = {}
    for phase in dict.fromkeys(r['phase'] for r in samples):
        if 'idle' in phase:
            rows = [r for r in samples if r['phase'] == phase]
            idle[phase] = {key: statistics.median(r[key] for r in rows) for key in metrics}
            idle[phase]['samples'] = len(rows)
    report = {'memory': {
        'sample_count': len(samples), 'idle_medians': idle,
        'sampled_peaks': {key: max(r[key] for r in samples) for key in metrics},
        'unique_device_hbm_MB': sorted({r['device_hbm_MB'] for r in samples}),
        'unique_npu_process_memory_MB': sorted({r['npu_process_memory_MB'] for r in samples}),
        'monitor_errors': json.loads((args.measurement / 'monitor_errors.json').read_text()),
    }, 'comparisons': {}}
    fields = ('token_ids', 'raw_text', 'text', 'stop_reason', 'crop_size', 'input_tokens', 'projected_image_tokens')
    folders = [('manual', args.manual_results)] + [(p.name, p / 'results') for p in sorted(args.measurement.glob('round*'))]
    for name, folder in folders:
        rows = read_rows(folder / 'results.jsonl')
        mismatch = {key: [] for key in fields}
        assert len(rows) == len(previous) == 1000
        assert len({r['sequence'] for r in rows}) == 1000
        assert read_rows(folder / 'schedule.jsonl') == schedule
        for row in rows:
            old = previous[row['sequence']]
            assert row['status'] == old['status'] == 'ok'
            assert row['request_id'] == old['request_id']
            for key in fields:
                if row['service_result']['response'][key] != old['service_result']['response'][key]:
                    mismatch[key].append(row['sequence'])
        summary = json.loads((folder / 'summary.json').read_text())
        report['comparisons'][name] = dict(
            results=str(folder), results_sha256=hashlib.sha256((folder / 'results.jsonl').read_bytes()).hexdigest(),
            exact_schedule_match=True, mismatch_sequences=mismatch,
            stop_reasons=dict(Counter(r['service_result']['response']['stop_reason'] for r in rows)),
            request_latency_s=summary['request_latency_s'],
            completed_qps=summary['completed_request_count'] / summary['run_wall_s'],
            failed_requests=summary['failed_request_count'],
        )
    reference_summary = json.loads((reference / 'summary.json').read_text())
    report['reference'] = dict(results=str(reference), request_latency_s=reference_summary['request_latency_s'])
    (args.measurement / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
