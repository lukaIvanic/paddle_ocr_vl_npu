"""Pair the same 100 arrivals before/after real-request warmup; no inference."""
import argparse
import json
from pathlib import Path
from statistics import mean


def load_rows(path):
    return {row['sequence']: row for row in
        (json.loads(line) for line in path.read_text().splitlines())}


def stats(values):
    ordered = sorted(values)
    def percentile(p):
        position = (len(ordered) - 1) * p
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return dict(mean=mean(values), p50=percentile(.5), p95=percentile(.95), max=max(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    folders = [args.root / 'b8' / name / 'results' for name in ('first', 'second')]
    first, second = [load_rows(folder / 'results.jsonl') for folder in folders]
    assert len(first) == 100 and first.keys() == second.keys()
    assert (folders[0] / 'schedule.jsonl').read_bytes() == (folders[1] / 'schedule.jsonl').read_bytes()
    pairs = []
    mismatch_fields = ('token_ids', 'raw_text', 'text', 'stop_reason', 'input_tokens', 'projected_image_tokens')
    mismatches = {field: [] for field in mismatch_fields}
    for sequence in sorted(first):
        a, b = first[sequence], second[sequence]
        assert a['request_id'] == b['request_id'] and a['status'] == b['status'] == 'ok'
        responses = [row['service_result']['response'] for row in (a, b)]
        for field in mismatch_fields:
            if responses[0][field] != responses[1][field]:
                mismatches[field].append(sequence)
        pair = dict(sequence=sequence, request_id=a['request_id'],
            scheduled_offset_s=a['scheduled_offset_s'],
            first_latency_s=a['request_latency_s'], second_latency_s=b['request_latency_s'],
            extra_first_latency_s=a['request_latency_s'] - b['request_latency_s'],
            first_scheduled_latency_s=a['scheduled_latency_s'], second_scheduled_latency_s=b['scheduled_latency_s'],
            first_dispatch_lag_s=a['dispatch_lag_s'], second_dispatch_lag_s=b['dispatch_lag_s'],
            vision_bucket=responses[0]['vision']['bucket'], text_bucket=responses[0]['text_prefill']['bucket'])
        for field in ('timing_s', 'device_stage_s'):
            pair[field] = {name: dict(first=responses[0][field][name], second=responses[1][field][name],
                extra_first=responses[0][field][name] - responses[1][field][name])
                for name in responses[0][field] if name in responses[1][field]}
        pairs.append(pair)
    report = dict(note='Same server, same saved Poisson schedule. Paired differences include queueing and run variation, not only first-use cost.',
        mismatches=mismatches, groups={}, requests=pairs)
    for name, rows in (('all', pairs), ('first_10', pairs[:10]), ('remaining_90', pairs[10:])):
        report['groups'][name] = {label: stats([p[key] for p in rows]) for label, key in (
            ('first_latency_s', 'first_latency_s'), ('second_latency_s', 'second_latency_s'),
            ('extra_first_latency_s', 'extra_first_latency_s'))}
    report['largest_first_pass_penalties'] = sorted(pairs, key=lambda p:p['extra_first_latency_s'], reverse=True)[:10]
    (args.root / 'paired_comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = ['# First-use versus warmed Poisson100', '', report['note'], '',
        'All times below are seconds. Positive difference means the first pass was slower.', '',
        '| Request order | Table | First pass | Second pass | Difference |',
        '| --- | --- | ---: | ---: | ---: |']
    for p in pairs:
        lines.append(f"| {p['sequence']} | {p['request_id']} | {p['first_latency_s']:.4f} | {p['second_latency_s']:.4f} | {p['extra_first_latency_s']:+.4f} |")
    (args.root / 'per_request_latency.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(groups=report['groups'], mismatches=mismatches,
        first_request=pairs[0], largest_penalty=report['largest_first_pass_penalties'][0]), indent=2))
    assert not any(mismatches.values()), 'Generation differences require investigation'


if __name__ == '__main__':
    main()
