"""Compare every occurrence in the saved step-2 and step-3 flagship runs."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('control', type=Path)
parser.add_argument('candidate', type=Path)
parser.add_argument('--output', type=Path)
args = parser.parse_args()


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def requests(root):
    rows = records(root / 'b8/qps6/measured/results.jsonl')
    keyed = {row['sequence']: row for row in rows}
    assert len(keyed) == len(rows) == 1000, 'Missing or duplicate occurrences'
    return keyed


control, candidate = requests(args.control), requests(args.candidate)
assert control.keys() == candidate.keys(), 'Occurrence sequences differ'
fields = ('token_ids', 'raw_text', 'text', 'stop_reason', 'input_tokens',
          'projected_image_tokens', 'crop_size', 'prompt', 'crop_type')
mismatches = {field: [] for field in fields}
identity_mismatches = []
errors = []
for sequence in sorted(control):
    old, new = control[sequence], candidate[sequence]
    if old['request_id'] != new['request_id']:
        identity_mismatches.append(sequence)
    if new.get('error') or new.get('status') != old.get('status'):
        errors.append({'sequence': sequence, 'status': new.get('status'),
                       'error': new.get('error')})
    a = old['service_result']['response']
    b = new['service_result']['response']
    for field in fields:
        if field not in a or field not in b or a[field] != b[field]:
            mismatches[field].append(sequence)

schedule = Path('b8/qps6/measured/schedule.jsonl')
schedule_equal = (args.control / schedule).read_bytes() == (args.candidate / schedule).read_bytes()
ownership = records(args.candidate / 'ownership.jsonl')
foreign = [r for r in ownership if set(r['pids']) - set(r['owned'])]
metrics_old = json.loads((args.control / 'results.json').read_text())[0]
metrics_new = json.loads((args.candidate / 'results.json').read_text())[0]
metric_fields = ('mean_s', 'p50_s', 'p90_s', 'p95_s', 'p99_s', 'max_s', 'completed_qps')
report = {
    'count': len(candidate),
    'sequence_sha256': hashlib.sha256(json.dumps([candidate[i]['request_id'] for i in sorted(candidate)]).encode()).hexdigest(),
    'schedule_byte_identical': schedule_equal,
    'request_id_mismatches': identity_mismatches,
    'errors_or_status_changes': errors,
    'mismatches_by_response_field': mismatches,
    'ownership_checks': len(ownership),
    'foreign_device_process_records': foreign,
    'final_device_pids': ownership[-1]['pids'],
    'control': metrics_old,
    'candidate': metrics_new,
    'metric_change_percent': {k: (metrics_new[k] / metrics_old[k] - 1) * 100 for k in metric_fields},
}
report['complete_output_and_workload_parity'] = not any(mismatches.values()) and not identity_mismatches and not errors and schedule_equal
print(json.dumps(report, indent=2))
if args.output:
    args.output.write_text(json.dumps(report, indent=2) + '\n')
