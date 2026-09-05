"""CPU-only threshold screening; deliberately NOT a C2 scheduler simulation."""
import hashlib
import json
from pathlib import Path
from statistics import mean

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
ORDINARY = ROOT / 'table_closed_loop_random100_seed1_3a745ba_20260903'
SPEC = ROOT / 'table_phase_reference_6099805a_20260905'
paths = {'b1': ORDINARY / 'b1', 'b2': ORDINARY / 'b2',
         'spec_c1': SPEC / 'async_c1', 'spec_c2': SPEC / 'async_c2'}
data = {name: {r['request_id']: r for r in map(json.loads,
        (path / 'results.jsonl').read_text().splitlines())}
        for name, path in paths.items()}
hashes = {name: hashlib.sha256((p / 'tables.jsonl').read_bytes()).hexdigest()
          for name, p in paths.items()}
assert len(set(hashes.values())) == 1
ids = list(data['b1'])
assert len(ids) == 100
assert all(set(rows) == set(ids) for rows in data.values())
assert all(r['status'] == 'ok' and r['service_result']['stop_reason'] == 'eos'
           for rows in data.values() for r in rows.values())

def percentile(values, p):
    a = sorted(values)
    x = (len(a) - 1) * p / 100
    i = int(x)
    return a[i] + (a[min(i + 1, len(a) - 1)] - a[i]) * (x - i)

def distribution(values):
    return dict(mean=mean(values), p50=percentile(values, 50),
                p90=percentile(values, 90), p95=percentile(values, 95),
                p99=percentile(values, 99), maximum=max(values),
                over_2s=sum(x > 2 for x in values))

rows = []
for k in ids:
    b = data['b1'][k]['service_result']['response']
    s = data['spec_c1'][k]['service_result']['response']
    # Native output-ID count including EOS, never tokenize decoded text.
    n = len(b['token_ids'])
    assert n == b['generated_tokens_including_eos']
    rows.append(dict(request_id=k, b1_native_tokens=n,
        current_spec=s['route_lane'] == 'spec',
        b1_spec_output_identical=b['token_ids'] == s['token_ids'],
        **{name + '_s': data[name][k]['latency_s'] for name in paths}))

bins = []
for lo, hi in [(0,256), (256,512), (512,768), (768,1024), (1024,1536), (1536,100000)]:
    group = [r for r in rows if r['current_spec'] and lo <= r['b1_native_tokens'] < hi]
    bins.append(dict(min_tokens=lo, max_tokens_exclusive=hi, count=len(group),
        spec_faster=sum(r['spec_c1_s'] < r['b1_s'] for r in group),
        b1_mean_s=mean(r['b1_s'] for r in group),
        spec_mean_s=mean(r['spec_c1_s'] for r in group),
        spec_net_saving_s=sum(r['b1_s']-r['spec_c1_s'] for r in group)))

sweep = []
for t in [0,256,384,512,640,768,896,1024,1280,1536,2048,100000]:
    selected = [r for r in rows if r['current_spec'] and r['b1_native_tokens'] >= t]
    removed = [r for r in rows if r['current_spec'] and r['b1_native_tokens'] < t]
    # Keep the 39 existing ordinary routes at their current C1 timings.
    # Replace only newly rerouted tables using their historical ordinary B1 timings.
    values = [r['b1_s'] if r in removed else r['spec_c1_s'] for r in rows]
    sweep.append(dict(threshold_inclusive=t, selected_spec=len(selected),
        newly_ordinary=len(removed),
        retained_historical_b2_over_2s=sum(r['b2_s'] > 2 for r in selected),
        retained_current_c2_over_2s=sum(r['spec_c2_s'] > 2 for r in selected),
        saved_sequential_request_seconds=sum(r['spec_c1_s']-r['b1_s'] for r in removed),
        projected_c1_distribution_s=distribution(values),
        serial_reciprocal_mean_proxy=1/mean(values)))

result = dict(contract={
    'kind': 'offline perfect-length oracle screen, not a C2 performance prediction',
    'selection_sha256': hashes,
    'rule': 'existing height eligibility AND ordinary-B1 native output-ID count >= threshold',
    'invariance': 'No arrival index, companion identity, latency or acceptance in the routing rule.',
    'limitations': [
        'Historical ordinary B1 timings and contemporaneous speculative C1 timings are different runs.',
        'Thresholds are exploratory on these 100 samples, not held-out validation.',
        'A perfect length estimate does not predict draft acceptance.',
        'No estimator overhead is included; no model execution or serving code was changed.',
        'C2 ordering, batching, interruption and completion dynamics must be remeasured.',
        'Serial reciprocal mean is not C2 QPS or measured serving capacity.']},
    measured={name: json.loads((p/'summary.json').read_text())['latency_s']
              for name,p in paths.items()},
    bins=bins, thresholds=sweep, per_table=rows)
(OUT/'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
source = ROOT/'table_b1_latency_full_04fbc8e/client/tables.jsonl'
counts = {r['request_id']: int(r['output_tokens'])
          for r in map(json.loads, source.read_text().splitlines())}
counts.update({r['request_id']: r['b1_native_tokens'] for r in rows})
(OUT/'oracle_counts.json').write_text(json.dumps({
    'format': 'ordinary_b1_output_length_oracle_v1',
    'definition': 'ordinary B1 generated token count including EOS',
    'source': str(source.relative_to(ROOT.parent.parent)),
    'measured_100_override': str((ORDINARY/'b1/results.jsonl').relative_to(ROOT.parent.parent)),
    'provenance': '100 measured samples counted from native IDs; remaining warmup candidates use saved B1 output_tokens.',
    'token_counts': counts}, indent=2)+'\n')
print(json.dumps({'bins': bins, 'thresholds': sweep}, indent=2))
