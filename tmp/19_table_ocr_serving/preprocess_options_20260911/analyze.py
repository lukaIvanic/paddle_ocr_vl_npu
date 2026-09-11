"""Compare identical 100-request sequences; retain timing and output changes."""
from collections import Counter
import difflib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VARIANTS = ('pillow', 'kornia', 'uint8', 'both')
TIMINGS = ('cpu_image_decode', 'cpu_image_and_prompt_preprocess',
           'cpu_mrope_index', 'cpu_pin_memory', 'cpu_preprocess_background_service',
           'cpu_preprocess_background_queue_wait')


def stats(values):
    values = sorted(values)
    def percentile(p):
        at = (len(values) - 1) * p
        lo, hi = math.floor(at), math.ceil(at)
        return values[lo] + (values[hi] - values[lo]) * (at - lo)
    return dict(mean=sum(values)/len(values), p50=percentile(.5),
                p95=percentile(.95), max=max(values), total=sum(values))


def load(batch, variant):
    folder = ROOT / f'b{batch}_{variant}_measured' / f'b{batch}' / 'measured/results'
    return ([json.loads(line) for line in (folder/'results.jsonl').read_text().splitlines()],
            json.loads((folder/'summary.json').read_text()))


def main():
    comparisons = []
    for batch in (2, 8):
        control, _ = load(batch, 'pillow')
        control = {r['sequence']: r for r in control}
        for variant in VARIANTS:
            rows, summary = load(batch, variant)
            assert len(rows) == 100 and summary['failed_request_count'] == 0
            responses = [r['service_result']['response'] for r in rows]
            differences = []
            for r, response in zip(rows, responses):
                before_row = control[r['sequence']]
                assert r['request_id'] == before_row['request_id']
                before = before_row['service_result']['response']
                for key in ('crop_size', 'input_tokens', 'projected_image_tokens'):
                    assert response[key] == before[key], (r['request_id'], key)
                if response['token_ids'] != before['token_ids']:
                    differences.append(dict(
                        request_id=r['request_id'],
                        baseline_token_count=len(before['token_ids']),
                        variant_token_count=len(response['token_ids']),
                        text_equal=response['text'] == before['text'],
                        raw_text_equal=response['raw_text'] == before['raw_text'],
                        stop_reasons=[before['stop_reason'],response['stop_reason']],
                        text_diff=list(difflib.unified_diff(
                            before['text'].splitlines(), response['text'].splitlines(),
                            fromfile='pillow',tofile=variant,lineterm='')),
                    ))
            metrics = {key: stats([r['timing_s'].get(key, 0) for r in responses]) for key in TIMINGS}
            for key in ('prefill_blocked_s', 'scheduler_idle_blocked_s'):
                values = [r['scheduling_metrics']['cpu_readiness'][key] for r in responses]
                metrics[key] = dict(**stats(values), over_1ms=sum(v>.001 for v in values))
            device_names = sorted(set().union(*(r['device_stage_s'] for r in responses)))
            device = {key: stats([r['device_stage_s'].get(key,0) for r in responses])
                      for key in device_names}
            ownership = [json.loads(line) for line in
                         (ROOT/f'b{batch}_{variant}_measured/ownership.jsonl').read_text().splitlines()]
            assert all(set(r['pids']) <= set(r['owned']) for r in ownership)
            assert ownership[-1]['pids'] == []
            comparisons.append(dict(batch=batch, variant=variant,
                tables_s=summary['completion_qps'], latency_s=summary['latency_s'],
                metrics=metrics, device_stage_s=device,
                generated_tokens=sum(r['generated_tokens_including_eos'] for r in responses),
                completion_reasons=dict(Counter(r['stop_reason'] for r in responses)),
                native_match_count=100-len(differences),
                text_match_count=100-sum(not r['text_equal'] for r in differences),
                differences=differences, ownership_snapshots=len(ownership)))
    (ROOT/'analysis.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    for r in comparisons:
        print(json.dumps({k:r[k] for k in ('batch','variant','tables_s','latency_s',
              'generated_tokens','completion_reasons','native_match_count','text_match_count')}))
        print(json.dumps(r['metrics']))


if __name__ == '__main__':
    main()
