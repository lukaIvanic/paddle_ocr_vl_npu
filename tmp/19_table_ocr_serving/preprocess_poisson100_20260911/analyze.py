"""Paired open-loop comparison; include queueing and every measured response."""
from collections import Counter
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def stats(values):
    v = sorted(values)
    def q(p):
        at = (len(v)-1)*p
        lo, hi = math.floor(at), math.ceil(at)
        return v[lo]+(v[hi]-v[lo])*(at-lo)
    return dict(mean=sum(v)/len(v), p50=q(.5), p95=q(.95), max=max(v))


def main():
    schedule = rows(ROOT/'schedule.jsonl')
    result = []
    control = None
    for variant in ('pillow', 'both'):
        folder = ROOT/f'b8_{variant}_measured/b8'
        measured = folder/'measured/results'
        assert rows(measured/'schedule.jsonl') == schedule
        requests = sorted(rows(measured/'results.jsonl'), key=lambda r:r['sequence'])
        summary = json.loads((measured/'summary.json').read_text())
        assert len(requests) == 100 and summary['failed_request_count'] == 0
        responses = [r['service_result']['response'] for r in requests]
        if control is None:
            control = responses
        mismatches = []
        for request, before, response, arrival in zip(requests, control, responses, schedule):
            assert request['status'] == 'ok'
            assert request['request_id'] == arrival['request_id']
            assert request['scheduled_offset_s'] == arrival['scheduled_offset_s']
            for key in ('input_tokens', 'projected_image_tokens', 'crop_size'):
                assert before[key] == response[key]
            if before['token_ids'] != response['token_ids']:
                mismatches.append(dict(request_id=request['request_id'],
                    text_equal=before['text']==response['text'],
                    before_text=before['text'], after_text=response['text']))
        ownership = rows(folder.parent/'ownership.jsonl')
        assert all(set(r['pids'])<=set(r['owned']) for r in ownership)
        assert ownership[-1]['pids'] == []
        dispatches = sorted(r['dispatch_offset_s'] for r in requests)
        peak = max(sum(t<=x<t+1 for x in dispatches) for t in dispatches)
        metrics = {k: stats([r['timing_s'][k] for r in responses]) for k in
                   ('cpu_preprocess_background_service', 'cpu_preprocess_background_queue_wait')}
        for key in ('prefill_blocked_s', 'scheduler_idle_blocked_s'):
            metrics[key] = stats([r['scheduling_metrics']['cpu_readiness'][key] for r in responses])
        result.append(dict(variant=variant, target_qps=6,
            completed_qps=100/summary['run_wall_s'],
            actual_arrival_qps=100/dispatches[-1], peak_1s_arrivals=peak,
            latency_s=summary['request_latency_s'], scheduled_latency_s=summary['scheduled_latency_s'],
            dispatch_lag_s=summary['dispatch_lag_s'],
            max_outstanding=summary['max_active_requests'], drain_s=summary['drain_after_last_arrival_s'],
            metrics=metrics, generated_tokens=sum(r['generated_tokens_including_eos'] for r in responses),
            stop_reasons=dict(Counter(r['stop_reason'] for r in responses)),
            native_matches=100-len(mismatches), differences=mismatches,
            ownership_snapshots=len(ownership)))
    (ROOT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
