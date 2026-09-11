"""Compare expanded-head requests with the saved 16k/full-head controls."""
import collections
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'lm_head_ab_20260911'
baseline=json.loads((OLD/'comparison.json').read_text())['rows']
rows=[]; comparisons=[]
def records(folder):
    return {r['sequence']:r for r in (json.loads(x) for x in (folder/'results.jsonl').read_text().splitlines())}
for batch in (2,8):
    folder=ROOT/f'b{batch}_expanded_measured'/f'b{batch}'
    measured=folder/'measured/results'
    summary=json.loads((measured/'summary.json').read_text())
    service=json.loads((folder/'service.json').read_text())['summary']
    actual=records(measured)
    responses=[r['service_result']['response'] for r in actual.values()]
    device_s=service['timing_s']['decode_model_and_argmax_device']
    row=dict(batch=batch,head='expanded_60416',count=len(actual),errors=summary['failed_request_count'],
        tables_s=summary['completion_qps'],latency_s=summary['latency_s'],
        generated_tokens=sum(len(r['token_ids']) for r in responses),
        stop_reasons=dict(collections.Counter(r['stop_reason'] for r in responses)),
        service_requests=service['requests'],graph_calls=service['graph_calls'],
        device_ms_per_call=device_s*1000/service['graph_calls'],
        device_calls_per_s=service['graph_calls']/device_s,
        physical_slots_per_device_s=service['raw_decode_token_slots']/device_s,
        useful_tokens_per_device_s=service['effective_decode_tokens']/device_s,
        device_scope='101 service requests including one warmup; events may include submission gaps')
    rows.append(row)
    for head in ('trimmed','full'):
        ref=records(OLD/f'b{batch}_{head}_measured'/f'b{batch}'/'measured/results')
        assert actual.keys()==ref.keys()
        differences=[]
        for seq in sorted(actual):
            assert actual[seq]['request_id']==ref[seq]['request_id']
            a=ref[seq]['service_result']['response']; b=actual[seq]['service_result']['response']
            changed=[k for k in ('token_ids','raw_text','text','stop_reason','input_tokens','projected_image_tokens','crop_size') if a[k]!=b[k]]
            if changed: differences.append(dict(request_id=actual[seq]['request_id'],changed=changed))
        b=next(r for r in baseline if r['batch']==batch and r['head']==head)
        comparisons.append(dict(batch=batch,reference=head,differences=differences,
            token_identical=sum(ref[s]['service_result']['response']['token_ids']==actual[s]['service_result']['response']['token_ids'] for s in actual),
            html_identical=sum(ref[s]['service_result']['response']['text']==actual[s]['service_result']['response']['text'] for s in actual),
            throughput_change_percent=100*(row['tables_s']/b['tables_s']-1),
            p95_change_percent=100*(row['latency_s']['p95']/b['latency_s']['p95']-1)))
ownership_checks=0
for p in ROOT.glob('*/ownership.jsonl'):
    samples=[json.loads(line) for line in p.read_text().splitlines()]
    assert samples and all(set(s['pids'])<=set(s['owned']) for s in samples),p
    assert samples[-1]['pids']==[],p
    ownership_checks+=len(samples)
result=dict(rows=rows,comparisons=comparisons,ownership_checks=ownership_checks,
    all_ownership_checks_clean=True,all_phases_released_device=True)
(ROOT/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
