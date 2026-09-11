"""Summarize completed head comparisons without changing benchmark inputs."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    rows = []
    outputs = {}
    for batch, head in ((2, 'trimmed'), (2, 'full'), (8, 'trimmed'), (8, 'full')):
        folder = args.root / f'b{batch}_{head}_measured' / f'b{batch}'
        if not (folder / 'service.json').exists():
            continue
        measured = folder / 'measured/results'
        summary = json.loads((measured / 'summary.json').read_text())
        service = json.loads((folder / 'service.json').read_text())['summary']
        records = [json.loads(line) for line in (measured / 'results.jsonl').read_text().splitlines()]
        outputs[batch, head] = {r['sequence']: r for r in records}
        tokens = sum(len(r['service_result']['response']['token_ids']) for r in records)
        device_s = service['timing_s']['decode_model_and_argmax_device']
        row = dict(batch=batch, head=head, count=summary['request_count'],
                   errors=summary['failed_request_count'], tables_s=summary['completion_qps'],
                   latency_s=summary['latency_s'], generated_tokens=tokens,
                   output_tokens_per_client_wall_s=tokens / summary['run_wall_s'],
                   device_decode_scope='service lifetime: 100 measured requests plus one real warmup',
                   service_requests=service['requests'], decode_calls=service['graph_calls'],
                   device_decode_s=device_s,
                   device_ms_per_call=device_s * 1000 / service['graph_calls'],
                   device_calls_per_s=service['graph_calls'] / device_s,
                   physical_decode_slots_per_device_s=service['raw_decode_token_slots'] / device_s,
                   useful_decode_tokens_per_device_s=service['effective_decode_tokens'] / device_s)
        rows.append(row)
    comparisons = []
    for batch in (2, 8):
        if (batch, 'trimmed') not in outputs or (batch, 'full') not in outputs:
            continue
        a, b = outputs[batch, 'trimmed'], outputs[batch, 'full']
        assert a.keys() == b.keys()
        differing = []
        for sequence in sorted(a):
            x, y = a[sequence], b[sequence]
            assert x['request_id'] == y['request_id']
            x, y = x['service_result']['response'], y['service_result']['response']
            changed = [k for k in ('token_ids', 'raw_text', 'text', 'stop_reason',
                                   'input_tokens', 'projected_image_tokens', 'crop_size') if x[k] != y[k]]
            if changed:
                differing.append(dict(sequence=sequence, request_id=a[sequence]['request_id'],
                                      changed=changed, trimmed_tokens=len(x['token_ids']),
                                      full_tokens=len(y['token_ids'])))
        comparisons.append(dict(batch=batch, differing=differing,
                                token_identical=sum(a[i]['service_result']['response']['token_ids'] ==
                                    b[i]['service_result']['response']['token_ids'] for i in a),
                                html_identical=sum(a[i]['service_result']['response']['text'] ==
                                    b[i]['service_result']['response']['text'] for i in a)))
    print(json.dumps(dict(rows=rows, comparisons=comparisons), indent=2))


if __name__ == '__main__':
    main()
