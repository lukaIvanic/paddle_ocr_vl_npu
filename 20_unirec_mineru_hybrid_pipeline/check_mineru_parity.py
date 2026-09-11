#!/usr/bin/env python3
"""NPU control: unchanged standalone stream versus hybrid ready-KV admission.

Uses the same real crop group, model, packing and B32 graph. This is a token
parity check, not a throughput benchmark or ground-truth accuracy evaluation.
"""
from collections import deque
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace

from run_pipeline import ROOT, build_parser, make_mineru
from hybrid_routing import MINERU_TASKS


def main():
    import torch
    from PIL import Image
    from streaming_decode import run_decode_stream

    args = build_parser().parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    actual = {}
    adapter = make_mineru(args, lambda key, text, ids, reason, model: actual.update({key: ids}))
    manifest = json.loads((ROOT / 'crops/manifest.json').read_text())
    requests = [SimpleNamespace(request_id=str(i), prompt=r['suggested_prompt'],
                               crop=Image.open(ROOT / 'crops' / r['file']).convert('RGB'))
                for i, r in enumerate(manifest) if r['suggested_prompt'] in MINERU_TASKS]
    reference = {}
    waiting = deque()
    with torch.inference_mode():
        for i, request in enumerate(requests):
            _, params, cpu = adapter.prepare_cpu(request)
            inputs, position, rope, _, _ = cpu
            waiting.append((int(request.request_id), adapter.client._finish_generation(inputs, params, position, rope)))

        class Source:
            @property
            def closed(self):
                return not waiting

            def pull(self, *, block=False):
                return waiting.popleft() if waiting else None

            def complete(self, index, ids):
                reference[str(index)] = ids

        reference_metrics = run_decode_stream(adapter.engine, Source())
        adapter.enqueue_page(requests)
        wakeup = Event()
        while not adapter.done:
            wakeup.clear()
            adapter.pump_preparation(wakeup.set)
            adapter.set_upstream(bool(adapter.pending), closed=True)
            if adapter.prefill_available:
                adapter.prefill()
            elif not adapter.pending or adapter.occupied >= adapter.capacity:
                adapter.advance(32)
            else:
                wakeup.wait()
    rows = [dict(request_id=r.request_id, prompt=r.prompt,
                 match=reference[r.request_id] == actual[r.request_id],
                 reference_ids=reference[r.request_id], hybrid_ids=actual[r.request_id])
            for r in requests]
    result = dict(all_match=all(r['match'] for r in rows), crops=len(rows),
                  reference_graph_calls=reference_metrics['graph_calls'],
                  hybrid_graph_calls=adapter.graph_calls, rows=rows,
                  ready_storage=adapter.ready_storage_summary())
    (output / 'parity.json').write_text(json.dumps(result, indent=2) + '\n')
    adapter.close()
    print('MINERU_READY_PARITY', result['all_match'], result['crops'], flush=True)
    if not result['all_match']:
        raise RuntimeError('Standalone/hybrid crop token mismatch; inspect parity.json')


if __name__ == '__main__':
    main()
