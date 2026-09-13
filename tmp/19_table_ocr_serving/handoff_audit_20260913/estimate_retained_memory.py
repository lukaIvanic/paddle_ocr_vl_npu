"""CPU-only estimate of lifetime history, using saved native generated IDs.

No inference, model import, tensor allocation, or production edits. Reconstruct
the four actual dataclasses after their NPU state has been released. This is
Python allocation accounting, not a measurement of the live server's RSS/HBM.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, field, fields
import gc
import json
from pathlib import Path
import statistics
import sys
import tracemalloc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--copies', type=int, default=3)
    args = parser.parse_args()
    source = args.repo / '19_table_ocr_serving/p02_serving_runtime.py'
    names = {'DecodeRequest', 'DecodeCompletion', 'CpuTiming', 'PrefillTiming'}
    tree = ast.parse(source.read_text())
    definitions = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
    assert len(definitions) == 4
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[
        ast.alias(name='annotations')], level=0), *definitions], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source), 'exec'), globals())
    records = [json.loads(line) for line in args.results.open()]
    responses = [r['service_result']['response'] for r in records if r['status'] == 'ok']
    assert len(responses) == len(records), 'Do not silently omit failed requests'

    # Force fresh scalar values as real requests produce them. Literal keys and
    # implementation names are shared, as in production. The finalized token
    # list has exact-size allocation (the runtime copies the growing slot list).
    def new_int(value):
        return int(str(value))

    def new_float(value):
        return float(str(value))

    def make_completion(response, sequence):
        timing = response['timing_s']
        cpu = CpuTiming(**{f.name: new_float(timing[f.name]) for f in fields(CpuTiming)})
        prefill = PrefillTiming(**{f.name: new_float(timing[f.name]) for f in fields(PrefillTiming)})
        vision = response['vision']
        text = response['text_prefill']
        request_id = f"{response['request_id']}:{sequence:032x}"
        ready = DecodeRequest(
            request_id=request_id, prompt='Table Recognition:',
            crop_size=tuple(new_int(v) for v in response['crop_size']),
            skip_special_tokens=True, cache=None, cache_lease=None,
            rope_deltas=None, cache_position=None, first_token_tensor=None,
            first_token=new_int(response['token_ids'][0]),
            prompt_length=new_int(response['input_tokens']),
            projected_image_tokens=new_int(response['projected_image_tokens']),
            vision={
                'execution': 'compiled',
                'real_vision_tokens': new_int(vision['real_vision_tokens']),
                'physical_vision_tokens': vision['physical_vision_tokens'],
                'padding_vision_tokens': new_int(vision['padding_vision_tokens']),
                'useful_token_fraction': new_float(vision['useful_token_fraction']),
                'bucket': vision['bucket'],
            },
            text_prefill={
                'execution': 'compiled',
                'real_text_tokens': new_int(text['real_text_tokens']),
                'physical_text_tokens': text['physical_text_tokens'],
                'padding_text_tokens': new_int(text['padding_text_tokens']),
                'useful_token_fraction': new_float(text['useful_token_fraction']),
                'bucket': text['bucket'],
                'private_cache_slot_index': sequence % 40,
                'private_cache_generation': sequence // 40 + 1,
            },
            cpu_timing=cpu, prefill_timing=prefill,
            request_started=new_float(sequence), prefill_finished=new_float(sequence + .1),
        )
        tokens = list(map(new_int, response['token_ids']))
        return DecodeCompletion(
            ready=ready, token_ids=list(tokens), stop_reason=sys.intern(response['stop_reason']),
            slot_index=response['decode_slot_index'], slot_epoch=sequence // 8 + 1,
            admitted_at=new_float(sequence + .2), first_decode_launched_at=new_float(sequence + .3),
            completed_at=new_float(sequence + response['timing_s']['request_total']),
            iterations_launched=new_int(response['decode_calls_executed']),
        )

    # Warm dataclass shared-key dictionaries before measuring steady growth.
    for index, response in enumerate(responses[:100]):
        make_completion(response, index)
    gc.collect()
    tracemalloc.start()
    submitted_order, submitted_ids, completions = [], set(), []
    start = tracemalloc.get_traced_memory()[0]
    checkpoints = []
    for repeat in range(args.copies):
        for response in responses:
            completion = make_completion(response, len(completions))
            request_id = completion.ready.request_id
            submitted_order.append(request_id)
            submitted_ids.add(request_id)
            completions.append(completion)
        gc.collect()
        checkpoints.append({'requests': len(completions),
                            'retained_python_bytes': tracemalloc.get_traced_memory()[0] - start})
    bytes_per_request = checkpoints[-1]['retained_python_bytes'] / len(completions)
    # Array storage plus fresh non-small Python ints. IDs <=256 use CPython's
    # shared small-int objects and do not cost a fresh integer per occurrence.
    token_bytes = statistics.mean(
        sys.getsizeof(list(r['token_ids'])) + sum(sys.getsizeof(t) for t in r['token_ids'] if t > 256)
        for r in responses)
    report = dict(
        python=sys.version, source=str(source), results=str(args.results),
        method='Reconstructed released completion records; tracemalloc, not live RSS/HBM',
        source_requests=len(responses),
        mean_native_tokens=statistics.mean(len(r['token_ids']) for r in responses),
        max_native_tokens=max(len(r['token_ids']) for r in responses),
        estimated_token_list_bytes_per_request=token_bytes,
        checkpoints=checkpoints, retained_python_bytes_per_request=bytes_per_request,
        ten_decimal_GB_requests=10_000_000_000 / bytes_per_request,
        ten_decimal_GB_hours_by_completed_qps={str(q): 10_000_000_000 / bytes_per_request / q / 3600 for q in (1, 3, 6)},
        ten_GiB_hours_at_6_completed_qps=10 * 1024**3 / bytes_per_request / 6 / 3600,
        caveats=['Excludes model, current requests, images, log files and allocator/RSS overhead.',
                 'Scalar sharing is approximate; workload and Python build affect bytes per request.',
                 'This is incremental retained history, not time until total machine memory is full.'],
    )
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
