#!/usr/bin/env python3
"""Report complete live page runs and their real vision length distributions."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from vision_timing_report import group_stats


def load(root):
    output=root/'output'
    s=json.loads((output/'run_summary_shard_00.json').read_text())
    assert (s['completed'],s['failed'],s['skipped']) == (1651,0,0)
    assert s['streaming']['layout_calls'] == 1651
    assert s['layout_backend'] == 'pp-doclayout-v3'
    assert s['processor_max_pixels'] == 602112
    samples=[json.loads(line) for line in (output/s['vision_timing']['raw_samples_file']).read_text().splitlines()]
    v=s['local_compiled_vision']
    assert sum(r['real_tokens'] for r in samples) == v['real_tokens']
    assert sum(r['physical_tokens'] for r in samples) == v['physical_tokens']
    assert abs(sum(r['device_s'] for r in samples)-s['vision_timing']['all']['device_s']) < 1e-6
    geometry={}
    for file in (output/'layout_regions').glob('*.json'):
        row=json.loads(file.read_text())
        # Timing metadata is deliberately excluded from the input comparison.
        geometry[file.name]=row.get('blocks',row.get('regions'))
    if len(geometry) != 1651 or any(value is None for value in geometry.values()):
        raise ValueError('unrecognized layout metadata; cannot silently claim matching crops')
    prompts={}
    outputs={}
    for line in (output/'generation_trace.jsonl').read_text().splitlines():
        row=json.loads(line)
        prompts[row['request_id']]=dict(ids=row['prompt_token_ids'],image_sha256=row['image_sha256'],max_new_tokens=row['max_new_tokens'])
        outputs[row['request_id']]=row['generated_token_ids']
    return s,samples,geometry,prompts,outputs


def length_rows(samples):
    groups=defaultdict(list)
    for row in samples:
        useful=row['real_tokens']
        high=next((v for v in [384,512,768,1024,1536,2048,3072] if useful <= v),None)
        if high is None:raise ValueError('crop input exceeds agreed cap')
        kind='packed' if row['members'] > 1 else 'single'
        groups[(high,kind)].append(row)
    return {f'{upper}:{kind}':dict(group_stats(rows),
        group_histogram=dict(Counter(json.dumps([r['physical_tokens'],sorted(r['member_lengths'])]) for r in rows)),
        actual_min=min(r['real_tokens'] for r in rows),actual_max=max(r['real_tokens'] for r in rows))
        for (upper,kind),rows in sorted(groups.items())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original',type=Path,required=True)
    p.add_argument('--approximate',type=Path)
    p.add_argument('--approximate-prewarm-audit',type=Path)
    p.add_argument('--chip',choices=['910B','310P'],required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    original=load(a.original)
    rows=[('original',original)]
    comparison=None
    if a.approximate:
        approximate=load(a.approximate)
        audit=[json.loads(line) for line in (a.approximate/'precision.jsonl').read_text().splitlines()]
        assert audit[-1]['event'] == 'completed' and audit[-1]['mode'] == 4
        assert a.chip != '310P' or audit[-1]['supported_310p']
        assert a.approximate_prewarm_audit, 'provide the mode4 prewarm audit, not only a filename labelled approximate'
        warm=[json.loads(line) for line in a.approximate_prewarm_audit.read_text().splitlines()]
        assert warm[-1]['event'] == 'completed' and warm[-1]['mode'] == 4
        assert {r['bucket'] for r in warm if r['event'] == 'prewarm_crop_complete'} == {384,512,768,1024,1536,2048,3072}
        assert all(r['inner_precise'] == 4 for r in warm if r['event'] == 'vision_ge_promptfa')
        rows.append(('approximate',approximate))
        comparison=dict(model_hashes_match=original[0]['model_hashes'] == approximate[0]['model_hashes'],
            layout_model_hashes_match=original[0]['layout_model_hashes'] == approximate[0]['layout_model_hashes'],
            crop_geometry_hashes_match=original[2] == approximate[2],
            prompt_ids_and_image_hashes_match=original[3] == approximate[3],
            changed_output_requests=sum(original[4].get(k) != approximate[4].get(k) for k in original[4].keys()|approximate[4].keys()),
            e2e_pg_s_gain_pct=100*(original[0]['pipeline_wall_s']/approximate[0]['pipeline_wall_s']-1))
        print('PAIR_INPUT_AND_OUTPUT_COMPARISON '+json.dumps(comparison))
    result=dict(chip=a.chip,comparison=comparison,lanes={})
    print('| Chip | Precision | Pages | Wall s | Setup s | pg/s | Output tokens incl EOS | Useful vision tokens | Padding positions | Padding % | Vision event tok/s |')
    print('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for label,(s,samples,geometry,prompts,generated) in rows:
        v=s['local_compiled_vision'];total=s['vision_timing']['all']
        result['lanes'][label]=dict(wall_s=s['pipeline_wall_s'],pg_s=1651/s['pipeline_wall_s'],
            setup_s=s['setup_s'],output_tokens_including_eos=sum(map(len,generated.values())),vision=total,lengths=length_rows(samples),
            precision_audit_checked=label == 'approximate')
        print(f"| {a.chip} | {label} | 1651 | {s['pipeline_wall_s']:.3f} | {s['setup_s']:.3f} | "
            f"{1651/s['pipeline_wall_s']:.6f} | {sum(map(len,generated.values()))} | {v['real_tokens']} | {v['physical_tokens']-v['real_tokens']} | "
            f"{100*(1-v['real_tokens']/v['physical_tokens']):.3f} | {total['real_tok_s']:.1f} |")
    print('\nVision event regions include launch gaps and initial cache loads, as in production. Not isolated kernels.')
    print('| Chip | Precision | Useful length bin / kind | Actual useful range | Groups / crops | Vision time share % | Mean / p50 / p99 ms | Useful tok/s |')
    print('|---|---|---|---|---|---:|---|---:|')
    for label,lane in result['lanes'].items():
        for key,s in lane['lengths'].items():
            lat=s['latency_ms']
            print(f"| {a.chip} | {label} | {key} | {s['actual_min']}–{s['actual_max']} | {s['calls']} / {s['members']} | "
                f"{100*s['device_s']/lane['vision']['device_s']:.2f} | {lat['mean']:.3f} / {lat['p50']:.3f} / {lat['p99']:.3f} | {s['real_tok_s']:.1f} |")
    if comparison:
        print('\n| Chip | Useful length bin / kind | Vision useful tok/s gain | Group length distributions match |')
        print('|---|---|---:|---|')
        base=result['lanes']['original']['lengths']; approx=result['lanes']['approximate']['lengths']
        result['vision_gains']={}
        for key in sorted(base.keys() & approx.keys()):
            gain=100*(approx[key]['real_tok_s']/base[key]['real_tok_s']-1)
            matched=base[key]['group_histogram'] == approx[key]['group_histogram']
            result['vision_gains'][key]=dict(useful_tok_s_gain_pct=gain,group_distributions_match=matched)
            print(f'| {a.chip} | {key} | {gain:+.2f}% | {matched} |')
        print('Throughput is shown even if outputs differ. Check precision sidecar and input-match flags before interpreting gains.')
    a.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
