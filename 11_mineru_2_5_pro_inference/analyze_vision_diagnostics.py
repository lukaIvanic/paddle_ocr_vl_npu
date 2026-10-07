#!/usr/bin/env python3
"""Chip-independent vision/projection profile accounting, with explicit waits.

Kernel duration sums and Wait Time are separate, non-additive counters. Observed
stream gaps are not a causal attribution to host submission. No kernel-count or
chip-name gate is used. Unknown operations stay visible in remaining_by_type.
"""
import argparse
import collections
import csv
import json
from pathlib import Path
import re


def number(value):
    try:
        return float(str(value).strip())
    except (ValueError, TypeError):
        return None


def bucket(row):
    kind = row.get('Type', '')
    key = re.sub('[^a-z0-9]', '', kind.lower())
    if 'attention' in key:
        return 'attention'
    if any(s in key for s in ('matmul', 'gemm')):
        return 'matmul'
    if any(s in key for s in ('transdata', 'formatcast', 'layoutconvert')):
        return 'format_conversion'
    if 'cast' in key and row.get('Input Formats') and row.get('Output Formats') and row['Input Formats'] != row['Output Formats']:
        return 'format_conversion'
    return 'remaining'


def aggregate(rows, forwards):
    duration = [number(r.get('Duration(us)')) for r in rows]
    waits = [number(r.get('Wait Time(us)')) for r in rows]
    def histogram(column):
        return dict(collections.Counter(r.get(column, 'MISSING') for r in rows))
    return dict(calls=len(rows), calls_per_forward=len(rows)/forwards,
                duration_ms_per_forward=sum(v for v in duration if v is not None)/1000/forwards,
                wait_ms_per_forward=sum(v for v in waits if v is not None)/1000/forwards,
                missing_duration=sum(v is None for v in duration), missing_wait=sum(v is None for v in waits),
                block_num=histogram('Block Num'), mix_block_num=histogram('Mix Block Num'),
                accelerator_core=histogram('Accelerator Core'), hf32_eligible=histogram('HF32 Eligible'),
                input_shapes=histogram('Input Shapes'), input_formats=histogram('Input Formats'),
                output_formats=histogram('Output Formats'))


def profile(rows, forwards):
    buckets = collections.defaultdict(list)
    types = collections.defaultdict(list)
    streams = collections.defaultdict(list)
    for row in rows:
        buckets[bucket(row)].append(row)
        types[row.get('Type', 'MISSING')].append(row)
        start, duration = number(row.get('Start Time(us)')), number(row.get('Duration(us)'))
        if start is not None and duration is not None:
            key = (row.get('Device_id'), row.get('Stream ID'), row.get('Step Id'))
            streams[str(key)].append((start, start+duration))
    gaps = {}
    for key, intervals in streams.items():
        intervals.sort()
        end, gap = intervals[0][1], 0
        for begin, finish in intervals[1:]:
            gap += max(0, begin-end)
            end = max(end, finish)
        gaps[key] = gap/1000/forwards
    columns = ['Name','Type','Step Id','Device_id','Stream ID','Duration(us)','Wait Time(us)',
               'Block Num','Mix Block Num','Accelerator Core','HF32 Eligible','Input Shapes',
               'Input Formats','Output Formats','Input Data Types','Output Data Types']
    total = aggregate(rows, forwards)
    return dict(profile_forwards=forwards, total=total,
                buckets={k:aggregate(buckets[k], forwards) for k in ['attention','matmul','format_conversion','remaining']},
                by_type={k:dict(bucket=bucket(v[0]), **aggregate(v, forwards)) for k,v in sorted(types.items())},
                names_found={k:sorted({r.get('Type','MISSING') for r in v}) for k,v in buckets.items()},
                unclassified_types=sorted({r.get('Type','MISSING') for r in buckets['remaining']}),
                conversion_directions=dict(collections.Counter(r.get('Input Formats','MISSING')+' -> '+r.get('Output Formats','MISSING') for r in buckets['format_conversion'])),
                kernel_calls=[{k:r.get(k) for k in columns} for r in rows if bucket(r) in ['attention','matmul']],
                observed_stream_gap_ms_per_forward=gaps,
                wait_note='CSV Wait Time sum is separate from kernel duration; not proven host delay. Do not add it to duration as elapsed time.',
                gap_note='Per device/stream/profile-step observed gaps; include profiler/synchronization effects. Streams may overlap; do not sum as host time.')


def analyze(root, default_forwards):
    lanes = []
    for path in sorted(root.rglob('result.json')):
        if any(x in path.parts for x in ['vision_cache', 'cache']):
            continue
        data = json.loads(path.read_text())
        if 'variant' not in data and data.get('kind') != 'matmul_calibration':
            continue
        files = list(path.parent.rglob('kernel_details.csv'))
        record = dict(lane=str(path.parent.relative_to(root)), device=data.get('device','unreported'),
                      kind=data.get('kind','full_vision_stack'), variant=data.get('variant'), route=data.get('route'),
                      execution=data.get('execution'), status=data.get('status','incomplete'),
                      config=data.get('vision_config'), timing=data.get('timing'),
                      wall_real_tok_s=data.get('wall_real_tok_s'), drift=data.get('full_encoder_parity'),
                      calibration_shape=data.get('shape'), calibration_tflops=data.get('achieved_tflops'), warnings=[])
        if len(files) != 1:
            record['warnings'].append(f'expected one kernel CSV, found {len(files)}; no aggregate invented')
        else:
            forwards = data.get('profile_forwards', default_forwards)
            if forwards <= 0:
                raise ValueError('profile_forwards must be positive')
            rows = list(csv.DictReader(files[0].open()))
            record['profile'] = profile(rows, forwards)
            record['profile_csv'] = str(files[0].relative_to(root))
            if data.get('kind') == 'matmul_calibration':
                m,k,n = (data['shape'][x] for x in ['M','K','N'])
                for typ, group in record['profile']['by_type'].items():
                    if group['bucket'] == 'matmul':
                        us = group['duration_ms_per_forward']*1000/max(group['calls_per_forward'], 1e-30)
                        group['achieved_tflops'] = 2*m*k*n/(us*1e6) if us > 0 and group['calls_per_forward']==1 and len(record['profile']['names_found']['matmul'])==1 else None
        lanes.append(record)
    return dict(schema=1, profile_forward_default=default_forwards, lanes=lanes,
                scope='Vision stack and separately labelled synthetic matmul calibration; never substitute calibration for model performance')


def compare_old(result, old):
    checked = 0
    for ref in old['records']:
        matches = [r for r in result['lanes'] if r['variant']==ref['variant'] and r['route']==ref['route']]
        if len(matches) != 1:
            raise ValueError(f'comparison coverage mismatch: {ref["variant"]}/{ref["route"]}')
        p = matches[0]['profile']
        pairs = [(p['total']['duration_ms_per_forward'], ref['kernel_ms']),
                 (p['buckets']['attention']['duration_ms_per_forward'],ref['attention_ms']),
                 (p['buckets']['matmul']['duration_ms_per_forward'],ref['linear_ms'])]
        if any(abs(a-b)>1e-6 for a,b in pairs):
            raise ValueError(f'normalization mismatch: {matches[0]["lane"]}')
        checked += 1
    return dict(status='pass', compared_lanes=checked, tolerance_ms=1e-6,
                fields=['kernel_ms','attention_ms','linear_ms'])


def report(result):
    lines=['# Vision diagnostic accounting','',
           'Kernel durations normalized by the recorded profile-forward count (default 3). Wait and observed gaps are separate, non-additive measurements.','',
           '| Chip | Lane | Mode | Wall ms | Event ms | Useful tok/s | Status |',
           '|---|---|---|---:|---:|---:|---|']
    def fmt(v): return 'NA' if v is None else f'{v:.3f}'
    for r in result['lanes']:
        t=r.get('timing') or {}
        lines.append(f"| {r['device']} | {r['lane']} | {r['execution']} | {fmt(t.get('wall_ms',{}).get('mean'))} | {fmt(t.get('device_ms',{}).get('mean'))} | {fmt(r.get('wall_real_tok_s'))} | {r['status']} |")
    lines += ['', '| Chip | Lane | Bucket/type | Calls/forward | Kernel ms/forward | Wait ms/forward | Block Num histogram | Mix Block Num | Accelerator Core |', '|---|---|---|---:|---:|---:|---|---|---|']
    for r in result['lanes']:
        p=r.get('profile')
        if not p:continue
        for name,g in p['by_type'].items():
            lines.append(f"| {r['device']} | {r['lane']} | {g['bucket']}: {name} | {g['calls_per_forward']:.1f} | {g['duration_ms_per_forward']:.6f} | {g['wait_ms_per_forward']:.6f} | {g['block_num']} | {g['mix_block_num']} | {g['accelerator_core']} |")
        lines.append(f"| {r['device']} | {r['lane']} | **TOTAL WAIT (separate)** | — | — | {p['total']['wait_ms_per_forward']:.6f} | — | — | — |")
    lines += ['', 'Full per-call columns, shape/format/HF32 histograms, conversion directions, remaining types and per-stream gaps are retained in the companion JSON. Unclassified does not mean erroneous; it means no semantic attribution was guessed.', '']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--profile-forwards',type=int,default=3)
    p.add_argument('--compare-analysis',type=Path)
    a=p.parse_args()
    result=analyze(a.run_dir,a.profile_forwards)
    if a.compare_analysis:
        result['reference_comparison']=compare_old(result,json.loads(a.compare_analysis.read_text()))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    a.output.with_suffix('.md').write_text(report(result))
    print(json.dumps(dict(lanes=len(result['lanes']),comparison=result.get('reference_comparison'),json=str(a.output),markdown=str(a.output.with_suffix('.md')))))


if __name__=='__main__':
    main()
