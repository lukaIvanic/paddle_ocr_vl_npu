#!/usr/bin/env python3
"""Read raw kernel CSVs: preserve zeros, distinguish missing PMU data, retain calls.

PMU engine times overlap and are not additive parts of kernel wall time.
Ratios and bandwidths are never summed. No compute/memory-bound classification.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def stats(values):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return dict(count=0, mean=None, min=None, p50=None, p99=None, max=None)
    def quantile(q):
        pos = (len(vals)-1)*q
        lo = int(pos)
        return vals[lo]+(vals[min(lo+1,len(vals)-1)]-vals[lo])*(pos-lo)
    return dict(count=len(vals), mean=statistics.mean(vals), min=vals[0],
                p50=quantile(.5), p99=quantile(.99), max=vals[-1])


def is_pmu(name):
    return name.lower().startswith(('aic_', 'aiv_', 'aicore_', 'cube_', 'vec_',
        'mac_', 'mte', 'scalar_', 'fixpipe_', 'ub_', 'l0', 'l1_', 'l2_', 'main_mem_', 'icache_'))


def analyze_csv(path, forwards):
    groups = {}
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        columns = [key for key in reader.fieldnames if is_pmu(key)]
        for row in reader:
            kind = row.get('Type') or row.get('Op Type') or row.get('Task Type') or ''
            if 'attention' not in kind.lower():
                continue
            duration = number(row.get('Duration(us)'))
            if duration is None:
                duration = number(row.get('Task Duration(us)'))
            if duration is None:
                raise ValueError(f'attention row lacks duration: {path}')
            shape = row.get('Input Shapes', '')
            core = row.get('Accelerator Core') or row.get('Core Type', '')
            key = (kind, shape, core)
            call = dict(name=row.get('Name') or row.get('Task Name'), duration_us=duration,
                input_formats=row.get('Input Formats'), input_dtypes=row.get('Input Data Types'),
                pmu={key:number(row.get(key)) for key in columns})
            groups.setdefault(key, []).append(call)
    result = []
    for (kind,shape,core), calls in groups.items():
        counters = {}
        for name in columns:
            valid = [(c['pmu'][name],c['duration_us']) for c in calls
                     if c['pmu'][name] is not None and c['pmu'][name] >= 0]
            field = stats([v for v,_ in valid])
            field['missing_count'] = sum(c['pmu'][name] is None for c in calls)
            field['invalid_negative_count'] = sum(c['pmu'][name] is not None and c['pmu'][name] < 0 for c in calls)
            denominator = sum(w for _,w in valid)
            field['duration_weighted_mean'] = sum(v*w for v,w in valid)/denominator if denominator else None
            if 'time' in name.lower() and '(us)' in name.lower():
                field['engine_time_sum_us_per_forward'] = sum(v for v,_ in valid)/forwards if valid else None
            counters[name] = field
        result.append(dict(kernel=kind, input_shapes=shape, core=core, count=len(calls),
            calls_per_forward=len(calls)/forwards, elapsed_us=stats([c['duration_us'] for c in calls]),
            elapsed_ms_per_forward=sum(c['duration_us'] for c in calls)/forwards/1000,
            pmu=counters, calls=calls))
    if not result:
        raise ValueError(f'no attention rows in {path}')
    return result


def collect(root, old_forwards=3):
    output = []
    for lane in sorted(root.iterdir()):
        if not lane.is_dir() or not (lane/'result.json').exists():
            continue
        result = json.loads((lane/'result.json').read_text())
        sessions_path = lane/'metric_sessions.json'
        if sessions_path.exists():
            sessions = json.loads(sessions_path.read_text())['profiles']
        else:
            sessions = [dict(metric='pipe', status='completed', profile_forwards=old_forwards)]
        for session in sessions:
            base = lane/'metrics'/session['metric'] if sessions_path.exists() else lane
            item = dict(lane=lane.name, variant=result['variant'], route=result['route'],
                device=result['device'], commit=result['commit'], capture_sha256=result['source_capture_sha256'],
                metric=session['metric'], status=session['status'], profile_forwards=session['profile_forwards'],
                warm_timing=result.get('timing'), feature_parity=result.get('full_encoder_parity'))
            if session['status'] == 'completed':
                paths = list((base/'profile').rglob('kernel_details.csv'))
                if len(paths) != 1:
                    raise ValueError(f'expected one raw kernel CSV under {base}, got {paths}')
                item.update(csv=str(paths[0]), attention=analyze_csv(paths[0],session['profile_forwards']))
                item['has_numeric_pmu'] = any(s['count'] for k in item['attention'] for s in k['pmu'].values())
            output.append(item)
    baselines = {r['lane']:sum(k['elapsed_ms_per_forward'] for k in r.get('attention',[]))
                 for r in output if r['metric']=='pipe' and r.get('attention')}
    for row in output:
        if row.get('attention'):
            baseline = baselines.get(row['lane'])
            row['elapsed_ratio_to_pipe'] = sum(k['elapsed_ms_per_forward'] for k in row['attention'])/baseline if baseline else None
            row['has_invalid_negative_pmu'] = any(s['invalid_negative_count'] for k in row['attention'] for s in k['pmu'].values())
            row['duration_perturbation_warning'] = bool(baseline and row['elapsed_ratio_to_pipe'] > 1.25)
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--old-profile-forwards', type=int, default=3)
    p.add_argument('--omit-calls', action='store_true', help='Compact summary; raw CSVs remain authoritative')
    args = p.parse_args()
    if args.old_profile_forwards <= 0:
        p.error('forward count must be positive')
    rows = collect(args.root,args.old_profile_forwards)
    for row in rows:
        for kernel in row.get('attention',[]):
            print(row['lane'],row['metric'],kernel['kernel'],
                f"calls/fwd={kernel['calls_per_forward']:g} ms/fwd={kernel['elapsed_ms_per_forward']:.4f}")
            if args.omit_calls:
                del kernel['calls']
    report = dict(notes=['PMU pipe times overlap: never sum as elapsed components.',
        'Zero is retained; missing/unsupported is null, not zero. Negative PMU values remain in raw calls but are excluded from statistics and flagged.',
        'Counter units and chip aggregation semantics remain those of the raw CSV headers.',
        'No bottleneck classification is made from utilization alone.'], rows=rows)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print('ATTENTION_PIPES_JSON',args.out)


if __name__ == '__main__':
    main()
