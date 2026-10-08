"""Dependency-free analysis of portable ColQwen profiles; JSON + console only.

Kernel sums are not wall time. Shape-based projection attribution and broad
operator families are hints, not exact fused-graph source maps. Missing PMU
values remain missing, including when an entire counter is unavailable.
"""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import re

from forward_profile_analysis import distribution, interval_union_us


def number(value):
    try:
        value = float(str(value).strip())
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def counter_field(name):
    name = name.lower()
    return name.startswith(('aic_', 'aiv_')) or any(x in name for x in (
        'utilization', 'bandwidth', 'mac_ratio', 'vec_ratio', 'mte', 'memory_bound', 'cube'))


def category(kind, name):
    value = (kind+' '+name).lower()
    if 'matmul' in value or 'gemm' in value:
        return 'matrix_multiplication'
    if 'flashattention' in value:
        return 'attention'
    if any(x in value for x in ('transpose', 'transdata', 'stridedslice', 'split', 'unpack', 'tile', 'broadcast', 'pad')):
        return 'layout_slice_repeat_padding'
    if any(x in value for x in ('scatter', 'indexput', 'indexbytensor', 'nonzero', 'gather', 'embedding')):
        return 'indexing_insertion_lookup'
    if any(x in value for x in ('norm', 'rsqrt', 'reducesum', 'reducemean', 'square')):
        return 'normalization_or_reduction_candidate'
    if any(x in value for x in ('swish', 'swiglu', 'silu', 'gelu')):
        return 'activation_gating'
    if 'rotary' in value:
        return 'rotary'
    if 'cast' in value:
        return 'cast_or_fused_cast'
    return 'other_elementwise_or_unclassified'


def projection_roles(shapes, outputs, model_map):
    def dims(s):
        return [int(v) for v in re.findall(r'\d+', s)]
    parts = shapes.split(';')
    if len(parts) < 2:
        return []
    a, b, out = dims(parts[0]), dims(parts[1]), dims(outputs.split(';')[0])
    if len(a) != 2 or len(b) != 2 or not out:
        return []
    return sorted({r['role'] for r in model_map
                   if a[-1] == r['in_features'] and out[-1] == r['out_features']
                   and b in ([r['in_features'],r['out_features']], [r['out_features'],r['in_features']])})


def analyze_csv(path, steps, model_map):
    if steps < 1:
        raise ValueError('Positive capture step count required')
    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        rows = list(reader)
    duration_key = next((k for k in ('Duration(us)','Task Duration(us)') if k in fields), None)
    if not rows or not duration_key:
        raise ValueError(f'Missing kernels/duration column: {path}; fields={fields}')
    counters = [k for k in fields if counter_field(k)]
    groups, families, intervals = {}, defaultdict(float), []
    valid_counters = set()
    for r in rows:
        duration = number(r.get(duration_key))
        if duration is None or duration < 0:
            raise ValueError(f'Invalid duration in {path}: {r.get(duration_key)!r}')
        kind = r.get('Type') or r.get('Op Type') or 'unknown'
        if kind.strip().lower() in ('n/a','na','none','unknown'):
            kind = 'unknown'
        name = r.get('Name') or r.get('Op Name') or ''
        shapes = r.get('Input Shapes','').strip('"')
        outputs = r.get('Output Shapes','').strip('"')
        # Level0 may omit both types and shapes. Never collapse all unnamed
        # types into one bucket and label that bucket from its first sample.
        discriminator = name if kind=='unknown' else ''
        key = (kind,shapes,r.get('Input Data Types',''),outputs,r.get('Output Data Types',''),discriminator)
        g = groups.setdefault(key,dict(count=0,us=0.,names=[],pmu={}))
        g['count'] += 1
        g['us'] += duration
        if name not in g['names'] and len(g['names']) < 3:
            g['names'].append(name)
        for c in counters:
            value = number(r.get(c))
            if value is not None:
                valid_counters.add(c)
                total,weight,count = g['pmu'].get(c,(0.,0.,0))
                g['pmu'][c] = (total+value*duration,weight+duration,count+1)
        start = number(r.get('Start Time(us)'))
        if start is not None and duration:
            intervals.append((start,start+duration))
    total = sum(g['us'] for g in groups.values())
    if total <= 0:
        raise ValueError('Kernel durations sum to zero')
    result = []
    for k,g in sorted(groups.items(),key=lambda kv:-kv[1]['us']):
        family = category(k[0], ' '.join(g['names']))
        families[family] += g['us']/steps/1000
        result.append(dict(type=k[0],input_shapes=k[1],input_dtypes=k[2],output_shapes=k[3],
            output_dtypes=k[4],names=g['names'],calls_per_forward=g['count']/steps,
            ms_per_forward=g['us']/steps/1000,kernel_sum_percent=100*g['us']/total,
            family_hint=family,projection_role_candidates=projection_roles(k[1],k[3],model_map)
            if family=='matrix_multiplication' else [],
            counters={c:dict(duration_weighted_mean=t/w if w else None,valid_rows=n)
                      for c,(t,w,n) in g['pmu'].items()}))
    envelope = max(b for _,b in intervals)-min(a for a,_ in intervals) if intervals else None
    union = interval_union_us(intervals) if intervals else None
    return dict(raw_csv=str(path),kernel_rows=len(rows),steps=steps,
        unknown_type_rows=sum(g['count'] for k,g in groups.items() if k[0]=='unknown'),
        metadata_note='Unknown types are kept separate by exact kernel name; absent shapes prevent projection-role matching.',
        kernel_sum_ms_per_forward=total/steps/1000,
        interval_envelope_ms=envelope/1000 if envelope is not None else None,
        interval_union_ms=union/1000 if union is not None else None,
        uncovered_envelope_ms=(envelope-union)/1000 if envelope is not None else None,
        interval_note='All captured steps together. Gaps include inter-step boundaries; not pure CPU overhead.',
        counter_columns_present=counters,counter_columns_with_values=sorted(valid_counters),
        counter_columns_without_values=sorted(set(counters)-valid_counters),
        missing_counter_note='Missing/NA is not zero. Counter means preserve original column units; no automatic roofline conclusion.',
        family_hints_ms=dict(sorted(families.items(),key=lambda kv:-kv[1])),groups=result)


def analyze_run(root):
    r = json.loads((root/'result.json').read_text())
    if r['status'] != 'completed':
        raise ValueError(f'Capture incomplete: {root}: {r["status"]}')
    out = dict(device=r['device'],execution=r['execution'],page=r['page'],
               scopes={},scope_definitions=r['scope_definitions'],source=r['source'],
               same_implementation_diagnostic=r['compiled_vs_eager'],model_map=r['model_map'],
               input_contract=r['input_contract'])
    out['text_options']=r.get('text_options',r.get('options',{}))
    out['weight_formats']=r.get('weight_formats')
    out['nz_vs_native']=r.get('nz_vs_native')
    for name,scope in r['scopes'].items():
        before,after = scope['before'],scope['after']
        clean = distribution(before['wall_samples_ms']+after['wall_samples_ms'])
        item = dict(clean_wall_ms=clean,clean_device_ms=distribution(
            before['device_samples_ms']+after['device_samples_ms']),
            before_mean_ms=before['wall_ms']['mean'],after_mean_ms=after['wall_ms']['mean'],
            before_after_change_percent=100*(after['wall_ms']['mean']/before['wall_ms']['mean']-1),profiles={})
        maps = r['model_map'] if name=='full' else [m for m in r['model_map'] if m['role'].startswith(name+'.')]
        for metric,record in scope['profiles'].items():
            raw = root/record['directory']
            paths = list(raw.glob('**/ASCEND_PROFILER_OUTPUT/kernel_details.csv'))
            if len(paths) != 1:
                raise ValueError(f'Expected one kernel_details.csv in {raw}, found {len(paths)}; retain raw output')
            profile = analyze_csv(paths[0],record['steps'],maps)
            profile.update(replay=record['replay'],profiled_wall_ms=record['profiled_wall_ms'],
                profiled_to_clean_wall_ratio=record['profiled_wall_ms']['mean']/clean['mean'])
            item['profiles'][metric] = profile
        out['scopes'][name] = item
    return out


def print_summary(lanes):
    for lane,r in lanes.items():
        print(f'PROFILE_RUN {lane} device={r["device"]} page={r["page"]["id"]}')
        print('All timings below are ms/forward. Kernel percentages use summed kernel duration, not wall time.')
        for scope,s in r['scopes'].items():
            print(f'CLEAN {scope}: mean={s["clean_wall_ms"]["mean"]:.3f} '
                  f'p50={s["clean_wall_ms"]["p50"]:.3f} '
                  f'before={s["before_mean_ms"]:.3f} after={s["after_mean_ms"]:.3f}')
            for metric,p in s['profiles'].items():
                print(f'KERNELS {scope}/{metric}: sum={p["kernel_sum_ms_per_forward"]:.3f} '
                      f'profile/clean-wall={p["profiled_to_clean_wall_ratio"]:.2f} '
                      f'replay_exact={p["replay"]["exact"]}')
                print('FAMILY_HINTS '+json.dumps(p['family_hints_ms']))
                print('COUNTERS_AVAILABLE '+json.dumps(p['counter_columns_with_values']))
                print('COUNTERS_UNAVAILABLE '+json.dumps(p['counter_columns_without_values']))
                if p['unknown_type_rows']:
                    print(f'METADATA_LIMITATION {p["unknown_type_rows"]}/{p["kernel_rows"]} rows lack operator types; '
                          'grouped by kernel name, not attributed to model roles without shapes.')
                # Keep console output actionable; summary.json retains every
                # group, counter and sample count without truncation.
                topn = (10 if lane=='torchair' else 5) if metric!='memory' else 4
                for g in p['groups'][:topn]:
                    print(f'  TOP {g["ms_per_forward"]:.3f}ms {g["kernel_sum_percent"]:.2f}% '
                          f'{g["calls_per_forward"]:g}x {g["type"]} '
                          f'in={g["input_shapes"]} out={g["output_shapes"]} '
                          f'roles={g["projection_role_candidates"]} names={g["names"]}')
                    priority=('aic_mac_ratio','aic_vec_ratio','aiv_vec_ratio','aic_mte2_ratio',
                              'aiv_mte2_ratio','aiv_mte3_ratio','cube_utilization(%)')
                    keys=[k for k in priority if k in g['counters']]
                    bandwidth=[k for k in g['counters'] if 'bandwidth' in k.lower() or '_bw' in k.lower()]
                    bandwidth.sort(key=lambda k:('main_mem' not in k,'l2' not in k,k))
                    keys += bandwidth
                    keys += [k for k in g['counters'] if 'memory' in k.lower() and 'ratio' in k.lower()]
                    values={k:round(g['counters'][k]['duration_weighted_mean'],5)
                            for k in keys[:10] if g['counters'][k]['duration_weighted_mean'] is not None}
                    if values:
                        print('    PMU_MEANS '+json.dumps(values))
                suspects = [g for g in p['groups'] if any(x in (g['type']+' '.join(g['names'])).lower()
                    for x in ('transpose','stridedslice','transdata','scatter','indexput','nonzero','rms',
                              'rsqrt','square','reducemean','swiglu','swish','rotary'))]
                if metric!='memory':
                    for g in suspects[:12]:
                        print(f'TARGETED_KERNEL {g["type"]} {g["ms_per_forward"]:.3f}ms '
                              f'{g["calls_per_forward"]:g}x in={g["input_shapes"]} names={g["names"]}')
                print('FULL_DETAILS summary.json includes all groups/counters and projection source locations.')
    if {'torchair','raw_eager'} <= lanes.keys():
        for key in ('vision_sha256','text_sha256'):
            if lanes['torchair']['input_contract'][key] != lanes['raw_eager']['input_contract'][key]:
                raise ValueError(f'Eager/compiled frozen input mismatch: {key}; do not compare timings as identical-input runs')
        for scope in lanes['torchair']['scopes']:
            c,e = (lanes[k]['scopes'][scope]['clean_wall_ms']['mean'] for k in ('torchair','raw_eager'))
            print(f'EAGER_COMPILED {scope}: eager_ms={e:.3f} compiled_ms={c:.3f} speedup={e/c:.3f}')
    print('INTERPRETATION: projection roles are shape-derived candidates; fused elementwise attribution is heuristic. '
          'Kernel duration alone does not prove compute/bandwidth limitation. Preserve raw traces for exact follow-up.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,required=True)
    args=p.parse_args()
    lanes={name:analyze_run(args.run_root/name) for name in ('torchair','raw_eager')
           if (args.run_root/name/'result.json').exists()}
    if not lanes:
        raise ValueError('No capture results found')
    print_summary(lanes)
    (args.run_root/'summary.json').write_text(json.dumps(lanes,indent=2)+'\n')


if __name__=='__main__':
    main()
