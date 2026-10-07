#!/usr/bin/env python3
"""Analyze paired isolated text profiles; no torch dependency or model execution.

Reads the original kernel CSVs and verifies the window contains the expected
text attention/projection calls and no vision, merger or image-insertion kernels.
Compiled semantic attribution uses fused names, dtypes and owned source shapes.
Shared residual-add/FP32-cast kernels are kept separate from RMSNorm work.
"""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import re

from forward_profile_analysis import distribution


def compiled_category(g):
    kind, names = g['type'], g['normalized_names']
    shape = g['input_shapes'].split(';')[0]
    rank = len(shape.split(','))
    if kind.startswith('MatMul'):
        return 'matrix_projections'
    if 'FlashAttention' in kind:
        return 'attention'
    if (kind == 'Square' or names == ['AddRsqrt'] or names == ['MulCast']
        or kind == 'Mul' and rank == 1
        or kind == 'Cast' and g['input_dtypes'] == 'FLOAT16' and g['output_dtypes'] == 'FLOAT'):
        return 'rmsnorm_excluding_shared_residual_cast'
    if (kind in ('Neg', 'ConcatV2D') or names == ['MulMulAdd']
        or kind == 'SplitVD' and rank == 4):
        return 'rotary'
    if kind in ('Transpose', 'SplitVD'):
        return 'layout_and_projection_split'
    if names == ['SwishMul']:
        return 'silu_gate_multiply'
    if kind == 'Add':
        return 'residual_deepstack_add_shared_norm_cast'
    return 'other'


def analyze(directory):
    result = json.loads((directory/'output/result.json').read_text())
    if result['status'] != 'completed':
        raise ValueError('Cannot analyze an incomplete text run')
    summary = dict(physical_npu=result['physical_npu'], execution=result['execution'],
                   frozen_inputs_sha256=result['frozen_inputs_sha256'], text_tokens=result['text_tokens'],
                   text_input_shapes=result['text_input_shapes'],
                   warm_wall_ms=distribution(result['before_profile']['wall_samples_ms'] +
                                             result['after_profile']['wall_samples_ms']),
                   before_wall_ms=result['before_profile']['wall_ms'],
                   after_wall_ms=result['after_profile']['wall_ms'],
                   warm_device_interval_ms=distribution(result['before_profile']['device_samples_ms'] +
                                                        result['after_profile']['device_samples_ms']),
                   vs_frozen_eager=result['vs_frozen_eager'], profiles={})
    for metric in result['profiles']:
        destination = directory/'output/profiles'/metric
        paths = list((destination/'raw').glob('**/ASCEND_PROFILER_OUTPUT/kernel_details.csv'))
        if len(paths) != 1:
            raise ValueError(f'Expected exactly one raw kernel CSV for {directory}/{metric}')
        with paths[0].open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        steps = result['profile_steps']
        groups = defaultdict(lambda: dict(count=0, duration_us=0., names=set(), examples=[]))
        for r in rows:
            key = (r['Type'], r['Input Shapes'].strip('"'), r['Input Data Types'],
                   r['Output Shapes'].strip('"'), r['Output Data Types'])
            g = groups[key]
            g['count'] += 1
            g['duration_us'] += float(r['Duration(us)'])
            g['names'].add(re.sub(r'_?\d+', '', r['Name']))
            if len(g['examples']) < 2 and r['Name'] not in g['examples']:
                g['examples'].append(r['Name'])
        values = [dict(type=k[0], input_shapes=k[1], input_dtypes=k[2], output_shapes=k[3],
                       output_dtypes=k[4], count_per_forward=v['count']/steps,
                       ms_per_forward=v['duration_us']/steps/1000,
                       normalized_names=sorted(v['names']), name_examples=v['examples'])
                  for k,v in sorted(groups.items(), key=lambda p: -p[1]['duration_us'])]
        counts = defaultdict(float)
        for g in values:
            counts[g['type']] += g['count_per_forward']
        layers = result['text_layers']
        if counts['PromptFlashAttention'] != layers:
            raise ValueError('Expected one text attention call per layer')
        if sum(v for k,v in counts.items() if k.startswith('MatMul')) != 4*layers:
            raise ValueError('Expected four fused text projections per layer')
        forbidden = {'MaskedScatter', 'LayerNormV3', 'Gelu', 'GeluV2', 'Unpack', 'Conv3D'}
        if forbidden.intersection(counts):
            raise ValueError('Non-text preparation/vision kernel in isolated capture')
        length = result['text_tokens']
        # Both Q and KV must attend over the frozen text length, rather than a vision grid.
        attention = [g for g in values if g['type'] == 'PromptFlashAttention']
        if any(any(f',{length},' not in s for s in g['input_shapes'].split(';')[:3]) for g in attention):
            raise ValueError('Attention shape is inconsistent with frozen text inputs')
        total = sum(g['ms_per_forward'] for g in values)
        captured = result['profiles'][metric]['kernel_accounting']
        if len(rows) != captured['kernel_rows'] or abs(total-captured['kernel_duration_sum_ms_per_forward']) > 1e-8:
            raise ValueError('Export does not conserve original kernel accounting')
        profile = dict(raw_csv=str(paths[0]), isolated_text_checks_passed=True,
                       kernel_count_per_forward=len(rows)/steps, kernel_sum_ms_per_forward=total,
                       groups=values)
        if result['execution'] == 'torchair':
            buckets = defaultdict(lambda: dict(ms_per_forward=0., kernels_per_forward=0.))
            for g in values:
                b = buckets[compiled_category(g)]
                b['ms_per_forward'] += g['ms_per_forward']
                b['kernels_per_forward'] += g['count_per_forward']
            profile['semantic_categories'] = dict(sorted(buckets.items(), key=lambda p:-p[1]['ms_per_forward']))
            if counts['Square'] != 4*layers+1:
                raise ValueError('Expected two hidden plus Q/K RMSNorms per layer and a final norm')
        (destination/'kernel_shape_groups.json').write_text(json.dumps(profile, indent=2)+'\n')
        summary['profiles'][metric] = {k:v for k,v in profile.items() if k != 'groups'}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    modes = {mode:analyze(args.run_dir/mode) for mode in ('raw_eager', 'torchair')}
    eager, compiled = modes['raw_eager'], modes['torchair']
    if eager['frozen_inputs_sha256'] != compiled['frozen_inputs_sha256']:
        raise ValueError('Paired profiles used different frozen inputs')
    a,b = eager['warm_wall_ms']['mean'], compiled['warm_wall_ms']['mean']
    output = dict(scope='36 text transformer layers, DeepStack adds and final RMSNorm; '
                        'kernel sums are distinct from clean warmed wall latency.',
                  **modes, identical_frozen_inputs=True, speedup=a/b,
                  latency_reduction_percent=100*(1-b/a))
    (args.run_dir/'comparison.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
