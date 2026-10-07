#!/usr/bin/env python3
"""Compare complete warmed text variants, retaining numerical rejection gates."""
import argparse
import json
from pathlib import Path

from analyze_text_profile import analyze
VARIANTS = ('baseline', 'bsnd', 'rotary_bnsd', 'apply_bnsd', 'apply_bsnd',
            'swiglu', 'apply_bsnd_swiglu')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    output = dict(scope='Frozen B1/S1274 text-only forward. Clean wall timing '
                        'outside profiler; kernel sums diagnose changes.', variants={})
    identities, cards = set(), set()
    for name in (*VARIANTS, 'baseline_end'):
        lanes = ('torchair',) if name == 'baseline_end' else ('raw_eager', 'torchair')
        record = {}
        for lane in lanes:
            directory = args.run_dir/name/lane
            result_path = directory/'output/result.json'
            if not result_path.exists():
                record[lane] = {'status': 'missing'}
                continue
            result = json.loads(result_path.read_text())
            if result['status'] != 'completed':
                record[lane] = {'status': result['status'], 'error': result.get('error')}
                continue
            record[lane] = dict(status='completed', **analyze(directory),
                               adoption_eligible=result['adoption_eligible'],
                               candidate_eager_vs_frozen=result['candidate_eager_vs_frozen'],
                               vs_candidate_eager=result['vs_candidate_eager'])
            identities.add(result['frozen_inputs_sha256'])
            cards.add(result['physical_npu'])
        output['variants'][name] = record
    if len(identities) != 1 or len(cards) != 1:
        raise ValueError('Variant matrix must retain one input snapshot and physical NPU')
    output['frozen_inputs_sha256'] = next(iter(identities))
    output['physical_npu'] = next(iter(cards))
    baseline = output['variants']['baseline']['torchair']['warm_wall_ms']['mean']
    end = output['variants']['baseline_end']['torchair']
    if end['status'] == 'completed':
        output['baseline_drift_percent'] = 100*(end['warm_wall_ms']['mean']/baseline-1)
    for name, record in output['variants'].items():
        compiled = record.get('torchair', {})
        if compiled.get('status') == 'completed':
            compiled['latency_reduction_vs_initial_control_percent'] = 100*(
                1-compiled['warm_wall_ms']['mean']/baseline)
        print('TEXT_VARIANT', name, json.dumps({lane:{
            k:value.get(k) for k in ('status','warm_wall_ms','adoption_eligible',
                                    'latency_reduction_vs_initial_control_percent')}
            for lane,value in record.items()}))
    (args.run_dir/'comparison.json').write_text(json.dumps(output, indent=2)+'\n')


if __name__ == '__main__':
    main()
