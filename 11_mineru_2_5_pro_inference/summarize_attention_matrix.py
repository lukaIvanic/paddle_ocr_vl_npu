#!/usr/bin/env python3
"""Read-only, standard-library summary of one or more attention matrices."""
import argparse
import json
from pathlib import Path


def summarize(roots):
    rows = []
    for root in roots:
        for path in sorted(root.glob('*/result.json')):
            data = json.loads(path.read_text())
            if 'variant' not in data:
                continue
            row = dict(path=str(path), device=data['device'], route=data['route'],
                variant=data['variant'], status=data.get('status', 'incomplete'),
                execution=data['execution'], capture_sha256=data['source_capture_sha256'])
            if 'timing' in data:
                row.update(device_ms=data['timing']['device_ms'], wall_ms=data['timing']['wall_ms'],
                    real_tok_s=data['real_tok_s'], physical_tok_s=data['physical_tok_s'],
                    parity=data['full_encoder_parity'])
            profile = path.with_name('parsed_profile_summary.json')
            if profile.exists():
                captures = json.loads(profile.read_text())['runs']
                if len(captures) != 1:
                    raise ValueError(f'expected exactly one capture: {profile}')
                kernels = captures[0]['kernel_details']
                row['profile_kernel_ms_per_forward'] = kernels['total_duration_us'] / 3000
                row['attention_kernel_types'] = [dict(name=k['name'],
                    calls_per_forward=k['count']/3, ms_per_forward=k['duration_us']/3000)
                    for k in kernels['top_kernel_types'] if 'attention' in k['name'].lower()]
            rows.append(row)
    print('All times are warm. Device-event intervals may include host launch gaps, especially raw_eager.')
    print('Profiles contain THREE full forwards: duration_us / 3 / 1000; never substitute aicore_time_us.')
    print('Layer/feature parity is not OCR accuracy. No default change is authorized by this report.')
    print('| Route | Variant | Execution | Mean/p50/p99 ms | Wall mean ms | Useful tok/s | Relative L2 |')
    print('|---|---|---|---|---|---|---|')
    for row in rows:
        if 'device_ms' not in row:
            print(f"| {row['route']} | {row['variant']} | {row['status']} | — | — | — | — |")
            continue
        d = row['device_ms']
        print(f"| {row['route']} | {row['variant']} | {row['execution']} | {d['mean']:.3f}/{d['p50']:.3f}/{d['p99']:.3f} | {row['wall_ms']['mean']:.3f} | {row['real_tok_s']:.0f} | {row['parity']['relative_l2']:.6g} |")
    print('ATTENTION_MATRIX_JSON ' + json.dumps(rows))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('roots', type=Path, nargs='+')
    summarize(p.parse_args().roots)
