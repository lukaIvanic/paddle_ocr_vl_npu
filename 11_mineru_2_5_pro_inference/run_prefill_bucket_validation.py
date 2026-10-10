"""Matched regression/smoke/full runs and the pull-only 310P prefill pair."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

from compare_generation_traces import compare
from prefill_buckets_config import PRODUCTION_PREFILL_OPTIONS
from run_host_overlap_validation import cache_artifacts, exact
from vision_diagnostic_runner import run_lane, write_new

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def route_tables(output):
    for name in ('vision', 'text_prefill'):
        groups = defaultdict(list)
        for line in (output / f'{name}_timing_shard_00.jsonl').read_text().splitlines():
            row = json.loads(line)
            groups[row['route']].append(row)
        print(f'\n{name}\n| Route | Calls | Real tokens | Physical tokens | Device s | Median ms excl first | Mean ms excl first | Real tok/s | Physical tok/s |')
        print('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
        for route, rows in sorted(groups.items()):
            rows.sort(key=lambda row: row['call_index'])
            seconds = sum(row['device_s'] for row in rows)
            real = sum(row['real_tokens'] for row in rows)
            physical = sum(row['physical_tokens'] for row in rows)
            times = [1000 * row['device_s'] for row in rows[1:]]
            median = f'{statistics.median(times):.3f}' if times else 'n/a'
            mean = f'{statistics.mean(times):.3f}' if times else 'n/a'
            print(f'| {route} | {len(rows)} | {real} | {physical} | {seconds:.3f} | {median} | {mean} | {real / seconds if seconds else 0:.1f} | {physical / seconds if seconds else 0:.1f} |')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--chip', choices=('910b', '310p'), required=True)
    p.add_argument('--phase', choices=('regression', 'smoke', 'full', 'pair'), required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--parent-repo', type=Path)
    for name in ('model', 'layout-model', 'dataset-json', 'images-dir',
                 'cache-decode', 'cache-vision', 'cache-text'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    a.root = a.root.resolve()
    a.root.mkdir(parents=True, exist_ok=True)
    assert (a.phase == 'pair') == (a.chip == '310p'), '310P uses the full pair; 910B uses individual validation phases'
    assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()

    def command(label, count, stage, *, candidate=False, parent=False, warm=False):
        source_repo = a.parent_repo.resolve() if parent else REPO
        cache = {}
        for key in ('decode', 'vision', 'text'):
            target = a.root / 'cache' / label / key
            if not target.exists():
                shutil.copytree(getattr(a, 'cache_' + key).resolve(), target, symlinks=False)
            cache[key] = target
        runner = 'run_page_pipeline_vision_precision.py' if a.chip == '310p' else 'run_page_pipeline.py'
        cmd = [sys.executable, '-u', str(source_repo / HERE.name / runner)]
        if a.chip == '310p':
            cmd += ['--vision-inner-precise', '4', '--precision-audit', str(stage / 'precision.jsonl')]
        for flag in ('model', 'layout_model', 'dataset_json', 'images_dir'):
            cmd += ['--' + flag.replace('_', '-'), str(getattr(a, flag).resolve())]
        for flag, key in (('local-torchair-cache-dir', 'decode'),
                          ('local-vision-torchair-cache-dir', 'vision'),
                          ('local-text-torchair-cache-dir', 'text')):
            cmd += ['--' + flag, str(cache[key])]
        cmd += ['--processor-max-pixels', '602112', '--offset', '0', '--limit', str(count),
                '--output-dir', str(stage / 'output'), '--local-vision-grid-device', 'cpu',
                '--local-input-transfer', 'pinned-nonblocking',
                '--no-local-prefill-metrics' if a.phase == 'regression' else '--local-prefill-metrics']
        if candidate:
            cmd += PRODUCTION_PREFILL_OPTIONS
        if warm:
            cmd += ['--local-warm-all-prefill-buckets']
        return cmd, source_repo

    def run(name, count, label, *, candidate=False, parent=False, warm=False, audit=False):
        stage = a.root / name
        stage.mkdir(exist_ok=False)
        cmd, source_repo = command(label, count, stage, candidate=candidate, parent=parent, warm=warm)
        expected_commit = subprocess.check_output(['git', '-C', str(source_repo), 'rev-parse', 'HEAD'], text=True).strip()
        before = cache_artifacts(cmd) if audit else None
        if before is not None:
            write_new(stage / 'cache_before.json', before)
        previous = Path.cwd()
        try:
            os.chdir(source_repo)
            result = run_lane(cmd, stage / 'receipt', 14400, stage / 'run.log',
                              require_idle_card=a.chip == '910b')
        finally:
            os.chdir(previous)
        (stage / 'exit_code.txt').write_text(str(result['exit_code']) + '\n')
        if audit:
            after = cache_artifacts(cmd)
            changed = [key for key in sorted(set(before) | set(after)) if before.get(key) != after.get(key)]
            write_new(stage / 'cache_after.json', after)
            write_new(stage / 'cache_audit.json', dict(unchanged=not changed, changed=changed))
            assert not changed, 'Graph artifacts changed during measurement; result invalid'
        s = json.loads((stage / 'output/run_summary_shard_00.json').read_text())
        assert s['git_commit'] == expected_commit
        assert (s['completed'], s['failed'], s['skipped']) == (count, 0, 0)
        assert s['processor_max_pixels'] == 602112 and s['processor_min_pixels'] == 25088
        assert s['layout_backend'] == 'pp-doclayout-v3' and s['streaming']['layout_calls'] == count
        assert not s['saved_layout_manifest'] and not s['crop_replay_manifest']
        assert s['local_vision_grid_device'] == 'cpu' and s['local_input_transfer'] == 'pinned-nonblocking'
        assert s['local_prefill_metrics'] == (a.phase != 'regression')
        if candidate:
            assert s['processor_text_max_pixels'] == 401408
            assert s['local_text_pack_target'] == 384 and s['local_text_prefill_schedule'] == 'window'
        if warm:
            assert s['warmup_only']
            assert set(map(int, s['all_prefill_bucket_warmup']['vision'])) == set(s['local_compiled_vision']['buckets'])
            assert set(map(int, s['all_prefill_bucket_warmup']['text'])) == set(s['local_compiled_text_prefill']['buckets'])
        if a.chip == '310p':
            records = [json.loads(line) for line in (stage / 'precision.jsonl').read_text().splitlines()]
            assert any(row.get('event') == 'configuration' and row.get('supported_310p') and row.get('allow_internal_format') for row in records)
            assert records[-1]['event'] == 'completed' and records[-1]['mode'] == 4
            assert all(row['stock_converter_restored'] for row in records if row['event'] == 'first_vision_call_finish')
        g = s['local_compiled_generation']
        print('PREFILL_RESULT ' + json.dumps(dict(chip=a.chip, lane=name, commit=expected_commit,
            warmup_only=warm, pages=count, wall_s=s['pipeline_wall_s'],
            pages_per_s=None if warm else count / s['pipeline_wall_s'],
            prefill_s=g['prefill_s'], decode_s=g['decode_s'],
            prefill_metrics=g['prefill_metrics'], streaming=s['streaming'])), flush=True)
        if s['local_prefill_metrics']:
            route_tables(stage / 'output')
        return stage / 'output'

    if a.phase == 'regression':
        assert a.parent_repo and subprocess.check_output(['git', '-C', str(a.parent_repo), 'rev-parse', '--short=8', 'HEAD'], text=True).strip() == '2ef68bdd'
        reference = run('parent64', 64, 'parent', parent=True)
        default = run('defaults64', 64, 'defaults')
        exact(reference, default, a.root / 'regression_compare.json')
    elif a.phase == 'smoke':
        assert (a.root / 'regression_compare.json').is_file(), 'Run the regression gate first'
        candidate = run('candidate_smoke64', 64, 'candidate', candidate=True, warm=True)
        result = compare(a.root / 'defaults64/output', candidate)
        write_new(a.root / 'smoke_compare.json', result)
        print('SMOKE_OUTPUT_DIFFERENCES ' + str(len(result['differences'])), flush=True)
    elif a.phase == 'full':
        assert (a.root / 'smoke_compare.json').is_file(), 'Run the smoke first'
        run('candidate_full1651', 1651, 'candidate', candidate=True, audit=True)
    else:
        for label, candidate in (('baseline', False), ('candidate', True)):
            run(label + '_warmup64', 64, label, candidate=candidate, warm=True)
            run(label + '_full1651', 1651, label, candidate=candidate, audit=True)


if __name__ == '__main__':
    main()
