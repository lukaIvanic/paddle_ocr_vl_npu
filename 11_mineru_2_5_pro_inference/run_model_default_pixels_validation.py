"""Two 910B runs using the production receipt, changing only pixels/buckets/KV."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from run_host_overlap_validation import cache_artifacts
from run_prefill_bucket_validation import route_tables
from vision_diagnostic_runner import run_lane, write_new

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
VISION_BUCKETS = '384,512,768,896,1024,1280,1536,1792,1920,2048,2560,3072,4096,5120,6144,7168,8192'
TEXT_BUCKETS = '128,256,384,512,576,832,1024,1536,2048,2176'
MODEL_DEFAULT_OPTIONS = {
    '--processor-min-pixels': '50176',
    '--processor-max-pixels': '1605632',
    '--local-vision-buckets': VISION_BUCKETS,
    '--local-text-buckets': TEXT_BUCKETS,
    '--local-compiled-cache-length': '8192',
}


def replace_option(argv, flag, value):
    if flag in argv:
        index = argv.index(flag)
        if value is None:
            del argv[index:index + 2]
        else:
            argv[index + 1] = str(value)
    elif value is not None:
        argv.extend([flag, str(value)])


def make_command(reference, root, count, warm):
    argv = list(reference)
    assert Path(argv[2]).name == 'run_page_pipeline.py'
    argv[2] = str(HERE / 'run_page_pipeline.py')
    # Build directly from the verified 910B receipt, not PRODUCTION_PREFILL_OPTIONS.
    replace_option(argv, '--processor-text-max-pixels', None)
    for flag, value in MODEL_DEFAULT_OPTIONS.items():
        replace_option(argv, flag, value)
    for flag, label in (('--local-torchair-cache-dir', 'decode'),
                        ('--local-vision-torchair-cache-dir', 'vision'),
                        ('--local-text-torchair-cache-dir', 'text')):
        replace_option(argv, flag, root / 'cache' / label)
    replace_option(argv, '--limit', count)
    replace_option(argv, '--output-dir', root / ('warmup64' if warm else 'full1651') / 'output')
    assert '--local-warm-all-prefill-buckets' not in argv
    if warm:
        argv.append('--local-warm-all-prefill-buckets')
    return argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--production-run', type=Path, required=True)
    args = parser.parse_args()
    os.chdir(REPO)
    assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    production = args.production_run.resolve()
    receipt = json.loads((production / 'receipt/command.json').read_text())
    reference = receipt['argv']
    baseline = json.loads((production / 'output/run_summary_shard_00.json').read_text())
    assert baseline['completed'] == 1651 and baseline['failed'] == 0
    model = Path(reference[reference.index('--model') + 1])
    defaults = json.loads((model / 'preprocessor_config.json').read_text())
    assert defaults['min_pixels'] == 50176 and defaults['max_pixels'] == 1605632
    write_new(root / 'checkpoint_pixel_defaults.json', defaults)
    write_new(root / 'production_receipt.json', receipt)
    for flag, label in (('--local-torchair-cache-dir', 'decode'),
                        ('--local-vision-torchair-cache-dir', 'vision'),
                        ('--local-text-torchair-cache-dir', 'text')):
        target = root / 'cache' / label
        # Refuse to resume a partially populated or previously used run root.
        shutil.copytree(Path(reference[reference.index(flag) + 1]), target)
    for count, warm in ((64, True), (1651, False)):
        stage = root / ('warmup64' if warm else 'full1651')
        stage.mkdir(exist_ok=False)
        command = make_command(reference, root, count, warm)
        before = cache_artifacts(command) if not warm else None
        if before is not None:
            write_new(stage / 'cache_before.json', before)
        result = run_lane(command, stage / 'receipt', 14400, stage / 'run.log', require_idle_card=True)
        (stage / 'exit_code.txt').write_text(str(result['exit_code']) + '\n')
        if before is not None:
            after = cache_artifacts(command)
            changed = [key for key in sorted(set(before) | set(after)) if before.get(key) != after.get(key)]
            write_new(stage / 'cache_after.json', after)
            write_new(stage / 'cache_audit.json', dict(unchanged=not changed, changed=changed))
            assert not changed, 'Graph artifacts changed during full-run measurement'
        s = json.loads((stage / 'output/run_summary_shard_00.json').read_text())
        assert s['git_commit'] == commit
        assert (s['completed'], s['failed'], s['skipped']) == (count, 0, 0)
        assert s['processor_min_pixels'] == 50176 and s['processor_max_pixels'] == 1605632
        assert s['processor_text_max_pixels'] is None
        assert s['local_compiled_cache_length'] == 8192
        assert s['local_text_pack_target'] == 384 and s['local_text_prefill_schedule'] == 'window'
        assert s['layout_backend'] == 'pp-doclayout-v3' and s['streaming']['layout_calls'] == count
        assert s['local_vision_grid_device'] == baseline['local_vision_grid_device']
        assert s['local_input_transfer'] == baseline['local_input_transfer']
        assert s['local_prefill_metrics'] == baseline['local_prefill_metrics']
        g = s['local_compiled_generation']
        checks = dict(vision_eager_overflow=s['local_compiled_vision']['route_counts'].get('eager_overflow', 0),
                      text_prefill_overflow=g['prefill_metrics'].get('text_prefill_overflow_count', 0))
        write_new(stage / 'overflow_checks.json', checks)
        assert not any(checks.values()), checks
        if warm:
            assert s['warmup_only']
            for name, buckets in (('vision', VISION_BUCKETS), ('text', TEXT_BUCKETS)):
                assert set(s['all_prefill_bucket_warmup'][name]) == set(buckets.split(','))
        print('MODEL_DEFAULT_PIXELS_RESULT ' + json.dumps(dict(lane=stage.name,
            pages=count, child_wall_s=result['child_wall_s'], setup_s=s['setup_s'],
            pipeline_wall_s=s['pipeline_wall_s'], pages_per_s=None if warm else count / s['pipeline_wall_s'],
            prefill_s=g['prefill_s'], decode_s=g['decode_s'], prefill_metrics=g['prefill_metrics'],
            streaming=s['streaming'], all_bucket_warmup=s.get('all_prefill_bucket_warmup'))), flush=True)
        route_tables(stage / 'output')
    env = dict(os.environ, RUN_ROOT=str(root / 'full1651'), CDM_WORKERS='96', LIMIT='1651')
    subprocess.run(['bash', str(HERE / 'run_serving_accuracy.sh')], env=env, check=True)


if __name__ == '__main__':
    main()
