"""Paired production/evaluation runs on pages with oversized recognition crops."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from run_vision_timing_production import build_command, validate


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def select_pages(dataset, trace_path, max_tokens, image_token_id):
    names = [Path(row['page_info']['image_path']).name for row in dataset]
    if len(names) != len(set(names)):
        raise ValueError('duplicate dataset names')
    affected = []
    layouts = Counter()
    request_ids = set()
    for line in trace_path.open():
        row = json.loads(line)
        if row['request_id'] in request_ids:
            raise ValueError('duplicate trace request')
        request_ids.add(row['request_id'])
        if row['page'] not in names:
            raise ValueError('trace page outside dataset')
        if row['phase'] == 'layout':
            layouts[row['page']] += 1
        elif row['phase'] == 'recognition':
            tokens = row['prompt_token_ids'].count(image_token_id) * 4
            if tokens > max_tokens:
                affected.append({k: row[k] for k in
                                 ('request_id', 'page', 'block_index', 'block_type', 'bbox', 'angle')}
                                | {'original_raw_vision_tokens': tokens})
    if layouts != Counter(names):
        raise ValueError('expected exactly one layout trace per dataset page')
    selected_names = {r['page'] for r in affected}
    indices = [i for i, name in enumerate(names) if name in selected_names]
    if not indices:
        raise ValueError('no affected pages')
    return [dataset[i] for i in indices], indices, affected


def execute(command, root, log_name='run.log'):
    (root / (log_name + '.command')).write_text(shlex.join(command) + '\n')
    started = time.monotonic()
    with (root / log_name).open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    (root / (log_name + '.exit_code')).write_text(str(result.returncode) + '\n')
    (root / (log_name + '.wall_s')).write_text(str(time.monotonic() - started) + '\n')
    if result.returncode:
        raise RuntimeError(f'exit={result.returncode}: {root / log_name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-summary', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--max-pixels', type=int, default=1103872)
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parent.parent)
    root = args.run_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / 'pid.txt').write_text(str(os.getpid()) + '\n')
    (root / 'commit.txt').write_text(subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True))
    (root / 'visible_device.txt').write_text(os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '') + '\n')
    code = 1
    try:
        reference = json.loads(args.reference_summary.read_text())
        original_dataset = Path(reference['dataset_json'])
        if sha(original_dataset) != reference['model_hashes']['dataset_json']:
            raise ValueError('reference dataset hash mismatch')
        model = Path(reference['model'])
        config = json.loads((model / 'config.json').read_text())
        preprocessor = json.loads((model / 'preprocessor_config.json').read_text())
        if preprocessor['patch_size'] != 14 or preprocessor['merge_size'] != 2:
            raise ValueError('unexpected checkpoint image patch geometry')
        original_max = preprocessor['max_pixels']
        if not reference['processor_min_pixels'] <= args.max_pixels < original_max:
            raise ValueError('cap must be between production min and original max')
        trace = args.reference_summary.parent / 'generation_trace.jsonl'
        subset, indices, crops = select_pages(json.loads(original_dataset.read_text()), trace,
                                             args.max_pixels // 196, config['image_token_id'])
        subset_path = root / 'OmniDocBench_affected.json'
        dump(subset_path, subset)
        dump(root / 'selection.json', {
            'reference_summary': str(args.reference_summary),
            'reference_summary_sha256': sha(args.reference_summary),
            'reference_trace_sha256': sha(trace),
            'original_dataset_sha256': sha(original_dataset),
            'subset_sha256': sha(subset_path),
            'selection_rule': 'recognition prompt image-pad count * 4 > capped raw token limit',
            'page_count': len(subset), 'crop_count': len(crops), 'original_indices': indices,
            'crop_types': dict(Counter(r['block_type'] for r in crops)), 'crops': crops,
        })
        print(f'SELECTION pages={len(subset)} crops={len(crops)}', flush=True)
        dump(root / 'reference_summary.json', reference)
        paired_reference = dict(reference, dataset_json=str(subset_path),
                                model_hashes=dict(reference['model_hashes'], dataset_json=sha(subset_path)))
        lock = Path(reference['local_torchair_cache_dir']) / 'vision_timing_production.lock'
        with lock.open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for label, maximum in [('original', original_max), ('capped', args.max_pixels)]:
                run = root / label
                run.mkdir()
                command = build_command(paired_reference, run / 'output', len(subset), maximum)
                print(f'INFERENCE_START {label} max_pixels={maximum}', flush=True)
                execute(command, run)
                summary = validate(run / 'output', paired_reference, len(subset))
                assert summary['processor_max_pixels'] == maximum
                if label == 'capped':
                    assert max(x for line in (run / 'output/vision_timing_shard_00.jsonl').open()
                               for x in json.loads(line)['member_lengths']) <= args.max_pixels // 196
                (run / 'exit_code.txt').write_text('0\n')
                print(f'INFERENCE_FINISH {label} pages={summary["completed"]} '
                      f'hot_s={summary["pipeline_wall_s"]}', flush=True)
        for label in ['original', 'capped']:
            run = root / label
            os.environ.update(RUN_ROOT=str(run), DATASET_JSON=str(subset_path), LIMIT=str(len(subset)))
            print(f'EVALUATION_START {label}', flush=True)
            execute(['bash', '11_mineru_2_5_pro_inference/run_serving_accuracy.sh'], run, 'evaluation_launcher.log')
            print(f'EVALUATION_FINISH {label}', flush=True)
        code = 0
    finally:
        (root / 'exit_code.txt').write_text(str(code) + '\n')


if __name__ == '__main__':
    main()
