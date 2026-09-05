"""Repeat the validated production preset using asset/cache paths from a completed run."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


def build_command(reference, output, limit, max_pixels=None):
    required = {
        'completed': 1651, 'failed': 0, 'skipped': 0,
        'batch_size': 32, 'page_batch_size': 32, 'streaming_page_window': 32,
        'global_request_stream': True, 'streaming_pages': True,
        'dtype': 'float16', 'local_compiled_cache_length': 4096,
        'local_decode_increfa_length_mode': 'pse_sentinel_310p',
        'local_decode_attention': 'increfa', 'local_decode_weight_format': 'decode_nz',
        'local_decode_rotary_impl': 'npu_apply', 'local_text_backend': 'torchair-packed',
        'local_vision_attention': 'prompt_flash_attention', 'local_vision_backend': 'torchair',
        'local_vision_pack_target': 768, 'local_vision_lookahead': 32,
        'local_prepare_prefetch_depth': 64, 'processor_min_pixels': 25088,
        'local_text_max_members': 32, 'layout_image_size': [1036, 1036],
        'local_vision_buckets': '384,512,768,1024,1536,2048,3072,4224,5632',
        'local_text_buckets': '128,256,512,1024', 'image_analysis': False,
    }
    for key, expected in required.items():
        if reference.get(key) != expected:
            raise ValueError(f'reference mismatch: {key}={reference.get(key)!r}, expected {expected!r}')
    vision = reference['local_compiled_vision']
    for key, expected in {'layer_norm_impl': 'manual_fp32', 'projection_impl': 'linear',
                          'promptfa_pad_head_dim_to': 0, 'mask_sparse_mode': 1}.items():
        if vision.get(key) != expected:
            raise ValueError(f'vision reference mismatch: {key}')
    command = [sys.executable, str(Path(__file__).with_name('run_official_transformers_omnidocbench.py')),
               '--backend', 'local-continuous-client', '--output-dir', str(output),
               '--offset', '0', '--limit', str(limit), '--warmup-pages', '2',
               '--no-resume', '--fail-fast', '--global-request-stream', '--streaming-pages',
               '--token-trace', '--hash-model-files', '--local-prefill-metrics',
               '--local-dtype', 'float16']
    for key in required:
        if key in ('completed', 'failed', 'skipped', 'dtype', 'image_analysis') or isinstance(required[key], bool):
            continue
        value = reference[key]
        command.append('--' + key.replace('_', '-'))
        command.extend(str(v) for v in value) if isinstance(value, list) else command.append(str(value))
    for key in ('model', 'dataset_json', 'images_dir', 'local_text_torchair_cache_dir',
                'local_vision_torchair_cache_dir', 'local_torchair_cache_dir'):
        path = Path(reference[key]).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parent.parent / path
        path = path.resolve()
        if not path.exists():
            raise ValueError(f'missing reference path: {key}={path}')
        command.extend(['--' + key.replace('_', '-'), str(path)])
    effective_max = max_pixels if max_pixels is not None else reference.get('processor_max_pixels')
    if effective_max is not None:
        if effective_max < reference['processor_min_pixels']:
            raise ValueError('processor max pixels must be at least the reference min pixels')
        command.extend(['--processor-max-pixels', str(effective_max)])
    return command


def validate(output, reference, limit):
    summary = json.loads((output/'run_summary_shard_00.json').read_text())
    assert summary['completed'] == limit and summary['failed'] == 0 and summary['skipped'] == 0
    assert summary['model_hashes'] == reference['model_hashes'], 'asset hashes changed'
    assert summary['local_decode_increfa_length_mode'] == 'pse_sentinel_310p'
    assert summary['local_compiled_vision']['layer_norm_impl'] == 'manual_fp32'
    timing = summary['vision_timing']
    rows = [json.loads(line) for line in (output/timing['raw_samples_file']).read_text().splitlines()]
    routes = summary['local_compiled_vision']['route_counts']
    assert len(rows) == sum(routes.values()) == timing['all']['calls']
    assert sum(r['real_tokens'] for r in rows) == summary['local_compiled_vision']['real_tokens']
    assert sum(r['physical_tokens'] for r in rows) == summary['local_compiled_vision']['physical_tokens']
    device = summary['streaming']['decode']['prefill_metrics']['vision_transformer_blocks']
    assert abs(timing['all']['device_s'] - device) < 1e-6, 'tagged and aggregate event totals differ'
    for route, count in routes.items():
        name = f'bucket_{route}' if route.isdigit() else route
        assert timing['by_route'][name]['calls'] == count
    for folder, suffix in [('progress', '*.json'), ('content_lists', '*.json'), ('predictions', '*.md')]:
        assert len(list((output/folder).glob(suffix))) == limit
    trace = [json.loads(line) for line in (output/'generation_trace.jsonl').read_text().splitlines()]
    assert len(trace) == summary['generation_trace']['requests']
    expected = {f'bucket_{b}' for b in summary['local_compiled_vision']['buckets']} | {'packed_768', 'eager_overflow'}
    print('VISION_TIMING_VALIDATION: PASS', flush=True)
    print('uncovered_routes=' + str(sorted(expected-set(timing['by_route']))), flush=True)
    return summary


def main():
    os.chdir(Path(__file__).resolve().parent.parent)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-summary', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=384)
    parser.add_argument('--processor-max-pixels', type=int,
                        help='Explicit resolution ablation; 1103872 caps raw vision tokens at 5632.')
    args = parser.parse_args()
    reference = json.loads(args.reference_summary.read_text())
    root = args.run_root.resolve()
    command = build_command(reference, root/'output', args.limit, args.processor_max_pixels)
    root.mkdir(parents=True, exist_ok=False)
    (root/'pid.txt').write_text(str(os.getpid())+'\n')
    (root/'reference_summary.json').write_text(json.dumps(reference, indent=2)+'\n')
    (root/'command.sh').write_text(shlex.join(command)+'\n')
    (root/'visible_device.txt').write_text(os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '')+'\n')
    (root/'commit.txt').write_text(subprocess.check_output(['git','rev-parse','HEAD'],text=True))
    lock = Path(reference['local_torchair_cache_dir'])/'vision_timing_production.lock'
    started = time.monotonic()
    code = 1
    try:
        with lock.open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with (root/'run.log').open('w') as log:
                code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
            (root/'inference_exit_code.txt').write_text(str(code)+'\n')
            if code != 0:
                raise RuntimeError(f'inference exit={code}; inspect {root}/run.log')
            code = 1
            result = validate(root/'output', reference, args.limit)
            from vision_timing_report import explain
            explain(result)
            code = 0
    finally:
        (root/'process_wall_s.txt').write_text(str(time.monotonic()-started)+'\n')
        (root/'exit_code.txt').write_text(str(code)+'\n')


if __name__ == '__main__':
    main()
