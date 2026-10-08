#!/usr/bin/env python3
"""Smoke then full live cap3072 reproduction using an exact d4e7fdd checkout."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from vision_diagnostic_runner import run_lane

D4 = 'd4e7fdd0efd90bf6408c21efab320cc3292f07a4'


def gate(root, count):
    output = root/'output'
    s = json.loads((output/'run_summary_shard_00.json').read_text())
    assert (s['completed'],s['failed'],s['skipped']) == (count,0,0)
    assert s['layout_backend'] == 'pp-doclayout-v3'
    assert s['streaming']['layout_source'] == 'PP-DocLayoutV3_live'
    assert s['streaming']['layout_calls'] == count
    assert s['streaming']['layout_model_dtype'] == 'torch.float32'
    assert not s['layout_graph_capture'] and not s['image_analysis']
    assert not s['saved_layout_manifest'] and not s['crop_replay_manifest']
    assert s['processor_max_pixels'] == 602112 and s['processor_min_pixels'] == 25088
    assert s['warmup']['executed_pages'] == 0
    assert s['local_decode_increfa_length_mode'] == 'pse_sentinel_310p'
    assert s['local_compiled_vision']['layer_norm_impl'] == 'manual_fp32'
    assert s['local_compiled_vision']['projection_impl'] == 'linear'
    assert s['batch_size'] == 32 and s['local_compiled_cache_length'] == 4096
    for folder,suffix in [('predictions','*.md'),('content_lists','*.json'),('layout_regions','*.json')]:
        assert len(list((output/folder).glob(suffix))) == count
    cfg = json.loads((Path(s['model'])/'config.json').read_text())
    ids = set()
    for line in (output/'generation_trace.jsonl').open():
        row = json.loads(line)
        assert row['phase'] == 'recognition' and row['request_id'] not in ids
        ids.add(row['request_id'])
        assert row['prompt_token_ids'].count(cfg['image_token_id'])*4 <= 3072
    assert len(ids) == s['generation_trace']['requests']
    assert len(ids) == s['streaming']['requests_completed_by_phase']['recognition']
    v = s['local_compiled_vision']
    result = dict(pages=count, pipeline_wall_s=s['pipeline_wall_s'],
        pg_s=count/s['pipeline_wall_s'], crops=len(ids), vision_real_tokens=v['real_tokens'],
        vision_physical_tokens=v['physical_tokens'], padding_tokens=v['physical_tokens']-v['real_tokens'],
        padding_fraction=1-v['real_tokens']/v['physical_tokens'], vision=s.get('vision_timing'),
        source_commit=s.get('git_commit'), setup=s.get('setup_s'), chip='Ascend910B2')
    (root/'gate.json').write_text(json.dumps(result,indent=2)+'\n')
    print('LIVE_D4_GATE_PASS '+json.dumps(result),flush=True)
    return s


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-repo',type=Path,required=True)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--reference-summary',type=Path,required=True)
    p.add_argument('--lock-file',type=Path,required=True)
    p.add_argument('--continue-after-smoke',action='store_true',help='Validate an existing successful smoke, then launch only the untouched full stage')
    a=p.parse_args()
    source=a.source_repo.resolve()
    assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip() == D4
    assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain','--untracked-files=no'],text=True).strip()
    ref=json.loads(a.reference_summary.read_text())
    root=a.run_root.resolve()
    if a.continue_after_smoke:
        assert (root/'source_commit.txt').read_text().strip() == D4
        assert json.loads((root/'smoke2'/'receipt'/'exit.json').read_text())['status'] == 'completed'
        gate(root/'smoke2',2)
        assert not (root/'full1651').exists()
    else:
        root.mkdir(parents=True,exist_ok=False)
        (root/'source_commit.txt').write_text(D4+'\n')
    common=[sys.executable,'-u',str(source/'11_mineru_2_5_pro_inference/run_page_pipeline.py'),
        '--layout-backend','pp-doclayout-v3','--no-layout-graph-capture',
        '--processor-min-pixels','25088','--processor-max-pixels','602112',
        '--warmup-pages','0','--no-resume','--fail-fast']
    for key in ['model','layout_model','images_dir','local_torchair_cache_dir',
                'local_vision_torchair_cache_dir','local_text_torchair_cache_dir']:
        path=Path(ref[key])
        if not path.is_absolute():
            path=a.reference_summary.resolve().parents[3]/path
        if not path.exists(): raise FileNotFoundError(f'{key}: {path}')
        common.extend(['--'+key.replace('_','-'),str(path)])
    images=Path(ref['images_dir'])
    smoke=common+['--limit','2','--output-dir',str(root/'smoke2'/'output'),'--input-images',
        str(images/'page-573c437e-c309-4483-a038-ef2f440b104a.png'),
        str(images/'page-9bcba6da-bdb0-4403-97cb-8874898ac8ab.png')]
    full=common+['--dataset-json',ref['dataset_json'],'--offset','0','--limit','1651',
        '--output-dir',str(root/'full1651'/'output')]
    os.chdir(source)  # Child receipts and pipeline metadata must name actual d4 source.
    with a.lock_file.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name,command,count,deadline in [('smoke2',smoke,2,1800),('full1651',full,1651,7200)]:
            if name == 'smoke2' and a.continue_after_smoke:
                continue
            stage=root/name;stage.mkdir()
            row=run_lane(command,stage/'receipt',deadline,stage/'run.log')
            (stage/'exit_code.txt').write_text(str(row['exit_code'])+'\n')
            s=gate(stage,count)
            for key,expected in ref['model_hashes'].items():
                if count == 2 and key == 'dataset_json':
                    continue  # Explicit-image smoke generates its own two-page manifest.
                assert s['model_hashes'][key] == expected,f'model/data hash differs: {key}'
    print('D4_REPRODUCTION_COMPLETED '+str(root),flush=True)


if __name__=='__main__':main()
