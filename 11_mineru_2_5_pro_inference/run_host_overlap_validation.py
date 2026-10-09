"""Bounded 910B validation stages for host-only MinerU changes.

Source must be clean and published before use. Each lane preserves an immutable
receipt and stops on any output difference; comparison never allows drift.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from compare_generation_traces import compare
from vision_diagnostic_runner import run_lane

HERE=Path(__file__).resolve().parent
FLAGS={
 'baseline':[],
 'c1':['--local-vision-grid-device','cpu'],
 'c2_only':['--local-input-transfer','pinned-nonblocking'],
 'c5_only':['--no-local-prefill-metrics'],
 'c2':['--local-vision-grid-device','cpu','--local-input-transfer','pinned-nonblocking'],
 'c5':['--local-vision-grid-device','cpu','--local-input-transfer','pinned-nonblocking','--no-local-prefill-metrics'],
}


def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def exact(reference,candidate,path):
    result=compare(reference,candidate)
    save(path,result)
    for key in ['missing_requests','extra_requests','differences','changed_pages','missing_pages','extra_pages','new_length_stops','unexpected_input_changes']:
        assert not result[key],f'{key}: see {path}'
    assert result['candidate_trace_accounting'] and all(result['candidate_trace_accounting'].values())
    return dict(exact_requests=result['candidate_requests'],exact_pages=result['byte_identical_pages'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-run',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--cache-root',type=Path,help='Shared study cache parent; each configuration still has its own directory')
    p.add_argument('--stage',choices=['vision','smoke','diagnostic','performance','full'],required=True)
    p.add_argument('--lanes',nargs='+',choices=list(FLAGS),default=['baseline','c1'])
    p.add_argument('--diagnostic-pages',type=int,default=8)
    p.add_argument('--diagnostic-drain',choices=['both','queued','drained'],default='both')
    p.add_argument('--diagnostic-profiler',type=Path,help='Optional standalone py-spy binary; diagnostic only')
    p.add_argument('--vision-transfer',default='blocking')
    p.add_argument('--vision-metrics-off',action='store_true')
    p.add_argument('--vision-grid',choices=['cpu','npu'],default='cpu')
    a=p.parse_args()
    a.root.mkdir(parents=True,exist_ok=True)
    if a.stage=='performance' and len(a.lanes)!=2:
        p.error('performance requires exactly two adjacent configurations, run in A B A B order')
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip()
    ref=json.loads((a.reference_run/'output/run_summary_shard_00.json').read_text())
    with (a.root/'ownership.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def cache(label):
            result={}
            for key in ['local_torchair_cache_dir','local_vision_torchair_cache_dir','local_text_torchair_cache_dir']:
                source=Path(ref[key]);assert source.is_absolute() and source.is_dir()
                target=(a.cache_root or a.root/'cache')/label/key
                if not target.exists():
                    target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copytree(source,target,symlinks=True)
                result[key]=str(target)
            return result
        def run(name,command,deadline=3600):
            stage=a.root/name;stage.mkdir(exist_ok=False)
            result=run_lane(command,stage/'receipt',deadline,stage/'run.log')
            (stage/'exit_code.txt').write_text(str(result['exit_code'])+'\n')
            return stage
        def command(label,count,name,diagnostic=None,drain=False):
            cmd=[sys.executable,'-u',str(HERE/('host_overlap_diagnostics.py' if diagnostic else 'run_page_pipeline.py'))]
            if diagnostic:
                cmd+=['--diagnostic-output',str(a.root/name/'diagnostic')]
                if drain:cmd+=['--drain-before-h2d']
            for key in ['model','layout_model','images_dir','dataset_json']:
                cmd+=['--'+key.replace('_','-'),str(ref[key])]
            for key,value in cache(label).items():cmd+=['--'+key.replace('_','-'),value]
            cmd+=['--processor-max-pixels','602112','--offset','0','--limit',str(count),
                '--output-dir',str(a.root/name/'output')]+FLAGS[label]
            return cmd
        if a.stage=='vision':
            label=f'vision_{a.vision_grid}_{a.vision_transfer}'+('_metrics_off' if a.vision_metrics_off else '')
            cmd=[sys.executable,'-u',str(HERE/'check_host_overlap_vision.py'),
                '--reference-run',str(a.reference_run),'--cache-root',cache('validation')['local_vision_torchair_cache_dir'],
                '--output',str(a.root/label/'check'),'--candidate-grid',a.vision_grid,
                '--candidate-transfer',a.vision_transfer]
            if a.vision_metrics_off:cmd+=['--candidate-metrics-off']
            run(label,cmd)
            return
        if a.stage=='diagnostic':
            for lane in a.lanes:
                for drain in ([False,True] if a.diagnostic_drain=='both' else [a.diagnostic_drain=='drained']):
                    name=f'diagnostic_{lane}_{"drained" if drain else "queued"}'
                    cmd=command(lane,a.diagnostic_pages,name,diagnostic=True,drain=drain)
                    if a.diagnostic_profiler:
                        cmd=[str(a.diagnostic_profiler),'record','--rate','49','--format','raw',
                             '--threads','--native','--idle','--output',str(a.root/name/'native_stacks.txt'),'--']+cmd
                    run(name,cmd)

            return
        count={'smoke':64,'performance':384,'full':1651}[a.stage]
        repeats=2 if a.stage=='performance' else 1
        reference=None;table=[]
        for repeat in range(repeats):
            for lane in a.lanes:
                name=f'{a.stage}_r{repeat+1}_{lane}'
                stage=run(name,command(lane,count,name),7200 if count==1651 else 3600)
                s=json.loads((stage/'output/run_summary_shard_00.json').read_text())
                assert (s['completed'],s['failed'],s['skipped'])==(count,0,0)
                assert s['processor_max_pixels']==602112 and s['layout_backend']=='pp-doclayout-v3'
                assert s['streaming']['layout_calls']==count
                if reference is None:reference=stage/'output';parity=None
                else:parity=exact(reference,stage/'output',stage/'exactness.json')
                g=s['local_compiled_generation'];m=g['prefill_metrics'];st=s['streaming']
                # Event regions can contain launch gaps; layout is a host span.
                # This residual is requested but is NOT measured device idle.
                measured=sum(float(m.get(k,0)) for k in ['token_embedding','vision_patch_embed','vision_position_prepare','vision_transformer_blocks','vision_merger','image_embed_scatter','mrope_prepare','text_transformer_prefill','text_kv_redistribute','prefill_lm_head'])+g['decode_s']
                table.append(dict(lane=lane,repeat=repeat+1,chip='910B2',pages=count,
                    wall_s=s['pipeline_wall_s'],pages_per_s=count/s['pipeline_wall_s'],
                    request_h2d_submit_s=st['request_h2d_submit_s'],cpu_prepare_wait_s=st['cpu_prepare_wait_s'],
                    vision_position_prepare_s=m.get('vision_position_prepare'),prefill_s=g['prefill_s'],
                    wall_minus_measured_prefill_and_decode_s=(s['pipeline_wall_s']-measured) if s['local_prefill_metrics'] else None,
                    idle_estimate_limit='Residual includes layout, other device work, launch gaps and host work; not measured NPU idle.',
                    exactness=parity))
                save(a.root/f'{a.stage}_results.json',table)
                print('HOST_LANE_RESULT '+json.dumps(table[-1]),flush=True)


if __name__=='__main__':main()
