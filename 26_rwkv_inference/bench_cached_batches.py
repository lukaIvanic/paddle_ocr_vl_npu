"""Same real cached pairs across batch sizes; full-pipeline warm timings/profiles."""
import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

from run_cached_nanobeir import Engine, StateLoader, load_task, unpack_device
from run_reranker_smoke import require
from run_reranker_buckets import profile
from run_cpu_reference import sha256
import numpy as np
import torch


def emit(label,value):
    print(label,json.dumps(value),flush=True)


def load_cases(a):
    data=load_task(a,'NanoSCIDOCSRetrieval')
    jobs=sorted(data['jobs'],key=lambda j:(len(data['queries'][j['query_id']]),j['query_id']))
    if not getattr(a,'all_queries',False):
        jobs=[jobs[i] for i in [0,len(jobs)//2,len(jobs)-1]]
    indices={d:i for i,d in enumerate(sorted(data['documents']))}
    old={x['query_id']:x for x in map(json.loads,(a.cache_task/'scores.jsonl').read_text().splitlines())}
    pairs=[];count=getattr(a,'candidates_per_query',32)
    for job in jobs:
        q=job['query_id'];reference=old[q];assert reference['corpus_ids']==job['corpus_ids']
        for d,score in zip(job['corpus_ids'][:count],reference['scores']['cached'][:count]):
            pairs.append((indices[d],data['queries'][q],score,q,d))
    assert len(pairs)==len(jobs)*count
    cache=np.load(a.cache_task/'document_states.npy',mmap_mode='r')
    assert cache.shape==(len(indices),3244032) and cache.dtype==np.float32
    return pairs,cache


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['output','prepared','models','reference-build','build-root','vector-build','cache-task']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--batch-size',type=int,choices=[1,2,4,8,16,32],required=True)
    p.add_argument('--size',default='large',choices=['large'])
    p.add_argument('--repeats',type=int,default=3)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    status=subprocess.check_output(['bash','-c','npu-status'],text=True)
    import os
    line=next(x for x in status.splitlines() if x.startswith('NPU '+os.environ['ASCEND_RT_VISIBLE_DEVICES']+': '))
    assert ': free ' in line and 'Health=OK' in line,line
    pairs,cache=load_cases(a)
    batches=[pairs[i:i+a.batch_size] for i in range(0,len(pairs),a.batch_size)]
    emit('INPUTS',dict(batch=a.batch_size,pairs=len(pairs),query_lengths=sorted({len(x[1]) for x in pairs}),pair_ids=[(q,d) for _,_,_,q,d in pairs],cache=str(a.cache_task),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),script_sha256=sha256(Path(__file__)),scope='Prepared tokens; existing RAM/file-backed FP32 states; includes gather/H2D/rearrange/query preparation/backbone/reranker/score D2H; excludes tokenization/cache build/setup/compile/profile export.'))
    e=Engine(a);loader=StateLoader(cache,a.batch_size)

    def pipeline(batch,backend,mode):
        ids=[x[0] for x in batch]
        if mode=='reuse':state=loader(ids)
        else:
            with torch.profiler.record_function('rwkv.host_gather'):
                selected=cache[ids]
            with torch.profiler.record_function('rwkv.host_extra_copy'):
                copied=np.array(selected,copy=True)
            with torch.profiler.record_function('rwkv.state_h2d'):
                v=torch.from_numpy(copied).to('npu')
            with torch.profiler.record_function('rwkv.state_rearrange'):
                state=unpack_device(v)
        with torch.profiler.record_function('rwkv.backbone_and_head'):
            scores=e.scores([x[1] for x in batch],state,backend)
        with torch.profiler.record_function('rwkv.score_d2h'):
            return scores.cpu().tolist()

    with torch.inference_mode():
        refs=torch.tensor([x[2] for x in pairs]);actual={}
        for backend in ['raw_eager','torchair']:
            # Warm full-model calls; then check every candidate against saved B4.
            begin=time.perf_counter()
            for _ in range(3):pipeline(batches[0],backend,'reuse')
            torch.npu.synchronize()
            emit('WARMUP',dict(batch=a.batch_size,backend=backend,seconds=time.perf_counter()-begin))
            got=torch.tensor(sum([pipeline(b,backend,'reuse') for b in batches],[]))
            actual[backend]=got
            emit('PARITY',dict(batch=a.batch_size,backend=backend,vs_saved_b4=require(got,refs,.02,.005)))
        emit('COMPILED_PARITY',require(actual['torchair'],actual['raw_eager'],.02,.005))
        modes=['legacy','reuse'] if a.batch_size==4 else ['reuse']
        if 'legacy' in modes:
            got=torch.tensor(sum([pipeline(b,'torchair','legacy') for b in batches],[]))
            assert torch.equal(got,actual['torchair']),'Host-copy change altered scores'
            emit('LOADER_PARITY',dict(bitwise_equal=True,pairs=96))
        for backend in ['raw_eager','torchair']:
            samples={m:[] for m in modes}
            for mode in modes:
                for _ in range(2):pipeline(batches[0],backend,mode)
            torch.npu.synchronize()
            for repeat in range(a.repeats):
                for mode in modes[::(-1 if repeat%2 else 1)]:
                    begin=time.perf_counter()
                    for batch in batches:pipeline(batch,backend,mode)
                    torch.npu.synchronize();samples[mode].append(time.perf_counter()-begin)
            for mode in modes:
                seconds=statistics.median(samples[mode])
                emit('TIMING',dict(batch=a.batch_size,backend=backend,loader=mode,pairs=96,seconds=samples[mode],median_seconds=seconds,pairs_per_second=96/seconds,ms_per_batch=seconds/len(batches)*1000,peak_hbm_bytes=torch.npu.max_memory_allocated()))
                # Profile actual full batches, including all host copies and D2H.
                cursor=0
                def call():
                    nonlocal cursor
                    batch=batches[cursor%len(batches)];cursor+=1
                    return pipeline(batch,backend,mode)
                emit('PROFILE_START',dict(batch=a.batch_size,backend=backend,loader=mode))
                profile(call,a.output/f'profile_{backend}_{mode}',f'rwkv.pipeline_{backend}_{mode}',warmup_iterations=3,active_iterations=2)
        emit('DONE',dict(batch=a.batch_size,all_checks_passed=True))


if __name__=='__main__':main()
