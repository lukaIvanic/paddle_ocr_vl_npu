"""Bounded pinned-buffer cached pipeline; matched serial/overlap real-pair runs."""
import argparse
import os
from pathlib import Path
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from bench_cached_batches import load_cases, emit
from run_cached_nanobeir import Engine, unpack_device
from probe_reranker_endpoint import compiled
from run_reranker_smoke import require
from run_reranker_buckets import profile


class Pipeline:
    def __init__(self, engine, cache, batch, cpus, depth=3):
        self.engine=engine;self.cache=cache;self.batch=batch
        self.pool=ThreadPoolExecutor(8,initializer=lambda:os.sched_setaffinity(0,cpus))
        self.producer=ThreadPoolExecutor(1)
        self.transfer=torch.npu.Stream();self.compute=torch.npu.current_stream()
        self.slots=[];affinity=os.sched_getaffinity(0)
        try:
            os.sched_setaffinity(0,cpus)
            for _ in range(depth):
                host=torch.empty((batch,cache.shape[1]),dtype=torch.float32,pin_memory=True)
                host.zero_()
                slot=dict(host=host,array=host.numpy(),device=torch.empty_like(host,device='npu'),
                    ids=torch.empty((batch,64),dtype=torch.long,pin_memory=True),
                    lens=torch.empty(batch,dtype=torch.int32,pin_memory=True),
                    scores=torch.empty(batch,dtype=torch.float32,pin_memory=True),
                    device_ids=torch.empty((batch,64),dtype=torch.long,device='npu'),
                    device_lens=torch.empty(batch,dtype=torch.int32,device='npu'),
                    ready=torch.npu.Event(),done=torch.npu.Event())
                assert all(slot[k].is_pinned() for k in ['host','ids','lens','scores'])
                self.slots.append(slot)
        finally:os.sched_setaffinity(0,affinity)
        torch.npu.synchronize()

    def gather(self,slot,batch,verify=False):
        start=time.perf_counter()
        def copy(worker):
            for row in range(worker,self.batch,8):
                np.copyto(slot['array'][row],self.cache[batch[row][0]])
        futures=[self.pool.submit(copy,i) for i in range(8)]
        for future in futures:future.result()
        ids=slot['ids'].numpy();lens=slot['lens'].numpy();ids.fill(0)
        for row,pair in enumerate(batch):
            tokens=pair[1];assert 0<len(tokens)<=64
            ids[row,:len(tokens)]=tokens;lens[row]=len(tokens)
            if verify:assert np.array_equal(slot['array'][row],self.cache[pair[0]])
        return time.perf_counter()-start

    def enqueue(self,slot,backend):
        with torch.npu.stream(self.transfer),torch.profiler.record_function('rwkv.async_h2d'):
            slot['device'].copy_(slot['host'],non_blocking=True)
            slot['device_ids'].copy_(slot['ids'],non_blocking=True)
            slot['device_lens'].copy_(slot['lens'],non_blocking=True)
            slot['ready'].record(self.transfer)
        self.compute.wait_event(slot['ready'])
        with torch.profiler.record_function('rwkv.state_rearrange'):
            state=unpack_device(slot['device'])
        with torch.profiler.record_function('rwkv.backbone_and_head'):
            call=self.engine.calls[64] if backend=='torchair' else self.engine.eager
            state=call(slot['device_ids'],slot['device_lens'],*state)
            logits=(self.engine.head if backend=='torchair' else self.engine.ranker)(state[1])
        with torch.profiler.record_function('rwkv.async_score_d2h'):
            slot['scores'].copy_(logits.reshape(-1),non_blocking=True)
            slot['done'].record(self.compute)
        # Keep outputs alive until the event completes, including graph outputs.
        slot['live']=(state,logits)

    def collect(self,slot):
        with torch.profiler.record_function('rwkv.slot_wait_and_scores'):
            slot['done'].synchronize()
            result=slot['scores'].tolist()
            slot.pop('live',None)
        return result

    def run(self,batches,backend,overlap,verify=False):
        results=[None]*len(batches);gather_times=[];waits=[]
        start=time.perf_counter()
        if not overlap:
            slot=self.slots[0]
            for i,batch in enumerate(batches):
                with torch.profiler.record_function('rwkv.host_gather'):
                    gather_times.append(self.gather(slot,batch,verify))
                self.enqueue(slot,backend);results[i]=self.collect(slot)
        else:
            pending={};depth=len(self.slots)
            for i in range(min(depth,len(batches))):
                pending[i]=self.producer.submit(self.gather,self.slots[i],batches[i],verify)
            for i in range(len(batches)):
                slot=self.slots[i%depth]
                before=time.perf_counter()
                with torch.profiler.record_function('rwkv.wait_gather'):
                    gather_times.append(pending.pop(i).result())
                waits.append(time.perf_counter()-before)
                self.enqueue(slot,backend)
                # Retire oldest after submitting this batch, leaving other slots
                # running while the producer refills the released slot.
                retire=i-depth+1
                if retire>=0:
                    old=self.slots[retire%depth];results[retire]=self.collect(old)
                    refill=retire+depth
                    if refill<len(batches):pending[refill]=self.producer.submit(self.gather,old,batches[refill],verify)
            for i in range(max(0,len(batches)-depth+1),len(batches)):
                results[i]=self.collect(self.slots[i%depth])
        seconds=time.perf_counter()-start
        return torch.tensor(sum(results,[])),dict(seconds=seconds,
            pairs_per_second=len(batches)*self.batch/seconds,
            ms_per_batch=seconds/len(batches)*1000,
            gather_ms_median=statistics.median(gather_times)*1000,
            host_wait_gather_ms=sum(waits)*1000)

    def close(self):
        torch.npu.synchronize();self.producer.shutdown();self.pool.shutdown()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['output','prepared','models','reference-build','build-root','vector-build','cache-task']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--batch-size',type=int,choices=[16,32,64],required=True)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--local-cpus',default=','.join(map(str,range(24,40))))
    a=p.parse_args();a.size='large';a.all_queries=True;a.candidates_per_query=64
    a.output.mkdir(parents=True,exist_ok=False)
    status=subprocess.check_output(['npu-status'],text=True)
    line=next(x for x in status.splitlines() if x.startswith('NPU '+os.environ['ASCEND_RT_VISIBLE_DEVICES']+': '))
    assert ': free ' in line and 'Health=OK' in line,line
    pairs,cache=load_cases(a);assert len(pairs)%a.batch_size==0
    batches=[pairs[i:i+a.batch_size] for i in range(0,len(pairs),a.batch_size)]
    assert max(len(p[1]) for p in pairs)<=64
    emit('CONFIG',dict(batch=a.batch_size,pairs=len(pairs),queries=len({p[3] for p in pairs}),
        query_lengths=sorted({len(p[1]) for p in pairs}),ring_depth=3,workers=8,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        scope='All selected real pairs, prepared tokens and warm file-backed states; gather through CPU score materialization, including fill/drain. Setup/compile excluded.'))
    e=Engine(a);e.calls[64]=compiled(e.module.forward,a.output/'continuation_t64')
    pipeline=Pipeline(e,cache,a.batch_size,set(map(int,a.local_cpus.split(','))))
    refs=torch.tensor([p[2] for p in pairs]);canonical={}
    try:
        with torch.inference_mode():
            for backend in ['raw_eager','torchair']:
                begin=time.perf_counter()
                for _ in range(3):pipeline.run(batches[:3],backend,False)
                emit('WARMUP',dict(backend=backend,seconds=time.perf_counter()-begin))
                for overlap in [False,True]:
                    got,stats=pipeline.run(batches,backend,overlap,verify=True)
                    if not overlap:canonical[backend]=got
                    assert torch.equal(got,canonical[backend]),'Overlap changed scores'
                    emit('PARITY',dict(backend=backend,overlap=overlap,exact_state_rows=True,
                        exact_vs_serial=True,vs_saved_b4=require(got,refs,.02,.005)))
            emit('COMPILED_PARITY',require(canonical['torchair'],canonical['raw_eager'],.02,.005))
            for backend in ['raw_eager','torchair']:
                samples={False:[],True:[]}
                for repeat in range(a.repeats):
                    for overlap in ([False,True] if repeat%2==0 else [True,False]):
                        got,stats=pipeline.run(batches,backend,overlap)
                        assert torch.equal(got,canonical[backend]),'Repeat/slot reuse changed scores'
                        samples[overlap].append(stats)
                        emit('REPEAT',dict(backend=backend,overlap=overlap,repeat=repeat,**stats))
                for overlap,runs in samples.items():
                    seconds=statistics.median(r['seconds'] for r in runs)
                    emit('TIMING',dict(backend=backend,overlap=overlap,batch=a.batch_size,
                        pairs=len(pairs),seconds=seconds,pairs_per_second=len(pairs)/seconds,
                        ms_per_batch=seconds/len(batches)*1000,peak_hbm_bytes=torch.npu.max_memory_allocated()))
                for overlap in [False,True]:
                    emit('PROFILE_START',dict(backend=backend,overlap=overlap))
                    profile(lambda:pipeline.run(batches[:12],backend,overlap),
                        a.output/f'profile_{backend}_{"overlap" if overlap else "serial"}',
                        'rwkv.cached_window',warmup_iterations=1,active_iterations=1)
            emit('DONE',dict(all_checks_passed=True))
    finally:pipeline.close()


if __name__=='__main__':main()
