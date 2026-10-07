"""Synchronous CPU gather alternatives inside the real cached NPU pipeline."""
import argparse
import collections
import ctypes
import os
from pathlib import Path
import resource
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

from bench_cached_batches import load_cases, emit
from run_cached_nanobeir import Engine, unpack_device
from run_reranker_smoke import require
from run_reranker_buckets import profile
import numpy as np
import torch


class Gather:
    def __init__(self,cache,batch,method,local_cpus,shared_buffer=None):
        self.cache=cache;self.plain=np.asarray(cache);self.method=method
        self.local_cpus=local_cpus;self.pool=None
        affinity=os.sched_getaffinity(0)
        try:
            if method.startswith('local_'):os.sched_setaffinity(0,local_cpus)
            self.buffer=np.empty((batch,cache.shape[1]),dtype=np.float32) if shared_buffer is None else shared_buffer
            self.buffer.fill(0)  # First-touch before timing, same for every method.
        finally:os.sched_setaffinity(0,affinity)
        self.tensor=torch.from_numpy(self.buffer)
        self.indices=torch.empty(batch,dtype=torch.long)
        if method=='torch_index':self.source=torch.from_numpy(self.plain)
        if 'threads' in method:
            self.workers=int(method.split('threads')[1])
            initialize=(lambda:os.sched_setaffinity(0,local_cpus)) if method.startswith('local_') else None
            self.pool=ThreadPoolExecutor(self.workers,initializer=initialize)
        self.memmove=ctypes.CDLL(None).memmove
        self.memmove.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t]
        self.memmove.restype=ctypes.c_void_p

    def copy_rows(self,rows,indices):
        for row in rows:np.copyto(self.buffer[row],self.cache[indices[row]])

    def __call__(self,indices):
        assert len(indices)==len(self.buffer) and all(0<=i<len(self.cache) for i in indices)
        if self.method=='take':
            # Validated indices preserve raise semantics; clip avoids NumPy's
            # documented mode='raise' output buffering.
            np.take(self.plain,indices,axis=0,out=self.buffer,mode='clip')
        elif self.method=='torch_index':
            for row,index in enumerate(indices):self.indices[row]=index
            torch.index_select(self.source,0,self.indices,out=self.tensor)
        elif self.pool is not None:
            futures=[self.pool.submit(self.copy_rows,range(i,len(indices),self.workers),indices) for i in range(self.workers)]
            for future in futures:future.result()  # Join all work BEFORE H2D.
        elif self.method=='memmove':
            width=self.buffer.strides[0]
            for row,index in enumerate(indices):self.memmove(self.buffer.ctypes.data+row*width,self.cache.ctypes.data+index*width,width)
        elif self.method=='plain_rows':
            for row,index in enumerate(indices):np.copyto(self.buffer[row],self.plain[index])
        elif self.method=='local_rows':
            affinity=os.sched_getaffinity(0)
            try:
                os.sched_setaffinity(0,self.local_cpus);self.copy_rows(range(len(indices)),indices)
            finally:os.sched_setaffinity(0,affinity)
        else:self.copy_rows(range(len(indices)),indices)
        return self.tensor

    def close(self):
        if self.pool:self.pool.shutdown(wait=True)


def placement(array):
    address=array.ctypes.data
    for line in Path('/proc/self/maps').read_text().splitlines():
        interval=line.split()[0];lo,hi=[int(x,16) for x in interval.split('-')]
        if lo<=address<hi:
            return next((x for x in Path('/proc/self/numa_maps').read_text().splitlines() if x.startswith(f'{lo:x} ')),line)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['output','prepared','models','reference-build','build-root','vector-build','cache-task']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--candidates-per-query',type=int,choices=[32,64],default=32);p.add_argument('--size',default='large')
    p.add_argument('--methods',nargs='+',default=['rows','plain_rows','take','torch_index','memmove','threads2','threads4','local_rows'])
    p.add_argument('--local-cpus',default='24,25,26,27');p.add_argument('--repeats',type=int,default=3)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    status=subprocess.check_output(['npu-status'],text=True)
    line=next(x for x in status.splitlines() if x.startswith('NPU '+os.environ['ASCEND_RT_VISIBLE_DEVICES']+': '));assert ': free ' in line and 'Health=OK' in line,line
    pairs,cache=load_cases(a);assert len(pairs)%a.batch_size==0
    batches=[pairs[i:i+a.batch_size] for i in range(0,len(pairs),a.batch_size)]
    e=Engine(a);local_cpus=set(map(int,a.local_cpus.split(',')))
    shared=np.empty((a.batch_size,cache.shape[1]),dtype=np.float32);shared.fill(0)
    loaders={m:Gather(cache,a.batch_size,m,local_cpus,None if m.startswith('local_') else shared) for m in a.methods}
    emit('CONFIG',dict(batch=a.batch_size,pairs=len(pairs),methods=a.methods,torch_threads=torch.get_num_threads(),cpu_affinity=sorted(os.sched_getaffinity(0)),numpy=np.__version__,query_lengths=sorted({len(x[1]) for x in pairs}),pair_ids=[(x[3],x[4]) for x in pairs],source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),no_overlap=True))
    gather_stats=collections.defaultdict(list)
    def pipeline(batch,backend,method):
        indices=[x[0] for x in batch];loader=loaders[method]
        before=resource.getrusage(resource.RUSAGE_SELF);t=time.perf_counter()
        with torch.profiler.record_function('rwkv.host_gather'):
            host=loader(indices)
        elapsed=time.perf_counter()-t;after=resource.getrusage(resource.RUSAGE_SELF)
        gather_stats[method].append((elapsed,after.ru_minflt-before.ru_minflt,after.ru_majflt-before.ru_majflt,after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime))
        with torch.profiler.record_function('rwkv.state_h2d'):device=host.to('npu')
        with torch.profiler.record_function('rwkv.state_rearrange'):state=unpack_device(device)
        with torch.profiler.record_function('rwkv.backbone_and_head'):scores=e.scores([x[1] for x in batch],state,backend)
        with torch.profiler.record_function('rwkv.score_d2h'):return scores.cpu().tolist()
    try:
        with torch.inference_mode():
            # Exact CPU-state checks on every real row, outside performance windows.
            for method,loader in loaders.items():
                for batch in batches:
                    indices=[x[0] for x in batch];loader(indices)
                    for row,index in enumerate(indices):assert np.array_equal(loader.buffer[row],cache[index]),(method,index)
                emit('STATE_PARITY',dict(method=method,bitwise_equal=True,destination=placement(loader.buffer)))
            emit('SOURCE_PLACEMENT',placement(cache))
            refs=torch.tensor([x[2] for x in pairs]);canonical={}
            for backend in ['raw_eager','torchair']:
                begin=time.perf_counter()
                for _ in range(3):pipeline(batches[0],backend,a.methods[0])
                emit('WARMUP',dict(backend=backend,seconds=time.perf_counter()-begin))
                for method in a.methods:
                    got=torch.tensor(sum([pipeline(b,backend,method) for b in batches],[]))
                    check=require(got,refs,.02,.005)
                    if backend not in canonical:canonical[backend]=got
                    assert torch.equal(got,canonical[backend]),(backend,method,'gather changed logits')
                    emit('SCORE_PARITY',dict(backend=backend,method=method,exact_vs_baseline=True,vs_saved_b4=check))
            emit('COMPILED_PARITY',require(canonical['torchair'],canonical['raw_eager'],.02,.005))
            for backend in ['raw_eager','torchair']:
                samples=collections.defaultdict(list);stats=collections.defaultdict(list)
                for repeat in range(a.repeats):
                    order=a.methods[repeat:]+a.methods[:repeat]
                    if repeat%2:order=order[::-1]
                    for method in order:
                        gather_stats[method].clear();t=time.perf_counter()
                        for batch in batches:pipeline(batch,backend,method)
                        torch.npu.synchronize();samples[method].append(time.perf_counter()-t)
                        stats[method].extend(gather_stats[method])
                for method in a.methods:
                    seconds=statistics.median(samples[method]);g=stats[method]
                    emit('TIMING',dict(batch=a.batch_size,backend=backend,method=method,seconds=samples[method],ms_per_batch=seconds/len(batches)*1000,pairs_per_second=len(pairs)/seconds,gather_ms_median=statistics.median(x[0] for x in g)*1000,gather_ms_max=max(x[0] for x in g)*1000,gather_cpu_ms_median=statistics.median(x[3] for x in g)*1000,minor_faults=sum(x[1] for x in g),major_faults=sum(x[2] for x in g),peak_hbm_bytes=torch.npu.max_memory_allocated()))
                if backend=='torchair':best=min(a.methods,key=lambda m:statistics.median(samples[m]))
            # Preserve matched eager/compiled full-pipeline profiles for baseline
            # and measured winner; no isolated transfer/copy microbenchmark.
            for backend in ['raw_eager','torchair']:
                for method in dict.fromkeys([a.methods[0],best]):
                    cursor=0
                    def call():
                        nonlocal cursor
                        batch=batches[cursor%len(batches)];cursor+=1
                        return pipeline(batch,backend,method)
                    emit('PROFILE_START',dict(backend=backend,method=method))
                    profile(call,a.output/f'profile_{backend}_{method}',f'rwkv.pipeline_{backend}_{method}',warmup_iterations=3,active_iterations=2)
            emit('DONE',dict(best=best,all_checks_passed=True))
    finally:
        for loader in loaders.values():loader.close()


if __name__=='__main__':main()
