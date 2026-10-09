"""Instrument the real page pipeline; these runs are NOT performance results.

The optional pre-transfer fence attributes pending device work separately from
Tensor.to host duration. It changes scheduling and is diagnostic only. Pool spans
include wall/thread CPU time, queue delay, affinity and intra-op thread settings.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import threading
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--diagnostic-output',type=Path,required=True)
    p.add_argument('--drain-before-h2d',action='store_true')
    args,rest=p.parse_known_args()
    import torch
    import native_custom_backend as native
    import run_page_pipeline as runner
    args.diagnostic_output.mkdir(parents=True,exist_ok=False)
    tls=threading.local()
    rows=[];jobs=[];lock=threading.Lock()
    original_factory=native.make_local_fixed_batch_vlm_client
    original_to=torch.Tensor.to
    original_submit=ThreadPoolExecutor.submit

    def tensor_to(tensor,*a,**kw):
        record=getattr(tls,'record',None)
        if record is None:return original_to(tensor,*a,**kw)
        start=time.perf_counter();cpu=time.thread_time()
        out=original_to(tensor,*a,**kw)
        record['copies'].append(dict(shape=list(tensor.shape),bytes=tensor.numel()*tensor.element_size(),
            source_device=str(tensor.device),source_dtype=str(tensor.dtype),target_device=str(out.device),
            target_dtype=str(out.dtype),non_blocking=kw.get('non_blocking',False),
            field=record['fields'].get(id(tensor)),wall_s=time.perf_counter()-start,
            thread_cpu_s=time.thread_time()-cpu))
        return out

    def factory(*a,**kw):
        client=original_factory(*a,**kw)
        original=client._finish_generation
        def finish(inputs,params,positions,deltas):
            fence=time.perf_counter()
            if args.drain_before_h2d:torch.npu.synchronize()
            fence=time.perf_counter()-fence if args.drain_before_h2d else 0
            record=dict(fields={id(v):k for k,v in inputs.items()},copies=[],fence_s=fence)
            record['fields'].update({id(positions):'position_ids',id(deltas):'rope_deltas'})
            start=time.perf_counter();cpu=time.thread_time();tls.record=record
            try:return original(inputs,params,positions,deltas)
            finally:
                tls.record=None
                record.update(wall_s=time.perf_counter()-start,thread_cpu_s=time.thread_time()-cpu)
                record.pop('fields');rows.append(record)
        client._finish_generation=finish
        return client

    def submit(executor,fn,*a,**kw):
        queued=time.perf_counter();name=executor._thread_name_prefix
        def job():
            start=time.perf_counter();cpu=time.thread_time()
            info=dict(pool=name,thread_id=threading.get_native_id(),queued=queued,start=start,
                torch_intraop_threads=torch.get_num_threads(),affinity=len(os.sched_getaffinity(0)),
                cpu_count=os.cpu_count())
            try:return fn(*a,**kw)
            finally:
                info.update(end=time.perf_counter(),thread_cpu_s=time.thread_time()-cpu)
                with lock:jobs.append(info)
        return original_submit(executor,job)

    torch.Tensor.to=tensor_to
    native.make_local_fixed_batch_vlm_client=factory
    ThreadPoolExecutor.submit=submit
    try:runner.main(runner.pipeline_args(rest))
    finally:
        torch.Tensor.to=original_to
        native.make_local_fixed_batch_vlm_client=original_factory
        ThreadPoolExecutor.submit=original_submit
        (args.diagnostic_output/'requests.json').write_text(json.dumps(rows,indent=2)+'\n')
        (args.diagnostic_output/'pool_spans.json').write_text(json.dumps(jobs,indent=2)+'\n')
        sums=Counter();fields={}
        for row in rows:
            for key in ['wall_s','thread_cpu_s','fence_s']:sums[key]+=row[key]
            for call in row['copies']:
                key=str(call['field'])+':'+call['source_device']+'->'+call['target_device']
                rec=fields.setdefault(key,Counter())
                for f in ['bytes','wall_s','thread_cpu_s']:rec[f]+=call[f]
                rec['calls']+=1
        summary=dict(scope='diagnostic only; instrumentation/fences perturb scheduling',
            drain_before_h2d=args.drain_before_h2d,requests=len(rows),totals=dict(sums),copies=fields)
        (args.diagnostic_output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print('HOST_DIAGNOSTIC '+json.dumps(summary),flush=True)


if __name__=='__main__':main()
