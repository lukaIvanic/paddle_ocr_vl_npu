"""Summarize diagnostic spans; never treat instrumented wall time as throughput."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics


def union_duration(spans):
    end=None;total=0.0
    for left,right in sorted(spans):
        total+=max(0.0,right-max(left,end if end is not None else left))
        end=max(right,end if end is not None else right)
    return total


def summarize(path):
    diagnostic=path/'diagnostic'
    rows=json.loads((diagnostic/'requests.json').read_text())
    jobs=json.loads((diagnostic/'pool_spans.json').read_text())
    pools={}
    for name in sorted({j['pool'] for j in jobs}):
        selected=[j for j in jobs if j['pool']==name]
        pools[name]=dict(jobs=len(selected),worker_busy_wall_s=sum(j['end']-j['start'] for j in selected),
            any_worker_busy_wall_s=union_duration([(j['start'],j['end']) for j in selected]),
            worker_thread_cpu_s=sum(j['thread_cpu_s'] for j in selected),
            native_threads=sorted({j['thread_id'] for j in selected}),
            affinity_cpu_counts=sorted({j['affinity'] for j in selected}),
            cpu_counts=sorted({j['cpu_count'] for j in selected}),
            torch_intraop_threads=sorted({j['torch_intraop_threads'] for j in selected}))
    copies=[c for r in rows for c in r['copies'] if c['field']=='pixel_values']
    fit=None;bins=[]
    if len(copies)>1:
        x=[c['bytes']/1e6 for c in copies];y=[c['wall_s']*1e3 for c in copies]
        mx=statistics.mean(x);my=statistics.mean(y)
        xx=sum((v-mx)**2 for v in x);yy=sum((v-my)**2 for v in y)
        xy=sum((a-mx)*(b-my) for a,b in zip(x,y))
        if xx and yy:fit=dict(slope_ms_per_MB=xy/xx,intercept_ms=my-xy/xx*mx,r_squared=xy*xy/xx/yy)
        grouped=defaultdict(list)
        for c in copies:
            mb=c['bytes']/1e6
            edge=next((e for e in [1,2,4,8,16,32] if mb<=e),float('inf'))
            grouped[edge].append(c)
        for edge,cs in sorted(grouped.items()):
            bins.append(dict(upper_MB=edge,requests=len(cs),mean_MB=statistics.mean(c['bytes']/1e6 for c in cs),
                median_wall_ms=statistics.median(c['wall_s']*1e3 for c in cs),
                median_thread_cpu_ms=statistics.median(c['thread_cpu_s']*1e3 for c in cs),
                total_wall_s=sum(c['wall_s'] for c in cs)))
    sampling=None
    profile=path/'native_stacks.txt'
    if profile.exists():
        counts=Counter();tops=Counter();waiting=Counter()
        for line in profile.read_text().splitlines():
            stack,count=line.rsplit(' ',1);count=int(count);counts['all_thread_samples']+=count
            if 'run_decode_stream (' in stack:
                counts['npu_owner_pipeline_samples']+=count
                tops[stack.split(';')[-1]]+=count
                if 'take_gil' in stack:
                    counts['npu_owner_take_gil_samples']+=count
                    waiting[stack.split(';')[-1]]+=count
                    if any(s in stack for s in ['pthread_cond','futex','__futex']):counts['npu_owner_gil_wait_samples']+=count
            if '_prepare_cpu (' in stack:counts['prep_worker_samples']+=count
            if any(s in stack for s in ['_layout_job (','_expand (','crops (']):counts['frontend_worker_samples']+=count
        sampling=dict(counts=counts,top_pipeline_owner_leaves=tops.most_common(20),take_gil_leaves=waiting.most_common(10),
            interpretation='49 Hz native/idle sampling; take_gil with futex/condition wait identifies sampled main-thread GIL waits. Worker samples alone do not establish GIL ownership or causality. Counts across threads are not additive wall time.')
    return dict(lane=str(path),scope='Diagnostic only; profiler/hooks/fences perturb scheduling',
        transfers=json.loads((diagnostic/'summary.json').read_text()),pools=pools,pixel_copy_size_fit=fit,pixel_copy_size_bins=bins,
        size_fit_limit='Observational per-request fit, not a controlled bandwidth experiment; includes host allocation/cast/GIL effects.',sampling=sampling)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('lanes',type=Path,nargs='+')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();result=[summarize(p) for p in args.lanes]
    content=json.dumps(result,indent=2)+'\n'
    if args.output:
        with args.output.open('x') as f:f.write(content)
    print(content)


if __name__=='__main__':main()
