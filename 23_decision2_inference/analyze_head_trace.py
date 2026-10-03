"""Read Ascend trace attribution; scopes overlap, kernel sums are not wall time."""
import argparse
import bisect
from collections import defaultdict
import csv
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    a=p.parse_args()
    outputs=list((a.root/'diagnostics/traces').glob('*/ASCEND_PROFILER_OUTPUT'))
    if len(outputs)!=1:
        raise ValueError('Expected exactly one worker capture')
    folder=outputs[0]
    operators=list(csv.DictReader((folder/'operator_details.csv').open()))
    aggregates=defaultdict(lambda:defaultdict(float))
    for row in operators:
        s=aggregates[row['Name']]
        s['calls']+=1
        for key in ('Host Self Duration(us)','Host Total Duration(us)','Device Total Duration(us)'):
            s[key]+=float(row[key])
    kernels=defaultdict(lambda:{'calls':0,'duration_us':0.})
    for row in csv.DictReader((folder/'kernel_details.csv').open()):
        s=kernels[row['Name']]
        s['calls']+=1
        s['duration_us']+=float(row['Duration(us)'])
    content=json.loads((folder/'trace_view.json').read_text())
    events=content['traceEvents'] if isinstance(content,dict) else content
    ranges=defaultdict(list)
    for e in events:
        if e.get('ph')=='X' and e.get('name','').startswith('decision2/'):
            ranges[e['name'],e['tid']].append((float(e['ts']),float(e['ts'])+float(e['dur'])))
    for rs in ranges.values():
        rs.sort()
    starts={k:[x[0] for x in v] for k,v in ranges.items()}
    def inside(name,event):
        key=(name,event.get('tid'))
        rs=ranges.get(key,[])
        t=float(event['ts'])
        i=bisect.bisect_right(starts.get(key,[]),t)-1
        return i>=0 and t+float(event.get('dur',0))<=rs[i][1]+1
    sync=defaultdict(lambda:defaultdict(lambda:{'calls':0,'host_duration_us':0.}))
    # Deepest matching main-thread source scope. Runtime PID is synthetic, so
    # match Linux TID + interval. Queue-thread work is kept unattributed.
    order=['positions_h2d','gather','head_fp32','pooler','pool_and_output','backbone','execute']
    for e in events:
        name=e.get('name','')
        if e.get('ph')!='X' or not name.startswith('AscendCL@') or not any(x in name for x in ('Synchronize','Memcpy')):
            continue
        scope=next((s for s in order if inside('decision2/'+s,e)),'unattributed_other_thread')
        s=sync[scope][name]
        s['calls']+=1
        s['host_duration_us']+=float(e['dur'])
    report={'source':str(folder),'warning':'Nested total durations overlap; do not add parent and child scopes. Kernel durations are not end-to-end time.',
            'annotated_ranges':{k:v for k,v in aggregates.items() if k.startswith('decision2/')},
            'sync_and_memcpy_by_source_scope':sync,
            'top_host_self_ops':sorted([{'name':k,**v} for k,v in aggregates.items()], key=lambda x:x['Host Self Duration(us)'],reverse=True)[:25],
            'top_kernels':sorted([{'name':k,**v} for k,v in kernels.items()],key=lambda x:x['duration_us'],reverse=True)[:25]}
    (a.root/'trace_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
