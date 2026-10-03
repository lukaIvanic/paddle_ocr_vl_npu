"""Match real-serving worker stages to measured HTTP windows; never add CPU and NPU times."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def summarize(rows):
    values=defaultdict(list)
    for row in rows:
        host={k:sum(v) for k,v in row['host_ms'].items()}
        for key,val in host.items():
            values['host_'+key+'_ms'].append(val)
        for key,val in row['stream_elapsed_ms'].items():
            if val is not None:
                values['stream_'+key+'_ms'].append(val)
        values['host_output_after_pooler_ms'].append(host.get('pool_and_output',0)-host.get('pooler',0))
        positions=row['host_ms'].get('positions_h2d',[])
        if positions:
            values['first_position_h2d_host_ms'].append(positions[0])
            values['remaining_position_h2d_host_ms'].append(sum(positions[1:]))
    return {'steps':len(rows),'requests':sum(r['batch_size'] for r in rows),
            'batch_sizes':[r['batch_size'] for r in rows],
            'metrics':{k:{'mean':statistics.mean(v),'median':statistics.median(v),'max':max(v),'sum':sum(v)} for k,v in values.items()}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    a=p.parse_args()
    stages=[json.loads(line) for file in (a.root/'diagnostics').glob('stages_*.jsonl') for line in file.open()]
    report={'units':'milliseconds unless specified','warning':'Stream intervals include host dispatch gaps, not just device kernel busy time. Host and stream measurements overlap; do not add them.','runs':{}}
    for name in ('uninstrumented','instrumented','profiled','after_profile','concurrency64'):
        path=a.root/(name+'.json')
        if not path.exists():
            continue
        groups=json.loads(path.read_text())['groups']
        output=[]
        for group in groups:
            if not group['label'].startswith('measured'):
                continue
            lo=group['started_at']
            hi=lo+group['wall_s']
            matching=[r for r in stages if lo<=r['time']<=hi and 'batch_size' in r]
            row={'label':group['label'],'wall_s':group['wall_s'],'pairs_s':len(group['results'])/group['wall_s'],
                 'items':len(group['results']),'worker':summarize(matching) if matching else None}
            if matching and sum(r['batch_size'] for r in matching)!=len(group['results']):
                raise ValueError(f'Client/worker count mismatch in {name}/{group["label"]}')
            row['by_batch_size']={str(b):summarize([r for r in matching if r['batch_size']==b]) for b in sorted({r['batch_size'] for r in matching})}
            output.append(row)
        report['runs'][name]=output
    (a.root/'stage_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
