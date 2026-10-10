"""Audit paired V2 captures; retain absolute counters and exact sample counts."""
import argparse
import csv
from collections import defaultdict
from decimal import Decimal
import json
import math
from pathlib import Path
import statistics

FIELDS=['Duration(us)','aiv_time(us)','aiv_vec_time(us)','aiv_scalar_time(us)',
        'aiv_mte2_time(us)','aiv_mte3_time(us)','aiv_vec_ratio','aiv_scalar_ratio',
        'aiv_mte2_ratio','aiv_mte3_ratio','aic_mac_time(us)','aiv_icache_miss_rate']


def stats(rows):
    result={'samples':len(rows),'blocks':sorted({int(r['Block Num']) for r in rows}),
            'cores':sorted({r['Accelerator Core'] for r in rows}),
            'shapes':sorted({r['Input Shapes'] for r in rows})}
    for field in FIELDS:
        vals=[]
        for row in rows:
            try: value=float(row.get(field,''))
            except ValueError: continue
            if math.isfinite(value): vals.append(value)
        result[field]={'mean':statistics.mean(vals) if vals else None,
                       'median':statistics.median(vals) if vals else None,'valid_samples':len(vals)}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); pooled=defaultdict(list); runs=[]
    for folder in sorted(a.root.glob('*_*_[01]')):
        phase,lane,repeat=folder.name.split('_')
        if phase not in ('direct','model'): continue
        result=json.loads((folder/'result.json').read_text()); assert result['passed']
        for summary in sorted((folder/'profiles').glob('*/summary.json')):
            meta=json.loads(summary.read_text()); steps=meta['active_forwards']
            csvpath,=summary.parent.glob('**/kernel_details.csv')
            rows=list(csv.DictReader(csvpath.open()))
            events=json.loads(csvpath.with_name('trace_view.json').read_text())
            if isinstance(events,dict): events=events['traceEvents']
            marks=[e for e in events if e.get('ph')=='X' and e.get('name','').startswith('ProfilerStep#')]
            assert len(marks)==steps
            selected_counts=[]
            for mark in marks:
                start=Decimal(str(mark['ts'])); end=start+Decimal(str(mark['dur']))
                selected=[r for r in rows if start<=Decimal(r['Start Time(us)'].strip()) and Decimal(r['Start Time(us)'].strip())+Decimal(r['Duration(us)'])<=end]
                selected_counts.append(len(selected))
            assert sum(selected_counts)==len(rows)
            v2=[r for r in rows if r['Type']=='AddLayerNormQuantV2']
            assert len(v2)==steps*(1 if phase=='direct' else 48),(folder,summary,len(v2))
            groups={'all_v2':v2}
            if phase=='model':
                groups['residual47']=[r for r in v2 if r['Name']!='AddLayerNormQuantV2']
                assert len(groups['residual47'])==47*steps
            for group,values in groups.items():
                pooled[(phase,lane,summary.parent.name,group)].extend(values)
                runs.append({'run':folder.name,'label':summary.parent.name,'group':group,
                             'steps':steps,'all_kernel_rows':len(rows),**stats(values)})
    aggregates=[{'phase':key[0],'lane':key[1],'label':key[2],'group':key[3],**stats(rows)} for key,rows in pooled.items()]
    pairs=[]
    for baseline in aggregates:
        if baseline['lane']!='baseline':continue
        candidate=next(r for r in aggregates if r['lane']=='candidate' and all(r[k]==baseline[k] for k in ('phase','label','group')))
        pairs.append({'phase':baseline['phase'],'label':baseline['label'],'group':baseline['group'],
                      'duration_change_percent':100*(candidate['Duration(us)']['mean']/baseline['Duration(us)']['mean']-1),
                      'baseline':baseline,'candidate':candidate})
    report={'coverage_audit':'passed','runs':runs,'comparisons':pairs,
            'note':'ABBA fresh-process captures on one physical 910B2. Pipeline times overlap; ratios are busy-cycle fractions, not peak throughput or HBM bandwidth.'}
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    for pair in pairs:
        print(pair['phase'],pair['label'],pair['group'],
              {lane:{k:round(pair[lane][k]['mean'],4) for k in FIELDS[:6]} for lane in ('baseline','candidate')},
              'change_percent',round(pair['duration_change_percent'],2))

if __name__=='__main__':main()
