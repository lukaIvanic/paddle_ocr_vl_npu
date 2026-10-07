"""Real full-task cached accuracy across globally selectable batch sizes."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from bench_cached_overlap import Pipeline
from run_cached_suites import device_engine,save,emit,identity
from run_cached_nanobeir import load_task
from run_nanobeir_reranker import bm25_reference
from run_reranker_smoke import require


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['output','prepared','reference','data-root','models','reference-build','build-root','vector-build']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--local-cpus',default=','.join(map(str,range(24,40))))
    p.add_argument('--batches',nargs='+',type=int,default=[32,16,4])
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);e=device_engine(a);_,metric=bm25_reference(a)
    data=load_task(a,'NanoNQRetrieval');folder=next(a.reference.glob('worker_*/NanoNQRetrieval'))
    cache=np.load(folder/'document_states.npy',mmap_mode='r');index={d:i for i,d in enumerate(sorted(data['documents']))}
    old={j['query_id']:j for j in map(json.loads,(folder/'scores.jsonl').read_text().splitlines())}
    pairs=[(index[d],data['queries'][j['query_id']],j['query_id'],d) for j in data['jobs'] for d in j['corpus_ids']]
    assert max(len(x[1]) for x in pairs)<=64
    result=dict(**identity(),tasks=[])
    with torch.inference_mode():
        for b in a.batches:
            pipe=Pipeline(e,cache,b,set(map(int,a.local_cpus.split(','))))
            batches=[]
            for off in range(0,len(pairs),b):
                rows=pairs[off:off+b];batches.append(rows+rows[-1:]*(b-len(rows)))
            try:
                for _ in range(2):pipe.run(batches[:1],'torchair',False)
                begin=time.perf_counter();values,_=pipe.run(batches,'torchair',True);values=values[:len(pairs)]
                scores={}
                for row,v in zip(pairs,values.tolist()):scores.setdefault(row[2],{})[row[3]]=v
                query=[]
                for j in data['jobs']:
                    q=j['query_id'];got=[scores[q][d] for d in j['corpus_ids']];ref=old[q]['scores']['cached']
                    check=require(torch.tensor(got),torch.tensor(ref),.02,.005)
                    query.append(dict(query=q,ndcg=metric(j['labels'],got),previous_ndcg=metric(j['labels'],ref),**check))
                current=100*np.mean([q['ndcg'] for q in query]);previous=100*np.mean([q['previous_ndcg'] for q in query])
                row=dict(batch=b,ndcg=current,previous_ndcg=previous,delta_pp=current-previous,seconds=time.perf_counter()-begin,queries=query)
                result['tasks'].append(row);save(a.output/'result.json',result);save(a.output/f'scores_b{b}.json',scores)
                emit('ACCURACY',dict(batch=b,ndcg=current,previous_ndcg=previous,delta_pp=current-previous,changed_queries=[q for q in query if q['ndcg']!=q['previous_ndcg']]))
            finally:pipe.close()
    result['all_checks_passed']=True;save(a.output/'result.json',result)


if __name__=='__main__':main()
