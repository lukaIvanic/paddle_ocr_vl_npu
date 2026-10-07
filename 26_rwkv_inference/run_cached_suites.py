"""Validate cached overlap on NanoBEIR; evaluate pinned Qwen candidate suites."""
import argparse
from collections import defaultdict
import gzip
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
import numpy as np
import torch
from bench_cached_overlap import Pipeline
from run_cached_nanobeir import Engine, load_task as load_nano, pack
from run_cpu_reference import CReference, sha256
from run_reranker_smoke import PREFIX, SUFFIX, require
from run_nanobeir_reranker import bm25_reference, tie_ndcg

QWEN=Path(__file__).resolve().parents[1]/'22_qwen3_embedding_benchmark'
sys.path.insert(0,str(QWEN))


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.new')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def emit(name,value):print(name,json.dumps(value),flush=True)


def identity():
    return dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources={n:sha256(Path(__file__).parent/n) for n in ['run_cached_suites.py','bench_cached_overlap.py',
        'run_cached_nanobeir.py','local_modeling_rwkv_embedding.py','local_modeling_rwkv_reranker.py']})


def task_data(name,suite):
    import mteb
    from protocol import validate_task
    from suite_protocol import ENGLISH
    from run_english_suite import load_task
    assert importlib.metadata.version('mteb')=='1.38.9'
    module=importlib.import_module('mteb.abstasks.AbsTaskRetrieval');original=module.load_dataset
    def cached_loader(repo,*args,**kwargs):
        if repo in {v[0] for v in ENGLISH.values()} and not args and kwargs.get('name') is None:kwargs['name']='default'
        return original(repo,*args,**kwargs)
    module.load_dataset=cached_loader
    try:
        if suite=='english':task,_=load_task(name,SimpleNamespace(state={}))
        else:task=mteb.get_tasks(tasks=[name])[0];validate_task(task);task.load_data()
    finally:module.load_dataset=original
    split='test' if suite=='english' else 'dev'
    return task,task.corpus[split],task.queries[split],task.relevant_docs[split]


def prepare(a):
    from mteb.evaluation.evaluators.RetrievalEvaluator import corpus_to_str
    refs=json.loads(a.baseline.read_text());expected={'english':882000,'chinese':3974000}
    a.output.mkdir(parents=True,exist_ok=True);c=CReference(a.runtime,4)
    report=dict(**identity(),baseline_sha256=sha256(a.baseline),tasks=[],policy='Full unpadded RWKV document prefix without EOS, query+EOS; no truncation; saved Qwen candidates and original judgments.')
    try:
        for ref in refs['tasks']:
            name=ref['task'];suite=ref['suite'];path=a.output/(name+'.json.gz');meta=a.output/(name+'.meta.json')
            if meta.exists():
                row=json.loads(meta.read_text());assert sha256(path)==row['sha256'];report['tasks'].append(row);continue
            begin=time.perf_counter();task,corpus,queries,qrels=task_data(name,suite)
            cp=a.candidates/suite/(f'{name}_default_predictions.json' if suite=='chinese' else f'{name}/mteb/{name}_default_predictions.json')
            assert sha256(cp)==ref['candidate_sha256'];candidates=json.loads(cp.read_text())
            assert set(candidates)==set(queries)==set(qrels)
            assert len(queries)==ref['baseline']['queries']
            jobs=[];used=set()
            for qid in sorted(candidates):
                ds=candidates[qid];assert len(ds)==100 and all(d in corpus and math.isfinite(v) for d,v in ds.items())
                dids=sorted(ds,key=lambda d:(-ds[d],d));used.update(dids)
                jobs.append(dict(query_id=qid,corpus_ids=dids))
            qs={q:c.tokenize(SUFFIX.format(query=queries[q])).tolist()+[65535] for q in sorted(queries)}
            docs={}
            for i,d in enumerate(sorted(used)):
                text=corpus_to_str([corpus[d]])[0]
                docs[d]=c.tokenize(PREFIX.format(document=text)).tolist()
                assert docs[d] and docs[d][-1]!=65535
                if i%2000==0:save(a.output/'progress.json',dict(phase='tokenize',task=name,done=i,total=len(used),seconds=time.perf_counter()-begin))
            for j in jobs[:2]:
                q=j['query_id']
                for d in j['corpus_ids'][:2]:
                    text=corpus_to_str([corpus[d]])[0]
                    assert docs[d]+qs[q]==c.tokenize(PREFIX.format(document=text)+SUFFIX.format(query=queries[q])).tolist()+[65535]
            data=dict(task=name,suite=suite,documents=docs,queries=qs,jobs=jobs,qrels=qrels,
                ignore_identical_ids=task.ignore_identical_ids,candidates=candidates)
            tmp=path.with_suffix('.tmp')
            with gzip.open(tmp,'wt') as f:json.dump(data,f,ensure_ascii=False)
            tmp.replace(path)
            row=dict(task=name,suite=suite,queries=len(qs),pairs=len(qs)*100,documents=len(docs),
                candidate_sha256=ref['candidate_sha256'],dataset=dict(task.metadata.dataset),
                query_lengths=dict(min=min(map(len,qs.values())),max=max(map(len,qs.values()))),
                document_tokens=sum(map(len,docs.values())),max_document_tokens=max(map(len,docs.values())),
                baseline_ndcg_at_10=ref['baseline'].get('ndcg_at_10',ref['baseline'].get('metrics',{}).get('reranker',{}).get('ndcg_cut_10')),
                file=path.name,sha256=sha256(path),seconds=time.perf_counter()-begin)
            assert row['baseline_ndcg_at_10'] is not None
            save(meta,row);report['tasks'].append(row);save(a.output/'manifest.json',report);emit('PREPARED',row)
        assert {s:sum(t['pairs'] for t in report['tasks'] if t['suite']==s) for s in expected}==expected
        report['all_checks_passed']=True;save(a.output/'manifest.json',report)
    finally:c.close()


def device_engine(a):
    status=subprocess.check_output(['npu-status'],text=True)
    line=next(x for x in status.splitlines() if x.startswith('NPU '+os.environ['ASCEND_RT_VISIBLE_DEVICES']+': '))
    assert ': free ' in line and 'Health=OK' in line,line
    os.sched_setaffinity(0,set(map(int,a.local_cpus.split(','))))
    a.size='large';a.batch_size=4
    return Engine(a)


def capacity(length):
    return next((x for x in [64,128,256,512,2048] if length<=x),math.ceil(length/2048)*2048)


def score_pairs(a,e,cache,pairs,check=False):
    # Pair tuple: cache row, query tokens, query ID, document ID. Every output is
    # attached to its original IDs; synthetic final-batch rows are discarded.
    grouped=defaultdict(list)
    for pair in pairs:grouped[capacity(len(pair[1]))].append(pair)
    output=[];stats=[]
    for cap,group in sorted(grouped.items()):
        batch=min(a.max_score_batch,max(4,4096//min(cap,2048)))
        pipe=Pipeline(e,cache,batch,set(map(int,a.local_cpus.split(','))),query_capacity=cap)
        batches=[];real=[]
        for offset in range(0,len(group),batch):
            rows=group[offset:offset+batch];real.append(len(rows));batches.append(rows+rows[-1:]*(batch-len(rows)))
        try:
            # Warmup and same-input controls are outside scoring timing.
            for _ in range(2):pipe.run(batches[:1],'torchair',False)
            if check:
                chosen=[batches[0],batches[-1],batches[0]]
                serial,_=pipe.run(chosen,'torchair',False,verify=True)
                async_scores,_=pipe.run(chosen,'torchair',True,verify=True)
                assert torch.equal(serial,async_scores),'Overlap or tail-slot corruption'
                eager,_=pipe.run(chosen[:1],'raw_eager',False)
                require(serial[:batch],eager,.02,.005)
                # Independently run Engine.states on the same tokens to check
                # query bucketing, EOS selection and long-query continuation.
                previous_batch=e.batch_size;previous_calls=e.calls
                e.batch_size=batch;e.calls={}
                pipe.gather(pipe.slots[0],chosen[0]);from run_cached_nanobeir import unpack_device
                ref=e.scores([r[1] for r in chosen[0]],unpack_device(pipe.slots[0]['host'].to('npu')),backend='raw_eager').cpu()
                e.batch_size=previous_batch;e.calls=previous_calls
                require(eager,ref,.02,.005)
                emit('BUCKET_CHECK',dict(capacity=cap,batch=batch,overlap_exact=True,engine_control=True))
            cursor=0;before=time.perf_counter()
            # Bounded windows keep progress visible without growing an unbounded
            # queue; 256 batches amortize the three-slot fill and drain.
            for start in range(0,len(batches),256):
                window=batches[start:start+256];scores,timing=pipe.run(window,'torchair',True)
                for i,n in enumerate(real[start:start+256]):
                    rows=window[i]
                    output.extend((row[2],row[3],float(v)) for row,v in zip(rows[:n],scores[i*batch:i*batch+n]))
                cursor+=sum(real[start:start+256])
                save(a.output/'progress.json',dict(phase='score',bucket=cap,done=cursor,total=len(group),seconds=time.perf_counter()-before))
            stats.append(dict(bucket=cap,batch=batch,pairs=len(group),seconds=time.perf_counter()-before))
        finally:pipe.close()
    assert len(output)==len(pairs) and all(math.isfinite(x[2]) for x in output)
    return output,stats


def nano(a):
    a.output.mkdir(parents=True,exist_ok=False);e=device_engine(a)
    manifest=json.loads((a.prepared/'manifest.json').read_text());_,metric=bm25_reference(a)
    results=[]
    with torch.inference_mode():
        for item in manifest['tasks'][a.worker_index::2]:
            name=item['task'];data=load_nano(a,name)
            folders=list(a.reference.glob('worker_*/'+name));assert len(folders)==1
            folder=folders[0];cache=np.load(folder/'document_states.npy',mmap_mode='r')
            docs=sorted(data['documents']);index={d:i for i,d in enumerate(docs)}
            assert cache.shape==(len(docs),3244032)
            old={j['query_id']:j for j in map(json.loads,(folder/'scores.jsonl').read_text().splitlines())}
            pairs=[]
            for j in data['jobs']:
                q=j['query_id'];assert j['corpus_ids']==old[q]['corpus_ids'] and j['labels']==old[q]['labels']
                pairs.extend((index[d],data['queries'][q],q,d) for d in j['corpus_ids'])
            # Check the document-state builder used by the full suites against
            # accepted cache rows, including long-prefix continuation.
            state_check=[]
            longest=max(range(len(docs)),key=lambda i:len(data['documents'][docs[i]]))
            for offset in sorted({0,(len(docs)//2)//4*4,longest//4*4,(len(docs)-1)//4*4}):
                ds=docs[offset:offset+4];rows=[data['documents'][d] for d in ds]
                rows+=rows[-1:]*(4-len(rows))
                fresh=pack(e.states(rows,document=True))[:len(ds)]
                state_check.append(require(torch.from_numpy(fresh),
                    torch.from_numpy(np.array(cache[offset:offset+len(ds)])),.02,.005))
            scored,timing=score_pairs(a,e,cache,pairs,check=True)
            got=defaultdict(dict)
            for q,d,s in scored:assert d not in got[q];got[q][d]=s
            current=[];prior=[];delta=[]
            for j in data['jobs']:
                q=j['query_id'];values=[got[q][d] for d in j['corpus_ids']];ref=old[q]['scores']['cached']
                require(torch.tensor(values),torch.tensor(ref),.02,.005)
                value=metric(j['labels'],values);assert abs(value-tie_ndcg(j['labels'],values))<1e-12
                current.append(value);prior.append(metric(j['labels'],ref));delta.extend(abs(x-y) for x,y in zip(values,ref))
            row=dict(task=name,queries=len(current),pairs=len(pairs),ndcg=100*np.mean(current),previous_ndcg=100*np.mean(prior),
                delta_pp=100*(np.mean(current)-np.mean(prior)),max_logit_delta=max(delta),state_builder_check=state_check,timing=timing)
            results.append(row);save(a.output/(name+'.scores.json'),scored);emit('NANO_TASK',row)
            save(a.output/'result.json',dict(**identity(),tasks=results,all_checks_passed=False))
            assert abs(row['delta_pp'])<=.2,(name,row)
    save(a.output/'result.json',dict(**identity(),tasks=results,all_checks_passed=True))


def aggregate_nano(a):
    reports=[json.loads((a.output/f'worker_{i}/result.json').read_text()) for i in range(2)]
    assert all(r['all_checks_passed'] and r['sources']==identity()['sources'] for r in reports)
    rows=sum([r['tasks'] for r in reports],[])
    assert len(rows)==len({r['task'] for r in rows})==11 and sum(r['pairs'] for r in rows)==57688
    now=float(np.mean([r['ndcg'] for r in rows]));old=float(np.mean([r['previous_ndcg'] for r in rows]))
    assert abs(now-old)<=.1 and all(abs(r['delta_pp'])<=.2 for r in rows)
    result=dict(**identity(),all_checks_passed=True,mean_ndcg=now,previous_mean_ndcg=old,delta_pp=now-old,tasks=rows)
    save(a.output/'result.json',result);emit('NANO_GATE',result)


def load_suite(a,name):
    manifest=json.loads((a.prepared/'manifest.json').read_text());assert manifest['all_checks_passed']
    meta=next(t for t in manifest['tasks'] if t['task']==name)
    path=a.prepared/meta['file'];assert sha256(path)==meta['sha256']
    with gzip.open(path,'rt') as f:return json.load(f),meta


def suite_worker(a):
    gate=json.loads((a.gate/'result.json').read_text());assert gate['all_checks_passed'] and gate['sources']==identity()['sources']
    a.output.mkdir(parents=True,exist_ok=True);e=device_engine(a)
    manifest=json.loads((a.prepared/'manifest.json').read_text())
    with torch.inference_mode():
        for meta in [t for t in manifest['tasks'] if t['suite']==a.suite]:
            name=meta['task'];data,meta=load_suite(a,name);docs=data['documents'];queries=data['queries']
            ids=sorted(docs);bydoc=defaultdict(list)
            for j in data['jobs']:
                for d in j['corpus_ids']:bydoc[d].append(j['query_id'])
            folder=a.output/name;folder.mkdir(exist_ok=True)
            for block,off in enumerate(range(0,len(ids),a.document_block)):
                if block%2!=a.worker_index:continue
                target=folder/f'block_{block:05d}.json';chosen=ids[off:off+a.document_block]
                if target.exists():
                    prior=json.loads(target.read_text());assert prior['documents']==chosen and prior['input_sha256']==meta['sha256'] and prior['sources']==identity()['sources'];continue
                begin=time.perf_counter();cache=np.empty((len(chosen),3244032),dtype=np.float32)
                for k in range(0,len(chosen),4):
                    ds=chosen[k:k+4];rows=[docs[d] for d in ds];rows+=rows[-1:]*(4-len(rows))
                    cache[k:k+len(ds)]=pack(e.states(rows,document=True))[:len(ds)]
                    if k%100==0:save(a.output/'progress.json',dict(suite=a.suite,task=name,phase='cache_build',block=block,blocks=math.ceil(len(ids)/a.document_block),done=k+len(ds),total=len(chosen),seconds=time.perf_counter()-begin))
                build=time.perf_counter()-begin
                pairs=[(i,queries[q],q,d) for i,d in enumerate(chosen) for q in bydoc[d]]
                # Deterministic order; preserve all original candidate IDs.
                pairs.sort(key=lambda p:(len(p[1]),p[2],p[3]))
                scored,timing=score_pairs(a,e,cache,pairs,check=True)
                row=dict(**identity(),task=name,block=block,documents=chosen,input_sha256=meta['sha256'],
                    pairs=scored,cache_build_seconds=build,scoring=timing,seconds=time.perf_counter()-begin)
                save(target,row);emit('BLOCK_DONE',dict(task=name,block=block,documents=len(chosen),pairs=len(scored),cache_build_seconds=build,scoring=timing))
                del cache
            save(folder/'done.json',dict(input_sha256=meta['sha256'],all_checks_passed=True))
    save(a.output/'done.json',dict(**identity(),suite=a.suite,all_checks_passed=True))


def aggregate_suite(a):
    from run_reranker_evaluation import metric_summary
    manifest=json.loads((a.prepared/'manifest.json').read_text());rows=[]
    assert all(json.loads((a.output/f'worker_{i}/done.json').read_text())['all_checks_passed'] for i in range(2))
    for meta in [t for t in manifest['tasks'] if t['suite']==a.suite]:
        data,meta=load_suite(a,meta['task']);name=meta['task'];pred=defaultdict(dict);seen=set()
        for path in sorted(a.output.glob('worker_*/'+name+'/block_*.json')):
            block=json.loads(path.read_text());assert block['input_sha256']==meta['sha256'] and block['sources']==identity()['sources']
            assert not (seen&set(block['documents']));seen.update(block['documents'])
            for q,d,s in block['pairs']:assert d not in pred[q] and math.isfinite(s);pred[q][d]=s
        assert seen==set(data['documents']) and sum(map(len,pred.values()))==meta['pairs']
        metrics,per_query=metric_summary(pred,data['candidates'],data['qrels'],data['ignore_identical_ids'])
        ndcg=100*metrics['reranker']['ndcg_cut_10'];baseline=100*meta['baseline_ndcg_at_10']
        row=dict(task=name,queries=meta['queries'],pairs=meta['pairs'],rwkv_ndcg=ndcg,qwen4b_ndcg=baseline,delta_pp=ndcg-baseline,metrics=metrics)
        save(a.output/name/'per_query_metrics.json',per_query);save(a.output/name/'result.json',row);rows.append(row);emit('SUITE_TASK',row)
    assert len(rows)==(10 if a.suite=='english' else 8)
    result=dict(**identity(),suite=a.suite,all_checks_passed=True,tasks=rows,pairs=sum(r['pairs'] for r in rows),
        rwkv_macro_ndcg=float(np.mean([r['rwkv_ndcg'] for r in rows])),qwen4b_macro_ndcg=float(np.mean([r['qwen4b_ndcg'] for r in rows])))
    result['delta_pp']=result['rwkv_macro_ndcg']-result['qwen4b_macro_ndcg'];save(a.output/'result.json',result);emit('SUITE_RESULT',result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','nano','aggregate-nano','suite-worker','aggregate-suite'])
    for n in ['output','baseline','candidates','runtime','prepared','reference','data-root','models','reference-build','build-root','vector-build','gate']:p.add_argument('--'+n,type=Path)
    p.add_argument('--suite',choices=['english','chinese']);p.add_argument('--worker-index',type=int,choices=[0,1])
    p.add_argument('--local-cpus',default=','.join(map(str,range(24,40))))
    p.add_argument('--document-block',type=int,default=4096)
    p.add_argument('--max-score-batch',type=int,choices=[4,16,32,64],default=32)
    a=p.parse_args();globals()[a.mode.replace('-','_')](a)


if __name__=='__main__':main()
