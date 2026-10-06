"""Pinned NanoBEIR 90M reranker, logical B32 preparation and B4 NPU workers."""
import argparse
import ast
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
import numpy as np
import torch
from run_cpu_reference import CReference, CHECKPOINT_SHA256, sha256
from run_nanobeir import REFERENCE
from run_nanoscidocs import ndcg_reference, save
from run_reranker_smoke import PREFIX, SUFFIX, require
from local_modeling_rwkv_embedding import Embedding
from local_modeling_rwkv_reranker import Reranker, CHECKPOINT_SHA256 as RERANKER_SHA256
from probe_reranker_endpoint import Backbone, compiled
from probe_wkv7 import load_bridge
from wkv7_endpoint import load_endpoint, register_converter


def progress(path, value):
    temp=path.with_suffix('.new');save(temp,value);temp.replace(path)


def prepare(args, report):
    """Flatten each task before padding; groups may cross query/shard boundaries."""
    manifest=json.loads((args.data_root/'manifest.json').read_text())
    assert manifest['mteb_version']=='1.38.60' and {t['task'] for t in manifest['tasks']}==set(REFERENCE)
    assert sum(t['query_count'] for t in manifest['tasks'])==649
    baseline=json.loads((args.candidates_root/'result.json').read_text());assert baseline['all_checks_passed']
    assert sha256(args.data_root/'manifest.json')==baseline['manifest_sha256']
    pin=json.loads((Path(__file__).parent/'data/reranker_padding_sources.json').read_text())
    wrapper=args.upstream/'wrapper.py';assert sha256(wrapper)==pin['files']['embedding/reranker/src/wrapper.py']['sha256']
    assert sha256(args.upstream/'mteb_retrieval_evaluator.py')==pin['mteb_retrieval_evaluator_sha256']
    tree=ast.parse(wrapper.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TokenizerWrapper')
    namespace={'torch':torch,'EOS_INDEX':65535}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(wrapper),'exec'),namespace)
    tokenizer=CReference(args.runtime,4)
    class Tokens:
        def encode(self,text):return tokenizer.tokenize(text).tolist()
    wrapper_tokenizer=namespace['TokenizerWrapper'](Tokens(),PREFIX+SUFFIX)
    jobs=[];report['tasks']=[]
    tasks=sorted(manifest['tasks'],key=lambda t:t['task']!='NanoSCIDOCSRetrieval')
    try:
        for task in tasks:
            name=task['task'];out=args.output/'prepared'/name;out.mkdir(parents=True)
            data_path=args.data_root/task['data_file'];assert sha256(data_path)==task['data_sha256']
            data=json.loads(data_path.read_text());assert data['revision']==task['revision']
            corpus={r['_id']:r['text'].strip() for r in data['corpus']}
            queries={r['_id']:r['text'] for r in data['queries']}
            assert len(queries)==task['query_count'] and len(corpus)==task['corpus_count']
            cp=args.candidates_root/name/'top100.json';candidates=json.loads(cp.read_text())
            prior=json.loads((args.candidates_root/name/'result.json').read_text())
            assert sha256(cp)==prior['artifacts']['top100.json']['sha256']
            assert set(candidates)==set(queries)
            pairs=[]
            for qid in queries:
                docs=candidates[qid]
                assert len(docs)==100 and len({d['document_id'] for d in docs})==100
                assert all(d['document_id'] in corpus and np.isfinite(d['score']) for d in docs)
                assert docs==sorted(docs,key=lambda d:d['score'],reverse=True)
                pairs.extend((qid,d['document_id']) for d in docs)
            arrays=[];lengths=[]
            # Outer 128 is divisible by wrapper B32; same boundaries as MTEB.
            for offset in range(0,len(pairs),32):
                group=pairs[offset:offset+32]
                rows=wrapper_tokenizer([(queries[q],corpus[d]) for q,d in group],return_tensors='list')
                assert len({len(r) for r in rows})==1 and all(r[-1]==65535 and 0<len(r)<=2048 for r in rows)
                arrays.append(np.asarray(rows,dtype=np.int32).reshape(-1));lengths.extend(map(len,rows))
            flat=np.concatenate(arrays);offsets=np.concatenate(([0],np.cumsum(lengths)))
            np.save(out/'tokens.npy',flat);np.save(out/'offsets.npy',offsets)
            for i,qid in enumerate(queries):
                lo=i*100;hi=lo+100;ls=lengths[lo:hi]
                assert all(len(set(ls[k:k+4]))==1 for k in range(0,100,4))
                cost=sum(.03415 if L<=512 else .080+.000006*L for L in ls[::4])
                digest=hashlib.sha256(flat[offsets[lo]:offsets[hi]].tobytes()+np.asarray(ls,dtype=np.int32).tobytes()).hexdigest()
                jobs.append(dict(task=name,query_id=qid,document_ids=[d for q,d in pairs[lo:hi]],
                                 row_start=lo,lengths=ls,input_sha256=digest,estimated_seconds=cost))
            report['tasks'].append(dict(**task,candidates_sha256=sha256(cp),
                prepared_files_sha256={f.name:sha256(f) for f in out.iterdir()},
                prepared_pairs=len(pairs),short_pairs=sum(L<=512 for L in lengths),
                length_min=min(lengths),length_max=max(lengths)))
            print('PREPARED',json.dumps(report['tasks'][-1]),flush=True)
    finally:tokenizer.close()
    assert len(jobs)==649
    loads=[0.]*len(args.devices);shards=[[] for _ in args.devices]
    for job in sorted(jobs,key=lambda j:j['estimated_seconds'],reverse=True):
        i=min(range(len(loads)),key=loads.__getitem__);loads[i]+=job['estimated_seconds'];shards[i].append(job)
    task_order={t['task']:i for i,t in enumerate(tasks)}
    query_order={(j['task'],j['query_id']):i for i,j in enumerate(jobs)}
    for i,shard in enumerate(shards):
        shard.sort(key=lambda j:(task_order[j['task']],query_order[j['task'],j['query_id']]))
        save(args.output/f'jobs_{i}.json',shard)
    report.update(dataset_manifest_sha256=sha256(args.data_root/'manifest.json'),
        baseline_mean_ndcg_at_10=baseline['mean_ndcg_at_10'],pairs=64900,
        shard_queries=list(map(len,shards)),estimated_shard_forward_seconds=loads,
        jobs_sha256={str(i):sha256(args.output/f'jobs_{i}.json') for i in range(len(shards))})


def worker(args):
    import torch_npu
    torch.set_num_threads(4)
    i=args.worker_index;out=args.output/f'worker_{i}';out.mkdir()
    report={'all_checks_passed':False,'physical_npu':os.environ['ASCEND_RT_VISIBLE_DEVICES'],'dense_dtype':args.dtype,'state_dtype':'fp32','cases':[]}
    start=time.perf_counter()
    try:
        status=subprocess.check_output(['/usr/local/bin/npu-status'],text=True)
        selected=next(s for s in status.splitlines() if s.startswith('NPU '+report['physical_npu']+': '))
        assert ': free ' in selected and 'Health=OK' in selected,selected
        torch.npu.set_device(0);torch.npu.set_compile_mode(jit_compile=False);torch.npu.config.allow_internal_format=False
        torch.npu.matmul.allow_hf32=False
        free,total=torch.npu.mem_get_info();assert free>3*1024**3
        report.update(device=torch.npu.get_device_name(0),hbm_before=dict(free_bytes=free,total_bytes=total),device_status=status,
                      torch=torch.__version__,torch_npu=torch_npu.__version__,matmul_allow_hf32=torch.npu.matmul.allow_hf32)
        root=Path(__file__).parent;load_bridge(root/'wkv7_npu',args.reference_build);load_endpoint(args.build_root);register_converter()
        dtype={'fp16':torch.float16,'fp32':torch.float32}[args.dtype]
        model=Embedding(args.checkpoint,'npu:0',dtype);ranker=Reranker(args.reranker,'npu:0',dtype)
        backbone=Backbone(model).eval()
        jobs=json.loads((args.output/f'jobs_{i}.json').read_text())
        # Each process owns fresh copies; no shared TorchAir cache writers.
        for n in ['backbone_cache','head_cache']:shutil.copytree(args.warm_cache_from/n,out/n)
        encode=compiled(backbone.forward,out/'backbone_cache');head=compiled(ranker.forward,out/'head_cache')
        loaded={}
        def rows(job,offset):
            name=job['task']
            if name not in loaded:
                p=args.output/'prepared'/name
                loaded[name]=(np.load(p/'tokens.npy',mmap_mode='r'),np.load(p/'offsets.npy',mmap_mode='r'))
            flat,offs=loaded[name];j=job['row_start']+offset
            return np.stack([flat[offs[k]:offs[k+1]] for k in range(j,j+4)])
        def score(cpu_ids):
            L=cpu_ids.shape[1]
            if L<=512:
                padded=np.pad(cpu_ids,((0,0),(0,512-L)))
                ni=torch.from_numpy(padded.astype(np.int64)).to('npu');lens=torch.full((4,),L,dtype=torch.int32,device='npu')
                return head(encode(ni,lens)[1]),'torchair_t512'
            ni=torch.from_numpy(cpu_ids.astype(np.int64)).to('npu')
            return ranker(model.encode_states(ni)[1][1]),'raw_eager_exact_length'
        with torch.inference_mode():
            # Actual logically padded rows, both backends if present, before scoring.
            for short in [True,False]:
                sample=next(((j,k) for j in jobs for k in range(0,100,4) if (j['lengths'][k]<=512)==short),None)
                if sample is None:continue
                job,k=sample;cpu_ids=rows(job,k);L=cpu_ids.shape[1]
                ni=torch.from_numpy(cpu_ids.astype(np.int64)).to('npu');_,batched,_=model.encode_states(ni)
                expected=ranker(batched[1]);single=[]
                for r in range(4):
                    _,s,_=model.encode_states(ni[r:r+1].contiguous());single.append(ranker(s[1]))
                single_check=require(expected,torch.cat(single),.02,.005)
                actual,backend=score(cpu_ids);check=require(actual,expected,.02,.005)
                assert torch.equal(actual,score(cpu_ids)[0]),'Repeat changed scores'
                record=dict(task=job['task'],query_id=job['query_id'],offset=k,length=L,backend=backend,
                            logits=actual.cpu().tolist(),reference_logits=expected.cpu().tolist(),
                            batch_vs_single=single_check,backend_vs_exact_eager=check)
                report['cases'].append(record);print('PREFLIGHT',json.dumps(record),flush=True)
                for _ in range(3):score(cpu_ids)
                torch.npu.synchronize()
            report['setup_and_preflight_seconds']=time.perf_counter()-start
            scoring=time.perf_counter();done=0;forward=0.;weight_done=0.
            with (out/'scores.jsonl').open('w') as scores,(out/'batch_timings.jsonl').open('w') as timings:
                for job in jobs:
                    if job['task'] not in loaded:
                        p=args.output/'prepared'/job['task']
                        loaded[job['task']]=(np.load(p/'tokens.npy',mmap_mode='r'),np.load(p/'offsets.npy',mmap_mode='r'))
                    flat,offs=loaded[job['task']]
                    lo=job['row_start'];hi=lo+100
                    assert hashlib.sha256(flat[offs[lo]:offs[hi]].tobytes()+np.asarray(job['lengths'],dtype=np.int32).tobytes()).hexdigest()==job['input_sha256']
                    values=[]
                    for k in range(0,100,4):
                        cpu_ids=rows(job,k);before=time.perf_counter();logits,backend=score(cpu_ids);torch.npu.synchronize()
                        elapsed=time.perf_counter()-before;forward+=elapsed
                        got=logits.cpu().tolist();assert len(got)==4 and all(np.isfinite(got));values.extend(got)
                        timings.write(json.dumps(dict(task=job['task'],query_id=job['query_id'],offset=k,length=cpu_ids.shape[1],backend=backend,forward_and_transfer_seconds=elapsed))+'\n')
                    scores.write(json.dumps(dict(task=job['task'],query_id=job['query_id'],scores=dict(zip(job['document_ids'],values))))+'\n');scores.flush()
                    done+=100;weight_done+=job['estimated_seconds'];wall=time.perf_counter()-scoring
                    record=dict(done=done,total=len(jobs)*100,elapsed_seconds=wall,pairs_per_second=done/wall,
                        forward_and_transfer_seconds=forward,estimated_remaining_seconds=wall*(sum(j['estimated_seconds'] for j in jobs)-weight_done)/weight_done,
                        task=job['task'],queries=done//100)
                    progress(out/'progress.json',record)
                    if done==100 or done%1000==0 or done==len(jobs)*100:print('PROGRESS',json.dumps(record),flush=True)
            report.update(scoring_wall_seconds=time.perf_counter()-scoring,pairs=done,forward_and_transfer_seconds=forward,all_checks_passed=True)
    except Exception as error:report['error']=f'{type(error).__name__}: {error}';raise
    finally:
        report['total_seconds']=time.perf_counter()-start
        if torch.npu.is_initialized():report['peak_hbm_bytes']=dict(allocated=torch.npu.max_memory_allocated(),reserved=torch.npu.max_memory_reserved())
        save(out/'result.json',report)


def coordinate(args):
    import pytrec_eval
    start=time.perf_counter();root=Path(__file__).parent;args.output.mkdir(parents=True,exist_ok=False)
    report=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256={n:sha256(root/n) for n in ['run_nanobeir_reranker.py','probe_reranker_endpoint.py','local_modeling_rwkv_embedding.py','local_modeling_rwkv_reranker.py','wkv7_endpoint.py','run_nanoscidocs.py']},
        checkpoint_sha256=sha256(args.checkpoint),reranker_sha256=sha256(args.reranker),devices=args.devices,
        dense_dtype=args.dtype,state_dtype='fp32',logical_batch_size=32,outer_batch_size=128,device_batch_size=4,
        preparation='Task-wide query order and saved candidate rank order; pinned wrapper B32 left padding/last2048/EOS; additional right padding only for compiled T512.',
        backend_policy='TorchAir T512 for prepared length<=512; exact-length raw eager for longer inputs',
        candidate_source=str(args.candidates_root),document_caching=False,task_results=[],all_checks_passed=False)
    children=[]
    try:
        assert len(args.devices)==2 and len(set(args.devices))==2
        assert report['checkpoint_sha256']==CHECKPOINT_SHA256 and report['reranker_sha256']==RERANKER_SHA256
        gates=json.loads(args.batch_evidence.read_text())
        for run in gates['runs']:
            p=root.parent/run['result_path'];assert sha256(p)==run['result_sha256']
            previous=json.loads(p.read_text());assert previous['all_checks_passed']
            for n in ['local_modeling_rwkv_embedding.py','local_modeling_rwkv_reranker.py','wkv7_endpoint.py']:
                assert report['source_sha256'][n]==previous['source_sha256'][n]
        report['batch_gate_sha256']=sha256(args.batch_evidence)
        cache=json.loads((args.warm_cache_from/'result.json').read_text())
        assert cache['all_checks_passed'] and cache['dtype']==args.dtype and cache['bucket']==512 and cache['batch_size']==4
        assert cache['backend']=='torchair'
        for n in ['local_modeling_rwkv_embedding.py','local_modeling_rwkv_reranker.py','wkv7_endpoint.py']:
            assert report['source_sha256'][n]==cache['source_sha256'][n]
        assert cache['checkpoint_sha256']=={'checkpoint':report['checkpoint_sha256'],'reranker':report['reranker_sha256']}
        report['warm_cache_reference_sha256']=sha256(args.warm_cache_from/'result.json')
        report['historical_batch_probe_sha256']=sorted({json.loads((root.parent/r['result_path']).read_text())['source_sha256']['probe_reranker_endpoint.py'] for r in gates['runs']})
        prepare(args,report);report['preparation_seconds']=time.perf_counter()-start;save(args.output/'result.json',report)
        launch=time.perf_counter();logs=[]
        for i,device in enumerate(args.devices):
            env=os.environ.copy();env['ASCEND_RT_VISIBLE_DEVICES']=str(device)
            log=(args.output/f'worker_{i}.log').open('w');logs.append(log)
            argv=[sys.executable,'-u',str(Path(__file__).resolve()),*sys.argv[1:],'--worker-index',str(i)]
            child=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT);children.append(child)
            save(args.output/f'worker_{i}_command.json',dict(argv=argv,physical_npu=device,pid=child.pid))
        previous=-1
        while any(c.poll() is None for c in children):
            if any(c.poll() not in [None,0] for c in children):raise RuntimeError('Worker failed; inspect worker logs/results')
            states=[json.loads(p.read_text()) for i in range(len(children)) if (p:=args.output/f'worker_{i}/progress.json').exists()]
            done=sum(s['done'] for s in states)
            if done!=previous and (done==0 or done-previous>=1000):
                wall=time.perf_counter()-launch
                record=dict(done=done,total=64900,worker_elapsed_seconds=wall,run_elapsed_seconds=time.perf_counter()-start,
                    aggregate_pairs_per_second=done/wall,estimated_remaining_seconds=max((s['estimated_remaining_seconds'] for s in states),default=None),workers=states)
                progress(args.output/'progress.json',record);print('PROGRESS',json.dumps(record),flush=True);previous=done
            time.sleep(5)
        for c,log in zip(children,logs):assert c.wait()==0;log.close()
        report['workers']=[json.loads((args.output/f'worker_{i}/result.json').read_text()) for i in range(2)]
        assert all(w['all_checks_passed'] for w in report['workers'])
        report['workers_wall_seconds']=time.perf_counter()-launch
        scores={t['task']:{} for t in report['tasks']}
        for i in range(2):
            for line in (args.output/f'worker_{i}/scores.jsonl').read_text().splitlines():
                row=json.loads(line);assert row['query_id'] not in scores[row['task']]
                scores[row['task']][row['query_id']]=row['scores']
        for task in report['tasks']:
            name=task['task'];data=json.loads((args.data_root/task['data_file']).read_text());candidates=json.loads((args.candidates_root/name/'top100.json').read_text())
            assert set(scores[name])==set(candidates) and all(set(scores[name][q])=={d['document_id'] for d in candidates[q]} for q in candidates)
            qrels={q:{} for q in candidates}
            for r in data['qrels']:qrels[r['query-id']][r['corpus-id']]=1
            retrieval={q:{d['document_id']:d['score'] for d in ds} for q,ds in candidates.items()}
            evaluator=pytrec_eval.RelevanceEvaluator(qrels,{'ndcg_cut.10'})
            measured=evaluator.evaluate(scores[name]);base=evaluator.evaluate(retrieval)
            for results,metrics in [(scores[name],measured),(retrieval,base)]:
                manual=ndcg_reference(qrels,results)
                assert set(metrics)==set(candidates) and all(abs(metrics[q]['ndcg_cut_10']-manual[q])<1e-12 for q in candidates)
            out=args.output/name;out.mkdir()
            with gzip.open(out/'pair_scores.json.gz','wt') as f:json.dump(scores[name],f)
            save(out/'per_query_scores.json',measured)
            ndcg=sum(v['ndcg_cut_10'] for v in measured.values())/len(candidates)*100
            baseline=sum(v['ndcg_cut_10'] for v in base.values())/len(candidates)*100
            row=dict(task=name,queries=len(candidates),pairs=len(candidates)*100,ndcg_at_10=ndcg,embedding_ndcg_at_10=baseline,delta_embedding_points=ndcg-baseline,metric_crosscheck_passed=True)
            save(out/'result.json',row);report['task_results'].append(row);print('TASK_RESULT',json.dumps(row),flush=True)
        assert sum(t['pairs'] for t in report['task_results'])==64900
        report.update(mean_ndcg_at_10=sum(t['ndcg_at_10'] for t in report['task_results'])/13,paper_mean_ndcg_at_10=63.41,
            all_checks_passed=True)
        report['delta_paper_points']=report['mean_ndcg_at_10']-63.41
    except Exception as error:report['error']=f'{type(error).__name__}: {error}';raise
    finally:
        for child in children:
            if child.poll() is None:child.terminate();child.wait(timeout=30)
        report['total_seconds']=time.perf_counter()-start;report['under_15_minutes']=report['total_seconds']<900
        save(args.output/'result.json',report);print('SUMMARY',json.dumps({k:v for k,v in report.items() if k not in ['tasks','source_sha256','workers']}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['checkpoint','reranker','runtime','upstream','build-root','reference-build','data-root','candidates-root','batch-evidence','warm-cache-from','output']:
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--dtype',choices=['fp16','fp32'],default='fp16')
    p.add_argument('--devices',type=int,nargs=2,required=True)
    p.add_argument('--worker-index',type=int,choices=[0,1])
    args=p.parse_args()
    worker(args) if args.worker_index is not None else coordinate(args)


if __name__=='__main__':main()
