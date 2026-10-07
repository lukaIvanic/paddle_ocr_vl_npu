"""Audit saved Qwen candidates and time the largest RWKV pair; no full evaluation."""
import argparse,ast,gzip,hashlib,importlib.metadata,json,math,os,random,subprocess,sys,time
from pathlib import Path
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'22_qwen3_embedding_benchmark'))

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,o):Path(p).write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n')
def prepare(a):
    import mteb,torch
    from types import SimpleNamespace
    from protocol import TASKS,validate_task
    from suite_protocol import ENGLISH
    from run_english_suite import load_task
    from mteb.evaluation.evaluators.RetrievalEvaluator import corpus_to_str
    from run_cpu_reference import CReference
    from run_reranker_smoke import PREFIX,SUFFIX
    assert importlib.metadata.version('mteb')=='1.38.9'
    a.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    refs=json.loads(a.baseline.read_text());assert {r['task'] for r in refs['tasks']}==set(TASKS)|set(ENGLISH)
    pin=json.loads((Path(__file__).parent/'data/reranker_padding_sources.json').read_text())
    wrapper=a.upstream/'wrapper.py';assert digest(wrapper)==pin['files']['embedding/reranker/src/wrapper.py']['sha256']
    node=next(n for n in ast.parse(wrapper.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='TokenizerWrapper')
    ns={'torch':torch,'EOS_INDEX':65535};exec(compile(ast.Module(body=[node],type_ignores=[]),str(wrapper),'exec'),ns)
    c=CReference(a.runtime,4)
    class Tokens:
        def encode(self,s):return c.tokenize(s).tolist()
    tokenizer=ns['TokenizerWrapper'](Tokens(),PREFIX+SUFFIX)
    report=dict(baseline_sha256=digest(a.baseline),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=digest(__file__),mteb='1.38.9',tasks=[],all_checks_passed=False,preparation='Saved candidate order; no positive injection; released RWKV document-first format, per-query logical B32 left padding then last2048/EOS, split into B4 without additional padding.',sampling='Two uniformly sampled queries per task, seed 20261007; all 100 candidates, not a quality estimate.')
    jobs=[]
    try:
        for ref in refs['tasks']:
            before=time.perf_counter();name=ref['task'];suite=ref['suite'];split='dev' if suite=='chinese' else 'test'
            cp=a.candidates/suite/(f'{name}_default_predictions.json' if suite=='chinese' else f'{name}/mteb/{name}_default_predictions.json')
            assert digest(cp)==ref['candidate_sha256'],name
            candidates=json.loads(cp.read_text())
            if suite=='english':task,counts=load_task(name,SimpleNamespace(state={}))
            else:
                task=mteb.get_tasks(tasks=[name])[0];validate_task(task);task.load_data();counts={}
            corpus,queries,qrels=task.corpus[split],task.queries[split],task.relevant_docs[split]
            assert set(candidates)==set(queries)==set(qrels),name
            assert len(candidates)==ref['baseline']['queries'],name
            assert all(len(ds)==100 and all(d in corpus for d in ds) and all(math.isfinite(v) for v in ds.values()) for ds in candidates.values()),name
            audit_s=time.perf_counter()-before;sample_qids=random.Random(20261007).sample(sorted(candidates),2)
            token_s=0.;lens=[];truncated=0
            for qid in sample_qids:
                before=time.perf_counter();dids=sorted(candidates[qid],key=lambda d:(-candidates[qid][d],d));docs=corpus_to_str([corpus[d] for d in dids]);arrays=[]
                pairs=[(queries[qid],d) for d in docs]
                for k in range(0,100,32):
                    group=pairs[k:k+32];rows=tokenizer(group,return_tensors='list')
                    assert len({len(x) for x in rows})==1 and all(x[-1]==65535 and 0<len(x)<=2048 for x in rows)
                    arrays.extend(rows)
                elapsed=time.perf_counter()-before;token_s+=elapsed
                raw_lengths=[len(c.tokenize(PREFIX.format(query=q,document=d)+SUFFIX.format(query=q,document=d)))+1 for q,d in pairs]
                truncated+=sum(L>2048 for L in raw_lengths);lens.extend(map(len,arrays))
                jobs.append(dict(task=name,suite=suite,query_id=qid,document_ids=dids,input_ids=arrays,tokenization_seconds=elapsed))
            row=dict(task=name,suite=suite,queries=len(queries),pairs=len(queries)*100,documents=len(corpus),candidate_path=str(cp),candidate_sha256=ref['candidate_sha256'],split=split,dataset=dict(task.metadata.dataset),ignore_identical_ids=task.ignore_identical_ids,audit_seconds=audit_s,sample_queries=sample_qids,sample_pairs=200,sample_preparation_seconds=token_s,sample_truncated_pairs=truncated,prepared_length_min=min(lens),prepared_length_max=max(lens),baseline_ndcg_at_10=ref['baseline'].get('ndcg_at_10',ref['baseline'].get('metrics',{}).get('reranker',{}).get('ndcg_cut_10')))
            report['tasks'].append(row);save(a.output/'audit.json',report);print('AUDITED',json.dumps(row),flush=True)
        with gzip.open(a.output/'sample.json.gz','wt') as f:json.dump(jobs,f)
        report.update(all_checks_passed=True,total_pairs=sum(r['pairs'] for r in report['tasks']),sample_pairs=sum(len(j['document_ids']) for j in jobs),sample_sha256=digest(a.output/'sample.json.gz'),total_seconds=time.perf_counter()-start)
        save(a.output/'audit.json',report)
    finally:c.close()

def measure(a):
    import numpy as np,torch,torch_npu
    from bench_reranker_sizes import pinned_pair
    from types import SimpleNamespace
    from local_modeling_rwkv_embedding import Embedding
    from local_modeling_rwkv_reranker import Reranker
    from probe_wkv7 import load_bridge
    from wkv7_endpoint import load_endpoint
    from run_reranker_smoke import require
    a.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();status=subprocess.check_output(['npu-status'],text=True)
    physical=os.environ['ASCEND_RT_VISIBLE_DEVICES'];line=next(x for x in status.splitlines() if x.startswith('NPU '+physical+': '));assert ': free ' in line and 'Health=OK' in line
    torch.set_num_threads(4);torch.npu.set_device(0);torch.npu.set_compile_mode(jit_compile=False);torch.npu.config.allow_internal_format=False;torch.npu.matmul.allow_hf32=False
    load_bridge(Path(__file__).parent/'wkv7_npu',a.reference_build);load_endpoint(a.build_root)
    emb,rank,eh,rh,*_=pinned_pair(SimpleNamespace(size='large',models=a.models));model=Embedding(emb,'npu:0',torch.float32,expected_sha256=eh);ranker=Reranker(rank,'npu:0',torch.float32,expected_sha256=rh)
    audit=json.loads((a.prepared/'audit.json').read_text());assert audit['all_checks_passed'] and digest(a.prepared/'sample.json.gz')==audit['sample_sha256']
    with gzip.open(a.prepared/'sample.json.gz','rt') as f:jobs=json.load(f)
    # One complete query per task per device; complementary queries give exact disjoint coverage.
    selected=[j for i,j in enumerate(jobs) if i%2==a.worker_index]
    report=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=digest(__file__),physical_npu=physical,device=torch.npu.get_device_name(0),checkpoint_sha256=eh,reranker_sha256=rh,audit_sha256=digest(a.prepared/'audit.json'),dtype='fp32',batch_size=4,backend='raw_eager',tasks=[],all_checks_passed=False)
    with torch.inference_mode():
        # Identical cross-device anchor, B4 versus independently scored B1 and replay.
        anchor=jobs[0]['input_ids'][:4];ni=torch.tensor(anchor,device='npu');ref=ranker(model.encode_states(ni)[1][1]);single=torch.cat([ranker(model.encode_states(ni[i:i+1].contiguous())[1][1]) for i in range(4)])
        report['batch_parity']=require(ref,single,.02,.005);assert torch.equal(ref,ranker(model.encode_states(ni)[1][1]));report['anchor_logits']=ref.cpu().tolist()
        for _ in range(3):ranker(model.encode_states(ni)[1][1])
        torch.npu.synchronize();report['setup_seconds']=time.perf_counter()-start
        for job in selected:
            before=time.perf_counter();values=[];lengths=[]
            for k in range(0,100,4):
                ids=torch.tensor(np.asarray(job['input_ids'][k:k+4],dtype=np.int64)).to('npu');values.extend(ranker(model.encode_states(ids)[1][1]).cpu().tolist());lengths.append(ids.shape[1])
            wall=time.perf_counter()-before;assert len(values)==100 and all(math.isfinite(v) for v in values)
            row=dict(task=job['task'],suite=job['suite'],query_id=job['query_id'],pairs=100,forward_transfer_output_seconds=wall,prepared_length_min=min(lengths),prepared_length_max=max(lengths),scores_sha256=hashlib.sha256(json.dumps(values).encode()).hexdigest())
            report['tasks'].append(row);save(a.output/'result.json',report);print('MEASURED',json.dumps(row),flush=True)
        report.update(all_checks_passed=True,total_seconds=time.perf_counter()-start,peak_reserved_hbm_bytes=torch.npu.max_memory_reserved());save(a.output/'result.json',report)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','measure'])
    for n in ['output','baseline','candidates','runtime','upstream','prepared','models','reference-build','build-root']:p.add_argument('--'+n,type=Path)
    p.add_argument('--worker-index',type=int,choices=[0,1]);a=p.parse_args();prepare(a) if a.mode=='prepare' else measure(a)
if __name__=='__main__':main()
