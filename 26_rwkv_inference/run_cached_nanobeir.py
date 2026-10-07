"""Released document-first cached policy: real-state gates and paired NanoBEIR evaluation."""
import argparse,gzip,json,os,subprocess,time,hashlib,statistics
from pathlib import Path
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
import numpy as np
import torch
from run_cpu_reference import CReference,sha256
from run_reranker_smoke import PREFIX,SUFFIX,require,metrics
from run_nanobeir_reranker import bm25_reference,tie_ndcg,progress
from run_nanoscidocs import save
from probe_reranker_endpoint import compiled,measure
from bench_reranker_sizes import pinned_pair
from local_modeling_rwkv_embedding import Embedding
from local_modeling_rwkv_reranker import Reranker
from probe_wkv7 import load_bridge
from wkv7_endpoint import load_endpoint,register_converter
from wkv7_vector_variants import load_variant


def prepare(a):
    a.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    old=json.loads((a.reference/'result.json').read_text());assert old['all_checks_passed'] and old['protocol']=='bm25-positives'
    manifest=json.loads((a.data_root/'manifest.json').read_text());assert sha256(a.data_root/'manifest.json')==old['dataset_manifest_sha256']
    jobs=sum([json.loads((a.reference/f'jobs_{i}.json').read_text()) for i in range(len(old['devices']))],[])
    report=dict(policy='Released cached example: unpadded full instruction+document+newline, no prefix EOS; separately tokenized Query+query+EOS. No truncation. Right bucket padding excluded by valid lengths. Same candidates/labels/metric as reference.',reference_sha256=sha256(a.reference/'result.json'),tasks=[],all_checks_passed=False)
    c=CReference(a.runtime,4)
    try:
        for task in old['tasks']:
            name=task['task'];source=next(x for x in manifest['tasks'] if x['task']==name)
            assert sha256(a.data_root/source['data_file'])==source['data_sha256']
            data=json.loads((a.data_root/source['data_file']).read_text());corpus={x['_id']:x['text'] for x in data['corpus']};queries={x['_id']:x['text'] for x in data['queries']}
            selected=[j for j in jobs if j['task']==name];dids=sorted({d for j in selected for d in j['corpus_ids']})
            docs={d:c.tokenize(PREFIX.format(document=corpus[d])).tolist() for d in dids}
            qs={q:c.tokenize(SUFFIX.format(query=queries[q])).tolist()+[65535] for q in {j['query_id'] for j in selected}}
            assert all(x and x[-1]!=65535 for x in docs.values())
            mismatch=0
            for j in selected:
                q=j['query_id']
                for d in j['corpus_ids']:
                    full=c.tokenize(PREFIX.format(document=corpus[d])+SUFFIX.format(query=queries[q])).tolist()+[65535]
                    mismatch+=int(full!=docs[d]+qs[q])
            value=dict(task=name,documents=docs,queries=qs,jobs=selected)
            path=a.output/(name+'.json.gz')
            with gzip.open(path,'wt') as f:json.dump(value,f)
            row=dict(task=name,queries=len(selected),pairs=sum(len(j['corpus_ids']) for j in selected),documents=len(docs),max_document_tokens=max(map(len,docs.values())),max_query_tokens=max(map(len,qs.values())),documents_over_2048=sum(len(x)>2048 for x in docs.values()),split_tokenization_diff_pairs=mismatch,sha256=sha256(path),file=path.name,estimated_work_tokens=sum(map(len,docs.values()))+sum(sum(len(docs[d])+len(qs[j['query_id']]) for d in j['corpus_ids']) for j in selected))
            report['tasks'].append(row);print('PREPARED',json.dumps(row),flush=True)
        assert sum(t['pairs'] for t in report['tasks'])==57688
        report.update(all_checks_passed=True,seconds=time.perf_counter()-start);save(a.output/'manifest.json',report)
    finally:c.close()


class Continue(torch.nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,ids,lengths,shift,matrix):
        return self.model.encode_states(ids,(shift,matrix),valid_lengths=lengths)[1]


class Engine:
    def __init__(self,a):
        import torch_npu
        torch.set_num_threads(4);torch.npu.set_device(0);torch.npu.set_compile_mode(jit_compile=False)
        torch.npu.config.allow_internal_format=False;torch.npu.matmul.allow_hf32=False
        root=Path(__file__).parent;load_bridge(root/'wkv7_npu',a.reference_build);load_endpoint(a.build_root);register_converter()
        emb,rank,eh,rh,*_=pinned_pair(a)
        self.model=Embedding(emb,'npu:0',torch.float16,expected_sha256=eh);self.ranker=Reranker(rank,'npu:0',torch.float16,expected_sha256=rh)
        op=load_variant(a.vector_build,'aiv-fp32')
        for b in [*self.model.blocks,*self.ranker.blocks]:b.group_norm_impl='layer_norm';b.vector_variant=op;b.vector_dtype=torch.float32
        self.module=Continue(self.model).eval();self.eager=self.module.forward
        self.calls={};self.cache_root=a.output;self.head=compiled(self.ranker.forward,a.output/'head_cache')
        self.report=dict(checkpoint_sha256=eh,reranker_sha256=rh,physical_npu=os.environ['ASCEND_RT_VISIBLE_DEVICES'],device=torch.npu.get_device_name(0),dtype='FP16 dense; FP32 shifts/matrices',torch=torch.__version__,torch_npu=torch_npu.__version__)
    def zero(self):return (torch.zeros(24,2,4,2048,device='npu'),torch.zeros(24,4,32,64,64,device='npu'))
    def states(self,rows,state=None,backend='torchair',document=False):
        assert len(rows)==4 and all(rows)
        state=self.zero() if state is None else state
        for offset in range(0,max(map(len,rows)),2048):
            parts=[x[offset:offset+2048] for x in rows];ls=[len(x) for x in parts]
            choices=[512,2048] if document else [64,128,256,512,2048]
            T=next(t for t in choices if t>=max(ls))
            ids=torch.tensor([x+[0]*(T-len(x)) for x in parts],dtype=torch.long,device='npu')
            lens=torch.tensor([max(1,L) for L in ls],dtype=torch.int32,device='npu')
            if backend=='torchair' and T not in self.calls:
                self.calls[T]=compiled(self.module.forward,self.cache_root/f'continuation_t{T}')
            got=(self.calls[T] if backend=='torchair' else self.eager)(ids,lens,*state)
            if min(ls)==0:
                active=torch.tensor([L>0 for L in ls],device='npu')
                state=(torch.where(active[None,None,:,None],got[0],state[0]),torch.where(active[None,:,None,None,None],got[1],state[1]))
            else:state=got
        return state
    def scores(self,rows,state=None,backend='torchair',document=False):
        s=self.states(rows,state,backend,document)
        return (self.head if backend=='torchair' else self.ranker)(s[1])


def pack(state):
    return torch.cat([state[0].permute(2,0,1,3).reshape(4,-1),state[1].permute(1,0,2,3,4).reshape(4,-1)],dim=1).cpu().numpy()


def unpack(array):
    v=torch.from_numpy(np.array(array,copy=True)).to('npu');n=24*2*2048
    return (v[:,:n].reshape(4,24,2,2048).permute(1,2,0,3).contiguous(),v[:,n:].reshape(4,24,32,64,64).permute(1,0,2,3,4).contiguous())


def load_task(a,name):
    manifest=json.loads((a.prepared/'manifest.json').read_text());assert manifest['all_checks_passed']
    item=next(t for t in manifest['tasks'] if t['task']==name);p=a.prepared/item['file'];assert sha256(p)==item['sha256']
    with gzip.open(p,'rt') as f:return json.load(f)


def smoke(a,e,report):
    from run_reranker_buckets import profile
    data=load_task(a,'NanoSCIDOCSRetrieval');jobs=data['jobs'];docs=data['documents'];qs=data['queries']
    pairs=[(j['query_id'],d) for j in jobs for d in j['corpus_ids']]
    pairs=sorted(pairs,key=lambda p:len(docs[p[1]]))
    chosen=[pairs[i] for i in [0,len(pairs)//2,3*len(pairs)//4,len(pairs)-1]]
    prefixes=[docs[d] for q,d in chosen];suffixes=[qs[q] for q,d in chosen];full=[p+q for p,q in zip(prefixes,suffixes)]
    report.update(cases=chosen,prefix_lengths=list(map(len,prefixes)),query_lengths=list(map(len,suffixes)),checks={},timings={})
    # Paired full-model control: one uninterrupted canonical sequence versus a
    # split at the document boundary. The chunk loop retains all tokens.
    with torch.inference_mode():
        fullstate=e.states(full,backend='raw_eager',document=True);base=e.ranker(fullstate[1]);pre=e.states(prefixes,backend='raw_eager',document=True)
        before=[s.clone() for s in pre];continued=e.states(suffixes,pre,backend='raw_eager');got=e.ranker(continued[1])
        report['checks']['split_score']=metrics(got,base)
        report['checks']['split_states']=[metrics(x,y) for x,y in zip(continued,fullstate)]
        print('SPLIT_CHECK',json.dumps(report['checks']),flush=True)
        require(got,base,.02,.005)
        assert all(x['finite'] and x['normalized_rmse']<=.002 for x in report['checks']['split_states']),report['checks']['split_states']
        assert all(torch.equal(x,y) for x,y in zip(pre,before))
        single=[]
        for tokens in full:
            _,st,_=e.model.encode_states(torch.tensor([tokens],dtype=torch.long,device='npu'))
            single.append(e.ranker(st[1]))
        report['checks']['batch_vs_single']=require(base,torch.cat(single),.02,.005)
        # Exercise cross-chunk continuation against full-sequence dense ops and
        # the already-validated stock recurrence, including three inactive rows.
        longdata=load_task(a,'NanoFiQA2018Retrieval')
        ld=max(longdata['documents'],key=lambda d:len(longdata['documents'][d]))
        lj=next(j for j in longdata['jobs'] if ld in j['corpus_ids'])
        lp=longdata['documents'][ld];lq=longdata['queries'][lj['query_id']]
        assert len(lp)>2048
        pref=prefixes[:3]+[lp];suff=suffixes[:3]+[lq]
        lpstate=e.states(pref,backend='raw_eager',document=True)
        lc=e.scores(suff,lpstate,backend='raw_eager')
        lf=e.scores([p+q for p,q in zip(pref,suff)],backend='raw_eager',document=True)
        report['checks']['long_split_score']=require(lc,lf,.02,.005)
        operations=[b.vector_variant for b in e.model.blocks]
        try:
            for b in e.model.blocks:b.vector_variant=None
            _,refstate,_=e.model.encode_states(torch.tensor([lp+lq],dtype=torch.long,device='npu'))
            refscore=e.ranker(refstate[1])
        finally:
            for b,op in zip(e.model.blocks,operations):b.vector_variant=op
        report['checks']['long_full_sequence_control']=require(lc[-1:],refscore,.02,.005)
        report['checks']['long_compiled_cached']=require(e.scores(suff,e.states(pref,document=True)),lc,.02,.005)
        report['long_control_tokens']=len(lp+lq)
        # Real saved state, not random tensors or a device-resident-only timing.
        path=a.output/'smoke_states.npy';np.save(path,pack(pre));disk=np.load(path,mmap_mode='r')
        restored=unpack(disk);assert all(torch.equal(x,y) for x,y in zip(pre,restored))
        compiled_prefix=e.states(prefixes,document=True)
        report['checks']['compiled_prefix']=[require(x,y,.02,.005) for x,y in zip(compiled_prefix,pre)]
        result=e.scores(suffixes,restored)
        report['checks']['compiled_cached_score']=require(result,got,.02,.005)
        report['checks']['compiled_uncached_score']=require(e.scores(full,document=True),base,.02,.005)
        assert torch.equal(result,e.scores(suffixes,restored))
        # Full pipeline: filesystem state access/copy, H2D, query/state forward,
        # readout, D2H and score write. File cache state is OS-managed, not cold.
        def pipeline(backend,cached):
            if cached:
                with torch.profiler.record_function('rwkv.cache_load_and_h2d'):s=unpack(np.load(path,mmap_mode='r'))
            else:s=None
            with torch.profiler.record_function('rwkv.backbone_and_head'):v=e.scores(suffixes if cached else full,s,backend,document=not cached)
            with torch.profiler.record_function('rwkv.d2h_and_write'):
                values=v.cpu().tolist();save(a.output/'smoke_scores.json',values)
            return values
        for backend in ['raw_eager','torchair']:
            for cached in [False,True]:
                name=backend+('_cached' if cached else '_uncached');call=lambda:pipeline(backend,cached)
                report['timings'][name]=measure(call,10,4)
                report.setdefault('profiles',{})[name]=profile(call,a.output/('profile_'+name),'rwkv.'+name,warmup_iterations=5,active_iterations=2)
                print('TIMING',name,json.dumps(report['timings'][name]),flush=True)
        report['cache_bytes_per_document']=path.stat().st_size//4
        report['smoke_scope']='Four distinct real pairs; prepared tokens. Timing includes mmap access, host copy, state H2D, model/head, D2H and JSON write; excludes tokenization/setup/cache build. OS page cache not evicted.'


def evaluate(a,e,report):
    assert a.gate and json.loads((a.gate/'result.json').read_text())['all_checks_passed']
    gate=json.loads((a.gate/'result.json').read_text());assert gate['source_sha256']==report['source_sha256']
    _,metric=bm25_reference(a);previous=json.loads((a.reference/'result.json').read_text());old={t['task']:t['ndcg_at_10'] for t in previous['task_results']}
    report['tasks']=[];start=time.perf_counter()
    with torch.inference_mode():
        for name in a.tasks:
            taskstart=time.perf_counter();data=load_task(a,name);docs=data['documents'];queries=data['queries'];jobs=data['jobs'];folder=a.output/name;folder.mkdir()
            ids=sorted(docs);index={d:i for i,d in enumerate(ids)};path=folder/'document_states.npy'
            cache=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(ids),24*2*2048+24*32*64*64))
            build=time.perf_counter()
            # Warm both document buckets before measuring build; compile cost
            # belongs to setup. Representative actual rows are used.
            for want in [False,True]:
                d=next((d for d in ids if (len(docs[d])>512)==want),None)
                if d is not None:
                    for _ in range(3):e.states([docs[d]]*4,document=True)
            torch.npu.synchronize();warm=time.perf_counter()-build;build=time.perf_counter()
            for off in range(0,len(ids),4):
                chosen=ids[off:off+4];rows=[docs[d] for d in chosen];rows+=rows[-1:]*(4-len(rows))
                cache[off:off+len(chosen)]=pack(e.states(rows,document=True))[:len(chosen)]
                if off%100==0:progress(a.output/'progress.json',dict(task=name,phase='build',done=off+len(chosen),total=len(ids),seconds=time.perf_counter()-build))
            cache.flush();builds=time.perf_counter()-build;del cache;cache=np.load(path,mmap_mode='r')
            # Warm real query lengths and uncached shapes before timed scoring.
            for j in jobs:
                q=queries[j['query_id']];dids=j['corpus_ids'][:4]
                for _ in range(1):
                    e.scores([q]*4,unpack(cache[[index[d] for d in dids]]))
            torch.npu.synchronize()
            for j in jobs[:2]:
                e.scores([docs[d]+queries[j['query_id']] for d in j['corpus_ids'][:4]],document=True)
            torch.npu.synchronize()
            sums={'cached':0.,'uncached':0.};values={'cached':[],'uncached':[]};deltas=[];done=0
            with (folder/'scores.jsonl').open('w') as log:
                for j in jobs:
                    q=queries[j['query_id']];per={'cached':[],'uncached':[]}
                    for off in range(0,len(j['corpus_ids']),4):
                        ds=j['corpus_ids'][off:off+4];real=len(ds);ds+=ds[-1:]*(4-real)
                        # Alternate order by query to reduce order-dependent timing bias.
                        modes=['cached','uncached'] if done%2==0 else ['uncached','cached']
                        for mode in modes:
                            t=time.perf_counter()
                            if mode=='cached':
                                with torch.profiler.record_function('rwkv.cache_gather_and_h2d'):state=unpack(cache[[index[d] for d in ds]])
                                logits=e.scores([q]*4,state)
                            else:logits=e.scores([docs[d]+q for d in ds],document=True)
                            scores=logits.cpu().tolist()[:real];sums[mode]+=time.perf_counter()-t;per[mode].extend(scores)
                    for mode in per:
                        assert len(per[mode])==len(j['labels']) and all(np.isfinite(per[mode]))
                        v=metric(j['labels'],per[mode]);assert abs(v-tie_ndcg(j['labels'],per[mode]))<1e-12;values[mode].append(v)
                    deltas.extend(abs(x-y) for x,y in zip(per['cached'],per['uncached']))
                    log.write(json.dumps(dict(query_id=j['query_id'],corpus_ids=j['corpus_ids'],labels=j['labels'],scores=per))+'\n');log.flush();done+=1
                    progress(a.output/'progress.json',dict(task=name,phase='paired_score',done=done,total=len(jobs),scoring_seconds=sums,elapsed_seconds=time.perf_counter()-taskstart))
            row=dict(task=name,queries=len(jobs),pairs=sum(len(j['corpus_ids']) for j in jobs),documents=len(ids),cache_bytes=path.stat().st_size,cache_build_seconds=builds,build_warmup_seconds=warm,scoring_seconds=sums,ndcg={m:sum(v)/len(v)*100 for m,v in values.items()},previous_ndcg=old[name],max_cached_uncached_logit_delta=max(deltas),total_seconds=time.perf_counter()-taskstart)
            save(folder/'result.json',row);report['tasks'].append(row);save(a.output/'result.json',report);print('TASK_RESULT',json.dumps(row),flush=True)
            del cache
    report['evaluation_seconds']=time.perf_counter()-start


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','smoke','evaluate'])
    for n in ['output','reference','data-root','runtime','prepared','models','reference-build','build-root','vector-build','gate']:p.add_argument('--'+n,type=Path)
    p.add_argument('--size',default='large',choices=['large']);p.add_argument('--tasks',nargs='+');a=p.parse_args()
    if a.mode=='prepare':prepare(a);return
    a.output.mkdir(parents=True,exist_ok=False);root=Path(__file__).parent
    report=dict(mode=a.mode,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256={n:sha256(root/n) for n in [Path(__file__).name,'local_modeling_rwkv_embedding.py','local_modeling_rwkv_reranker.py','probe_reranker_endpoint.py','wkv7_vector_variants.py']},all_checks_passed=False)
    start=time.perf_counter()
    try:
        status=subprocess.check_output(['npu-status'],text=True);line=next(x for x in status.splitlines() if x.startswith('NPU '+os.environ['ASCEND_RT_VISIBLE_DEVICES']+': '));assert ': free ' in line and 'Health=OK' in line
        e=Engine(a);report.update(e.report)
        with torch.inference_mode():
            smoke(a,e,report) if a.mode=='smoke' else evaluate(a,e,report)
        report['all_checks_passed']=True
    except Exception as error:report['error']=f'{type(error).__name__}: {error}';raise
    finally:report['total_seconds']=time.perf_counter()-start;save(a.output/'result.json',report)

if __name__=='__main__':main()
