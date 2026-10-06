"""Released 90M RWKV reranker on pinned NanoSCIDOCS top-100 candidates."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time

os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import numpy as np
import torch
from local_modeling_rwkv_embedding import Embedding
from local_modeling_rwkv_reranker import Reranker, CHECKPOINT_SHA256 as RERANKER_SHA256
from probe_wkv7 import load_bridge
from run_cpu_reference import CReference, CHECKPOINT_SHA256, sha256
from run_nanoscidocs import ndcg_reference, save
from run_reranker_smoke import PREFIX, SUFFIX, cpu_models, metrics, require

DATA_SHA256 = 'bfc261e4b59f9b5e3e9915cc9cde95241d2dfe466ee4595a012c7369f4d34190'
CANDIDATES_SHA256 = 'dcd8a6d3de984c5d0ca47d36e211838cfe496cb13681021e6c960ae0c38c74b6'
REVISION = '484eb90549fc3f0b9c42b3551e80ceb999515537'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint','reranker','runtime','upstream','build-root','data','candidates','smoke-result','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--batch-size',type=int,choices=(1,2),default=2)
    p.add_argument('--allow-shared-device',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).parent;start=time.perf_counter();c=None
    report={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'source_sha256':{n:sha256(root/n) for n in ['run_nanoscidocs_reranker.py','local_modeling_rwkv_embedding.py','local_modeling_rwkv_reranker.py','run_reranker_smoke.py','run_nanoscidocs.py']},
            'checkpoint_sha256':sha256(args.checkpoint),'reranker_sha256':sha256(args.reranker),
            'data_sha256':sha256(args.data),'candidates_sha256':sha256(args.candidates),
            'dataset':'zeta-alpha-ai/NanoSCIDOCS','revision':REVISION,'split':'train',
            'mteb_contract_version':'1.38.60','mteb_framework_used':False,
            'physical_npu':os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),'hostname':platform.node(),
            'backend':'raw_eager','batch_size':args.batch_size,'dense_dtype':'fp16','state_pointwise_dtype':'float32',
            'prompt':PREFIX+SUFFIX,'padding':'left zero padding to batch longest; padding participates in recurrence',
            'truncation':'last 2048 tokens, after padding','eos':'one terminal 65535; no embedding repetition/multi-EOS preprocessing',
            'candidate_source':'Full NanoBEIR 0.1B embedding reproduction c3166fd1, reference B4',
            'ordering':'dataset query order, saved candidate rank order; no pair reordering',
            'ignore_identical_ids':False,'document_caching':False,'cpu_threads':4,
            'thresholds':{'state_normalized_rmse':.02,'hidden_normalized_rmse':.02,'logit_atol':.02,'logit_rtol':.005},
            'all_checks_passed':False}
    try:
        import torch_npu
        import pytrec_eval
        torch.set_num_threads(4)
        assert report['data_sha256']==DATA_SHA256 and report['candidates_sha256']==CANDIDATES_SHA256
        assert report['checkpoint_sha256']==CHECKPOINT_SHA256 and report['reranker_sha256']==RERANKER_SHA256
        smoke=json.loads(args.smoke_result.read_text());assert smoke['all_checks_passed']
        for name in ['local_modeling_rwkv_embedding.py','local_modeling_rwkv_reranker.py','run_reranker_smoke.py']:
            assert smoke['source_sha256'][name]==report['source_sha256'][name]
        report['smoke_gate']={'path':str(args.smoke_result),'sha256':sha256(args.smoke_result),'source_commit':smoke['source_commit']}
        data=json.loads(args.data.read_text());assert data['revision']==REVISION
        corpus={x['_id']:x['text'].strip() for x in data['corpus']}
        queries={x['_id']:x['text'] for x in data['queries']}
        candidates=json.loads(args.candidates.read_text())
        assert len(corpus)==2210 and len(queries)==50 and set(candidates)==set(queries)
        qrels={qid:{} for qid in queries}
        for x in data['qrels']:
            assert x['query-id'] in queries and x['corpus-id'] in corpus
            qrels[x['query-id']][x['corpus-id']]=1
        assert sum(map(len,qrels.values()))==244 and all(qrels.values())
        pairs=[];retrieval={}
        for qid,query in queries.items():
            docs=candidates[qid];assert len(docs)==100 and len({x['document_id'] for x in docs})==100
            assert all(x['document_id'] in corpus and np.isfinite(x['score']) for x in docs)
            retrieval[qid]={x['document_id']:x['score'] for x in docs}
            pairs.extend((qid,x['document_id']) for x in docs)
        report['coverage']={'queries':50,'corpus_documents':2210,'judgments':244,'candidate_pairs':len(pairs)}
        assert len(pairs)==5000 and len(set(pairs))==5000
        load_bridge(root/'wkv7_npu',args.build_root)
        status=subprocess.check_output(['/usr/local/bin/npu-status'],text=True)
        selected=next((x for x in status.splitlines() if x.startswith(f"NPU {report['physical_npu']}: ")), '')
        if 'Health=OK' not in selected or (not args.allow_shared_device and ': free ' not in selected):
            raise RuntimeError('Unhealthy device or unauthorized shared execution')
        torch.npu.set_device(0);torch.npu.set_compile_mode(jit_compile=False);torch.npu.config.allow_internal_format=False
        free,total=torch.npu.mem_get_info()
        if free<2*1024**3:raise RuntimeError('Need 2 GiB free HBM')
        report.update(device=torch.npu.get_device_name(0),torch=torch.__version__,torch_npu=torch_npu.__version__,
                      device_status=status,hbm_before={'free_bytes':free,'total_bytes':total},
                      pytrec_eval=importlib.metadata.version('pytrec-eval-terrier'))
        c=CReference(args.runtime,4)
        embed=Embedding(args.checkpoint,'npu:0',torch.float16)
        ranker=Reranker(args.reranker,'npu:0',torch.float16)
        torch.npu.reset_peak_memory_stats()
        # Tokenize once, keeping every original pair and the wrapper's exact batch order.
        tokens=[c.tokenize(PREFIX.format(document=corpus[did])+SUFFIX.format(query=queries[qid])).tolist()+[65535]
                for qid,did in pairs]
        def prepared(offset):
            rows=tokens[offset:offset+args.batch_size];length=max(map(len,rows))
            return [[0]*max(0,min(2048,length)-len(ids))+ids[-2048:] for ids in rows]
        report['length_audit']={'maximum_raw_pair_tokens':max(map(len,tokens)),
                               'pairs_truncated':sum(len(x)>2048 for x in tokens),
                               'maximum_device_tokens':min(2048,max(map(len,tokens)))}
        # This is equivalent to padding first and then retaining the last 2048 positions.
        assert all(len(set(map(len,prepared(i))))==1 for i in range(0,len(pairs),args.batch_size))
        report['setup_seconds']=time.perf_counter()-start
        with torch.inference_mode():
            longest=max(range(0,len(pairs),args.batch_size),key=lambda i:max(map(len,tokens[i:i+args.batch_size])))
            rows=prepared(longest)
            print('PREFLIGHT_START',json.dumps({'offset':longest,'shape':[len(rows),len(rows[0])],'pairs':pairs[longest:longest+len(rows)]}),flush=True)
            before=time.perf_counter();cpu,cpu_ranker,indices=cpu_models(args)
            state=cpu.generate_zero_state(len(rows))
            expected_hidden=cpu.forward_seq_batch(rows,state,True)
            expected=cpu_ranker(torch.stack([state[1][i] for i in indices])).reshape(-1)
            ni=torch.tensor(rows,dtype=torch.long,device='npu')
            hidden,nstate,_=embed.encode_states(ni);actual=ranker(nstate[1])
            sm=[metrics(a,b) for a,b in zip(nstate,state)];hm=metrics(hidden,expected_hidden)
            assert all(x['finite'] and x['normalized_rmse']<=.02 for x in sm+[hm])
            score_match=require(actual,expected,.02,.005)
            single=[]
            for i in range(len(rows)):
                _,s,_=embed.encode_states(ni[i:i+1]);single.append(ranker(s[1]))
            batch_match=require(torch.cat(single),actual,.02,.005)
            np.savez_compressed(args.output/'preflight.npz',input_ids=np.asarray(rows),cpu_logits=expected.numpy(),npu_logits=actual.cpu().numpy(),cpu_shift=state[0].numpy(),cpu_matrix=state[1].numpy())
            report['preflight']={'passed':True,'offset':longest,'shape':list(ni.shape),'states':sm,'hidden':hm,
                                 'logits':score_match,'batch_vs_single':batch_match,
                                 'cpu_logits':expected.tolist(),'npu_logits':actual.cpu().tolist(),
                                 'seconds':time.perf_counter()-before}
            print('PREFLIGHT',json.dumps(report['preflight']),flush=True)
            del cpu,cpu_ranker,state,expected_hidden,expected,hidden,nstate,actual,s,single,ni
            # Two actual B2 batches warm both model stages outside evaluation timing.
            ni=torch.tensor(prepared(0),dtype=torch.long,device='npu')
            for _ in range(2):
                _,state,_=embed.encode_states(ni);ranker(state[1])
            torch.npu.synchronize();del state,ni
            report['preflight_and_warmup_seconds']=time.perf_counter()-start-report['setup_seconds']
            save(args.output/'result.json',report)
            evaluation_start=time.perf_counter();scores={qid:{} for qid in queries};forward_total=0;batch_seconds=[]
            with (args.output/'batch_timings.jsonl').open('w') as log:
                for offset in range(0,len(pairs),args.batch_size):
                    rows=prepared(offset);ni=torch.tensor(rows,dtype=torch.long,device='npu')
                    torch.npu.synchronize();before=time.perf_counter()
                    _,state,_=embed.encode_states(ni);logits=ranker(state[1]);torch.npu.synchronize()
                    elapsed=time.perf_counter()-before;forward_total+=elapsed;batch_seconds.append(elapsed)
                    values=logits.cpu().tolist();assert all(np.isfinite(values))
                    for (qid,did),value in zip(pairs[offset:offset+len(rows)],values):
                        assert did not in scores[qid];scores[qid][did]=value
                    log.write(json.dumps({'offset':offset,'pairs':pairs[offset:offset+len(rows)],'shape':list(ni.shape),
                              'input_sha256':hashlib.sha256(np.asarray(rows,dtype=np.int64).tobytes()).hexdigest(),'forward_seconds':elapsed})+'\n')
                    del ni,state,logits
                    done=offset+len(rows)
                    if offset==0 or done//500!=offset//500 or done==len(pairs):
                        wall=time.perf_counter()-evaluation_start
                        progress={'done':done,'total':len(pairs),'elapsed_seconds':wall,'pairs_per_second':done/wall,
                                  'estimated_remaining_seconds':wall*(len(pairs)-done)/done}
                        save(args.output/'progress.json',progress);print('PROGRESS',json.dumps(progress),flush=True)
            report.update(scoring_wall_seconds=time.perf_counter()-evaluation_start,forward_seconds=forward_total,
                          batches=len(batch_seconds),batch_forward_median_seconds=float(np.median(batch_seconds)),
                          batch_forward_p95_seconds=float(np.percentile(batch_seconds,95)))
            assert set(scores)==set(queries) and all(set(scores[q])==set(retrieval[q]) for q in queries)
            evaluator=pytrec_eval.RelevanceEvaluator(qrels,{'ndcg_cut.10'})
            reranked=evaluator.evaluate(scores);baseline=evaluator.evaluate(retrieval)
            for results,measured in [(scores,reranked),(retrieval,baseline)]:
                manual=ndcg_reference(qrels,results)
                assert set(measured)==set(queries) and all(abs(manual[q]-measured[q]['ndcg_cut_10'])<1e-12 for q in queries)
            ndcg=sum(x['ndcg_cut_10'] for x in reranked.values())/50
            base=sum(x['ndcg_cut_10'] for x in baseline.values())/50
            assert round(base,5)*100==41.058
            save(args.output/'pair_scores.json',scores);save(args.output/'per_query_scores.json',reranked)
            save(args.output/'ranked_top100.json',{qid:sorted(scores[qid],key=lambda did:(scores[qid][did],did),reverse=True) for qid in queries})
            report.update(ndcg_at_10=round(ndcg,5)*100,ndcg_at_10_unrounded=ndcg,
                          embedding_baseline_ndcg_at_10=round(base,5)*100,delta_embedding_points=(ndcg-base)*100,
                          metric_crosscheck_passed=True,coverage_passed=True,all_checks_passed=True,
                          evaluation_wall_seconds=time.perf_counter()-evaluation_start,
                          published_per_task_reranker_reference=None)
            report['artifacts']={p.name:{'sha256':sha256(p),'bytes':p.stat().st_size} for p in args.output.iterdir() if p.name!='result.json'}
            print('SUMMARY',json.dumps({k:v for k,v in report.items() if k not in ['device_status','artifacts']}),flush=True)
    except Exception as e:
        report['error']=f'{type(e).__name__}: {e}';raise
    finally:
        if c is not None:c.close()
        report['total_seconds']=time.perf_counter()-start
        if hasattr(torch, 'npu') and torch.npu.is_initialized():
            report['peak_hbm_allocated_bytes']=torch.npu.max_memory_allocated()
            report['peak_hbm_reserved_bytes']=torch.npu.max_memory_reserved()
        save(args.output/'result.json',report)

if __name__=='__main__':main()
