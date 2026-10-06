"""Complete pinned NanoBEIR embedding evaluation, preserving reference B4 inputs.

Long inputs keep the release's full preprocessing; the owned model carries WKV
state between <=2048-token kernel calls. Device row splitting changes no tokens.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import time

os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import numpy as np
import torch
from probe_wkv7 import load_bridge
from run_cpu_reference import CReference, CHECKPOINT_SHA256, INSTRUCTION, sha256
from run_npu_embedding import Embedding, comparison
from run_nanoscidocs import ndcg_reference, save

REFERENCE = {
    'NanoArguAnaRetrieval': (55.982, 55.989), 'NanoClimateFeverRetrieval': (38.144, 38.122),
    'NanoDBPediaRetrieval': (54.678, 54.789), 'NanoFEVERRetrieval': (86.418, 86.418),
    'NanoFiQA2018Retrieval': (50.029, 49.767), 'NanoHotpotQARetrieval': (71.790, 71.827),
    'NanoMSMARCORetrieval': (49.752, 49.779), 'NanoNFCorpusRetrieval': (32.538, 32.539),
    'NanoNQRetrieval': (58.963, 59.686), 'NanoQuoraRetrieval': (92.626, 92.626),
    'NanoSCIDOCSRetrieval': (41.058, 40.988), 'NanoSciFactRetrieval': (78.971, 78.956),
    'NanoTouche2020Retrieval': (54.845, 55.172),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint', 'runtime', 'build-root', 'data-root', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--allow-shared-device', action='store_true')
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--preflight-from', type=Path)
    p.add_argument('--task', choices=sorted(REFERENCE))
    p.add_argument('--max-token-slots', type=int, default=8192)
    args = p.parse_args()
    if args.max_token_slots < 2048: p.error('max-token-slots must be >=2048')
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    root = Path(__file__).parent
    report = dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        script_sha256=sha256(__file__), model_script_sha256=sha256(root/'run_npu_embedding.py'),
        checkpoint_sha256=CHECKPOINT_SHA256, manifest_sha256=sha256(args.data_root/'manifest.json'),
        physical_npu=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'), hostname=platform.node(),
        backend='raw_eager',dense_dtype='fp16',state_pointwise_dtype='float32',
        reference_batch_size=4,max_device_batch_size=4,max_token_slots=args.max_token_slots,
        mteb_contract_version='1.38.60',mteb_framework_used=False,ignore_identical_ids=False,
        context=2048,eos_chunk=512,instruction=INSTRUCTION,task_results=[],all_checks_passed=False,
        quality_review_limits=dict(per_task_cpu_delta_points=.25,macro_cpu_delta_points=.10))
    cpu = None
    try:
        import torch_npu
        import pytrec_eval
        torch.set_num_threads(4)
        manifest = json.loads((args.data_root/'manifest.json').read_text())
        assert manifest['mteb_version']=='1.38.60'
        assert {t['task'] for t in manifest['tasks']}==set(REFERENCE)
        assert sum(t['corpus_count'] for t in manifest['tasks'])==56723
        assert sum(t['query_count'] for t in manifest['tasks'])==649
        for task in manifest['tasks']:
            assert sha256(args.data_root/task['data_file'])==task['data_sha256']
        report['dataset_manifest']=manifest
        load_bridge(root/'wkv7_npu',args.build_root)
        status = subprocess.check_output(['/usr/local/bin/npu-status'],text=True)
        selected = next((x for x in status.splitlines() if x.startswith(f"NPU {report['physical_npu']}: ")), '')
        if 'Health=OK' not in selected or (not args.allow_shared_device and ': free ' not in selected):
            raise RuntimeError('Selected NPU unhealthy or unauthorized shared use')
        report['device_status']=status
        torch.npu.set_device(0)
        free,total=torch.npu.mem_get_info()
        if free < 2*1024**3: raise RuntimeError('Need 2 GiB free HBM')
        report.update(device=torch.npu.get_device_name(0),hbm_before_model=dict(free_bytes=free,total_bytes=total),
                      torch=torch.__version__,torch_npu=torch_npu.__version__)
        torch.npu.set_compile_mode(jit_compile=False)
        torch.npu.config.allow_internal_format=False
        cpu=CReference(args.runtime,4)
        model=Embedding(args.checkpoint,'npu:0',torch.float16)
        torch.npu.reset_peak_memory_stats()
        report['setup_seconds']=time.perf_counter()-start
        with torch.inference_mode():
            if args.preflight_from:
                previous=json.loads(args.preflight_from.read_text())
                for key in ['model_script_sha256','checkpoint_sha256','manifest_sha256']:
                    assert previous[key]==report[key],key
                assert previous['preflight_passed']
                report.update(preflight_passed=True,preflight=previous['preflight'],preflight_reused_from=str(args.preflight_from))
            else:
                candidates=[]
                for task in manifest['tasks']:
                    data=json.loads((args.data_root/task['data_file']).read_text())
                    for role in ['corpus','queries']:
                        rows=json.loads((args.data_root/task['task']/'lengths.json').read_text())[role]
                        lookup={r['_id']:r['text'] for r in data[role]}
                        for row in rows:
                            if row['prepared']>2048:
                                text=lookup[row['id']].strip() if role=='corpus' else INSTRUCTION.format(query=lookup[row['id']])
                                candidates.append((row['prepared'],task['task'],role,row['id'],text))
                assert candidates, 'Expected long inputs in the full suite'
                samples=[min(candidates),max(candidates)]
                report['preflight']=[]
                tiny=next(t for t in manifest['tasks'] if t['task']=='NanoSCIDOCSRetrieval')
                tiny_data=json.loads((args.data_root/tiny['data_file']).read_text())
                ids,mask=cpu.prepare_batch([INSTRUCTION.format(query=r['text']) for r in tiny_data['queries'][:4]])
                expected,_=cpu.encode(ids,mask,False)
                ni=torch.from_numpy(ids.astype(np.int64)).to('npu');nm=torch.from_numpy(mask.astype(np.float32)).to('npu')
                batched=model(ni,nm).cpu().numpy()
                split=np.concatenate([model(ni[i:i+1],nm[i:i+1]).cpu().numpy() for i in range(4)])
                metrics=comparison(batched,expected);row_metrics=comparison(split,batched)
                assert metrics['max_abs']<=.002 and np.min((batched*expected).sum(-1))>=.9995
                assert row_metrics['max_abs']<=.002 and np.min((split*batched).sum(-1))>=.9995
                report['row_split_preflight']=dict(shape=list(ids.shape),cpu_comparison=metrics,split_vs_batched=row_metrics)
                print('ROW_SPLIT_PREFLIGHT',json.dumps(report['row_split_preflight']),flush=True)
                del ni,nm,expected,batched,split
                for length,task,role,key,text in samples:
                    print('PREFLIGHT_START',task,role,key,length,flush=True)
                    ids,mask=cpu.prepare_batch([text]);assert ids.shape[1]==length
                    before=time.perf_counter()
                    expected,_=cpu.encode(ids,mask,False)
                    ni=torch.from_numpy(ids.astype(np.int64)).to('npu')
                    nm=torch.from_numpy(mask.astype(np.float32)).to('npu')
                    got=model(ni,nm).cpu().numpy()
                    metrics=comparison(got,expected);cosine=float((got*expected).sum())
                    assert metrics['max_abs']<=.002 and cosine>=.9995 and np.isfinite(got).all(),metrics
                    assert np.array_equal(got,model(ni,nm).cpu().numpy())
                    row=dict(task=task,role=role,id=key,shape=list(ids.shape),embedding=metrics,cosine=cosine,
                             seconds=time.perf_counter()-before)
                    report['preflight'].append(row);print('PREFLIGHT',json.dumps(row),flush=True)
                    del ni,nm,got
                torch.npu.empty_cache()
                report['preflight_passed']=True
            report['preflight_seconds']=time.perf_counter()-start-report['setup_seconds']
            if args.preflight_only:
                report['all_checks_passed']=True
                return
            evaluation_start=time.perf_counter()
            for task in manifest['tasks']:
                if args.task and args.task!=task['task']: continue
                name=task['task'];out=args.output/name;out.mkdir()
                data=json.loads((args.data_root/task['data_file']).read_text())
                assert data['revision']==task['revision']
                corpus={r['_id']:r['text'].strip() for r in data['corpus']}
                queries={r['_id']:INSTRUCTION.format(query=r['text']) for r in data['queries']}
                assert len(corpus)==task['corpus_count'] and len(queries)==task['query_count']
                qrels={qid:{} for qid in queries}
                for row in data['qrels']:
                    assert row['query-id'] in queries and row['corpus-id'] in corpus
                    qrels[row['query-id']][row['corpus-id']]=1
                assert all(qrels.values())
                begin=time.perf_counter();vectors={};id_sets={};stages={}
                with (out/'batch_timings.jsonl').open('w') as timings:
                    for group,texts in [('queries',queries),('corpus',corpus)]:
                        order=list(texts) if group=='queries' else sorted(texts,reverse=True)
                        id_sets[group]=order;embeddings=[];stage_start=time.perf_counter();forward_seconds=0
                        for offset in range(0,len(order),4):
                            keys=order[offset:offset+4]
                            ids,mask=cpu.prepare_batch([texts[k] for k in keys])
                            # Keep B4 preprocessing exact while bounding peak activation memory.
                            rows_per_call=max(1,min(4,args.max_token_slots//ids.shape[1]))
                            outputs=[];seconds=0
                            for row in range(0,len(keys),rows_per_call):
                                ni=torch.from_numpy(ids[row:row+rows_per_call].astype(np.int64)).to('npu')
                                nm=torch.from_numpy(mask[row:row+rows_per_call].astype(np.float32)).to('npu')
                                torch.npu.synchronize();before=time.perf_counter();got=model(ni,nm);torch.npu.synchronize()
                                seconds+=time.perf_counter()-before
                                outputs.append(got.cpu().numpy())
                            result=np.concatenate(outputs)
                            assert np.isfinite(result).all() and np.allclose(np.linalg.norm(result,axis=-1),1,atol=1e-4)
                            embeddings.append(result);forward_seconds+=seconds
                            timings.write(json.dumps(dict(group=group,offset=offset,ids=keys,shape=list(ids.shape),
                                rows_per_call=rows_per_call,forward_seconds=seconds))+'\n')
                            done=offset+len(keys)
                            if offset==0 or done//256!=offset//256 or done==len(order):
                                elapsed=time.perf_counter()-stage_start
                                progress=dict(task=name,group=group,done=done,total=len(order),elapsed_seconds=elapsed,
                                    texts_per_second=done/elapsed,estimated_remaining_seconds=elapsed*(len(order)-done)/done)
                                save(args.output/'progress.json',progress);print('PROGRESS',json.dumps(progress),flush=True)
                        vectors[group]=np.concatenate(embeddings)
                        stages[group]=dict(texts=len(order),wall_seconds=time.perf_counter()-stage_start,forward_seconds=forward_seconds)
                np.savez_compressed(out/'embeddings.npz',**vectors);save(out/'ids.json',id_sets)
                sims=torch.from_numpy(vectors['queries'])@torch.from_numpy(vectors['corpus']).T
                results={};top100={}
                for i,qid in enumerate(id_sets['queries']):
                    values,positions=torch.topk(sims[i],min(1000,len(corpus)),sorted=True)
                    results[qid]={id_sets['corpus'][j]:float(s) for j,s in zip(positions.tolist(),values.tolist())}
                    top100[qid]=[dict(document_id=id_sets['corpus'][j],score=float(s))
                                 for j,s in zip(positions[:100].tolist(),values[:100].tolist())]
                scores=pytrec_eval.RelevanceEvaluator(qrels,{'ndcg_cut.10'}).evaluate(results)
                manual=ndcg_reference(qrels,results)
                assert set(scores)==set(queries) and all(abs(scores[q]['ndcg_cut_10']-manual[q])<1e-12 for q in queries)
                score=round(sum(scores[q]['ndcg_cut_10'] for q in queries)/len(queries),5)*100
                save(out/'per_query_scores.json',scores);save(out/'top100.json',top100)
                row=dict(task=name,ndcg_at_10=score,upstream_cpu_fp32=REFERENCE[name][0],upstream_gpu_bf16=REFERENCE[name][1],
                    delta_cpu_points=score-REFERENCE[name][0],delta_gpu_points=score-REFERENCE[name][1],
                    wall_seconds=time.perf_counter()-begin,stages=stages,metric_crosscheck_passed=True,
                    artifacts={p.name:dict(sha256=sha256(p),bytes=p.stat().st_size) for p in out.iterdir()})
                save(out/'result.json',row);report['task_results'].append(row)
                save(args.output/'result.json',report);print('TASK_RESULT',json.dumps(row),flush=True)
            report['eval_wall_seconds']=time.perf_counter()-evaluation_start
            report['mean_ndcg_at_10']=sum(r['ndcg_at_10'] for r in report['task_results'])/len(report['task_results'])
            if not args.task:
                assert {r['task'] for r in report['task_results']}==set(REFERENCE)
                report.update(upstream_cpu_mean=58.907231,upstream_gpu_mean=58.973692,paper_mean=59.10,
                    delta_cpu_mean=report['mean_ndcg_at_10']-58.907231,delta_gpu_mean=report['mean_ndcg_at_10']-58.973692,
                    delta_paper_mean=report['mean_ndcg_at_10']-59.10)
                report['quality_review_required']=(abs(report['delta_cpu_mean'])>.10 or
                    any(abs(r['delta_cpu_points'])>.25 for r in report['task_results']))
            report['all_checks_passed']=True
            print('SUMMARY',json.dumps({k:v for k,v in report.items() if k not in ['dataset_manifest','task_results']}),flush=True)
    except Exception as error:
        report['error']=f'{type(error).__name__}: {error}'
        raise
    finally:
        if cpu is not None: cpu.close()
        report['total_seconds']=time.perf_counter()-start
        if torch.npu.is_initialized():
            report['peak_hbm_allocated_bytes']=torch.npu.max_memory_allocated()
            report['peak_hbm_reserved_bytes']=torch.npu.max_memory_reserved()
        save(args.output/'result.json',report)


if __name__=='__main__': main()
