"""Full pinned Chinese CMTEB-R with independent TP1 query-shard workers.

Reuses the historical evaluator's candidate stream, journal, score conversion
and pytrec_eval metrics. Only prompt order and query partitioning are explicit
adaptations. Whole query groups stay on one worker; metrics merge by query.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from protocol import TASKS, validate_task
from reranker_protocol import tokenize_pairs, PREFIX, SUFFIX, MAX_LENGTH
from reranker_serving_sweep import digest, save
from run_evaluation import Observer, emit
import run_reranker_evaluation as evaluator
from run_english_suite import servers


def inventory(args):
    result = {}
    for name in TASKS:
        sources = [args.historical_root/run/'evaluation'/name/'manifest.json' for run in
            ['reranker_cmteb_6npu_fc41475e','reranker_ecom_e70e8473']]
        path = next(p for p in sources if p.is_file())
        original = json.loads(path.read_text())
        candidate = Path(original['candidate_file'])
        if not candidate.is_absolute():
            candidate = args.historical_root.parent.parent/candidate
        assert digest(candidate) == original['sha256'], name
        baseline = json.loads(candidate.read_text())
        assert len(baseline) == original['queries'], name
        assert all(len(docs)==100 for docs in baseline.values()), name
        assert (original['dataset_revision'],original['qrels_revision'],original['instruction']) == TASKS[name][:3]
        result[name] = dict(candidate_file=str(candidate), candidate_sha256=original['sha256'],
            historical_task_manifest=str(path), historical_manifest_sha256=digest(path),
            queries=len(baseline), pairs=len(baseline)*100, query_ids=sorted(baseline))
    assert sum(x['queries'] for x in result.values()) == 39740
    assert sum(x['pairs'] for x in result.values()) == 3974000
    directories = {str(Path(x['candidate_file']).parent) for x in result.values()}
    if len(directories) != 1:
        raise ValueError('Historical candidate files do not share one directory')
    save(args.root/'inventory.json', result)
    return result, directories.pop()


def reference(args):
    import mteb
    import torch
    import torch_npu
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from mteb.evaluation.evaluators.RetrievalEvaluator import corpus_to_str
    inventory = json.loads((args.root/'inventory.json').read_text())
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    workloads = []
    for name, item in inventory.items():
        task = mteb.get_tasks(tasks=[name])[0]
        validate_task(task); task.load_data()
        baseline = json.loads(Path(item['candidate_file']).read_text())
        assert set(baseline)==set(task.relevant_docs['dev']), name
        qids = [sorted(baseline)[0],sorted(baseline)[-1]]
        pairs=[]
        for q in qids:
            dids=sorted(baseline[q],key=lambda d:(-baseline[q][d],d))
            chosen=[dids[i] for i in [0,1,10,99]]
            docs=corpus_to_str([task.corpus['dev'][d] for d in chosen])
            pairs.extend(dict(qid=q,did=d,query=task.queries['dev'][q],document=doc)
                for d,doc in zip(chosen,docs))
        ids,truncated=tokenize_pairs(tok,name,pairs,'document_first')
        workloads.append(dict(task=name,pairs=pairs,input_ids=ids,truncated=truncated))
        emit('reference_inputs',task=name,pairs=len(ids),tokens=sum(map(len,ids)),truncated=truncated)
        del task,baseline
    torch.set_num_threads(8);torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    model=AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,
        dtype=torch.float16,attn_implementation='eager').eval().to('npu:0')
    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert {str(p.dtype) for p in model.parameters()}=={'torch.float16'}
    no,yes=tok.convert_tokens_to_ids('no'),tok.convert_tokens_to_ids('yes')
    with torch.inference_mode():
        for w in workloads:
            scores=[]
            for row in w['input_ids']:
                x=torch.tensor([row],device='npu:0')
                logits=model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False,
                    logits_to_keep=1).logits[:,-1,[no,yes]]
                scores.append(float(torch.softmax(logits.float(),dim=-1)[0,1].cpu()))
            w['scores']=scores
            emit('hf_reference',task=w['task'],pairs=len(scores))
    save(args.root/'hf_reference.json',workloads)


def worker(args):
    from transformers import AutoTokenizer
    output=args.root/'workers'/f'npu_{args.devices[args.worker_index]}'
    output.mkdir(parents=True,exist_ok=False)
    args.output=output
    args.resume_partial_run=None
    args.candidates=Path(args.candidates)
    tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    observer=Observer();observer.thread.start()
    rows=[]
    try:
        for name in TASKS:
            row=evaluator.evaluate(args,tok,[args.endpoint],observer,name,
                tokenizer_fn=lambda t,n,p:tokenize_pairs(t,n,p,'document_first'),
                query_shard=(args.worker_index,len(args.devices)))
            rows.append(row);save(output/'results.json',rows)
        save(output/'completion.json',dict(status='complete'))
    except BaseException as exc:
        save(output/'failure.json',dict(error=repr(exc)));raise
    finally:
        observer.stop.set();observer.thread.join(timeout=1)


def merge_query_metrics(shards, expected_ids):
    merged={'reranker':{},'embedding':{}}
    for shard in shards:
        for kind in merged:
            if set(merged[kind]).intersection(shard[kind]):
                raise ValueError('Query appeared in more than one shard')
            merged[kind].update(shard[kind])
    if any(set(rows)!=set(expected_ids) for rows in merged.values()):
        raise ValueError('Merged metric query coverage differs from full evaluation split')
    means={kind:{k:sum(r[k] for r in rows.values())/len(rows)
        for k in next(iter(rows.values()))} for kind,rows in merged.items()}
    for q in expected_ids:
        if abs(merged['reranker'][q]['recall_100']-merged['embedding'][q]['recall_100'])>1e-12:
            raise ValueError('Reranking changed recall@100')
    return means,merged


def merge_task(args,name,item):
    folders=[args.root/'workers'/f'npu_{d}'/name for d in args.devices]
    rows=[json.loads((p/'result.json').read_text()) for p in folders]
    per_query=[]
    for p in folders:
        manifest=json.loads((p/'manifest.json').read_text())
        assert manifest['sha256']==item['candidate_sha256']
        assert manifest['full_queries']==item['queries']
        assert manifest['input_order']=='document_first'
        per_query.append(json.loads((p/'per_query_metrics.json').read_text()))
    metrics,merged=merge_query_metrics(per_query,item['query_ids'])
    assert sum(r['queries'] for r in rows)==item['queries']
    assert sum(r['pairs'] for r in rows)==item['pairs']
    wall=max(r['scoring_ended_at'] for r in rows)-min(r['scoring_started_at'] for r in rows)
    row=dict(task=name,queries=item['queries'],pairs=item['pairs'],tokens=sum(r['tokens'] for r in rows),
        truncated_pairs=sum(r['truncated_pairs'] for r in rows),
        max_input_length=max(r['max_input_length'] for r in rows),
        scoring_wall_s=wall,scoring_worker_seconds=sum(r['scoring_wall_s'] for r in rows),
        task_wall_s=max(r['scoring_ended_at'] for r in rows)-min(r['task_started_at'] for r in rows),
        input_tok_s=sum(r['tokens'] for r in rows)/wall,npu_count=len(rows),metrics=metrics,
        candidate_sha256=item['candidate_sha256'],query_shards_verified=True,
        recall100_unchanged=True,worker_results=rows)
    folder=args.root/'tasks'/name;folder.mkdir(parents=True)
    save(folder/'per_query_metrics.json',merged);save(folder/'result.json',row)
    emit('task_finished',**{k:v for k,v in row.items() if k!='worker_results'})
    return row


def run(args):
    if args.root.exists() or args.devices != [0,1,2,3]:
        raise ValueError('Require fresh output and the explicitly authorized devices 0–3')
    args.root.mkdir(parents=True)
    started=time.monotonic()
    snapshot=subprocess.check_output(['npu-smi','info'],text=True,timeout=120)
    busy=[(int(a),int(b)) for a,b in re.findall(r'^\|\s*(\d+)\s+\d+\s*\|\s*(\d+)\s*\|',snapshot,re.M)]
    if any(d in args.devices and (d!=0 or pid!=635074) for d,pid in busy):
        raise RuntimeError(f'Unexpected process on selected NPU: {busy}')
    (args.root/'initial_npu_snapshot.txt').write_text(snapshot)
    items,candidates=inventory(args)
    export=json.loads((Path(args.model)/'export_manifest.json').read_text())
    assert export['update']==500 and export['bitwise_export_verified']
    assert export['model_files']=={p.name:digest(p) for p in Path(args.model).glob('*.safetensors')}
    args.input_order='document_first';args.output=args.root;args.allow_occupied_devices=[0]
    evaluator.MODEL=args.model
    manifest=dict(model=args.model,update=500,input_order=args.input_order,chip='910B2',
        dtype='float16',devices=args.devices,tasks=list(TASKS),queries=39740,pairs=3974000,
        mteb=importlib.metadata.version('mteb'),prefix=PREFIX,suffix=SUFFIX,max_length=MAX_LENGTH,
        task_instructions={n:TASKS[n][2] for n in TASKS},candidate_directory=candidates,
        model_files=export['model_files'],source_checkpoint_sha256=export['checkpoint_sha256'],
        server_command=evaluator.command(args.port),client_batch=128,concurrency_per_worker=2,
        partition='sorted full dev query IDs, round-robin across 4 independent worker processes',
        scoring='unchanged Qwen activated yes-minus-no relevance; historical pytrec_eval metrics',
        tokenization_timed=True,aggregation='unweighted mean of all 8 full task NDCG@10 scores',
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    save(args.root/'manifest.json',manifest)
    common=[sys.executable,__file__,'--root',str(args.root),'--model',args.model,
        '--devices',*map(str,args.devices)]
    with (args.root/'reference.log').open('w') as log:
        subprocess.run(common+['--phase','reference'],stdout=log,stderr=subprocess.STDOUT,check=True)
    # The HF subprocess has exited, releasing its context before serving startup.
    refs=json.loads((args.root/'hf_reference.json').read_text())
    observer=Observer();observer.thread.start()
    processes=[];logs=[];results=[]
    try:
        with servers(args,'reranker',observer) as endpoints:
            for endpoint in endpoints:
                for w in refs:
                    values=evaluator.score(endpoint,w['input_ids'])
                    delta=max(abs(a-b) for a,b in zip(values,w['scores']))
                    with (args.root/'hf_parity.jsonl').open('a') as log:
                        log.write(json.dumps(dict(endpoint=endpoint,task=w['task'],pairs=len(values),max_abs_diff=delta))+'\n')
                    emit('hf_parity',endpoint=endpoint,task=w['task'],max_abs_diff=delta)
                    if delta>.03:raise ValueError('HF/serving parity failed')
            (args.root/'workers').mkdir()
            for i,endpoint in enumerate(endpoints):
                log=(args.root/f'worker_{args.devices[i]}.log').open('w');logs.append(log)
                p=subprocess.Popen(common+['--phase','worker','--worker-index',str(i),
                    '--endpoint',endpoint,'--candidates',candidates],stdout=log,stderr=subprocess.STDOUT,
                    start_new_session=True)
                processes.append(p)
                save(args.root/f'worker_{args.devices[i]}.json',dict(pid=p.pid,endpoint=endpoint,device=args.devices[i]))
            pending=set(TASKS)
            while pending:
                if any(p.poll() not in (None,0) for p in processes):
                    raise RuntimeError('Evaluation worker failed; see worker logs and failure.json')
                for name in list(TASKS):
                    if name in pending and all((args.root/'workers'/f'npu_{d}'/name/'result.json').exists() for d in args.devices):
                        results.append(merge_task(args,name,items[name]));pending.remove(name)
                        save(args.root/'results.json',results)
                if all(p.poll()==0 for p in processes) and pending:
                    raise RuntimeError('Workers ended with missing tasks')
                time.sleep(1)
            for p in processes:
                if p.wait(timeout=60)!=0:raise RuntimeError('Worker exit failure')
        summary=evaluator.suite_summary(results)
        assert summary['scope']=='full_CMTEB-R' and summary['pairs']==3974000
        summary['published_reranker_macro_percent']=None;summary['delta_vs_published_pp']=None
        summary['historical_4b_macro_percent']=75.97906278867693
        summary['delta_vs_historical_4b_pp']=summary['macro_ndcg_at_10_percent']['reranker']-75.97906278867693
        save(args.root/'summary.json',summary)
        save(args.root/'completion.json',dict(status='complete',wall_s=time.monotonic()-started,
            all_queries_and_candidates_verified=True,all_task_recall100_unchanged=True,summary=summary))
    finally:
        for p in processes:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        for p in processes:
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        for log in logs:log.close()
        observer.stop.set();observer.thread.join(timeout=1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['run','reference','worker'],default='run')
    p.add_argument('--root',type=Path,required=True);p.add_argument('--model',required=True)
    p.add_argument('--historical-root',type=Path)
    p.add_argument('--devices',type=int,nargs='+',default=[0,1,2,3])
    p.add_argument('--port',type=int,default=18730)
    p.add_argument('--worker-index',type=int);p.add_argument('--endpoint');p.add_argument('--candidates')
    args=p.parse_args();args.input_order='document_first'
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    assert importlib.metadata.version('mteb')=='1.38.9'
    try:{'run':run,'reference':reference,'worker':worker}[args.phase](args)
    except BaseException as exc:
        if args.phase=='run' and args.root.exists():save(args.root/'failure.json',dict(error=repr(exc)))
        raise


if __name__=='__main__':main()
