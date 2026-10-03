"""Pinned CMTEB candidates -> native Decision2 prompts -> real HTTP scoring.

Prepare runs in the pinned MTEB environment; run uses the Decision2 environment.
No candidate changes, gold injection, truncation, prompt search, or isolated timing.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '22_qwen3_embedding_benchmark'))
from protocol import TASKS, validate_task


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def emit(event, **kw):
    print(json.dumps({'event': event, 'time': time.time(), **kw}, ensure_ascii=False), flush=True)


def prepare(a):
    import importlib.metadata
    import mteb
    from mteb.evaluation.evaluators.RetrievalEvaluator import corpus_to_str
    from run_reranker_evaluation import candidate_rows
    assert importlib.metadata.version('mteb') == '1.38.9'
    a.output.mkdir(parents=True, exist_ok=False)
    task = mteb.get_tasks(tasks=[a.task])[0]
    validate_task(task)
    task.load_data()
    baseline = json.loads(a.candidates.read_text())
    qrels = task.relevant_docs['dev']
    rr_manifest = json.loads((a.reranker / 'manifest.json').read_text())
    assert rr_manifest['sha256'] == digest(a.candidates)
    assert rr_manifest['dataset_revision'] == TASKS[a.task][0]
    assert rr_manifest['qrels_revision'] == TASKS[a.task][1]
    reranker = json.loads((a.reranker / 'predictions.json').read_text())
    assert set(baseline) == set(qrels) == set(reranker)
    for q in baseline:
        assert len(baseline[q]) == 100 and set(baseline[q]) == set(reranker[q])
    for name, obj in [('embedding.json', baseline), ('reranker.json', reranker), ('qrels.json', qrels)]:
        save(a.output / name, obj)
    count = 0
    with (a.output / 'pairs.jsonl').open('w') as f:
        for pair in candidate_rows(baseline, task.queries['dev'], task.corpus['dev'], corpus_to_str):
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
            count += 1
    manifest = {'task': a.task, 'queries': len(qrels), 'pairs': count,
                'dataset_revision': TASKS[a.task][0], 'qrels_revision': TASKS[a.task][1],
                'task_instruction': TASKS[a.task][2], 'ignore_identical_ids': task.ignore_identical_ids,
                'candidate_sha256': digest(a.candidates), 'reranker_source': str(a.reranker),
                'reranker_predictions_sha256': digest(a.reranker / 'predictions.json'),
                'pairs_sha256': digest(a.output / 'pairs.jsonl'),
                'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}
    save(a.output / 'manifest.json', manifest)
    emit('prepared', **manifest)


def run(a):
    from transformers import AutoTokenizer
    a.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(a.bundle))
    from decision2._vendor.dev2model.decision_model import encode
    from decision2._vendor.dev2model.infer import question_to_row, product_answer
    tok = AutoTokenizer.from_pretrained(str(a.bundle), local_files_only=True)
    manifest = json.loads((a.prepared / 'manifest.json').read_text())
    assert digest(a.prepared / 'pairs.jsonl') == manifest['pairs_sha256']
    instruction = 'Does the document satisfy this retrieval instruction for the query? ' + manifest['task_instruction']
    manifest.update(model=a.served_model, bundle=str(a.bundle), model_manifest_sha256=digest(a.bundle/'MODEL_MANIFEST.json'),
                    instruction=instruction, question_type='noul', temperature=1.0,
                    max_length=a.max_length, concurrency=a.concurrency, chip='910B2', physical_device=7,
                    dtype='BF16 backbone, original FP32 head', truncation='none; reject overlength',
                    server_command=a.server_command.read_text(), argv=sys.argv,
                    commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    timing='Scoring wall includes native encoding, bounded submission, HTTP, readout and flushed pair logs; warmup excluded')
    save(a.output/'manifest.json', manifest)
    state = {'section':'preflight_all_input_lengths', 'completed_pairs':0, 'total_pairs':manifest['pairs']}
    stop = threading.Event()
    def heartbeat():
        while not stop.wait(5):
            emit('heartbeat', **state)
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    def encoded(pair):
        row = question_to_row({'id':pair['qid']+'/'+pair['did'], 'state':{'query':pair['query'], 'document':pair['document']}},
                              'relevance', {'type':'noul', 'instructions':instruction})
        return row, encode(row, tok, a.max_length)
    try:
        lengths=[]
        first=[]
        for line in (a.prepared/'pairs.jsonl').open():
            pair=json.loads(line)
            _, e=encoded(pair)
            lengths.append(len(e['ids']))
            if len(first)<64:
                first.append(pair)
            state['completed_pairs']=len(lengths)
        save(a.output/'input_lengths.json', {'count':len(lengths), 'sum':sum(lengths), 'max':max(lengths),
             'mean':statistics.mean(lengths), 'overlength':0, 'truncated':0})
        def one(pair):
            start=time.perf_counter()
            row,e=encoded(pair)
            payload={'model':a.served_model, 'input':e['ids'], 'task':'classify', 'use_activation':False,
                     'add_special_tokens':False, 'encoding_format':'float',
                     'decision2':{'candidate_positions':e['candidate_positions'], 'query_position':e['query_position'], 'token_count':len(e['ids'])}}
            request=urllib.request.Request(a.url+'/pooling', data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request, timeout=300) as response:
                data=json.load(response)
            assert data['usage']['prompt_tokens']==len(e['ids'])
            logits=data['data'][0]['data']
            assert len(logits)==len(e['keys']) and all(math.isfinite(x) for x in logits)
            score=product_answer('noul',e['keys'],logits,1.0,[o['description'] for o in row['options']])['noul']
            assert math.isfinite(score) and 0<=score<=1
            return {'qid':pair['qid'], 'did':pair['did'], 'score':score, 'logits':logits, 'tokens':len(e['ids']), 'wall_s':time.perf_counter()-start}
        state.update(section='warmup', completed_pairs=0)
        with ThreadPoolExecutor(max_workers=a.concurrency) as pool:
            list(pool.map(one, first))
        predictions={}
        latencies=[]
        tokens=0
        state.update(section='scoring', completed_pairs=0)
        started=time.perf_counter()
        with (a.prepared/'pairs.jsonl').open() as source, (a.output/'requests.jsonl').open('w') as log, ThreadPoolExecutor(max_workers=a.concurrency) as pool:
            iterator=(json.loads(line) for line in source)
            pending=set()
            def submit():
                pair=next(iterator,None)
                if pair is not None:
                    pending.add(pool.submit(one,pair))
            for _ in range(a.concurrency):
                submit()
            while pending:
                done,pending=wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    r=future.result()
                    by_doc=predictions.setdefault(r['qid'],{})
                    assert r['did'] not in by_doc
                    by_doc[r['did']]=r['score']
                    log.write(json.dumps(r)+'\n')
                    log.flush()
                    tokens+=r['tokens']
                    latencies.append(r['wall_s'])
                    elapsed=time.perf_counter()-started
                    state.update(completed_pairs=len(latencies), pairs_s=len(latencies)/elapsed,
                                 input_tok_s=tokens/elapsed, elapsed_s=elapsed,
                                 eta_s=(manifest['pairs']-len(latencies))*elapsed/len(latencies))
                    if len(by_doc)==100:
                        emit('query_finished', qid=r['qid'], **state)
                    submit()
        wall=time.perf_counter()-started
        assert len(latencies)==manifest['pairs']
        save(a.output/'predictions.json', predictions)
        save(a.output/'timing.json', {'pairs':len(latencies), 'tokens':tokens, 'wall_s':wall,
             'pairs_s':len(latencies)/wall, 'input_tok_s':tokens/wall,
             'request_latency_s':{'p50':statistics.median(latencies), 'p99':sorted(latencies)[int(.99*(len(latencies)-1))], 'max':max(latencies)}})
        save(a.output/'completion.json', {'status':'complete', 'pairs':len(latencies)})
        emit('scoring_complete', **state)
    finally:
        stop.set()
        thread.join()


def metrics(a):
    from run_reranker_evaluation import metric_summary
    def read(p):
        return json.loads(p.read_text())
    manifest=read(a.prepared/'manifest.json')
    assert read(a.output/'completion.json')['status']=='complete'
    base=read(a.prepared/'embedding.json')
    qrels=read(a.prepared/'qrels.json')
    nox, per_nox=metric_summary(read(a.output/'predictions.json'),base,qrels,manifest['ignore_identical_ids'])
    rr, per_rr=metric_summary(read(a.prepared/'reranker.json'),base,qrels,manifest['ignore_identical_ids'])
    result={'task':manifest['task'], 'queries':manifest['queries'], 'pairs':manifest['pairs'],
            'metrics':{'embedding':nox['embedding'], 'nox':nox['reranker'], 'qwen3_reranker_4b':rr['reranker']},
            'nox_timing':read(a.output/'timing.json'),
            'scope':'Full pinned task dev set; identical saved top100; native fixed Noul prompt; no prompt tuning or truncation'}
    save(a.output/'result.json',result)
    save(a.output/'per_query_metrics.json', {'nox':per_nox['reranker'], 'qwen3_reranker_4b':per_rr['reranker'], 'embedding':per_nox['embedding']})
    emit('evaluation_complete', **result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='mode', required=True)
    prep=sub.add_parser('prepare')
    prep.add_argument('--task', choices=list(TASKS), required=True)
    prep.add_argument('--candidates', type=Path, required=True)
    prep.add_argument('--reranker', type=Path, required=True)
    prep.add_argument('--output', type=Path, required=True)
    r=sub.add_parser('run')
    r.add_argument('--prepared', type=Path, required=True)
    r.add_argument('--output', type=Path, required=True)
    r.add_argument('--bundle', type=Path, required=True)
    r.add_argument('--server-command', type=Path, required=True)
    r.add_argument('--served-model', default='nox-4b')
    r.add_argument('--url', default='http://127.0.0.1:18423')
    r.add_argument('--concurrency', type=int, default=64)
    r.add_argument('--max-length', type=int, default=8192)
    m=sub.add_parser('metrics')
    m.add_argument('--prepared', type=Path, required=True)
    m.add_argument('--output', type=Path, required=True)
    a=p.parse_args()
    {'prepare':prepare,'run':run,'metrics':metrics}[a.mode](a)


if __name__=='__main__':
    main()
