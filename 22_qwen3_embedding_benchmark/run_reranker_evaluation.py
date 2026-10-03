"""Full pinned CMTEB-R: saved top100 -> FP16 vLLM-Ascend -> query metrics.

One eager TP1 server per explicitly reserved healthy NPU. Bounded client work
is distributed across servers, including T2, rather than assigning a whole task
to one device. No embedding recomputation or gold-document injection.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import shlex
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

from protocol import TASKS, validate_task
from reranker_protocol import MAX_LENGTH, PREFIX, SUFFIX, tokenize_pairs
from reranker_serving_sweep import MODEL, BASE_PYTHON, save, digest, request, scores_from_response
from run_evaluation import Observer, emit


def candidate_rows(candidates, queries, corpus, corpus_formatter):
    """Stable full-query ordering, preserving the exact saved candidate set."""
    for qid in sorted(candidates):
        dids = sorted(candidates[qid], key=lambda d: (-candidates[qid][d], d))
        if len(dids) != 100:
            raise ValueError(f'{qid}: expected exactly 100 candidates, got {len(dids)}')
        docs = corpus_formatter([corpus[d] for d in dids])
        for did, doc in zip(dids, docs):
            yield {'qid': qid, 'did': did, 'query': queries[qid], 'document': doc}


def batches(rows, size):
    batch = []
    for row in rows:
        batch.append(row)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def restore_journal(path):
    """Read flushed per-pair results; fail closed on corruption or duplicates."""
    predictions, records = {}, []
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            pairs = row.pop('pairs_scored')
            if len(pairs) != row['pairs'] or len(row['lengths']) != row['pairs']:
                raise ValueError('Incomplete journal record')
            if sum(row['lengths']) != row['tokens']:
                raise ValueError('Journal token count mismatch')
            for qid, did, score_value in pairs:
                by_doc = predictions.setdefault(qid, {})
                if did in by_doc or not 0 <= score_value <= 1:
                    raise ValueError('Duplicate/invalid journal score')
                by_doc[did] = score_value
            records.append(row)
    return predictions, records


def unscored_rows(rows, predictions):
    return (row for row in rows if row['did'] not in predictions.get(row['qid'], {}))


def metric_summary(predictions, baseline, qrels, ignore_identical_ids):
    import pytrec_eval
    # Exactly the pinned MTEB retrieval semantics: remove self-matches when asked.
    def filtered(source):
        return {q: {d: v for d, v in ds.items() if not (ignore_identical_ids and q == d)}
                for q, ds in source.items()}
    if set(predictions) != set(qrels) or set(baseline) != set(qrels):
        raise ValueError('Prediction/query/qrels coverage mismatch')
    for q in baseline:
        if set(predictions[q]) != set(baseline[q]):
            raise ValueError(f'Candidate membership changed for {q}')
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'ndcg_cut.10', 'recall.10,100'})
    after, before = evaluator.evaluate(filtered(predictions)), evaluator.evaluate(filtered(baseline))
    if set(after) != set(qrels) or set(before) != set(qrels):
        raise ValueError('Evaluator silently omitted queries')
    if any(abs(after[q]['recall_100'] - before[q]['recall_100']) > 1e-12 for q in qrels):
        raise ValueError('Reranking changed recall@100')
    mean = lambda rows: {k: sum(v[k] for v in rows.values()) / len(rows)
                         for k in next(iter(rows.values()))}
    return {'reranker': mean(after), 'embedding': mean(before)}, {'reranker': after, 'embedding': before}


def command(port):
    return [BASE_PYTHON, '-m', 'vllm.entrypoints.openai.api_server', '--model', MODEL,
            '--served-model-name', 'reranker-diagnostic', '--host', '127.0.0.1', '--port', str(port),
            '--runner', 'pooling', '--hf-overrides', json.dumps({
                'architectures': ['Qwen3ForSequenceClassification'],
                'classifier_from_token': ['no', 'yes'], 'is_original_qwen3_reranker': True}),
            '--dtype', 'float16', '--max-model-len', str(MAX_LENGTH), '--enforce-eager',
            '--gpu-memory-utilization', '0.45', '--max-num-seqs', '32', '--block-size', '128',
            '--max-num-batched-tokens', '16384', '--no-enable-prefix-caching',
            '--no-enable-chunked-prefill', '--no-async-scheduling']


def suite_summary(rows):
    names = [r['task'] for r in rows]
    if len(names) != len(set(names)) or not set(names).issubset(TASKS):
        raise ValueError('Duplicate or unknown task in aggregate')
    full = set(names) == set(TASKS)
    means = {kind: 100*sum(r['metrics'][kind]['ndcg_cut_10'] for r in rows)/len(rows)
             for kind in ['reranker', 'embedding']}
    return {'scope': 'full_CMTEB-R' if full else 'partial_CMTEB-R', 'tasks': names,
            'queries': sum(r['queries'] for r in rows), 'pairs': sum(r['pairs'] for r in rows),
            'tokens': sum(r['tokens'] for r in rows), 'macro_ndcg_at_10_percent': means,
            'published_reranker_macro_percent': 75.94 if full else None,
            'delta_vs_published_pp': means['reranker']-75.94 if full else None,
            'aggregation': 'unweighted mean of task nDCG@10, not query-weighted; no adjustment for first-stage discrepancy'}


@contextmanager
def servers(args, observer):
    owned, handles, endpoints = [], [], []
    try:
        observer.state = {'section': 'server_start', 'devices': args.devices}
        for index, device in enumerate(args.devices):
            port = args.port + index
            with socket.socket() as probe:
                if probe.connect_ex(('127.0.0.1', port)) == 0:
                    raise RuntimeError(f'Port {port} already occupied; will not touch its owner')
            folder = args.output / f'npu_{device}'
            folder.mkdir()
            cmd = command(port)
            save(folder / 'command.json', cmd)
            env = {**os.environ, 'ASCEND_RT_VISIBLE_DEVICES': str(device)}
            log = (folder / 'server.log').open('w')
            handles.append(log)
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                       env=env, start_new_session=True)
            owned.append(process)
            save(folder / 'pid.json', {'pid': process.pid, 'physical_device': device})
            endpoints.append(f'http://127.0.0.1:{port}')
        started = time.monotonic()
        pending = set(range(len(owned)))
        while pending and time.monotonic() - started < 600:
            for i in list(pending):
                if owned[i].poll() is not None:
                    raise RuntimeError(f'NPU {args.devices[i]} server exited; inspect its log')
                try:
                    with urllib.request.urlopen(endpoints[i] + '/health', timeout=2):
                        pending.remove(i)
                    emit('server_ready', device=args.devices[i], startup_s=time.monotonic()-started)
                except OSError:
                    pass
            if pending:
                time.sleep(1)
        if pending:
            raise TimeoutError(f'Servers not ready: {pending}')
        yield endpoints
    finally:
        # Signal only subprocess groups created by this invocation, never broad pkill.
        for p in owned:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        for p in owned:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait(timeout=10)
        for log in handles:
            log.close()


def score(endpoint, ids):
    data = request(endpoint + '/pooling', {'model': 'reranker-diagnostic', 'input': ids,
        'task': 'classify', 'use_activation': True, 'encoding_format': 'float', 'add_special_tokens': False})
    values = scores_from_response(data, len(ids)).tolist()
    if data['usage']['prompt_tokens'] != sum(map(len, ids)):
        raise ValueError('Server token usage does not match explicit inputs')
    return values


def parity(args, tok, endpoints, observer):
    workloads = json.loads((args.prepared / 'workloads.json').read_text())
    refs = {r['task']: r for r in json.loads((args.prepared / 'hf_reference.json').read_text())}
    prep = json.loads((args.prepared / 'manifest.json').read_text())
    assert digest(args.prepared / 'workloads.json') == prep['workloads_sha256']
    for name, expected in prep['model_files'].items():
        assert digest(Path(MODEL) / name) == expected, f'Model changed: {name}'
    for endpoint in endpoints:
        for w in workloads:
            observer.state = {'section': 'hf_parity', 'endpoint': endpoint, 'task': w['task']}
            full_ids, _ = tokenize_pairs(tok, w['task'], w['pairs'])
            assert hashlib.sha256(json.dumps(full_ids).encode()).hexdigest() == w['input_ids_sha256']
            ref = refs[w['task']]
            vals = score(endpoint, [full_ids[i] for i in ref['indices']])
            delta = max(abs(a-b) for a,b in zip(vals, ref['scores']))
            row = {'endpoint': endpoint, 'task': w['task'], 'scores': vals,
                   'hf_scores': ref['scores'], 'max_abs_diff': delta, 'threshold': .03}
            with (args.output / 'hf_parity.jsonl').open('a') as log:
                log.write(json.dumps(row) + '\n')
            emit('hf_parity', endpoint=endpoint, task=w['task'], max_abs_diff=delta)
            if delta > .03:
                raise ValueError('HF parity diagnostic failed; refusing full evaluation')


def evaluate(args, tok, endpoints, observer, name):
    import mteb
    import numpy as np
    from mteb.evaluation.evaluators.RetrievalEvaluator import corpus_to_str
    observer.state = {'section': 'loading_task', 'task': name}
    task_start = time.monotonic()
    task = mteb.get_tasks(tasks=[name])[0]
    validate_task(task)
    task.load_data()
    candidate_path = args.candidates / f'{name}_default_predictions.json'
    baseline = json.loads(candidate_path.read_text())
    qrels = task.relevant_docs['dev']
    if set(baseline) != set(qrels):
        raise ValueError('Saved candidates do not cover the exact full dev query set')
    folder = args.output / name
    folder.mkdir()
    save(folder / 'manifest.json', {'candidate_file': str(candidate_path), 'sha256': digest(candidate_path),
         'dataset_revision': TASKS[name][0], 'qrels_revision': TASKS[name][1],
         'instruction': TASKS[name][2], 'queries': len(qrels), 'pairs': len(qrels)*100,
         'ignore_identical_ids': task.ignore_identical_ids})
    predictions, records = {}, []
    resume_folder = args.resume_partial_run / name if args.resume_partial_run else None
    historical_wall = None
    if resume_folder:
        old_manifest = json.loads((resume_folder/'manifest.json').read_text())
        if old_manifest != json.loads((folder/'manifest.json').read_text()):
            raise ValueError('Resumed task manifest differs from original')
        source = resume_folder/'requests.jsonl'
        predictions, records = restore_journal(source)
        for qid, docs in predictions.items():
            if qid not in baseline or not set(docs).issubset(baseline[qid]):
                raise ValueError('Resumed scores contain unknown candidate pairs')
        restored_pairs = sum(r['pairs'] for r in records)
        # The original observer timestamps contain the scoring window, including
        # any earlier contention. Do not pass off a 6->5 NPU run as pure 5-NPU speed.
        old_log = args.resume_partial_run.parent/'run.log'
        if old_log.is_file():
            for line in old_log.open():
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get('event') == 'batch_finished' and event.get('task') == name and event.get('completed_pairs') == restored_pairs:
                    historical_wall = event.get('total_scoring_elapsed_s', event.get('elapsed_s'))
        shutil.copyfile(source, folder/'requests.jsonl')
        save(folder/'resume.json', {'source': str(source), 'sha256': digest(source),
             'restored_pairs': restored_pairs, 'historical_scoring_wall_s': historical_wall,
             'old_devices': json.loads((args.resume_partial_run/'manifest.json').read_text())['devices'],
             'new_devices': args.devices})
        emit('journal_restored', task=name, pairs=restored_pairs, remaining=len(qrels)*100-restored_pairs)
    iterator = iter(batches(unscored_rows(candidate_rows(baseline, task.queries['dev'], task.corpus['dev'], corpus_to_str), predictions), 128))
    completed, tokens, truncated, finished_queries = 0, 0, 0, 0
    completed = sum(r['pairs'] for r in records)
    tokens = sum(r['tokens'] for r in records)
    truncated = sum(r['truncated'] for r in records)
    finished_queries = sum(len(v)==100 for v in predictions.values())
    initial_completed, initial_tokens = completed, tokens
    scoring_start = time.monotonic()
    state = {'section': 'scoring', 'task': name, 'total_pairs': len(qrels)*100,
             'completed_pairs': 0, 'completed_queries': 0, 'npu_count': len(endpoints)}
    observer.state = state

    def submit(pool, endpoint):
        rows = next(iterator, None)
        if rows is None:
            return None
        t0 = time.monotonic()
        ids, count = tokenize_pairs(tok, name, rows)
        tokenize_s = time.monotonic()-t0
        lengths = list(map(len, ids))
        def one():
            start = time.monotonic()
            values = score(endpoint, ids)
            return rows, values, {'endpoint': endpoint, 'pairs': len(rows), 'tokens': sum(lengths),
                'lengths': lengths, 'truncated': count, 'tokenize_s': tokenize_s,
                'http_s': time.monotonic()-start}
        return pool.submit(one)

    with (folder / 'requests.jsonl').open('a') as log, ThreadPoolExecutor(max_workers=2*len(endpoints)) as pool:
        pending = {}
        for endpoint in endpoints:
            for _ in range(2):
                future = submit(pool, endpoint)
                if future is not None:
                    pending[future] = endpoint
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                endpoint = pending.pop(future)
                rows, values, record = future.result()
                for p, val in zip(rows, values):
                    by_doc = predictions.setdefault(p['qid'], {})
                    if p['did'] in by_doc:
                        raise ValueError('Duplicate completed pair')
                    by_doc[p['did']] = val
                    if len(by_doc) == 100:
                        finished_queries += 1
                # Per-pair scores and lengths survive interruption, without duplicated document text.
                log.write(json.dumps({**record, 'pairs_scored': [[p['qid'], p['did'], v]
                                  for p,v in zip(rows, values)]}) + '\n')
                log.flush()
                records.append(record)
                completed += record['pairs']
                tokens += record['tokens']
                truncated += record['truncated']
                elapsed = time.monotonic()-scoring_start
                state.update(completed_pairs=completed, completed_queries=finished_queries,
                             elapsed_s=elapsed, restored_pairs=initial_completed,
                             total_scoring_elapsed_s=(historical_wall or 0)+elapsed if not initial_completed or historical_wall is not None else None,
                             input_tok_s=(tokens-initial_tokens)/elapsed,
                             eta_s=(state['total_pairs']-completed)*elapsed/(completed-initial_completed))
                emit('batch_finished', **state, batch_http_s=record['http_s'])
                replacement = submit(pool, endpoint)
                if replacement is not None:
                    pending[replacement] = endpoint
    scoring_wall = time.monotonic()-scoring_start
    observer.state = {'section': 'metrics_and_save', 'task': name}
    metrics, per_query = metric_summary(predictions, baseline, qrels, task.ignore_identical_ids)
    save(folder / 'predictions.json', predictions)
    save(folder / 'per_query_metrics.json', per_query)
    lengths = [x for r in records for x in r['lengths']]
    combined_wall = scoring_wall if not resume_folder else (historical_wall+scoring_wall if historical_wall is not None else None)
    row = {'task': name, 'queries': len(qrels), 'pairs': completed, 'tokens': tokens,
        'truncated_pairs': truncated, 'max_input_length': max(lengths),
        'scoring_wall_s': combined_wall, 'task_wall_s': time.monotonic()-task_start,
        'input_tok_s': tokens/combined_wall if combined_wall else None,
        'pairs_s': completed/combined_wall if combined_wall else None,
        'query_s': len(qrels)/combined_wall if combined_wall else None,
        'npu_count': None if resume_folder else len(endpoints), 'metrics': metrics,
        'resumed_phase': {'npu_count': len(endpoints), 'restored_pairs': initial_completed,
                          'scoring_wall_s': scoring_wall, 'pairs': completed-initial_completed,
                          'input_tok_s': (tokens-initial_tokens)/scoring_wall} if resume_folder else None,
        'ndcg_change_pp': 100*(metrics['reranker']['ndcg_cut_10']-metrics['embedding']['ndcg_cut_10']),
        'request_latency_s': {k: float(np.percentile([r['http_s'] for r in records], p))
                              for k,p in [('p50',50),('p99',99),('max',100)]}}
    save(folder / 'result.json', row)
    emit('task_finished', **row)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks', nargs='+', choices=list(TASKS), required=True)
    p.add_argument('--devices', nargs='+', type=int, required=True,
                   help='Physical devices verified healthy and free immediately before launch')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--candidates', type=Path, required=True)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--completed-run', type=Path, action='append', default=[],
                   help='Include previously completed disjoint tasks in the final suite summary')
    p.add_argument('--resume-partial-run', type=Path,
                   help='Stopped evaluation directory: recover requested task journals and retain its completed tasks')
    p.add_argument('--port', type=int, default=18325)
    args = p.parse_args()
    if len(set(args.devices)) != len(args.devices) or len(set(args.tasks)) != len(args.tasks):
        p.error('Duplicate devices or tasks')
    if args.output.exists():
        p.error('Output must be fresh')
    args.output.mkdir(parents=True)
    save(args.output / 'invocation.json', {'argv': [sys.executable, *sys.argv],
         'cwd': os.getcwd(), 'started_at': time.time()})
    (args.output / 'command.txt').write_text(
        'commit=' + subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() + '\n'
        + 'hostname=' + socket.gethostname() + '\n'
        + 'physical_devices=' + ','.join(map(str, args.devices)) + '\n'
        + 'ASCEND_RT_VISIBLE_DEVICES=' + os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '') + '\n'
        + shlex.join([sys.executable, *sys.argv]) + '\n')
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    observer = Observer()
    observer.thread.start()
    started = time.monotonic()
    try:
        from transformers import AutoTokenizer
        assert importlib.metadata.version('mteb') == '1.38.9'
        manifest = {'commit': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
             'model': MODEL, 'chip': '910B2', 'dtype': 'float16', 'devices': args.devices,
             'tasks': args.tasks, 'max_length': MAX_LENGTH, 'prefix': PREFIX, 'suffix': SUFFIX,
             'client_batch': 128, 'concurrency_per_server': 2, 'eager': True,
             'prompt_source': 'HF Transformers example with pinned task instructions; single newline between fields',
             'scope': 'full dev queries, fixed saved embedding top100; no gold injection',
             'timing': 'scoring wall includes streaming corpus-to-text, tokenization, HTTP, checkpoint logging; setup separately',
             'completed_runs': [str(path) for path in args.completed_run],
             'model_files': {n: digest(Path(MODEL)/n) for n in
                             ['config.json', 'tokenizer_config.json', 'model.safetensors.index.json']}}
        save(args.output / 'manifest.json', manifest)
        prior = []
        if args.resume_partial_run:
            previous_manifest = json.loads((args.resume_partial_run/'manifest.json').read_text())
            for key in ['model', 'chip', 'dtype', 'max_length', 'prefix', 'suffix', 'eager', 'prompt_source', 'scope', 'model_files']:
                assert previous_manifest[key] == manifest[key], f'Resume contract differs: {key}'
            prior.extend(json.loads((args.resume_partial_run/'results.json').read_text()))
            manifest['resume_partial_run'] = str(args.resume_partial_run)
            save(args.output/'manifest.json', manifest)
        for path in args.completed_run:
            assert json.loads((path/'completion.json').read_text())['status'] == 'complete'
            previous_manifest = json.loads((path/'manifest.json').read_text())
            for key in ['model', 'chip', 'dtype', 'max_length', 'prefix', 'suffix', 'eager', 'prompt_source', 'scope']:
                assert previous_manifest[key] == manifest[key], f'Previous run contract differs: {key}'
            prior.extend(json.loads((path/'results.json').read_text()))
        if len({r['task'] for r in prior}) != len(prior) or {r['task'] for r in prior}.intersection(args.tasks):
            raise ValueError('Previous runs contain duplicated or newly requested tasks')
        tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        results = []
        with servers(args, observer) as endpoints:
            parity(args, tok, endpoints, observer)
            for name in args.tasks:
                results.append(evaluate(args, tok, endpoints, observer, name))
                save(args.output / 'results.json', results)
                save(args.output / 'combined_results.json', prior + results)
                summary = suite_summary(prior + results)
                save(args.output / 'suite_summary.json', summary)
                emit('suite_summary', **summary)
        save(args.output / 'completion.json', {'status': 'complete', 'wall_s': time.monotonic()-started})
        (args.output / 'exit_code.txt').write_text('0\n')
    except BaseException as exc:
        save(args.output / 'completion.json', {'status': 'failed', 'error': repr(exc), 'wall_s': time.monotonic()-started})
        (args.output / 'exit_code.txt').write_text('1\n')
        raise
    finally:
        observer.stop.set()
        observer.thread.join(timeout=1)


if __name__ == '__main__':
    main()
