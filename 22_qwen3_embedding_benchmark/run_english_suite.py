"""English MTEB-R: pinned datasets -> embedding top100 -> reranking.

Real evaluation through five TP1 vLLM-Ascend servers, no synthetic microbench.
The Chinese evaluator is unchanged. Disk caches and pair journals are scoped to
an immutable run manifest; --resume checks it before reusing anything.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import signal
import shutil
import socket
import subprocess
import sys
import threading
import time

from protocol import MODEL_REVISION
from reranker_protocol import PREFIX, SUFFIX, MAX_LENGTH
from run_evaluation import Observer, emit
import run_reranker_evaluation as reranker_evaluator
from run_reranker_evaluation import (batches, candidate_rows, command as reranker_command,
    metric_summary, restore_journal, score, unscored_rows, parity)
from reranker_serving_sweep import BASE_PYTHON, MODEL as RERANKER_MODEL, request
from suite_protocol import (BENCHMARK, ENGLISH, MTEB_VERSION, QWEN_COMMIT,
                            aggregate, format_embedding, validate_tasks)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '13_qwen3_reranker'))
EMBEDDING_MODEL = '/workspace/models/Qwen3-Embedding-0.6B'


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def load_task(name, observer):
    import mteb
    module = importlib.import_module('mteb.abstasks.AbsTaskRetrieval')
    task = mteb.get_tasks(tasks=[name])[0]
    observer.state = {'section': 'dataset_loading', 'task': name}
    path, revision, _, _ = ENGLISH[name]
    original = module.load_dataset
    calls = []

    def pinned_loader(repo, *args, **kwargs):
        # MTEB 1.38.9's generic HFDataLoader drops metadata.dataset.revision.
        # Scope this fix to the owned evaluation process and restore afterwards.
        if repo != path or kwargs.get('revision', revision) != revision:
            raise ValueError(f'Unexpected dataset request: {repo} {kwargs}')
        kwargs['revision'] = revision
        if not args and 'name' not in kwargs:
            kwargs['name'] = 'default'  # Explicit original qrels config for offline cache.
        calls.append({'path': repo, 'revision': revision, 'args': list(args)})
        return original(repo, *args, **kwargs)

    module.load_dataset = pinned_loader
    try:
        task.load_data()
    finally:
        module.load_dataset = original
    if len(calls) != 3:
        raise ValueError(f'{name}: expected three pinned dataset reads, got {calls}')
    corpus, queries, qrels = task.corpus['test'], task.queries['test'], task.relevant_docs['test']
    if set(queries) != set(qrels):
        raise ValueError('Dataset query/qrels coverage mismatch')
    # ArguAna's pinned corpus lacks five positively judged documents. Preserve
    # those queries and judgments (as upstream does); NEVER inject documents or
    # drop difficult queries to increase the score.
    missing = [(q,d,s) for q,ds in qrels.items() for d,s in ds.items() if d not in corpus]
    return task, {'task': name, 'documents': len(corpus), 'queries': len(queries),
                  'rerank_pairs': len(queries)*100, 'dataset_reads': calls,
                  'judgments_missing_from_corpus':len(missing), 'missing_judgment_examples':missing[:10],
                  'split': 'test', 'subset': 'default',
                  'query_instruction': ENGLISH[name][2],
                  'document_instruction': ENGLISH[name][2] if ENGLISH[name][3] else ''}


def embedding_command(port):
    return [BASE_PYTHON, '-m', 'vllm.entrypoints.openai.api_server',
            '--model', EMBEDDING_MODEL, '--served-model-name', 'qwen3-embedding-0.6b',
            '--host', '127.0.0.1', '--port', str(port), '--runner', 'pooling', '--convert', 'embed',
            '--dtype', 'float16', '--max-model-len', '8192', '--pooler-config',
            '{"pooling_type":"LAST","use_activation":true}', '--enforce-eager',
            '--gpu-memory-utilization', '0.35', '--max-num-seqs', '128', '--block-size', '128',
            '--max-num-batched-tokens', '32768', '--no-enable-prefix-caching',
            '--no-enable-chunked-prefill', '--async-scheduling']


def validate_device_snapshot(snapshot, devices, allow_occupied=()):
    for device in devices:
        healthy = re.search(r'^\|\s*'+str(device)+r'\s+910B2\s*\|\s*OK\s*\|', snapshot, re.M)
        free = f'No running processes found in NPU {device}' in snapshot
        if not healthy or (not free and device not in allow_occupied):
            raise RuntimeError(f'NPU {device} is not confirmed healthy and idle; refusing to start')


@contextmanager
def servers(args, stage, observer):
    processes, logs, endpoints = [], [], []
    root = args.output / stage / f'servers_{time.time_ns()}'
    root.mkdir(parents=True)
    observer.state = {'stage': stage, 'section': 'server_start', 'devices': args.devices}
    try:
        snapshot = subprocess.check_output(['npu-smi','info'],text=True,timeout=120)
        (root/'device_preflight.txt').write_text(snapshot)
        validate_device_snapshot(snapshot,args.devices,args.allow_occupied_devices)
        for i, device in enumerate(args.devices):
            port = args.port + i
            with socket.socket() as probe:
                if probe.connect_ex(('127.0.0.1', port)) == 0:
                    raise RuntimeError(f'Port {port} occupied; will not touch its owner')
            cmd = embedding_command(port) if stage == 'embedding' else reranker_command(port)
            log = (root / f'npu_{device}.log').open('w')
            logs.append(log)
            env = {**os.environ, 'ASCEND_RT_VISIBLE_DEVICES': str(device)}
            p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                 env=env, start_new_session=True)
            processes.append(p)
            save(root / f'npu_{device}.json', {'pid': p.pid, 'device': device, 'command': cmd})
            endpoints.append(f'http://127.0.0.1:{port}')
        start = time.monotonic()
        pending = set(range(len(processes)))
        import urllib.request
        while pending and time.monotonic()-start < 900:
            for i in list(pending):
                if processes[i].poll() is not None:
                    raise RuntimeError(f'{stage}: NPU {args.devices[i]} server exited; see {root}')
                try:
                    with urllib.request.urlopen(endpoints[i]+'/health', timeout=1):
                        pending.remove(i)
                    emit('server_ready', stage=stage, device=args.devices[i], startup_s=time.monotonic()-start)
                except OSError:
                    pass
            if pending:
                time.sleep(1)
        if pending:
            raise TimeoutError(f'{stage}: servers not ready: {pending}')
        yield endpoints
    finally:
        for p in processes:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        for p in processes:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait(timeout=10)
        for log in logs:
            log.close()


def dispatch(items, endpoints, prepare, execute, consume):
    """Two bounded requests per server; refill whichever server finishes first."""
    iterator = iter(items)
    with ThreadPoolExecutor(max_workers=2*len(endpoints)) as pool:
        pending = {}

        def submit(endpoint):
            item = next(iterator, None)
            if item is not None:
                payload = prepare(item)
                pending[pool.submit(execute, endpoint, payload)] = endpoint

        for endpoint in endpoints:
            submit(endpoint)
            submit(endpoint)
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                endpoint = pending.pop(future)
                consume(future.result())
                submit(endpoint)


class Encoder:
    def __init__(self, args, endpoints, observer):
        import numpy as np
        from transformers import AutoTokenizer
        from mteb.model_meta import ModelMeta
        self.np, self.args, self.endpoints, self.observer = np, args, endpoints, observer
        self.tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL, local_files_only=True, padding_side='left')
        self.mteb_model_meta = ModelMeta(name='local/Qwen3-Embedding-0.6B-vllm-ascend-fp16',
            revision=MODEL_REVISION, release_date=None, languages=None, n_parameters=None,
            memory_usage_mb=None, max_tokens=8192, embed_dim=1024, license='apache-2.0',
            open_weights=True, public_training_code=None, public_training_data=None,
            framework=['PyTorch'], similarity_fn_name='cosine', use_instructions=True, training_datasets=None)
        self.cache = args.output / 'embedding' / 'batch_cache'
        self.cache.mkdir(parents=True, exist_ok=True)
        self.log = (args.output / 'embedding' / 'batches.jsonl').open('a')
        self.totals = {'texts': 0, 'tokens': 0, 'truncated': 0, 'cache_hits': 0}

    def encode(self, sentences, *, task_name, prompt_type=None, **kwargs):
        role = getattr(prompt_type, 'value', prompt_type)
        np = self.np
        vectors = np.empty((len(sentences), 1024), dtype=np.float32)
        start = time.monotonic()
        state = {'stage': 'embedding', 'section': 'encoding', 'task': task_name,
                 'role': role, 'total_texts': len(sentences), 'completed_texts': 0, 'tokens': 0}
        self.observer.state = state

        def prepare(offset):
            texts = [format_embedding(t, task_name, role) for t in sentences[offset:offset+128]]
            key = hashlib.sha256(json.dumps([task_name, role, texts], ensure_ascii=False).encode()).hexdigest()
            ids = self.tokenizer(texts, add_special_tokens=True, padding=False, truncation=False)['input_ids']
            truncated = sum(len(x)>8192 for x in ids)
            ids = [x[:8192] for x in ids]
            if any(not x for x in ids):
                raise ValueError('Empty tokenized input')
            return offset, key, ids, truncated

        def execute(endpoint, prepared):
            offset, key, ids, truncated = prepared
            begin = time.monotonic()
            cache_path = self.cache / f'{key}.npy'
            hit = cache_path.exists()
            if hit:
                values = np.load(cache_path, allow_pickle=False)
            else:
                result = request(endpoint+'/v1/embeddings', {'model':'qwen3-embedding-0.6b',
                    'input': ids, 'encoding_format':'base64'})
                rows = sorted(result['data'], key=lambda r:r['index'])
                if [r['index'] for r in rows] != list(range(len(ids))):
                    raise ValueError('Missing/reordered embedding indices')
                values = np.stack([np.frombuffer(base64.b64decode(r['embedding'], validate=True), dtype='<f4') for r in rows])
                if result['usage']['prompt_tokens'] != sum(map(len, ids)):
                    raise ValueError('Embedding token usage mismatch')
            if values.shape != (len(ids), 1024) or not np.isfinite(values).all():
                raise ValueError('Invalid embeddings')
            if float(np.abs(np.linalg.norm(values, axis=1)-1).max()) > .005:
                raise ValueError('Embeddings not unit normalized')
            if not hit:
                partial = cache_path.with_suffix(f'.partial.{threading.get_ident()}')
                with partial.open('wb') as stream:
                    np.save(stream, values, allow_pickle=False)
                partial.replace(cache_path)
            return offset, values, {'task': task_name, 'role': role, 'offset': offset,
                'texts':len(ids), 'tokens':sum(map(len,ids)), 'lengths':list(map(len,ids)),
                'truncated':truncated, 'cache_hit':hit, 'endpoint':endpoint,
                'request_wall_s':time.monotonic()-begin}

        def consume(result):
            offset, values, record = result
            vectors[offset:offset+len(values)] = values
            self.log.write(json.dumps(record)+'\n')
            self.log.flush()
            state['completed_texts'] += len(values)
            state['tokens'] += record['tokens']
            for k in ('texts','tokens','truncated'):
                self.totals[k] += record[k]
            self.totals['cache_hits'] += int(record['cache_hit'])
            state.update(elapsed_s=time.monotonic()-start)
            emit('embedding_batch_finished', **state, cache_hit=record['cache_hit'],
                 request_wall_s=record['request_wall_s'], tok_s=state['tokens']/state['elapsed_s'])

        dispatch(range(0,len(sentences),128), self.endpoints, prepare, execute, consume)
        self.observer.state = {'stage':'embedding','section':'retrieval_scoring','task':task_name}
        return vectors


def tokenize_rerank(tokenizer, name, rows, input_order="query_first"):
    prefix = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix = tokenizer.encode(SUFFIX, add_special_tokens=False)
    limit = MAX_LENGTH-len(prefix)-len(suffix)
    from training_smoke_data import body
    texts = [body(ENGLISH[name][2], p['query'], p['document'], input_order) for p in rows]
    ids = tokenizer(texts, add_special_tokens=False, padding=False, truncation=False)['input_ids']
    return [prefix+x[:limit]+suffix for x in ids], sum(len(x)>limit for x in ids)


def evaluate_embedding(args, encoder, observer, name):
    import mteb
    folder = args.output / 'embedding' / name
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / 'result.json'
    if result_path.exists():
        row = json.loads(result_path.read_text())
        if digest(folder/'mteb'/f'{name}_default_predictions.json') != row['candidates_sha256']:
            raise ValueError('Completed candidate file changed')
        emit('task_reused', stage='embedding', task=name)
        return row
    task, counts = load_task(name, observer)
    start = time.monotonic()
    # overwrite_results recomputes only incomplete task metrics, reusing exact
    # content-addressed batches; it never reruns inference from completed tasks.
    result = mteb.MTEB(tasks=[task]).run(encoder, output_folder=str(folder/'mteb'),
        eval_splits=['test'], encode_kwargs={'batch_size':128}, corpus_chunk_size=8192,
        top_k=100, save_predictions=True, raise_error=True, overwrite_results=True)
    scores = result[0].scores['test']
    if len(scores) != 1 or scores[0]['hf_subset'] != 'default':
        raise ValueError('Unexpected result subsets')
    candidates_path = folder/'mteb'/f'{name}_default_predictions.json'
    candidates = json.loads(candidates_path.read_text())
    if set(candidates) != set(task.queries['test']) or any(len(v)!=100 for v in candidates.values()):
        raise ValueError('Incomplete top100 candidate coverage')
    row = {**counts, 'ndcg_at_10':scores[0]['ndcg_at_10'],
           'recall_at_100':scores[0]['recall_at_100'], 'evaluation_wall_s':time.monotonic()-start,
           'candidates_sha256':digest(candidates_path)}
    save(result_path, row)
    emit('task_finished', stage='embedding', **row)
    return row


def evaluate_reranker(args, tok, endpoints, observer, name):
    import numpy as np
    from mteb.evaluation.evaluators.RetrievalEvaluator import corpus_to_str
    folder = args.output/'reranker'/name
    folder.mkdir(parents=True, exist_ok=True)
    candidate_path = args.output/'embedding'/name/'mteb'/f'{name}_default_predictions.json'
    candidate_digest = digest(candidate_path)
    if (folder/'result.json').exists():
        row = json.loads((folder/'result.json').read_text())
        if row['candidates_sha256'] != candidate_digest:
            raise ValueError('Reranker candidates changed')
        emit('task_reused', stage='reranker', task=name)
        return row
    task, counts = load_task(name, observer)
    baseline = json.loads(candidate_path.read_text())
    manifest = {**counts, 'candidates_sha256':candidate_digest, 'max_length':8192,
                'prompt':'HF Transformers single-newline prefix/body/suffix; task-specific instruction',
                'input_order':args.input_order}
    if (folder/'manifest.json').exists() and json.loads((folder/'manifest.json').read_text()) != manifest:
        raise ValueError('Reranker manifest mismatch')
    save(folder/'manifest.json', manifest)
    journal = folder/'requests.jsonl'
    predictions, records = restore_journal(journal) if journal.exists() else ({}, [])
    for q, docs in predictions.items():
        if q not in baseline or not set(docs).issubset(baseline[q]):
            raise ValueError('Unexpected restored pair')
    restored = sum(len(v) for v in predictions.values())
    restored_tokens = sum(r['tokens'] for r in records)
    start = time.monotonic()
    state = {'stage':'reranker','task':name,'section':'scoring','total_pairs':counts['rerank_pairs'],
             'completed_pairs':restored,'tokens':restored_tokens,'restored_pairs':restored}
    observer.state = state
    rows = candidate_rows(baseline, task.queries['test'], task.corpus['test'], corpus_to_str)

    def prepare(rows):
        ids, truncated = tokenize_rerank(tok, name, rows, args.input_order)
        return rows, ids, truncated

    def execute(endpoint, payload):
        rows, ids, truncated = payload
        t0 = time.monotonic()
        values = score(endpoint, ids)
        return rows, values, {'pairs':len(rows),'lengths':list(map(len,ids)),
            'tokens':sum(map(len,ids)),'truncated':truncated,'http_s':time.monotonic()-t0,'endpoint':endpoint}

    with journal.open('a') as log:
        def consume(result):
            rows, values, record = result
            for r, v in zip(rows,values):
                target = predictions.setdefault(r['qid'],{})
                if r['did'] in target:
                    raise ValueError('Duplicate result')
                target[r['did']] = v
            log.write(json.dumps({**record,'pairs_scored':[[r['qid'],r['did'],v] for r,v in zip(rows,values)]})+'\n')
            log.flush()
            records.append(record)
            state['completed_pairs'] += record['pairs']
            state['tokens'] += record['tokens']
            elapsed = time.monotonic()-start
            rate = (state['completed_pairs']-restored)/elapsed
            state.update(elapsed_s=elapsed, pairs_s=rate,
                         input_tok_s=(state['tokens']-restored_tokens)/elapsed,
                         eta_s=(state['total_pairs']-state['completed_pairs'])/rate)
            emit('reranker_batch_finished', **state)
        dispatch(batches(unscored_rows(rows,predictions),128), endpoints,prepare,execute,consume)
    metrics, per_query = metric_summary(predictions,baseline,task.relevant_docs['test'],task.ignore_identical_ids)
    save(folder/'predictions.json',predictions)
    save(folder/'per_query_metrics.json',per_query)
    lengths = [n for r in records for n in r['lengths']]
    row = {**counts,'candidates_sha256':candidate_digest,
           'ndcg_at_10':metrics['reranker']['ndcg_cut_10'],
           'recall_at_100':metrics['reranker']['recall_100'], 'metrics':metrics,
           'scoring_wall_s':time.monotonic()-start,'restored_pairs':restored,
           'tokens':sum(r['tokens'] for r in records),'truncated_pairs':sum(r['truncated'] for r in records),
           'length_percentiles':{str(p):float(np.percentile(lengths,p)) for p in (50,95,99,100)},
           'batch_latency_s':{str(p):float(np.percentile([r['http_s'] for r in records],p)) for p in (50,95,99,100)}}
    save(folder/'result.json',row)
    emit('task_finished',stage='reranker',**row)
    return row


def reuse_embedding(source, output, contract):
    """Copy only verified historical top100 inputs; never old reranker outputs."""
    source = Path(source)
    original = json.loads((source/'manifest.json').read_text())
    for key in ('benchmark', 'mteb', 'tasks', 'embedding_revision',
                'embedding_max_length', 'dtype', 'top_k'):
        if original[key] != contract[key]:
            raise ValueError(f'Saved embedding contract differs: {key}')
    summary = json.loads((source/'embedding/summary.json').read_text())
    if not summary['complete'] or set(summary['completed_tasks']) != set(ENGLISH):
        raise ValueError('Saved embedding suite is incomplete')
    files = ['summary.json', 'results.json']
    hashes = {}
    for name in ENGLISH:
        candidate = f'{name}/mteb/{name}_default_predictions.json'
        result = json.loads((source/'embedding'/name/'result.json').read_text())
        if digest(source/'embedding'/candidate) != result['candidates_sha256']:
            raise ValueError(f'Saved candidates changed: {name}')
        files += [candidate, f'{name}/result.json']
    for name in files:
        src, dst = source/'embedding'/name, output/'embedding'/name
        hashes[name] = digest(src)
        if dst.exists():
            if digest(dst) != hashes[name]:
                raise ValueError(f'Existing saved input differs: {name}')
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    save(output/'saved_embedding_provenance.json',
         {'source':str(source), 'manifest_sha256':digest(source/'manifest.json'),
          'files_sha256':hashes})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--devices',type=int,nargs='+',default=[0,1,2,3,6])
    p.add_argument('--port',type=int,default=18530)
    p.add_argument('--allow-occupied-devices',type=int,nargs='*',default=[],
                   help='Explicitly authorized device sharing; does not bypass health or runtime HBM checks')
    p.add_argument('--reranker-model', default=RERANKER_MODEL)
    p.add_argument('--input-order', choices=['query_first','document_first'], default='query_first')
    p.add_argument('--reranker-reference', type=float, default=69.76)
    p.add_argument('--saved-embedding', type=Path)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--prepared',type=Path,default=Path('tmp/22_qwen3_embedding_benchmark/reranker_serving_cc6c6841/prepared'))
    args = p.parse_args()
    reranker_evaluator.MODEL = args.reranker_model
    if len(set(args.devices))!=len(args.devices) or set(args.devices)-{0,1,2,3,6}:
        p.error('Only the five reserved healthy devices may be used')
    if set(args.allow_occupied_devices)-set(args.devices):
        p.error('Shared devices must be selected evaluation devices')
    if args.output.exists() and not args.resume:
        p.error('Output exists; use explicit --resume')
    args.output.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    observer=Observer()
    observer.thread.start()
    started=time.monotonic()
    try:
        import mteb
        import torch
        from transformers import AutoTokenizer
        if importlib.metadata.version('mteb')!=MTEB_VERSION:
            raise ValueError('Wrong evaluator version')
        torch.set_num_threads(8)
        tasks=[t for t in mteb.get_benchmark(BENCHMARK).tasks if t.metadata.type=='Retrieval']
        validate_tasks(tasks)
        contract={'schema':1,'benchmark':BENCHMARK,'mteb':MTEB_VERSION,'qwen_commit':QWEN_COMMIT,
                  'reranker_model':args.reranker_model,'reranker_published_percent':args.reranker_reference,
                  'input_order':args.input_order,
                  'model_files':{p.name:digest(p) for p in sorted(Path(args.reranker_model).glob('*.safetensors'))},
                  'allow_occupied_devices':args.allow_occupied_devices,
                  'saved_embedding':str(args.saved_embedding) if args.saved_embedding else None,
                  'tasks':ENGLISH,'embedding_revision':MODEL_REVISION,'devices':args.devices,
                  'embedding_max_length':8192,'reranker_max_length':8192,
                  'dtype':'float16','compiled':False,'top_k':100,
                  'embedding_server':embedding_command(args.port),'reranker_server':reranker_command(args.port),
                  'source_files':{n:digest(Path(__file__).parent/n) for n in
                    ['suite_protocol.py','run_english_suite.py','reranker_protocol.py']}}
        # JSON normalizes tuple values into lists before equality checks.
        contract=json.loads(json.dumps(contract))
        manifest=args.output/'manifest.json'
        if manifest.exists() and json.loads(manifest.read_text())!=contract:
            raise ValueError('Resume contract differs')
        save(manifest,contract)
        save(args.output/f'invocation_{time.time_ns()}.json',{'argv':sys.argv,'hostname':socket.gethostname(),
             'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'time':time.time()})
        if args.saved_embedding:
            reuse_embedding(args.saved_embedding,args.output,contract)
            emit('saved_embedding_verified',source=str(args.saved_embedding))
        inventory=[]
        for name in ENGLISH:
            _, counts=load_task(name,observer)
            inventory.append(counts)
            save(args.output/'inventory.json',inventory)
            emit('dataset_ready',**counts)
        if args.prepare_only:
            emit('preparation_complete',tasks=len(inventory),documents=sum(x['documents'] for x in inventory),
                 queries=sum(x['queries'] for x in inventory),rerank_pairs=sum(x['rerank_pairs'] for x in inventory))
            return
        observer.state={'section':'checkpoint_verification'}
        verified=subprocess.check_output([BASE_PYTHON,str(Path(__file__).with_name('verify_checkpoint.py')),EMBEDDING_MODEL],text=True)
        save(args.output/'embedding_checkpoint.json',json.loads(verified))
        for stage in ('embedding','reranker'):
            if (args.output/stage/'summary.json').exists() and json.loads((args.output/stage/'summary.json').read_text())['complete']:
                emit('stage_already_complete',stage=stage)
                continue
            with servers(args,stage,observer) as endpoints:
                encoder=None
                if stage=='embedding':
                    encoder=Encoder(args,endpoints,observer)
                else:
                    tok=AutoTokenizer.from_pretrained(args.reranker_model,local_files_only=True)
                    parity(args,tok,endpoints,observer,
                           tokenizer_fn=lambda tokenizer,name,pairs: tokenize_rerank(tokenizer,name,pairs,args.input_order))
                rows=[]
                try:
                    for name in ENGLISH:
                        row=(evaluate_embedding(args,encoder,observer,name) if stage=='embedding' else
                             evaluate_reranker(args,tok,endpoints,observer,name))
                        rows.append(row)
                        summary=aggregate(rows,stage,published_percent=args.reranker_reference if stage=='reranker' else None)
                        save(args.output/stage/'results.json',rows)
                        save(args.output/stage/'summary.json',summary)
                        emit('suite_summary',**summary)
                finally:
                    if encoder:
                        encoder.log.close()
                        save(args.output/stage/'encoding_totals.json',encoder.totals)
        save(args.output/'completion.json',{'status':'complete','wall_s_this_invocation':time.monotonic()-started})
    except BaseException as exc:
        save(args.output/'failure.json',{'error':repr(exc),'state':observer.state,'time':time.time()})
        raise
    finally:
        observer.stop.set()


if __name__=='__main__':
    main()
