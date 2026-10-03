"""Real-text -> embeddings -> retrieval sweep on one separate NPU.

All cases remain FP16 with the same model, prompts, 8192-token limit and
scoring. Setup/data loading and tiny full-pipeline warmup are reported separately.
This does not change the active accuracy server on port 18222.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import threading
import time
import urllib.request

from protocol import format_text, validate_task
from run_evaluation import Observer, emit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--subset', type=Path, required=True)
    parser.add_argument('--experiment', choices=['batching', 'graph', 'async'], default='batching')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output must be fresh')
    args.output.mkdir(parents=True)
    observer = Observer()
    observer.thread.start()
    server = None
    server_log = None
    log = (args.output / 'requests.jsonl').open('w')
    try:
        import numpy as np
        import torch
        import mteb
        import pytrec_eval
        from transformers import AutoTokenizer
        from mteb.evaluation.evaluators.RetrievalEvaluator import corpus_to_str, DenseRetrievalExactSearch
        torch.set_num_threads(8)
        model_path = '/workspace/models/Qwen3-Embedding-0.6B'
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, padding_side='left')
        lock = threading.Lock()
        subset = json.loads((args.subset / 'manifest.json').read_text())
        workloads = []
        for name in ['EcomRetrieval', 'CmedqaRetrieval', 'T2Retrieval']:
            observer.state = {'section': 'loading_data', 'task': name}
            task = mteb.get_tasks(tasks=[name])[0]
            validate_task(task)
            task.load_data()
            rng = random.Random(20261003)
            if name == 'EcomRetrieval':
                qids, dids = subset['qids'], subset['dids']
            else:
                qids = sorted(rng.sample(sorted(task.queries['dev']), 64))
                dids = sorted(set(rng.sample(sorted(task.corpus['dev']), 2048)) |
                              {d for q in qids for d in task.relevant_docs['dev'][q]})
            w = {'task': name, 'qids': qids, 'dids': dids,
                 'queries': [format_text(task.queries['dev'][q], name, 'query') for q in qids],
                 'documents': corpus_to_str([task.corpus['dev'][d] for d in dids]),
                 'qrels': {q: task.relevant_docs['dev'][q] for q in qids},
                 'ignore_identical_ids': task.ignore_identical_ids}
            workloads.append(w)
        (args.output / 'workloads.json').write_text(json.dumps(workloads, ensure_ascii=False) + '\n')
        manifest = {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    'device': os.environ.get('ASCEND_RT_VISIBLE_DEVICES'), 'chip': '910B2',
                    'dtype': 'float16', 'experiment': args.experiment, 'max_model_len': 8192,
                    'workloads_sha256': hashlib.sha256((args.output / 'workloads.json').read_bytes()).hexdigest(),
                    'timing_scope': 'prepared real text -> tokenization -> HTTP embeddings -> exact retrieval metrics; excludes model/data setup',
                    'quality_scope': 'fixed reduced corpora; not published full-corpus benchmark scores'}
        (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        emit('workloads_ready', tasks=[{'task': w['task'], 'queries': len(w['qids']), 'documents': len(w['dids'])} for w in workloads])

        def encode(texts, client, context):
            def one(offset):
                start = time.monotonic()
                with lock:
                    ids = tokenizer(texts[offset:offset + client['batch']], padding=False,
                                    truncation=True, max_length=8192)['input_ids']
                payload = json.dumps({'model': 'embedding-batching-diagnostic', 'input': ids,
                                      'encoding_format': client['format']}).encode()
                request = urllib.request.Request('http://127.0.0.1:18224/v1/embeddings', data=payload,
                                                 headers={'Content-Type': 'application/json'})
                http_start = time.monotonic()
                with urllib.request.urlopen(request, timeout=300) as response:
                    raw = response.read()
                rows = sorted(json.loads(raw)['data'], key=lambda row: row['index'])
                if [row['index'] for row in rows] != list(range(len(ids))):
                    raise ValueError('Missing or reordered output indices')
                if client['format'] == 'base64':
                    vectors = np.stack([np.frombuffer(base64.b64decode(row['embedding'], validate=True), dtype='<f4') for row in rows])
                else:
                    vectors = np.asarray([row['embedding'] for row in rows], dtype=np.float32)
                if vectors.shape != (len(ids), 1024) or not np.isfinite(vectors).all():
                    raise ValueError('Invalid embeddings')
                if float(np.abs(np.linalg.norm(vectors, axis=1) - 1).max()) > .005:
                    raise ValueError('Non-unit embeddings')
                record = {**context, 'offset': offset, 'texts': len(ids), 'tokens': sum(map(len, ids)),
                          'token_lengths': list(map(len, ids)), 'http_decode_s': time.monotonic() - http_start,
                          'batch_wall_s': time.monotonic() - start, 'response_bytes': len(raw)}
                with lock:
                    log.write(json.dumps(record) + '\n')
                    log.flush()
                return vectors, record['tokens'], record['response_bytes']

            with ThreadPoolExecutor(max_workers=client['concurrency']) as executor:
                results = list(executor.map(one, range(0, len(texts), client['batch'])))
            return np.concatenate([r[0] for r in results]), sum(r[1] for r in results), sum(r[2] for r in results)

        reference = {}
        details = {}
        results = []

        def pipeline(w, client, context, warmup=False):
            observer.state = {**context, 'task': w['task'], 'section': 'encoding', 'warmup': warmup}
            start = time.monotonic()
            q, qt, qb = encode(w['queries'], client, {**context, 'task': w['task'], 'role': 'query', 'warmup': warmup})
            d, dt, db = encode(w['documents'], client, {**context, 'task': w['task'], 'role': 'passage', 'warmup': warmup})
            encoding_s = time.monotonic() - start
            observer.state['section'] = 'scoring'
            qmap, dmap = dict(zip(w['qids'], q)), dict(zip(w['dids'], d))

            class RecordedEncoder:
                def encode(self, sentences, *, prompt_type, **kwargs):
                    role = getattr(prompt_type, 'value', prompt_type)
                    return np.stack([qmap[x] for x in sentences] if role == 'query'
                                    else [dmap[x['text']] for x in sentences])

            run = DenseRetrievalExactSearch(RecordedEncoder(), corpus_chunk_size=4096).search(
                {x: {'text': x} for x in w['dids']}, {x: x for x in w['qids']}, 1000, w['task'])
            if w['ignore_identical_ids']:
                for qid in w['qids']:
                    run[qid].pop(qid, None)
            per_query = pytrec_eval.RelevanceEvaluator(w['qrels'], {'recall.10,100', 'ndcg_cut.10'}).evaluate(run)
            pipeline_s = time.monotonic() - start
            metrics = {k: sum(r[k] for r in per_query.values()) / len(per_query)
                       for k in ['recall_10', 'recall_100', 'ndcg_cut_10']}
            row = {**context, 'task': w['task'], 'client': client, 'tokens': qt + dt,
                   'texts': len(q) + len(d), 'response_bytes': qb + db,
                   'encoding_wall_s': encoding_s, 'pipeline_wall_s': pipeline_s,
                   'encoding_tok_s': (qt + dt) / encoding_s, 'pipeline_tok_s': (qt + dt) / pipeline_s,
                   'texts_s': (len(q) + len(d)) / encoding_s, 'metrics': metrics}
            if not warmup:
                if w['task'] not in reference:
                    reference[w['task']] = (q.copy(), d.copy(), per_query)
                rq, rd, rs = reference[w['task']]
                cosine = np.sum(d * rd, axis=1) / (np.linalg.norm(d, axis=1) * np.linalg.norm(rd, axis=1))
                row['corpus_cosine_min_vs_baseline'] = float(cosine.min())
                row['metric_changed_queries_vs_baseline'] = sum(per_query[x] != rs[x] for x in w['qids'])
                details[f"{context['case']}/{context['client_name']}/{context['repeat']}/{w['task']}"] = per_query
                results.append(row)
                (args.output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
                (args.output / 'per_query.json').write_text(json.dumps(details, indent=2) + '\n')
            emit('pipeline_finished', warmup=warmup, **row)

        def stop_server():
            nonlocal server, server_log
            if server is not None:
                if server.poll() is None:
                    os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait(timeout=10)
                server = None
            if server_log is not None:
                server_log.close()
                server_log = None

        default = {'batch': 128, 'concurrency': 1, 'format': 'float'}
        default_name = 'float_b128_c1'
        cases = [('s32_t16k', 32, 16384, False), ('s64_t16k', 64, 16384, False),
                 ('s128_t16k', 128, 16384, False), ('s128_t32k', 128, 32768, False),
                 ('s32_t16k_recheck', 32, 16384, False)]
        if args.experiment in ('graph', 'async'):
            default = {'batch': 128, 'concurrency': 2, 'format': 'base64'}
            default_name = 'base64_b128_c2'
            cases = [('eager_before', 128, 32768, False),
                     ('piecewise', 128, 32768, True), ('eager_after', 128, 32768, False)]
            if args.experiment == 'async':
                cases = [('sync_before', 128, 32768, False),
                         ('async_eager', 128, 32768, False), ('sync_after', 128, 32768, False)]
        for case, seqs, budget, graph in cases:
            observer.state = {'case': case, 'section': 'server_start'}
            case_dir = args.output / case
            case_dir.mkdir()
            command = ['/usr/local/python3.12.13/bin/python3', '-m', 'vllm.entrypoints.openai.api_server',
                       '--model', model_path, '--served-model-name', 'embedding-batching-diagnostic',
                       '--host', '127.0.0.1', '--port', '18224', '--runner', 'pooling', '--convert', 'embed',
                       '--dtype', 'float16', '--max-model-len', '8192', '--pooler-config',
                       '{"pooling_type":"LAST","use_activation":true}',
                       '--gpu-memory-utilization', '0.35', '--max-num-seqs', str(seqs), '--block-size', '128',
                       '--max-num-batched-tokens', str(budget), '--no-enable-prefix-caching', '--no-enable-chunked-prefill']
            if graph:
                command += ['--compilation-config', json.dumps({
                    'mode': 3, 'cudagraph_mode': 'PIECEWISE',
                    'cudagraph_capture_sizes': [1024, 4096, 8192, 16384, 32768],
                    'max_cudagraph_capture_size': 32768,
                })]
            else:
                command += ['--enforce-eager']
            if args.experiment == 'graph':
                command += ['--cudagraph-metrics']
            if args.experiment == 'async':
                command += ['--async-scheduling' if case == 'async_eager' else '--no-async-scheduling']
            (case_dir / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
            server_log = (case_dir / 'server.log').open('w')
            server = subprocess.Popen(command, stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
            started = time.monotonic()
            try:
                for _ in range(180):
                    if server.poll() is not None:
                        raise RuntimeError(f'Server failed: {case}; see server.log')
                    try:
                        with urllib.request.urlopen('http://127.0.0.1:18224/health', timeout=2):
                            break
                    except OSError:
                        time.sleep(5)
                else:
                    raise TimeoutError('Server readiness timeout')
                emit('server_ready', case=case, startup_wall_s=time.monotonic() - started)
                warm = dict(workloads[0])
                warm.update(qids=warm['qids'][:8], queries=warm['queries'][:8],
                            dids=warm['dids'][:128], documents=warm['documents'][:128],
                            qrels={q: warm['qrels'][q] for q in warm['qids'][:8]})
                pipeline(warm, default, {'case': case, 'client_name': default_name, 'repeat': -1}, warmup=True)
                clients = [(default_name, default)]
                if args.experiment == 'batching' and case == 's128_t32k':
                    clients += [('base64_b128_c1', {'batch': 128, 'concurrency': 1, 'format': 'base64'}),
                                ('base64_b128_c2', {'batch': 128, 'concurrency': 2, 'format': 'base64'}),
                                ('base64_b512_c2', {'batch': 512, 'concurrency': 2, 'format': 'base64'})]
                for client_name, client in clients:
                    for repeat in range(2):
                        for w in workloads:
                            pipeline(w, client, {'case': case, 'client_name': client_name, 'repeat': repeat})
                with urllib.request.urlopen('http://127.0.0.1:18224/metrics', timeout=10) as response:
                    (case_dir / 'metrics.txt').write_bytes(response.read())
            finally:
                stop_server()
        (args.output / 'summary.json').write_text(json.dumps({'complete': True, 'manifest': manifest, 'results': results}, indent=2) + '\n')
        emit('sweep_complete', trials=len(results))
    except BaseException as exc:
        (args.output / 'failure.json').write_text(json.dumps({'error': repr(exc), 'state': observer.state}, indent=2))
        raise
    finally:
        if server is not None and server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(timeout=10)
        if server_log is not None:
            server_log.close()
        observer.stop.set()
        log.close()


if __name__ == '__main__':
    main()
