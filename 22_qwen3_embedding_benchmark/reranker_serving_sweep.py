"""FP16/eager real CMTEB top-100 -> reranker HTTP -> rankings on one 910B2.

prepare: pinned datasets, deterministic query selection and HF FP16 reference.
sweep: built-in vLLM classifier conversion, fixed prompts, no shortened inputs.
All timings include client tokenization and HTTP scoring. No isolated kernels.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request

from protocol import TASKS, validate_task
from reranker_protocol import MAX_LENGTH, PREFIX, SUFFIX, tokenize_pairs
from run_evaluation import Observer, emit

MODEL = '/workspace/models/Qwen3-Reranker-4B'
BASE_PYTHON = '/usr/local/python3.12.13/bin/python3'
NAMES = ['EcomRetrieval', 'CmedqaRetrieval', 'T2Retrieval']


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(args, observer):
    import mteb
    import torch
    import torch_npu
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from mteb.evaluation.evaluators.RetrievalEvaluator import corpus_to_str
    assert importlib.metadata.version('mteb') == '1.38.9'
    torch.set_num_threads(8)
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, padding_side='left')
    workloads = []
    manifest = {'chip': '910B2', 'device': os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
                'model': MODEL, 'dtype': 'float16', 'max_length': MAX_LENGTH,
                'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'prompt_source': 'Qwen3-Reranker-4B HF Transformers example; task instructions from pinned Qwen evaluation',
                'prefix': PREFIX, 'suffix': SUFFIX, 'candidate_files': {},
                'model_files': {n: digest(Path(MODEL) / n) for n in
                                ['config.json', 'tokenizer_config.json', 'model.safetensors.index.json']},
                'quality_scope': '8 deterministic dev queries per task, complete saved top100 from full corpus; not published aggregate'}
    for name in NAMES:
        observer.state = {'section': 'loading_data', 'task': name}
        task = mteb.get_tasks(tasks=[name])[0]
        validate_task(task)
        task.load_data()
        path = args.candidates / f'{name}_default_predictions.json'
        candidates = json.loads(path.read_text())
        manifest['candidate_files'][name] = {'path': str(path), 'sha256': digest(path)}
        qids = sorted(random.Random(20261003).sample(sorted(candidates), args.queries))
        pairs = []
        for qid in qids:
            dids = sorted(candidates[qid], key=lambda d: (-candidates[qid][d], d))[:100]
            assert len(dids) == 100
            docs = corpus_to_str([task.corpus['dev'][d] for d in dids])
            for did, doc in zip(dids, docs):
                pairs.append({'qid': qid, 'did': did, 'query': task.queries['dev'][qid],
                              'document': doc, 'retrieval_score': candidates[qid][did]})
        ids, truncated = tokenize_pairs(tok, name, pairs)
        lengths = list(map(len, ids))
        w = {'task': name, 'pairs': pairs, 'qids': qids,
             'qrels': {q: task.relevant_docs['dev'][q] for q in qids},
             'ignore_identical_ids': task.ignore_identical_ids,
             'lengths': lengths, 'truncated': truncated,
             'input_ids_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest()}
        workloads.append(w)
        emit('workload_ready', task=name, queries=len(qids), pairs=len(pairs),
             tokens=sum(lengths), mean_tokens=sum(lengths)/len(lengths), max_tokens=max(lengths))
    save(args.output / 'workloads.json', workloads)
    manifest['workloads_sha256'] = digest(args.output / 'workloads.json')
    save(args.output / 'manifest.json', manifest)
    observer.state = {'section': 'hf_reference_load'}
    assert torch_npu.npu.is_available(), 'NPU required, no CPU inference fallback'
    torch_npu.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True,
                torch_dtype=torch.float16, attn_implementation='eager').eval().to('npu:0')
    dtypes = sorted({str(p.dtype) for p in model.parameters()})
    assert dtypes == ['torch.float16'], dtypes
    yes, no = tok.convert_tokens_to_ids('yes'), tok.convert_tokens_to_ids('no')
    refs = []
    with torch.inference_mode():
        for w in workloads:
            # Include high and low retrieval ranks, independent of gold labels.
            indices = [q * 100 + rank for q in range(min(4, args.queries)) for rank in (0, 1, 10, 99)]
            ids, _ = tokenize_pairs(tok, w['task'], [w['pairs'][i] for i in indices])
            scores = []
            for j, row in enumerate(ids):
                observer.state = {'section': 'hf_reference', 'task': w['task'], 'done': j, 'total': len(ids)}
                batch = {'input_ids': torch.tensor([row], device='npu:0'),
                         'attention_mask': torch.ones((1, len(row)), dtype=torch.long, device='npu:0')}
                output = model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1, :]
                assert output.dtype == torch.float16
                # Exact model-card arithmetic (two-token log_softmax then exp).
                score = torch.nn.functional.log_softmax(output[:, [no, yes]], dim=-1)[:, 1].exp()
                scores.append(float(score.cpu()[0]))
            refs.append({'task': w['task'], 'indices': indices, 'scores': scores,
                         'parameter_dtypes': dtypes, 'logit_dtype': str(output.dtype)})
            emit('hf_reference_finished', task=w['task'], pairs=len(ids))
    save(args.output / 'hf_reference.json', refs)


def request(endpoint, payload):
    req = urllib.request.Request(endpoint, data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f'HTTP {exc.code}: {exc.read().decode()}') from exc


def scores_from_response(data, count):
    import numpy as np
    rows = sorted(data['data'], key=lambda r: r['index'])
    if [r['index'] for r in rows] != list(range(count)):
        raise ValueError('Missing/duplicate response indices')
    result = np.asarray([r['data'] for r in rows], dtype=np.float32)
    if result.shape != (count, 1) or not np.isfinite(result).all():
        raise ValueError(f'Invalid score shape/values: {result.shape}')
    if (result < 0).any() or (result > 1).any():
        raise ValueError('Activated probability out of range')
    return result[:, 0]


def sweep(args, observer):
    import numpy as np
    import pytrec_eval
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    workloads = json.loads((args.prepared / 'workloads.json').read_text())
    prep_manifest = json.loads((args.prepared / 'manifest.json').read_text())
    assert digest(args.prepared / 'workloads.json') == prep_manifest['workloads_sha256']
    refs = {r['task']: r for r in json.loads((args.prepared / 'hf_reference.json').read_text())}
    for w in workloads:
        ids, _ = tokenize_pairs(tok, w['task'], w['pairs'])
        assert hashlib.sha256(json.dumps(ids).encode()).hexdigest() == w['input_ids_sha256']
    manifest = {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'prepared': str(args.prepared), 'workloads_sha256': prep_manifest['workloads_sha256'],
                'device': os.environ.get('ASCEND_RT_VISIBLE_DEVICES'), 'chip': '910B2',
                'dtype': 'float16', 'eager': True, 'max_length': MAX_LENGTH,
                'timing': 'real text -> client tokenization -> HTTP pooling/classify -> rankings; setup and warmup separate',
                'prefix_cache': False}
    save(args.output / 'manifest.json', manifest)
    endpoint = f'http://127.0.0.1:{args.port}'
    # Refuse to benchmark against an unrelated existing server.
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1', args.port)) == 0:
            raise RuntimeError(f'Port {args.port} already occupied')
    lock = threading.Lock()
    logs = (args.output / 'requests.jsonl').open('w')
    results, baseline = [], {}

    def run(w, batch_size, concurrency, context, warmup=False, indices=None):
        pairs = w['pairs'] if indices is None else [w['pairs'][i] for i in indices]
        total = len(pairs)
        started = time.monotonic()
        observer.state = {**context, 'task': w['task'], 'section': 'tokenize_score', 'completed': 0, 'pairs': total}

        def one(offset):
            t0 = time.monotonic()
            with lock:
                ids, truncated = tokenize_pairs(tok, w['task'], pairs[offset:offset + batch_size])
            t1 = time.monotonic()
            data = request(endpoint + '/pooling', {'model': 'reranker-diagnostic', 'input': ids,
                'task': 'classify', 'use_activation': True, 'encoding_format': 'float', 'add_special_tokens': False})
            scores = scores_from_response(data, len(ids))
            token_count = sum(map(len, ids))
            if data['usage']['prompt_tokens'] != token_count:
                raise ValueError('Server token count differs from explicit input')
            record = {**context, 'task': w['task'], 'warmup': warmup, 'offset': offset,
                      'pairs': len(ids), 'tokens': token_count, 'lengths': list(map(len, ids)),
                      'truncated': truncated, 'tokenize_s': t1-t0, 'http_s': time.monotonic()-t1,
                      'batch_s': time.monotonic()-t0}
            with lock:
                logs.write(json.dumps(record) + '\n')
                logs.flush()
            return offset, scores, record

        records, scores = [], np.empty(total, dtype=np.float32)
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(one, i) for i in range(0, total, batch_size)]
            for future in as_completed(futures):
                offset, vals, record = future.result()
                scores[offset:offset+len(vals)] = vals
                records.append(record)
                observer.state['completed'] += len(vals)
                emit('batch_finished', **context, task=w['task'], completed=observer.state['completed'],
                     pairs=total, batch_s=record['batch_s'], warmup=warmup)
        score_wall = time.monotonic() - started
        if indices is not None:
            return scores
        predictions, before = {}, {}
        for p, val in zip(pairs, scores):
            if w['ignore_identical_ids'] and p['qid'] == p['did']:
                continue
            predictions.setdefault(p['qid'], {})[p['did']] = float(val)
            before.setdefault(p['qid'], {})[p['did']] = p['retrieval_score']
        evaluator = pytrec_eval.RelevanceEvaluator(w['qrels'], {'ndcg_cut.10', 'recall.10,100'})
        per_query = evaluator.evaluate(predictions)
        raw = evaluator.evaluate(before)
        metrics = {k: float(np.mean([p[k] for p in per_query.values()])) for k in next(iter(per_query.values()))}
        initial = {k: float(np.mean([p[k] for p in raw.values()])) for k in next(iter(raw.values()))}
        wall = time.monotonic() - started
        tokens = sum(r['tokens'] for r in records)
        latency = [r['http_s'] for r in records]
        row = {**context, 'task': w['task'], 'client_batch': batch_size, 'concurrency': concurrency,
               'pairs': total, 'tokens': tokens, 'mean_tokens': tokens / total,
               'score_wall_s': score_wall, 'pipeline_wall_s': wall, 'pairs_s': total/score_wall,
               'input_tok_s': tokens/score_wall, 'pipeline_tok_s': tokens/wall,
               'request_latency_s': {k: float(np.percentile(latency, p)) for k, p in [('p50', 50), ('p99', 99), ('max', 100)]},
               'metrics': metrics, 'embedding_metrics': initial}
        if w['task'] not in baseline:
            baseline[w['task']] = scores.copy()
        row['max_abs_score_diff_vs_baseline'] = float(np.max(np.abs(scores-baseline[w['task']])))
        key = f"{context['case']}_{context['client']}_{context['repeat']}_{w['task']}"
        save(args.output / (key + '_predictions.json'), predictions)
        results.append(row)
        save(args.output / 'results.json', results)
        emit('trial_finished', **row)
        return scores

    cases = [('s32_t16k', 32, 16384, False), ('s128_t32k', 128, 32768, False),
             ('s256_t32k', 256, 32768, False), ('s128_t32k_async', 128, 32768, True),
             ('s32_t16k_recheck', 32, 16384, False)]
    if args.smoke:
        cases = cases[:1]
    try:
        for case, seqs, budget, async_schedule in cases:
            observer.state = {'section': 'server_start', 'case': case}
            case_dir = args.output / case
            case_dir.mkdir()
            cmd = [BASE_PYTHON, '-m', 'vllm.entrypoints.openai.api_server', '--model', MODEL,
                   '--served-model-name', 'reranker-diagnostic', '--host', '127.0.0.1', '--port', str(args.port),
                   '--runner', 'pooling', '--hf-overrides', json.dumps({'architectures': ['Qwen3ForSequenceClassification'],
                       'classifier_from_token': ['no', 'yes'], 'is_original_qwen3_reranker': True}),
                   '--dtype', 'float16', '--max-model-len', str(MAX_LENGTH), '--enforce-eager',
                   '--gpu-memory-utilization', '0.45', '--max-num-seqs', str(seqs), '--block-size', '128',
                   '--max-num-batched-tokens', str(budget), '--no-enable-prefix-caching', '--no-enable-chunked-prefill',
                   '--async-scheduling' if async_schedule else '--no-async-scheduling']
            save(case_dir / 'command.json', cmd)
            with (case_dir / 'server.log').open('w') as server_log:
                server = subprocess.Popen(cmd, stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
                save(case_dir / 'pid.json', {'pid': server.pid})
                try:
                    start = time.monotonic()
                    while time.monotonic() - start < 600:
                        if server.poll() is not None:
                            raise RuntimeError(f'{case} server exited; see server.log')
                        try:
                            with urllib.request.urlopen(endpoint + '/health', timeout=2):
                                break
                        except OSError:
                            time.sleep(2)
                    else:
                        raise TimeoutError('Server startup timeout')
                    emit('server_ready', case=case, setup_s=time.monotonic()-start)
                    for w in workloads:
                        ref = refs[w['task']]
                        vals = run(w, 16, 1, {'case': case, 'client': 'hf_parity', 'repeat': 0},
                                   warmup=True, indices=ref['indices'])
                        delta = np.abs(vals-np.asarray(ref['scores']))
                        comparison = {'task': w['task'], 'case': case, 'hf_scores': ref['scores'],
                                      'vllm_scores': vals.tolist(), 'max_abs_diff': float(delta.max()),
                                      'mean_abs_diff': float(delta.mean()), 'diagnostic_tolerance': 0.03}
                        save(case_dir / (w['task']+'_hf_parity.json'), comparison)
                        emit('hf_parity', **{k:v for k,v in comparison.items() if not k.endswith('scores')})
                        if delta.max() > 0.03:
                            raise ValueError('Large HF/vLLM score discrepancy; inspect before benchmarking')
                    if args.smoke:
                        continue
                    clients = [(128, 1), (128, 2)]
                    if case == 's128_t32k':
                        clients += [(32, 1), (256, 4)]
                    for repeat in range(2):
                        # Reverse client order on the second pass.
                        for batch, concurrency in clients[::1 if repeat == 0 else -1]:
                            for w in workloads:
                                run(w, batch, concurrency, {'case': case, 'client': f'b{batch}_c{concurrency}', 'repeat': repeat})
                finally:
                    if server.poll() is None:
                        os.killpg(server.pid, signal.SIGTERM)
                    try:
                        server.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(server.pid, signal.SIGKILL)
                        server.wait(timeout=10)
    finally:
        logs.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['prepare', 'sweep'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prepared', type=Path)
    p.add_argument('--candidates', type=Path)
    p.add_argument('--queries', type=int, default=8)
    p.add_argument('--port', type=int, default=18225)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if args.output.exists():
        p.error('Output must be fresh')
    args.output.mkdir(parents=True)
    observer = Observer()
    observer.thread.start()
    try:
        (prepare if args.phase == 'prepare' else sweep)(args, observer)
        save(args.output / 'completion.json', {'status': 'complete'})
    except Exception as exc:
        save(args.output / 'completion.json', {'status': 'failed', 'error': repr(exc)})
        raise
    finally:
        observer.stop.set()
        observer.thread.join(timeout=1)


if __name__ == '__main__':
    main()
