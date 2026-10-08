"""Small adaptive FP16/eager serving sweep, followed by replica-scaling checks.

Reuses saved real candidate texts, never creates or changes candidate pools.
Model-card query-first inputs and the historical 8192-token policy are preserved.
Timings include tokenization, HTTP, and score parsing; setup is recorded separately.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.request

from protocol import TASKS
from reranker_protocol import PREFIX, SUFFIX, MAX_LENGTH
from reranker_serving_sweep import request, scores_from_response, digest
from run_evaluation import emit
from suite_protocol import ENGLISH


def save(path, data):
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def tokenize(tok, task, pairs):
    instruction = (ENGLISH if task in ENGLISH else TASKS)[task][2]
    prefix, suffix = (tok.encode(x, add_special_tokens=False) for x in (PREFIX, SUFFIX))
    limit = MAX_LENGTH - len(prefix) - len(suffix)
    texts = [f'<Instruct>: {instruction}\n<Query>: {p["query"]}\n<Document>: {p["document"]}' for p in pairs]
    ids = tok(texts, add_special_tokens=False, padding=False, truncation=False)['input_ids']
    return [prefix + row[:limit] + suffix for row in ids], sum(len(row) > limit for row in ids)


def assert_free(devices):
    snapshot = subprocess.check_output(['npu-smi', 'info'], text=True)
    busy = {int(m[0]): int(m[1]) for m in re.findall(
        r'^\|[ \t]+(\d+)[ \t]+\d+[ \t]*\|[ \t]*(\d+)[ \t]*\|', snapshot, flags=re.M)}
    # This coordinator's released HF reference may retain its runtime context.
    conflicts = {d: busy[d] for d in devices if d in busy and busy[d] != os.getpid()}
    if conflicts:
        raise RuntimeError(f'Devices acquired by other processes: {conflicts}; will not touch them')
    return snapshot


def prepare(args, tok):
    source = args.prepared / 'workloads.json'
    manifest = json.loads((args.prepared / 'manifest.json').read_text())
    assert digest(source) == manifest['workloads_sha256']
    workloads = json.loads(source.read_text())
    for w in workloads:
        ids, trunc = tokenize(tok, w['task'], w['pairs'])
        assert hashlib.sha256(json.dumps(ids).encode()).hexdigest() == w['input_ids_sha256'], w['task']
        assert trunc == w['truncated']
    if args.english_evaluation:
        from mteb.evaluation.evaluators.RetrievalEvaluator import corpus_to_str
        from run_english_suite import load_task
        import importlib
        import random
        class Observer:
            state = {}
        for name in ['ArguAna', 'ClimateFEVERHardNegatives']:
            # Offline datasets cannot infer the default qrels config when
            # corpus/queries/default are all cached; select that same config explicitly.
            loader_module = importlib.import_module('mteb.abstasks.AbsTaskRetrieval')
            original_loader = loader_module.load_dataset
            def explicit_default(repo, *a, **kw):
                if not a and 'name' not in kw:
                    kw['name'] = 'default'
                return original_loader(repo, *a, **kw)
            loader_module.load_dataset = explicit_default
            try:
                task, meta = load_task(name, Observer())
            finally:
                loader_module.load_dataset = original_loader
            path = args.english_evaluation / 'embedding' / name / 'mteb' / f'{name}_default_predictions.json'
            candidates = json.loads(path.read_text())
            qids = sorted(random.Random(20261008).sample(sorted(candidates), 8))
            pairs = []
            for qid in qids:
                dids = sorted(candidates[qid], key=lambda d: (-candidates[qid][d], d))[:100]
                assert len(dids) == 100
                docs = corpus_to_str([task.corpus['test'][d] for d in dids])
                pairs += [dict(qid=qid, did=d, query=task.queries['test'][qid], document=doc,
                               retrieval_score=candidates[qid][d]) for d, doc in zip(dids, docs)]
            ids, trunc = tokenize(tok, name, pairs)
            workloads.append(dict(task=name, pairs=pairs, qids=qids,
                qrels={q: task.relevant_docs['test'][q] for q in qids},
                ignore_identical_ids=task.ignore_identical_ids,
                lengths=list(map(len, ids)), truncated=trunc,
                candidate_source=str(path), candidate_sha256=digest(path), pinned_task=meta))
    save(args.output / 'workloads.json', workloads)
    for w in workloads:
        emit('workload', task=w['task'], pairs=len(w['pairs']), tokens=sum(w['lengths']),
             max_length=max(w['lengths']), truncated=w['truncated'])
    return workloads


def reference(args, tok, workloads):
    import torch
    import torch_npu
    from transformers import AutoModelForCausalLM
    assert_free([args.devices[0]])
    torch.set_num_threads(8)
    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
        torch_dtype=torch.float16, attn_implementation='eager').eval().to('npu:0')
    no, yes = tok.convert_tokens_to_ids('no'), tok.convert_tokens_to_ids('yes')
    assert {str(p.dtype) for p in model.parameters()} == {'torch.float16'}
    refs = {}
    with torch.inference_mode():
        for w in workloads:
            indices = [q*100+r for q in range(4) for r in (0, 1, 10, 99)]
            ids, _ = tokenize(tok, w['task'], [w['pairs'][i] for i in indices])
            values = []
            for row in ids:
                x = torch.tensor([row], device='npu:0')
                logits = model(input_ids=x, attention_mask=torch.ones_like(x),
                               use_cache=False, logits_to_keep=1).logits[:, -1, :]
                values.append(float(torch.nn.functional.log_softmax(logits[:, [no, yes]], dim=-1)[:, 1].exp()[0].cpu()))
            refs[w['task']] = dict(indices=indices, scores=values)
            emit('hf_reference', task=w['task'], pairs=len(values))
    save(args.output / 'hf_reference.json', refs)
    del model, x, logits
    gc.collect()
    torch.npu.empty_cache()
    return refs


@contextmanager
def servers(args, devices, config, name):
    assert_free(devices)
    owned, handles, endpoints = [], [], []
    folder = args.output / name
    folder.mkdir()
    try:
        for i, device in enumerate(devices):
            port = args.port + i
            with socket.socket() as probe:
                if probe.connect_ex(('127.0.0.1', port)) == 0:
                    raise RuntimeError(f'Port {port} occupied; will not touch its owner')
            cmd = [args.server_python, '-m', 'vllm.entrypoints.openai.api_server',
                '--model', args.model, '--served-model-name', 'reranker-diagnostic',
                '--host', '127.0.0.1', '--port', str(port), '--runner', 'pooling',
                '--hf-overrides', json.dumps(dict(architectures=['Qwen3ForSequenceClassification'],
                    classifier_from_token=['no', 'yes'], is_original_qwen3_reranker=True)),
                '--dtype', 'float16', '--max-model-len', str(MAX_LENGTH), '--enforce-eager',
                '--gpu-memory-utilization', '0.45', '--max-num-seqs', str(config[0]),
                '--max-num-batched-tokens', str(config[1]), '--block-size', '128',
                '--no-enable-prefix-caching', '--no-enable-chunked-prefill', '--no-async-scheduling']
            save(folder / f'command_{device}.json', cmd)
            log = (folder / f'server_{device}.log').open('w')
            handles.append(log)
            p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, 'ASCEND_RT_VISIBLE_DEVICES': str(device)}, start_new_session=True)
            owned.append(p)
            save(folder / f'pid_{device}.json', dict(pid=p.pid, device=device))
            endpoints.append(f'http://127.0.0.1:{port}')
        started = time.monotonic()
        pending = set(range(len(owned)))
        while pending and time.monotonic()-started < 600:
            for i in list(pending):
                if owned[i].poll() is not None:
                    raise RuntimeError(f'{name}: server exited; see saved log')
                try:
                    with urllib.request.urlopen(endpoints[i]+'/health', timeout=2):
                        pending.remove(i)
                except OSError:
                    pass
            if pending:
                time.sleep(2)
        if pending:
            raise TimeoutError('Server startup timeout')
        emit('servers_ready', name=name, devices=devices, setup_s=time.monotonic()-started)
        yield endpoints
    finally:
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


def trial(args, tok, workloads, endpoints, batch, concurrency, tag, repeat, refs=None):
    import numpy as np
    import pytrec_eval
    result = []
    for w in workloads:
        pairs = w['pairs'] if refs is None else [w['pairs'][i] for i in refs[w['task']]['indices']]
        started = time.monotonic()
        tokenize_s, records = 0., []
        # Tokenization stays in the timed pipeline; HTTP requests overlap it.
        scores = np.empty(len(pairs), dtype=np.float32)
        pools = [ThreadPoolExecutor(max_workers=concurrency) for _ in endpoints]
        futures = []
        try:
            for j, offset in enumerate(range(0, len(pairs), batch)):
                t0 = time.monotonic()
                ids, truncated = tokenize(tok, w['task'], pairs[offset:offset+batch])
                tokenize_s += time.monotonic()-t0
                def one(endpoint, ids=ids, offset=offset, truncated=truncated):
                    t0 = time.monotonic()
                    response = request(endpoint+'/pooling', dict(model='reranker-diagnostic',
                        input=ids, task='classify', use_activation=True,
                        encoding_format='float', add_special_tokens=False))
                    values = scores_from_response(response, len(ids))
                    tokens = sum(map(len, ids))
                    assert response['usage']['prompt_tokens'] == tokens
                    return offset, values, dict(tokens=tokens, pairs=len(ids), truncated=truncated,
                                               http_s=time.monotonic()-t0)
                i = j % len(endpoints)
                futures.append(pools[i].submit(one, endpoints[i]))
            for future in as_completed(futures):
                offset, values, record = future.result()
                scores[offset:offset+len(values)] = values
                records.append(record)
        finally:
            for pool in pools:
                pool.shutdown(wait=True)
        elapsed = time.monotonic()-started
        row = dict(tag=tag, repeat=repeat, task=w['task'], replicas=len(endpoints),
            client_batch=batch, concurrency_per_replica=concurrency, pairs=len(pairs),
            tokens=sum(r['tokens'] for r in records), score_wall_s=elapsed,
            tokenize_s=tokenize_s, pairs_s=len(pairs)/elapsed,
            http_p50_s=float(np.median([r['http_s'] for r in records])))
        row['input_tok_s'] = row['tokens']/elapsed
        if refs is not None:
            delta = np.abs(scores-np.asarray(refs[w['task']]['scores']))
            row['max_abs_hf_score_diff'] = float(delta.max())
            if not np.isfinite(scores).all() or delta.max() > .03:
                raise ValueError(f'HF/vLLM parity failed: {row}')
        else:
            predictions = {}
            for pair, value in zip(pairs, scores):
                if w['ignore_identical_ids'] and pair['qid'] == pair['did']:
                    continue
                predictions.setdefault(pair['qid'], {})[pair['did']] = float(value)
            metrics = pytrec_eval.RelevanceEvaluator(w['qrels'], {'ndcg_cut.10'}).evaluate(predictions)
            row['ndcg10'] = sum(x['ndcg_cut_10'] for x in metrics.values())/len(metrics)
            save(args.output / f'{tag}_r{repeat}_{w["task"]}_scores.json', scores.tolist())
        emit('parity' if refs else 'trial', **row)
        result.append(row)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='/workspace/models/Qwen3-Reranker-0.6B')
    p.add_argument('--server-python', default='/usr/local/python3.12.13/bin/python3')
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--english-evaluation', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--devices', type=int, nargs='+', required=True)
    p.add_argument('--port', type=int, default=18360)
    p.add_argument('--reference-only', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.reference_only:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True, padding_side='left')
        reference(args, tok, json.loads((args.output/'workloads.json').read_text()))
        return
    if args.output.exists() or len(set(args.devices)) != len(args.devices):
        p.error('Require a fresh output and distinct devices')
    args.output.mkdir(parents=True)
    os.environ['ASCEND_RT_VISIBLE_DEVICES'] = str(args.devices[0])
    results = []
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True, padding_side='left')
        config = dict(model=args.model, devices=args.devices, max_length=MAX_LENGTH,
            dtype='float16', eager=True, prefix_cache=False, input_order='query_first',
            commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
            source_prepared=str(args.prepared), model_files={x.name: digest(x) for x in
                Path(args.model).glob('*.json')},
            scope='real workload throughput sample; not full benchmark accuracy')
        save(args.output/'manifest.json', config)
        save(args.output/'initial_npu_snapshot.json', assert_free(args.devices))
        workloads = prepare(args, tok)
        # End the HF worker completely before allocating serving replicas.
        # empty_cache() alone does not release the NPU runtime context.
        subprocess.run([sys.executable, __file__, '--reference-only', '--model', args.model,
            '--prepared', str(args.prepared), '--output', str(args.output),
            '--devices', str(args.devices[0])], check=True)
        refs = json.loads((args.output/'hf_reference.json').read_text())
        def measure(endpoints, client, tag):
            for endpoint in endpoints:
                trial(args, tok, workloads, [endpoint], 16, 1, tag+'_parity', 0, refs)
            for repeat in range(2):
                rows = trial(args,tok,workloads,endpoints,*client,tag,repeat)
                results.extend(rows)
                save(args.output/'results.json',results)
        clients = [(64,1),(128,1),(128,2),(256,2),(256,4)]
        # First isolate client load, then change server scheduler capacity.
        with servers(args,args.devices[:1],(32,16384),'s32_t16k') as endpoints:
            for i, client in enumerate(clients):
                measure(endpoints,client,f's32_b{client[0]}_c{client[1]}')
        def cost(tag):
            return statistics.mean(sum(r['score_wall_s'] for r in results
                if r['tag']==tag and r['repeat']==rep) for rep in range(2))
        tags = sorted({r['tag'] for r in results})
        first = min(tags,key=cost)
        example = next(r for r in results if r['tag']==first)
        client = (example['client_batch'],example['concurrency_per_replica'])
        choices = [(first,(32,16384))]
        for seqs in (128,256):
            tag=f's{seqs}_b{client[0]}_c{client[1]}'
            with servers(args,args.devices[:1],(seqs,32768),f's{seqs}_t32k') as endpoints:
                measure(endpoints,client,tag)
            choices.append((tag,(seqs,32768)))
        best, server_config = min(choices,key=lambda x:cost(x[0]))
        emit('selected', tag=best, client=client, server_config=server_config,
             selection='minimum mean summed scoring time across fixed sampled tasks')
        # Recheck the winner and use exactly the same workload for replica scaling.
        with servers(args,args.devices[:1],server_config,'winner_recheck') as endpoints:
            measure(endpoints,client,'winner_recheck')
        scaling_status = 'not_requested'
        if len(args.devices)>1:
            try:
                assert_free(args.devices)
            except RuntimeError as exc:
                scaling_status = str(exc)
                emit('scaling_unavailable', reason=scaling_status)
            else:
                with servers(args,args.devices,server_config,'replica_scaling') as endpoints:
                    measure(endpoints,client,'replica_scaling')
                scaling_status = 'complete'
        save(args.output/'summary.json', dict(status='complete',best_sweep_tag=best,
            selected_server_config=server_config,selected_client=client,
            measured_replicas=max(r['replicas'] for r in results), scaling_status=scaling_status, results=results))
        save(args.output/'completion.json',dict(status='complete'))
    except Exception as exc:
        save(args.output/'completion.json',dict(status='failed',error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
