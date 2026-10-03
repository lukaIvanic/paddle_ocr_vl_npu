"""Small paired retrieval diagnostic, NOT a published-benchmark reproduction.

Shared pool: union of fixed 64 queries' existing vLLM top100, their gold docs,
and 2048 seeded random corpus documents. Selection is vLLM-conditioned and can
miss HF-only candidates outside this pool. Never compare its absolute scores
to full-corpus scores. Both backends independently encode queries AND documents.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time
import urllib.request

from protocol import TASKS, format_text, validate_task


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Choose a fresh output directory')
    args.output.mkdir(parents=True)
    import numpy as np
    import torch
    import torch_npu
    import mteb
    import pytrec_eval
    from transformers import AutoModel, AutoTokenizer
    from mteb.evaluation.evaluators.RetrievalEvaluator import corpus_to_str, DenseRetrievalExactSearch

    assert torch_npu.npu.is_available(), 'No CPU inference fallback'
    torch_npu.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.set_num_threads(8)
    model_path = '/workspace/models/Qwen3-Embedding-0.6B'
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, padding_side='left')
    task = mteb.get_tasks(tasks=['EcomRetrieval'])[0]
    validate_task(task)
    task.load_data()
    predictions = json.loads(args.predictions.read_text())
    rng = random.Random(20261003)
    qids = sorted(rng.sample(sorted(task.queries['dev']), 64))
    dids = sorted(set(rng.sample(sorted(task.corpus['dev']), 2048)) |
                  {d for q in qids for d in predictions[q]} |
                  {d for q in qids for d in task.relevant_docs['dev'][q]})
    queries = [format_text(task.queries['dev'][q], 'EcomRetrieval', 'query') for q in qids]
    documents = corpus_to_str([task.corpus['dev'][d] for d in dids])
    texts = queries + documents
    # Deliberate prompt perturbation, diagnostic only: never silently adopt it.
    spaced = [f"Instruct: {TASKS['EcomRetrieval'][2]}\nQuery: {task.queries['dev'][q]}" for q in qids]
    input_texts = texts + spaced
    ids = tokenizer(input_texts, padding=False, truncation=True, max_length=8192)['input_ids']
    qrels = {q: task.relevant_docs['dev'][q] for q in qids}
    manifest = {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'chip': '910B2', 'hf_physical_device': os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
                'queries': len(qids), 'documents': len(dids), 'qids': qids, 'dids': dids,
                'seed': 20261003, 'selection': __doc__, 'ignore_identical_ids': task.ignore_identical_ids,
                'input_ids_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                'predictions_sha256': hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
                'reference': 'Transformers 4.51.3 eager FP16 on Ascend, not CUDA FA2'}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'event': 'subset_ready', 'queries': len(qids), 'documents': len(dids)}), flush=True)

    def endpoint():
        vectors = []
        for offset in range(0, len(ids), 128):
            batch = ids[offset:offset + 128]
            req = urllib.request.Request('http://127.0.0.1:18222/v1/embeddings',
                data=json.dumps({'model': 'qwen3-embedding-0.6b', 'input': batch, 'encoding_format': 'float'}).encode(),
                headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=120) as response:
                records = sorted(json.load(response)['data'], key=lambda r: r['index'])
            assert [r['index'] for r in records] == list(range(len(batch)))
            vectors.extend(r['embedding'] for r in records)
            print(json.dumps({'event': 'encoded', 'backend': 'vllm', 'done': len(vectors), 'total': len(ids)}), flush=True)
        return np.asarray(vectors, dtype=np.float32)

    start = time.monotonic()
    vllm = endpoint()
    vllm_s = time.monotonic() - start
    model = AutoModel.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.float16,
                                     attn_implementation='eager').eval().to('npu:0')
    hf_rows = []
    start = time.monotonic()
    with torch.inference_mode():
        for offset in range(0, len(ids), 8):
            batch = tokenizer(input_texts[offset:offset + 8], padding=True, truncation=True,
                              max_length=8192, return_tensors='pt')
            assert all(t == batch['input_ids'][j][batch['attention_mask'][j].bool()].tolist()
                       for j, t in enumerate(ids[offset:offset + 8]))
            batch = batch.to('npu:0')
            hidden = model(**batch, use_cache=False).last_hidden_state[:, -1]
            hf_rows.append(torch.nn.functional.normalize(hidden, p=2, dim=1).float().cpu().numpy())
            if offset % 256 == 0:
                print(json.dumps({'event': 'encoded', 'backend': 'hf', 'done': offset + len(batch['input_ids']),
                                  'total': len(ids)}), flush=True)
    hf_s = time.monotonic() - start
    hf = np.concatenate(hf_rows)
    assert np.isfinite(hf).all() and np.isfinite(vllm).all()
    np.savez_compressed(args.output / 'embeddings.npz', hf=hf, vllm=vllm)
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'recall.10,100', 'ndcg_cut.10'})
    n = len(qids)

    def evaluate(emb, space, depth, chunk_size=4096):
        query_vectors = emb[len(texts):] if space else emb[:n]
        qmap = dict(zip(qids, query_vectors))
        dmap = dict(zip(dids, emb[n:len(texts)]))

        class RecordedEncoder:
            def encode(self, sentences, *, prompt_type, **kwargs):
                role = getattr(prompt_type, 'value', prompt_type)
                if role == 'query':
                    return np.stack([qmap[q] for q in sentences])
                return np.stack([dmap[d['text']] for d in sentences])

        # Exercise the installed evaluator's real chunked top-k/heap selection.
        # No model subsection is replayed: only saved embeddings are rescored.
        search = DenseRetrievalExactSearch(RecordedEncoder(), corpus_chunk_size=chunk_size)
        run = search.search({d: {'text': d} for d in dids}, {q: q for q in qids}, depth, 'EcomRetrieval')
        if task.ignore_identical_ids:
            for q in qids:
                run[q].pop(q, None)
        per_query = evaluator.evaluate(run)
        return {key: sum(row[key] for row in per_query.values()) / len(per_query)
                for key in ('recall_10', 'recall_100', 'ndcg_cut_10')}, per_query

    report = {'diagnostic_only': True, 'queries': n, 'documents': len(dids), 'timing_s': {'vllm': vllm_s, 'hf': hf_s},
              'nonpadding_token_ids_equal': True, 'results': {}}
    all_details = {}
    for backend, emb in [('vllm', vllm), ('hf', hf)]:
        for space in (False, True):
            for depth in (100, 1000):
                key = f'{backend}_space{int(space)}_top{depth}'
                report['results'][key], all_details[key] = evaluate(emb, space, depth)
        for depth in (100, 1000):
            key = f'{backend}_space0_top{depth}_chunk50000'
            report['results'][key], all_details[key] = evaluate(emb, False, depth, 50000)
    a, b = all_details['vllm_space0_top1000'], all_details['hf_space0_top1000']
    report['changed_queries'] = [{ 'qid': q, 'vllm': a[q], 'hf': b[q]} for q in qids if a[q] != b[q]]
    (args.output / 'per_query.json').write_text(json.dumps(all_details, indent=2) + '\n')
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
