"""Same real Ecom inputs: HF eager FP16 on NPU vs existing vLLM endpoint.

Diagnostic only: this is not the original CUDA/FlashAttention2 implementation,
nor a full retrieval evaluation. Does not change the benchmark or its service.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

from protocol import format_text, validate_task


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', default='/workspace/models/Qwen3-Embedding-0.6B')
    args = p.parse_args()
    if args.output.exists():
        p.error('Choose a fresh output path')
    started = time.monotonic()
    import numpy as np
    import torch
    import torch_npu
    import mteb
    from transformers import AutoModel, AutoTokenizer
    from mteb.evaluation.evaluators.RetrievalEvaluator import corpus_to_str

    assert torch_npu.npu.is_available(), 'NPU required; no CPU inference fallback'
    torch_npu.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.set_num_threads(8)
    task = mteb.get_tasks(tasks=['EcomRetrieval'])[0]
    validate_task(task)
    task.load_data()
    predictions = json.loads(args.predictions.read_text())
    qids = list(task.queries['dev'])[:32]
    dids = sorted({did for q in qids for did in
                   list(sorted(predictions[q], key=predictions[q].get, reverse=True)[:5])
                   + list(task.relevant_docs['dev'][q])})
    texts = [format_text(task.queries['dev'][q], 'EcomRetrieval', 'query') for q in qids]
    texts += corpus_to_str([task.corpus['dev'][d] for d in dids])
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = 'left'
    ids = tokenizer(texts, padding=False, truncation=True, max_length=8192)['input_ids']
    official_style = tokenizer(texts, padding=True, truncation=True, max_length=8192, return_tensors='pt')
    assert all(t == official_style['input_ids'][i][official_style['attention_mask'][i].bool()].tolist()
               for i, t in enumerate(ids)), 'Non-padding token IDs differ'
    print(json.dumps({'event': 'inputs_ready', 'queries': len(qids), 'documents': len(dids),
                      'tokens': sum(map(len, ids)), 'max_length': max(map(len, ids))}), flush=True)

    def endpoint(batch_size):
        vectors = []
        for start in range(0, len(ids), batch_size):
            batch = ids[start:start + batch_size]
            data = json.dumps({'model': 'qwen3-embedding-0.6b', 'input': batch,
                               'encoding_format': 'float'}).encode()
            request = urllib.request.Request('http://127.0.0.1:18222/v1/embeddings',
                                             data=data, headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=120) as response:
                rows = sorted(json.load(response)['data'], key=lambda row: row['index'])
            assert [r['index'] for r in rows] == list(range(len(batch)))
            vectors.extend(r['embedding'] for r in rows)
        return np.asarray(vectors, dtype=np.float32)

    vllm32 = endpoint(32)
    vllm8 = endpoint(8)
    print(json.dumps({'event': 'endpoint_comparisons_done'}), flush=True)
    model = AutoModel.from_pretrained(args.model, local_files_only=True,
                                     torch_dtype=torch.float16, attn_implementation='eager').eval().to('npu:0')

    def hf(batch_size):
        vectors = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = tokenizer(texts[start:start + batch_size], padding=True,
                                  truncation=True, max_length=8192, return_tensors='pt').to('npu:0')
                hidden = model(**batch, use_cache=False).last_hidden_state[:, -1]
                # Mirror Qwen's FP16 normalization before cast to FP32 on output.
                vectors.append(torch.nn.functional.normalize(hidden, p=2, dim=1).float().cpu().numpy())
        return np.concatenate(vectors)

    hf8 = hf(8)
    hf1 = hf(1)

    def compare(a, b):
        unit_a = a / np.linalg.norm(a, axis=1, keepdims=True)
        unit_b = b / np.linalg.norm(b, axis=1, keepdims=True)
        cos = np.sum(unit_a * unit_b, axis=1)
        sa = unit_a[:len(qids)] @ unit_a[len(qids):].T
        sb = unit_b[:len(qids)] @ unit_b[len(qids):].T
        order_a, order_b = np.argsort(-sa, axis=1), np.argsort(-sb, axis=1)
        gold_rank = []
        for i, qid in enumerate(qids):
            gold = set(task.relevant_docs['dev'][qid])
            ra = min(j + 1 for j, idx in enumerate(order_a[i]) if dids[idx] in gold)
            rb = min(j + 1 for j, idx in enumerate(order_b[i]) if dids[idx] in gold)
            if ra != rb:
                gold_rank.append({'qid': qid, 'rank_a': ra, 'rank_b': rb})
        return {'embedding_cosine_min': float(cos.min()), 'embedding_cosine_mean': float(cos.mean()),
                'embedding_max_abs_diff': float(np.abs(a-b).max()),
                'pair_score_abs_diff_max': float(np.abs(sa-sb).max()),
                'pair_score_abs_diff_mean': float(np.abs(sa-sb).mean()),
                'candidate_pool_top1_changed': int((order_a[:, 0] != order_b[:, 0]).sum()),
                'candidate_pool_gold_rank_changes': gold_rank}

    report = {'diagnostic_only': True, 'chip': '910B2',
              'physical_device': os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
              'reference': 'Transformers 4.51.3 eager FP16 on Ascend, not original CUDA FA2',
              'sample': 'first 32 dev queries; union of their stored top5 and gold documents',
              'queries': len(qids), 'documents': len(dids),
              'nonpadding_token_ids_equal': True,
              'input_ids_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
              'vllm32_vs_vllm8': compare(vllm32, vllm8),
              'hf8_vs_hf1': compare(hf8, hf1), 'vllm8_vs_hf8': compare(vllm8, hf8),
              'elapsed_s': time.monotonic() - started}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(args.output.with_suffix('.npz'), vllm32=vllm32, vllm8=vllm8,
                        hf8=hf8, hf1=hf1, qids=qids, dids=dids)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
