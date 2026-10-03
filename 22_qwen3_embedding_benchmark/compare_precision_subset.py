"""FP32/BF16 diagnostic on the exact saved small retrieval subset.

FP16 embeddings are reused, not recomputed. The BF16 vLLM endpoint is separate
from the full benchmark. HF uses eager attention with HF32/downcasting disabled.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

from protocol import TASKS, format_text, validate_task
from run_evaluation import Observer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Choose a fresh output directory')
    args.output.mkdir(parents=True)
    observer = Observer()
    observer.thread.start()
    try:
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
        torch_npu.npu.set_option({'ACL_PRECISION_MODE': 'must_keep_origin_dtype'})
        torch_npu.npu.matmul.allow_hf32 = False
        torch_npu.npu.conv.allow_hf32 = False
        torch_npu.npu.matmul.cube_math_type = torch_npu.npu.CubeMathType.KEEP_DTYPE
        torch.set_num_threads(8)
        baseline = json.loads((args.baseline / 'manifest.json').read_text())
        qids, dids = baseline['qids'], baseline['dids']
        task = mteb.get_tasks(tasks=['EcomRetrieval'])[0]
        validate_task(task)
        task.load_data()
        model_path = '/workspace/models/Qwen3-Embedding-0.6B'
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, padding_side='left')
        texts = [format_text(task.queries['dev'][q], 'EcomRetrieval', 'query') for q in qids]
        texts += corpus_to_str([task.corpus['dev'][d] for d in dids])
        texts += [f"Instruct: {TASKS['EcomRetrieval'][2]}\nQuery: {task.queries['dev'][q]}" for q in qids]
        ids = tokenizer(texts, padding=False, truncation=True, max_length=8192)['input_ids']
        digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        assert digest == baseline['input_ids_sha256'], 'Input subset or tokenization changed'
        old = np.load(args.baseline / 'embeddings.npz')
        variants = {'vllm_fp16': old['vllm'], 'hf_fp16': old['hf']}
        report = {'diagnostic_only': True, 'queries': len(qids), 'documents': len(dids),
                  'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  'chip': '910B2', 'physical_device': os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
                  'baseline': str(args.baseline), 'input_ids_sha256': digest,
                  'hf_precision_options': {'ACL_PRECISION_MODE': 'must_keep_origin_dtype',
                     'allow_matmul_hf32': torch_npu.npu.matmul.allow_hf32,
                     'allow_conv_hf32': torch_npu.npu.conv.allow_hf32,
                     'cube_math_type': str(torch_npu.npu.matmul.cube_math_type)},
                  'hf_observed_dtypes': {}, 'timing_s': {}, 'results': {}}
        print(json.dumps({'event': 'precision_inputs_verified', **report['hf_precision_options']}), flush=True)
        start = time.monotonic()
        vectors = []
        for offset in range(0, len(ids), 128):
            observer.state = {'backend': 'vllm_bf16', 'done': offset, 'total': len(ids)}
            batch = ids[offset:offset + 128]
            req = urllib.request.Request('http://127.0.0.1:18223/v1/embeddings',
                data=json.dumps({'model': 'qwen3-embedding-bf16-diagnostic', 'input': batch,
                                 'encoding_format': 'float'}).encode(),
                headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=120) as response:
                rows = sorted(json.load(response)['data'], key=lambda r: r['index'])
            assert [r['index'] for r in rows] == list(range(len(batch)))
            vectors.extend(r['embedding'] for r in rows)
        variants['vllm_bf16'] = np.asarray(vectors, dtype=np.float32)
        report['timing_s']['vllm_bf16'] = time.monotonic() - start

        for name, dtype in [('hf_bf16', torch.bfloat16), ('hf_fp32', torch.float32)]:
            observer.state = {'backend': name, 'section': 'model_loading'}
            model = AutoModel.from_pretrained(model_path, local_files_only=True, torch_dtype=dtype,
                                             attn_implementation='eager').eval().to('npu:0')
            parameter_dtypes = sorted({str(x.dtype) for x in model.parameters()})
            assert parameter_dtypes == [str(dtype)], parameter_dtypes
            rows, fp32norm_rows = [], []
            start = time.monotonic()
            with torch.inference_mode():
                for offset in range(0, len(ids), 8):
                    observer.state = {'backend': name, 'done': offset, 'total': len(ids)}
                    batch = tokenizer(texts[offset:offset + 8], padding=True, truncation=True,
                                      max_length=8192, return_tensors='pt')
                    assert all(t == batch['input_ids'][j][batch['attention_mask'][j].bool()].tolist()
                               for j, t in enumerate(ids[offset:offset + 8]))
                    hidden = model(**batch.to('npu:0'), use_cache=False).last_hidden_state[:, -1]
                    assert hidden.dtype == dtype
                    rows.append(torch.nn.functional.normalize(hidden, p=2, dim=1).float().cpu().numpy())
                    if name == 'hf_bf16':
                        fp32norm_rows.append(torch.nn.functional.normalize(hidden.float(), p=2, dim=1).cpu().numpy())
            variants[name] = np.concatenate(rows)
            if fp32norm_rows:
                variants['hf_bf16_fp32norm'] = np.concatenate(fp32norm_rows)
            report['hf_observed_dtypes'][name] = {'parameters': parameter_dtypes, 'last_hidden_state': str(hidden.dtype)}
            report['timing_s'][name] = time.monotonic() - start
            del model, hidden, batch
            gc.collect()
            torch_npu.npu.empty_cache()

        observer.state = {'section': 'retrieval_scoring'}
        qrels = {q: task.relevant_docs['dev'][q] for q in qids}
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'recall.10,100', 'ndcg_cut.10'})
        per_query = {}
        for name, embeddings in variants.items():
            assert embeddings.shape == (len(ids), 1024) and np.isfinite(embeddings).all()
            n = len(qids)
            qmap = dict(zip(qids, embeddings[:n]))
            dmap = dict(zip(dids, embeddings[n:n + len(dids)]))

            class RecordedEncoder:
                def encode(self, sentences, *, prompt_type, **kwargs):
                    role = getattr(prompt_type, 'value', prompt_type)
                    return np.stack([qmap[q] for q in sentences] if role == 'query'
                                    else [dmap[d['text']] for d in sentences])

            run = DenseRetrievalExactSearch(RecordedEncoder(), corpus_chunk_size=4096).search(
                {d: {'text': d} for d in dids}, {q: q for q in qids}, 1000, 'EcomRetrieval')
            if task.ignore_identical_ids:
                for q in qids:
                    run[q].pop(q, None)
            scores = evaluator.evaluate(run)
            per_query[name] = scores
            report['results'][name] = {key: sum(row[key] for row in scores.values()) / len(scores)
                for key in ('recall_10', 'recall_100', 'ndcg_cut_10')}
        report['changed_queries_vs_vllm_fp16'] = {
            name: [{'qid': q, 'baseline': per_query['vllm_fp16'][q], 'variant': rows[q]}
                   for q in qids if rows[q] != per_query['vllm_fp16'][q]]
            for name, rows in per_query.items() if name != 'vllm_fp16'}
        np.savez_compressed(args.output / 'embeddings.npz', **variants)
        (args.output / 'per_query.json').write_text(json.dumps(per_query, indent=2) + '\n')
        (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)
    except BaseException as exc:
        (args.output / 'failure.json').write_text(json.dumps({'error': repr(exc), 'state': observer.state}, indent=2))
        raise
    finally:
        observer.stop.set()


if __name__ == '__main__':
    main()
