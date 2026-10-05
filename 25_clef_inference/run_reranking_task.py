"""Score a frozen complete reranking task from existing BF16 document caches."""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from run_local_smoke import NoTransformers, encode_document_prefix, encode_record
from run_reranking_smoke import BENCHMARK_DIR, cache_cast, digest, request_for, save, sync_saved


@contextmanager
def ordinary_preload_pages():
    """Avoid huge-page allocation stalls for owned CPU caches; restore on exit."""
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 4
    prctl.restype = ctypes.c_int
    previous = prctl(42, 0, 0, 0, 0)  # PR_GET_THP_DISABLE, supported by server kernel.
    if previous < 0 or prctl(41, 1, 0, 0, 0) != 0:  # PR_SET_THP_DISABLE
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    try:
        yield
    finally:
        if prctl(41, previous, 0, 0, 0) != 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code))


def validate_predictions(fixture, predictions, complete):
    if set(predictions) - set(fixture['queries']):
        raise ValueError('Unknown query in predictions')
    for qid, values in predictions.items():
        expected = set(fixture['queries'][qid]['candidates'])
        if set(values) != expected or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values.values()):
            raise ValueError('Incomplete query or invalid score: ' + qid)
    if complete and set(predictions) != set(fixture['queries']):
        raise ValueError('Missing query predictions')


def evaluate(args):
    if importlib.metadata.version('mteb') != '1.38.9':
        raise ValueError('Use the original MTEB 1.38.9 environment')
    sys.path.insert(0, str(BENCHMARK_DIR))
    from run_reranker_evaluation import metric_summary
    fixture = json.loads(args.fixture.read_text())
    result = json.loads(args.output.read_text())
    if result['status'] != 'completed' or result['contract']['fixture_sha256'] != digest(args.fixture):
        raise ValueError('Incomplete run or changed fixture')
    predictions = result['predictions']
    validate_predictions(fixture, predictions, complete=True)
    candidates = {q: r['candidates'] for q, r in fixture['queries'].items()}
    qwen = {q: r['qwen_scores'] for q, r in fixture['queries'].items()}
    qrels = {q: r['qrels'] for q, r in fixture['queries'].items()}
    ignore = fixture['ignore_identical_ids']
    metrics, per_query = metric_summary(predictions, candidates, qrels, ignore)
    reference, reference_per_query = metric_summary(qwen, candidates, qrels, ignore)
    result['evaluation'] = {
        'mteb_version': importlib.metadata.version('mteb'),
        'metrics': {'clef': metrics['reranker'], 'qwen3_reranker_4b': reference['reranker'],
                    'embedding': metrics['embedding']},
        'ndcg10_difference_pp': 100 * (metrics['reranker']['ndcg_cut_10'] - reference['reranker']['ndcg_cut_10']),
        'per_query': {'clef': per_query['reranker'], 'qwen3_reranker_4b': reference_per_query['reranker'],
                      'embedding': per_query['embedding']},
        'scope': 'complete frozen task; same top100 candidates and full qrels as saved Qwen baseline'}
    save(args.output, result)
    sync_saved(args.output)
    print(json.dumps(result['evaluation']['metrics'], indent=2), flush=True)


def score(args):
    fixture = json.loads(args.fixture.read_text())
    manifest = json.loads(args.cache_manifest.read_text())
    if fixture['scope'] != 'complete_task_candidate_documents' or fixture['truncation'] != 'none':
        raise ValueError('Expected complete, untruncated task fixture')
    if any(len(q['candidates']) != 100 or set(q['candidates']) != set(q['qwen_scores'])
           for q in fixture['queries'].values()):
        raise ValueError('Expected the same top100 candidates as Qwen')
    # For self-match protocols, include a document if it also occurs for another query.
    documents = {d for qid, q in fixture['queries'].items() for d in q['candidates']
                 if not (fixture['ignore_identical_ids'] and qid == d)}
    if set(fixture['documents']) != documents or set(manifest['documents']) != documents:
        raise ValueError('Cache/document coverage mismatch')
    if (manifest['status'] != 'completed' or manifest['task'] != fixture['task'] or
        manifest['contract']['fixture_sha256'] != digest(args.fixture) or
        manifest['contract']['dtype'] != 'float32' or manifest['contract']['storage_dtype'] != 'bfloat16'):
        raise ValueError('Incompatible or incomplete cache manifest')
    if digest(args.model / 'tokenizer.json') != fixture['tokenizer_sha256']:
        raise ValueError('Tokenizer changed')
    contract = {'fixture_sha256': digest(args.fixture), 'cache_manifest_sha256': digest(args.cache_manifest),
                'model_source_sha256': digest(Path(__file__).with_name('local_modeling_clef.py')),
                'compute_dtype': 'float32', 'storage_dtype': 'bfloat16',
                'ranking_score': 'unrounded P(true)', 'request_template': 'existing request_for'}
    result = json.loads(args.output.read_text()) if args.output.exists() else {
        'status': 'running', 'task': fixture['task'], 'contract': contract, 'predictions': {}, 'queries': {}}
    if result['contract'] != contract:
        raise ValueError('Resume contract changed')
    validate_predictions(fixture, result['predictions'], complete=False)
    if result['status'] == 'completed':
        validate_predictions(fixture, result['predictions'], complete=True)
        print('Already scored; use evaluate for metrics', flush=True)
        return
    sys.meta_path.insert(0, NoTransformers())
    import torch
    import torch_npu
    from torch_npu.npu.npu_config import _CubeMathType
    from tokenizers import Tokenizer
    from local_modeling_clef import DocumentCache, load_model
    torch.npu.set_option({'ACL_PRECISION_MODE': 'must_keep_origin_dtype'})
    torch.npu.matmul.allow_hf32 = False
    torch.npu.conv.allow_hf32 = False
    torch.npu.matmul.cube_math_type = _CubeMathType.KEEP_DTYPE
    torch.set_float32_matmul_precision('highest')
    if not torch.npu.is_available() or '910B' not in torch.npu.get_device_name(0):
        raise RuntimeError('910B required; no CPU model fallback')
    torch.npu.set_device(0)
    torch.set_num_threads(8)
    if torch.npu.mem_get_info()[0] < 50 * 1024**3:
        raise RuntimeError('Insufficient free NPU memory for FP32 model')
    result.update(status='running', git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  hostname=platform.node(), physical_device=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
                  torch=torch.__version__, torch_npu=torch_npu.__version__,
                  total_queries=len(fixture['queries']), total_pairs=sum(len(q['candidates']) for q in fixture['queries'].values()))
    result.pop('error', None)
    save(args.output, result)
    start = time.perf_counter()
    try:
        tokenizer = Tokenizer.from_file(str(args.model / 'tokenizer.json'))
        tokenizer.no_padding()
        tokenizer.no_truncation()
        model = load_model(args.model, 'npu:0', progress=lambda step: print(step, flush=True)).float()
        torch.npu.synchronize()
        torch.npu.empty_cache()
        if model.cache_identity != manifest['model_identity'] or any(p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError('Model identity or dtype differs from precomputation')
        ram, by_document = {}, {}
        preload_start = time.perf_counter()
        with ordinary_preload_pages():
            for index, (did, doc) in enumerate(fixture['documents'].items()):
                prefix = encode_document_prefix(tokenizer, doc['text'])
                key = hashlib.sha256(json.dumps([model.cache_identity, 'torch.bfloat16', prefix]).encode()).hexdigest()
                row = manifest['documents'][did]
                path = args.cache_dir / (key + '.safetensors')
                if row['key'] != key or row['prefix_tokens'] != len(prefix) or path.stat().st_size != row['file_bytes']:
                    raise ValueError('Cache identity, prefix length or size changed: ' + did)
                if key not in ram:
                    cache = DocumentCache.load(path)
                    if (cache.prefix_ids != prefix or cache.model_identity != model.cache_identity or
                        cache.nbytes != row['cache_bytes'] or any(t.dtype != torch.bfloat16 for t in cache.tensors().values())):
                        raise ValueError('Incompatible cache contents: ' + did)
                    ram[key] = cache
                by_document[did] = key
                if (index + 1) % 100 == 0:
                    print(json.dumps({'stage': 'preload', 'documents': index + 1, 'of': len(documents),
                                      'seconds': time.perf_counter() - preload_start}), flush=True)
        result.update(ram_cache_bytes=sum(c.nbytes for c in ram.values()), ram_preload_s=time.perf_counter() - preload_start,
                      model_identity=model.cache_identity, preload_thp_disabled=True)
        save(args.output, result)
        print(json.dumps({'stage': 'scoring', 'ram_cache_bytes': result['ram_cache_bytes']}), flush=True)
        with torch.inference_mode():
            for qid, query in sorted(fixture['queries'].items()):
                if qid in result['predictions']:
                    continue
                values, rows = {}, {}
                query_start = time.perf_counter()
                for did in sorted(query['candidates'], key=lambda d: (-query['candidates'][d], d)):
                    if fixture['ignore_identical_ids'] and qid == did:
                        # Keep membership intact; the established evaluator filters this entry.
                        values[did] = 0.0
                        rows[did] = {'excluded_self_match': True}
                        continue
                    began = time.perf_counter()
                    pair = {'task': fixture['task'], 'qid': qid, 'did': did, 'query': query['text'],
                            'document': fixture['documents'][did]['text'], 'instruction': fixture['instruction']}
                    record = encode_record(tokenizer, request_for(pair))
                    ids = torch.tensor([record.input_ids], dtype=torch.long, device='npu:0')
                    packed = ram[by_document[did]].to('npu:0')
                    cache = cache_cast(packed, torch.float32)
                    del packed
                    logits = model(ids, record, cache=cache)[0]
                    probabilities = logits.float().softmax(-1).cpu().tolist()
                    raw = logits.float().cpu().tolist()
                    if len(raw) != 2 or not all(math.isfinite(v) for v in raw + probabilities):
                        raise ValueError('Invalid logits or probabilities')
                    values[did] = dict(zip(record.questions[0].option_ids, probabilities))['true']
                    rows[did] = {'logits': raw, 'input_tokens': len(record.input_ids),
                                 'prefix_tokens': len(cache.prefix_ids), 'wall_s': time.perf_counter() - began}
                    del ids, cache, logits
                result['predictions'][qid] = values
                result['queries'][qid] = {'pairs': rows, 'wall_s': time.perf_counter() - query_start}
                result.update(completed_queries=len(result['predictions']),
                              completed_pairs=sum(len(v) for v in result['predictions'].values()))
                save(args.output, result)
                sync_saved(args.output)
                print(json.dumps({'stage': 'query_saved', 'qid': qid, 'completed_queries': result['completed_queries'],
                                  'of': result['total_queries'], 'completed_pairs': result['completed_pairs'],
                                  'query_s': result['queries'][qid]['wall_s']}), flush=True)
        validate_predictions(fixture, result['predictions'], complete=True)
        result['status'] = 'completed'
    except BaseException as exc:
        result.update(status='failed', error=repr(exc))
        raise
    finally:
        result['elapsed_this_run_s'] = time.perf_counter() - start
        result['transformers_imported'] = any(n == 'transformers' or n.startswith('transformers.') for n in sys.modules)
        if result['status'] == 'completed' and result['transformers_imported']:
            result.update(status='failed', error='Unexpected Transformers import')
        save(args.output, result)
        sync_saved(args.output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('score', 'evaluate'))
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', type=Path)
    p.add_argument('--cache-dir', type=Path)
    p.add_argument('--cache-manifest', type=Path)
    args = p.parse_args()
    if args.command == 'score' and not all((args.model, args.cache_dir, args.cache_manifest)):
        p.error('score requires --model, --cache-dir and --cache-manifest')
    {'score': score, 'evaluate': evaluate}[args.command](args)


if __name__ == '__main__':
    main()
