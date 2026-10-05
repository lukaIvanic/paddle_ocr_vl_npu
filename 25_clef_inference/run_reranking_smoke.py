"""Six real MTEB-R/CMTEB-R pairs: prepare with the pinned evaluator, run uncached.

One fixed label-balanced query per language is a protocol smoke, not a quality
estimate. The document is STATE; query and task instruction are in the schema.
"""
import argparse
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

from run_local_smoke import NoTransformers, encode_record, systemone_answer

BENCHMARK_DIR = Path(__file__).resolve().parents[1] / '22_qwen3_embedding_benchmark'


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix('.partial')
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    partial.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def request_for(pair):
    return {'id': pair['task'] + '/' + pair['qid'] + '/' + pair['did'],
            'state': pair['document'],
            'questions': {'relevance': {'type': 'noul', 'instructions':
                'Task: ' + pair['instruction'] + '\nQuery: ' + pair['query'] +
                '\nDoes the document in STATE meet the task requirements for this query?'}}}


def choose_pairs(candidates, qrels, ignore_identical_ids):
    # Explicitly choose a label-balanced plumbing sample, never a score sample.
    # Selection does not inspect Qwen or Clef scores.
    for qid in sorted(candidates):
        docs = sorted(candidates[qid], key=lambda d: (-candidates[qid][d], d))
        if len(docs) != 100:
            raise ValueError('Expected the original top-100 candidate set')
        docs = [d for d in docs if not (ignore_identical_ids and d == qid)]
        positive = [d for d in docs if qrels[qid].get(d, 0) > 0]
        negative = [d for d in docs if qrels[qid].get(d, 0) == 0]
        if positive and len(negative) >= 2:
            return qid, [positive[0], negative[0], negative[-1]]
    raise ValueError('No query with a positive and two negative candidates')


def load_english_task(name, load_task):
    """Resolve MTEB's implicit default qrels config in the offline cache."""
    import importlib
    from types import SimpleNamespace
    module = importlib.import_module('mteb.abstasks.AbsTaskRetrieval')
    original = module.load_dataset

    def explicit_default(repo, *args, **kwargs):
        if not args and 'name' not in kwargs:
            kwargs['name'] = 'default'
        data = original(repo, *args, **kwargs)
        # datasets' offline cache loader can ignore revision and choose latest.
        # Verify every returned Arrow cache belongs to the requested revision.
        parts = list(data.values()) if hasattr(data, 'values') else [data]
        files = [f['filename'] for part in parts for f in part.cache_files]
        revision = kwargs['revision']
        if not files or any(revision not in Path(f).parts for f in files):
            raise ValueError('Offline English dataset cache revision mismatch')
        return data

    module.load_dataset = explicit_default
    try:
        return load_task(name, SimpleNamespace(state={}))
    finally:
        module.load_dataset = original


def prepare(args):
    sys.path.insert(0, str(BENCHMARK_DIR))
    import mteb
    from mteb.evaluation.evaluators.RetrievalEvaluator import corpus_to_str
    from protocol import TASKS, validate_task
    from run_english_suite import load_task
    from suite_protocol import ENGLISH

    if importlib.metadata.version('mteb') != '1.38.9':
        raise ValueError('Use the original MTEB 1.38.9 evaluator')
    if args.output.exists():
        raise FileExistsError(args.output)
    pairs, groups = [], {}
    sources = [('FiQA2018', args.english / 'reranker/FiQA2018', 'test'),
               ('EcomRetrieval', args.chinese / 'EcomRetrieval', 'dev')]
    for name, folder, split in sources:
        source = json.loads((folder / 'manifest.json').read_text())
        if split == 'test':
            task, _ = load_english_task(name, load_task)
            candidate_path = args.english / 'embedding' / name / 'mteb' / (name + '_default_predictions.json')
            candidate_hash = source['candidates_sha256']
            instruction = ENGLISH[name][2]
            if source['query_instruction'] != instruction:
                raise ValueError('English task instruction changed')
            revision = ENGLISH[name][1]
        else:
            task = mteb.get_tasks(tasks=[name])[0]
            validate_task(task)
            task.load_data()
            candidate_path = Path(source['candidate_file'])
            if not candidate_path.is_absolute():
                candidate_path = args.source_repo / candidate_path
            candidate_hash = source['sha256']
            instruction = TASKS[name][2]
            revision = TASKS[name][0]
            if source['dataset_revision'] != revision or source['qrels_revision'] != TASKS[name][1] or source['instruction'] != instruction:
                raise ValueError('Chinese task contract changed')
        if digest(candidate_path) != candidate_hash:
            raise ValueError('Saved embedding candidates changed')
        candidates = json.loads(candidate_path.read_text())
        predictions_path = folder / 'predictions.json'
        predictions = json.loads(predictions_path.read_text())
        qrels = task.relevant_docs[split]
        if set(candidates) != set(qrels) or set(predictions) != set(qrels):
            raise ValueError('Saved query coverage changed')
        qid, dids = choose_pairs(candidates, qrels, task.ignore_identical_ids)
        if set(candidates[qid]) != set(predictions[qid]):
            raise ValueError('Saved reranker candidate membership changed')
        docs = corpus_to_str([task.corpus[split][did] for did in dids])
        for did, doc in zip(dids, docs):
            pairs.append({'task': name, 'qid': qid, 'did': did, 'query': task.queries[split][qid],
                          'document': doc, 'instruction': instruction, 'relevance': qrels[qid].get(did, 0),
                          'embedding_score': candidates[qid][did], 'qwen_score': predictions[qid][did]})
        groups[name] = {'qid': qid, 'split': split, 'dataset_revision': revision,
                        'qrels_revision': TASKS[name][1] if split == 'dev' else revision,
                        'qrels': qrels[qid], 'ignore_identical_ids': task.ignore_identical_ids,
                        'candidate_file': str(candidate_path), 'candidate_sha256': candidate_hash,
                        'qwen_predictions_sha256': digest(predictions_path),
                        'qwen_manifest': source, 'qwen_manifest_sha256': digest(folder / 'manifest.json'),
                        'selected_document_ids': dids}
    save(args.output, {'scope': 'six_pair_protocol_smoke', 'mteb_version': '1.38.9',
                      'selection': 'first sorted query with a positive and two negatives in saved top100; first positive, first and last negative in embedding order',
                      'truncation': 'none', 'score': 'unrounded P(true) from noul logits',
                      'groups': groups, 'pairs': pairs})
    print(json.dumps({'prepared': str(args.output), 'pairs': len(pairs), 'tasks': list(groups)}))


def evaluate(args):
    sys.path.insert(0, str(BENCHMARK_DIR))
    from run_reranker_evaluation import metric_summary
    fixture = json.loads(args.fixture.read_text())
    result = json.loads(args.result.read_text())
    if result['status'] != 'completed' or result['fixture_sha256'] != digest(args.fixture):
        raise ValueError('Incomplete run or changed fixture')
    scores = {(r['task'], r['qid'], r['did']): r['score'] for r in result['rows']}
    expected = {(p['task'], p['qid'], p['did']) for p in fixture['pairs']}
    if set(scores) != expected or len(result['rows']) != len(expected):
        raise ValueError('Run does not cover exactly the frozen fixture')
    metrics, ordering = {}, {}
    for name, group in fixture['groups'].items():
        pairs = [p for p in fixture['pairs'] if p['task'] == name]
        qid = group['qid']
        baseline = {qid: {p['did']: p['embedding_score'] for p in pairs}}
        qwen = {qid: {p['did']: p['qwen_score'] for p in pairs}}
        clef = {qid: {p['did']: scores[(name, qid, p['did'])] for p in pairs}}
        qrels = {qid: group['qrels']}
        clef_metrics, _ = metric_summary(clef, baseline, qrels, group['ignore_identical_ids'])
        qwen_metrics, _ = metric_summary(qwen, baseline, qrels, group['ignore_identical_ids'])
        metrics[name] = {'clef': clef_metrics['reranker'], 'qwen': qwen_metrics['reranker'],
                         'embedding': clef_metrics['embedding']}
        ordering[name] = {label: sorted(values[qid], key=lambda d: (-values[qid][d], d))
                          for label, values in [('clef', clef), ('qwen', qwen), ('embedding', baseline)]}
    # Independent tiny analytic check of the existing evaluator's gain/discount.
    toy, _ = metric_summary({'q': {'irrelevant': 2., 'relevant': 1.}},
                            {'q': {'irrelevant': 2., 'relevant': 1.}},
                            {'q': {'relevant': 1}}, False)
    if not math.isclose(toy['reranker']['ndcg_cut_10'], 1 / math.log2(3), abs_tol=1e-12):
        raise AssertionError('Unexpected NDCG semantics')
    result.update(metrics=metrics, ordering=ordering,
                  quality_scope='one label-balanced query and three selected top100 candidates per task; not benchmark accuracy',
                  metric_check='existing pytrec_eval NDCG@10; independent two-document analytic check passed')
    save(args.result, result)
    print(json.dumps({'metrics': metrics, 'ordering': ordering}, ensure_ascii=False))


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    fixture = json.loads(args.fixture.read_text())
    if len(fixture['pairs']) != 6 or len(fixture['groups']) != 2:
        raise ValueError('Expected the fixed six-pair, two-task sample')
    sys.meta_path.insert(0, NoTransformers())
    import torch
    import torch_npu  # noqa: F401
    from tokenizers import Tokenizer
    from local_modeling_clef import load_model

    if not torch.npu.is_available() or '910B' not in torch.npu.get_device_name(0):
        raise RuntimeError('910B NPU required; no CPU fallback')
    torch.npu.set_device(0)
    torch.set_num_threads(8)
    free, _ = torch.npu.mem_get_info()
    if free < 26 * 1024**3:
        raise RuntimeError('Less than 26 GiB free; refusing model load')
    tokenizer = Tokenizer.from_file(str(args.model / 'tokenizer.json'))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    requests = [request_for(pair) for pair in fixture['pairs']]
    encoded = [encode_record(tokenizer, request, args.max_length) for request in requests]
    result = {'status': 'running', 'scope': fixture['scope'], 'cache_reuse': False, 'dtype': 'bfloat16',
              'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'fixture_sha256': digest(args.fixture), 'hostname': platform.node(),
              'physical_device': os.environ.get('ASCEND_RT_VISIBLE_DEVICES'), 'device_name': torch.npu.get_device_name(0),
              'request_template': 'document as STATE; task and query in relevance-question instructions',
              'ranking_score': fixture['score'], 'max_length': args.max_length, 'rows': []}
    try:
        model = load_model(args.model, 'npu:0', progress=lambda step: print(step, flush=True))
        with torch.inference_mode():
            for index in [-1, *range(6)]:
                item = max(index, 0)
                record = encoded[item]
                ids = torch.tensor([record.input_ids], device='npu:0', dtype=torch.long)
                torch.npu.synchronize()
                start = time.perf_counter()
                logits = model(ids, record)[0]
                torch.npu.synchronize()
                elapsed = time.perf_counter() - start
                values = logits.float().cpu()
                if logits.dtype != torch.bfloat16 or values.shape != (2,) or not torch.isfinite(values).all():
                    raise ValueError('Invalid noul logits')
                # NPU softmax, as in the existing local runner and official release.
                probabilities = dict(zip(record.questions[0].option_ids, logits.float().softmax(-1).cpu().tolist()))
                if index < 0:
                    continue
                pair = fixture['pairs'][item]
                spans = [(q.question_id, q.question_span, q.option_spans, q.option_ids) for q in record.questions]
                row = {k: pair[k] for k in ('task', 'qid', 'did', 'relevance', 'qwen_score', 'embedding_score')}
                row.update(input_tokens=len(record.input_ids), input_sha256=hashlib.sha256(json.dumps([record.input_ids, spans]).encode()).hexdigest(),
                           logits=values.tolist(), score=probabilities['true'], model_s=elapsed,
                           answer=systemone_answer(requests[item]['questions']['relevance'], probabilities))
                result['rows'].append(row)
                save(args.output, result)
                print(json.dumps(row, ensure_ascii=False), flush=True)
                del ids, logits
        result['status'] = 'completed'
        result['transformers_imported'] = any(n == 'transformers' or n.startswith('transformers.') for n in sys.modules)
        if result['transformers_imported']:
            raise AssertionError('Transformers imported during local inference')
    except BaseException as exc:
        result.update(status='failed', error=str(exc))
        raise
    finally:
        save(args.output, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--english', type=Path, required=True)
    prep.add_argument('--chinese', type=Path, required=True)
    prep.add_argument('--source-repo', type=Path, required=True)
    prep.add_argument('--output', type=Path, required=True)
    infer = commands.add_parser('run')
    infer.add_argument('--fixture', type=Path, required=True)
    infer.add_argument('--model', type=Path, required=True)
    infer.add_argument('--output', type=Path, required=True)
    infer.add_argument('--max-length', type=int, default=3072)
    metrics = commands.add_parser('evaluate')
    metrics.add_argument('--fixture', type=Path, required=True)
    metrics.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    {'prepare': prepare, 'run': run, 'evaluate': evaluate}[args.command](args)
