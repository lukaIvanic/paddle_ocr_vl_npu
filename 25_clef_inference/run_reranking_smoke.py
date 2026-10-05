"""Small pinned MTEB-R/CMTEB-R fixtures: prepare, run uncached, evaluate.

The six-pair fixture checks protocol; the forty-pair fixture covers input
lengths. Neither estimates benchmark accuracy. Document is STATE; the query
and task instruction are in the schema. The cache command measures actual document reuse and storage transfers.
"""
import argparse
import bisect
import gc
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

from run_local_smoke import NoTransformers, SYSTEM_PROMPT, encode_record, systemone_answer

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


def prepare_task(args):
    """Freeze every original top-100 candidate, storing each document once."""
    from run_local_smoke import encode_document_prefix
    sys.path.insert(0, str(BENCHMARK_DIR))
    from suite_protocol import ENGLISH
    from run_english_suite import load_task
    from mteb.evaluation.evaluators.RetrievalEvaluator import corpus_to_str
    from tokenizers import Tokenizer
    if importlib.metadata.version('mteb') != '1.38.9':
        raise ValueError('Use the original MTEB 1.38.9 evaluator')
    if args.output.exists():
        raise FileExistsError(args.output)
    sizes = []
    for name in ENGLISH:
        folder = args.english / 'reranker' / name
        source = json.loads((folder / 'manifest.json').read_text())
        path = args.english / 'embedding' / name / 'mteb' / (name + '_default_predictions.json')
        if digest(path) != source['candidates_sha256']:
            raise ValueError('Candidate hash changed: ' + name)
        candidates = json.loads(path.read_text())
        if any(len(docs) != 100 for docs in candidates.values()):
            raise ValueError('Expected original top100 candidates')
        sizes.append({'task': name, 'queries': len(candidates),
                      'pairs': sum(map(len, candidates.values())),
                      'unique_candidate_documents': len({d for docs in candidates.values() for d in docs}),
                      'corpus_documents': source['documents']})
    name = args.task
    if name not in ENGLISH:
        raise ValueError('Not in our pinned MTEB-R suite')
    source_folder = args.english / 'reranker' / name
    source = json.loads((source_folder / 'manifest.json').read_text())
    path = args.english / 'embedding' / name / 'mteb' / (name + '_default_predictions.json')
    candidates = json.loads(path.read_text())
    task, _ = load_english_task(name, load_task)
    qrels = task.relevant_docs['test']
    predictions = json.loads((source_folder / 'predictions.json').read_text())
    if set(candidates) != set(qrels) or set(predictions) != set(qrels):
        raise ValueError('Query coverage changed')
    if source['query_instruction'] != ENGLISH[name][2]:
        raise ValueError('Instruction changed')
    if any(set(candidates[q]) != set(predictions[q]) for q in candidates):
        raise ValueError('Qwen candidate membership changed')
    tokenizer = Tokenizer.from_file(str(args.model / 'tokenizer.json'))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    # Keep original candidate membership in queries; ignore-identical applies at scoring.
    dids = sorted({d for q, docs in candidates.items() for d in docs
                   if not (task.ignore_identical_ids and d == q)})
    texts = corpus_to_str([task.corpus['test'][d] for d in dids])
    documents = {d: {'text': text, 'prefix_tokens': len(encode_document_prefix(tokenizer, text))}
                 for d, text in zip(dids, texts)}
    queries = {q: {'text': task.queries['test'][q], 'candidates': candidates[q],
                   'qrels': qrels[q], 'qwen_scores': predictions[q]} for q in sorted(candidates)}
    lengths = [v['prefix_tokens'] for v in documents.values()]
    result = {'scope': 'complete_task_candidate_documents', 'task': name, 'split': 'test',
        'mteb_version': '1.38.9', 'instruction': ENGLISH[name][2], 'dataset_revision': ENGLISH[name][1],
        'ignore_identical_ids': task.ignore_identical_ids, 'truncation': 'none',
        'candidate_sha256': digest(path), 'source_manifest_sha256': digest(source_folder / 'manifest.json'),
        'qwen_predictions_sha256': digest(source_folder / 'predictions.json'),
        'tokenizer_sha256': digest(args.model / 'tokenizer.json'),
        'suite_sizes': sorted(sizes, key=lambda r: r['pairs']),
        'prefix_lengths': length_summary([lengths]),
        'estimated_fp32_cache_bytes': sum(50.25 * 1024**2 + n * 80 * 1024 for n in lengths),
        'queries': queries, 'documents': documents}
    save(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('queries', 'documents')}, indent=2), flush=True)


def length_summary(parts):
    """Equal task weight; input parts are a list of per-task length lists."""
    ordered = sorted((n, 1 / len(part)) for part in parts for n in part)
    total = sum(w for _, w in ordered)
    result = {'count': len(ordered), 'min': ordered[0][0], 'max': ordered[-1][0]}
    for percentile in (10, 25, 50, 75, 90, 95, 99):
        target, cumulative = total * percentile / 100, 0.
        for n, weight in ordered:
            cumulative += weight
            if cumulative >= target:
                result['p' + str(percentile)] = n
                break
    result['above_3072'] = sum(n > 3072 for n, _ in ordered)
    return result


def length_cdf(pool, key):
    ordered = sorted(row[key] for row in pool)
    return lambda value: (bisect.bisect_left(ordered, value) +
                          bisect.bisect_right(ordered, value)) / (2 * len(ordered))


def load_chinese_task(name, mteb, validate_task, expected):
    """Pin both corpus/query and qrels revisions, including offline Arrow files."""
    import importlib
    module = importlib.import_module('mteb.abstasks.AbsTaskRetrieval')
    task = mteb.get_tasks(tasks=[name])[0]
    validate_task(task)
    path = task.metadata.dataset['path']
    original = module.load_dataset

    def pinned(repo, *args, **kwargs):
        if repo == path:
            revision = expected[0]
        elif repo == path + '-qrels':
            revision = expected[1]
        else:
            raise ValueError('Unexpected Chinese dataset: ' + repo)
        kwargs['revision'] = revision
        data = original(repo, *args, **kwargs)
        parts = list(data.values()) if hasattr(data, 'values') else [data]
        files = [f['filename'] for part in parts for f in part.cache_files]
        if not files or any(revision not in Path(f).parts for f in files):
            raise ValueError('Offline Chinese dataset cache revision mismatch: ' + repo)
        return data

    module.load_dataset = pinned
    try:
        task.load_data()
    finally:
        module.load_dataset = original
    return task


def prepare_lengths(args):
    """Build a length-coverage fixture, with no model-score-based selection."""
    sys.path.insert(0, str(BENCHMARK_DIR))
    import mteb
    from mteb.evaluation.evaluators.RetrievalEvaluator import corpus_to_str
    from tokenizers import Tokenizer
    from protocol import TASKS, validate_task
    from run_english_suite import load_task
    from suite_protocol import ENGLISH

    if importlib.metadata.version('mteb') != '1.38.9':
        raise ValueError('Use the original MTEB 1.38.9 evaluator')
    if args.output.exists():
        raise FileExistsError(args.output)
    tokenizer = Tokenizer.from_file(str(args.model / 'tokenizer.json'))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    tokenizer_hash = digest(args.model / 'tokenizer.json')
    prefix = tokenizer.encode(f'<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n', add_special_tokens=False).ids
    profiles, pool = [], []
    for language, contracts in (('en', ENGLISH), ('zh', TASKS)):
        for name, contract in contracts.items():
            folder = args.english / 'reranker' / name if language == 'en' else (
                args.ecom / name if name == 'EcomRetrieval' else
                args.t2 / name if name == 'T2Retrieval' else args.chinese / name)
            source = json.loads((folder / 'manifest.json').read_text())
            if language == 'en':
                candidate = args.english / 'embedding' / name / 'mteb' / (name + '_default_predictions.json')
                expected_hash = source['candidates_sha256']
                revision, qrevision, instruction, split = contract[1], contract[1], contract[2], 'test'
                if source['query_instruction'] != instruction:
                    raise ValueError('English instruction changed')
            else:
                candidate = Path(source['candidate_file'])
                if not candidate.is_absolute():
                    candidate = args.source_repo / candidate
                expected_hash = source['sha256']
                revision, qrevision, instruction, split = contract[0], contract[1], contract[2], 'dev'
                if (source['dataset_revision'], source['qrels_revision'], source['instruction']) != (revision, qrevision, instruction):
                    raise ValueError('Chinese task contract changed')
            if digest(candidate) != expected_hash:
                raise ValueError('Candidate hash changed: ' + name)
            contract_record = {'task': name, 'language': language, 'split': split,
                'dataset_revision': revision, 'qrels_revision': qrevision,
                'candidate_file': str(candidate), 'candidate_sha256': expected_hash,
                'source_manifest_sha256': digest(folder / 'manifest.json'),
                'tokenizer_sha256': tokenizer_hash, 'profile_queries_per_task': 64,
                'selection_seed': 'clef-cache-lengths-v1', 'source_code_sha256': digest(Path(__file__))}
            cached = args.profile_dir / (name + '.json')
            if cached.exists():
                profile = json.loads(cached.read_text())
                if profile['contract'] != contract_record:
                    raise ValueError('Length-profile resume contract changed: ' + name)
            else:
                print(json.dumps({'profiling': name, 'language': language}), flush=True)
                candidates = json.loads(candidate.read_text())
                task = load_english_task(name, load_task)[0] if language == 'en' else load_chinese_task(name, mteb, validate_task, contract)
                qrels = task.relevant_docs[split]
                if set(candidates) != set(qrels):
                    raise ValueError('Query coverage changed: ' + name)
                qids = sorted(candidates, key=lambda q: hashlib.sha256(('clef-cache-lengths-v1/' + name + '/' + q).encode()).hexdigest())[:64]
                lengths = {key: [] for key in ('document_tokens', 'prefix_tokens', 'question_side_tokens', 'input_tokens', 'query_tokens')}
                query_rows = []
                for qid in qids:
                    if len(candidates[qid]) != 100:
                        raise ValueError('Expected original top100 candidates')
                    dids = sorted(d for d in candidates[qid] if not (task.ignore_identical_ids and d == qid))
                    docs = corpus_to_str([task.corpus[split][did] for did in dids])
                    encodings = tokenizer.encode_batch(docs, add_special_tokens=False)
                    doc_lengths = [len(e.ids) for e in encodings]
                    query = task.queries[split][qid]
                    base = {'task': name, 'qid': qid, 'did': '', 'query': query, 'document': '', 'instruction': instruction}
                    # An empty STATE has zero tokens; the remainder is exactly the original schema and suffix.
                    empty = encode_record(tokenizer, request_for(base), max_length=2**31-1)
                    question_side = len(empty.input_ids) - len(prefix)
                    query_length = len(tokenizer.encode(query, add_special_tokens=False).ids)
                    for size in doc_lengths:
                        lengths['document_tokens'].append(size)
                        lengths['prefix_tokens'].append(len(prefix) + size)
                        lengths['question_side_tokens'].append(question_side)
                        lengths['input_tokens'].append(len(prefix) + size + question_side)
                        lengths['query_tokens'].append(query_length)
                    ordered = sorted(range(len(dids)), key=lambda i: (doc_lengths[i], dids[i]))
                    selected = [ordered[round(p*(len(ordered)-1))] for p in (.1, .5, .9, .99)]
                    if len(set(selected)) != 4:
                        raise ValueError('Insufficient distinct documents')
                    pair_rows = []
                    for percentile, i in zip((10, 50, 90, 99), selected):
                        pair = {**base, 'did': dids[i], 'document': docs[i], 'language': language,
                            'relevance': qrels[qid].get(dids[i], 0), 'embedding_score': candidates[qid][dids[i]],
                            'document_length_percentile': percentile, 'document_tokens': doc_lengths[i],
                            'prefix_tokens': len(prefix) + doc_lengths[i], 'question_side_tokens': question_side,
                            'query_tokens': query_length, 'input_tokens': len(prefix) + doc_lengths[i] + question_side}
                        record = encode_record(tokenizer, request_for(pair), max_length=pair['input_tokens'])
                        if len(record.input_ids) != pair['input_tokens'] or record.input_ids[:len(prefix)] != tuple(prefix):
                            raise ValueError('Counted lengths differ from original encoding')
                        pair['input_sha256'] = hashlib.sha256(json.dumps(record.input_ids).encode()).hexdigest()
                        pair_rows.append(pair)
                    query_rows.append({'task': name, 'language': language, 'qid': qid,
                        'query_tokens': query_length, 'median_document_tokens': doc_lengths[ordered[len(ordered)//2]],
                        'question_side_tokens': question_side, 'qrels': qrels[qid],
                        'ignore_identical_ids': task.ignore_identical_ids, 'pairs': pair_rows})
                profile = {'contract': contract_record, 'lengths': lengths, 'queries': query_rows}
                save(cached, profile)
                del task, candidates
                gc.collect()
            profiles.append(profile)
            pool.extend(profile['queries'])
            print(json.dumps({'profiled': name, 'queries': len(profile['queries']), 'pairs': len(profile['lengths']['input_tokens'])}), flush=True)

    selected_queries = []
    for language in ('en', 'zh'):
        available = [q for q in pool if q['language'] == language]
        query_cdf = length_cdf(available, 'query_tokens')
        doc_cdf = length_cdf(available, 'median_document_tokens')
        used = set()
        for target in (.1, .3, .5, .7, .9):
            row = min((q for q in available if q['task'] not in used),
                      key=lambda q: (abs(query_cdf(q['query_tokens'])-target) + abs(doc_cdf(q['median_document_tokens'])-target), q['task'], q['qid']))
            selected_queries.append({**row, 'target_length_percentile': int(target*100)})
            used.add(row['task'])
    groups, pairs = {}, []
    # Qwen scores are read only after every selection is fixed.
    for query in selected_queries:
        name = query['task']
        folder = args.english / 'reranker' / name if query['language'] == 'en' else (
            args.ecom / name if name == 'EcomRetrieval' else args.t2 / name if name == 'T2Retrieval' else args.chinese / name)
        predictions = json.loads((folder / 'predictions.json').read_text())
        qid = query['qid']
        provenance = next(p['contract'] for p in profiles if p['contract']['task'] == name)
        for pair in query['pairs']:
            pairs.append({**pair, 'qwen_score': predictions[qid][pair['did']]})
        groups[name + '/' + qid] = {**provenance, 'qid': qid, 'qrels': query['qrels'],
            'ignore_identical_ids': query['ignore_identical_ids'], 'selected_document_ids': [p['did'] for p in query['pairs']],
            'query_tokens': query['query_tokens'], 'question_side_tokens': query['question_side_tokens'],
            'target_length_percentile': query['target_length_percentile'], 'qwen_predictions_sha256': digest(folder / 'predictions.json')}
    reference = {language: {key: length_summary([p['lengths'][key] for p in profiles if p['contract']['language'] == language])
                            for key in ('document_tokens', 'prefix_tokens', 'query_tokens', 'question_side_tokens', 'input_tokens')}
                 for language in ('en', 'zh')}
    sample = {language: {key: length_summary([[p[key] for p in pairs if p['language'] == language]])
                         for key in ('document_tokens', 'prefix_tokens', 'query_tokens', 'question_side_tokens', 'input_tokens')}
              for language in ('en', 'zh')}
    save(args.output, {'scope': 'length_stratified_cache_check', 'mteb_version': '1.38.9', 'expected_pairs': 40,
        'selection': '64 hash-sampled queries per task across all 18 pinned tasks; five distinct tasks per language nearest joint query/median-document length percentiles 10/30/50/70/90; four original candidates at document-length percentiles 10/50/90/99',
        'quality_scope': 'length coverage and cached/uncached correctness; tail-enriched, not frequency-representative benchmark accuracy',
        'selection_uses_model_scores': False, 'selection_uses_relevance_labels': False,
        'truncation': 'none', 'score': 'unrounded P(true) from noul logits', 'tokenizer_sha256': tokenizer_hash,
        'reference_weighting': 'equal task weight for length summaries; up to 64 deterministic sampled queries/task, all original top100 except protocol self-matches',
        'reference_lengths': reference, 'sample_lengths': sample, 'profiled_tasks': [p['contract'] for p in profiles],
        'groups': groups, 'pairs': pairs})
    print(json.dumps({'prepared': str(args.output), 'pairs': len(pairs), 'queries': len(groups), 'sample_lengths': sample}), flush=True)


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
        task_name = group.get('task', name)
        qid = group['qid']
        pairs = [p for p in fixture['pairs'] if p['task'] == task_name and p['qid'] == qid]
        baseline = {qid: {p['did']: p['embedding_score'] for p in pairs}}
        qwen = {qid: {p['did']: p['qwen_score'] for p in pairs}}
        clef = {qid: {p['did']: scores[(task_name, qid, p['did'])] for p in pairs}}
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
                  quality_scope=fixture.get('quality_scope', 'one label-balanced query and three selected top100 candidates per task; not benchmark accuracy'),
                  metric_check='existing pytrec_eval NDCG@10; independent two-document analytic check passed')
    save(args.result, result)
    print(json.dumps({'metrics': metrics, 'ordering': ordering}, ensure_ascii=False))


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    fixture = json.loads(args.fixture.read_text())
    if len(fixture['pairs']) != fixture.get('expected_pairs', 6):
        raise ValueError('Fixture pair count differs from its declared scope')
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
            for index in [-1, *range(len(fixture['pairs']))]:
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


def precache_task(args):
    """Persist one cache per candidate document, with a resumable manifest."""
    import shutil
    from run_local_smoke import encode_document_prefix
    fixture = json.loads(args.fixture.read_text())
    if fixture['scope'] != 'complete_task_candidate_documents' or fixture['truncation'] != 'none':
        raise ValueError('Expected an untruncated full-task fixture')
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
        raise RuntimeError('910B NPU required; no CPU fallback')
    torch.npu.set_device(0)
    torch.set_num_threads(8)
    if torch.npu.mem_get_info()[0] < (50 if args.dtype == 'float32' else 28) * 1024**3:
        raise RuntimeError('Insufficient free NPU memory')
    if digest(args.model / 'tokenizer.json') != fixture['tokenizer_sha256']:
        raise ValueError('Tokenizer changed')
    tokenizer = Tokenizer.from_file(str(args.model / 'tokenizer.json'))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    prefixes = {did: encode_document_prefix(tokenizer, doc['text']) for did, doc in fixture['documents'].items()}
    if any(len(p) != fixture['documents'][d]['prefix_tokens'] for d, p in prefixes.items()):
        raise ValueError('Document token count changed')
    if max(map(len, prefixes.values())) > args.max_prefix_length:
        raise ValueError('Document exceeds explicit prefix limit; no truncation')
    contract = {'fixture_sha256': digest(args.fixture), 'dtype': args.dtype,
                'cache_dir': str(args.cache_dir.resolve()), 'max_prefix_length': args.max_prefix_length}
    result = json.loads(args.output.read_text()) if args.output.exists() else {
        'status': 'running', 'task': fixture['task'], 'contract': contract, 'documents': {}}
    if result['contract'] != contract:
        raise ValueError('Resume contract changed')
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    bytes_per_token = 81920 if args.dtype == 'float32' else 40960
    fixed_bytes = int((50.25 if args.dtype == 'float32' else 49.125) * 1024**2)
    remaining_bytes = sum(fixed_bytes + len(p) * bytes_per_token + 65536
                          for d, p in prefixes.items() if d not in result['documents'])
    if shutil.disk_usage(args.cache_dir).free < remaining_bytes:
        raise RuntimeError('Not enough disk space for remaining document caches')
    try:
        model = load_model(args.model, 'npu:0', progress=lambda step: print(step, flush=True))
        if args.dtype == 'float32':
            model.float()
            torch.npu.empty_cache()
        if result.get('model_identity', model.cache_identity) != model.cache_identity:
            raise ValueError('Resume model identity changed')
        result.update(status='running', model_identity=model.cache_identity,
            git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            physical_device=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
            torch=torch.__version__, torch_npu=torch_npu.__version__, total_documents=len(prefixes))
        result.pop('error', None)
        save(args.output, result)
        # Longest first: prove peak memory fits before committing hours of work.
        ordered = sorted(prefixes, key=lambda d: (-len(prefixes[d]), d))
        start = time.perf_counter()
        with torch.inference_mode():
            for did in ordered:
                prefix = prefixes[did]
                key = hashlib.sha256(json.dumps([model.cache_identity, str(getattr(torch, args.dtype)), tuple(prefix)]).encode()).hexdigest()
                path = args.cache_dir / (key + '.safetensors')
                if did in result['documents']:
                    row = result['documents'][did]
                    if row['key'] != key or not path.exists() or path.stat().st_size != row['file_bytes']:
                        raise ValueError('Saved cache missing or incompatible: ' + did)
                    continue
                torch.npu.synchronize()
                began = time.perf_counter()
                cache = model.prepare_document(prefix)
                torch.npu.synchronize()
                prepare_s = time.perf_counter() - began
                if cache.key != key:
                    raise AssertionError('Unexpected document identity')
                began = time.perf_counter()
                cpu = cache.to('cpu')
                del cache
                cpu.save(path)
                roundtrip = False
                if not result['documents']:
                    reloaded = DocumentCache.load(path)
                    if reloaded.key != key or any(not torch.equal(t, reloaded.tensors()[n]) for n, t in cpu.tensors().items()):
                        raise AssertionError('Saved cache roundtrip changed tensors')
                    del reloaded
                    roundtrip = True
                result['documents'][did] = {'key': key, 'prefix_tokens': len(prefix),
                    'cache_bytes': cpu.nbytes, 'file_bytes': path.stat().st_size,
                    'prepare_s': prepare_s, 'offload_save_s': time.perf_counter() - began,
                    'roundtrip_checked': roundtrip}
                del cpu
                result.update(completed_documents=len(result['documents']), elapsed_this_run_s=time.perf_counter() - start)
                save(args.output, result)
                print(json.dumps({'completed': len(result['documents']), 'of': len(prefixes), 'did': did,
                                  **result['documents'][did]}), flush=True)
        result.update(status='completed', cache_bytes=sum(r['cache_bytes'] for r in result['documents'].values()),
                      file_bytes=sum(r['file_bytes'] for r in result['documents'].values()))
    except BaseException as exc:
        result.update(status='failed', error=str(exc))
        raise
    finally:
        save(args.output, result)


def run_cache(args):
    """Prepare once, persist, eagerly preload RAM, then measure real reuse."""
    import statistics
    from dataclasses import replace
    from run_local_smoke import encode_document_prefix
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.repeats < 1:
        raise ValueError('At least one timing repeat is required')
    fixture = json.loads(args.fixture.read_text())
    if len(fixture['pairs']) != fixture.get('expected_pairs', 6):
        raise ValueError('Fixture pair count changed')
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
        raise RuntimeError('910B NPU required; no CPU fallback')
    torch.npu.set_device(0)
    torch.set_num_threads(8)
    target_dtype = getattr(torch, args.dtype)
    required_gib = 50 if target_dtype == torch.float32 else 28
    if torch.npu.mem_get_info()[0] < required_gib * 1024**3:
        raise RuntimeError('Insufficient free NPU memory')
    tokenizer = Tokenizer.from_file(str(args.model / 'tokenizer.json'))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    if fixture.get('tokenizer_sha256') and digest(args.model / 'tokenizer.json') != fixture['tokenizer_sha256']:
        raise ValueError('Tokenizer changed')
    requests = [request_for(p) for p in fixture['pairs']]
    records = [encode_record(tokenizer, r, args.max_length) for r in requests]
    prefixes = [encode_document_prefix(tokenizer, r['state']) for r in requests]
    for pair, record, prefix in zip(fixture['pairs'], records, prefixes):
        if record.input_ids[:len(prefix)] != prefix:
            raise ValueError('Prefix encoding mismatch')
        if pair.get('input_sha256') and hashlib.sha256(json.dumps(record.input_ids).encode()).hexdigest() != pair['input_sha256']:
            raise ValueError('Frozen input changed')
    previous = {}
    if args.reference:
        reference = json.loads(args.reference.read_text())
        if reference['status'] != 'completed' or reference['fixture_sha256'] != digest(args.fixture):
            raise ValueError('Invalid prior baseline')
        previous = {(r['task'], r['qid'], r['did']): r for r in reference['rows']}
    result = {'status': 'running', 'scope': 'actual_document_cache_reuse', 'cache_reuse': True,
        'dtype': args.dtype, 'fixture_sha256': digest(args.fixture),
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'hostname': platform.node(), 'physical_device': os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
        'torch': torch.__version__, 'torch_npu': torch_npu.__version__, 'max_length': args.max_length,
        'timing_repeats': args.repeats, 'timing_scope': 'synchronized eager forward; pre-encoded IDs; no dtype audit; median repeats',
        'disk_read_scope': 'eager CPU preload after writes; filesystem cache may be warm, not cold-SSD latency',
        'prepared': [], 'rows': []}
    save(args.output, result)

    def timed(fn):
        torch.npu.synchronize()
        start = time.perf_counter()
        value = fn()
        torch.npu.synchronize()
        return value, time.perf_counter() - start

    def measure(fn):
        times, first = [], None
        for _ in range(args.repeats):
            value, seconds = timed(fn)
            if value.dtype != target_dtype or value.shape != (2,) or not torch.isfinite(value).all():
                raise ValueError('Invalid model logits')
            if first is None:
                first = value.clone()
            elif not torch.equal(first, value):
                raise AssertionError('Repeated forward changed logits')
            times.append(seconds)
        return first, {'median_s': statistics.median(times), 'samples_s': times}

    def values(logits, record):
        probs = dict(zip(record.questions[0].option_ids, logits.float().softmax(-1).cpu().tolist()))
        return {'logits': logits.float().cpu().tolist(), 'score': probs['true']}

    try:
        model = load_model(args.model, 'npu:0', progress=lambda step: print(step, flush=True))
        if target_dtype == torch.float32:
            model.float()
            torch.npu.empty_cache()
        assert all(p.dtype == target_dtype for p in model.parameters())
        result['model_cache_identity'] = model.cache_identity
        files = []
        with torch.inference_mode():
            ids = torch.tensor([records[0].input_ids], dtype=torch.long, device='npu:0')
            model(ids, records[0])
            warm_cache = model.prepare_document(prefixes[0])
            model(ids, records[0], warm_cache)
            del ids, warm_cache
            for index, prefix in enumerate(prefixes):
                cache, prepare_s = timed(lambda: model.prepare_document(prefix))
                cpu_cache, npu_to_ram_s = timed(lambda: cache.to('cpu'))
                path = args.cache_dir / (cache.key + '.safetensors')
                start = time.perf_counter()
                cpu_cache.save(path)
                save_s = time.perf_counter() - start
                files.append(path)
                result['prepared'].append({'pair': index, 'prefix_tokens': len(prefix), 'cache_bytes': cache.nbytes,
                    'file_bytes': path.stat().st_size, 'prepare_s': prepare_s, 'npu_to_ram_s': npu_to_ram_s,
                    'disk_save_s': save_s, 'path': str(path), 'key': cache.key})
                print(json.dumps({'prepared': index+1, 'of': len(prefixes), **result['prepared'][-1]}), flush=True)
                save(args.output, result)
                del cache, cpu_cache
            # This is the startup preload stage: every selected file becomes
            # an owned CPU allocation before the query loop begins.
            start = time.perf_counter()
            ram = [DocumentCache.load(path) for path in files]
            result['ram_preload_s'] = time.perf_counter() - start
            result['ram_cache_bytes'] = sum(c.nbytes for c in ram)
            for index, (pair, record, cpu_cache) in enumerate(zip(fixture['pairs'], records, ram)):
                ids = torch.tensor([record.input_ids], dtype=torch.long, device='npu:0')
                full, full_time = measure(lambda: model(ids, record)[0])
                transfers = []
                for _ in range(args.repeats):
                    cache, seconds = timed(lambda: cpu_cache.to('npu:0'))
                    transfers.append(seconds)
                    del cache
                cache = cpu_cache.to('npu:0')
                cached, cached_time = measure(lambda: model(ids, record, cache)[0])
                returned = cache.to('cpu')
                if not all(torch.equal(t, returned.tensors()[k]) for k, t in cpu_cache.tensors().items()):
                    raise AssertionError('Query modified document cache')
                row = {k: pair[k] for k in ('task', 'qid', 'did')}
                row.update(input_tokens=len(record.input_ids), prefix_tokens=len(cache.prefix_ids),
                    suffix_tokens=len(record.input_ids)-len(cache.prefix_ids), cache_bytes=cache.nbytes,
                    uncached=values(full, record), cached=values(cached, record),
                    uncached_timing=full_time, cached_timing=cached_time,
                    ram_to_npu_timing={'median_s': statistics.median(transfers), 'samples_s': transfers},
                    cache_unchanged=True)
                row['score_abs_diff'] = abs(row['cached']['score']-row['uncached']['score'])
                row['logit_abs_diff'] = float((cached.float()-full.float()).abs().max())
                row['resident_speedup'] = full_time['median_s']/cached_time['median_s']
                row['ram_speedup'] = full_time['median_s']/(statistics.median(transfers)+cached_time['median_s'])
                prior = previous.get((pair['task'], pair['qid'], pair['did']))
                if prior and target_dtype == torch.bfloat16:
                    row['uncached_matches_prior'] = row['uncached']['logits'] == prior['modes']['whole']['logits']
                    if not row['uncached_matches_prior']:
                        raise AssertionError('Default uncached arithmetic changed')
                result['rows'].append(row)
                print(json.dumps({'completed': index+1, 'of': len(records), **row}), flush=True)
                save(args.output, result)
                del ids, cache, returned, full, cached
            # A/B/A reuse against one unchanged cache, plus incompatible-prefix
            # and model-identity rejection, followed by another valid request.
            cache = ram[0].to('npu:0')
            record_a = records[0]
            request_b = request_for({**fixture['pairs'][-1], 'document': fixture['pairs'][0]['document']})
            record_b = encode_record(tokenizer, request_b, args.max_length)
            ids_a = torch.tensor([record_a.input_ids], dtype=torch.long, device='npu:0')
            ids_b = torch.tensor([record_b.input_ids], dtype=torch.long, device='npu:0')
            before = model(ids_a, record_a, cache)[0]
            b_cached = model(ids_b, record_b, cache)[0]
            b_full = model(ids_b, record_b)[0]
            after = model(ids_a, record_a, cache)[0]
            assert torch.equal(before, after)
            rejected = []
            for label, bad_cache in [('prefix', replace(cache, prefix_ids=(cache.prefix_ids[0]+1, *cache.prefix_ids[1:]))),
                                     ('model_identity', replace(cache, model_identity='incompatible'))]:
                try:
                    model(ids_a, record_a, bad_cache)
                except ValueError:
                    rejected.append(label)
                else:
                    raise AssertionError('Invalid cache accepted')
            assert torch.equal(after, model(ids_a, record_a, cache)[0])
            after_cpu = cache.to('cpu')
            assert all(torch.equal(t, after_cpu.tensors()[k]) for k,t in ram[0].tensors().items())
            result['reuse_check'] = {'a_repeat_exact': True, 'cache_unchanged': True, 'rejected': rejected,
                'b_uncached': values(b_full, record_b), 'b_cached': values(b_cached, record_b)}
        rankings = {}
        for name, group in fixture['groups'].items():
            task = group.get('task', name)
            rows = [r for r in result['rows'] if (r['task'],r['qid']) == (task,group['qid'])]
            orders = {mode: [r['did'] for r in sorted(rows,key=lambda r:(-r[mode]['score'],r['did']))]
                      for mode in ('uncached','cached')}
            rankings[name] = {'orders': orders, 'unchanged': orders['uncached']==orders['cached']}
        result['rankings'] = rankings
        result['summary'] = {'score_abs_diff_max': max(r['score_abs_diff'] for r in result['rows']),
            'logit_abs_diff_max': max(r['logit_abs_diff'] for r in result['rows']),
            'rankings_unchanged': sum(v['unchanged'] for v in rankings.values()), 'queries': len(rankings),
            'uncached_mean_s': statistics.mean(r['uncached_timing']['median_s'] for r in result['rows']),
            'resident_cached_mean_s': statistics.mean(r['cached_timing']['median_s'] for r in result['rows']),
            'ram_to_npu_mean_s': statistics.mean(r['ram_to_npu_timing']['median_s'] for r in result['rows'])}
        result['transformers_imported'] = any(n=='transformers' or n.startswith('transformers.') for n in sys.modules)
        assert not result['transformers_imported']
        result['status'] = 'completed'
        print(json.dumps(result['summary']), flush=True)
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
    lengths = commands.add_parser('prepare-lengths')
    for flag in ('english', 'chinese', 'ecom', 't2', 'source-repo', 'model', 'profile-dir', 'output'):
        lengths.add_argument('--' + flag, type=Path, required=True)
    task = commands.add_parser('prepare-task')
    task.add_argument('--task', required=True)
    for flag in ('english', 'model', 'output'):
        task.add_argument('--' + flag, type=Path, required=True)
    infer = commands.add_parser('run')
    infer.add_argument('--fixture', type=Path, required=True)
    infer.add_argument('--model', type=Path, required=True)
    infer.add_argument('--output', type=Path, required=True)
    infer.add_argument('--max-length', type=int, default=3072)
    cached = commands.add_parser('cache')
    for flag in ('fixture', 'model', 'output', 'cache-dir'):
        cached.add_argument('--' + flag, type=Path, required=True)
    cached.add_argument('--reference', type=Path)
    cached.add_argument('--dtype', choices=('bfloat16', 'float32'), default='bfloat16')
    cached.add_argument('--max-length', type=int, default=3072)
    cached.add_argument('--repeats', type=int, default=3)
    precache = commands.add_parser('precache-task')
    for flag in ('fixture', 'model', 'cache-dir', 'output'):
        precache.add_argument('--' + flag, type=Path, required=True)
    precache.add_argument('--dtype', choices=('bfloat16', 'float32'), default='float32')
    precache.add_argument('--max-prefix-length', type=int, required=True)
    metrics = commands.add_parser('evaluate')
    metrics.add_argument('--fixture', type=Path, required=True)
    metrics.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    {'prepare': prepare, 'prepare-lengths': prepare_lengths, 'prepare-task': prepare_task, 'run': run, 'cache': run_cache, 'precache-task': precache_task, 'evaluate': evaluate}[args.command](args)
