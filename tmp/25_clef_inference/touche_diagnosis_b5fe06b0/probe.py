"""Small outcome-selected diagnosis; never a benchmark accuracy estimate."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / '25_clef_inference'))
from run_local_smoke import NoTransformers, encode_document_prefix, encode_record
from run_reranking_smoke import cache_cast, request_for, save


def requests(pair):
    original = request_for(pair)
    criteria = {
        'true': 'The document is relevant to the search query, including arguments either supporting or opposing it.',
        'false': 'The document is unrelated to the search query or provides no useful argument about it.',
    }
    explicit = json.loads(json.dumps(original))
    explicit['questions']['relevance']['criteria'] = criteria
    neutral = json.loads(json.dumps(explicit))
    neutral['questions']['relevance']['instructions'] = (
        'Assess document relevance, not agreement. Search query: ' + pair['query'] +
        '\nDoes STATE contain an argument addressing this topic? Arguments for and against are equally relevant. '
        'Do not answer the search query itself. Decide whether the document discusses it.')
    graded = json.loads(json.dumps(neutral))
    graded['questions']['relevance'] = {
        'type': 'score', 'instructions': neutral['questions']['relevance']['instructions'],
        'criteria': ['Unrelated to the topic or contains no argument answering the search query.',
                     'Partly relevant: mentions the topic but supplies limited useful reasoning.',
                     'Directly relevant: gives useful arguments answering the search query, for or against.'],
    }
    stance = json.loads(json.dumps(original))
    stance['questions']['relevance']['instructions'] = pair['query']
    return {'baseline': original, 'explicit_criteria': explicit,
            'stance_neutral': neutral, 'graded_relevance': graded, 'direct_question': stance}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--benchmark-result', type=Path, required=True)
    p.add_argument('--cache-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
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
    torch.set_num_threads(8)
    if not torch.npu.is_available() or '910B' not in torch.npu.get_device_name(0):
        raise RuntimeError('910B required')
    torch.npu.set_device(0)
    model = load_model(args.model, 'npu:0', progress=lambda x: print(x, flush=True)).float()
    torch.npu.synchronize()
    tokenizer = Tokenizer.from_file(str(args.model / 'tokenizer.json'))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    spec = importlib.util.spec_from_file_location('pinned_official_clef', args.model / 'joint_schema_model.py')
    official = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = official
    spec.loader.exec_module(official)

    class Adapter:
        def __call__(self, text, add_special_tokens=False):
            return SimpleNamespace(input_ids=tokenizer.encode(text, add_special_tokens=add_special_tokens).ids)

    f = json.loads(args.fixture.read_text())
    reference = json.loads(args.benchmark_result.read_text())
    selection = {
        '1': [('supports', '51530f3f-2019-04-18T18:15:02Z-00004-000'),
              ('opposes', '51530f3f-2019-04-18T18:15:02Z-00005-000')],
        '9': [('supports', 'a7f5e454-2019-04-18T13:43:30Z-00003-000'),
              ('opposes', 'a7f5e454-2019-04-18T13:43:30Z-00002-000')],
        '33': [('supports', '8eeab760-2019-04-18T16:21:32Z-00007-000'),
               ('opposes', '7eabc63c-2019-04-18T12:18:12Z-00003-000')],
    }
    out = {'scope': 'outcome-selected diagnosis: three queries, judged-relevant opposing stances plus synthetic off-topic controls',
           'physical_device': os.environ.get('ASCEND_RT_VISIBLE_DEVICES'), 'model_identity': model.cache_identity,
           'compute_dtype': 'float32', 'storage_dtype': 'bfloat16',
           'head_scales': {name: getattr(model.head, name).float().item()
                           for name in ['prior_logit_scale', 'joint_logit_scale', 'residual_gate']}, 'rows': []}

    def run(request, cache=None):
        rec = encode_record(tokenizer, request)
        upstream = official.encode_record(Adapter(), request)
        fields = ['question_id', 'question_type', 'question_span', 'option_spans', 'option_ids']
        if rec.input_ids != upstream.input_ids or any(
                getattr(a, name) != getattr(b, name)
                for a, b in zip(rec.questions, upstream.questions) for name in fields):
            raise ValueError('Official encoding differs')
        ids = torch.tensor([rec.input_ids], device='npu:0', dtype=torch.long)
        began = time.perf_counter()
        logits = model(ids, rec, cache=cache)[0].float()
        probabilities = logits.softmax(-1).cpu().tolist()
        return {'logits': logits.cpu().tolist(), 'probabilities': dict(zip(rec.questions[0].option_ids, probabilities)),
                'expected_score': sum(i * v for i, v in enumerate(probabilities)) if len(probabilities) == 3 else None,
                'input_tokens': len(rec.input_ids), 'wall_s': time.perf_counter() - began, 'official_encoding_equal': True}

    with torch.inference_mode():
        for qid, selected in selection.items():
            selected = selected + [('unrelated', None)]
            for stance, did in selected:
                doc = f['documents'][did]['text'] if did else (
                    'To bake a plain loaf of bread, mix flour, water, yeast and salt. Knead the dough, '
                    'let it rise until doubled in size, then bake it in a hot oven until the crust is golden.')
                pair = {'task': f['task'], 'qid': qid, 'did': did or 'synthetic',
                        'query': f['queries'][qid]['text'], 'document': doc, 'instruction': f['instruction']}
                prefix = encode_document_prefix(tokenizer, doc)
                row = {'qid': qid, 'did': did, 'stance': stance, 'query': pair['query'],
                       'qrel': f['queries'][qid]['qrels'][did] if did else None, 'prefix_tokens': len(prefix)}
                fresh = model.prepare_document(prefix)
                if did:
                    key = hashlib.sha256(json.dumps([model.cache_identity, 'torch.bfloat16', prefix]).encode()).hexdigest()
                    stored = cache_cast(DocumentCache.load(args.cache_dir / (key + '.safetensors')).to('npu:0'), torch.float32)
                    original = request_for(pair)
                    row['saved_benchmark_score'] = reference['predictions'][qid][did]
                    row['uncached'] = run(original)
                    row['fresh_fp32_cache'] = run(original, fresh)
                    row['fresh_bf16_cache'] = run(original, cache_cast(cache_cast(fresh, torch.bfloat16), torch.float32))
                    del fresh
                else:
                    stored = cache_cast(cache_cast(fresh, torch.bfloat16), torch.float32)
                    del fresh
                row['variants'] = {name: run(request, stored) for name, request in requests(pair).items()}
                del stored
                out['rows'].append(row)
                save(args.output, out)
                print(json.dumps({'qid': qid, 'stance': stance, 'scores': {
                    name: v['probabilities'].get('true', v['expected_score']) for name, v in row['variants'].items()}}), flush=True)
    out['transformers_imported'] = any(n == 'transformers' or n.startswith('transformers.') for n in sys.modules)
    out['status'] = 'completed'
    save(args.output, out)


if __name__ == '__main__':
    main()
