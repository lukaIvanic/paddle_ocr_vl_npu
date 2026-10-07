"""Small, untruncated prompt-order probe; no training or cache implementation."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

from reranker_protocol import PREFIX, SUFFIX


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    import torch_npu
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(8)
    torch.npu.set_device('npu:0')
    torch.npu.set_compile_mode(jit_compile=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    prefix, suffix = encode(PREFIX), encode(SUFFIX)
    fixture = json.loads(args.fixture.read_text())
    pairs = []
    # Fixed query IDs; two explicit judgments per grade, shortest then median.
    # Unjudged candidates are never silently called negative in this diagnostic.
    for qid in ('1', '9', '33'):
        query = fixture['queries'][qid]
        for grade in (0, 1, 2):
            eligible = sorted((d for d in query['candidates']
                               if query['qrels'].get(d) == grade
                               and len(encode(fixture['documents'][d]['text'])) <= 1500),
                              key=lambda d: (len(encode(fixture['documents'][d]['text'])), d))
            for index in sorted({0, len(eligible) // 2}) if eligible else []:
                did = eligible[index]
                pairs.append(dict(id=f'{qid}/{did}', query_id=qid, document_id=did,
                                  query=query['text'], document=fixture['documents'][did]['text'],
                                  grade=grade, source='Touche2020Retrieval.v3',
                                  instruction=fixture['instruction'], saved_qwen_score=query['qwen_scores'][did]))
    for index, (query, positive, negative) in enumerate([
        ('What is the capital of France?', 'Paris is the capital of France.', 'Bananas are a tropical fruit.'),
        ('How do I reset a forgotten password?', 'Click Forgot password, enter your email, and follow the reset link.', 'The store closes at six every evening.'),
        ('Why do leaves look green?', 'Chlorophyll absorbs red and blue light and reflects green light.', 'A bicycle usually has two wheels.'),
    ]):
        for grade, document in [(1, positive), (0, negative)]:
            pairs.append(dict(id=f'synthetic-{index}-{grade}', query_id=f'synthetic-{index}',
                              query=query, document=document, grade=grade, source='synthetic_sanity',
                              instruction='Given a web search query, retrieve relevant passages that answer the query'))

    result = dict(status='running', scope='selected-pair diagnostic, not a benchmark accuracy estimate',
                  environment=dict(host=platform.node(), chip='Ascend 910B2', physical_npu=os.getenv('ASCEND_RT_VISIBLE_DEVICES'),
                                   torch=torch.__version__, torch_npu=torch_npu.__version__, transformers=transformers.__version__,
                                   dtype='bfloat16', attention='sdpa', batch_size=1, use_cache=False),
                  commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  fixture_sha256=digest(args.fixture), source_sha256=digest(__file__),
                  tokenizer_sha256=digest(args.model / 'tokenizer.json'),
                  prefix=PREFIX, suffix=SUFFIX, prefix_ids=prefix, suffix_ids=suffix,
                  bos_token_id=tokenizer.bos_token_id, special_tokens_map=tokenizer.special_tokens_map,
                  add_special_tokens=False, truncation=False, pairs=[])
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                attn_implementation='sdpa', local_files_only=True).eval().to('npu:0')
    yes, no = tokenizer.convert_tokens_to_ids('yes'), tokenizer.convert_tokens_to_ids('no')
    assert encode('yes') == [yes] and encode('no') == [no]
    result['answer_token_ids'] = dict(yes=yes, no=no)
    with torch.inference_mode():
        for pair in pairs:
            instruction = f'<Instruct>: {pair["instruction"]}'
            query = f'<Query>: {pair["query"]}'
            document = f'<Document>: {pair["document"]}'
            bodies = dict(query_first='\n'.join([instruction, query, document]),
                          swapped_contents='\n'.join([instruction, f'<Query>: {pair["document"]}', f'<Document>: {pair["query"]}']),
                          document_first='\n'.join([instruction, document, query]),
                          document_then_instruction='\n'.join([document, instruction, query]),
                          query_first_split_boundaries='\n'.join([instruction, query, document]),
                          document_first_split_boundaries='\n'.join([instruction, document, query]))
            boundary = encode('\n')
            split_ids = dict(query_first_split_boundaries=prefix + encode(instruction) + boundary + encode(query) + boundary + encode(document) + suffix,
                             document_first_split_boundaries=prefix + encode(instruction) + boundary + encode(document) + boundary + encode(query) + suffix)
            assert Counter(split_ids['query_first_split_boundaries']) == Counter(split_ids['document_first_split_boundaries'])
            row = {**pair, 'variants': {}}
            baseline_ids = None
            for variant, body in bodies.items():
                ids = split_ids.get(variant, prefix + encode(body) + suffix)
                assert len(ids) <= 8192, 'Probe must not truncate any input'
                assert tokenizer.decode(ids, skip_special_tokens=False) == PREFIX + body + SUFFIX
                if baseline_ids is None:
                    baseline_ids = ids
                tensor = torch.tensor([ids], dtype=torch.long, device='npu:0')
                start = time.monotonic()
                logits = model(input_ids=tensor, attention_mask=torch.ones_like(tensor),
                               use_cache=False, logits_to_keep=1).logits[0, -1, [no, yes]].float()
                values = logits.cpu().tolist()
                score = logits.softmax(-1)[1].item()
                row['variants'][variant] = dict(score=score, no_logit=values[0], yes_logit=values[1],
                    seconds=time.monotonic()-start, tokens=len(ids), input_ids=ids,
                    token_multiset_matches_baseline=Counter(ids) == Counter(baseline_ids),
                    special_token_sequence=[i for i in ids if i in tokenizer.all_special_ids],
                    body=body)
            result['pairs'].append(row)
            save(args.output, result)
            print(json.dumps(dict(id=pair['id'], grade=pair['grade'],
                                  scores={v: x['score'] for v,x in row['variants'].items()})), flush=True)
    variants = list(bodies)
    groups = {}
    for row in result['pairs']:
        groups.setdefault(row['query_id'], []).append(row)
    comparisons = {}
    for variant in variants:
        total = wins = ties = 0
        for rows in groups.values():
            for positive in rows:
                for negative in rows:
                    if positive['grade'] > negative['grade']:
                        delta = positive['variants'][variant]['score'] - negative['variants'][variant]['score']
                        total += 1
                        wins += delta > 0
                        ties += delta == 0
        comparisons[variant] = dict(correct=wins, total=total, ties=ties)
    result['summary'] = dict(pairs=len(pairs), graded_pair_orderings=comparisons,
        baseline_max_abs_difference_from_saved=max(abs(r['variants']['query_first']['score']-r['saved_qwen_score'])
                  for r in result['pairs'] if 'saved_qwen_score' in r),
        split_boundary_controls_have_identical_token_multisets=all(
            Counter(r['variants']['query_first_split_boundaries']['input_ids']) == Counter(r['variants']['document_first_split_boundaries']['input_ids'])
            for r in result['pairs']),
        all_token_multisets_identical=all(v['token_multiset_matches_baseline'] for r in result['pairs'] for v in r['variants'].values()),
        all_special_token_sequences_identical=all(v['special_token_sequence'] == r['variants']['query_first']['special_token_sequence']
                                                 for r in result['pairs'] for v in r['variants'].values()))
    result['status'] = 'completed'
    save(args.output, result)
    print(json.dumps(result['summary'], indent=2), flush=True)


if __name__ == '__main__':
    main()
