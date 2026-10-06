"""Controlled Qwen reranker field-role investigation, without training.

Calibration uses the saved 22-pair probe. Holdout uses explicitly judged
documents from every remaining Touché query, with a fixed length/sample rule.
Scores are binary-normalized yes/no probabilities, not full-vocabulary ones.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess

from reranker_protocol import PREFIX, SUFFIX


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def variants(pair, holdout=False):
    task, question, passage = pair['instruction'], pair['query'], pair['document']
    reverse_task = ('Given an argument, retrieve a question that the argument addresses with detailed and persuasive reasoning'
                    if pair['source'] == 'Touche2020Retrieval.v3' else
                    'Given a passage, retrieve a question that the passage answers')
    # The complete 2 x 2 design independently changes content order and label assignment.
    result = {
        'normal': (PREFIX, task, [('Query', question), ('Document', passage)], False),
        'document_first': (PREFIX, task, [('Document', passage), ('Query', question)], False),
        'swapped_contents': (PREFIX, task, [('Query', passage), ('Document', question)], False),
        'query_first_wrong_labels': (PREFIX, task, [('Document', question), ('Query', passage)], False),
    }
    if not holdout:
        for name, spec in list(result.items()):
            result[name + '_atomic'] = (*spec[:3], True)
    # A role clarification keeps the original task text and appends a controlled sentence.
    clarify = task + '. The question is in <Query>; the passage is in <Document>.'
    remap = task + '. The question is in <Document>; the passage is in <Query>.'
    result.update({
        'normal_clarified': (PREFIX, clarify, [('Query', question), ('Document', passage)], False),
        'document_first_clarified': (PREFIX, clarify, [('Document', passage), ('Query', question)], False),
        'swapped_contents_remapped': (PREFIX, remap, [('Query', passage), ('Document', question)], False),
        'swapped_contents_reverse_task': (PREFIX, reverse_task, [('Query', passage), ('Document', question)], False),
        'normal_general_task': (PREFIX, 'Given a web search query, retrieve relevant passages that answer the query',
                                [('Query', question), ('Document', passage)], False),
        'swapped_contents_general_task': (PREFIX, 'Given a web search query, retrieve relevant passages that answer the query',
                                          [('Query', passage), ('Document', question)], False),
        'swapped_contents_system_remapped': (
            PREFIX.replace('the Document', 'the __PASSAGE__').replace('the Query', 'the Document').replace('the __PASSAGE__', 'the Query'),
            task, [('Query', passage), ('Document', question)], False),
    })
    return result


def ordering(rows, names, key='score'):
    result = {}
    for name in names:
        comparisons = [a['variants'][name][key] - b['variants'][name][key]
                       for a in rows for b in rows
                       if a['query_id'] == b['query_id'] and a['grade'] > b['grade']]
        result[name] = dict(correct=sum(d > 0 for d in comparisons), ties=sum(d == 0 for d in comparisons),
                            total=len(comparisons))
    return result


def summarize(rows):
    names = list(rows[0]['variants'])
    result = dict(pairs=len(rows), queries=len({r['query_id'] for r in rows}))
    for source in sorted({r['source'] for r in rows}):
        subset = [r for r in rows if r['source'] == source]
        metrics = {}
        for name in names:
            shifts = [r['variants'][name]['margin'] - r['variants']['normal']['margin'] for r in subset]
            metrics[name] = dict(mean_margin_shift=statistics.mean(shifts),
                                mean_absolute_score_change=statistics.mean(abs(r['variants'][name]['score'] - r['variants']['normal']['score']) for r in subset),
                                mean_margin_by_grade={str(g): statistics.mean(r['variants'][name]['margin'] for r in subset if r['grade'] == g)
                                                      for g in sorted({r['grade'] for r in subset})})
        result[source] = dict(pairs=len(subset), queries=len({r['query_id'] for r in subset}),
                              orderings=ordering(subset, names), fp32_head_orderings=ordering(subset, names, 'fp32_head_margin'), metrics=metrics)
    return result


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--prior-result', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=['calibration', 'holdout'], required=True)
    parser.add_argument('--max-document-tokens', type=int, default=900,
                        help='Holdout selection limit only; documents are never truncated')
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
    prior = json.loads(args.prior_result.read_text())
    if args.phase == 'calibration':
        pairs = [{k: v for k, v in r.items() if k != 'variants'} for r in prior['pairs']]
    else:
        fixture = json.loads(args.fixture.read_text())
        excluded = {r['query_id'] for r in prior['pairs']}
        pairs = []
        for qid in sorted(fixture['queries'], key=int):
            if qid in excluded:
                continue
            query = fixture['queries'][qid]
            for grade in (0, 1, 2):
                eligible = sorted((d for d in query['candidates'] if query['qrels'].get(d) == grade
                                   and len(encode(fixture['documents'][d]['text'])) <= args.max_document_tokens),
                                  key=lambda d: (len(encode(fixture['documents'][d]['text'])), d))
                for index in sorted({0, len(eligible) // 2}) if eligible else []:
                    did = eligible[index]
                    pairs.append(dict(id=f'{qid}/{did}', query_id=qid, document_id=did,
                                      query=query['text'], document=fixture['documents'][did]['text'],
                                      grade=grade, instruction=fixture['instruction'], source='Touche2020Retrieval.v3'))
    assert pairs
    yes, no = tokenizer.convert_tokens_to_ids('yes'), tokenizer.convert_tokens_to_ids('no')
    assert encode('yes') == [yes] and encode('no') == [no]
    # Include explicitly written control tokens even if all_special_ids omits them.
    controls = {text: encode(text) for text in ('<|im_start|>', '<|im_end|>', '<think>', '</think>')}
    assert all(len(ids) == 1 for ids in controls.values())
    control_ids = {ids[0] for ids in controls.values()}
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                attn_implementation='sdpa', local_files_only=True).eval().to('npu:0')
    head_weights = model.lm_head.weight[[no, yes]].detach().float()
    head_bias = None if model.lm_head.bias is None else model.lm_head.bias[[no, yes]].detach().float()
    head_input = {}
    def capture_head(module, inputs):
        head_input['last'] = inputs[0][:, -1, :].detach().float()
    handle = model.lm_head.register_forward_pre_hook(capture_head)
    result = dict(status='running', phase=args.phase, scope='controlled selected-pair diagnostic, not full benchmark',
                  holdout_max_document_tokens=args.max_document_tokens,
                  commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  source_sha256=sha(__file__), prior_sha256=sha(args.prior_result), fixture_sha256=sha(args.fixture),
                  tokenizer_sha256=sha(args.model / 'tokenizer.json'),
                  environment=dict(host=platform.node(), chip='Ascend 910B2', physical_npu=os.getenv('ASCEND_RT_VISIBLE_DEVICES'),
                                   torch=torch.__version__, torch_npu=torch_npu.__version__, transformers=transformers.__version__,
                                   dtype='bfloat16', attention='sdpa', use_cache=False, batch_size=1),
                  controls=controls, suffix=SUFFIX, suffix_ids=encode(SUFFIX),
                  normal_prefix=PREFIX, normal_prefix_ids=encode(PREFIX), answer_token_ids=dict(no=no, yes=yes), pairs=[])
    with torch.inference_mode():
        for pair in pairs:
            if args.phase == 'holdout':
                torch.npu.empty_cache()
            row = {**pair, 'variants': {}}
            atomic_bags = []
            for name, (prefix, task, fields, atomic) in variants(pair, args.phase == 'holdout').items():
                body = '<Instruct>: ' + task + '\n' + '\n'.join(f'<{label}>: {content}' for label, content in fields)
                if atomic:
                    ids = encode(prefix) + encode('<Instruct>:') + encode(' ' + task) + encode('\n')
                    for i, (label, content) in enumerate(fields):
                        if i:
                            ids += encode('\n')
                        ids += encode(f'<{label}>:') + encode(' ' + content)
                    ids += encode(SUFFIX)
                    atomic_bags.append(Counter(ids))
                else:
                    ids = encode(prefix) + encode(body) + encode(SUFFIX)
                assert len(ids) <= 8192, 'No truncation permitted'
                assert tokenizer.decode(ids, skip_special_tokens=False) == prefix + body + SUFFIX
                assert [i for i in ids if i in control_ids] == [i for i in row['variants']['normal']['input_ids'] if i in control_ids] if row['variants'] else True
                tensor = torch.tensor([ids], dtype=torch.long, device='npu:0')
                full_logits = model(input_ids=tensor, attention_mask=torch.ones_like(tensor), use_cache=False,
                                    logits_to_keep=1).logits[0, -1].float()
                binary_logits = full_logits[[no, yes]]
                fp32_head_logits = torch.nn.functional.linear(head_input.pop('last'), head_weights, head_bias)[0]
                nl, yl = binary_logits.cpu().tolist()
                fn, fy = fp32_head_logits.cpu().tolist()
                top = full_logits.topk(5)
                top_ids = top.indices.cpu().tolist()
                probabilities = full_logits.softmax(-1)[[no, yes]].cpu().tolist()
                row['variants'][name] = dict(prefix=prefix, body=body, input_ids=ids, tokens=len(ids),
                    no_logit=nl, yes_logit=yl, margin=yl-nl, score=binary_logits.softmax(-1)[1].item(),
                    fp32_head_no_logit=fn, fp32_head_yes_logit=fy, fp32_head_margin=fy-fn,
                    fp32_head_score=fp32_head_logits.softmax(-1)[1].item(),
                    full_vocabulary_no_probability=probabilities[0], full_vocabulary_yes_probability=probabilities[1],
                    top_tokens=[dict(id=i, text=tokenizer.decode([i]), logit=value) for i, value in zip(top_ids, top.values.cpu().tolist())])
            assert all(bag == atomic_bags[0] for bag in atomic_bags) if atomic_bags else True
            baseline_bag = Counter(row['variants']['normal']['input_ids'])
            for value in row['variants'].values():
                bag = Counter(value['input_ids'])
                value['token_inventory_changes'] = [dict(id=i, text=tokenizer.decode([i]),
                                                        normal_count=baseline_bag[i], variant_count=bag[i])
                    for i in sorted(baseline_bag.keys() | bag.keys()) if baseline_bag[i] != bag[i]]
            if args.phase == 'calibration':
                previous = next(r for r in prior['pairs'] if r['id'] == pair['id'])
                assert row['variants']['normal']['input_ids'] == previous['variants']['query_first']['input_ids']
                row['normal_score_change_from_prior'] = row['variants']['normal']['score'] - previous['variants']['query_first']['score']
            result['pairs'].append(row)
            if len(result['pairs']) % 5 == 0:
                save(args.output, result)
            print(json.dumps(dict(pair=pair['id'], done=len(result['pairs']), total=len(pairs),
                                  scores={k: round(v['score'], 6) for k, v in row['variants'].items()})), flush=True)
    handle.remove()
    result['summary'] = summarize(result['pairs'])
    result['summary']['all_control_token_sequences_preserved'] = True
    result['summary']['atomic_token_multisets_identical'] = args.phase == 'calibration'
    result['status'] = 'completed'
    save(args.output, result)
    print(json.dumps(result['summary'], indent=2), flush=True)


if __name__ == '__main__':
    main()
