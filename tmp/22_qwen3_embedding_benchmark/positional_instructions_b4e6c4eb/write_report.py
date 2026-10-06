"""Render the positional-instruction result tables from the recorded output."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--result', type=Path, required=True)
parser.add_argument('--summary', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
p = json.loads(args.result.read_text())
s = json.loads(args.summary.read_text())
assert p['status'] == 'completed'
heldout = p['partition_summaries']['holdout']['Touche2020Retrieval.v3']
cal = p['partition_summaries']['calibration']
names = ('normal', 'document_first', 'swapped_contents', 'query_first_wrong_labels', 'document_then_instruction')
labels = ('Normal query-first', 'Document-first, correct labels', 'Document-first, swapped contents', 'Query-first, swapped labels', 'Document, instruction, query')
example = next(r for r in p['pairs'] if r['id'] == 'synthetic-0-1')
example_variant = example['variants']['swapped_contents_position_both']
text = ['# Explicit first/second-field instruction results', '',
    'Verified on Ascend 910B2, 2026-10-06. This is a follow-up to [the prompt-role investigation](PROMPT_ROLE_INVESTIGATION.md).', '',
    'The original relevance task is retained. The new explanation identifies document/query roles by field position, regardless of labels. Its order is adapted to the actual content order.', '',
    'For document-first Touché inputs, the task instruction is:', '', '```text',
    next(r for r in p['pairs'] if r['source']=='Touche2020Retrieval.v3')['variants']['swapped_contents_position_both']['body'].split('\n')[0], '```', '',
    'The new system instruction is:', '', '```text', example_variant['prefix'].split('system\n',1)[1].split('<|im_end|>',1)[0], '```', '',
    'For query-first inputs, the explanation says the first input field contains the query and the second contains the document. The task instruction is explicitly excluded from the count of input fields, including the layout where it occurs between the document and query.', '',
    'Each layout is tested unchanged, with the explanation in the task only, in the system only, and in both. Ordinary body tokenization is used for the broad comparison. The initial short examples additionally repeat every layout with separately tokenized labels, contents, and newlines.', '',
    '## Results on the same 46-query Touché sample', '',
    'Every cell is correct comparisons out of 478. A tie is not counted as correct.', '',
    '| Layout | Unchanged | Task explanation | System explanation | Both |',
    '|---|---:|---:|---:|---:|']
for name, label in zip(names, labels):
    values = [heldout['orderings'][name+suffix]['correct'] for suffix in ('','_position_task','_position_system','_position_both')]
    text.append('| ' + label + ' | ' + ' | '.join(str(v) for v in values) + ' |')
text += ['', 'The position explanation in both places improves correctly labeled document-first inputs from 237 to 246 correct comparisons, a gain of 9/478 (1.9 percentage points). For swapped contents, the strongest tested placement is system-only: 270 to 271. None of the new document-first variants matches the unchanged query-first baseline of 328/478; the strongest document-first result remains the unchanged document/instruction/query layout at 276/478. These results describe this wording and sample, not every possible instruction.', '',
    'The same 257 explicitly judged documents from the other 46 Touché queries were selected with the unchanged shortest/median-per-grade rule and a 400-document-token eligibility limit. No text was truncated. This is graded pair-ordering accuracy on a selected short-document sample, not full top-100 NDCG@10.', '',
    '## Ties', '', '| Layout | Unchanged | Task explanation | System explanation | Both |', '|---|---:|---:|---:|---:|']
for name,label in zip(names,labels):
    values = [heldout['orderings'][name+suffix]['ties'] for suffix in ('','_position_task','_position_system','_position_both')]
    text.append('| ' + label + ' | ' + ' | '.join(str(v) for v in values) + ' |')
text += ['', '## Synthetic sanity tests', '',
    'The six original synthetic pairs are retained. Each positive-versus-unrelated comparison is tested with every layout and instruction placement. Scores below show the three relevant answers with the explanation in both the system and task.', '',
    '| Answer | Normal unchanged | Document-first, correct labels | Swapped contents |', '|---|---:|---:|---:|']
for row in p['pairs']:
    if row['source']=='synthetic_sanity' and row['grade']==1:
        text.append('| '+row['document'].replace('|','\\|')+' | '+' | '.join(f"{row['variants'][name]['score']:.3f}" for name in ('normal','document_first_position_both','swapped_contents_position_both'))+' |')
text += ['', 'All 20 ordinary variants still rank the relevant answer above the unrelated one in all three synthetic comparisons. All synthetic ordering counts, calibration results, atomic controls, FP32-head comparisons, and grade-specific margins are in summary.json.', '',
    '## Validation and limits', '',
    f"The maximum absolute score change from previous unmodified baselines is {s['audit']['maximum_baseline_score_change_from_prior']}. Input IDs for matching old layouts are also identical. Original control-token sequences and the assistant suffix were preserved; task-only changes preserve the full original prefix. System-only and combined changes deliberately modify system prose while retaining its chat markers.", '',
    'Twenty-one short initial pairs are evaluated separately: 15 Touché documents across the initial three queries and six synthetic pairs. The single initial long document is excluded by the same 400-token selection limit, without truncation. Atomic token inventories match within groups sharing identical system/task wording; instructions that assign different first/second roles naturally have different wording and are not asserted to have identical inventories.', '',
    f"Maximum complete input length: {s['audit']['max_input_tokens']} tokens. The backbone remains BF16; only the shadow two-row output projection is recomputed in FP32. These are accuracy runs on shared physical NPUs 2 and 3, not latency measurements.", '',
    'The first attempt hit a shared-device memory limit after saving 115 completed rows. The continuation enabled expandable_segments in the NPU allocator, recomputed an overlap example with all 20 variants, and required an exact score match before reusing rows. It preserved the document selection, prompts, precision, attention implementation, and score formula. The allocator setting is documented by [Ascend](https://github.com/Ascend/pytorch/blob/master/docs/zh/api/environment_variable/memory_management/PYTORCH_NPU_ALLOC_CONF.md).', '',
    'Code: investigate_prompt_roles.py --positional-instructions. Source commit: '+p['commit']+'.', '',
    'Evidence: [summary](../tmp/22_qwen3_embedding_benchmark/positional_instructions_b4e6c4eb/summary.json), [exact command](../tmp/22_qwen3_embedding_benchmark/positional_instructions_b4e6c4eb/command.txt), [example prompts](../tmp/22_qwen3_embedding_benchmark/positional_instructions_b4e6c4eb/example_prompts.txt). Full raw token/logit output is saved as result.json.gz with a SHA-256 manifest. analyze.py and write_report.py reproduce the derived files.', '']
args.output.write_text('\n'.join(text))
prompt_lines = []
for name in names:
    for suffix in ('', '_position_task','_position_system','_position_both'):
        value = example['variants'][name+suffix]
        prompt_lines += [name+suffix, value['prefix']+value['body']+p['suffix'], '']
prompt_lines.append('End of prompt examples.')
(args.result.parent/'example_prompts.txt').write_text('\n'.join(prompt_lines) + '\n')
