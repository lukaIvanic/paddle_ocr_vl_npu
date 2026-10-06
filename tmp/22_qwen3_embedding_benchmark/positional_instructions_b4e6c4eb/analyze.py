"""Validate and summarize the positional-instruction rerun."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import statistics

parser = argparse.ArgumentParser()
parser.add_argument('--result', type=Path, required=True)
parser.add_argument('--previous-holdout', type=Path, required=True)
parser.add_argument('--previous-calibration', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
p = json.loads(args.result.read_text())
assert p['status'] == 'completed'
previous_holdout = {r['id']: r for r in json.loads(args.previous_holdout.read_text())['pairs']}
previous_calibration = {r['id']: r for r in json.loads(args.previous_calibration.read_text())['pairs']}
heldout = [r for r in p['pairs'] if r['partition'] == 'holdout']
calibration = [r for r in p['pairs'] if r['partition'] == 'calibration']
assert len(heldout) == 257 and len(calibration) == 21
assert {r['id'] for r in heldout} == set(previous_holdout)
core = ('normal', 'document_first', 'swapped_contents', 'query_first_wrong_labels', 'document_then_instruction')
controls = {ids[0] for ids in p['controls'].values()}
head_errors = []
reproducibility_errors = []
for row in p['pairs']:
    canonical = row['variants']['normal']['input_ids']
    for name, value in row['variants'].items():
        assert [i for i in value['input_ids'] if i in controls] == [i for i in canonical if i in controls]
        assert value['input_ids'][-len(p['suffix_ids']):] == p['suffix_ids']
        if name in core or '_position_task' in name:
            assert value['input_ids'][:len(p['normal_prefix_ids'])] == p['normal_prefix_ids']
        head_errors.append(abs(value['margin'] - value['fp32_head_margin']))
    previous = (previous_holdout if row['partition'] == 'holdout' else previous_calibration)[row['id']]
    for name in core:
        if name in previous['variants']:
            assert row['variants'][name]['input_ids'] == previous['variants'][name]['input_ids']
            reproducibility_errors.append(abs(row['variants'][name]['score'] - previous['variants'][name]['score']))
summary = {k: p[k] for k in ('status', 'commit', 'source_sha256', 'fixture_sha256', 'tokenizer_sha256', 'environment', 'partition_summaries')}
summary['audit'] = dict(heldout_pairs=257, calibration_pairs=21, original_markers_preserved=True,
    suffix_ids_preserved=True, original_prefix_preserved_for_task_only_changes=True,
    maximum_baseline_score_change_from_prior=max(reproducibility_errors),
    max_input_tokens=max(v['tokens'] for r in p['pairs'] for v in r['variants'].values()),
    max_fp32_head_margin_difference=max(head_errors), mean_fp32_head_margin_difference=statistics.mean(head_errors),
    atomic_inventory_groups_match=all(r['atomic_token_inventory_groups_match'] for r in calibration))
summary['resume'] = p.get('resume')
summary['layout_instruction_effects'] = {}
for name in core:
    outcomes = {}
    for suffix in ('', '_position_task', '_position_system', '_position_both'):
        variant = name + suffix
        outcomes[variant] = dict(mean_score=statistics.mean(r['variants'][variant]['score'] for r in heldout),
            mean_margin_change_from_same_layout=statistics.mean(r['variants'][variant]['margin'] - r['variants'][name]['margin'] for r in heldout),
            mean_absolute_score_change_from_same_layout=statistics.mean(abs(r['variants'][variant]['score'] - r['variants'][name]['score']) for r in heldout))
    summary['layout_instruction_effects'][name] = outcomes
summary['synthetic_positive_scores'] = [{k: r[k] for k in ('id', 'query', 'document')} | {
    'scores': {name: v['score'] for name, v in r['variants'].items()}} for r in calibration if r['source'] == 'synthetic_sanity' and r['grade'] == 1]
args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
print('Audit:', json.dumps(summary['audit']))
metrics = p['partition_summaries']['holdout']['Touche2020Retrieval.v3']['orderings']
for name in core:
    print(name, {suffix or 'unchanged': metrics[name+suffix] for suffix in ('', '_position_task', '_position_system', '_position_both')})
raw = args.result.read_bytes()
compressed = gzip.compress(raw, mtime=0)
args.result.with_suffix('.json.gz').write_bytes(compressed)
(args.result.parent/'raw_output_manifest.json').write_text(json.dumps(dict(raw_filename=args.result.name,
    raw_sha256=hashlib.sha256(raw).hexdigest(), compressed_filename=args.result.name+'.gz',
    compressed_sha256=hashlib.sha256(compressed).hexdigest(), raw_bytes=len(raw)), indent=2) + '\n')
