"""Reproduce the per-query comparison metrics and query-cluster bootstrap."""
import argparse
import json
from pathlib import Path
import random
import statistics

parser = argparse.ArgumentParser()
parser.add_argument('--main', type=Path, required=True)
parser.add_argument('--joint', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
main = json.loads(args.main.read_text())
joint_run = json.loads(args.joint.read_text())
assert main['status'] == joint_run['status'] == 'completed'
rows = main['pairs']
joint = {r['id']: r for r in joint_run['pairs'] if r['partition'] == 'holdout'}
assert len(joint) == len(rows) == 257
for row in rows:
    assert row['variants']['normal']['score'] == joint[row['id']]['variants']['normal']['score']
    row['variants']['swapped_contents_joint_remap'] = joint[row['id']]['variants']['swapped_contents_joint_remap']
by_query = {q: [r for r in rows if r['query_id'] == q] for q in sorted({r['query_id'] for r in rows}, key=int)}
metrics = {}
for variant in rows[0]['variants']:
    per_query = []
    for q, subset in by_query.items():
        comparisons = [(a, b) for a in subset for b in subset if a['grade'] > b['grade']]
        if not comparisons:
            continue
        def accuracy(name):
            return sum(a['variants'][name]['score'] > b['variants'][name]['score'] for a, b in comparisons) / len(comparisons)
        per_query.append(dict(query_id=q, comparisons=len(comparisons), accuracy=accuracy(variant), normal_accuracy=accuracy('normal')))
    differences = [r['accuracy'] - r['normal_accuracy'] for r in per_query]
    rng = random.Random(20261006)
    bootstrap = sorted(statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(5000))
    metrics[variant] = dict(per_query=per_query, macro_accuracy=statistics.mean(r['accuracy'] for r in per_query),
                            macro_difference_from_normal=statistics.mean(differences),
                            bootstrap_95pct_macro_difference=[bootstrap[125], bootstrap[4874]],
                            bootstrap_note='Query-cluster resampling, 5000 draws, seed 20261006; descriptive for this selected short-document sample only.')
args.output.write_text(json.dumps(metrics, indent=2) + '\n')
