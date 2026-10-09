"""Score changed instructions with the frozen teacher and prove exact-input cache reuse."""
import argparse
import collections
import json
import math
from pathlib import Path
import time

from distill_runtime import Runtime, read, digest, model_manifest, save
from bge_instruction_ablation import SECTIONS, validate_derivation
from margin_distillation import benchmark_metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'control-dataset', 'old-teacher', 'manifest', 'model', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    assert not (args.output / 'teacher.json').exists(), 'Refuse duplicate teacher run'
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    data, control, manifest, old = map(read, (args.dataset, args.control_dataset, args.manifest, args.old_teacher))
    assert digest(args.control_dataset) == old['dataset_sha256'] == manifest['control_dataset_sha256']
    assert digest(args.old_teacher) == data['derivation']['control_teacher_sha256']
    assert digest(args.manifest) == data['derivation']['instruction_manifest_sha256']
    changes = validate_derivation(data, control, manifest)
    runtime = Runtime(args.model)
    records = {s: runtime.records(data[s], s) for s in SECTIONS}
    lengths = dict(runtime.lengths)
    assert all(v['truncated'] == 0 for v in lengths.values())
    weights = model_manifest(args.model)
    assert weights == old['model_sha256']
    config = {'order':'query_first','margin':'yes-minus-no raw logits','max_length':8192,
        'body_truncation':'right before fixed suffix','weights':'FP32','autocast':'BF16',
        'attention':'fusion_attention','tokenizer_files':{f.name:digest(f) for f in args.model.glob('*token*') if f.is_file()},
        'prefix_suffix_source_sha256':digest(Path(__file__).parent/'transformers_rerank.py')}
    assert old['scoring_config'] == config
    target = {'dataset_sha256':digest(args.dataset), 'model_sha256':weights,
        'lengths':lengths, 'scoring':old['scoring'], 'scoring_config':config,
        'instruction_changes':changes, 'control_teacher_sha256':digest(args.old_teacher),
        'instruction_manifest_sha256':digest(args.manifest), 'scores':{}, 'seconds':{}, 'reuse':{}}
    model = runtime.load(args.model)
    for section in SECTIONS:
        changed = set(changes[section])
        old_groups = {g['id']:g for g in control[section]}
        reused = [g for g in data[section] if g['id'] not in changed]
        assert all(g == old_groups[g['id']] for g in reused)
        scores = {g['id']:old['scores'][section][g['id']] for g in reused}
        target['reuse'][section] = {'reused':len(reused), 'rescored':len(changed)}
        rows = [row for row in records[section] if row['group_id'] in changed]
        elapsed = 0.0
        # Preserve score()'s full-section length-sorted microbatch plan; chunks
        # could change padding and BF16 rounding, so use the original scorer once.
        if rows:
            fresh, elapsed = runtime.score(model, rows, section)
            assert set(fresh) == changed
            scores.update(fresh)
        assert set(scores) == {g['id'] for g in data[section]}
        assert all(len(scores[g['id']]) == len(g['documents']) and
                   all(math.isfinite(v) for v in scores[g['id']]) for g in data[section])
        target['scores'][section], target['seconds'][section] = scores, elapsed
        if 'benchmark' in section:
            target.setdefault('metrics', {})[section] = benchmark_metrics(data[section], scores)
        save(args.output / 'partial_teacher.json', target)
        print('TEACHER_SECTION', json.dumps({'section':section, **target['reuse'][section], 'seconds':elapsed}), flush=True)
    target['status'] = 'completed'
    target['total_seconds'] = time.monotonic() - started
    save(args.output / 'teacher.json', target)
    save(args.output / 'completion.json', {'status':'completed','teacher_sha256':digest(args.output/'teacher.json'),
         'reuse':target['reuse'], 'seconds':target['total_seconds'], 'lengths':lengths})
    print('TEACHER_COMPLETE', json.dumps(target['reuse']), flush=True)


if __name__ == '__main__':
    main()
