"""Compare page-weighted accuracy and raw production traces for a paired cap run."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean


def load(path):
    return json.loads(path.read_text())


def trace(path):
    with path.open() as f:
        return {r['request_id']: r for r in map(json.loads, f)}


def page_average(samples, metric=None):
    grouped = defaultdict(list)
    for key, value in samples.items():
        grouped[key.rsplit('_[', 1)[0]].append(value if metric is None else value[metric])
    return {page: mean(values) for page, values in grouped.items()}


def metrics(run):
    result = run / 'evaluation/work/result'
    prefix = 'predictions_quick_match_'
    raw = load(result / (prefix + 'metric_result.json'))
    values = {
        'text_accuracy': 100 * (1 - raw['text_block']['page']['Edit_dist']['ALL']),
        'formula_cdm': 100 * raw['display_formula']['page']['CDM']['ALL'],
        'table_teds': 100 * raw['table']['page']['TEDS']['ALL'],
        'table_structure_teds': 100 * raw['table']['page']['TEDS_structure_only']['ALL'],
        'reading_order_edit_distance': raw['reading_order']['page']['Edit_dist']['ALL'],
    }
    values['overall'] = mean(values[k] for k in ['text_accuracy', 'formula_cdm', 'table_teds'])
    pages = {
        'text_accuracy': {p: 1 - v for p, v in load(result / (prefix + 'text_block_per_page_edit.json')).items()},
        'formula_cdm': page_average(load(result / (prefix + 'display_formula_per_sample_CDM.json'))),
        'table_teds': page_average(load(result / (prefix + 'table_per_table_TEDS.json')), 'TEDS'),
    }
    for key, rows in pages.items():
        assert abs(mean(rows.values()) * 100 - values[key]) < 1e-6, key
    return values, pages, raw


def summarize(root):
    assert (root / 'exit_code.txt').read_text().strip() == '0'
    values, pages, summaries, traces = {}, {}, {}, {}
    for label in ['original', 'capped']:
        run = root / label
        assert (run / 'evaluation/exit_code.txt').read_text().strip() == '0'
        values[label], pages[label], raw = metrics(run)
        summaries[label] = load(run / 'output/run_summary_shard_00.json')
        traces[label] = trace(run / 'output/generation_trace.jsonl')
        values[label]['evaluation_diagnostics'] = {
            'match': raw['match_debug'], 'teds': raw['table'].get('metric_debug'),
            'run_summary': load(run / 'evaluation/work/result/predictions_quick_match_run_summary.json'),
        }
    report = {
        'selection': {k: v for k, v in load(root / 'selection.json').items() if k != 'crops'},
        'accuracy': values,
        'delta_capped_minus_original': {k: values['capped'][k] - v for k, v in values['original'].items()
                                        if isinstance(v, (int, float))},
        'page_changes': {}, 'performance': {},
    }
    for metric in pages['original']:
        a, b = pages['original'][metric], pages['capped'][metric]
        common = a.keys() & b.keys()
        changes = sorted([{'page': p, 'original': a[p] * 100, 'capped': b[p] * 100,
                           'delta_pp': (b[p] - a[p]) * 100} for p in common], key=lambda r: r['delta_pp'])
        report['page_changes'][metric] = {
            'original_pages': len(a), 'capped_pages': len(b), 'common_pages': len(common),
            'original_only': sorted(a.keys() - b.keys()), 'capped_only': sorted(b.keys() - a.keys()),
            'improved': sum(r['delta_pp'] > 1e-9 for r in changes),
            'worsened': sum(r['delta_pp'] < -1e-9 for r in changes),
            'unchanged': sum(abs(r['delta_pp']) <= 1e-9 for r in changes),
            'worst': changes[:10], 'best': changes[-10:][::-1], 'all': changes,
        }
    for label, s in summaries.items():
        report['performance'][label] = {
            'completed': s['completed'], 'processor_max_pixels': s['processor_max_pixels'],
            'hot_wall_s': s['pipeline_wall_s'], 'hot_pg_s': s['measured_group_pages_per_s'],
            'vision': s['vision_timing']['all'],
            'stop_reasons': dict(Counter(r['stop_reason'] for r in traces[label].values())),
            'generated_tokens': sum(len(r['generated_token_ids']) for r in traces[label].values()),
        }
    a, b = traces['original'], traces['capped']
    audit = Counter()
    changed_layout = []
    changed_resolution = []
    for key in a.keys() & b.keys():
        left, right = a[key], b[key]
        same_tokens = left['generated_token_ids'] == right['generated_token_ids']
        if left['phase'] == 'layout':
            audit['layout_requests'] += 1
            audit['layout_exact_tokens'] += same_tokens
            if not same_tokens:
                changed_layout.append(left['page'])
        else:
            old = left['prompt_token_ids'].count(151655) * 4
            new = right['prompt_token_ids'].count(151655) * 4
            changed = old != new
            tag = 'changed_resolution' if changed else 'unchanged_resolution'
            audit[tag + '_requests'] += 1
            audit[tag + '_exact_tokens'] += same_tokens
            if changed:
                changed_resolution.append({'request_id': key, 'type': left['block_type'],
                                           'old_raw_tokens': old, 'new_raw_tokens': new,
                                           'exact_output_tokens': same_tokens})
    predictions = list((root / 'original/output/predictions').glob('*.md'))
    audit['markdown_exact_pages'] = sum(p.read_bytes() == (root / 'capped/output/predictions' / p.name).read_bytes()
                                        for p in predictions)
    report['trace_audit'] = dict(audit, changed_layout_pages=changed_layout,
                               original_only=sorted(a.keys() - b.keys()), capped_only=sorted(b.keys() - a.keys()),
                               changed_resolution=changed_resolution)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root', type=Path)
    args = parser.parse_args()
    report = summarize(args.run_root)
    (args.run_root / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    for k, v in report['accuracy'].items():
        print(k, {n: x for n, x in v.items() if n != 'evaluation_diagnostics'})
    print('DELTA', report['delta_capped_minus_original'])
    print('PERFORMANCE', report['performance'])
    print('TRACE', {k: v for k, v in report['trace_audit'].items() if k != 'changed_resolution'})
    print('PAGE_CHANGES', {k: {n: x for n, x in v.items() if n != 'all'} for k, v in report['page_changes'].items()})
