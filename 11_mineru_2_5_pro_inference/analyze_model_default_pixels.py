"""Compare completed model-default pixel evidence with the production 910B run."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re


def read(path):
    return json.loads(path.read_text())


def page_scores(run):
    directory = run / 'evaluation/work/result'
    prefix = 'predictions_quick_match_'
    text = {page: 1 - value for page, value in read(directory / (prefix + 'text_block_per_page_edit.json')).items()}
    result = {'text': text}
    for category, filename, metric in (
        ('formula_CDM', 'display_formula_per_sample_CDM.json', None),
        ('table_TEDS', 'table_per_table_TEDS.json', 'TEDS'),
    ):
        groups = defaultdict(list)
        for identity, value in read(directory / (prefix + filename)).items():
            groups[identity.rsplit('_', 1)[0]].append(value if metric is None else value[metric])
        result[category] = {page: sum(values) / len(values) for page, values in groups.items()}
    summary = read(directory / (prefix + 'run_summary.json'))
    metrics = summary['notebook_metric_summary']['metrics']
    accuracy = dict(overall=summary['notebook_metric_summary']['overall_notebook'],
        text=100 * (1 - metrics['text_block_Edit_dist']['raw']),
        table_TEDS=metrics['table_TEDS']['notebook_value'],
        formula_CDM=metrics['display_formula_CDM']['notebook_value'])
    # Verify our per-page aggregation is exactly the evaluator's published score.
    for category, pages in result.items():
        assert abs(100 * sum(pages.values()) / len(pages) - accuracy[category]) < 1e-8, category
    return result, accuracy, summary


def layout_comparison(production, candidate):
    previous = {p.name: p for p in (production / 'output/layout_regions').glob('*.json')}
    current = {p.name: p for p in (candidate / 'output/layout_regions').glob('*.json')}
    missing = sorted(previous.keys() - current.keys())
    added = sorted(current.keys() - previous.keys())
    differences = []
    hashes = [hashlib.sha256(), hashlib.sha256()]
    total_blocks = 0
    for name in sorted(previous.keys() & current.keys()):
        old, new = read(previous[name]), read(current[name])
        total_blocks += len(new['blocks'])
        for digest, value in zip(hashes, (old, new)):
            digest.update(name.encode())
            digest.update(json.dumps(value, sort_keys=True, separators=(',', ':')).encode())
        if old != new:
            differences.append(dict(page=name,
                changed_blocks=[i for i, pair in enumerate(zip(old['blocks'], new['blocks'])) if pair[0] != pair[1]],
                old_block_count=len(old['blocks']), new_block_count=len(new['blocks'])))
    return dict(reference_pages=len(previous), candidate_pages=len(current), missing=missing, added=added,
                identical=not (missing or added or differences), differences=differences,
                candidate_blocks=total_blocks, canonical_sha256=[h.hexdigest() for h in hashes])


def trace_lengths(run):
    stops = Counter()
    length_stops = []
    over4096 = []
    invalid_allowances = []
    request_count = 0
    for line in (run / 'output/generation_trace.jsonl').open():
        row = json.loads(line)
        prompt = len(row['prompt_token_ids'])
        generated = len(row['generated_token_ids'])
        value = {key: row[key] for key in ('request_id', 'page', 'block_type', 'stop_reason')}
        value.update(prompt_length=prompt, generated_length=generated, total_length=prompt + generated)
        stops[row['stop_reason']] += 1
        request_count += 1
        if row['stop_reason'] == 'length':
            length_stops.append(value)
        if prompt + generated > 4096:
            over4096.append(value)
        if row['max_new_tokens'] != 8192 - prompt:
            invalid_allowances.append(dict(value, max_new_tokens=row['max_new_tokens']))
    return dict(request_count=request_count, stop_counts=dict(stops), length_stops=length_stops,
                over4096=over4096, unexpected_output_allowances=invalid_allowances)


def cdm_timing(run, summary):
    sample_count = summary['stage_execution']['metrics']['display_formula']['CDM']['sample_count']
    log = (run / 'evaluation/run.log').read_text(errors='replace')
    completed = []
    for match in re.finditer(r'CDM:\s*100%[^\r\n]*?\|\s*(\d+)/(\d+)\s*\[([\d:]+)<', log):
        done, total, elapsed = match.groups()
        if int(done) == int(total) == sample_count:
            seconds = 0
            for component in elapsed.split(':'):
                seconds = seconds * 60 + int(component)
            completed.append(seconds)
    seconds = completed[-1] if completed else None
    return dict(workers=summary['stage_execution']['metrics']['display_formula']['CDM']['workers'],
                samples=sample_count, cdm_elapsed_s=seconds,
                cdm_samples_per_s=sample_count / seconds if seconds else None,
                timing_source='Final completed CDM progress elapsed time (one-second resolution)',
                evaluation_wall_s=float((run / 'evaluation/wall_s.txt').read_text()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--production-run', required=True, type=Path)
    parser.add_argument('--candidate-run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    old_pages, old_accuracy, _ = page_scores(args.production_run)
    new_pages, new_accuracy, evaluation = page_scores(args.candidate_run)
    changes = {}
    for category, pages in old_pages.items():
        candidate = new_pages[category]
        counters = Counter(better=0, worse=0, unchanged=0)
        details = []
        for page in sorted(pages.keys() & candidate.keys()):
            delta = candidate[page] - pages[page]
            direction = 'better' if delta > 1e-12 else 'worse' if delta < -1e-12 else 'unchanged'
            counters[direction] += 1
            if direction != 'unchanged':
                details.append(dict(page=page, reference=pages[page], candidate=candidate[page], delta=delta))
        changes[category] = dict(counters, missing=sorted(pages.keys() - candidate.keys()),
                                added=sorted(candidate.keys() - pages.keys()), changed_pages=details)
    result = dict(production_run=str(args.production_run), candidate_run=str(args.candidate_run),
        accuracy_reference=old_accuracy, accuracy_candidate=new_accuracy,
        accuracy_delta={key: new_accuracy[key] - old_accuracy[key] for key in old_accuracy},
        page_changes=changes, layout=layout_comparison(args.production_run, args.candidate_run),
        lengths=trace_lengths(args.candidate_run), evaluation=cdm_timing(args.candidate_run, evaluation))
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ('page_changes', 'lengths')}, indent=2))
    print('PAGE_COUNTS ' + json.dumps({k: {f: v[f] for f in ('better', 'worse', 'unchanged', 'missing', 'added')} for k, v in changes.items()}))
    print('LENGTH_STOPS ' + json.dumps(result['lengths'], ensure_ascii=False))


if __name__ == '__main__':
    main()
