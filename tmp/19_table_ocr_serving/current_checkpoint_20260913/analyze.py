"""Recompute output, formatting, repeatability and ownership checks locally."""
import ast
from collections import Counter, defaultdict
import difflib
import hashlib
import html
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parent
REFERENCE = BASE.parent / 'step3_b8qps6_0976fa33_20260911_cached'


def load(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main():
    relative = Path('b8/qps6/measured/results.jsonl')
    old_path, new_path = REFERENCE / relative, BASE / 'cached' / relative
    old = {r['sequence']:r for r in load(old_path)}
    new = {r['sequence']:r for r in load(new_path)}
    assert old.keys() == new.keys() and len(new) == 1000
    source = (ROOT / '19_table_ocr_serving/p03_crop_processing.py').read_text()
    nodes = [n for n in ast.parse(source).body
             if isinstance(n, ast.FunctionDef) and n.name in ('convert_otsl_to_html', '_parse_otsl_rows')
             or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_OTSL_TOKEN' for t in n.targets)]
    namespace = {'html':html, 're':re}
    exec('from __future__ import annotations\n' + ast.unparse(ast.Module(body=nodes, type_ignores=[])), namespace)

    def formatted(raw):
        return namespace['convert_otsl_to_html'](raw) or raw

    counts, unique = Counter(), defaultdict(set)
    variants, examples = defaultdict(set), {}
    for index in sorted(new):
        assert old[index]['request_id'] == new[index]['request_id']
        a, b = old[index]['service_result']['response'], new[index]['service_result']['response']
        request_id = new[index]['request_id']
        variants[request_id].add(tuple(b['token_ids']))
        checks = dict(native_changed=a['token_ids'] != b['token_ids'], raw_changed=a['raw_text'] != b['raw_text'],
                      html_changed=a['text'] != b['text'], current_formatter_mismatch=formatted(b['raw_text']) != b['text'],
                      html_changed_with_identical_raw=a['raw_text'] == b['raw_text'] and a['text'] != b['text'],
                      html_changed_after_common_formatter=formatted(a['raw_text']) != formatted(b['raw_text']))
        for key, changed in checks.items():
            counts[key] += changed
            if changed:
                unique[key].add(request_id)
        if checks['raw_changed'] and request_id not in examples:
            differences = []
            for op, left, right, start, stop in difflib.SequenceMatcher(None, a['raw_text'], b['raw_text'], autojunk=False).get_opcodes():
                if op != 'equal':
                    differences.append(dict(operation=op, old=a['raw_text'][left:right], new=b['raw_text'][start:stop]))
            examples[request_id] = dict(sequence=index, differences=differences)
    ownership = {phase:load(BASE / phase / 'ownership.jsonl') for phase in ('compile', 'cached')}
    report = dict(
        source_sha256={'historical_results':hashlib.sha256(old_path.read_bytes()).hexdigest(),
                       'current_results':hashlib.sha256(new_path.read_bytes()).hexdigest()},
        occurrence_counts=dict(counts), unique_counts={key:len(value) for key,value in unique.items()},
        unique_tables=len(variants), repeated_input_token_variants={key:len(value) for key,value in variants.items() if len(value)>1},
        raw_text_changes_by_table=examples,
        ownership={phase:dict(checks=len(rows), foreign_process_checks=sum(bool(set(r['pids'])-set(r['owned'])) for r in rows),
                              final_device_pids=rows[-1]['pids']) for phase,rows in ownership.items()},
    )
    # The earlier short run already used the 60k head and Kornia/uint8 path.
    # Compare shared inputs, but do not equate its different load with this run.
    short = load(BASE.parent / 'preprocess_poisson100_20260911/b8_both_measured/b8/measured/results/results.jsonl')
    current_by_id = {row['request_id']:row['service_result']['response'] for row in new.values()}
    fields = ('token_ids', 'raw_text', 'stop_reason', 'crop_size', 'input_tokens', 'projected_image_tokens')
    report['previous_same_head_preprocessing_100'] = {
        'count':len(short),
        'mismatches':{field:[row['request_id'] for row in short
                             if row['service_result']['response'][field] != current_by_id[row['request_id']][field]]
                      for field in fields},
        'qualification':'Shared-input output comparison only; different arrival workload and instrumentation, not performance parity.',
    }
    accuracy_path = BASE / 'accuracy/comparison.json'
    if accuracy_path.exists():
        accuracy = json.loads(accuracy_path.read_text())
        historical = json.loads((BASE / 'accuracy/historical/scores.json').read_text())
        page_counts = Counter(row['page_name'] for row in historical['per_table'])
        contributions = defaultdict(float)
        improved = worsened = 0
        for row in accuracy['changed_tables']:
            delta = row['after']['TEDS'] - row['before']['TEDS']
            group = 'raw_changed_may_also_include_postprocessing' if row['request_id'] in examples else 'postprocessing_only'
            contributions[group] += 100 * delta / page_counts[row['before']['page_name']] / len(page_counts)
            improved += delta > 0
            worsened += delta < 0
        report['accuracy_change_groups'] = dict(page_teds_percentage_point_contributions=dict(contributions),
                                               improved_table_scores=improved, worsened_table_scores=worsened)
    (BASE / 'output_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key:value for key,value in report.items() if key!='raw_text_changes_by_table'}, indent=2))


if __name__ == '__main__':
    main()
