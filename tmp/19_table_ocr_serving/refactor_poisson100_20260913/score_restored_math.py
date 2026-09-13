"""Rescore all 665 saved table generations with restored production formatting.

CPU only: no model imports, request submissions, tokenization or new inference.
Retain prior scores and artifacts unchanged; write a separate result directory.
"""
import argparse
import ast
import hashlib
import html
import importlib.util
import json
from pathlib import Path
import re
import runpy
import subprocess

ROOT = Path('/workspace/repos/paddle_ocr_vl_npu')
SAVED = Path('/workspace/repos/table_step1_be691de1_20260910/tmp/19_table_ocr_serving/current_checkpoint_20260913')
OUTPUT = SAVED.parent / 'refactor_poisson100_20260913/accuracy_restored_math'


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    scorer_path = ROOT / '09_persistent_page_engine/scripts/run_omnidocbench_table_api.py'
    assert hashlib.sha256(scorer_path.read_bytes()).hexdigest() == 'cf2db682c278598b615bf8482bbbd7bf1d0c3e76f91ba222147edd63e51121b2'
    evaluator_root = Path('/workspace/repos/OmniDocBench_eval')
    assert subprocess.check_output(['git', '-C', str(evaluator_root), 'rev-parse', 'HEAD'], text=True).strip() == '2b161d010d2e3aff77a0edef359ea3a6411d23cd'
    spec = importlib.util.spec_from_file_location('saved_table_evaluation', scorer_path)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    previous = runpy.run_path(str(ROOT / 'tmp/19_table_ocr_serving/current_checkpoint_20260913/score_saved_outputs.py'))
    input_path = SAVED / 'cached/b8/qps6/measured/results.jsonl'
    responses = previous['unique_responses'](input_path)
    source = (ROOT / '19_table_ocr_serving/p03_crop_processing.py').read_text()
    # Execute just the production string functions, avoiding torch/Kornia imports.
    names = {'normalize_math_delimiters', 'convert_otsl_to_html', '_parse_otsl_rows'}
    nodes = [node for node in ast.parse(source).body
             if isinstance(node, ast.FunctionDef) and node.name in names
             or isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_OTSL_TOKEN' for t in node.targets)]
    formatting_source = ast.unparse(ast.Module(body=nodes, type_ignores=[]))
    ns = {'html': html, 're': re}
    exec('from __future__ import annotations\n' + formatting_source, ns)
    dataset = Path('/workspace/datasets/OmniDocBench/OmniDocBench.json').read_bytes()
    assert hashlib.sha256(dataset).hexdigest() == evaluator.EXPECTED_JSON_SHA256
    records = []
    for page_index, page in enumerate(json.loads(dataset)):
        for annotation_index, annotation in enumerate(page.get('layout_dets') or []):
            if annotation.get('ignore') or annotation.get('category_type') != 'table':
                continue
            rid = f"page_{page_index:06d}_table_{annotation.get('anno_id', annotation_index)}"
            formatted = ns['normalize_math_delimiters'](responses[rid]['raw_text'])
            prediction = ns['convert_otsl_to_html'](formatted) or formatted
            records.append(dict(request_id=rid, page_name=Path(page['page_info']['image_path']).name,
                annotation_index=annotation_index, gt_html=annotation.get('html') or '', pred_html=prediction))
    assert len(records) == len(responses) == 665
    records.sort(key=lambda row: row['request_id'])
    (OUTPUT / 'predictions.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records))
    provenance = dict(runtime_commit=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        saved_input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
        formatting_source_sha256=hashlib.sha256(formatting_source.encode()).hexdigest(),
        dataset_sha256=hashlib.sha256(dataset).hexdigest(),
        note='CPU-only formatting replay of the saved current 1000-request run, deduplicated to 665 tables. Not a new generation run.')
    (OUTPUT / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    scores = evaluator._score(records, argparse.Namespace(evaluator_root=evaluator_root,
        output_dir=OUTPUT, teds_workers=12, teds_timeout_s=120.))
    assert scores['teds_error_count'] == scores['teds_timeout_count'] == 0
    old = json.loads((SAVED / 'accuracy/comparison.json').read_text())
    fields = ('sample_TEDS', 'sample_TEDS_structure_only', 'page_TEDS', 'page_TEDS_structure_only',
              'table_count', 'table_page_count', 'teds_error_count', 'teds_timeout_count')
    report = dict(historical=old['historical'], before_restoration=old['current'],
        restored={key: scores[key] for key in fields})
    report['page_teds_percentage_point_delta_vs_historical'] = 100 * (scores['page_TEDS'] - old['historical']['page_TEDS'])
    historical = {row['request_id']: row for row in json.loads((SAVED / 'accuracy/historical/scores.json').read_text())['per_table']}
    report['changed_tables'] = [dict(request_id=row['request_id'], before=historical[row['request_id']], after=row)
                              for row in scores['per_table'] if row['TEDS'] != historical[row['request_id']]['TEDS']
                              or row['TEDS_structure_only'] != historical[row['request_id']]['TEDS_structure_only']]
    (OUTPUT / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'changed_tables'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
