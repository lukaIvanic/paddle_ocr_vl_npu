"""CPU-only TEDS comparison of saved outputs; never submit inference requests."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path('/workspace/repos/paddle_ocr_vl_npu')
BASE = Path('/workspace/repos/table_step1_be691de1_20260910/tmp/19_table_ocr_serving/current_checkpoint_20260913')
REFERENCE = ROOT / 'tmp/19_table_ocr_serving/step3_b8qps6_0976fa33_20260911_cached'


def unique_responses(path):
    result = {}
    for row in sorted(map(json.loads, path.read_text().splitlines()), key=lambda row: row['sequence']):
        assert row['status'] == 'ok'
        response = row['service_result']['response']
        request_id = row['request_id']
        if request_id in result:
            assert response['token_ids'] == result[request_id]['token_ids'], request_id
        else:
            result[request_id] = response
    assert len(result) == 665
    return result


def main():
    source = ROOT / '09_persistent_page_engine/scripts/run_omnidocbench_table_api.py'
    spec = importlib.util.spec_from_file_location('saved_table_evaluation', source)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    dataset = Path('/workspace/datasets/OmniDocBench/OmniDocBench.json').read_bytes()
    assert hashlib.sha256(dataset).hexdigest() == evaluator.EXPECTED_JSON_SHA256
    ground_truth = {}
    for page_index, page in enumerate(json.loads(dataset)):
        for annotation_index, annotation in enumerate(page.get('layout_dets') or []):
            if annotation.get('ignore') or annotation.get('category_type') != 'table':
                continue
            request_id = f"page_{page_index:06d}_table_{annotation.get('anno_id', annotation_index)}"
            ground_truth[request_id] = dict(
                request_id=request_id, page_name=Path(page['page_info']['image_path']).name,
                annotation_index=annotation_index, gt_html=annotation.get('html') or '',
            )
    assert len(ground_truth) == 665
    scores = {}
    for name, folder in [('historical', REFERENCE), ('current', BASE / 'cached')]:
        responses = unique_responses(folder / 'b8/qps6/measured/results.jsonl')
        assert responses.keys() == ground_truth.keys()
        records = [dict(ground_truth[rid], pred_html=responses[rid]['text']) for rid in sorted(responses)]
        output = BASE / 'accuracy' / name
        output.mkdir(parents=True, exist_ok=False)
        args = argparse.Namespace(evaluator_root=Path('/workspace/repos/OmniDocBench_eval'),
                                  output_dir=output, teds_workers=12, teds_timeout_s=120.)
        scores[name] = evaluator._score(records, args)
    fields = ('sample_TEDS', 'sample_TEDS_structure_only', 'page_TEDS', 'page_TEDS_structure_only',
              'table_count', 'table_page_count', 'teds_timeout_count', 'teds_error_count')
    old = {r['request_id']: r for r in scores['historical']['per_table']}
    changed = []
    for row in scores['current']['per_table']:
        previous = old[row['request_id']]
        if row['TEDS'] != previous['TEDS'] or row['TEDS_structure_only'] != previous['TEDS_structure_only']:
            changed.append(dict(request_id=row['request_id'], before=previous, after=row))
    result = {
        'note': '665 unique tables; repeated benchmark occurrences are not extra accuracy samples. Both runs scored with the same existing evaluator and normalization. Historical head/preprocessing/postprocessing differ.',
        'dataset_sha256': hashlib.sha256(dataset).hexdigest(),
        'evaluator_script_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'historical': {k:scores['historical'][k] for k in fields},
        'current': {k:scores['current'][k] for k in fields},
        'changed_tables': changed,
    }
    (BASE / 'accuracy/comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'changed_tables'}, indent=2), flush=True)
    print(f'Changed table scores: {len(changed)}', flush=True)
    assert all(scores[name]['teds_timeout_count'] == scores[name]['teds_error_count'] == 0 for name in scores)


if __name__ == '__main__':
    main()
