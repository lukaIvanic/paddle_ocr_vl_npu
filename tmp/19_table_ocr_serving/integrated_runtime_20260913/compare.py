"""Compare saved requests, native generations, formatting and latency; no inference."""
import argparse
import ast
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--current',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--normalize-reference-math',action='store_true')
    args=parser.parse_args()
    old={row['sequence']:row for row in rows(args.reference/'results.jsonl')}
    new={row['sequence']:row for row in rows(args.current/'results.jsonl')}
    assert old.keys()==new.keys()
    assert rows(args.reference/'schedule.jsonl')==rows(args.current/'schedule.jsonl')
    fields=('token_ids','raw_text','text','stop_reason','crop_size','input_tokens','projected_image_tokens')
    mismatches={field:[] for field in fields}
    formatting=None
    if args.normalize_reference_math:
        source=(ROOT/'19_table_ocr_serving/p03_crop_processing.py').read_text()
        names={'normalize_math_delimiters','convert_otsl_to_html','_parse_otsl_rows'}
        nodes=[node for node in ast.parse(source).body
            if isinstance(node,ast.FunctionDef) and node.name in names
            or isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_OTSL_TOKEN' for t in node.targets)]
        formatting={'html':html,'re':re}
        exec('from __future__ import annotations\n'+ast.unparse(ast.Module(body=nodes,type_ignores=[])),formatting)
    for sequence in sorted(old):
        a,b=old[sequence],new[sequence]
        assert a['request_id']==b['request_id'] and a['status']==b['status']=='ok'
        a,b=(dict(row['service_result']['response']) for row in (a,b))
        if formatting is not None:
            content=formatting['normalize_math_delimiters'](a['raw_text'])
            a['text']=formatting['convert_otsl_to_html'](content) or content
        for field in fields:
            if a[field]!=b[field]: mismatches[field].append(old[sequence]['request_id'])
    performance={}
    for label,folder in (('reference',args.reference),('integrated',args.current)):
        summary=json.loads((folder/'summary.json').read_text())
        performance[label]=dict(latency_s=summary['request_latency_s'],
            scheduled_latency_s=summary['scheduled_latency_s'],
            completed_qps=summary['completed_request_count']/summary['run_wall_s'],
            completed=summary['completed_request_count'],errors=summary['failed_request_count'],
            max_outstanding=summary['max_active_requests'])
    report=dict(requests=len(new),schedule_identical=True,
        reference_math_formatting_replayed=args.normalize_reference_math,
        mismatches=mismatches,performance=performance,
        stop_reasons=dict(Counter(row['service_result']['response']['stop_reason'] for row in new.values())),
        reference=str(args.reference),current=str(args.current),
        result_sha256={label:hashlib.sha256((folder/'results.jsonl').read_bytes()).hexdigest()
            for label,folder in (('reference',args.reference),('integrated',args.current))})
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    assert not any(mismatches.values()),'Output differences require investigation'


if __name__=='__main__': main()
