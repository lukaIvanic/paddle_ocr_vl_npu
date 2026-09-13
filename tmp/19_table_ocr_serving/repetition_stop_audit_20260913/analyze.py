"""Count recorded repetition stops; replay current tracker on native historical IDs.

CPU-only analysis of saved results. No re-tokenization or inference.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys
import subprocess
import types
import zipfile

ROOT = Path(__file__).resolve().parents[3]
# Preserve the audited version after repetition support is removed from product code.
SOURCE = '564da03f:19_table_ocr_serving/_support/serving/repetition.py'
source_bytes = subprocess.check_output(['git', '-C', str(ROOT), 'show', SOURCE])
module = types.ModuleType('audited_repetition')
sys.modules[module.__name__] = module
exec(compile(source_bytes, SOURCE, 'exec'), module.__dict__)


def recorded_counts(path):
    stops = Counter()
    identities = set()
    hits = []
    missing_evidence = 0
    for line in path.open():
        row = json.loads(line)
        service = row.get('service_result') or {}
        result = service.get('response') or {}
        reason = result.get('stop_reason', service.get('stop_reason', 'missing'))
        stops[reason] += 1
        identities.add(row['request_id'])
        missing_evidence += 'repetition' not in result
        evidence = result.get('repetition') or {}
        if reason == 'repetition' or evidence:
            hits.append({'request_id': row['request_id'], 'evidence': evidence})
    return dict(source=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                generations=sum(stops.values()), unique_input_ids=len(identities),
                stop_reasons=dict(stops), missing_repetition_field=missing_evidence, hits=hits)


def replay_historical():
    prior = ROOT / 'tmp/09_persistent_page_engine/repetition_audit_910b_full_4fa1538_r2/report.json'
    archive = Path(json.loads(prior.read_text())['input'])
    counts = Counter()
    stops = Counter()
    hits = []
    with zipfile.ZipFile(archive) as z:
        with z.open('recognition_trace.jsonl') as f:
            digest = hashlib.sha256()
            for line in f:
                digest.update(line)
                row = json.loads(line)
                counts[row['label']] += 1
                stops[row['stop_reason']] += 1
                tokens = row['token_ids']
                # The real scheduler checks EOS before repetition.
                if row['stop_reason'] == 'eos':
                    tokens = tokens[:-1]
                tracker = module.ExactCycleTracker()
                for token in tokens:
                    evidence = tracker.update(token)
                    if evidence:
                        hits.append(dict(request_id=row['request_id'], label=row['label'],
                                         original_stop_reason=row['stop_reason'],
                                         original_tokens=len(tokens), evidence=evidence.to_dict(),
                                         later_tokens_not_generated=len(tokens)-evidence.trigger_length,
                                         already_generated_tokens_trimmed=evidence.trigger_length-evidence.trim_length))
                        break
    return dict(source=str(archive.relative_to(ROOT)), trace_sha256=digest.hexdigest(),
                generations=sum(counts.values()), by_label=dict(counts), original_stops=dict(stops),
                hits_by_label=dict(Counter(h['label'] for h in hits)), hits=hits)


if __name__ == '__main__':
    paths = sorted(p for p in (ROOT/'tmp/19_table_ocr_serving').rglob('results.jsonl')
                   if 'measured' in p.parts and 'warm' not in p.parts)
    paths += [ROOT/'tmp/09_persistent_page_engine'/p for p in (
        'step1_b8qps6_20260910/b8/qps6/measured/results.jsonl',
        'table_poisson_frontier_screen1000_be691de1_20260907/b8/qps6/measured/results.jsonl')]
    report = dict(tracker_source_sha256=hashlib.sha256(source_bytes).hexdigest(),
                  measured_runs=[recorded_counts(p) for p in paths], historical_replay=replay_historical())
    output = Path(__file__).with_name('report.json')
    output.write_text(json.dumps(report, indent=2)+'\n')
    for r in report['measured_runs']:
        print(r['source'], r['generations'], r['unique_input_ids'], r['stop_reasons'], 'hits',len(r['hits']))
    h=report['historical_replay']
    print('HISTORICAL', h['generations'],h['by_label'],h['hits_by_label'])
    print('LATER_TOKENS_NOT_GENERATED',sum(x['later_tokens_not_generated'] for x in h['hits']))
    print('ALREADY_GENERATED_TOKENS_TRIMMED',sum(x['already_generated_tokens_trimmed'] for x in h['hits']))
    print('REPORT', output)
