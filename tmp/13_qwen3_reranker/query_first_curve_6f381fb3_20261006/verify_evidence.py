"""Verify saved experiment evidence without loading a model or accelerator."""
import argparse, collections, gzip, hashlib, json, math, pathlib, statistics, sys
p=argparse.ArgumentParser()
p.add_argument('run',type=pathlib.Path)
p.add_argument('--repo',type=pathlib.Path,required=True)
a=p.parse_args()
sys.path.insert(0,str(a.repo/'13_qwen3_reranker'))
from curve_metrics import validation_metrics
from mixture_data import text_hash
x=json.loads((a.run/'result.json').read_text())
raw=(a.run/'inputs/mixture.json.gz').read_bytes()
d=json.loads(gzip.decompress(raw))
assert hashlib.sha256(raw).hexdigest()==x['dataset_sha256']
assert x['status']=='completed'
assert (a.run/'exit_code.txt').read_text().strip()=='0'
assert [u['step'] for u in x['updates']]==list(range(1,251))
assert set(x['evaluations'])=={'0','10','50','100','250'}
assert len(x['validation_ids'])==384 and len(set(x['validation_ids']))==384
assert len(x['touche_query_ids'])==49 and len(set(x['touche_query_ids']))==49
assert x['training']['checkpoints_saved'] is False
for u in x['updates']:
 assert u['parameter_changed'] and u['pairs']==32
 assert math.isfinite(u['loss']) and math.isfinite(u['gradient_norm'])
train_queries={text_hash(g['query']) for g in d['train']}
val_queries={text_hash(g['query']) for g in d['validation']}
train_documents={text_hash(t) for g in d['train'] for t in g['documents']}
val_documents={text_hash(t) for g in d['validation'] for t in g['documents']}
assert not train_queries & val_queries and not train_documents & val_documents
blocked=d['provenance']['touche_exclusion']
assert blocked['fixture_sha256']==x['fixture_sha256']
assert not (train_queries|val_queries)&set(blocked['query_hashes'])
assert not (train_documents|val_documents)&set(blocked['document_hashes'])
excluded=set(d['provenance']['excluded_families'])
assert not {g['source'] for g in d['train']+d['validation']} & excluded
assert len({g['source'] for g in d['train']})==55
val_by_id={g['id']:g for g in d['validation']}
vg=[val_by_id[i] for i in x['validation_ids']]
for e in x['evaluations'].values():
 assert e['validation_queries']==384 and e['touche_queries']==49
 assert e['seconds']['round']<180
 assert set(e['touche_margins'])==set(x['touche_query_ids'])
 assert all(len(m)==100 and all(math.isfinite(v) for v in m.values()) for m in e['touche_margins'].values())
 assert all(all(math.isfinite(v) for v in pair) for pair in e['validation_logits'])
 recomputed=validation_metrics(vg,e['validation_logits'])
 assert recomputed.keys()==e['validation'].keys()
 assert recomputed['overall']==e['validation']['overall']
 assert recomputed['per_source']==e['validation']['per_source']
 assert math.isclose(recomputed['source_macro_ordering_accuracy'],e['validation']['source_macro_ordering_accuracy'],rel_tol=1e-12)
 assert math.isclose(recomputed['pointwise_cross_entropy'],e['validation']['pointwise_cross_entropy'],rel_tol=1e-12)
consumed=x['training_order_ids'][:4000]
assert len(set(consumed))==4000
train_by_id={g['id']:g for g in d['train']}
counts=collections.Counter(train_by_id[i]['source'] for i in consumed)
recorded=collections.Counter()
for u in x['updates']:recorded.update(u['pairs_by_source'])
assert recorded==collections.Counter({s:2*n for s,n in counts.items()})
summary={'all_checks_passed':True,'total_seconds':x['total_seconds'],'training_update_seconds':sum(u['seconds'] for u in x['updates']),'mean_update_seconds':statistics.mean(u['seconds'] for u in x['updates']),'evaluation_seconds':sum(e['seconds']['round'] for e in x['evaluations'].values()),'peak_allocated_gib':max(u['peak_allocated_gib'] for u in x['updates']),'peak_reserved_gib':max(u['peak_reserved_gib'] for u in x['updates']),'training_queries_consumed':4000,'labeled_pairs_consumed':8000,'source_query_counts_consumed':dict(sorted(counts.items())),'source_count_consumed':len(counts),'sample_sha256':x['dataset_sha256'],'fixture_sha256':x['fixture_sha256'],'environment':x['environment'],'hf_control':x['hf_control'],'evaluations':{s:{'touche_ndcg10_percent':100*e['touche']['ndcg10'],'validation_ordering_percent':100*e['validation']['overall']['ordering_accuracy'],'validation_source_macro_ordering_percent':100*e['validation']['source_macro_ordering_accuracy'],'validation_cross_entropy':e['validation']['pointwise_cross_entropy'],'round_seconds':e['seconds']['round']} for s,e in x['evaluations'].items()}}
(a.run/'verification_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
