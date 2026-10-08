"""Verify the completed overlap re-audit against the actual prepared training pool."""
import argparse,collections,json
from pathlib import Path
from distill_runtime import read,digest,save
from prepare_bge_family_filter import EXCLUDED

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
 (r/'audit_review.json').unlink(missing_ok=True)
 prep=read(r/'prepared/preparation.json');data=read(r/'prepared/dataset.json.gz');pool=read(r/'prepared/retained_pool.json.gz');ex=read(r/'approved_exclusions.json');audit=read(r/'audit/summary.json')
 assert ex['status']=='approved_by_user'
 assert audit['status']=='completed' and len(audit['tasks'])==18
 assert audit['excluded_groups_sha256']==digest(r/'approved_exclusions.json')==data['derivation']['overlap_exclusions_sha256']
 assert audit['parent_sha256']==prep['parent_sha256']
 assert digest(r/'prepared/dataset.json.gz')==prep['dataset_sha256']
 assert len(pool)==audit['retained_groups'] and len(data['train'])==audit['scheduled_groups']==15690
 assert [g['id'] for g in pool[:15690]]==[g['parent_group_id'] for g in data['train']]
 assert not {g['id'] for g in pool}&set(ex['groups'])
 assert not {g['source'] for g in pool}&EXCLUDED
 punctuation=[];coverage={};hashes={};flags=[]
 for task in audit['tasks']:
  f=r/'audit'/f'{task}.json';x=read(f);hashes[task]=digest(f)
  flags += [{'task':task,'kind':'query','match':v} for v in x['query_neighbors']]
  flags += [{'task':task,'kind':'near_document','match':v} for v in x['near_documents']]
  for v in x['exact_documents']:
   if any(c.isalnum() for c in v['text']):flags.append({'task':task,'kind':'exact_document','match':v})
   else:punctuation.append({'task':task,'text':v['text'],'origins':v['origins']})
  coverage[task]={'queries':x['evaluation_queries'],'candidate_document_ids':x['candidate_document_ids'],'candidate_sha256':x['candidate_sha256']}
 assert sum(x['queries'] for x in coverage.values())==48560
 if flags:
  save(r/'unresolved_audit_flags.json',flags)
  raise RuntimeError(f'{len(flags)} substantive audit flags remain; training gate closed')
 review={'launch_authorized':True,'authorization':'User requested six-family removal and fresh training to update500 with horizon1800, then approved excluding flagged groups and re-auditing. No recipe changes.','dataset_sha256':prep['dataset_sha256'],'audit_summary_sha256':digest(r/'audit/summary.json'),'task_audit_sha256':hashes,'excluded_sources':sorted(EXCLUDED),'excluded_groups':len(ex['groups']),'audited_pool_groups':len(pool),'training_groups':len(data['train']),'unique_training_queries':prep['unique_queries'],'source_counts':dict(collections.Counter(g['source'] for g in data['train'])),'coverage':coverage,'substantive_detected_matches':0,'punctuation_only_equalities':punctuation,'methods':audit['methods'],'interpretation':'No remaining substantive overlap detected by these exact/lexical checks. This does not certify absence of semantic/translation leakage or audit the original Qwen pretraining.'}
 save(r/'audit_review.json',review)
 print(json.dumps({k:v for k,v in review.items() if k not in ['coverage','task_audit_sha256','source_counts','punctuation_only_equalities','methods']},ensure_ascii=False))
if __name__=='__main__':main()
