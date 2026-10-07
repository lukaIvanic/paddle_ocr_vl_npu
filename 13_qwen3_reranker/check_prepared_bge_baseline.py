"""Verify scheduled candidates against the original records and pinned sampler."""
import argparse, collections, random, types
from pathlib import Path
from datasets import load_dataset
from bge_baseline import reference_dataset_class
from distill_runtime import read, save, digest

def main():
 p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--extracted',type=Path,required=True);a=p.parse_args()
 data=read(a.prepared/'dataset.json.gz');audit=read(a.prepared/'preparation.json');blocked=set(read(a.prepared/'query_audit.json')['blocked_normalized_query_hashes'])
 assert digest(a.prepared/'dataset.json.gz')==audit['dataset_sha256']
 assert all(f['cap_removed']==0 for f in audit['files']), 'Handle capped row-index mapping before verification'
 groups=data['validation']+data['train'];byfile=collections.defaultdict(list)
 for g in groups:byfile[g['file']].append(g)
 raw={}
 for file,gs in byfile.items():
  ds=load_dataset('json',data_files=str(a.extracted/file),split='train',cache_dir=str(a.prepared/'arrow_cache'))
  for g in gs:raw[g['id']]=ds[g['row_after_cap']]
 Base=reference_dataset_class();ref=object.__new__(Base)
 ref.args=types.SimpleNamespace(train_group_size=8,shuffle_ratio=0.0,query_instruction_for_rerank=None,passage_instruction_for_rerank=None,knowledge_distillation=False)
 ref.create_one_example=lambda q,d:d
 random.seed(data['sampling']['candidate_seed'])
 for g in groups:
  r=raw[g['id']];assert r['query']==g['query'];assert g['query_hash'] not in blocked
  assert [r[o['pool']][o['index']] for o in g['candidate_origins']]==g['documents']
  ref.dataset=[r];assert ref[0][0]==g['documents'],g['id']
 assert not ({g['query_hash'] for g in data['train']}&{g['query_hash'] for g in data['validation']})
 examples=[data['train'][0]]
 for predicate in [lambda g:g['pool_sizes']['neg']<7,lambda g:g['pool_sizes']['pos']>1]:
  found=next((g for g in data['train'] if predicate(g) and g not in examples),None)
  if found:examples.append(found)
 for g in data['train']:
  if len(examples)>=3:break
  if g['source'] not in {x['source'] for x in examples}:examples.append(g)
 result={'passed':True,'groups_checked':len(groups),'original_files_checked':len(byfile),'upstream_sampler_replayed_for_every_occurrence':True,'candidate_text_and_origin_exact':True,'train_validation_normalized_query_disjoint':True,'training_short_pool_groups':sum(g['pool_sizes']['neg']<7 for g in data['train']),'examples':[{k:v for k,v in g.items() if k!='documents'}|{'candidate_text_prefixes':[d[:160] for d in g['documents']]} for g in examples]}
 save(a.prepared/'prepared_check.json',result);print('PREPARED_CHECK',result,flush=True)
if __name__=='__main__':main()
