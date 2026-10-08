"""Append audited supplied BGE groups without changing the first 500 updates."""
import argparse,collections,copy,json,math
from pathlib import Path
from distill_runtime import read,digest,save
from prepare_bge_baseline import save as save_data
from prepare_bge_family_filter import EXCLUDED,lengths
from training_smoke_data import body
from transformers_rerank import PREFIX,SUFFIX

def flags(folder):
 s=read(folder/'summary.json');assert s['status']=='completed' and len(s['tasks'])==18
 hits={}
 for task in s['tasks']:
  x=read(folder/f'{task}.json')
  for h in x['query_neighbors']:hits.setdefault(h['train_group'],[]).append({'task':task,'kind':'query','cosine':h['cosine']})
  for kind in ('exact_documents','near_documents'):
   for h in x[kind]:
    if kind=='exact_documents' and not any(c.isalnum() for c in h['text']):continue
    for o in h['origins']:hits.setdefault(o['id'],[]).append({'task':task,'kind':kind})
 return hits

def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['pool','exclude','finalize']);p.add_argument('--root',type=Path,required=True);p.add_argument('--audit',type=Path);p.add_argument('--exclusions',type=Path);a=p.parse_args();r=a.root
 base=Path('/workspace/results/qwen_bge_family_overlap_filtered_500_20261008');parent=Path('/workspace/results/qwen_bge_document_first_10h')
 old=read(base/'prepared/dataset.json.gz');oldpool=read(base/'prepared/retained_pool.json.gz');oldprep=read(base/'prepared/preparation.json')
 if a.stage=='exclude':
  found=flags(a.audit);excluded=read(a.exclusions)['groups'] if a.exclusions and a.exclusions.exists() else {}
  for k,v in found.items():excluded.setdefault(k,[]).extend(v)
  out={'status':'approved_by_user','authorization':'User approved excluding flagged groups and re-auditing; same rule applied to the authorized appended data.','groups':excluded,'last_audit_sha256':digest(a.audit/'summary.json'),'new_flags':len(found)}
  save(r/'append_exclusions.json',out);print(json.dumps({'excluded':len(excluded),'new_flags':len(found)}));return
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained('/workspace/models/Qwen3-Reranker-0.6B',local_files_only=True)
 if a.stage=='pool':
  raw=read(r/'raw/dataset.json.gz');originalraw=read(parent/'raw/dataset.json.gz')
  assert raw['train'][:len(originalraw['train'])]==originalraw['train']
  sourceaudit=read(r/'raw/preparation.json');oldsource=read(parent/'raw/preparation.json')
  for k in ('file_list','cap','full_source_distribution','eligible_training_rows','files'):assert sourceaudit[k]==oldsource[k],k
  candidates=[g for g in raw['train'][len(originalraw['train']):] if g['source'] not in EXCLUDED]
  limit=8192-len(tok.encode(PREFIX,add_special_tokens=False))-len(tok.encode(SUFFIX,add_special_tokens=False));kept=[];removed=[]
  for i in range(0,len(candidates),32):
   batch=candidates[i:i+32];enc={o:tok([body(g['instruction'],g['query'],d,o) for g in batch for d in g['documents']],add_special_tokens=False)['input_ids'] for o in ('query_first','document_first')}
   for j,g in enumerate(batch):
    if any(len(ids)>limit for rows in enc.values() for ids in rows[j*8:j*8+8]):removed.append(g['id'])
    else:kept.append(g)
   if i%2048==0:print('APPEND_LENGTH',i,len(candidates),len(removed),flush=True)
  pool=copy.deepcopy(old);pool['train']=kept
  save_data(r/'append_pool.json.gz',pool)
  save(r/'append_pool_preparation.json',{'raw_dataset_sha256':digest(r/'raw/dataset.json.gz'),'append_pool_sha256':digest(r/'append_pool.json.gz'),'raw_prefix_equal':True,'file_list_cap_and_counts_equal':True,'new_groups_before_length_filter':len(candidates),'new_groups':len(kept),'length_removed_ids':removed,'excluded_sources':sorted(EXCLUDED)})
  print('APPEND_POOL',len(kept),flush=True);return
 assert a.audit and a.exclusions and not flags(a.audit)
 audit=read(a.audit/'summary.json');assert audit['parent_sha256']==digest(r/'append_pool.json.gz') and audit['excluded_groups_sha256']==digest(a.exclusions)
 excluded=read(a.exclusions)['groups'];append=[g for g in read(r/'append_pool.json.gz')['train'] if g['id'] not in excluded]
 assert len(append)==audit['retained_groups']
 orig=read(parent/'prepared/dataset.json.gz');slots=[g['id'] for g in orig['train']]
 assert len(slots)==56537 and max(int(s.split('/')[1])//32 for s in slots)==1799
 selected=oldpool+append;assert len(selected)>=len(slots),(len(selected),len(slots))
 assert len({g['global_row'] for g in selected})==len(selected)
 train=[]
 for slot,g in zip(slots,selected):
  g=copy.deepcopy(g);g['parent_group_id']=g['id'];g['id']=slot;train.append(g)
 assert train[:len(old['train'])]==old['train']
 data=copy.deepcopy(old);data['train']=train
 data['derivation'].update(extension=True,prefix_dataset_sha256=digest(base/'prepared/dataset.json.gz'),append_pool_sha256=digest(r/'append_pool.json.gz'),append_exclusions_sha256=digest(a.exclusions),policy='Exact existing first500 prefix; continue stable deletion of original uniform draw; append prefix-verified uniform extension with same sampler and filters. No repeated rows or invented candidates.')
 data['distribution']['train']=dict(collections.Counter(g['source'] for g in train));data['sampling'].update(requested_exposures=len(train),unique_rows=len(train),unique_query_hashes=len({g['query_hash'] for g in train}))
 save_data(r/'prepared/dataset.json.gz',data)
 token_audit={o:lengths(data,tok,o) for o in ('query_first','document_first')}
 from bge_filtered_runtime import update_windows
 counts=list(map(len,update_windows(train,1800,32,'retained_original_slots')))
 assert counts[:500]==oldprep['group_counts']
 review={'dataset_sha256':digest(r/'prepared/dataset.json.gz'),'prefix_dataset_sha256':digest(base/'prepared/dataset.json.gz'),'prefix_audit_review_sha256':digest(base/'audit_review.json'),'prefix_audited_pool_groups':len(oldpool),'append_audit_summary_sha256':digest(a.audit/'summary.json'),'append_audit_path':str(a.audit),'append_task_sha256':{t:digest(a.audit/f'{t}.json') for t in audit['tasks']},'append_exclusions_sha256':digest(a.exclusions),'append_excluded_groups':len(excluded),'append_audited_pool_groups':len(append),'substantive_detected_append_matches':0,'coverage':audit['tasks'],'limitations':audit['methods']['limitations'],'authorization':'User requested both training runs through1800 and expansion after unchanged update500; existing approved family, length and overlap rules preserved.'}
 save(r/'audit_review.json',review)
 save(r/'prepared/preparation.json',{'dataset_sha256':digest(r/'prepared/dataset.json.gz'),'lengths':token_audit,'group_counts':counts,'groups':len(train),'unique_queries':len({g['query_hash'] for g in train}),'source_counts':data['distribution']['train'],'first500_exact':True,'audit_review_sha256':digest(r/'audit_review.json'),'prefix_suffix_source_sha256':digest(Path(__file__).parent/'transformers_rerank.py')})
 print('PREPARED_EXTENSION',len(train),'groups',flush=True)
if __name__=='__main__':main()
