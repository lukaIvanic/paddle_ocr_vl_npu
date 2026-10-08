"""Remove six explicitly authorized families; preserve old update sizes and targets."""
import argparse,array,collections,copy,hashlib,json,math
from pathlib import Path
from distill_runtime import read,digest,save
from prepare_bge_baseline import save as save_data
from bge_teacher_stream import group_signature
from training_smoke_data import body
from transformers_rerank import PREFIX,SUFFIX
EXCLUDED={'hotpotqa','msmarco','mmarco_chinese','dureader','t2ranking','cMedQAv2'}

def select_groups(parent,counts,excluded_groups=()):
 available=[g for g in parent if g['source'] not in EXCLUDED and g['id'] not in excluded_groups]
 slots=[g['id'] for g in parent if int(g['id'].split('/')[1])<len(counts)*32]
 assert len(slots)==sum(counts)<=len(available)
 selected=[]
 for original,slot in zip(available,slots):
  g=copy.deepcopy(original);g['parent_group_id']=g['id'];g['id']=slot;selected.append(g)
 assert [sum(int(g['id'].split('/')[1])//32==i for g in selected) for i in range(len(counts))]==counts
 return selected,available

def lengths(data,tok,order):
 pre=tok.encode(PREFIX,add_special_tokens=False);suf=tok.encode(SUFFIX,add_special_tokens=False);result={}
 for section in ('train','validation','benchmark','reserved_benchmark'):
  h=hashlib.sha256();pairs=tokens=truncated=0
  for offset in range(0,len(data[section]),32):
   gs=data[section][offset:offset+32]
   raws=tok([body(g['instruction'],g['query'],doc,order) for g in gs for doc in g['documents']],add_special_tokens=False)['input_ids']
   for raw in raws:
    truncated+=len(raw)>8192-len(pre)-len(suf)
    ids=pre+raw[:8192-len(pre)-len(suf)]+suf;pairs+=1;tokens+=len(ids);h.update(array.array('I',[len(ids)]+ids).tobytes())
  result[section]={'pairs':pairs,'tokens':tokens,'truncated':truncated,'token_ids_sha256':h.hexdigest()}
 assert not any(x['truncated'] for x in result.values())
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--tokenizer',type=Path,default=Path('/workspace/models/Qwen3-Reranker-0.6B'));p.add_argument('--stop-at',type=int,default=500);p.add_argument('--exclude-groups',type=Path);a=p.parse_args()
 assert not (a.output/'prepared/preparation.json').exists(),'Refuse overwrite of completed preparation'
 parent=read(a.parent/'prepared/dataset.json.gz');cfg=read(a.parent/'run_configuration.json');old=read(a.parent/'student/result.json')
 assert digest(a.parent/'prepared/dataset.json.gz')==cfg['dataset_sha256']==old['dataset_sha256']
 counts=[x['queries'] for x in old['updates'][:a.stop_at]];assert len(counts)==a.stop_at
 exclusions=read(a.exclude_groups) if a.exclude_groups else {'groups':{}}
 if a.exclude_groups:assert exclusions['status']=='approved_by_user'
 data=copy.deepcopy(parent);data['train'],available=select_groups(parent['train'],counts,set(exclusions['groups']))
 data['derivation']={'kind':'family_exclusion','parent_dataset_sha256':cfg['dataset_sha256'],'reference_dataset_sha256':parent['derivation']['reference_dataset_sha256'],'excluded_sources':sorted(EXCLUDED),'benchmark_unchanged':True,'validation_unchanged':True,'policy':'Stable deletion from original uniform draw; no replacement candidates; fill same first-500 batch counts with consecutive retained groups; original IDs in parent_group_id','parent_first500_group_counts':counts}
 data['derivation']['overlap_exclusions_sha256']=digest(a.exclude_groups) if a.exclude_groups else None
 data['derivation']['overlap_excluded_group_ids']=sorted(exclusions['groups'])
 data['distribution']['train']=dict(collections.Counter(g['source'] for g in data['train']))
 data['sampling'].update(requested_exposures=len(data['train']),unique_rows=len({g['global_row'] for g in data['train']}),unique_query_hashes=len({g['query_hash'] for g in data['train']}),family_exclusion=True)
 for s in ('validation','benchmark','reserved_benchmark'):assert data[s]==parent[s]
 a.output.mkdir(parents=True,exist_ok=True)
 if (a.output/'prepared/dataset.json.gz').exists():assert read(a.output/'prepared/dataset.json.gz')==data, 'Partial preparation differs'
 save_data(a.output/'prepared/dataset.json.gz',data)
 save_data(a.output/'prepared/retained_pool.json.gz',available)
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
 token_audit={order:lengths(data,tok,order) for order in ('query_first','document_first')}
 teacher=read(a.parent/'teacher/teacher.json');oldhash=digest(a.parent/'teacher/teacher.json');progress=read(a.parent/'teacher/progress.json')
 assert progress['status']=='completed' and progress['full_token_audit_passed']
 assert progress['teacher_manifest_sha256']==oldhash
 assert teacher['dataset_sha256']==cfg['dataset_sha256']
 assert teacher['scoring_config']['order']=='query_first'
 teacher_model=Path('/workspace/models/Qwen3-Reranker-4B')
 assert teacher['scoring_config']['tokenizer_files']=={f.name:digest(f) for f in teacher_model.glob('*token*') if f.is_file()}
 teacher_tok=AutoTokenizer.from_pretrained(teacher_model,local_files_only=True)
 assert lengths(data,teacher_tok,'query_first')==token_audit['query_first'], 'Teacher/student query-first token streams differ'
 assert teacher['scoring_config']['prefix_suffix_source_sha256']==digest(Path(__file__).parent/'transformers_rerank.py')
 original_by_id={g['id']:g for g in parent['train']};chunkids={g['id']:i//teacher['stream']['groups_per_chunk'] for i,g in enumerate(parent['train'])};cache={};chunk_hashes={};targets={}
 for g in data['train']:
  pid=g['parent_group_id'];index=chunkids[pid]
  if index not in cache:
   path=a.parent/'teacher'/teacher['stream']['directory']/f'{index:06d}.json';ch=read(path)
   assert ch['teacher_manifest_sha256']==oldhash and ch['dataset_sha256']==cfg['dataset_sha256'];assert ch['chunk']==index
   cache[index]=ch;chunk_hashes[str(index)]=digest(path)
  original=original_by_id[pid];ch=cache[index]
  assert ch['group_signatures'][pid]==group_signature(original)
  assert all(g[k]==original[k] for k in ('query','instruction','documents','candidate_origins'))
  values=ch['scores'][pid];assert len(values)==8 and all(math.isfinite(v) for v in values)
  targets[g['id']]=values
 teacher.pop('stream');teacher['scores']['train']=targets;teacher['dataset_sha256']=digest(a.output/'prepared/dataset.json.gz');teacher['lengths']=token_audit['query_first']
 teacher['cache_derivation']={'parent_teacher_sha256':oldhash,'chunk_sha256':chunk_hashes,'exact_group_signatures_verified':len(targets),'candidate_contents_unchanged':True,'all_parent_tokens_audited':True}
 (a.output/'teacher').mkdir(parents=True,exist_ok=True)
 save(a.output/'teacher/teacher.json',teacher)
 summary={'dataset_sha256':teacher['dataset_sha256'],'teacher_sha256':digest(a.output/'teacher/teacher.json'),'parent_sha256':cfg['dataset_sha256'],'excluded_sources':sorted(EXCLUDED),'original_groups':len(parent['train']),'retained_available':len(available),'scheduled_groups':len(data['train']),'unique_queries':len({g['query_hash'] for g in data['train']}),'group_counts':counts,'source_counts':data['distribution']['train'],'lengths':token_audit,'teacher_cache_verification':teacher['cache_derivation']}
 save(a.output/'prepared/preparation.json',summary)
 print('PREPARED',json.dumps({k:v for k,v in summary.items() if k not in ('group_counts','teacher_cache_verification')},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
