"""Filter whole overlength groups while preserving every retained input and target."""
import argparse, array, copy, hashlib, json
from pathlib import Path
from transformers import AutoTokenizer
from distill_runtime import read, digest, save
from prepare_bge_baseline import save as save_prepared
from training_smoke_data import body
from transformers_rerank import PREFIX, SUFFIX

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--root',type=Path,default=Path('/workspace/results/qwen_bge_baseline'))
 p.add_argument('--tokenizer',type=Path,default=Path('/workspace/models/Qwen3-Reranker-0.6B'))
 a=p.parse_args();out=a.root/'document_first_filtered';out.mkdir(parents=True,exist_ok=True)
 parent_path=a.root/'prepared/dataset.json.gz';teacher_path=a.root/'teacher/teacher.json';reference_path=a.root/'query_first_lr1e5/result.json'
 parent=read(parent_path);old_teacher=read(teacher_path);audit=read(a.root/'document_first_length_audit.json')
 assert audit['dataset_sha256']==digest(parent_path)==old_teacher['dataset_sha256']
 assert audit['teacher_sha256']==digest(teacher_path)
 assert old_teacher['scoring_config']['prefix_suffix_source_sha256']==digest(Path(__file__).parent/'transformers_rerank.py')
 tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True);enc=lambda text:tok.encode(text,add_special_tokens=False)
 prefix=enc(PREFIX);suffix=enc(SUFFIX);limit=8192-len(prefix)-len(suffix)
 data=copy.deepcopy(parent);lengths={};removed={}
 for section,details in audit['sections'].items():
  removed[section]=[g['id'] for g in details['removed']];bad=set(removed[section])
  if 'benchmark' in section:assert not bad,'Benchmark membership must stay fixed'
  data[section]=[g for g in parent[section] if g['id'] not in bad]
  for order in ['query_first','document_first']:
   h=hashlib.sha256();tokens=0;pairs=0
   for g in data[section]:
    assert g==next(x for x in parent[section] if x['id']==g['id'])
    for doc in g['documents']:
     raw=enc(body(g['instruction'],g['query'],doc,order));assert len(raw)<=limit
     ids=prefix+raw+suffix;pairs+=1;tokens+=len(ids);h.update(array.array('I',[len(ids)]+ids).tobytes())
   lengths.setdefault(order,{})[section]={'pairs':pairs,'truncated':0,'tokens':tokens,'token_ids_sha256':h.hexdigest()}
  print('FILTERED',section,len(data[section]),flush=True)
 data['derivation']={'kind':'whole_group_length_filter','parent_dataset_sha256':digest(parent_path),'parent_teacher_sha256':digest(teacher_path),'parent_reference_sha256':digest(reference_path),'removed_group_ids':removed,'benchmark_unchanged':True,'policy':'Exclude whole groups with any pair over8192 in either input order; no replacement, repetition, or candidate changes'}
 import collections
 data['distribution']={s:dict(collections.Counter(g['source'] for g in data[s])) for s in ['train','validation']}
 data['sampling']=dict(parent['sampling'],requested_exposures=len(data['train']),unique_query_hashes=len({g['query_hash'] for g in data['train']}),unique_rows=len({g['global_row'] for g in data['train']}))
 save_prepared(out/'dataset.json.gz',data)
 teacher=copy.deepcopy(old_teacher);teacher['dataset_sha256']=digest(out/'dataset.json.gz');teacher['lengths']=lengths['query_first'];teacher['cache_derivation']=data['derivation']
 teacher['scores']={s:{g['id']:old_teacher['scores'][s][g['id']] for g in data[s]} for s in audit['sections']}
 teacher['seconds']={};teacher['cache_reuse_note']='Retained records are byte-for-byte identical objects from the hash-bound parent dataset; original query-first token hashes checked in length audit; no teacher inference repeated.'
 save(out/'teacher.json',teacher)
 counts=[sum(g['id'] not in set(removed['train']) for g in parent['train'][i:i+32]) for i in range(0,len(parent['train']),32)]
 summary={'parent_dataset_sha256':digest(parent_path),'dataset_sha256':digest(out/'dataset.json.gz'),'teacher_sha256':digest(out/'teacher.json'),'removed_group_ids':removed,'retained_groups':{s:len(data[s]) for s in audit['sections']},'original_update_slot_group_counts':counts,'lengths':lengths,'cached_teacher_scores_reused_without_changes':True,'parent_query_first_training_had_additional_groups':len(removed['train']),'schedule_status':'awaiting choice; no training schedule silently substituted'}
 save(out/'preparation.json',summary);print('PREPARED',json.dumps(summary),flush=True)
if __name__=='__main__':main()
