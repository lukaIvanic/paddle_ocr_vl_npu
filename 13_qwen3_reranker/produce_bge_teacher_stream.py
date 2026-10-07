"""Score immutable candidate groups ahead of training using the frozen query-first 4B."""
import argparse,array,copy,hashlib,json,os,time
from pathlib import Path
from distill_runtime import Runtime,read,digest,model_manifest,save
from bge_teacher_stream import group_signature

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--dataset',type=Path,required=True);p.add_argument('--preparation',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--model',type=Path,default=Path('/workspace/models/Qwen3-Reranker-4B'))
 p.add_argument('--reference-root',type=Path,default=Path('/workspace/results/qwen_bge_baseline'))
 p.add_argument('--chunk-groups',type=int,default=256);a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=True);chunks=a.output/'chunks';chunks.mkdir(exist_ok=True)
 data=read(a.dataset);prep=read(a.preparation);old=read(a.reference_root/'document_first_filtered/dataset.json.gz');cached=read(a.reference_root/'document_first_filtered/teacher.json')
 assert digest(a.dataset)==prep['dataset_sha256']
 runtime=Runtime(a.model);fingerprint=model_manifest(a.model)
 assert fingerprint==cached['model_sha256']
 assert {f.name:digest(f) for f in a.model.glob('*token*') if f.is_file()}==cached['scoring_config']['tokenizer_files']
 assert digest(Path(__file__).parent/'transformers_rerank.py')==cached['scoring_config']['prefix_suffix_source_sha256']==prep['prefix_suffix_source_sha256']
 for section in ('validation','benchmark','reserved_benchmark'):
  assert data[section]==old[section]
  runtime.records(data[section],section)
  assert runtime.lengths[section]==cached['lengths'][section]==prep['lengths']['query_first'][section]
 teacher={k:copy.deepcopy(cached[k]) for k in ('model_sha256','scoring','scoring_config')}
 teacher.update(dataset_sha256=digest(a.dataset),lengths=prep['lengths']['query_first'],scores={s:cached['scores'][s] for s in ('validation','benchmark','reserved_benchmark')},stream={'directory':'chunks','groups_per_chunk':a.chunk_groups,'total_groups':len(data['train']),'policy':'Query-first frozen teacher, microbatch16/tokenbudget16384; length sorting within each chunk; publish complete chunks atomically'},evaluation_cache_parent_sha256=digest(a.reference_root/'document_first_filtered/teacher.json'))
 old_train = {g['id']:g for g in old['train']}
 reusable = {g['id']:cached['scores']['train'][g['id']] for g in data['train'] if g['id'] in old_train and group_signature(g)==group_signature(old_train[g['id']])}
 teacher['stream']['verified_cached_training_groups'] = len(reusable)
 teacher['stream']['cache_reuse_policy'] = 'Exact query, instruction, ordered documents, model weights, tokenizer, prompt source and scoring configuration match'
 manifest=a.output/'teacher.json'
 if manifest.exists():assert read(manifest)==teacher
 else:save(manifest,teacher)
 manifest_sha=digest(manifest)
 model=runtime.load(a.model);started=time.monotonic();h=hashlib.sha256();tokens=0;pairs=0
 progress={'status':'running','physical_npu':os.getenv('ASCEND_RT_VISIBLE_DEVICES'),'teacher_manifest_sha256':manifest_sha,'dataset_sha256':digest(a.dataset),'total_groups':len(data['train']),'completed_groups':0,'chunks':[]}
 save(a.output/'progress.json',progress)
 try:
  for offset in range(0,len(data['train']),a.chunk_groups):
   index=offset//a.chunk_groups;groups=data['train'][offset:offset+a.chunk_groups]
   rows=runtime.records(groups,'train_chunk')
   assert runtime.lengths['train_chunk']['truncated']==0
   for row in rows:
    ids=row['ids'];h.update(array.array('I',[len(ids)]+ids).tobytes());pairs+=1;tokens+=len(ids)
   target=chunks/f'{index:06d}.json';signatures={g['id']:group_signature(g) for g in groups}
   if target.exists():
    previous=read(target);assert previous['teacher_manifest_sha256']==manifest_sha and previous['group_signatures']==signatures
    scores=previous['scores'];seconds=0
   else:
    scores = {g['id']:reusable[g['id']] for g in groups if g['id'] in reusable}
    fresh_rows = [row for row in rows if row['group_id'] not in scores]
    fresh,seconds = runtime.score(model,fresh_rows,f'train_chunk_{index}') if fresh_rows else ({},0)
    scores.update(fresh)
    assert set(scores)==set(signatures)
    save(target,{'chunk':index,'teacher_manifest_sha256':manifest_sha,'dataset_sha256':teacher['dataset_sha256'],'group_signatures':signatures,'scores':scores,'seconds':seconds,'cached_groups_reused':sum(g['id'] in reusable for g in groups),'token_metadata':runtime.lengths['train_chunk']})
   progress['completed_groups']=offset+len(groups);progress['seconds']=time.monotonic()-started
   progress['chunks'].append({'chunk':index,'sha256':digest(target),'groups':len(groups),'seconds':seconds})
   save(a.output/'progress.json',progress)
   print('TEACHER_CHUNK',json.dumps(progress['chunks'][-1]|{'completed_groups':progress['completed_groups'],'total_seconds':progress['seconds']}),flush=True)
  actual={'pairs':pairs,'tokens':tokens,'truncated':0,'token_ids_sha256':h.hexdigest()}
  assert actual==teacher['lengths']['train'],('Full token audit mismatch',actual,teacher['lengths']['train'])
  progress['status']='completed';progress['full_token_audit_passed']=True;save(a.output/'progress.json',progress)
 except Exception as error:
  save(a.output/'failure.json',{'error':repr(error)});raise
if __name__=='__main__':main()
