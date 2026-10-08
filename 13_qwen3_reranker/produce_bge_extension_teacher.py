"""Complete frozen teacher cache for a prefix-preserving filtered extension."""
import argparse,copy,json,math,time,os
from pathlib import Path
from distill_runtime import Runtime,read,digest,save,model_manifest
from bge_teacher_stream import group_signature

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
 (r/'teacher/chunks').mkdir(parents=True,exist_ok=True)
 data=read(r/'prepared/dataset.json.gz');prep=read(r/'prepared/preparation.json');oldroot=Path('/workspace/results/qwen_bge_document_first_10h');old=read(oldroot/'prepared/dataset.json.gz');manifest=read(oldroot/'teacher/teacher.json');progress=read(oldroot/'teacher/progress.json')
 assert progress['status']=='completed' and progress['full_token_audit_passed']
 assert progress['teacher_manifest_sha256']==digest(oldroot/'teacher/teacher.json')
 cache={};chunk_hashes={}
 for i in range(math.ceil(len(old['train'])/manifest['stream']['groups_per_chunk'])):
  path=oldroot/'teacher'/manifest['stream']['directory']/f'{i:06d}.json';chunk=read(path);chunk_hashes[str(i)]=digest(path)
  assert chunk['teacher_manifest_sha256']==progress['teacher_manifest_sha256']
  for g in old['train'][i*manifest['stream']['groups_per_chunk']:(i+1)*manifest['stream']['groups_per_chunk']]:
   sig=group_signature(g);assert sig==chunk['group_signatures'][g['id']];cache[sig]=chunk['scores'][g['id']]
 modeldir=Path('/workspace/models/Qwen3-Reranker-4B');runtime=Runtime(modeldir)
 assert model_manifest(modeldir)==manifest['model_sha256']
 assert {f.name:digest(f) for f in modeldir.glob('*token*') if f.is_file()}==manifest['scoring_config']['tokenizer_files']
 assert digest(Path(__file__).parent/'transformers_rerank.py')==manifest['scoring_config']['prefix_suffix_source_sha256']
 records={s:runtime.records(data[s],s) for s in ('train','validation','benchmark','reserved_benchmark')}
 assert runtime.lengths==prep['lengths']['query_first']
 base=Path('/workspace/results/qwen_bge_family_overlap_filtered_500_20261008');oldfiltered=read(base/'prepared/dataset.json.gz');oldtargets=read(base/'teacher/teacher.json')
 for s in ('validation','benchmark','reserved_benchmark'):assert data[s]==oldfiltered[s]
 out={k:copy.deepcopy(manifest[k]) for k in ('model_sha256','scoring','scoring_config')}
 out.update(dataset_sha256=prep['dataset_sha256'],lengths=runtime.lengths,scores={s:copy.deepcopy(oldtargets['scores'][s]) for s in ('validation','benchmark','reserved_benchmark')})
 out['scores']['train']={};out['cache_derivation']={'parent_manifest_sha256':progress['teacher_manifest_sha256'],'chunk_sha256':chunk_hashes,'policy':'Exact group signatures and model/tokenizer/scoring configuration; new selected groups scored by same frozen query-first teacher'}
 model=None;started=time.monotonic();cached_count=fresh_count=0
 for offset in range(0,len(data['train']),256):
  gs=data['train'][offset:offset+256];signatures={g['id']:group_signature(g) for g in gs};path=r/'teacher/chunks'/f'{offset//256:06d}.json'
  if path.exists():
   ch=read(path);assert ch['dataset_sha256']==prep['dataset_sha256'] and ch['signatures']==signatures
   scores=ch['scores'];cached=ch['cached'];fresh=ch['fresh']
  else:
   lookup={g['id']:group_signature(dict(g,id=g['parent_group_id'])) for g in gs}
   scores={g['id']:cache[lookup[g['id']]] for g in gs if lookup[g['id']] in cache};cached=len(scores);fresh=len(gs)-cached
   if fresh:
    if model is None:model=runtime.load(modeldir)
    rows=[row for row in records['train'][offset*8:(offset+len(gs))*8] if row['group_id'] not in scores]
    values,seconds=runtime.score(model,rows,f'append_chunk_{offset//256}');scores.update(values)
   assert set(scores)==set(signatures) and all(len(v)==8 and all(math.isfinite(x) for x in v) for v in scores.values())
   save(path,{'dataset_sha256':prep['dataset_sha256'],'signatures':signatures,'scores':scores,'cached':cached,'fresh':fresh})
  out['scores']['train'].update(scores);cached_count+=cached;fresh_count+=fresh
  status={'status':'running','completed_groups':offset+len(gs),'total_groups':len(data['train']),'cached':cached_count,'fresh':fresh_count,'seconds':time.monotonic()-started,'physical_npu':os.getenv('ASCEND_RT_VISIBLE_DEVICES'),'full_token_audit_passed':True}
  save(r/'teacher/progress.json',status);print('TEACHER_PROGRESS',json.dumps(status),flush=True)
 assert all(out['scores']['train'][k]==v for k,v in oldtargets['scores']['train'].items())
 save(r/'teacher/teacher.json',out);status.update(status='completed',teacher_manifest_sha256=digest(r/'teacher/teacher.json'));save(r/'teacher/progress.json',status)
 print('TEACHER_COMPLETE',json.dumps(status),flush=True)
if __name__=='__main__':main()
