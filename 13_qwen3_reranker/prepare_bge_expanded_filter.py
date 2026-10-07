"""Audit and filter a larger draw from exactly the existing BGE preparation recipe."""
import argparse,array,collections,copy,hashlib,json,time
from pathlib import Path
from distill_runtime import read,digest,save
from prepare_bge_baseline import save as save_prepared
from bge_filtered_runtime import update_windows
from training_smoke_data import body
from transformers_rerank import PREFIX,SUFFIX

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--reference-root',type=Path,default=Path('/workspace/results/qwen_bge_baseline'))
 p.add_argument('--tokenizer',type=Path,default=Path('/workspace/models/Qwen3-Reranker-0.6B'))
 p.add_argument('--steps',type=int,default=1800);a=p.parse_args()
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
 pre=tok.encode(PREFIX,add_special_tokens=False);suf=tok.encode(SUFFIX,add_special_tokens=False);limit=8192-len(pre)-len(suf)
 data=read(a.input);old=read(a.reference_root/'prepared/dataset.json.gz');filtered=read(a.reference_root/'document_first_filtered/dataset.json.gz')
 audit=read(a.input.parent/'preparation.json');old_audit=read(a.reference_root/'prepared/preparation.json')
 for k in ('reference','file_list','cap','full_source_distribution','eligible_training_rows'):
  assert audit[k]==old_audit[k], ('Preparation scope changed',k)
 for section in ('validation','benchmark','reserved_benchmark'):assert data[section]==old[section],section
 assert len(data['train'])==a.steps*32
 assert data['sampling']['unique_rows']==len(data['train']), 'No repeated rows in this expanded pass'
 a.output.mkdir(parents=True,exist_ok=True)
 lengths={o:{} for o in ('query_first','document_first')};removed={};started=time.monotonic()
 for section in ('train','validation','benchmark','reserved_benchmark'):
  retained=[];removed[section]=[];hashes={o:hashlib.sha256() for o in lengths};tokens=collections.Counter();pairs=0
  raw_groups=data[section]
  for offset in range(0,len(raw_groups),32):
   batch=raw_groups[offset:offset+32]
   encoded={o:tok([body(g['instruction'],g['query'],doc,o) for g in batch for doc in g['documents']],add_special_tokens=False)['input_ids'] for o in lengths}
   cursor=0
   for g in batch:
    n=len(g['documents']);spans={o:rows[cursor:cursor+n] for o,rows in encoded.items()};cursor+=n
    if any(len(ids)>limit for rows in spans.values() for ids in rows):
     removed[section].append({'id':g['id'],'source':g.get('source',g.get('task')),'max_body_tokens':{o:max(map(len,rows)) for o,rows in spans.items()}});continue
    retained.append(g);pairs+=n
    for o,rows in spans.items():
     for raw in rows:
      ids=pre+raw+suf;tokens[o]+=len(ids);hashes[o].update(array.array('I',[len(ids)]+ids).tobytes())
   if offset%1024==0:print('LENGTH_PROGRESS',json.dumps({'section':section,'groups_checked':min(offset+32,len(raw_groups)),'total':len(raw_groups),'removed':len(removed[section]),'seconds':time.monotonic()-started}),flush=True)
  data[section]=retained
  for o in lengths:lengths[o][section]={'pairs':pairs,'truncated':0,'tokens':tokens[o],'token_ids_sha256':hashes[o].hexdigest()}
  if section!='train':assert data[section]==filtered[section],('Fixed held-out inputs changed',section)
 data['derivation']={'kind':'expanded_bge_length_filter','raw_sample_sha256':digest(a.input),'reference_dataset_sha256':digest(a.reference_root/'prepared/dataset.json.gz'),'fixed_filtered_dataset_sha256':digest(a.reference_root/'document_first_filtered/dataset.json.gz'),'benchmark_unchanged':True,'validation_unchanged':True,'removed_group_ids':{s:[x['id'] for x in v] for s,v in removed.items()},'policy':'Same upstream preparation; larger uniform row draw; whole-group length filter in both orders; retain original update slots; no replacement or invented candidates'}
 data['distribution']={s:dict(collections.Counter(g['source'] for g in data[s])) for s in ('train','validation')}
 data['sampling'].update(requested_exposures=len(data['train']),raw_sample_exposures=a.steps*32,unique_query_hashes=len({g['query_hash'] for g in data['train']}),unique_rows=len({g['global_row'] for g in data['train']}))
 windows=update_windows(data['train'],a.steps,32,'retained_original_slots')
 save_prepared(a.output/'dataset.json.gz',data)
 schedule=[{k:v for k,v in g.items() if k not in ('query','instruction','documents')} for g in data['train']]
 save_prepared(a.output/'schedule.json.gz',schedule)
 summary={'dataset_sha256':digest(a.output/'dataset.json.gz'),'schedule_sha256':digest(a.output/'schedule.json.gz'),'lengths':lengths,'removed':removed,'retained_groups':{s:len(data[s]) for s in removed},'source_distribution':data['distribution'],'sampling':data['sampling'],'group_counts_per_update':list(map(len,windows)),'seconds':time.monotonic()-started,'tokenizer_files':{f.name:digest(f) for f in a.tokenizer.glob('*token*') if f.is_file()},'prefix_suffix_source_sha256':digest(Path(__file__).parent/'transformers_rerank.py')}
 save(a.output/'preparation.json',summary)
 print('PREPARED',json.dumps({k:summary[k] for k in ('dataset_sha256','retained_groups','sampling','seconds')}),flush=True)
if __name__=='__main__':main()
