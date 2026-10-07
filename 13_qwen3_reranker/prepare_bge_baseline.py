"""Prepare a bounded, globally sampled run from the full pinned BGE release.

No source/language quotas, alternative corpora, or invented candidates. The
unmodified pinned loader supplies per-file cap and concatenation behavior.
"""
import argparse,collections,gzip,hashlib,json,pathlib,random,sys,tarfile,time,types
from bge_baseline import PIN,reference_dataset_class,candidate_indices
from training_smoke_data import normalize,body
from distill_runtime import read,save as save_json,digest
def save(path,data):
 if str(path).endswith(".gz"):
  path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
  path.write_bytes(gzip.compress(json.dumps(data,ensure_ascii=False).encode(),mtime=0))
 else:save_json(path,data)
DEFAULT_INSTRUCTION='Given a web search query, retrieve relevant passages that answer the query'
def key(t):return normalize(t)
def qhash(t):return hashlib.sha256(key(t).encode()).hexdigest()
def extract(archive,root):
 marker=root/'extracted.json'
 if marker.exists():return read(marker)
 assert archive.stat().st_size==PIN['archive']['size'] and digest(archive)==PIN['archive']['lfs']['oid']
 root.mkdir(parents=True,exist_ok=True);items=[]
 with tarfile.open(archive,mode='r|gz') as tf:
  for m in tf:
   if m.isdir():continue
   if not m.isfile():raise ValueError('Unexpected non-file tar member: '+m.name)
   path=root/m.name
   assert path.resolve().is_relative_to(root.resolve())
   path.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256()
   with tf.extractfile(m) as src,path.open('wb') as dst:
    while b:=src.read(8*1024**2):dst.write(b);h.update(b)
   items.append({'path':m.name,'bytes':m.size,'sha256':h.hexdigest()})
   print('EXTRACTED',m.name,m.size,flush=True)
 save(marker,items);return items

def blocklist(args):
 from datasets import load_dataset
 root=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'22_qwen3_embedding_benchmark'))
 from suite_protocol import ENGLISH
 panels=read(args.panels);blocked=set();coverage={};eval_ids={}
 for name,meta in panels['provenance'].items():
  if name in ENGLISH:
   repo,rev,*_=ENGLISH[name];ds=load_dataset(repo,'queries',revision=rev);qs=next(iter(ds.values()))
  else:
   repo=meta['dataset']['path'];rev=meta['dataset']['revision'];qs=load_dataset(repo,'default',revision=rev)['queries']
  if name in ENGLISH:
   judgments=load_dataset(repo,'default',revision=rev)[meta['split']];qidcol='query-id'
   qrel_repo=repo;qrel_rev=rev
  else:
   qrel_repo=repo+'-qrels';qrel_rev=meta['dataset']['qrel_revision']
   judgments=load_dataset(qrel_repo,'default',revision=qrel_rev)[meta['split']];qidcol='qid'
  target_ids={str(i) for i in judgments[qidcol]};idcol='_id' if '_id' in qs.column_names else 'id'
  target_queries={str(r[idcol]):r['text'] for r in qs if str(r[idcol]) in target_ids}
  assert set(target_queries)==target_ids,(name,len(target_queries),len(target_ids))
  blocked.update(qhash(q) for q in target_queries.values());eval_ids[name]=target_ids
  coverage[name]={'path':repo,'revision':rev,'target_split':meta['split'],'query_table_rows':len(qs),'evaluation_queries_checked':len(target_queries),'qrels_path':qrel_repo,'qrels_revision':qrel_rev,'coverage':'every query ID in the complete pinned target evaluation judgments'}

 for path in args.heldout:
  data=read(path);gs=data['validation'];blocked.update(qhash(g['query']) for g in gs)
  coverage['heldout:'+str(path)]={'queries':len(gs),'sha256':digest(path)}
 # Official mMARCO IDs link English and both released Chinese translations.
 from audit_missing_sources import tsv_queries
 translations={};known_ids=set()
 for path in sorted(args.audit_root.glob('*-*-*.tsv')):
  if path.name.startswith(('google-','helsinki-')):
   qs=tsv_queries(path);translations[path.name]=qs
   known_ids.update(i for i,q in qs.items() if qhash(q) in blocked)
 for qs in translations.values():blocked.update(qhash(q) for i,q in qs.items() if i in known_ids)
 coverage['translation_links']={'assets':{str(args.audit_root/k):digest(args.audit_root/k) for k in translations},'blocked_original_query_ids':len(known_ids)}
 save(args.output/'query_audit.json',{'coverage':coverage,'blocked_normalized_query_hashes':sorted(blocked),'normalization':'Unicode NFKC, casefold, whitespace collapse','limitations':['No exhaustive semantic/paraphrase overlap claim','Original record IDs used only where namespace is established; no matching unrelated numeric IDs','Negative-miner per-row provenance not supplied upstream; retained as upstream limitation']})
 return blocked,coverage,translations,known_ids

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--archive',type=pathlib.Path,default='/workspace/results/qwen_bge_baseline/release/bge-m3-data.tar.gz')
 p.add_argument('--extracted',type=pathlib.Path,default='/workspace/results/qwen_bge_baseline/release/extracted')
 p.add_argument('--output',type=pathlib.Path,required=True)
 p.add_argument('--panels',type=pathlib.Path,default='/workspace/results/qwen_margin_distill/data_fast/benchmark_panels.json.gz')
 p.add_argument('--heldout',type=pathlib.Path,nargs='*',default=[pathlib.Path('/workspace/results/qwen_margin_distill/data_fast/dataset.json.gz')])
 p.add_argument('--audit-root',type=pathlib.Path,default='/workspace/results/qwen_margin_distill/missing_source_audit')
 p.add_argument('--tokenizer',type=pathlib.Path,default='/workspace/models/Qwen3-Reranker-0.6B')
 p.add_argument('--seed',type=int,default=1047);p.add_argument('--steps',type=int,default=50);p.add_argument('--queries-per-update',type=int,default=32);p.add_argument('--validation-queries',type=int,default=64)
 p.add_argument('--defer-length-audit',action='store_true',help='Defer token-length audit to the subsequent mandatory two-order whole-group filter')
 p.add_argument('--max-example-num-per-dataset',type=int,default=100000000)
 p.add_argument('--stage',choices=['audit','extract','prepare'],default='prepare');args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
 if args.stage=='audit':blocklist(args);return
 assets=extract(args.archive,args.extracted)
 if args.stage=='extract':return
 blocked,coverage,translations,blocked_ids=blocklist(args)
 files=sorted(args.extracted/r['path'] for r in assets if pathlib.Path(r['path']).suffix in {'.json','.jsonl'})
 if not files:raise ValueError('No upstream JSON training files')
 manifest=[];start=time.monotonic();Base=reference_dataset_class()
 class Loader(Base):
  def _load_dataset(self,file_path):
   ds=super()._load_dataset(file_path)
   rel=str(pathlib.Path(file_path).relative_to(args.extracted));source=pathlib.Path(file_path).stem.split('_len-')[0]
   if source.startswith('len-') or source.replace('-','').isdigit():source=pathlib.Path(file_path).parent.name
   counts=collections.Counter();keep=[];hashes=[];row_ids=[]
   for i,row in enumerate(ds):
    reason=None
    if not isinstance(row.get('query'),str) or not row['query'].strip():reason='malformed_query'
    elif not isinstance(row.get('pos'),list) or not isinstance(row.get('neg'),list):reason='malformed_pool_type'
    elif not row['pos']:reason='empty_positive_pool'
    elif not row['neg']:reason='empty_negative_pool'
    elif any(not isinstance(d,str) or not d.strip() for d in row['pos']+row['neg']):reason='malformed_candidate'
    elif qhash(row['query']) in blocked:reason='evaluation_or_existing_validation_overlap'
    if reason:counts[reason]+=1;continue
    keep.append(i);hashes.append(qhash(row['query']));row_ids.append(i)
    if len(row['neg'])<7:counts['included_short_negative_pool']+=1
   counts['included']=len(keep);total=ds.info.splits['train'].num_examples if ds.info.splits and 'train' in ds.info.splits else len(ds)
   manifest.append({'file':rel,'source':source,'raw_rows':total,'after_cap':len(ds),'cap_removed':total-len(ds),'counts':dict(counts)})
   print('FILE_AUDIT',json.dumps(manifest[-1]),flush=True)
   selected=ds.select(keep)
   # Retain released data text/pools. Extra columns are schedule provenance only.
   selected=selected.add_column('__file',[rel]*len(keep)).add_column('__source',[source]*len(keep)).add_column('__row',row_ids).add_column('__qhash',hashes)
   save(args.output/'file_audit.partial.json',manifest)
   return selected
 random.seed(args.seed)
 loader_args=types.SimpleNamespace(train_data=[str(f) for f in files],max_example_num_per_dataset=args.max_example_num_per_dataset,knowledge_distillation=False,cache_path=str(args.output/'arrow_cache'),query_max_len=0,passage_max_len=8192)
 loader=Loader(loader_args,None);dataset=loader.dataset
 # Runtime reduction: uniform row sampling over the concatenation, no quotas.
 rng=random.Random(args.seed+1);all_indices=list(range(len(dataset)))
 validation_ids=[];val_hashes=set()
 query_hash_column=dataset['__qhash']
 for i in rng.sample(all_indices,len(all_indices)):
  h=query_hash_column[i]
  if h not in val_hashes:validation_ids.append(i);val_hashes.add(h)
  if len(validation_ids)==args.validation_queries:break
 # Propagate newly held-out translation identities before selecting training rows.
 val_linked_ids={i for qs in translations.values() for i,q in qs.items() if qhash(q) in val_hashes}
 for qs in translations.values():val_hashes.update(qhash(q) for i,q in qs.items() if i in val_linked_ids)
 eligible=[i for i,h in enumerate(query_hash_column) if h not in val_hashes]
 needed=args.steps*args.queries_per_update;order=[];epoch=0
 while len(order)<needed:
  epoch_indices=rng.sample(eligible,min(len(eligible),needed-len(order)))
  if not epoch_indices:raise ValueError('No eligible training rows')
  order.extend((epoch,i) for i in epoch_indices);epoch+=1
 crng=random.Random(args.seed+2);schedule=[]
 def group(i,occurrence,epoch,section):
  r=dataset[i];pi,ni=candidate_indices(r['pos'],r['neg'],crng)
  g={'id':f'{section}/{occurrence:06d}','source':r['__source'],'query':r['query'],'instruction':DEFAULT_INSTRUCTION,
   'documents':[r['pos'][pi]]+[r['neg'][j] for j in ni],'file':r['__file'],'row_after_cap':r['__row'],'global_row':i,'epoch':epoch,
   'candidate_origins':[{'pool':'pos','index':pi}]+[{'pool':'neg','index':j} for j in ni],
   'pool_sizes':{'pos':len(r['pos']),'neg':len(r['neg'])},'query_hash':r['__qhash']}
  schedule.append({k:v for k,v in g.items() if k not in ['query','instruction','documents']});return g
 val=[group(i,n,0,'validation') for n,i in enumerate(validation_ids)]
 train=[group(i,n,e,'train') for n,(e,i) in enumerate(order)]
 panels=read(args.panels);distribution={s:dict(collections.Counter(g['source'] for g in gs)) for s,gs in [('train',train),('validation',val)]}
 data={'train':train,'validation':val,'benchmark':panels['panel'],'reserved_benchmark':panels['reserved'],'distribution':distribution,
  'checks':{'evaluation_queries_filtered':True,'new_validation_overlap':not bool({g['query_hash'] for g in train}&val_hashes),'query_audit_coverage':coverage,'unresolved':'No exhaustive semantic/paraphrase check; source IDs absent or unmapped in some released records'},
  'reference':PIN,'sampling':{'seed':args.seed,'loader_seed':args.seed,'row_order_seed':args.seed+1,'candidate_seed':args.seed+2,'runtime_reduction':'uniform without-replacement row sample per pass over concatenated eligible rows','requested_exposures':needed,'unique_query_hashes':len({g['query_hash'] for g in train}),'unique_rows':len({g['global_row'] for g in train}),'resampling':'fresh upstream candidate draw for every occurrence, including revisits'}}
 save(args.output/'dataset.json.gz',data);save(args.output/'schedule.json.gz',schedule)
 from transformers import AutoTokenizer
 from transformers_rerank import PREFIX,SUFFIX
 tok=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True);enc=lambda s:tok.encode(s,add_special_tokens=False);pre=enc(PREFIX);suf=enc(SUFFIX);limit=8192-len(pre)-len(suf);lengths={}
 for section in ['train','validation','benchmark','reserved_benchmark']:
  lens=[];truncated=0;removed=0;per_source=collections.Counter();query_cut=0
  if args.defer_length_audit:continue
  for g in data[section]:
   for doc in g['documents']:
    ids=enc(body(g['instruction'],g['query'],doc,'query_first'));truncated+=len(ids)>limit;removed+=max(0,len(ids)-limit);lens.append(min(len(ids),limit)+len(pre)+len(suf))
    if len(ids)>limit:per_source[g.get('source',g.get('task'))]+=1
    query_cut+=len(enc(f"<Instruct>: {g['instruction']}\n<Query>: {g['query']}\n<Document>: "))>limit
  import numpy as np
  lengths[section]={'pairs':len(lens),'truncated_pairs':truncated,'removed_body_tokens':removed,'query_prefix_exceeds_body_budget':query_cut,'mean_tokens':float(np.mean(lens)),'p50_p90_p99_max':np.quantile(lens,[.5,.9,.99,1]).tolist(),'truncated_by_source':dict(per_source)}
 audit={'reference':PIN,'full_archive_included':True,'file_list':[str(f.relative_to(args.extracted)) for f in files],'files':manifest,'cap':args.max_example_num_per_dataset,'shuffle_ratio':0.0,'concat_rows_after_filters':len(dataset),'new_validation_excluded_rows':len(dataset)-len(eligible),'eligible_training_rows':len(eligible),'source_distribution':distribution,'full_source_distribution':dict(collections.Counter({source:sum(m['counts']['included'] for m in manifest if m['source']==source) for source in {m['source'] for m in manifest}})),'sampling':data['sampling'],'lengths':lengths,'length_audit_deferred':args.defer_length_audit,'dataset_sha256':digest(args.output/'dataset.json.gz'),'schedule_sha256':digest(args.output/'schedule.json.gz'),'seconds':time.monotonic()-start}
 save(args.output/'preparation.json',audit)
 print('PREPARED',json.dumps({k:v for k,v in audit.items() if k not in ['files','file_list','reference']}),flush=True)
if __name__=='__main__':main()
