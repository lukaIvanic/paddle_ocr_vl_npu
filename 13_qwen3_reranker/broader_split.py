"""Audited, quota-controlled BGE retrieval split; no model training.

Reopened benchmark families require membership in pinned upstream training
queries. Other sources retain the BGE authors' training-release provenance.
All suite queries (including Chinese dev) are blocked. Corpus overlap is
reported separately, not confused with query/label leakage.
"""
import argparse, collections, concurrent.futures, csv, functools, gzip, hashlib
import io, json, pathlib, random, sys, time, unicodedata, urllib.request, zipfile
from mixture_data import MIRROR, REVISION, quotas
from training_smoke_data import body, normalize

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'22_qwen3_embedding_benchmark'))
from protocol import TASKS
from suite_protocol import ENGLISH

# Engineering quotas, not an attempt to reconstruct unpublished Qwen weights.
# source: language, train queries, validation queries, instruction
GENERIC = 'Given a web search query, retrieve relevant passages that answer the query'
SPECS = {
 'hotpotqa': ('en',800,40,ENGLISH['HotpotQAHardNegatives'][2]),
 'nq': ('en',640,32,GENERIC),
 'miracl_en': ('en',640,32,GENERIC),
 'trivia': ('en',480,24,GENERIC),
 'mldr_en': ('en',320,16,'Given a question, retrieve relevant documents that answer the question'),
 'pubmed_qa_labeled': ('en',160,8,'Given a biomedical question, retrieve research abstracts that help answer the question'),
 'colliee': ('en',160,8,'Given a legal question, retrieve relevant legal provisions that help answer the question'),
 't2ranking': ('zh',1120,56,TASKS['T2Retrieval'][2]),
 'cMedQAv2': ('zh',960,48,TASKS['CmedqaRetrieval'][2]),
 'miracl_zh': ('zh',640,32,GENERIC),
 'mldr_zh': ('zh',320,16,'Given a question, retrieve relevant documents that answer the question'),
 'law_gpt': ('zh',160,8,'Given a legal question, retrieve answers that address the legal question'),
}
PENDING = {
 'msmarco/mmarco_chinese':'Need original query-ID mapping across English and Chinese translations',
 'dureader':'Need verify exact upstream variant and training-query membership',
 'Multi-CPR':'Not in this BGE mirror inventory; acquire official train data separately',
 'lecardv2':'Defer case retrieval until long-query handling is explicit',
 'NLI/sentence-matching':'No generic passage-relevance relabeling; separate task design required',
}

def digest(b): return hashlib.sha256(b).hexdigest()
def key(t): return normalize(t)
def strict_key(t): return ''.join(c for c in key(t) if c.isalnum())
def save(p, x):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 b=(json.dumps(x,ensure_ascii=False)+'\n').encode()
 tmp=p.with_suffix(p.suffix+'.partial');tmp.write_bytes(gzip.compress(b,mtime=0) if p.suffix=='.gz' else b);tmp.replace(p)
def read(p):
 p=pathlib.Path(p);return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())
def url_read(url):
 return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=60)
def download(url, p):
 p=pathlib.Path(p)
 if not p.exists():
  p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.partial')
  with url_read(url) as stream,tmp.open('wb') as out:
   while b:=stream.read(1024**2):out.write(b)
  tmp.replace(p)
 return p

def remote_parquet(url):
 import fsspec,aiohttp,pyarrow.parquet as pq
 stream=fsspec.open(url,mode='rb',block_size=1024**2,
  client_kwargs={'trust_env':True,'timeout':aiohttp.ClientTimeout(total=None,sock_connect=20,sock_read=60)},
  headers={'User-Agent':'Mozilla/5.0'}).open()
 return pq.ParquetFile(stream)

def registries(out):
 """Pin upstream assets; match BGE query text to IDs, never trust split name alone."""
 cache=out/'registries.json.gz'
 if cache.exists():return read(cache)
 root=out/'upstream';root.mkdir(exist_ok=True)
 result={}
 t2rev='2a369a430a70979223f1b9a41b1919774d46b432'
 base=f'https://hf-mirror.com/datasets/THUIR/T2Ranking/resolve/{t2rev}/data/'
 for split in ['train','dev','test']:
  fn=f'queries.{split}.tsv';p=download(base+fn,root/('t2-'+fn))
  rows=[line.rstrip('\n').split('\t',1) for line in p.read_text().splitlines()]
  result.setdefault('t2ranking',{})[split]={key(t):i for i,t in rows}
 result['t2ranking']['provenance']={'repository':'THUIR/T2Ranking','revision':t2rev}
 print('REGISTRY_READY','t2ranking',flush=True)
 rev='1908d6afbbead072334abe2965f91bd2709910ab'
 result['hotpotqa']={'train':{},'dev':{},'test':{},'provenance':{'repository':'hotpotqa/hotpot_qa','revision':rev}}
 paths=[('train',f'distractor/train-0000{i}-of-00002.parquet') for i in range(2)]+[('dev','distractor/validation-00000-of-00001.parquet'),('test','fullwiki/test-00000-of-00001.parquet')]
 for split,fn in paths:
  # Fetch only question/id column chunks, not 500 MB of contexts.
  url=f'https://hf-mirror.com/datasets/hotpotqa/hotpot_qa/resolve/{rev}/{fn}?download=true'
  rows=remote_parquet(url).read(columns=['id','question']).to_pylist()
  result['hotpotqa'][split].update({key(r['question']):r['id'] for r in rows})
 print('REGISTRY_READY','hotpotqa',flush=True)
 rev='85feb9278c3ae552c591205cbf3e828368c91f8f'
 base=f'https://raw.githubusercontent.com/zhangsheng93/cMedQA2/{rev}/'
 def zip_rows(fn):
  p=download(base+fn,root/fn)
  z=zipfile.ZipFile(p);files=[n for n in z.namelist() if not n.endswith('/') and not n.startswith('__MACOSX')]
  assert len(files)==1,files
  return list(csv.reader(io.StringIO(z.read(files[0]).decode('utf-8-sig'))))
 questions=zip_rows('question.zip');print('CMED_COLUMNS',questions[0],flush=True)
 qmap={r[0]:r[1] for r in questions[1:]}
 result['cMedQAv2']={'provenance':{'repository':'zhangsheng93/cMedQA2','revision':rev}}
 for split in ['train','dev','test']:
  rows=zip_rows(split+'_candidates.zip');print('CMED_SPLIT_COLUMNS',split,rows[0],flush=True)
  ids={r[0] for r in rows[1:]};assert ids<=qmap.keys()
  result['cMedQAv2'][split]={key(qmap[i]):i for i in ids}
 result['_asset_sha256']={p.name:digest(p.read_bytes()) for p in root.iterdir() if p.is_file()}
 save(cache,result);return result

def acquire(args):
 out=args.output;out.mkdir(parents=True,exist_ok=True)
 metadata=read(args.metadata);assert metadata['sha']==REVISION
 counts={r['config_name']:sum(s['num_examples'] for s in r['splits']) for r in metadata['cardData']['dataset_info']}
 jobs=[]
 for source,(_,n,v,_) in SPECS.items():
  cs={c:nr for c,nr in counts.items() if c.split('_len-')[0]==source}
  # Finite uniform sample stratified by published length bucket, never only shortest bucket.
  q=quotas(min(sum(cs.values()),max(3*(n+v),512)),cs,cs,minimum=2)
  jobs.extend((source,c,k) for c,k in q.items() if k)
 def get(job):
  source,config,n=job;p=out/'rows'/(config+'.json.gz')
  if p.exists():
   cached=read(p)
   if cached['seed']!=args.seed or cached['revision']!=REVISION:raise ValueError('Raw cache identity mismatch')
   if all(len(r['pos'])==r['original_positive_count'] for r in cached['rows']):
    return {'config':config,'cached':True,'rows':len(cached['rows'])}
  rng=random.Random(f'{args.seed}/{config}')
  chosen_set=set()
  while len(chosen_set)<n:
   start=rng.randrange(counts[config])
   for j in range(min(512,n-len(chosen_set))):chosen_set.add((start+j)%counts[config])
  chosen=sorted(chosen_set);result=[];base=0;assets=[]
  files=sorted(s['rfilename'] for s in metadata['siblings'] if s['rfilename'].startswith(config+'/') and s['rfilename'].endswith('.parquet'))
  t=time.monotonic()
  for fn in files:
   url=f'https://hf-mirror.com/datasets/{MIRROR}/resolve/{REVISION}/{fn}?download=true'
   pf=remote_parquet(url);assets.append({'url':url,'rows':pf.metadata.num_rows})
   for i in range(pf.num_row_groups):
    nr=pf.metadata.row_group(i).num_rows;indices=[x-base for x in chosen if base<=x<base+nr]
    if indices:
     table=pf.read_row_group(i).take(indices)
     print('ROW_GROUP',config,base,len(indices),flush=True)
     for idx,row in zip(indices,table.to_pylist()):
      rr=random.Random(f'{args.seed}/{config}/{base+idx}')
      pos=row['pos'];neg=row['neg']
      result.append({'id':f'{config}/{base+idx}','source':source,'config':config,'query':row['query'],
       'pos':pos,'neg':rr.sample(neg,min(len(neg),32)),
       'original_positive_count':len(pos),'original_negative_count':len(neg)})
    base+=nr
   if base>chosen[-1]:break
  assert len(result)==n,(config,len(result),n)
  save(p,{'sampling':'seeded random circular blocks of at most 512 rows within source/length strata','rows':result,'revision':REVISION,'seed':args.seed,'assets':assets,'pool_cap':{'positive':None,'negative':32}})
  return {'config':config,'rows':n,'seconds':round(time.monotonic()-t,2),'bytes':p.stat().st_size}
 with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
  for future in concurrent.futures.as_completed([pool.submit(get,j) for j in jobs]):
   print('ACQUIRED',json.dumps(future.result()),flush=True)
 registries(out)

def full_benchmark_queries(panels):
 from datasets import load_dataset
 rows=[];sources={}
 for name,meta in panels['provenance'].items():
  if name in ENGLISH:
   repo,revision,*_=ENGLISH[name]
   ds=load_dataset(repo,'queries',revision=revision)
   qs=next(iter(ds.values()))
  else:
   repo=meta['dataset']['path'];revision=meta['dataset']['revision']
   qs=load_dataset(repo,'default',revision=revision)['queries']
  rows.extend(qs['text']);sources[name]={'queries':len(qs),'revision':revision,'path':repo,'eval_split':meta['split']}
 return rows,sources

class NearIndex:
 """Conservative lexical near-duplicate screen, not semantic/translation proof."""
 def __init__(self,texts):
  from sklearn.feature_extraction.text import TfidfVectorizer
  self.texts=sorted(set(key(t) for t in texts if key(t)))
  self.exact={strict_key(t) for t in self.texts}
  self.v=TfidfVectorizer(analyzer='char',ngram_range=(3,5),max_features=400000,dtype='float32')
  self.matrix=self.v.fit_transform(self.texts)
 def blocked(self,texts):
  result={};matrix=self.v.transform([key(t) for t in texts])
  for start in range(0,len(texts),128):
   sims=(matrix[start:start+128]@self.matrix.T).tocsr()
   for j in range(sims.shape[0]):
    i=start+j;q=texts[i];row=sims.getrow(j)
    if strict_key(q) in self.exact:result[i]={'kind':'normalized_exact'}
    elif row.nnz:
     k=row.data.argmax();score=float(row.data[k]);target=self.texts[row.indices[k]]
     # Length guard reduces matches caused by a small shared template.
     ratio=min(len(key(q)),len(target))/max(len(key(q)),len(target),1)
     if score>=.92 and ratio>=.85:result[i]={'kind':'lexical_near','cosine':score,'matched_query':target}
  return result

def clean_pool(rows, source, registry, blocked):
 counts=collections.Counter();out=[];seen=set()
 for r in rows:
  if len(r['pos']) != r['original_positive_count']:
   raise ValueError('Positive pool was capped; reacquire before candidate construction')
  q=key(r['query']);strict=strict_key(q)
  if not strict or strict in blocked:counts['benchmark_or_upstream_holdout_query']+=1;continue
  if strict in seen:counts['duplicate_query']+=1;continue
  if registry is not None and q not in registry['train']:counts['not_verified_upstream_train']+=1;continue
  seen.add(strict);pos=[];neg=[];used=set()
  for label,docs in [('pos',r['pos']),('neg',r['neg'])]:
   for d in docs:
    h=key(d)
    if not h or h in used:continue
    used.add(h);(pos if label=='pos' else neg).append(d)
  if not pos:counts['no_positive']+=1;continue
  out.append(r|{'pos':pos,'neg':neg,'upstream_query_id':registry['train'][q] if registry else None,
   'split_evidence':'upstream_train_query_match' if registry else 'BGE_author_training_release'})
 return out,dict(counts)

def sample_group(r, fits, seed):
 rng=random.Random(f'{seed}/{r["id"]}');p=list(r['pos']);n=list(r['neg']);rng.shuffle(p);rng.shuffle(n)
 pos=next((d for d in p if fits(r,d)),None)
 if pos is None:return None
 docs=[pos];origins=['supplied_positive'];labels=[1]
 for d in n:
  if len(docs)==8:break
  if fits(r,d):docs.append(d);origins.append('supplied_negative');labels.append(0)
 if len(docs)!=8:return None
 return {k:r[k] for k in ['id','source','query','upstream_query_id','split_evidence']}|{
  'language':SPECS[r['source']][0],'instruction':SPECS[r['source']][3],'documents':docs,
  'candidate_origin':origins,'supplied_labels':labels,'positive_pool':p,'negative_pool':n,
  'original_positive_count':r['original_positive_count'],'original_negative_count':r['original_negative_count'],
  'supervised_ce_ready':False}

def quantiles(xs):
 import numpy as np
 return dict(zip(['count','mean','p50','p90','p99','max'],[len(xs),float(np.mean(xs)),*map(float,np.quantile(xs,[.5,.9,.99])),max(xs)]))

def build(args):
 from transformers import AutoTokenizer
 from transformers_rerank import PREFIX,SUFFIX
 out=args.output;panels=read(args.panels);registry=registries(out)
 allq,benchmark_meta=full_benchmark_queries(panels)
 # Retain all upstream dev/test queries even when the pinned suite uses a subset.
 for source in SPECS:
  if source in registry:allq.extend(list(registry[source]['dev'])+list(registry[source]['test']))
 prior=read(args.previous_dataset);allq.extend(g['query'] for g in prior['validation'])
 blocked={strict_key(q) for q in allq};index=NearIndex(allq)
 tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True)
 @functools.lru_cache(maxsize=100000)
 def nt(text):return len(tokenizer.encode(text,add_special_tokens=False))
 overhead=nt(PREFIX)+nt(SUFFIX)
 def fits(r,d):
  inst=SPECS[r['source']][3]
  return all(nt(body(inst,r['query'],d,o))+overhead<=8192 for o in ['query_first','document_first'])
 cleaned={};audit={'sources':{},'benchmark_registry':benchmark_meta,'blocked_query_count':len(blocked)}
 for source in SPECS:
  rows=[]
  for p in sorted((out/'rows').glob(source+'_len-*.json.gz')):rows.extend(read(p)['rows'])
  rows,counts=clean_pool(rows,source,registry.get(source),blocked)
  near=index.blocked([r['query'] for r in rows]);rows=[r for i,r in enumerate(rows) if i not in near]
  counts['lexical_near_excluded']=len(near);audit['sources'][source]={'cleaning':counts,'available':len(rows)}
  random.Random(f'{args.seed}/{source}/split').shuffle(rows);cleaned[source]=rows
  print('CLEAN',source,len(rows),counts,flush=True)
 # Choose validation first; then screen every training candidate against ALL validation queries.
 validation=[];train=[];global_seen=set();val_ids=set()
 for source,(_,n,v,_) in SPECS.items():
  for r in cleaned[source]:
   if strict_key(r['query']) in global_seen:continue
   g=sample_group(r,fits,args.seed)
   if g:
    validation.append(g);val_ids.add(r['id']);global_seen.add(strict_key(r['query']))
   if sum(g['source']==source for g in validation)==v:break
  assert sum(g['source']==source for g in validation)==v,('validation quota',source)
 vindex=NearIndex([g['query'] for g in validation])
 for source,(_,n,v,_) in SPECS.items():
  rows=[r for r in cleaned[source] if r['id'] not in val_ids]
  blocked_by_val=vindex.blocked([r['query'] for r in rows]);rows=[r for i,r in enumerate(rows) if i not in blocked_by_val]
  selected=[]
  for r in rows:
   if strict_key(r['query']) in global_seen:continue
   g=sample_group(r,fits,args.seed)
   if g:selected.append(g);global_seen.add(strict_key(r['query']))
   if len(selected)==n:break
  assert len(selected)==n,('training quota',source,len(selected),n)
  train.extend(selected);audit['sources'][source]['validation_near_excluded']=len(blocked_by_val)
 # Round-robin shuffled source queues make the first 1600 a precise quarter of each quota.
 # Each 1600-query block is balanced; each source's query selection stays seeded.
 ordered=[]
 for block in range(4):
  chunk=[]
  for s,(_,n,_,_) in SPECS.items():
   gs=[g for g in train if g['source']==s];chunk.extend(gs[block*n//4:(block+1)*n//4])
  random.Random(args.seed+block).shuffle(chunk);ordered.extend(chunk)
 train=ordered;random.Random(args.seed).shuffle(validation)
 evalgroups=panels['panel']+panels['reserved']
 docset=lambda gs:{digest(key(d).encode()) for g in gs for d in g['documents']}
 td,vd,ed=map(docset,[train,validation,evalgroups])
 audit['overlap']={'train_validation_shared_document_texts':len(td&vd),'train_panel_shared_document_texts':len(td&ed),
  'validation_panel_shared_document_texts':len(vd&ed),'document_overlap_policy':'allowed shared corpus; query supervision kept separate',
  'train_validation_exact_queries':len({strict_key(g['query']) for g in train}&{strict_key(g['query']) for g in validation})}
 audit['lengths']={}
 for section,groups in [('train',train),('validation',validation),('benchmark',evalgroups)]:
  buckets=collections.defaultdict(lambda:collections.defaultdict(list))
  for g in groups:
   s=g.get('source',g.get('task'));b=buckets[s];b['query_tokens'].append(nt(g['query']))
   for i,d in enumerate(g['documents']):
    b['document_tokens'].append(nt(d))
    b['prompt_tokens'].append(nt(body(g['instruction'],g['query'],d,'document_first'))+overhead)
    if 'candidate_origin' in g:b[g['candidate_origin'][i]+'_document_tokens'].append(nt(d))
  audit['lengths'][section]={s:{k:quantiles(v) for k,v in b.items()} for s,b in buckets.items()}
 audit['candidate_origins']=dict(collections.Counter(o for g in train for o in g['candidate_origin']))
 audit['candidate_difficulty']='NOT YET TEACHER-SCORED; supplied negative does not establish hardness or correctness'
 audit['checks']={'query_exact_and_lexical_near_screened':True,'upstream_train_membership_for_reopened_families':True,
  'all_training_and_validation_prompts_fit_both_orders_8192':True,'every_candidate_from_its_own_released_row':True}
 audit['limitations']=['Lexical screen is not a semantic or translation decontamination guarantee',
  'BGE sources outside reopened families retain author-release provenance; original ID mapping not reconstructed',
  'No new candidate mining or teacher difficulty/false-negative audit yet',
  'Argument/counterargument, fact verification, commerce/video and duplicate-question coverage still incomplete',
  'Existing benchmark panels are development data; fresh confirmation remains separate']
 result={'train':train,'validation':validation,'benchmark':panels['panel'],'reserved_benchmark':panels['reserved'],
  'distribution':{s:v[1] for s,v in SPECS.items()},'checks':audit['checks'],
  'provenance':{'recipe':'broader_bge_retrieval_v1','seed':args.seed,'mirror':MIRROR,'revision':REVISION,
   'quotas':SPECS,'pending_sources':PENDING,'registries_sha256':digest((out/'registries.json.gz').read_bytes()),
   'raw_sample_sha256':{p.name:digest(p.read_bytes()) for p in sorted((out/'rows').glob('*.json.gz'))},
   'benchmark_panel_sha256':digest(args.panels.read_bytes()),'instructions':'pinned Qwen task prompts where matching; explicitly authored descriptive prompts for legal/biomedical/long-document sources',
   'candidate_selection':'seeded positive and seven distinct supplied negatives from the same released row; no supplementation; fail if quota cannot be met',
   'supervised_loss_status':'not approved by this artifact; teacher/label disagreement audit still required'}}
 save(out/'dataset.json.gz',result);audit['dataset_sha256']=digest((out/'dataset.json.gz').read_bytes())
 audit['counts']={k:len(result[k]) for k in ['train','validation','benchmark','reserved_benchmark']}
 save(out/'audit.json',audit)
 save(out/'pilot_1600.json.gz',result|{'train':train[:1600],'distribution':dict(collections.Counter(g['source'] for g in train[:1600]))})
 print('PREPARED',json.dumps({'counts':audit['counts'],'distribution':result['distribution'],'sha256':audit['dataset_sha256'],'candidate_origins':audit['candidate_origins']}),flush=True)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['acquire','build'])
 p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--metadata',type=pathlib.Path,default='/tmp/qwen_bge_info.json')
 p.add_argument('--panels',type=pathlib.Path,default='/workspace/results/qwen_margin_distill/data_fast/benchmark_panels.json.gz')
 p.add_argument('--previous-dataset',type=pathlib.Path,default='/workspace/results/qwen_margin_distill/data_fast/dataset.json.gz')
 p.add_argument('--tokenizer',default='/workspace/models/Qwen3-Reranker-0.6B');p.add_argument('--seed',type=int,default=2047);p.add_argument('--workers',type=int,default=4)
 args=p.parse_args();(acquire if args.mode=='acquire' else build)(args)
if __name__=='__main__':main()
