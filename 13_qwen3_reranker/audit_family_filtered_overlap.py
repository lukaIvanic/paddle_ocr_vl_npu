"""CPU audit: exact query/document matches, lexical query neighbors, shingle document neighbors.
No filtering decisions are made by this audit. Near-match thresholds are diagnostic.
"""
import os
for k,v in {'HF_HOME':'/workspace/.cache/huggingface','HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','TORCH_DEVICE_BACKEND_AUTOLOAD':'0','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}.items():os.environ[k]=v
import argparse,collections,gzip,hashlib,html,json,re,sys,time,unicodedata,zlib
from pathlib import Path
EXCLUDED={'hotpotqa','msmarco','mmarco_chinese','dureader','t2ranking','cMedQAv2'}
def read(p):
 p=Path(p);b=p.read_bytes();return json.loads(gzip.decompress(b) if str(p).endswith('.gz') else b)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def norm(t):return ' '.join(unicodedata.normalize('NFKC',t).casefold().split())
def clean(t):return ' '.join(re.findall(r'[a-z0-9]+|[\u3400-\u9fff]|[^\W\d_]+',norm(html.unescape(re.sub(r'<[^>]+>',' ',t))),re.UNICODE))
def shingles(t):
 units=t.split();n=5
 return {zlib.crc32(' '.join(units[i:i+n]).encode()) for i in range(max(0,len(units)-n+1))}
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--eval-repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 sys.path.insert(0,str(a.eval_repo/'22_qwen3_embedding_benchmark'))
 from protocol import TASKS,validate_task
 from suite_protocol import ENGLISH
 from run_english_suite import load_task
 from mteb.evaluation.evaluators.RetrievalEvaluator import corpus_to_str
 import mteb,types,numpy as np
 from sklearn.feature_extraction.text import TfidfVectorizer
 data=read(a.parent/'prepared/dataset.json.gz');cfg=read(a.parent/'run_configuration.json');assert digest(a.parent/'prepared/dataset.json.gz')==cfg['dataset_sha256']
 allgroups=[g for g in data['train'] if g['source'] not in EXCLUDED];exposure=sum(x['queries'] for x in read(a.parent/'student/result.json')['updates'][:500]);used={g['id'] for g in allgroups[:exposure]}
 tq=[norm(g['query']) for g in allgroups];docmap=collections.defaultdict(list)
 for g in allgroups:
  for j,doc in enumerate(g['documents']):docmap[norm(doc)].append({'id':g['id'],'source':g['source'],'candidate':j,'pool':g['candidate_origins'][j]['pool'],'query':g['query'],'scheduled':g['id'] in used})
 texts=list(docmap);anchors=collections.defaultdict(list);cleantexts=[]
 for i,t in enumerate(texts):
  c=clean(t);cleantexts.append(c)
  for h in sorted(shingles(c))[:8]:anchors[h].append(i)
 print('INDEX',json.dumps({'retained_groups':len(allgroups),'scheduled_groups':exposure,'unique_documents':len(texts),'anchors':len(anchors)}),flush=True)
 inv=read('/workspace/results/qwen500_chinese_cmtebr_20261008_bf2fb7b7/evaluation/inventory.json');em=read('/workspace/results/qwen500_english_mtebr_20261008_c4778168/evaluation/manifest.json')
 summary={'parent_sha256':cfg['dataset_sha256'],'excluded_sources':sorted(EXCLUDED),'retained_groups':len(allgroups),'scheduled_groups':exposure,'methods':{'query':'NFKC/casefold/whitespace exact + character3-5 TF-IDF cosine, top2 per eval query >=0.65, fitted jointly per task; diagnostic only','documents':'Normalized full-text exact + HTML/punctuation-normalized 5-unit shingle neighbors (English words, Chinese characters), 8 minimum CRC32 anchors, ignore anchors occurring in >200 training documents; verify Jaccard>=0.60 or shorter-text containment>=0.85 with >=20 shared shingles and >=60 characters','limitations':['Lexical audit is not exhaustive semantic/paraphrase/translation detection','Documents checked are union of actual top100 candidates, not full corpora','Anchor screening is approximate; query thresholds are review triggers, not exclusion rules']},'tasks':{}}
 for name in list(TASKS)+list(ENGLISH):
  started=time.monotonic();print('LOAD',name,flush=True)
  existing=a.output/f'{name}.json'
  if existing.exists():
   prior=read(a.output/'summary.json');assert prior['parent_sha256']==cfg['dataset_sha256'] and prior['excluded_sources']==sorted(EXCLUDED)
   summary['tasks'][name]=prior['tasks'][name];print('REUSE',name,flush=True);continue
  if name in TASKS:
   task=mteb.get_tasks(tasks=[name])[0];validate_task(task);task.load_data();split='dev';cp=Path(inv[name]['candidate_file']);sha=inv[name]['candidate_sha256']
  else:
   task,_=load_task(name,types.SimpleNamespace());split='test';cp=Path(em['saved_embedding'])/'embedding'/name/'mteb'/f'{name}_default_predictions.json';sha=read(cp.parents[1]/'result.json')['candidates_sha256']
  assert digest(cp)==sha
  candidates=read(cp);qrels=task.relevant_docs[split];assert set(candidates)==set(qrels)
  qids=list(candidates);eq=[norm(task.queries[split][q]) for q in qids]
  vec=TfidfVectorizer(analyzer='char',ngram_range=(3,5),dtype=np.float32);mat=vec.fit_transform(tq+eq);trainmat=mat[:len(tq)];querymat=mat[len(tq):];qhits=[]
  for start in range(0,len(eq),128):
   sims=(querymat[start:start+128]@trainmat.T).tocsr()
   for j in range(sims.shape[0]):
    row=sims.getrow(j);inds=np.argsort(row.data)[-2:][::-1]
    for ix in inds:
     score=float(row.data[ix]);idx=int(row.indices[ix]);ei=start+j
     if score<.65:continue
     g=allgroups[idx];qhits.append({'eval_qid':qids[ei],'eval_query':eq[ei],'train_group':g['id'],'source':g['source'],'train_query':g['query'],'cosine':score,'exact':eq[ei]==tq[idx],'scheduled':g['id'] in used})
  exact=[];near=[];selected={d for ds in candidates.values() for d in ds};relevant={d for ds in qrels.values() for d,v in ds.items() if v>0}
  for counter,did in enumerate(sorted(selected)):
   raw=corpus_to_str([task.corpus[split][did]])[0];t=norm(raw)
   if t in docmap:
    exact.append({'eval_did':did,'judged_relevant_somewhere':did in relevant,'origins':docmap[t],'text':t[:500]});continue
   c=clean(raw)
   if len(c)<60:continue
   sh=shingles(c)
   if not sh:continue
   possible=set()
   for h in sorted(sh)[:8]:
    ids=anchors.get(h,[])
    if len(ids)<=200:possible.update(ids)
   best=None
   for idx in possible:
    tc=cleantexts[idx]
    if len(tc)<60:continue
    ts=shingles(tc);shared=len(sh&ts)
    if shared<20:continue
    jac=shared/len(sh|ts);contain=shared/min(len(sh),len(ts))
    if jac<.6 and contain<.85:continue
    row={'eval_did':did,'judged_relevant_somewhere':did in relevant,'jaccard':jac,'shorter_containment':contain,'shared_shingles':shared,'origins':docmap[texts[idx]],'eval_text':t[:1000],'train_text':texts[idx][:1000]}
    if best is None or (jac,contain)>(best['jaccard'],best['shorter_containment']):best=row
   if best:near.append(best)
   if counter and counter%25000==0:print('DOC_PROGRESS',name,counter,len(selected),flush=True)
  qhits.sort(key=lambda x:-x['cosine']);near.sort(key=lambda x:-x['jaccard'])
  out={'task':name,'candidate_sha256':sha,'evaluation_queries':len(eq),'candidate_document_ids':len(selected),'query_neighbors':qhits,'exact_query_count':sum(x['exact'] for x in qhits),'exact_documents':exact,'near_documents':near,'seconds':time.monotonic()-started}
  save(a.output/f'{name}.json',out)
  compact={'queries':len(eq),'candidate_documents':len(selected),'query_neighbors_ge065':len(qhits),'query_neighbors_ge085':sum(x['cosine']>=.85 for x in qhits),'exact_queries':sum(x['exact'] for x in qhits),'exact_documents':len(exact),'near_documents':len(near),'scheduled_exact_documents':sum(any(o['scheduled'] for o in x['origins']) for x in exact),'scheduled_near_documents':sum(any(o['scheduled'] for o in x['origins']) for x in near),'seconds':out['seconds']}
  summary['tasks'][name]=compact;save(a.output/'summary.json',summary);print('TASK',json.dumps({'task':name,**compact}),flush=True)
  del task,mat,querymat,trainmat,candidates,qrels
 summary['status']='completed';save(a.output/'summary.json',summary);print('COMPLETE',flush=True)
if __name__=='__main__':main()
