"""Acquire upstream query/label assets and audit their pinned benchmark mapping.
No final training selection, candidate generation, or model execution.
"""
import argparse,concurrent.futures,collections,hashlib,json,pathlib,time,urllib.request
from broader_split import key,read,save,download

MMARCO_REV='6d039c4638c0ba3e46a9cb7b498b145e7edc6230'
CPR_REV='a4e467182a3e2c110528a1575d79b33cc449d2c3'

def tsv_queries(p):
 result={}
 for line in p.read_text().splitlines():
  qid,text=line.split('\t',1);result[qid]=text
 return result

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=pathlib.Path,required=True)
 p.add_argument('--panels',type=pathlib.Path,default='/workspace/results/qwen_margin_distill/data_fast/benchmark_panels.json.gz')
 args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
 jobs={}
 for version,langs in [('google',['english','chinese']),('helsinki',['chinese'])]:
  for lang in langs:
   for split in ['train','dev']:
    name=f'{version}-{lang}-{split}.tsv'
    fn=f'data/{version}/queries/{split}/{lang}_queries.{split}.tsv'
    jobs[name]=f'https://hf-mirror.com/datasets/unicamp-dl/mmarco/resolve/{MMARCO_REV}/{fn}?download=true'
 for domain in ['ecom','video','medical']:
  for fn in ['train.query.txt','dev.query.txt','qrels.train.tsv','qrels.dev.tsv']:
   jobs[f'cpr-{domain}-{fn}']=f'https://raw.githubusercontent.com/Alibaba-NLP/Multi-CPR/{CPR_REV}/data/{domain}/{fn}'
 def get(item):
  name,url=item;t=time.monotonic()
  for attempt in range(3):
   try:
    f=download(url,out/name);break
   except Exception:
    if attempt==2:raise
    time.sleep(2)
  r={'url':url,'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'seconds':time.monotonic()-t}
  print('ASSET',name,json.dumps(r),flush=True);return name,r
 with concurrent.futures.ThreadPoolExecutor(4) as pool:assets=dict(pool.map(get,jobs.items()))
 save(out/'assets.json',assets)
 from datasets import load_dataset
 panels=read(args.panels);bench={}
 for task in ['MMarcoRetrieval','DuRetrieval','EcomRetrieval','VideoRetrieval','MedicalRetrieval']:
  meta=panels['provenance'][task]['dataset']
  ds=load_dataset(meta['path'],'default',revision=meta['revision'])['queries']
  idcol='_id' if '_id' in ds.column_names else 'id'
  bench[task]={str(r[idcol]):r['text'] for r in ds}
 audit={'assets':assets,'mMARCO':{},'Multi-CPR':{},'DuReader':{'status':'official archive located; train membership audit pending'}}
 mm={name:tsv_queries(out/name) for name in jobs if not name.startswith('cpr-')}
 mmbench={key(x) for x in bench['MMarcoRetrieval'].values()};blocked_ids=set()
 for name,qs in mm.items():
  ids={i for i,q in qs.items() if key(q) in mmbench};blocked_ids.update(ids)
  audit['mMARCO'][name]={'queries':len(qs),'benchmark_text_matches':len(ids),'matched_benchmark_unique_texts':len({key(qs[i]) for i in ids})}
 # IDs provide the bridge; block every corresponding English or Chinese translation.
 for name,qs in mm.items():
  audit['mMARCO'][name]['benchmark_linked_ids']=len(set(qs)&blocked_ids)
  if name.endswith('-train.tsv'):
   devname=name.replace('-train.tsv','-dev.tsv')
   audit['mMARCO'][name]['own_dev_id_overlap']=len(set(qs)&set(mm[devname]))
 audit['mMARCO']['pinned_benchmark_query_count']=len(bench['MMarcoRetrieval'])
 audit['mMARCO']['blocked_original_query_ids']=sorted(blocked_ids)
 for domain,task in [('ecom','EcomRetrieval'),('video','VideoRetrieval'),('medical','MedicalRetrieval')]:
  train=tsv_queries(out/f'cpr-{domain}-train.query.txt');dev=tsv_queries(out/f'cpr-{domain}-dev.query.txt')
  b={key(x) for x in bench[task].values()};t={key(x) for x in train.values()};d={key(x) for x in dev.values()}
  rels=[]
  for line in (out/f'cpr-{domain}-qrels.train.tsv').read_text().splitlines():
   fields=line.split();assert len(fields)==4;rels.append(fields)
  audit['Multi-CPR'][domain]={'train_queries':len(train),'dev_queries':len(dev),'train_dev_id_overlap':len(set(train)&set(dev)),
   'train_dev_normalized_text_overlap':len(t&d),'pinned_benchmark_queries':len(bench[task]),'benchmark_texts_in_official_dev':len(b&d),
   'benchmark_texts_in_official_train':len(b&t),'training_qrels':len(rels),'qrel_ids_not_in_train':len({r[0] for r in rels}-set(train)),
   'candidate_status':'positive judgments available; official reranker code requires a separate retrieved candidate run; no negatives invented'}
 save(out/'audit.json',audit)
 print('AUDIT',json.dumps({k:v for k,v in audit.items() if k!='assets' and k!='mMARCO'},ensure_ascii=False),flush=True)
 print('MMARCO',json.dumps({k:v for k,v in audit['mMARCO'].items() if k!='blocked_original_query_ids'}),flush=True)
if __name__=='__main__':main()
