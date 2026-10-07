"""Inspect a candidate external release; does not approve it or construct a split."""
import argparse,collections,concurrent.futures,hashlib,json,pathlib,re,time
from broader_split import download,key,save
from audit_missing_sources import tsv_queries
REV='e9443ab6f5d4dc29c79cea03834e932428ed6ab1'
ASSETS=[('train-00000-of-00003.parquet',81533907,'147ff0bf9cfc55c2d4af808098f117db8cd933ea51fc861a9ff3d98e502a69e4'),('train-00001-of-00003.parquet',143589207,'f42e9dace762dc9c522beca78e4140963d218f8a6ff24be1e60c32df004d3c11'),('train-00002-of-00003.parquet',31103946,'823d84fdf6839b30c988dad88e56643b7036c744e2808028f8fe56c1320943b6')]
def unwrap(text):
 # Recognize an explicit released instruction wrapper; retain both in audit.
 m=re.match(r'^(Instruct: [^\n]*\n\s*Query:\s*)(.*)$',text,re.DOTALL)
 return (m.group(2),m.group(1)) if m else (text,'')
def main():
 import pyarrow.parquet as pq
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=pathlib.Path,required=True);args=p.parse_args();out=args.output;root=out/'kalm';root.mkdir(exist_ok=True)
 def get(asset):
  fn,size,sha=asset;t=time.monotonic();url=f'https://hf-mirror.com/datasets/KaLM-Embedding/KaLM-embedding-finetuning-data/resolve/{REV}/Multi-CPR/{fn}?download=true'
  for attempt in range(3):
   try:
    path=download(url,root/fn);assert path.stat().st_size==size;assert hashlib.sha256(path.read_bytes()).hexdigest()==sha;break
   except Exception:
    if attempt==2:raise
  result={'file':fn,'url':url,'bytes':size,'sha256':sha,'seconds':time.monotonic()-t};print('VERIFIED',json.dumps(result),flush=True);return result
 with concurrent.futures.ThreadPoolExecutor(3) as pool:assets=list(pool.map(get,ASSETS))
 train={domain:{key(q):i for i,q in tsv_queries(out/f'cpr-{domain}-train.query.txt').items()} for domain in ['ecom','video','medical']}
 dev={key(q) for domain in train for q in tsv_queries(out/f'cpr-{domain}-dev.query.txt').values()}
 counts=collections.Counter();domains=collections.Counter();pools=collections.Counter();wrappers=collections.Counter();examples=[];seen=set();elig=collections.Counter()
 for asset in assets:
  for batch in pq.ParquetFile(root/asset['file']).iter_batches(batch_size=1024):
   for r in batch.to_pylist():
    q,wrapper=unwrap(r['query']);k=key(q);wrappers[wrapper]+=1;counts['rows']+=1
    matches=[d for d,qs in train.items() if k in qs];domains['+'.join(matches) or 'unmatched']+=1
    if len(matches)==1 and k not in dev:elig[matches[0]]+=1
    if k in dev:counts['official_dev_query_matches']+=1
    if k in seen:counts['duplicate_normalized_query_rows']+=1
    seen.add(k);pools[f"{len(r['pos'])}pos/{len(r['neg'])}neg"]+=1
    pos={key(unwrap(d)[0]) for d in r['pos']};neg=[key(unwrap(d)[0]) for d in r['neg']]
    counts['rows_positive_also_negative']+=bool(pos&set(neg));counts['rows_duplicate_negatives']+=len(set(neg))<len(neg)
    if not matches and len(examples)<10:examples.append(r['query'])
 result={'dataset':'KaLM-Embedding/KaLM-embedding-finetuning-data','revision':REV,'assets':assets,'counts':dict(counts),'official_train_domain_matches':dict(domains),'official_train_not_dev_rows_by_domain':dict(elig),'candidate_count_histogram':dict(pools),'query_wrappers':dict(wrappers),'unmatched_examples':examples,'scope':'All three published Multi-CPR shards; query provenance and candidate integrity only. No training selection. Positive document IDs and exact mining lineage remain unaudited.'}
 save(root/'audit.json',result);print('KALM_CPR',json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
