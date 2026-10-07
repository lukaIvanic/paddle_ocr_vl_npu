"""Audit official DuReader query splits and BGE row provenance (no training)."""
import argparse,collections,concurrent.futures,hashlib,json,pathlib
from broader_split import key,read,save,remote_parquet,MIRROR,REVISION
from audit_missing_sources import tsv_queries

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,required=True)
 args=p.parse_args();out=args.output;root=out/'dureader/extracted/dureader-retrieval-baseline-dataset'
 queries=set();pos=collections.Counter();neg=collections.Counter();n=0
 with (root/'train/cross.train.tsv').open() as f:
  for line in f:
   fields=line.rstrip('\n').split('\t');assert len(fields)==4
   q,_,d,label=fields;assert label in ['0','1'];q=key(q);queries.add(q)
   (pos if label=='1' else neg)[q]+=1;n+=1
 dev={key(q):i for q,i in read(root/'dev/q2qid.dev.json').items()}
 from datasets import load_dataset
 panels=read('/workspace/results/qwen_margin_distill/data_fast/benchmark_panels.json.gz')
 meta=panels['provenance']['DuRetrieval']['dataset']
 qs=load_dataset(meta['path'],'default',revision=meta['revision'])['queries'];bq={key(q) for q in qs['text']}
 audit={'official_archive':read(out/'dureader/verified.tar.verified.json'),
  'cross_training_rows':n,'training_unique_queries':len(queries),'official_dev_queries':len(dev),
  'pinned_benchmark_queries':len(qs),'pinned_benchmark_unique_texts':len(bq),
  'benchmark_texts_in_official_dev':len(bq&dev.keys()),'benchmark_texts_in_training':len(bq&queries),
  'train_dev_query_overlap':len(queries&dev.keys()),'training_label_rows':{'positive':sum(pos.values()),'negative':sum(neg.values())},
  'negatives_per_query_row_count_histogram':dict(collections.Counter(neg.values())),
  'candidate_provenance':'Official baseline dense-retriever negatives; BGE uses its own released pools, not assumed identical'}
 save(out/'dureader/train_query_registry.json.gz',{'train':sorted(queries),'dev':dev,'archive_sha256':audit['official_archive']['sha256']})
 save(out/'dureader/audit.json',audit)
 print('DUREADER',json.dumps(audit),flush=True)
 # A lineage spot check, deliberately NOT a representative training sample or whole-release clearance.
 metadata=read('/tmp/qwen_bge_info.json');cpr=read(out/'audit.json')
 en=tsv_queries(out/'google-english-train.tsv');zhg=tsv_queries(out/'google-chinese-train.tsv');zhh=tsv_queries(out/'helsinki-chinese-train.tsv')
 blocked=set(cpr['mMARCO']['blocked_original_query_ids'])
 tables={'msmarco':{key(q) for i,q in en.items() if i not in blocked},
  'mmarco_chinese':{key(q) for qs in [zhg,zhh] for i,q in qs.items() if i not in blocked},'dureader':queries-bq-set(dev)}
 def inspect(source):
  names=sorted(r['rfilename'] for r in metadata['siblings'] if r['rfilename'].startswith(source+'_len-0-500/') and r['rfilename'].endswith('.parquet'))
  fn=names[0];url=f'https://hf-mirror.com/datasets/{MIRROR}/resolve/{REVISION}/{fn}?download=true'
  pf=remote_parquet(url);rows=pf.read_row_group(0,columns=['query']).slice(0,512).to_pylist()
  eligible=sum(key(r['query']) in tables[source] for r in rows)
  result={'source':source,'sample':'first up to 512 rows of first short-length Parquet row group; provenance spot check only',
   'file':fn,'revision':REVISION,'queries_checked':len(rows),'verified_train_and_not_blocked':eligible,
   'unmatched_or_blocked_examples':[r['query'] for r in rows if key(r['query']) not in tables[source]][:10]}
  print('BGE_MEMBERSHIP',json.dumps(result,ensure_ascii=False),flush=True);return result
 with concurrent.futures.ThreadPoolExecutor(3) as pool:checks=list(pool.map(inspect,tables))
 save(out/'bge_missing_family_spot_checks.json',checks)
if __name__=='__main__':main()
