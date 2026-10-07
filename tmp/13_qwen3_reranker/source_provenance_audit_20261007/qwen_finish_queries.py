import sys,json,pathlib,gzip,collections
sys.path.insert(0,'/workspace/repos/qwen-margin-distillation/13_qwen3_reranker')
from broader_split import key,read,save
from audit_missing_sources import tsv_queries
from datasets import load_dataset
out=pathlib.Path('/workspace/results/qwen_margin_distill/missing_source_audit')
panels=read('/workspace/results/qwen_margin_distill/data_fast/benchmark_panels.json.gz');m=panels['provenance']['MMarcoRetrieval']['dataset'];qs=load_dataset(m['path'],'default',revision=m['revision'])['queries'];b={key(q) for q in qs['text']};dev={key(q) for q in tsv_queries(out/'google-chinese-dev.tsv').values()}
a=read(out/'audit.json');a['mMARCO']['pinned_benchmark_unique_texts']=len(b);a['mMARCO']['pinned_benchmark_texts_missing_from_google_dev']=sorted(b-dev)
for source,query in [('msmarco','what is blains'),('mmarco_chinese','团队领导的平均工资')]:
 hits={}
 for fn in ['google-english-train.tsv','google-english-dev.tsv','google-chinese-train.tsv','google-chinese-dev.tsv','helsinki-chinese-train.tsv','helsinki-chinese-dev.tsv']:
  ids=[i for i,q in tsv_queries(out/fn).items() if key(q)==key(query)]
  if ids:hits[fn]={'ids':ids,'blocked':[i for i in ids if i in set(a['mMARCO']['blocked_original_query_ids'])]}
 print('EXCLUDED_SPOT_ROW',source,query,hits)
 a.setdefault('spot_row_diagnostics',{})[source]={'query':query,'matches':hits}
save(out/'audit.json',a);print('MMARCO_UNIQUE',len(b),'MISSING_FROM_DEV',sorted(b-dev))
p=pathlib.Path('/workspace/results/qwen_margin_distill/data_fast');raw=read(p/'raw_pool.json.gz')['rows'];by={r['id']:r for r in raw};d=read(p/'dataset.json.gz');result={}
for split in ['train','validation']:
 c=collections.defaultdict(collections.Counter);ids=[]
 for g in d[split]:
  r=by[g['id']];known={key(x) for x in r['pos']+r['neg']};extra=[x for x in g['documents'] if key(x) not in known]
  c[g['source']]['queries']+=1;c[g['source']]['queries_with_added_candidates']+=bool(extra);c[g['source']]['added_candidates']+=len(extra)
  if extra:ids.append(g['id'])
 result[split]={'sources':dict(c),'affected_group_ids':ids}
save(out/'historical_supplementation_audit.json',result);print('HISTORICAL',json.dumps(result))
