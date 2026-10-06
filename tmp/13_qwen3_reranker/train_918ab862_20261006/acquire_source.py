import concurrent.futures,hashlib,json,urllib.parse,urllib.request
from pathlib import Path
dataset="hotchpotch/bge-m3-data-finetune-unified"
revision="b51dd24cbce7d89255911410ef74a36e07bfbab9"
config="pubmed_qa_labeled_len-0-500"
def fetch(offset):
 url="https://datasets-server.huggingface.co/rows?"+urllib.parse.urlencode(dict(dataset=dataset,config=config,split="train",offset=offset,length=100))
 raw=urllib.request.urlopen(url,timeout=30).read();d=json.loads(raw)
 assert not d.get("partial"),"Partial source response"
 return {"offset":offset,"url":url,"sha256":hashlib.sha256(raw).hexdigest(),"rows":d["rows"]}
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as p: responses=list(p.map(fetch,range(0,500,100)))
info=json.load(urllib.request.urlopen("https://huggingface.co/api/datasets/"+dataset,timeout=30))
assert info["sha"]==revision,"Source changed during acquisition"
rows=[dict(source_config=config,source_row_index=r["row_idx"],**r["row"]) for page in responses for r in page["rows"] if not r.get("truncated_cells")]
result={"provenance":{"original_dataset":"Shitao/bge-m3-data","mirror_dataset":dataset,"mirror_revision":revision,"mirror_claim":"format-only repack; not independently verified against 24 GB original archive","source_allowlist":[config],"benchmark_contracts":["Qwen MTEB-R: MTEB(eng, v2) retrieval","Qwen CMTEB-R: MTEB(cmn, v1) retrieval"],"filter_scope":"source-family exclusion, plus run-specific train/eval query and document deduplication; not exhaustive cross-benchmark text decontamination","responses":[{k:v for k,v in x.items() if k!="rows"} for x in responses]},"rows":rows}
Path("/tmp/qwen_bge_pubmed_smoke_source.json").write_text(json.dumps(result,ensure_ascii=False)+"\n")
print(json.dumps({"rows":len(rows),"path":"/tmp/qwen_bge_pubmed_smoke_source.json","bytes":Path("/tmp/qwen_bge_pubmed_smoke_source.json").stat().st_size}))
