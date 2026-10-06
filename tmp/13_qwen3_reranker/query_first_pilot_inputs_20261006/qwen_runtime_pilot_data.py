import gzip,hashlib,json,pathlib,sys,urllib.parse
sys.path.insert(0,'/tmp/qwen-document-first-training-smoke-small/13_qwen3_reranker')
from mixture_data import EXCLUDED,clean_row,select_split,source_name
info=json.loads(pathlib.Path('/tmp/qwen_bge_info.json').read_text())
meta=json.loads(gzip.decompress(pathlib.Path('/tmp/qwen-touche-exclusion.json.gz').read_bytes()))
blocked={k:set(meta[k]) for k in ('query_hashes','document_hashes')}
pool=[]
for path in sorted(pathlib.Path('/tmp/qwen-mixture-http-cache').glob('*.json.gz')):
    response=json.loads(gzip.decompress(path.read_bytes()))
    urls=response.get('acquisition_parquet_url')
    if not urls:continue
    if isinstance(urls,str):urls=[urls]
    config=urllib.parse.urlparse(urls[0]).path.split('/')[-2]
    if source_name(config) in EXCLUDED:continue
    for row in response['rows']:
        if row.get('truncated_cells'):continue
        item=clean_row(row['row'],config,row['row_idx'],blocked)
        if item:pool.append(item)
family_counts={f:sum(r['source']==f for r in pool) for f in {r['source'] for r in pool}}
families=sorted((f for f,n in family_counts.items() if n>=32),key=lambda f:-family_counts[f])[:12]
pool=[r for r in pool if r['source'] in families]
counts={c['config_name']:sum(s['num_examples'] for s in c['splits']) for c in info['cardData']['dataset_info'] if source_name(c['config_name']) in families}
x=select_split(pool,counts,256,48,731)
x['provenance']={'excluded_families':EXCLUDED,'touche_exclusion':meta,'purpose':'Runtime pilot from already downloaded sources, not the representative curve dataset','hub_endpoint':'https://hf-mirror.com','builder_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
path=pathlib.Path('/tmp/qwen-bge-runtime-pilot.json.gz')
path.write_bytes(gzip.compress(json.dumps(x,ensure_ascii=False).encode(),mtime=0))
print(json.dumps({'train_queries':len(x['train']),'validation_queries':len(x['validation']),'sources':families,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}))
