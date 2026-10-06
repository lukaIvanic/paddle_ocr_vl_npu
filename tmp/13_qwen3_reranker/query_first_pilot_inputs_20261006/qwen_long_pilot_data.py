import gzip,hashlib,json,pathlib,sys,urllib.parse
sys.path.insert(0,'/tmp/qwen-document-first-training-smoke-small/13_qwen3_reranker')
from mixture_data import clean_row,text_hash
from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained("/workspace/models/Qwen3-Reranker-0.6B",local_files_only=True)
x=json.loads(gzip.decompress(pathlib.Path('/tmp/qwen-bge-runtime-pilot.json.gz').read_bytes()))
meta=x['provenance']['touche_exclusion']
blocked={k:set(meta[k]) for k in ('query_hashes','document_hashes')}
vq={text_hash(r['query']) for r in x['validation']}
vd={text_hash(d) for r in x['validation'] for d in r['documents']}
long=[]
for path in sorted(pathlib.Path('/tmp/qwen-mixture-http-cache').glob('*.json.gz')):
    response=json.loads(gzip.decompress(path.read_bytes()))
    urls=response.get('acquisition_parquet_url')
    if not urls:continue
    if isinstance(urls,str):urls=[urls]
    config=urllib.parse.urlparse(urls[0]).path.split('/')[-2]
    if not config.startswith('mldr_'):continue
    for row in response['rows']:
        r=clean_row(row['row'],config,row['row_idx'],blocked)
        if not r or text_hash(r['query']) in vq:continue
        docs=[r['positives'][0],r['negatives'][0]]
        if min(map(len,docs))<10000 or vd & {text_hash(d) for d in docs}:continue
        if min(len(tokenizer.encode(d,add_special_tokens=False)) for d in docs)<8192:continue
        long.append({k:r[k] for k in ('id','source','config','query')}|{'documents':docs,'labels':[1,0]})
        if len(long)==2:break
    if len(long)==2:break
assert long
x['train']=long
info=json.loads(pathlib.Path('/tmp/qwen_bge_info.json').read_text())
sizes={c['config_name'].split('_len-')[0]:0 for c in info['cardData']['dataset_info']}
for c in info['cardData']['dataset_info']:sizes[c['config_name'].split('_len-')[0]]+=sum(s['num_examples'] for s in c['splits'])
x['distribution']={s:{'released_rows':sizes[s],'train_queries':sum(r['source']==s for r in long),'validation_queries':sum(r['source']==s for r in x['validation'])} for s in {r['source'] for r in long+x['validation']}}
x['provenance']['purpose']='Worst-length memory and backward preflight from completed MLDR downloads; not a quality run'
x['provenance']['builder_sha256']=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
p=pathlib.Path('/tmp/qwen-bge-long-pilot.json.gz')
p.write_bytes(gzip.compress(json.dumps(x,ensure_ascii=False).encode(),mtime=0))
print(json.dumps({'train_queries':len(long),'characters_per_pair':[[len(d) for d in r['documents']] for r in long],'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))
