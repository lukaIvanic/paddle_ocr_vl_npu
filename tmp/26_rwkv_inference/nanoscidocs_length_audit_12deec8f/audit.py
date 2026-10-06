from pathlib import Path
import concurrent.futures,urllib.request,urllib.parse,json,importlib.util,hashlib,math,statistics
root=Path('/tmp/rwkv_nanoscidocs_audit')
repo='zeta-alpha-ai/NanoSCIDOCS'
def fetch(url):
 with urllib.request.urlopen(url,timeout=30) as r: return json.load(r)
meta=fetch('https://huggingface.co/api/datasets/'+repo)
revision=meta['sha']
def page(config,offset):
 url='https://datasets-server.huggingface.co/rows?'+urllib.parse.urlencode(dict(dataset=repo,config=config,split='train',offset=offset,length=100))
 data=fetch(url)
 if data.get('dataset_git_revision') not in (None,revision): raise ValueError('Dataset revision mismatch')
 return [x['row'] for x in data['rows']]
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
 corpus=[x for rows in pool.map(lambda off:page('corpus',off),range(0,2210,100)) for x in rows]
queries=page('queries',0)
assert len(corpus)==2210 and len(queries)==50
assert fetch('https://huggingface.co/api/datasets/'+repo)['sha']==revision
assets=Path('/tmp/rwkv-reranker-inspect/embedding/eval/tokenizer')
module_path=assets/'rwkv_tokenizer.py'
vocab_path=assets/'rwkv_vocab_v20230424.txt'
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(vocab_path)=='e6dee3d4e31b4d5c40ac99508ac6c701ceef4bed681bf2167ce9a908552bca89'
spec=importlib.util.spec_from_file_location('rwkv_tokenizer',module_path)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
tok=mod.TRIE_TOKENIZER(str(vocab_path))
instruction='Instruct: Given a query, retrieve documents that answer the query\nQuery: {query}'
rows={}
for role,data in [('corpus',corpus),('queries',queries)]:
 output=[]
 for row in data:
  text=((row.get('title') or '')+' '+(row.get('text') or '')).strip() if role=='corpus' else instruction.format(query=row['text'])
  n=len(tok.encode(text));clipped=min(n,2048*8)
  prepared=(clipped+1)*4 if clipped<512 else clipped+4*math.ceil(clipped/2048) if clipped>2044 else clipped+4
  aligned=(prepared+15)//16*16
  output.append({'id':row['_id'],'raw_tokens':n,'prepared_tokens':aligned})
 rows[role]=output
summary={'dataset':repo,'revision':revision,'upstream_commit':'3c306736c58550f4be6d384be068512ba9bfbd72','vocabulary_sha256':sha(vocab_path),'tokenizer_sha256':sha(module_path),'ctx_len':2048,'eos_chunk_size':512,'query_instruction':instruction,'corpus_text':'(title + space + text).strip(); dataset has no title field','groups':{}}
for role,output in rows.items():
 counts=sorted(x['raw_tokens'] for x in output);prepared=sorted(x['prepared_tokens'] for x in output)
 summary['groups'][role]={'count':len(output),'raw_min':counts[0],'raw_median':statistics.median(counts),'raw_p95':counts[math.ceil(.95*len(counts))-1],'raw_max':counts[-1],'prepared_min':prepared[0],'prepared_median':statistics.median(prepared),'prepared_p95':prepared[math.ceil(.95*len(prepared))-1],'prepared_max':prepared[-1],'prepared_over_2048':sum(x>2048 for x in prepared),'raw_over_2048':sum(x>2048 for x in counts),'longest':sorted(output,key=lambda x:x['prepared_tokens'],reverse=True)[:5]}
(root/'input_rows.json').write_text(json.dumps({'revision':revision,'corpus':corpus,'queries':queries},ensure_ascii=False))
summary['input_rows_sha256']=sha(root/'input_rows.json')
(root/'lengths.json').write_text(json.dumps(rows,indent=2)+'\n')
(root/'result.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
