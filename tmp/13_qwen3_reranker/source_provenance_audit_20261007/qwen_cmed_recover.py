import urllib.request,json,base64,hashlib,pathlib,concurrent.futures,csv,zipfile,io,gzip,unicodedata
root=pathlib.Path('/workspace/results/qwen_margin_distill/broader_v1/upstream')
files=[{'path': 'dev_candidates.zip', 'mode': '100644', 'type': 'blob', 'sha': 'e5eadeed0a96594d93cbacc9fb93555ec16ba88f', 'size': 2395726, 'url': 'https://api.github.com/repos/zhangsheng93/cMedQA2/git/blobs/e5eadeed0a96594d93cbacc9fb93555ec16ba88f'}, {'path': 'question.zip', 'mode': '100644', 'type': 'blob', 'sha': 'd2a52ca01f33b513b00c6c110dc963d69c716130', 'size': 7701487, 'url': 'https://api.github.com/repos/zhangsheng93/cMedQA2/git/blobs/d2a52ca01f33b513b00c6c110dc963d69c716130'}, {'path': 'test_candidates.zip', 'mode': '100644', 'type': 'blob', 'sha': 'ca296d4135ebf93906616f3601042bc5264ab330', 'size': 2394296, 'url': 'https://api.github.com/repos/zhangsheng93/cMedQA2/git/blobs/ca296d4135ebf93906616f3601042bc5264ab330'}, {'path': 'train_candidates.zip', 'mode': '100644', 'type': 'blob', 'sha': '728eab844e477a75f8e10dde517d222d130f9e69', 'size': 23740116, 'url': 'https://api.github.com/repos/zhangsheng93/cMedQA2/git/blobs/728eab844e477a75f8e10dde517d222d130f9e69'}]

def get(r):
 p=root/r['path']
 if p.exists():data=p.read_bytes()
 else:
  x=json.load(urllib.request.urlopen(r['url'],timeout=40));data=base64.b64decode(x['content'])
 assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==r['sha']
 p.write_bytes(data);print('CMED_VERIFIED',p.name,len(data),flush=True)
with concurrent.futures.ThreadPoolExecutor(4) as pool:list(pool.map(get,files))
def rows(fn):
 z=zipfile.ZipFile(root/fn);ns=[n for n in z.namelist() if not n.endswith('/') and not n.startswith('__MACOSX')];assert len(ns)==1
 return list(csv.reader(io.StringIO(z.read(ns[0]).decode('utf-8-sig'))))
qs={r[0]:r[1] for r in rows('question.zip')[1:]}
result={}
for split in ['train','dev','test']:
 rs=rows(split+'_candidates.zip');ids={r[0] for r in rs[1:]};assert ids<=qs.keys();result[split]={i:qs[i] for i in ids};print('CMED_SPLIT',split,len(ids),flush=True)
p=root.parent/'cmed_official_queries.json.gz';p.write_bytes(gzip.compress(json.dumps(result,ensure_ascii=False).encode(),mtime=0))
