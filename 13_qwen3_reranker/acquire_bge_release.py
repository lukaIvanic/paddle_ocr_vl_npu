"""Download the pinned author archive in resumable, hash-checked ranges."""
import argparse,concurrent.futures,hashlib,json,os,pathlib,time,urllib.request
PIN=json.loads((pathlib.Path(__file__).parent/'bge_reference/PIN.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(8*1024**2):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--workers',type=int,default=8);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 meta=PIN['archive'];dest=a.output/meta['path'];expected=meta['lfs']['oid'];size=meta['size']
 if dest.exists():
  assert dest.stat().st_size==size and sha(dest)==expected;print('ALREADY_VERIFIED',str(dest),flush=True);return
 url=f"https://hf-mirror.com/datasets/{PIN['dataset']}/resolve/{PIN['dataset_revision']}/{meta['path']}?download=true"
 parts=a.output/'parts';parts.mkdir(exist_ok=True);chunk=256*1024**2;started=time.monotonic();progress={}
 def get(i):
  start=i*chunk;end=min(start+chunk,size)-1;final=parts/f'{i:04d}.part';info=final.with_suffix('.json')
  if final.exists() and info.exists():
   m=json.loads(info.read_text());assert m['release_sha256']==expected and final.stat().st_size==end-start+1 and sha(final)==m['sha256'];progress[i]=end-start+1;return
  for attempt in range(5):
   try:
    req=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}','User-Agent':'Mozilla/5.0'})
    with urllib.request.urlopen(req,timeout=60) as r,final.with_suffix('.partial').open('wb') as f:
     assert r.status==206 and r.headers['Content-Range']==f'bytes {start}-{end}/{size}',dict(r.headers)
     h=hashlib.sha256();n=0
     while b:=r.read(1024**2):f.write(b);h.update(b);n+=len(b);progress[i]=n
     assert n==end-start+1
    final.with_suffix('.partial').replace(final);info.write_text(json.dumps({'release_sha256':expected,'sha256':h.hexdigest()}));return
   except Exception as e:
    print('RETRY',i,attempt,repr(e),flush=True)
    if attempt==4:raise
    time.sleep(2)
 with concurrent.futures.ThreadPoolExecutor(a.workers) as pool:
  fs=[pool.submit(get,i) for i in range((size+chunk-1)//chunk)]
  while not all(f.done() for f in fs):
   done=sum(progress.values());elapsed=time.monotonic()-started;rate=done/max(elapsed,.01)
   print('TRANSFER',json.dumps({'bytes':done,'total':size,'MB_s':rate/1e6,'eta_s':(size-done)/rate if rate else None}),flush=True);time.sleep(10)
  for f in fs:f.result()
 h=hashlib.sha256()
 with dest.with_suffix('.partial').open('wb') as f:
  for i in range(len(fs)):
   with (parts/f'{i:04d}.part').open('rb') as src:
    while b:=src.read(8*1024**2):f.write(b);h.update(b)
 assert h.hexdigest()==expected,(h.hexdigest(),expected)
 dest.with_suffix('.partial').replace(dest)
 (a.output/'release_verified.json').write_text(json.dumps(PIN|{'download_url':url,'seconds':time.monotonic()-started,'sha256':h.hexdigest()},indent=2))
 print('VERIFIED',h.hexdigest(),flush=True)
if __name__=='__main__':main()
