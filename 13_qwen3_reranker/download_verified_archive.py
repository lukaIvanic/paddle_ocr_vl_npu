"""Resumable parallel HTTP-range download with full-length and optional MD5 verification."""
import argparse,base64,concurrent.futures,hashlib,json,os,pathlib,time,urllib.request

def main():
 p=argparse.ArgumentParser();p.add_argument('--url',required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--workers',type=int,default=4)
 args=p.parse_args();head=urllib.request.urlopen(urllib.request.Request(args.url,method='HEAD'),timeout=45)
 size=int(head.headers['Content-Length']);etag=head.headers.get('ETag');md5=head.headers.get('x-bce-meta-md5')
 args.output.parent.mkdir(exist_ok=True,parents=True);tmp=args.output.with_suffix('.partial')
 fd=os.open(tmp,os.O_CREAT|os.O_RDWR,0o600);os.ftruncate(fd,size)
 progress={};started=time.monotonic();jobs=[]
 # Per-range markers only reused with matching URL/size/ETag.
 manifest=args.output.with_suffix('.ranges.json')
 identity={'url':args.url,'size':size,'etag':etag,'workers':args.workers}
 if manifest.exists():assert json.loads(manifest.read_text())==identity
 else:manifest.write_text(json.dumps(identity))
 for i in range(args.workers):jobs.append((i,size*i//args.workers,size*(i+1)//args.workers-1))
 def get(job):
  i,start,end=job;marker=args.output.with_suffix(f'.range{i}.done')
  if marker.exists():progress[i]=end-start+1;return
  for attempt in range(3):
   try:
    req=urllib.request.Request(args.url,headers={'Range':f'bytes={start}-{end}','If-Match':etag})
    with urllib.request.urlopen(req,timeout=60) as r:
     assert r.status==206,(r.status,r.headers)
     assert r.headers['Content-Range']==f'bytes {start}-{end}/{size}',r.headers
     pos=start
     while b:=r.read(1024**2):
      assert pos+len(b)<=end+1
      view=memoryview(b)
      while view:
       n=os.pwrite(fd,view,pos);pos+=n;view=view[n:]
      progress[i]=pos-start
     assert pos==end+1
    marker.write_text('complete\n');return
   except Exception:
    if attempt==2:raise
    time.sleep(2)
 with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
  futures=[pool.submit(get,j) for j in jobs]
  while not all(f.done() for f in futures):
   done=sum(progress.values());elapsed=time.monotonic()-started;rate=done/max(elapsed,.01)
   print('TRANSFER',json.dumps({'bytes':done,'total':size,'MB_s':rate/1e6,'eta_s':(size-done)/rate if rate else None}),flush=True)
   time.sleep(10)
  for f in futures:f.result()
 os.fsync(fd);os.close(fd)
 h=hashlib.md5();sha=hashlib.sha256()
 with tmp.open('rb') as f:
  while b:=f.read(8*1024**2):h.update(b);sha.update(b)
 if md5:assert base64.b64encode(h.digest()).decode()==md5,('MD5 mismatch',h.hexdigest(),md5)
 tmp.replace(args.output)
 args.output.with_suffix('.verified.json').write_text(json.dumps(identity|{'md5':h.hexdigest(),'sha256':sha.hexdigest(),'seconds':time.monotonic()-started}))
 print('VERIFIED',sha.hexdigest(),flush=True)
if __name__=='__main__':main()
