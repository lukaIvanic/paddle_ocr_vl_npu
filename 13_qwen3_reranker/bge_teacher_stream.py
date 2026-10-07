"""Hash-bound teacher chunks; consumers never train on missing or provisional targets."""
import json,math,time
from pathlib import Path
import hashlib

def read(path):
 return json.loads(Path(path).read_bytes())

def digest(path):
 return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def group_signature(group):
 import hashlib
 fields={k:group[k] for k in ('id','query','instruction','documents')}
 return hashlib.sha256(json.dumps(fields,ensure_ascii=False,sort_keys=True).encode()).hexdigest()

class TeacherStream:
 def __init__(self,path,teacher,groups,timeout=600):
  self.path=Path(path);self.teacher=teacher;self.sha=digest(self.path);self.timeout=timeout;self.cache={};self.consumed={}
  self.directory=self.path.parent/teacher['stream']['directory'];size=teacher['stream']['groups_per_chunk']
  self.group_chunk={g['id']:i//size for i,g in enumerate(groups)}
  self.signatures={g['id']:group_signature(g) for g in groups}
 def get(self,key):
  failure=self.path.parent/'failure.json'
  if failure.exists():raise RuntimeError(f'Teacher failed: {read(failure)}')
  index=self.group_chunk[key]
  if index not in self.cache:
   path=self.directory/f'{index:06d}.json';started=time.monotonic();last_log=0
   while not path.exists():
    failure=self.path.parent/'failure.json'
    if failure.exists():raise RuntimeError(f'Teacher failed: {read(failure)}')
    if time.monotonic()-started>self.timeout:raise TimeoutError(f'Teacher chunk {index} unavailable for {self.timeout}s')
    if time.monotonic()-last_log>60:
     print('WAIT_TEACHER',json.dumps({'chunk':index,'seconds':time.monotonic()-started}),flush=True);last_log=time.monotonic()
    time.sleep(2)
   chunk=read(path)
   assert chunk['teacher_manifest_sha256']==self.sha
   assert chunk['dataset_sha256']==self.teacher['dataset_sha256']
   assert chunk['chunk']==index
   expected={k for k,v in self.group_chunk.items() if v==index}
   assert set(chunk['scores'])==set(chunk['group_signatures'])==expected
   for k,values in chunk['scores'].items():
    assert chunk['group_signatures'][k]==self.signatures[k]
    assert len(values)==8 and all(math.isfinite(v) for v in values)
   self.cache[index]=chunk['scores'];self.consumed[str(index)]=digest(path)
  return self.cache[index][key]
