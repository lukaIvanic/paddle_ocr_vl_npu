import concurrent.futures,hashlib,pathlib,subprocess,time,json
source=pathlib.Path('/tmp/qwen-bge-filtered-mixture.json.gz').read_bytes()
ssh=['ssh','-F','/home/luka/Documents/Codex/2026-10-01/can-you-connect-to-my-mac/work/ssh-blue-zone/config','-o','ControlPath=none','-o','ControlMaster=no','-o','ConnectTimeout=10','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=3','blue_zone_npu_server']
container='research_vllm_ascend_023_external_workspace'
subprocess.run(ssh+[f'docker exec {container} mkdir -p /tmp/qwen-mixture-chunks'],check=True,timeout=30,capture_output=True)
chunks=[source[i:i+512*1024] for i in range(0,len(source),512*1024)]
start=time.monotonic()
def upload(item):
    index,data=item
    path=f'/tmp/qwen-mixture-chunks/{index:04d}'
    command=f"docker exec -i {container} bash -c 'cat > {path} && sha256sum {path}'"
    for attempt in range(3):
        t=time.monotonic()
        try:
            response=subprocess.run(ssh+[command],input=data,capture_output=True,timeout=100)
            if response.returncode:raise RuntimeError(response.stderr.decode()[-300:])
            checksum=response.stdout.decode().split()[0]
            assert checksum==hashlib.sha256(data).hexdigest()
            return {'chunk':index,'bytes':len(data),'seconds':time.monotonic()-t,'sha256':checksum}
        except Exception:
            if attempt==2:raise
            time.sleep(2*(attempt+1))
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    futures=[pool.submit(upload,item) for item in enumerate(chunks)]
    result=[]
    for future in concurrent.futures.as_completed(futures):
        item=future.result();result.append(item);print('UPLOADED',json.dumps(item),flush=True)
command=f"docker exec {container} bash -c 'cat /tmp/qwen-mixture-chunks/[0-9][0-9][0-9][0-9] > /tmp/qwen-bge-filtered-mixture.incoming.gz && sha256sum /tmp/qwen-bge-filtered-mixture.incoming.gz'"
response=subprocess.run(ssh+[command],capture_output=True,timeout=30,check=True)
checksum=response.stdout.decode().split()[0]
assert checksum==hashlib.sha256(source).hexdigest()
report={'bytes':len(source),'chunks':len(chunks),'sha256':checksum,'seconds':time.monotonic()-start,'parts':sorted(result,key=lambda r:r['chunk'])}
pathlib.Path('/tmp/qwen-mixture-upload.json').write_text(json.dumps(report,indent=2))
print('TRANSFER_VERIFIED',json.dumps({k:v for k,v in report.items() if k!='parts'}),flush=True)
