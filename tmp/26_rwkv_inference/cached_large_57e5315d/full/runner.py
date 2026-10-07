import subprocess,time,json
from pathlib import Path
run=Path('/workspace/repos/rwkv-cpu-reference/tmp/26_rwkv_inference/cached_large_57e5315d/full');base=run.parent
children=[];logs=[]
try:
 while not (base/'exit_code.txt').exists():
  if (base/'interrupted.json').exists():raise RuntimeError('Gate interrupted')
  time.sleep(5)
 assert (base/'exit_code.txt').read_text().strip()=='0'
 assert json.loads((base/'smoke/result.json').read_text())['all_checks_passed']
 for i,cmd in enumerate(['source /usr/local/bin/npu-setup; export TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 ASCEND_RT_VISIBLE_DEVICES=1; source /workspace/rwkv_reference/wkv7_build_178d16e6/install/vendors/rwkv_reference/bin/set_env.bash; source /workspace/rwkv_reference/wkv7_endpoint_d7348673/install/vendors/rwkv_endpoint/bin/set_env.bash; source /workspace/rwkv_reference/rwkv_aiv32_9db3477a/install/vendors/rwkv_aiv32/bin/set_env.bash; export PATH=/usr/local/python3.12.13/bin:$PATH; cd /workspace/repos/rwkv-matrix-a3822c41; exec /workspace/venvs/rwkv_cpu_py312/bin/python -u /workspace/repos/rwkv-matrix-a3822c41/26_rwkv_inference/run_cached_nanobeir.py evaluate --reference /workspace/repos/rwkv-cpu-reference/tmp/26_rwkv_inference/nanobeir_large_bucketed_afde9124/probe --data-root /workspace/rwkv_reference/nanobeir_bm25_v5_1_2 --runtime /workspace/rwkv_reference/runtime --models /workspace/rwkv_reference/models --reference-build /workspace/rwkv_reference/wkv7_build_178d16e6 --build-root /workspace/rwkv_reference/wkv7_endpoint_d7348673 --vector-build /workspace/rwkv_reference/rwkv_aiv32_9db3477a --prepared /workspace/repos/rwkv-cpu-reference/tmp/26_rwkv_inference/cached_large_f0b47337/prepared --output /workspace/repos/rwkv-cpu-reference/tmp/26_rwkv_inference/cached_large_57e5315d/full/worker_0 --gate /workspace/repos/rwkv-cpu-reference/tmp/26_rwkv_inference/cached_large_57e5315d/smoke --tasks NanoSCIDOCSRetrieval NanoNFCorpusRetrieval NanoFEVERRetrieval NanoNQRetrieval NanoMSMARCORetrieval', 'source /usr/local/bin/npu-setup; export TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 ASCEND_RT_VISIBLE_DEVICES=3; source /workspace/rwkv_reference/wkv7_build_178d16e6/install/vendors/rwkv_reference/bin/set_env.bash; source /workspace/rwkv_reference/wkv7_endpoint_d7348673/install/vendors/rwkv_endpoint/bin/set_env.bash; source /workspace/rwkv_reference/rwkv_aiv32_9db3477a/install/vendors/rwkv_aiv32/bin/set_env.bash; export PATH=/usr/local/python3.12.13/bin:$PATH; cd /workspace/repos/rwkv-matrix-a3822c41; exec /workspace/venvs/rwkv_cpu_py312/bin/python -u /workspace/repos/rwkv-matrix-a3822c41/26_rwkv_inference/run_cached_nanobeir.py evaluate --reference /workspace/repos/rwkv-cpu-reference/tmp/26_rwkv_inference/nanobeir_large_bucketed_afde9124/probe --data-root /workspace/rwkv_reference/nanobeir_bm25_v5_1_2 --runtime /workspace/rwkv_reference/runtime --models /workspace/rwkv_reference/models --reference-build /workspace/rwkv_reference/wkv7_build_178d16e6 --build-root /workspace/rwkv_reference/wkv7_endpoint_d7348673 --vector-build /workspace/rwkv_reference/rwkv_aiv32_9db3477a --prepared /workspace/repos/rwkv-cpu-reference/tmp/26_rwkv_inference/cached_large_f0b47337/prepared --output /workspace/repos/rwkv-cpu-reference/tmp/26_rwkv_inference/cached_large_57e5315d/full/worker_1 --gate /workspace/repos/rwkv-cpu-reference/tmp/26_rwkv_inference/cached_large_57e5315d/smoke --tasks NanoClimateFeverRetrieval NanoSciFactRetrieval NanoFiQA2018Retrieval NanoHotpotQARetrieval NanoDBPediaRetrieval NanoQuoraRetrieval']):
  log=(run/f'worker_{i}.log').open('w');logs.append(log)
  children.append(subprocess.Popen(['bash','-c',cmd],stdout=log,stderr=subprocess.STDOUT))
 print('STARTED',[(p.pid) for p in children],flush=True)
 while any(p.poll() is None for p in children):
  if any(p.poll() not in [None,0] for p in children):raise RuntimeError('Worker failed')
  time.sleep(5)
 assert all(p.wait()==0 for p in children)
 results=[json.loads((run/f'worker_{i}/result.json').read_text()) for i in range(2)]
 assert all(r['all_checks_passed'] for r in results)
 tasks=sum([r['tasks'] for r in results],[]);assert len(tasks)==11 and sum(t['pairs'] for t in tasks)==57688
 summary=dict(tasks=tasks,all_checks_passed=True,mean_ndcg={m:sum(t['ndcg'][m] for t in tasks)/11 for m in ['cached','uncached']},cache_build_seconds_per_worker=[sum(t['cache_build_seconds'] for t in r['tasks']) for r in results],scoring_seconds_per_worker={m:[sum(t['scoring_seconds'][m] for t in r['tasks']) for r in results] for m in ['cached','uncached']},worker_total_seconds=[r['total_seconds'] for r in results])
 (run/'result.json').write_text(json.dumps(summary,indent=2));rc=0
except Exception as e:
 print(type(e).__name__,str(e),flush=True);rc=1
finally:
 for p in children:
  if p.poll() is None:p.terminate();p.wait(timeout=30)
 for log in logs:log.close()
 (run/'exit_code.txt').write_text(str(rc))
