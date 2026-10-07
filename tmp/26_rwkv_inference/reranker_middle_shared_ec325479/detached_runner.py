import subprocess
from pathlib import Path
r=Path('/workspace/repos/rwkv-cpu-reference/tmp/26_rwkv_inference/reranker_middle_shared_ec325479')
commands=['source /usr/local/bin/npu-setup; export TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 ASCEND_RT_VISIBLE_DEVICES=7; source /workspace/rwkv_reference/wkv7_build_178d16e6/install/vendors/rwkv_reference/bin/set_env.bash; source /workspace/rwkv_reference/wkv7_endpoint_d7348673/install/vendors/rwkv_endpoint/bin/set_env.bash; export PATH=/usr/local/python3.12.13/bin:$PATH; cd /workspace/repos/rwkv-cpu-reference; exec /workspace/venvs/rwkv_cpu_py312/bin/python -u /workspace/repos/rwkv-cpu-reference/26_rwkv_inference/bench_reranker_sizes.py --models /workspace/rwkv_reference/models --cases-root /workspace/repos/rwkv-cpu-reference/tmp/26_rwkv_inference --upstream /workspace/rwkv_reference/upstream/reranker --reference-build /workspace/rwkv_reference/wkv7_build_178d16e6 --build-root /workspace/rwkv_reference/wkv7_endpoint_d7348673 --download-status /workspace/rwkv_reference/large_models_download/status.json --worker --allow-shared-device --size base --dtype fp32 --repeats 10 --bucket 512 --output /workspace/repos/rwkv-cpu-reference/tmp/26_rwkv_inference/reranker_middle_shared_ec325479/probe_t512', 'source /usr/local/bin/npu-setup; export TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 ASCEND_RT_VISIBLE_DEVICES=7; source /workspace/rwkv_reference/wkv7_build_178d16e6/install/vendors/rwkv_reference/bin/set_env.bash; source /workspace/rwkv_reference/wkv7_endpoint_d7348673/install/vendors/rwkv_endpoint/bin/set_env.bash; export PATH=/usr/local/python3.12.13/bin:$PATH; cd /workspace/repos/rwkv-cpu-reference; exec /workspace/venvs/rwkv_cpu_py312/bin/python -u /workspace/repos/rwkv-cpu-reference/26_rwkv_inference/bench_reranker_sizes.py --models /workspace/rwkv_reference/models --cases-root /workspace/repos/rwkv-cpu-reference/tmp/26_rwkv_inference --upstream /workspace/rwkv_reference/upstream/reranker --reference-build /workspace/rwkv_reference/wkv7_build_178d16e6 --build-root /workspace/rwkv_reference/wkv7_endpoint_d7348673 --download-status /workspace/rwkv_reference/large_models_download/status.json --worker --allow-shared-device --size base --dtype fp32 --repeats 10 --bucket 2048 --output /workspace/repos/rwkv-cpu-reference/tmp/26_rwkv_inference/reranker_middle_shared_ec325479/probe_t2048']
rc=0
try:
 for cmd in commands:
  print('START_WORKER',cmd,flush=True)
  rc=subprocess.run(['bash','-c',cmd],timeout=1800).returncode
  if rc:break
except Exception as e:
 print(type(e).__name__,str(e),flush=True);rc=124
finally:
 (r/'exit_code.txt').write_text(str(rc)+'\n')
