import subprocess,time,json
from pathlib import Path
run=Path('/workspace/repos/rwkv-cpu-reference/tmp/26_rwkv_inference/nanobeir_large_bucketed_afde9124')
probes=['/workspace/repos/rwkv-cpu-reference/tmp/26_rwkv_inference/reranker_shape_matrix_300d63a0/b4_b1_t512/probe', '/workspace/repos/rwkv-cpu-reference/tmp/26_rwkv_inference/reranker_shape_matrix_300d63a0/b4_b1_t2048/probe']
try:
 for name in probes:
  p=Path(name)
  while not (p.parent/"exit_code.txt").exists():time.sleep(10)
  assert (p.parent/"exit_code.txt").read_text().strip()=="0"
  assert json.loads((p/"result.json").read_text())["all_checks_passed"]
 print("START_SUITE",flush=True)
 rc=subprocess.run(["bash","-c",'source /usr/local/bin/npu-setup; export TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1; source /workspace/rwkv_reference/wkv7_build_178d16e6/install/vendors/rwkv_reference/bin/set_env.bash; source /workspace/rwkv_reference/wkv7_endpoint_d7348673/install/vendors/rwkv_endpoint/bin/set_env.bash; source /workspace/rwkv_reference/rwkv_aiv32_9db3477a/install/vendors/rwkv_aiv32/bin/set_env.bash; export PATH=/usr/local/python3.12.13/bin:$PATH; cd /workspace/repos/rwkv-matrix-a3822c41; exec /workspace/venvs/rwkv_cpu_py312/bin/python -u /workspace/repos/rwkv-matrix-a3822c41/26_rwkv_inference/run_nanobeir_reranker.py --size large --inference-backend bucketed --protocol bm25-positives --dtype fp16 --devices 1 3 --reranker /workspace/rwkv_reference/models/rwkv1b3-reranker.pth --checkpoint /workspace/rwkv_reference/models/rwkv1b4-emb-curriculum.pth --runtime /workspace/rwkv_reference/runtime --upstream /workspace/rwkv_reference/upstream/reranker --build-root /workspace/rwkv_reference/wkv7_endpoint_d7348673 --reference-build /workspace/rwkv_reference/wkv7_build_178d16e6 --data-root /workspace/rwkv_reference/nanobeir_bm25_v5_1_2 --candidates-root /workspace/repos/rwkv-cpu-reference/tmp/26_rwkv_inference/nanobeir_eager_b4_fp16_c3166fd1/probe --batch-evidence /workspace/repos/rwkv-cpu-reference/tmp/26_rwkv_inference/nanobeir_large_bm25_dp2_fp32_9d712af2/batch_gate.json --warm-cache-from /workspace/repos/rwkv-cpu-reference/tmp/26_rwkv_inference/reranker_shape_matrix_300d63a0/b4_b1_t512/probe --prepared-reference /workspace/repos/rwkv-cpu-reference/tmp/26_rwkv_inference/nanobeir_large_bm25_dp2_fp32_9d712af2/probe --short-batch-seconds 0.1909257249935763 --long-intercept-seconds 0.023207223326608073 --long-per-token-seconds 0.0003275751985682973 --output /workspace/repos/rwkv-cpu-reference/tmp/26_rwkv_inference/nanobeir_large_bucketed_afde9124/probe --bucket-probes /workspace/repos/rwkv-cpu-reference/tmp/26_rwkv_inference/reranker_shape_matrix_300d63a0/b4_b1_t512/probe /workspace/repos/rwkv-cpu-reference/tmp/26_rwkv_inference/reranker_shape_matrix_300d63a0/b4_b1_t2048/probe --vector-build /workspace/rwkv_reference/rwkv_aiv32_9db3477a'],timeout=7200).returncode
except Exception as e:
 print(type(e).__name__,str(e),flush=True);rc=1
(run/"exit_code.txt").write_text(str(rc))
