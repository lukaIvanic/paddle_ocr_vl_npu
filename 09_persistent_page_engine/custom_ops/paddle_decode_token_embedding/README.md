# Paddle B1 decode token embedding

This independent AscendC operator replaces the TBE/TIK `GatherV2` emitted for
the PaddleOCR-VL decode embedding. It supports only the production experiment
contract: FP16 weight `[103424, 1024]`, INT64 input IDs `[1, 1]`, and FP16
output `[1, 1, 1024]` on Ascend 910B.

The specialization is intentional. Its purpose is to make the embedding
subkernel eligible for strict TorchAir SuperKernel binary fusion.

Build and install on the Blue Zone container:

```sh
cd /workspace/repos/paddle_ocr_vl_npu
source npu-setup
bash 09_persistent_page_engine/custom_ops/paddle_decode_token_embedding/build.sh

09_persistent_page_engine/custom_ops/paddle_decode_token_embedding/build_out/\
custom_opp_ubuntu_aarch64.run \
  --quiet \
  --install-path=/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/paddle_decode_token_embedding_opp

source .runtime_cache/paddle_decode_token_embedding_opp/vendors/\
paddle_decode_token_embedding/bin/set_env.bash
```

Use a new TorchAir cache key after every rebuild. Validate the operator by
itself before accepting it as a subkernel of the full decoder SuperKernel.

```sh
export PYTHONPATH="$PWD/09_persistent_page_engine:${PYTHONPATH}"
/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python \
  09_persistent_page_engine/scripts/probes/compare_paddle_decode_token_embedding.py \
  --cache-dir .runtime_cache/paddle_decode_token_embedding_probe \
  --output tmp/09_persistent_page_engine/paddle_decode_token_embedding_probe.json
```
