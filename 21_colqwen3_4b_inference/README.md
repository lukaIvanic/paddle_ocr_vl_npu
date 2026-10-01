# 21 — Ops-ColQwen3-4B Hugging Face baseline

Direct checkpoint-supplied Hugging Face `AutoModel` and `AutoProcessor` on
Ascend NPU. No vLLM, custom model implementation, TorchAir compilation, NZ
conversion, quantization, or processor-resolution overrides. FP16 and HF eager
attention are explicit; this is a correctness anchor, not an optimized path.

Model: https://huggingface.co/OpenSearch-AI/Ops-Colqwen3-4B

The checkpoint's reviewed local Python code is loaded with
`trust_remote_code=True, local_files_only=True`. Its processor owns query
augmentation and image prompts; its model owns projection, masking and
normalization. Full-dimensional embeddings (2560) are retained. MaxSim uses
the supplied processor with an explicit NPU device, avoiding its auto-CPU fallback.
The model card's CUDA FlashAttention-2 example is not used on Ascend.

## Earlier work

`qwen_vllm_research/02_ops_colqwen3_embedding_smoke` recorded a 910B vLLM-Ascend
query/synthetic-image smoke with block size 128. It was not a Hugging Face
baseline or retrieval evaluation. The cached model remains at
`/workspace/models/Ops-Colqwen3-4B`. No 310P result is claimed here.

## Run on 910B

Use a healthy, free device selected by `npu-setup`; check health before running.
The current container can be reached through host SSH plus `docker exec`;
the historical port-22021 SSH shortcut may be unavailable. Source changes
still follow local commit/push and server pull only.

```sh
source npu-setup
cd /workspace/repos/paddle_ocr_vl_npu
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/workspace/.cache/huggingface
RUN_ROOT="tmp/21_colqwen3_4b_inference/hf_smoke_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN_ROOT"
set -o pipefail
/workspace/venvs/mineru_pro_vllm_py312/bin/python \
  21_colqwen3_4b_inference/run_hf_baseline.py \
  --model /workspace/models/Ops-Colqwen3-4B \
  --output-dir "$RUN_ROOT/output" --hash-weights \
  2>&1 | tee "$RUN_ROOT/run.log"
RUN_EXIT=${PIPESTATUS[0]}
printf '%s\n' "$RUN_EXIT" > "$RUN_ROOT/exit_code.txt"
```

This initially reuses the existing Python environment without modifying its
packages. Any necessary dependency changes must use a separate environment.

Default inputs are two text queries and two real committed document crops.
Images run individually; queries form one batch. No synthetic image, crop
resize override, model modification, or CPU model fallback is permitted.
`--images` and `--queries` accept explicit alternative inputs.

Outputs include source/model/input hashes, package versions, exact loading
diagnostics, embedding dimensions and norms, repeat differences, MaxSim scores,
and CPU `.pt` copies of processor tensors and embeddings for future parity.
First forward and three warm repeats are reported separately. Forward timing
uses NPU synchronization and excludes preprocessing, transfer, and saving.
No ranking/accuracy conclusion follows from these smoke inputs or scores.

Reject missing/unexpected checkpoint weights, nonfinite/empty embeddings,
unit-norm error over 0.02, or repeat max-absolute drift over 0.001. Record
failures rather than silently patching checkpoint code or changing attention.

## Status

Implementation prepared; NPU validation pending. Existing experiments and their
environments are unchanged.
