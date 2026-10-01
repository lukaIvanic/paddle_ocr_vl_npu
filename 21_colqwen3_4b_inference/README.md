# 21 — Ops-ColQwen3-4B HF reference and local eager model

The reference uses checkpoint-supplied Hugging Face `AutoModel` and `AutoProcessor`
on Ascend NPU. That lane has no custom model implementation, vLLM, TorchAir compilation, NZ
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

Prepare the experiment-owned environment once:

```sh
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  bash 21_colqwen3_4b_inference/setup_environment.sh
```

This inherits the container's torch/torch_npu binaries but locally pins
Transformers 4.57.1 (the checkpoint's recorded version), tokenizers 0.22.2,
HF Hub 0.36.2 and PyArrow 21.0.0. Existing environments are unchanged. Inherited
vLLM packages require Transformers 5.5.4 and are deliberately not used here;
this is an HF-only environment, not a vLLM environment.

Transformers 5.5.4 was tested: text inference worked, but image inference failed
because the checkpoint wrapper does not forward the newly required
`mm_token_type_ids`. Pinning the reference version avoids patching the model.

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
/workspace/venvs/colqwen3_hf_py312/bin/python \
  21_colqwen3_4b_inference/run_hf_baseline.py \
  --model /workspace/models/Ops-Colqwen3-4B \
  --output-dir "$RUN_ROOT/output" --hash-weights \
  2>&1 | tee "$RUN_ROOT/run.log"
RUN_EXIT=${PIPESTATUS[0]}
printf '%s\n' "$RUN_EXIT" > "$RUN_ROOT/exit_code.txt"
```

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

Validated on one **910B2 (physical device 7), 2026-10-01**. Both the two-crop
smoke and a two-full-page ViDoRe smoke passed with FP16, Transformers 4.57.1,
checkpoint-supplied model/processor, and true uncompiled eager attention.
All 715 checkpoint tensors matched the expected keys/shapes. Embeddings were
finite, approximately unit-normalized, and bit-exact across three warm repeats.

The two original ViDoRe pages took 0.4734 s and 0.4581 s per warm image forward
(B1); preprocessing, transfers and scoring are excluded. This is **not** a
full-corpus throughput measurement or retrieval accuracy evaluation.

See [validated results and caveats](references/RESULTS.md),
[full-page run JSON](references/vidore_smoke_910b/result.json), and
[crop run JSON](references/hf4571_smoke_910b/result.json).
Processor inputs and embeddings remain on the server for future parity tests.
Existing experiments and their environments are unchanged.

Target evaluation dataset and download procedure: [ViDoRe v3](VIDORE_V3.md).

## Local modeling replacement

`local_modeling_colqwen3.py` owns the complete still-image/text embedding forward
in ordinary PyTorch, with no Transformers imports. `config.py` validates the
specific checkpoint architecture instead of guessing missing model dimensions.
The unchanged HF processor remains the preprocessing/tokenization boundary.

The implementation preserves the 24-layer D64 vision tower, learned spatial
position interpolation, 2×2 merger, DeepStack taps at blocks 5/11/17 and their
injection after text layers 0/1/2, 36-layer causal GQA text backbone with Q/K
RMSNorm, interleaved MRoPE, and the learned 2560-dimensional retrieval projection.
Attention uses explicit matmul and FP32 softmax, as in the HF eager reference.
No LM head, token generation, KV cache, TorchAir, quantization, or NZ conversion
is involved. Video input, generation/cache arguments, and alternative model
architectures are intentionally unsupported; they must not be silently accepted.

`LocalColQwen3.from_pretrained` loads all safetensors weights strictly, including
projection bias and DeepStack-specific normalization shapes. Its forward accepts
the checkpoint processor's `input_ids`, `attention_mask`, padded `pixel_values`
and `image_grid_thw`, returning `[batch, sequence, 2560]` normalized embeddings.

### Parity validation

The validation runner imports HF only for the independent reference and unchanged
processor. It compares saved anchors, fresh left/right-padded text and image
batches, exact position IDs, 22 selected intermediate module outputs (when used),
final embeddings, repeat stability and MaxSim scores. It saves processor tensors
and both outputs for each case. Instrumented wall times include CPU trace copies
and are **not** performance measurements.

```sh
source npu-setup
cd /workspace/repos/paddle_ocr_vl_npu
/workspace/venvs/colqwen3_hf_py312/bin/python -u \
  21_colqwen3_4b_inference/run_local_parity.py \
  --model /workspace/models/Ops-Colqwen3-4B \
  --reference-dirs \
    tmp/21_colqwen3_4b_inference/hf4571_smoke_b25734d4/output \
    tmp/21_colqwen3_4b_inference/vidore_smoke_247c121b/output \
  --output-dir tmp/21_colqwen3_4b_inference/local_parity_new/output
```

Those anchor paths are specific to our 910B workspace. Elsewhere, regenerate
anchors with `run_hf_baseline.py`; do not substitute synthetic image tensors.
CPU unit/structural tests (`python3 -m unittest discover -s
21_colqwen3_4b_inference -p 'test_*.py'`) are separate from NPU inference parity.

**Local eager validation passed on 910B2, physical device 7, 2026-10-01,
source `038ce189`.** All 10 cases had bit-exact embeddings, all 168 captured
intermediate comparisons were exact, both MaxSim matrices were exact, and all
position IDs matched. Both saved HF anchor groups were reproduced exactly.
Maximum active embedding-row norm error was 0.000452, with zero padding rows.
See the [successful report](references/local_parity_910b/result.json),
[command](references/local_parity_910b/command.txt), and
[log](references/local_parity_910b/run.log).

The [initial failed report](references/local_parity_910b/initial_failed_result.json)
is retained. It isolated a numerical-fidelity pitfall: computing rotary inverse
frequencies on NPU differed from HF's CPU FP32 initialization by up to
3.73e-9 (vision) / 7.45e-9 (text). Those small differences propagated through
the network. Matching CPU initialization restored exact parity without relaxing
any tolerance or changing the reference. Keep this initialization detail when
adding later optimizations; do not infer bit parity merely from equivalent math.

This is not a 310P result, a full ViDoRe retrieval score, or an optimized runtime.
The next gate is broader retrieval evaluation; speed work must remain a separate
lane from this eager correctness anchor.
