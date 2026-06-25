# Experiment 07: Vision Prefill Optimization

This experiment isolates PaddleOCR-VL recognition prefill for real OmniDocBench
crops. It intentionally excludes decode, hot-swapping, EOS handling, and page
layout detection.

## Reference Bundle

Create the authoritative NPU baseline:

```sh
cd 07_vision_prefill_optimization
ASCEND_RT_VISIBLE_DEVICES=1 bash run_npu_make_reference_baseline.sh
```

The reference contract is eager/non-compiled fp16 PromptFA. It stores one `.pt`
file per selected crop under `baselines/promptfa_fp16_eager_64/tensors/` plus a
`reference_manifest.json`.

Each tensor file stores:

- `visual_features`: native-resolution visual encoder output
- `image_embeds`: adaptive MLP projector output
- `prefill_logits`: next-token logits after text prefill

## Candidate Compare

Compare a candidate path to the stored baseline:

```sh
python vision_prefill_bench.py compare \
  --baseline baselines/promptfa_fp16_eager_64 \
  --candidate-name candidate_name \
  --device npu:0 \
  --dtype fp16 \
  --vision-attention prompt_flash_attention \
  --output outputs/candidate_name.json
```

The compare command re-extracts the exact manifest crops, verifies tensor file
SHA256s, computes the same three intermediate targets, and reports per-item plus
aggregate diffs.

## CUDA Smoke

CUDA smoke uses manual attention only and is not authoritative NPU evidence:

```sh
bash run_cuda_manual_smoke.sh
```
