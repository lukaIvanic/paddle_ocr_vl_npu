# Experiment 07: Vision Prefill Optimization

Experiment 07 isolates PaddleOCR-VL's vision transformer so attention and
compilation changes can be measured without mixing in layout inference,
preprocessing, projection, text prefill, or decode.

## Current benchmark

On Blue Zone:

```sh
ssh blue_zone_npu_container
cd /workspace/repos/paddle_ocr_vl_npu
source npu-setup
bash 07_vision_prefill_optimization/run_npu_small_visual_encoder_matrix.sh
```

The runner reconstructs the exact 160 layout crops from the existing five-page
Experiment08 result, preprocesses them at the requested `min_pixels`, and selects
the highest-fill real crops in each fixed vision-token bucket.

It compares this matrix at `B=1`:

```text
manual attention, eager
manual attention, TorchAir compiled
PromptFA, eager
PromptFA, TorchAir compiled
```

The initial bucket set is:

```text
min_pixels=56448:  (0, 384]
min_pixels=112896: (0, 640], (640, 768]
```

Override it with, for example:

```sh
BUCKET_CASES="56448:0:384 56448:384:512 112896:0:640 112896:640:768 112896:768:896" \
bash 07_vision_prefill_optimization/run_npu_small_visual_encoder_matrix.sh
```

The benchmark records three repeated timing blocks after warmup. The timed
callable is only:

```text
27 vision encoder layers + post LayerNorm
```

Patch embedding, absolute positions, RoPE, mask construction, projector, text
prefill, and decode are excluded. Results include physical padded tokens/s,
effective real tokens/s, useful-token fraction, first-call/compile time,
per-block dispersion, and eager-versus-compiled numerical checks. Correctness
continues outside the timed region through the real adaptive projector and text
prefill, including next-token argmax agreement.

Outputs are written under:

```text
tmp/07_vision_prefill_optimization/small_visual_encoder_<UTC>/
  case_*.json
  case_*.log
  summary.json
  summary.tsv
  compiled_vs_eager.tsv
```

TorchAir artifacts are cached under `.runtime_cache/`, not committed.

## Stock comparison and workarounds

The primary matrix starts with the stock-shaped math:

```text
LayerNorm: module
LayerNorm-fed Linear: normal
PromptFA head dimension: 72
```

Older 310P experiments needed provisional alternatives such as manual fp32
LayerNorm, grouped matmul, and padding PromptFA's call dimension from 72 to 80.
Those are not enabled silently. If native compilation fails correctness on 910B,
record the failed stock case and run an explicitly labeled workaround matrix,
for example:

```sh
PROMPTFA_PAD_HEAD_DIM_TO=80 \
OUT_ROOT="$PWD/tmp/07_vision_prefill_optimization/promptfa_d80" \
bash 07_vision_prefill_optimization/run_npu_small_visual_encoder_matrix.sh
```

## Important files

- `benchmark_small_visual_encoder.py`: one attention/backend/shape case.
- `run_npu_small_visual_encoder_matrix.sh`: current 910B matrix runner.
- `summarize_small_visual_encoder_matrix.py`: comparison tables.
- `validate_static_visual_batched_encoder.py`: reusable static encoder boundary
  and TorchAir compile wrapper.
- `vision_prefill_bench.py`: older full vision-prefill reference and diagnostic
  framework.
- `repro_attention_only_compile.py` and
  `repro_inline_single_layer_compile.py`: narrow correctness probes.
- `profile_static_visual_batched_encoder.py`: NPU profiling after a valid speed
  result exists.

The remaining `run_npu_*` scripts are historical diagnostics from the earlier
310P investigation. Several have old path defaults; pass current model, dataset,
Python, and device values explicitly when using them.

## Local checks

```sh
cd 07_vision_prefill_optimization
PYTHONPYCACHEPREFIX=/tmp/exp07_pycache python3 test_small_visual_encoder_benchmark.py
PYTHONPYCACHEPREFIX=/tmp/exp07_pycache python3 -m py_compile \
  benchmark_small_visual_encoder.py \
  summarize_small_visual_encoder_matrix.py \
  validate_static_visual_batched_encoder.py
bash -n run_npu_small_visual_encoder_matrix.sh
```
