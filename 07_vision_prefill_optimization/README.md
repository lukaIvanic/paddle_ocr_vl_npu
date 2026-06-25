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

Pass a local model directory with `--model` or set `MODEL`. Hugging Face download
is disabled in this experiment so missing model paths fail fast.

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
  --vision-prompt-fa-mask-sparse-mode 1 \
  --output outputs/candidate_name.json
```

The compare command re-extracts the exact manifest crops, verifies tensor file
SHA256s, computes the same three intermediate targets, and reports per-item plus
aggregate diffs.

Candidate timing defaults to `--timing-mode standard`. This records the visual
tower metric and the full-prefill wrapper metric separately in the same run.

The headline visual-prefill speed metric is `visual_tower_e2e_s`: the crop's
pixel tensor is already on the target device, static inputs such as `cu_seqlens`
are prepared, the script synchronizes, calls the visual tower, and synchronizes
again. When static padding is enabled, this timed visual tower returns physical
padded rows; the script slices back to real rows only after the sync/timer stop,
before projector/logit correctness checks. The JSON reports both
`visual_tower_effective_tokens_per_s` and `visual_tower_physical_tokens_per_s`.

The secondary wrapper metric is `full_prefill_e2e_s`, which measures visual
tower plus adaptive MLP projector plus text prefill. Use `--timing-mode
phase_sync` only for diagnostic phase breakdowns because it synchronizes around
every named phase.

Candidate compare always uses the compile-shaped static visual boundary. Use
`--vision-compile-backend none` for the noncompiled path, and use the TorchAir
backend to test compilation:

```sh
python vision_prefill_bench.py compare \
  --baseline baselines/promptfa_fp16_eager_64 \
  --candidate-name static_visual_torchair \
  --device npu:0 \
  --dtype fp16 \
  --vision-attention prompt_flash_attention \
  --vision-prompt-fa-mask-sparse-mode 1 \
  --vision-compile-backend torchair \
  --output outputs/static_visual_torchair.json
```

The static visual path hoists `cu_seqlens`, absolute position embeddings, and
vision RoPE out of the forward graph and compiles with `fullgraph=True,
dynamic=False`. This path currently uses plain `torch.compile`; it does not yet
load or write a TorchAir GE compile cache.

The static visual candidate has one padding-capable encoder path. The automatic
padding policy is reported as `static_visual_pad_policy`. Normal candidate runs
always add masked dummy rows. On Atlas inference cards, masked PromptFA needs
the physical sequence length to be aligned: tiny shapes use 16 alignment, and
normal crop shapes with `S > 128` use 128 alignment. The visual tower timing
stops after the synchronized physical padded output; real rows are sliced only
afterward for projector/logit correctness checks.

For debugging only, `--debug-static-visual-no-padding` runs the no-mask control,
and `--debug-static-visual-min-pad-tokens` /
`--debug-static-visual-pad-to-multiple` can adjust the padded physical shape.
Use these to separate padding/mask numerics from TorchAir compile numerics; do
not use them for normal throughput claims.

For padded PromptFA on the 310P3/CANN 8.2.RC1 work box, sparse mode `1` is the
normal setting because the synthetic probe showed that mode `1` honors the full
custom padding mask while mode `0` ignored it. Sparse modes are not
padding-placement settings: modes 2/3/4 are causal/band patterns with stricter
mask constraints, and `actual_seq_lengths` is not a viable 310P/Atlas-inference
path for this experiment. Keep mode `0` as a diagnostic only.

Check PromptFA mask semantics directly before interpreting OCR-level padded
drift:

```sh
python vision_prefill_bench.py probe-promptfa-mask \
  --device npu:0 \
  --dtype fp16 \
  --npu-jit-compile off
```

The key summary field is `recommended_full_mask_semantics_passed`. It should be
`true`, with `recommended_mask_sparse_mode` equal to `1` on the current 310P3
work box. Mode `1` with a non-null block mask should match the masked manual
reference and differ from the unmasked reference.

If padded eager matches but padded TorchAir diverges, isolate the compiled
PromptFA operator before blaming the visual tower:

```sh
python vision_prefill_bench.py probe-promptfa-compile \
  --device npu:0 \
  --dtype fp16 \
  --npu-jit-compile off \
  --vision-compile-backend torchair \
  --seq-lens 640,768 \
  --cases no_mask,all_false_mask,block_mask \
  --output outputs/promptfa_compile_probe.json
```

This probe does not load PaddleOCR-VL. It compiles a tiny PromptFA-only module
with `fullgraph=True, dynamic=False`, reports that it does not use
`cache_compile` / a GE cache directory, and compares eager vs compiled for:
unmasked PromptFA, sparse-mode-1 all-false mask, and sparse-mode-1 block mask.
The headline field is `compiled_second_matches_eager_all`.

The NPU equivalence gate has already passed: backend `none` matched the stored
eager PromptFA truth bundle with 0.0 diffs before the padding path was unified.
After changes to the static visual path, rerun backend `none` first and only
interpret TorchAir results if the noncompiled static path still matches the
stored baseline.

For TorchAir correctness investigations, add
`--validate-compiled-against-static-eager`. This runs the same static visual
candidate wrapper eagerly once during compiled setup and records
`compiled_vs_static_eager_validation.physical` plus
`compiled_vs_static_eager_validation.real_rows` inside each item's
`vision_compile` object. The `real_rows` diff is the key check for whether
compilation changed the rows that are actually compared to the stored baseline.

## MSIT GE-vs-FX Dump Compare

After `--torchair-run-eagerly` proves the FX graph semantics are clean, use the
native MSIT TorchAir dump path to localize GE/CANN lowering drift:

```sh
ASCEND_RT_VISIBLE_DEVICES=1 bash run_npu_msit_ge_fx_compare.sh
```

The runner executes one-crop static visual compare twice:

- GE target dump: `--torchair-msit-dump-kind ge`
- FX golden dump: `--torchair-msit-dump-kind fx`

`msit_llm` comes from the optional MSIT LLM component. The benchmark prefers
`from msit_llm.dump import torchair_dump` when it is installed, but it also has a
local compatibility fallback that applies the same TorchAir dump config fields
directly. The final official comparison still needs the `msit` CLI. If the CLI
is missing, install MSIT LLM in the active NPU Python environment:

```sh
python -m pip install msit
msit install llm
msit check llm
```

If dumps were already generated and only the official compare failed, do not
rerun the dump step. Retry compare on the existing output directory:

```sh
OUT_ROOT=outputs/msit_ge_fx_YYYYMMDDTHHMMSSZ bash run_npu_msit_compare_existing.sh
```

The compare scripts set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` by
default, which avoids the common generated-`pb2` versus protobuf-runtime crash.
If that still fails with a protobuf descriptor error, pin protobuf in the active
NPU environment:

```sh
python -m pip install 'protobuf==3.20.2'
```

Then it runs:

```sh
msit llm compare --my-path GE_DUMP --golden-path FX_DUMP --output OUT/msit_compare
```

The script defaults to GE `dump_mode=output` to avoid huge dumps. If the MSIT
compare output needs inputs too, rerun with `MSIT_DUMP_MODE=all`. Keep GE and
FX dump directories separate; MSIT warns that reusing a dump path can mix data
and invalidate the comparison.

## LayerNorm Compile Probe

If MSIT compare points at an early `LayerNormV3` row, isolate LayerNorm before
changing the vision tower:

```sh
ASCEND_RT_VISIBLE_DEVICES=1 bash run_npu_layernorm_compile_probe.sh
```

The runner creates `outputs/layernorm_compile_probe_*/torchair_default.json` and
`torchair_run_eagerly.json`. It tests synthetic `[S, 1152]` inputs for
`S=580,640,768` plus the real first baseline crop tensor immediately before
vision layer-0 `layer_norm1`. It compares:

- `nn.LayerNorm`
- `torch.nn.functional.layer_norm`
- manual mean/variance normalization with explicit fp32 reduction
- manual mean/variance normalization with input-dtype reduction, exposed as
  `manual_fp16_reduce` for fp16 overflow diagnostics
- `torch_npu.npu_layer_norm_eval` when available

Read the summary this way:

- `torchair_run_eagerly` matching eager means the FX graph semantics are correct.
- `torchair_default` mismatching while `run_eagerly` matches means GE/CANN graph
  execution lowered the LayerNorm path incorrectly.
- Synthetic pass plus real-crop fail means LayerNorm itself is probably not
  enough; the tensor produced by preceding compiled visual ops is the likely
  trigger.
- `manual` passing while `nn`/`functional` fail is consistent with a
  LayerNormV3 accumulation/lowering problem. `manual_fp16_reduce` is expected to
  be less stable on large synthetic scales and exists to make the fp16-reduction
  hypothesis explicit.
- Real-crop fail with `compiled_nonfinite_count > 0` reproduces the MSIT
  `my_data includes NAN or inf` symptom in a smaller, LayerNorm-only boundary.

For an explicit fp16 overflow stress test, rerun with a smaller matrix set and
larger synthetic values:

```sh
SEQ_LENS=640 SYNTHETIC_INPUT_SCALES=1,64,128 \
  ASCEND_RT_VISIBLE_DEVICES=1 bash run_npu_layernorm_compile_probe.sh
```

## CUDA Smoke

CUDA smoke uses manual attention only and is not authoritative NPU evidence:

```sh
bash run_cuda_manual_smoke.sh
```
