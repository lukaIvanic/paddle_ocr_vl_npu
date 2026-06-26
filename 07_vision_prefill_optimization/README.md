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

## LayerNorm -> QKV Compile Probe

The current first-bad-edge investigation points at the first visual encoder
block's `LayerNorm -> QKV Linear` producer-consumer edge, not PromptFA itself.
The prefix probe showed that patch embedding, position embedding, and
`layer_norm1` match eager, while QKV projection is the first compiled GE/CANN
divergence. The QKV probe then showed that materialized eager LN output fed to
QKV is clean, but `patch_pos -> LayerNorm -> QKV` inside one compiled graph can
produce fp16-saturated garbage.

Use the dedicated runner for the smallest real-crop check:

```sh
ASCEND_RT_VISIBLE_DEVICES=1 bash run_npu_qkv_linear_compile_probe.sh
```

By default it tests one real crop, one Q projection, and the most important
handoff barriers:

- `none`: reproduce the bad GE handoff.
- `format_cast_nd`: explicit `torch_npu.npu_format_cast(tensor, 2)` between LN
  and QKV, based on older Ascend transformer patches that use format `2` as an
  ND/base-format materialization boundary.
- `format_cast_nz_then_nd`: diagnostic format transition control.
- `transpose_roundtrip`: known-good but potentially heavier materialization
  barrier.

Optional diagnostic axes:

```sh
LN_IMPLS=module,functional,manual_fp32 \
BRIDGES=none,format_cast_nd,transpose_roundtrip \
ASCEND_RT_VISIBLE_DEVICES=1 bash run_npu_qkv_linear_compile_probe.sh

NPU_MM_BMM_FORMAT_ND=enable \
BRIDGES=none,format_cast_nd \
RUN_TORCHAIR_EAGERLY=0 \
ASCEND_RT_VISIBLE_DEVICES=1 bash run_npu_qkv_linear_compile_probe.sh

BRIDGES=none \
IMPLS=functional_q,addmm_q,mm_q,bmm_q,matmul_3d_q,einsum_q,conv1d_q,npu_linear_q \
RUN_TORCHAIR_EAGERLY=0 \
ASCEND_RT_VISIBLE_DEVICES=1 bash run_npu_qkv_linear_compile_probe.sh

BRIDGES=none \
IMPLS=npu_bmm_v2_q,npu_grouped_matmul_q \
RUN_TORCHAIR_EAGERLY=0 \
ASCEND_RT_VISIBLE_DEVICES=1 bash run_npu_qkv_linear_compile_probe.sh
```

Interpretation:

- If `format_cast_nd` matches eager, prefer it over `transpose_roundtrip` for
  the next full-vision candidate because it directly tests the suspected
  internal-format contract.
- If a manual LN implementation fixes `bridge=none`, the problem depends on the
  fused `LayerNormV3` producer. If manual LN still fails, the problem is broader
  GE layout propagation into MatMul.
- If `NPU_MM_BMM_FORMAT_ND=enable` fixes `bridge=none`, the consumer-side MatMul
  format policy is enough and may be cleaner than inserting per-layer bridges.
- If any `BRIDGES=none` alternate consumer such as `bmm_q`, `conv1d_q`, or
  `npu_linear_q` matches eager, then the failure is not simply "LN output into
  any cube op"; it is specific to the failing QKV consumer lowering. If all
  alternate consumers fail, the post-LN format boundary is still the cleanest
  direction.
- `npu_grouped_matmul_q` is a diagnostic only. Grouped MatMul is built for
  grouped/expert-style matmuls, not a normal single Linear, so an eager or
  compile error is useful evidence rather than a benchmark failure. The probe
  uses a one-expert 3D weight tensor `[1, K, O]`, matching the op's grouped
  weight-rank contract instead of passing a plain 2D Linear weight. With that
  one-expert form, the bias is passed as `[1, O]` rather than a plain 1D
  Linear bias.

Once the grouped Q probe passes, test the real static visual path, not another
synthetic probe:

```sh
ASCEND_RT_VISIBLE_DEVICES=1 bash run_npu_static_visual_grouped_compare.sh
```

This runner compares real baseline crops in three modes:

- `eager_grouped_qkv_mlp_fc1`: backend `none`, grouped matmul for both direct
  LayerNorm consumers. This must match the stored baseline before compiled
  results are meaningful.
- `torchair_grouped_qkv`: compiled grouped QKV only. If this still shows large
  visual/logit drift, the remaining likely direct producer-consumer edge is
  `layer_norm2 -> mlp.fc1`.
- `torchair_grouped_qkv_mlp_fc1`: compiled grouped QKV plus grouped `mlp.fc1`.
  This is the first full static-visual candidate that replaces all Linear ops
  directly fed by vision LayerNorm.

The default `MAX_ITEMS=4` keeps compile cost low. Increase to `MAX_ITEMS=8` only
after the 4-crop signal is clean.

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

## Visual Prefix Compile Probe

If isolated LayerNorm matches eager but the full static visual graph still
diverges, bisect the real-crop compiled prefix:

```sh
ASCEND_RT_VISIBLE_DEVICES=1 bash run_npu_visual_prefix_compile_probe.sh
```

By default the runner writes two JSON files: `torchair_default.json` for the
normal GE/CANN execution path, and `torchair_run_eagerly.json` for the traced FX
graph executed eagerly. If run-eagerly matches but default diverges, the Python
graph semantics are clean and the bug is in GE/CANN lowering or execution.

This uses the same first baseline crop and compiles progressively larger
prefixes of the current static visual path:

- `patch_conv`: patch embedding Conv2D over the pre-extracted patch tensors
- `patch_flat`: Conv2D output flattened to token rows
- `patch_pad`: optional static dummy rows appended
- `patch_pos`: absolute position embeddings added
- `ln1`: first vision encoder layer `layer_norm1`

The default stops at `ln1` because that is where the MSIT CSV symptom was
reported. If these all pass, go deeper with:

```sh
STAGES=qkv,qk_rope_v,attn_out,layer0_out \
  ASCEND_RT_VISIBLE_DEVICES=1 bash run_npu_visual_prefix_compile_probe.sh
```

Read the summary this way:

- First mismatch at `patch_conv` points at Conv2D/TorchAir/format lowering.
- First mismatch at `patch_pos` points at padding or absolute-position add
  layout/format behavior.
- `ln1` mismatch with prior stages clean means LayerNorm is only problematic
  inside the larger compiled prefix, not in isolation.
- `compiled_nonfinite_count > 0` is the small-prefix version of the MSIT
  `my_data includes NAN or inf` symptom.

## QKV Linear Compile Probe

If the visual prefix probe first fails at `qkv`, isolate that projection:

```sh
ASCEND_RT_VISIBLE_DEVICES=1 bash run_npu_qkv_linear_compile_probe.sh
```

The runner uses the same first baseline crop and writes `torchair_default.json`
plus `torchair_run_eagerly.json`. It tests two input sources:

- `ln1`: compile only the QKV projection, fed by the materialized eager
  `layer_norm1` output. If this fails, the QKV linear/addmm/MatMul lowering is
  enough to reproduce the bug.
- `patch_pos`: compile `layer_norm1 -> QKV`, fed by the materialized
  patch-plus-position tensor. If this fails while `ln1` passes, the culprit is
  the internal GE handoff from LayerNorm output to linear/MatMul, not standalone
  LayerNorm or standalone QKV.

The implementation variants are intentionally mechanical controls:
`nn.Linear`, `F.linear`, explicit `matmul + bias`, single concatenated QKV
weight, Q-only, and no-bias Q-only. The summary prints every
`source`/`impl` case with max diff and compiled finite min/max so fp16-range
garbage is visible without opening the full JSON.

After `source=ln1` passes and `source=patch_pos` fails, the runner defaults to
a small handoff-barrier matrix: `source=patch_pos`, `functional_q`, and bridges
inserted after LayerNorm but before QKV (`contiguous`, `clone`, `add_zero`,
reshape-contiguous, and transpose round-trip). This asks whether an explicit
materialization barrier can break the bad GE layout handoff. If one bridge
works, expand the projection implementations for that bridge:

```sh
BRIDGES=the_working_bridge \
  IMPLS=module_q,functional_q,matmul_q,functional_q_no_bias,matmul_q_no_bias \
  ASCEND_RT_VISIBLE_DEVICES=1 bash run_npu_qkv_linear_compile_probe.sh
```

To rerun the original broad matrix:

```sh
SOURCES=ln1,patch_pos BRIDGES=none \
  IMPLS=module_three,functional_three,matmul_three,functional_single,matmul_single,module_q,functional_q,matmul_q,functional_q_no_bias,matmul_q_no_bias \
  ASCEND_RT_VISIBLE_DEVICES=1 bash run_npu_qkv_linear_compile_probe.sh
```

## Visual Layer Edge Probe

If grouped QKV or grouped QKV+MLP-fc1 still diverges in the full static visual
compare, use the layer-edge probe before changing the candidate path again:

```sh
ASCEND_RT_VISIBLE_DEVICES=1 bash run_npu_visual_layer_edge_probe.sh
```

Default settings compile one item with
`LN_LINEAR_MODE=grouped_qkv_mlp_fc1` and check the first visual layer at:

- `qkv`
- `qk_rope_v`
- `attn_kernel_out`
- `attn_out_proj`
- `attn_residual`
- `ln2`
- `mlp_fc1`
- `mlp_act`
- `mlp_fc2`
- `layer0_out`

The critical output is `first_mismatch` plus the printed `STAGE_TABLE`. This is
not a throughput benchmark: it compiles one graph per stage so that the first
bad producer-consumer edge is visible. Start with `MAX_ITEMS=1`. If every stage
matches for the first item, rerun with `MAX_ITEMS=2`; if the first mismatch is
already clear, stop and report it.

Interpretation:

- If `qkv` passes but `attn_kernel_out` fails, the problem is inside the
  attention kernel or the Q/K/V layout handed into it.
- If `attn_kernel_out` passes but `attn_out_proj` fails, the next normal Linear
  consumer (`out_proj`) is misreading the attention output.
- If `attn_residual` and `ln2` pass but `mlp_fc1` fails, grouped `fc1` did not
  solve the second LayerNorm-to-Linear edge in the full graph.
- If `mlp_fc1` and `mlp_act` pass but `mlp_fc2` fails, the activation output is
  being handed to the normal `fc2` Linear in a bad format.
- If all single-layer stages pass but full static visual still fails, the bug is
  likely a later-layer or cross-layer format propagation issue; add a deeper
  edge probe before broad rewrites.

## Inline Single-Layer Repro

For the most compact reproduction, use the self-contained inline script:

```sh
ASCEND_RT_VISIBLE_DEVICES=1 bash run_npu_inline_single_layer_repro.sh
```

This script does not import `vision_prefill_bench.py`. It rebuilds one real
baseline crop, loads layer 0 weights, defines one inline transformer block in a
single `nn.Module`, runs that module eagerly and with `torch.compile(...,
fullgraph=True, dynamic=False)`, and compares all returned intermediate tensors
from the same compiled graph.

Default settings match the current suspected path:

- real baseline crop `ITEM_INDEX=0`
- `ATTENTION=prompt_flash_attention`
- `LN_IMPL=module`
- `LN_LINEAR_MODE=grouped_qkv_mlp_fc1`
- `PRE_PROMPTFA_BRIDGE=none`

The printed `STAGE_TABLE` includes:

- `q_bnsd`, `k_bnsd`, `v_bnsd`: exact tensors handed to PromptFA
- `attn_kernel_bnsd`: raw PromptFA result before transpose/view
- `attn_kernel_out`: PromptFA result after returning to `[S, hidden]`
- downstream `out_proj`, residual, `ln2`, `fc1`, activation, `fc2`, and
  `layer0_out`
- overall diffs plus `real_*` and `pad_*` split diffs. BNSD tensors are split
  on sequence axis 2; `[S, hidden]` tensors are split on axis 0.

Interpretation:

- If `q_bnsd`, `k_bnsd`, and `v_bnsd` match but `attn_kernel_bnsd` is the first
  bad stage, the script is a direct compiled-PromptFA repro.
- If `attn_kernel_bnsd` matches but `attn_kernel_out` fails, the transpose/view
  after PromptFA is the bad handoff.
- If PromptFA stages match and `attn_out_proj` fails, the normal Linear
  consumer after attention is the bad handoff.

Useful controls, in order:

```sh
# FX/Python semantics control. Should match eager if the GE/CANN graph is the bug.
TORCHAIR_RUN_EAGERLY=1 ASCEND_RT_VISIBLE_DEVICES=1 bash run_npu_inline_single_layer_repro.sh

# Preserve LayerNorm semantics but avoid fused LayerNormV3.
LN_IMPL=manual_fp32 ASCEND_RT_VISIBLE_DEVICES=1 bash run_npu_inline_single_layer_repro.sh

# Padding/mask influence control. This removes the static pad rows and therefore
# removes the PromptFA pad mask for this one-crop diagnostic.
LN_IMPL=manual_fp32 NO_PADDING=1 ASCEND_RT_VISIBLE_DEVICES=1 bash run_npu_inline_single_layer_repro.sh

# Check whether functional LayerNorm lowers differently from module LayerNorm.
LN_IMPL=functional ASCEND_RT_VISIBLE_DEVICES=1 bash run_npu_inline_single_layer_repro.sh

# Manual attention control. If this passes while PromptFA fails, compiled PromptFA is isolated.
ATTENTION=manual ASCEND_RT_VISIBLE_DEVICES=1 bash run_npu_inline_single_layer_repro.sh

# Real activation barrier before PromptFA. If this fixes PromptFA, the issue is likely input layout.
PRE_PROMPTFA_BRIDGE=transpose_roundtrip ASCEND_RT_VISIBLE_DEVICES=1 bash run_npu_inline_single_layer_repro.sh

# Current D-alignment workaround candidate. This keeps RoPE inside the compiled
# graph, pads only the PromptFA Q/K/V call dimension from D=72 to D=80, keeps
# scale_value=1/sqrt(72), and slices the PromptFA output back to D=72 before
# out_proj. Use manual LayerNorm to avoid the fused LayerNormV3 GE bug.
LN_IMPL=manual_fp32 LN_LINEAR_MODE=grouped_qkv_mlp_fc1 PROMPTFA_PAD_HEAD_DIM_TO=80 ASCEND_RT_VISIBLE_DEVICES=1 bash run_npu_inline_single_layer_repro.sh
```

The Qwen3-Embedding reference project uses RMSNorm, not LayerNorm. Treat that as
a clue that fused `LayerNormV3` may be the bad GE producer; do not treat RMSNorm
as a valid PaddleOCR-VL replacement. `LN_IMPL=manual_fp32` is the relevant test
because it keeps LayerNorm mean/variance/bias semantics while avoiding the fused
LayerNorm operator.

The JSON and printed summary include `promptfa_contract`, which records the
physical sequence length, mod-16/mod-128 alignment, Q/K/V call shape, sparse
mode, mask shape/counts, and the current pad-mask policy. Use it before drawing
conclusions about whether PromptFA was called on a 128-aligned padded sequence.
For `PROMPTFA_PAD_HEAD_DIM_TO=80`, check that `promptfa_call_head_dim=80`,
`promptfa_head_dim_pad_extra=8`,
`promptfa_call_head_dim_fp16_32b_aligned=true`, and
`promptfa_output_sliced_back_to_real_head_dim=true`.
The summary also includes `first_bad_stage_real_rows` and
`first_bad_stage_pad_rows`; prefer the real-row summary when judging whether
compiled visual features would survive slicing padded rows away before
downstream consumers.

If the inline D=80 run has no nonfinites and the first bad stage is acceptably
small, move to the full static visual candidate:

```sh
STATIC_VISUAL_LN_IMPL=manual_fp32 PROMPTFA_PAD_HEAD_DIM_TO=80 MAX_ITEMS=4 ASCEND_RT_VISIBLE_DEVICES=1 bash run_npu_static_visual_grouped_compare.sh
```

Read `torchair_grouped_qkv_mlp_fc1.json` first. The important fields are:

- `summary.argmax_match_count` versus `compared_count`
- `summary.visual_features`, `summary.image_embeds`, and `summary.prefill_logits`
- per-item `vision_compile.compiled_vs_static_eager_validation.real_rows`
- per-item `vision_compile.first_real_output_nonfinite_count`
- `vision_compile.static_visual_promptfa_call_head_dim` and
  `vision_compile.static_visual_promptfa_call_head_dim_fp16_32b_aligned`

Do not scale to 32/64 crops unless the 4-crop run has no real-row nonfinites and
the prefill logits/argmax checks are acceptable. If it fails, report the first
bad item and the compiled-vs-static-eager real-row diff instead of creating a new
script.

## Attention-Only Repro

Use `repro_attention_only_compile.py` when the full inline layer has already
localized the failure to attention. It materializes real-crop Q/K/V tensors
eagerly with manual LayerNorm and vision RoPE, then compiles only this graph:

```python
attention(q_bnsd, k_bnsd, v_bnsd) -> attn_kernel_bnsd
```

The compiled graph does not include patch embedding, position embeddings,
LayerNorm, QKV projection, residuals, MLP, or output projection. This is the
cleanest place to compare manual attention against `npu_prompt_flash_attention`
with and without masks.

Runner:

```sh
ASCEND_RT_VISIBLE_DEVICES=1 bash run_npu_attention_only_repro.sh
```

Useful controls:

```sh
# Known-good manual attention control over the same materialized Q/K/V.
ATTENTION=manual ASCEND_RT_VISIBLE_DEVICES=1 bash run_npu_attention_only_repro.sh

# PromptFA eager, no TorchAir GE lowering.
ATTENTION=prompt_flash_attention COMPILE_BACKEND=none ASCEND_RT_VISIBLE_DEVICES=1 bash run_npu_attention_only_repro.sh

# PromptFA compiled, no padding/mask.
ATTENTION=prompt_flash_attention NO_PADDING=1 MASK_KIND=none ASCEND_RT_VISIBLE_DEVICES=1 bash run_npu_attention_only_repro.sh

# Same no-padding/no-mask case with BSND call layout. Output is normalized back
# to BNSD before comparison.
ATTENTION=prompt_flash_attention PROMPTFA_LAYOUT=bsnd NO_PADDING=1 MASK_KIND=none ASCEND_RT_VISIBLE_DEVICES=1 bash run_npu_attention_only_repro.sh

# Boundary ladder, no padding/mask. This progressively moves more producer
# operations into the compiled graph before PromptFA:
# - bnsd: PromptFA receives already-materialized BNSD Q/K/V.
# - snhd_rope_done: graph includes SNHD -> PromptFA layout conversion.
# - snhd_pre_rope: graph includes RoPE plus layout conversion.
# - qkv_flat_pre_rope: graph includes QKV chunk/view plus RoPE plus layout conversion.
ATTENTION=prompt_flash_attention INPUT_BOUNDARY=bnsd NO_PADDING=1 MASK_KIND=none ASCEND_RT_VISIBLE_DEVICES=1 bash run_npu_attention_only_repro.sh
ATTENTION=prompt_flash_attention INPUT_BOUNDARY=snhd_rope_done NO_PADDING=1 MASK_KIND=none ASCEND_RT_VISIBLE_DEVICES=1 bash run_npu_attention_only_repro.sh
ATTENTION=prompt_flash_attention INPUT_BOUNDARY=snhd_pre_rope NO_PADDING=1 MASK_KIND=none ASCEND_RT_VISIBLE_DEVICES=1 bash run_npu_attention_only_repro.sh
ATTENTION=prompt_flash_attention INPUT_BOUNDARY=qkv_flat_pre_rope NO_PADDING=1 MASK_KIND=none ASCEND_RT_VISIBLE_DEVICES=1 bash run_npu_attention_only_repro.sh

# PromptFA compiled, padded but no mask.
ATTENTION=prompt_flash_attention MASK_KIND=none ASCEND_RT_VISIBLE_DEVICES=1 bash run_npu_attention_only_repro.sh

# PromptFA compiled with the current real<->pad block mask.
ATTENTION=prompt_flash_attention MASK_KIND=current MASK_RANK=4 ASCEND_RT_VISIBLE_DEVICES=1 bash run_npu_attention_only_repro.sh

# Padded/masked boundary checks once the no-padding ladder is understood.
ATTENTION=prompt_flash_attention INPUT_BOUNDARY=snhd_rope_done MASK_KIND=current MASK_RANK=4 ASCEND_RT_VISIBLE_DEVICES=1 bash run_npu_attention_only_repro.sh
ATTENTION=prompt_flash_attention INPUT_BOUNDARY=snhd_pre_rope MASK_KIND=current MASK_RANK=4 ASCEND_RT_VISIBLE_DEVICES=1 bash run_npu_attention_only_repro.sh
ATTENTION=prompt_flash_attention INPUT_BOUNDARY=qkv_flat_pre_rope MASK_KIND=current MASK_RANK=4 ASCEND_RT_VISIBLE_DEVICES=1 bash run_npu_attention_only_repro.sh

# If a boundary produces NaN, rerun that exact case with the 310P-supported
# TorchAir mode. Compare the nonfinite mask, not ordinary allclose.
ATTENTION=prompt_flash_attention INPUT_BOUNDARY=snhd_pre_rope NO_PADDING=1 MASK_KIND=none TORCHAIR_MODE=max-autotune ASCEND_RT_VISIBLE_DEVICES=1 bash run_npu_attention_only_repro.sh

# PromptFA head-dim pad/barrier probe. PaddleOCR-VL vision uses D=72, which is
# not fp16 32-byte aligned. Padding to D=80/96 tests aligned call widths, while
# D=73/88 tests whether merely inserting Pad/Slice breaks a bad GE handoff.
ATTENTION=prompt_flash_attention INPUT_BOUNDARY=bnsd NO_PADDING=1 MASK_KIND=none PROMPTFA_PAD_HEAD_DIM_TO=80 ASCEND_RT_VISIBLE_DEVICES=1 bash run_npu_attention_only_repro.sh
ATTENTION=prompt_flash_attention INPUT_BOUNDARY=snhd_pre_rope NO_PADDING=1 MASK_KIND=none PROMPTFA_PAD_HEAD_DIM_TO=73 ASCEND_RT_VISIBLE_DEVICES=1 bash run_npu_attention_only_repro.sh
ATTENTION=prompt_flash_attention INPUT_BOUNDARY=snhd_pre_rope NO_PADDING=1 MASK_KIND=none PROMPTFA_PAD_HEAD_DIM_TO=80 ASCEND_RT_VISIBLE_DEVICES=1 bash run_npu_attention_only_repro.sh
ATTENTION=prompt_flash_attention INPUT_BOUNDARY=snhd_pre_rope NO_PADDING=1 MASK_KIND=none PROMPTFA_PAD_HEAD_DIM_TO=88 ASCEND_RT_VISIBLE_DEVICES=1 bash run_npu_attention_only_repro.sh
ATTENTION=prompt_flash_attention INPUT_BOUNDARY=snhd_pre_rope NO_PADDING=1 MASK_KIND=none PROMPTFA_PAD_HEAD_DIM_TO=96 ASCEND_RT_VISIBLE_DEVICES=1 bash run_npu_attention_only_repro.sh
ATTENTION=prompt_flash_attention INPUT_BOUNDARY=snhd_pre_rope MASK_KIND=current MASK_RANK=4 PROMPTFA_PAD_HEAD_DIM_TO=80 ASCEND_RT_VISIBLE_DEVICES=1 bash run_npu_attention_only_repro.sh

# Mask-rank contract checks.
ATTENTION=prompt_flash_attention MASK_KIND=current MASK_RANK=2 ASCEND_RT_VISIBLE_DEVICES=1 bash run_npu_attention_only_repro.sh
ATTENTION=prompt_flash_attention MASK_KIND=current MASK_RANK=3 ASCEND_RT_VISIBLE_DEVICES=1 bash run_npu_attention_only_repro.sh
```

The reference is always manual attention over the same boundary inputs and mask.
For non-`bnsd` boundaries, the manual reference replays the same producer ops
before attention, so this is a producer-boundary test rather than a different
math reference. The JSON reports overall and real/pad split diffs for the attention
output only. For PromptFA candidates, the script also computes eager PromptFA in
the same process and reports `candidate_vs_eager_promptfa` and
`eager_promptfa_vs_manual`, so GE/CANN drift can be separated from ordinary
PromptFA-vs-manual numerical differences. It also reports top diff locations and
per-head diff summaries, plus explicit nonfinite locations.

`PROMPTFA_PAD_HEAD_DIM_TO` is a PromptFA-only D-pad/barrier probe. It must not
change the manual reference. If set above `72`, the script pads Q/K/V only for
PromptFA, preserves `scale_value=1/sqrt(72)`, then slices the PromptFA output
back to D=72 before diffing. Do not conclude that D-alignment fixed anything
unless misaligned padded targets such as `73` and `88` fail while aligned targets
such as `80` and `96` pass. If all padded targets pass, the likely effect is a
Pad/Slice graph barrier rather than D-axis alignment. Check
`config.promptfa_call_head_dim`, `config.promptfa_head_dim_pad_extra`,
`config.promptfa_call_head_dim_fp16_32b_aligned`,
`attention_input_meta.promptfa_call_q_shape`, and
`attention_input_meta.promptfa_output_sliced_back_to_head_dim` in the JSON.

When NaNs are present, `candidate_first_vs_second_allclose_5e_2` is expected to
be false because `NaN != NaN`. Use
`candidate_first_vs_second_nonfinite_mask.nonfinite_mask_match` and
`candidate_first_vs_second_nonfinite_mask.nan_mask_match` to decide whether the
compiled graph is structurally deterministic. Use
`nonfinite_pattern_candidate_second.per_head` to check whether NaNs are isolated
to a head, row region, sequence range, or head-dimension range.

## CUDA Smoke

CUDA smoke uses manual attention only and is not authoritative NPU evidence:

```sh
bash run_cuda_manual_smoke.sh
```
