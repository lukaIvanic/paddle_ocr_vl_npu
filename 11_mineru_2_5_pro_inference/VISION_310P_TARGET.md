# Active MinerU 310P target: vision encoding

The optimization target is the **32-layer vision transformer**, not text decode.
Luka reports about 3.5k useful vision tokens/s on 310P versus 45k+ on 910B.
The saved matched 910B full-run reference records 44,143.9 useful raw vision
tokens/s. Those figures concern vision tokens, not generated text tokens.
Different masks, padding densities and crop distributions change the rate;
matched inputs and denominators are required for a cross-chip diagnosis.

The new decode KV-cache benchmark on `codex/mineru-full-model-kv-benchmark`
measured a different stage. Its rates, paged-cache layouts and decoder NZ
experiments do **not** explain this vision bottleneck. Keep that evidence as
separate decode work. `FULL_MODEL_KV_310P_HANDOFF.md` is not the handoff for this
investigation.

MinerU vision attention is full bidirectional attention with fresh Q/K/V in
each block. The active custom path is FP16, 16 heads, D80, manual FP32
LayerNorm, ordinary linears and compiled masked PromptFA. Packing preserves
independent image components, including an independent filler component.
There is no persistent autoregressive KV cache in this vision path.

The inspected vLLM-Ascend checkout on the 910B host is revision
`80610e4438dba05011b05f89fc45d91e96992671`. Its
`vllm_ascend/_310p/ops/mm_encoder_attention.py` uses:

- `_npu_flash_attention_unpad`, with fresh TND Q/K/V and CPU int32 lengths;
- zero-padding D80 to D128, retaining the D80 scale, then slicing back;
- no decode block tables or persistent KV cache.

Although that source declares a CPU-length cache and comments describe reuse,
the inspected `forward_oot` actually runs `torch.diff(cu_seqlens).to("cpu")`
on every invocation. Do not assume the comments prove cached metadata behavior.
The receiving agent must inspect its own installed revision.

The generic unquantized linear loader converts FP16 weights to NZ on 310P.
This is a separate variable from attention layout. The custom vision's 128
projection weights were native/ND in the saved captures. Stock runtime behavior
must still be verified on the actual 310P environment; the source audit is not
an observation of its running model.

Prior work already measured full 32-block vision with real production inputs:
[route profiles](references/vision_profiles_910b_20260905/RESULTS.md),
[attention matrix](references/attention_matrix_910b_20260905/RESULTS.md), and
[PMU investigation](references/attention_pipes_910b_20260906/RESULTS.md).
The large S5632 sample is MinerU's own layout image; it is not representative
of recognition-only crops in the PP-DocLayoutV3 hybrid. Keep it distinct from
crop routes, whose current cap is 602112 pixels / 3072 raw vision tokens.
The old 310P report also required correction of three-forward denominators
and duration_us versus AICore counters; do not repeat its uncorrected ratios.

## Current controlled experiment

`bench_production_vision_attention.py capture-crops` now runs actual crop images
through the existing patch embedding, position preparation, full compiled
32-block vision and merger. It saves the real block inputs, rotary factors,
mask and baseline features, plus image/model/tensor hashes. It never generates
synthetic hidden states or reconstructs Q/K/V.

`replay` reruns the **complete 32-block stack** on those captured inputs:

- `baseline`: compiled PromptFA D80, native weights;
- `pfa_nz_weights`: same compiled PromptFA, only 128 projection weights NZ;
- `pfa_d128`: same compiled PromptFA and native weights, only D128 padding;
- `eager_pfa`: full eager PromptFA control;
- `unpad_d128`: full eager stack using the stock private op's D128 contract;
- `unpad_d128_nz_weights`: same eager unpad stack, projection weights NZ.

Compare attention contracts within eager mode, and weight formats within the
same execution/attention path. A compiled-versus-eager ratio also includes
fusion and dispatch differences. This is the stock operator contract inside
the owned vision stack, not a benchmark of the entire stock vLLM wrapper.
Lengths are prepared once before replay; private-op CPU metadata consumption
remains inside each timed layer. Stock wrapper CPU preparation/copy behavior
may differ and is not represented as an exact vLLM runtime reproduction.

Headline rates count useful raw crop tokens once per completed full encoder
forward: sum(real_tokens) / sum(seconds). Padding tokens are reported separately.
CPU preprocessing/H2D, patch embedding, initial positions and merger are outside
the block replay timer; they actually run during capture. This boundary matches
the existing vision_transformer_blocks production metric. Record synchronized
wall time and NPU-event time; no page/s or decode tok/s claim belongs here.
Compile, weight conversion and profiler passes are excluded from warm timings.
Timing fails if new graphs or recompilation warnings occur inside the timer.
Feature drift is quantified independently and does not suppress timings merely
because outputs differ. Nonfinite outputs and NPU failures stop progression.

Inspect actual profiler formats: a loaded NZ descriptor is insufficient proof
that the compiled matmul consumes NZ. Count 32 attention calls and 128 projection
matmuls per full forward. Separate attention, linears, rotary, LayerNorm and
conversion costs using kernel duration_us with the correct forward denominator.
PMU engine times overlap; do not sum them as elapsed time.

The receiving-agent procedure is
[VISION_CROP_310P_HANDOFF.md](VISION_CROP_310P_HANDOFF.md). Its primary lanes
and compiled padding control are validated on 910B; actual 310P measurements
remain out of band. It also names a separate 310P-only approximate-precision
matrix rather than treating a 910B unsupported-device skip as validation.
See [the twelve-lane 910B results](references/vision_crop_contracts_910b_20261007/RESULTS.md)
for warm full-encoder rates, actual kernel formats and independent feature drift.
