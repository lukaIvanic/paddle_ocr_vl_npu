# Vision weight NZ pre-formatting: read-only audit

Date: 2026-09-05. No model, cache, compiler or weight-format changes were made;
no new NPU inference was run for this audit.

## Confirmed custom-path facts

- `configure_decode_weight_format` in `local_modeling_mineru.py` iterates
  `model.model.layers` and creates a separate LM-head weight copy. It does not
  visit `model.visual`.
- `StaticMinerUVisionBlocks` uses the original visual blocks. Its production
  `projection_impl="linear"` calls the QKV, attention output, MLP FC1 and MLP
  FC2 linears without explicit vision weight format conversion.
- The saved 910B S5632 pipe profile lists ND/ND/ND inputs (activation, weight,
  bias) for all four projection families. Each has 96 calls across three
  forwards. The reported 2D weight shapes are:

  | Projection | Weight shape | Kernel | Duration / forward |
  |---|---|---|---:|
  | QKV | 3840 × 1280 | MatMulV3 | 5.68548 ms |
  | Attention output | 1280 × 1280 | MatMulV3 | 2.11901 ms |
  | MLP FC1 | 5120 × 1280 | MatMulV3 | 7.45938 ms |
  | MLP FC2 | 1280 × 5120 | MatMulV3 | 7.79681 ms |

- There are no separate TransData kernel records in that capture. This does
  not prove an absence of packing or format work inside the MatMul kernel.

## A relevant generic vLLM-Ascend difference

Read directly from the clean files in `/vllm-workspace/vllm-ascend` on the
910B host, commit `80610e4438dba05011b05f89fc45d91e96992671`, tag
`v0.21.0rc1`:

- `vllm_ascend/ops/linear.py:78–95`:
  `AscendUnquantizedLinearMethod.process_weights_after_loading` calls
  `maybe_trans_nz(layer.weight.data)` when the prefix does not contain `conv1d`.
- `vllm_ascend/ops/linear.py:123–124`: the shared linear base selects
  `AscendUnquantizedLinearMethod` when no quantization config is supplied.
  The Ascend column/row parallel linear classes invoke that base initializer.
- `vllm_ascend/utils.py:206–246`: `_should_trans_nz` rejects FP32 and meta
  tensors, then returns true for 310P. `maybe_trans_nz` performs
  `torch_npu.npu_format_cast(weight, ACL_FORMAT_FRACTAL_NZ)` when permitted.
- The associated vLLM checkout, commit
  `ad7125a431e176d4161099480a66f0169609a690`, uses column/row parallel linears
  for Qwen2-VL's vision QKV, output projection and MLP (`qwen2_vl.py:254–309`).

This supports an expected **generic stock-310P vision-weight conversion path**,
not a MinerU-specific attention patch. The code was inspected on the 910B host;
this is not a fresh observation of the work server's loaded stock model tensors.
The 310P agent should first report formats visible in its existing custom
captures; no stock model load or new experiment is authorized by the correction
handoff.

## Why NZ is worth testing, and what it would not establish

Huawei's [Ascend C matrix-multiplication guide](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/80RC1alpha001/devguide/opdevg/ascendcopdevg/atlas_ascendc_10_0060.html)
describes NZ as a special blocked format introduced for efficient Cube
computation, with support covering 310P and Atlas A2. This makes weight layout
a technically motivated variable, not proof of a speedup in this graph.

Pre-formatting fixed weights could change the selected matmul path and move
some weight-format work to setup. Whether it actually does so must be measured
and the resulting loaded/compiled formats verified. An ND label alone does not
quantify conversion overhead, and the absence of separate TransData kernels
does not resolve it.

The first controlled candidate would convert only the 128 transformer-block
projection weights (four per layer), once after loading and before compiling.
Keep FP16 values, biases, ordinary linear calls, manual-FP32 LayerNorm, rotary,
native D80 PromptFA, masks, packing and real production inputs unchanged.
Do not conflate this with Q/K/V head-dimension padding, grouped matmul, activation
format changes or quantization. Patch embedding and merger remain unchanged in
that first candidate.

Such a test needs an explicit NZ-specific cache identity while retaining the
baseline cache. The current vision runtime cache key has no explicit weight-
format field; do not assume changing weights behind an existing cached callable
is safe. No new cache was created in this audit.

Compare logical weights before/after conversion, full vision features,
production outputs, warm latency, formats and kernel breakdown. A declared
conversion is not enough: verify `get_npu_format` and actual graph input formats
so silent ND fallback is detected. This remains a proposed experiment, not a
validated implementation or an instruction to the work agent to modify code.

Weight conversion directly targets projection matmuls. It does not directly
replace PromptFA's activation×activation attention computation or FP32
LayerNorm. If corrected 310P accounting still puts projections near 20% of
S5632 kernel time, eliminating that entire portion would only give about 1.25×
kernel-region speedup with everything else unchanged. Real NZ gains would be
smaller under that assumption; smaller shapes, where projections contribute
more, could have a different payoff. Recompute the share using duration_us
before making a quantitative prediction.
