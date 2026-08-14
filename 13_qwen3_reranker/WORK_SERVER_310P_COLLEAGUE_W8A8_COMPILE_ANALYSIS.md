# 310P colleague W8A8 graph-compile failure: problem statement and analysis

## Purpose

This is a pull-only investigation for the 310P work-server agent. The colleague's
actual implementation exists only on that server. Inspect that code directly.
Do not infer its behavior from this repository's implementation.

The goal is to identify the first causal error in the colleague's compiled W8A8
path and determine whether it is:

1. an unsupported Atlas 310P operator configuration;
2. a tensor dtype, shape, format, or transpose contract violation;
3. a TorchAir/GE lowering problem;
4. or a TBE compiler-process/resource failure.

Do not edit tracked files, create commits, push, or create branches on the 310P
server. If a source change is needed, report the smallest proposed diff with the
exact source location. The local authoring agent will implement and push it.

Report progress immediately after each phase below. Do not wait until the whole
investigation finishes to report the first useful result.

## Reported environment and symptom

- Product: Atlas 310P.
- CANN path in the error: `/usr/local/cann-9.1.0-beta.1/...`.
- Execution mode: static graph compilation.
- Failing model shape reported by the colleague: batch 8, total sequence length
  128, therefore flattened linear-token dimension `M = 8 * 128 = 1024`.
- Quantization calls reported by the colleague:
  - `torch_npu.npu_quantize(..., axis=-1, div_mode=False, dtype=torch.qint8)`;
  - `torch_npu.npu_quant_matmul(..., scale=self.deq_scale,
    bias=self.quant_bias, output_dtype=torch.float16)`.
- Weight processing reportedly includes:
  - a repeated FP16 activation scale;
  - reciprocal/input-offset preparation;
  - INT8 weight conversion to `ACL_FORMAT_FRACTAL_NZ`;
  - a static dequant scale created with
    `torch_npu.npu_trans_quant_param(...)`.
- Partial compiler failure reported by the colleague:
  - `Failed to compile Op Mul_3`;
  - the referenced implementation is a TBE dynamic `mul.py`;
  - this is followed by failures to recompile unrelated single operations such
    as `Swish` and `trans_TransData_229`.

The operation name in the final `E40021` line is not yet established as the
root cause. Repeated failures for unrelated operations can be secondary errors
after the TBE compiler's main process exits.

## Known 310P operator contract

Treat “Atlas inference products” in the Huawei documentation as the product
family containing Atlas 310P.

### `torch_npu.npu_quantize`

The reported `axis=-1`, `div_mode=False`, `dtype=torch.qint8` combination is
documented for Atlas inference products.

For this mode:

- input may be FP16 or FP32 in ND format;
- scale may be FP16 or FP32;
- a one-dimensional scale must have length 1 or match the last input axis;
- `zero_points` may be `None`;
- when non-`None`, `zero_points` must have the same shape and dtype as `scales`;
- `axis=-2` and INT4 output are not supported on Atlas inference products;
- `div_mode=False` means that the supplied scale is used as a multiplier, not
  as a divisor.

Reference:

- <https://www.hiascend.com/document/detail/zh/Pytorch/700/apiref/apilist/ptaoplist_000535.html>

### `torch_npu.npu_quant_matmul`

The conservative FP16-output 310P contract is:

- `x1`: INT8, logical shape `[M, K]`, ND;
- `x2`: INT8, logical shape `[K, N]` as observed by QuantMatmul;
- `scale`: packed INT64/UINT64, shape `[1]` or `[N]`;
- `offset`: `None`;
- `pertoken_scale`: `None` because 310P does not support it here;
- `bias`: `None` or INT32, normally shape `[N]` for a two-dimensional output;
- `output_dtype`: FP16;
- the relevant K or N dimension must not exceed 65535.

For graph-mode FP16 output without per-token scaling, pass a prepacked scale
created by `torch_npu.npu_trans_quant_param`. Do not construct or pack it inside
the compiled forward.

References:

- <https://www.hiascend.com/document/detail/zh/Pytorch/700/apiref/apilist/ptaoplist_000532.html>
- <https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta3/API/aolapi/context/ops-nn/aclnnQuantMatmulV3.md>
- <https://www.hiascend.com/document/detail/zh/Pytorch/710/apiref/torchnpuCustomsapi/context/torch_npu-npu_trans_quant_param.md>

### FRACTAL_NZ weight ordering

The Atlas inference route differs from the A2/910B route.

For the documented high-performance 310P layout:

1. begin with the stored INT8 weight in linear form `[N, K]`;
2. convert that stored `[N, K]` tensor to FRACTAL_NZ once;
3. express `.transpose(-2, -1)` inside the compiled forward;
4. verify that QuantMatmul therefore observes logical `[K, N]`.

Do not assume that a tensor reporting shape `[N, K]` after the format cast can
be supplied directly as logical `[K, N]`. This exact mistake previously caused
the error `x1=[512,2560]`, `x2=[9728,2560]`: both K axes appeared in the last
dimension, so the matrix multiplication contract was invalid.

This repository's current reference is:

- `13_qwen3_reranker/local_reranker_w8a8.py`, especially
  `W8A8Linear.quant_matmul_from_quantized` and
  `prepare_w8a8_weight_format`;
- commit `4b9091b`, which moved the 310P transpose into the graph;
- commit `410f6c5`, which promoted the validated 310P layout.

Use it only as a comparison. First establish what the colleague's code actually
does.

## Important unsupported or misleading alternatives

- FP16 or FP32 QuantMatmul bias is supported on newer A2/A3 products but not on
  Atlas 310P. A 310P QuantMatmul bias must be INT32 or absent.
- QuantMatmul per-token scale is not supported on 310P.
- `torch_npu.contrib.module.LinearA8W8Quant` is only a wrapper around
  `npu_quant_matmul`; it does not provide a different lowering path.
- `torch_npu.npu_weight_quant_batchmatmul` is a valid A16W8 fallback, but it is
  not the A8W8 path under investigation.
- Public `torch_npu.npu_ffn` and `torch_npu.npu_swiglu` support targets newer
  A2/A3 products, not 310P. Do not substitute them as a 310P fix.
- Plain FP16 SiLU/Swish and elementwise Mul are supported on 310P. Their names
  appearing late in a compiler-failure cascade does not prove that either
  operation is unsupported.

## Phase 1: inspect the colleague's real code

Before running or changing anything, report the exact files, classes, methods,
and line numbers for:

1. weight quantization and weight loading;
2. `process_weights_after_loading` or its equivalent;
3. activation scale and zero-point construction;
4. `npu_quantize`;
5. `npu_quant_matmul`;
6. graph compilation and cache configuration;
7. the Qwen MLP activation and gate multiplication around the failing linear.

For every relevant tensor, report both how it is constructed and its runtime
metadata immediately before the operator call:

| Tensor | Required evidence |
|---|---|
| Quantize input | shape, dtype, NPU format, contiguous status |
| Quantize scale | shape, dtype, value summary, static or graph-produced |
| Quantize zero-point | `None` or shape/dtype/value summary |
| Quantized activation | shape, dtype, NPU format |
| Stored weight | shape, dtype, format before and after FRACTAL_NZ conversion |
| QuantMatmul x2 expression | exact source expression, including transpose |
| Dequant scale | shape, dtype, where `npu_trans_quant_param` runs |
| Quant bias | shape, dtype, formula used to compute it |
| QuantMatmul output | requested and observed dtype |

Also answer these specific questions:

- Does `self.quant_bias` have dtype `torch.int32` on the NPU?
- Is `self.deq_scale` already packed before graph capture?
- Is the activation multiplier reciprocal consistent with
  `div_mode=False`?
- If `zero_points` is not `None`, does it exactly match the scale's shape and
  dtype?
- Does the graph contain the 310P weight transpose, or is the formatted
  `[N,K]` tensor passed directly?
- Does the implementation flatten `[B,S,K]` to `[B*S,K]` before quantization
  and QuantMatmul?
- Is any `.to(...)`, scale multiplication, reciprocal, repeat, format cast, or
  `npu_trans_quant_param` unexpectedly executing inside the compiled forward?

Report Phase 1 immediately. If a deterministic contract violation is found,
stop before running the expensive full graph and propose the minimal correction.

## Phase 2: find the first causal compiler error

If Phase 1 finds no definite contract violation, reproduce the failing
`B=8, T=128` compile once with a new, isolated compile-cache directory.

Preserve and report:

- exact command;
- git commit and any untracked colleague-code identity available;
- Python, PyTorch, TorchNPU, TorchAir, and CANN versions;
- physical NPU and device name;
- all relevant compile environment variables;
- exit code;
- host free memory before and during compilation;
- the first error, not only the final `E40021` summary.

Search at least 200 lines before the first `Failed to compile Op` message for:

- Python traceback;
- `Killed`, exit code `-9`, or OOM messages;
- SIGSEGV or exit code `-11`;
- `main process disappeared`;
- `ConnectionRefusedError`;
- `EOFError`;
- `bad_alloc`;
- a concrete unsupported dtype/format/shape diagnostic.

Determine from graph dumps or source mapping which model expression became
`Mul_3`. It could be activation quantization, SiLU's multiplication, the gated
MLP multiplication, or another scale operation. Do not infer this from the GE
node name alone.

Report Phase 2 immediately with the earliest causal log excerpt.

## Phase 3: isolate operator support from full-graph pressure

Use fresh Python processes and distinct cache directories for each lane. Do not
run concurrent compiles. Keep all shapes static and use `M=1024`.

Run the smallest available tests in this order:

1. quantize-only: FP16 `[1024,K]` to INT8;
2. QuantMatmul-only with a prequantized INT8 `[1024,K]` activation;
3. one combined `npu_quantize -> npu_quant_matmul` linear;
4. one complete colleague MLP layer;
5. full colleague model only if the preceding lanes pass.

As an environment control, compare against this repository's existing isolated
probe at the same `M=1024`:

- `13_qwen3_reranker/probe_310p_w8a8_ops.py --tokens 1024`.

Resolve the server's existing model and cache paths from prior successful runs;
do not invent or hardcode a new server path. Reuse model weights, but use a new
compile-cache directory for this diagnosis.

For one diagnostic pass, reduce compile concurrency:

```bash
export TE_PARALLEL_COMPILER=1
export MAX_COMPILE_CORE_NUMBER=1
```

Record the original values first. This diagnostic will compile more slowly but
reduces TBE peak host-memory pressure. Do not present it as the final performance
configuration.

Report each lane immediately after it finishes, with PASS/FAIL and the first
causal error.

## Interpretation

- Quantize-only failure indicates a scale, zero-point, axis, dtype, or
  quantizer-lowering problem.
- Prequantized QuantMatmul-only failure indicates x2 logical layout, FRACTAL_NZ,
  dequant scale, bias, or QuantMatmul lowering.
- Both isolated operations passing but the combined linear failing indicates a
  converter/fusion interaction.
- A combined linear passing but one MLP layer failing points to SiLU/gate
  interaction or graph size.
- All small lanes passing while the full graph fails, especially only at
  `M=1024`, points to compiler resources, cache corruption, or a whole-graph
  lowering issue rather than unsupported W8A8 operations.
- The failure disappearing with single-process compilation, together with high
  host-memory use or a killed TBE process, is evidence of compiler resource
  pressure. It is not proof that the named final operation was unsupported.

## Required final report

Return one concise report with:

1. `CODE_AUDIT`: exact source locations and the tensor-contract table;
2. `FIRST_CAUSAL_ERROR`: earliest relevant log excerpt and interpretation;
3. `ISOLATION_RESULTS`: one row per lane, including the M=1024 control;
4. `ROOT_CAUSE`: confirmed, probable, or unresolved, with confidence;
5. `MINIMAL_PROPOSED_CHANGE`: exact file/location and pseudodiff, or `none`;
6. `NO_SERVER_SOURCE_MUTATION`: confirmation that no tracked source was edited;
7. paths to all preserved logs and graph dumps.

Do not report a later `Mul`, `Swish`, or `TransData` compilation message as the
root cause unless the earlier logs and the isolated lane prove it.
