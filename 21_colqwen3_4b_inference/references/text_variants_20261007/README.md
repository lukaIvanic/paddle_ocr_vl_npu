# ColQwen text-forward rotary/layout/SwiGLU results — 910B2 NPU 0

**Joint native ApplyRotary + BSND attention lowers warmed compiled text-forward
latency by 4.54% with bit-exact hidden states on this B1/S1274 case.** An
alternated same-process check measured **63.110 ms control → 60.248 ms
candidate**, across 120 clean forwards each. The complete separate-process
matrix also measured eager and compiled execution and profiled every variant.
Separate RotaryMul has an eager/compiled numerical discrepancy; native SwiGLU
changes the hidden states beyond the existing gate. Those results remain
diagnostic. The baseline and opt-in candidate implementations are both retained.

## Workload and measurement

The entire matrix ran on physical **NPU 0, Ascend 910B2**, source commit
`3e026452`, on `liteserver-c001-4`. The user explicitly reserved that card.
The resident Clef vLLM service on port 18123 was left running; its running,
waiting, request-success and token counters remained zero before and after
the matrix. [The activity check](evidence/reservation_activity.json) records
both samples. This checks that service's activity, without claiming exclusive
use of the shared host.

All cases reuse the exact seven resident tensors and frozen eager hidden-state
reference from [the original isolated text experiment](../text_forward_20261007/README.md):
**B1, FP16, S1274, 36 layers, H2560, Q32/KV8, D128**. The timed and profiled
callable contains the complete text transformer, three DeepStack additions and
final norm. Vision, preprocessing, input/model transfers, retrieval projection,
compile and validation are outside the window. There is no generation or KV
cache. Native weights, internal formats enabled, HF32 disabled, NPU JIT disabled,
and four CPU threads match the control.

Each variant executes in separate raw-eager and TorchAir processes: five
warmups, 30 clean forwards before profiling, three pipe-profiled forwards, and
30 clean forwards afterwards. Each candidate owns an isolated compile cache
keyed by its options and source hash. The table reports the mean of all 60
clean full-forward wall timings; profiler kernel sums are a separate diagnostic.

Baseline compiled output must remain bit-exact to the frozen eager reference.
Candidates retain the existing elementwise **atol=0.002 / rtol=0.002** gate
against both the frozen baseline and their own raw-eager implementation.
`--diagnostic-parity` permits profiling rejected candidates while retaining
`adoption_eligible=false`; it does not relax that threshold.

## Complete matrix

| Variant | Raw eager ms | TorchAir ms | Latency reduction vs opening compiled control | Numerical result |
|---|---:|---:|---:|---|
| Original BNSD control | 78.033 | 61.871 | +0.00% | Bit-exact in both lanes |
| BSND only | 74.491 | 62.335 | -0.75% | Bit-exact in both lanes |
| Separate RotaryMul + BNSD | 74.927 | 61.057 | +1.32% | Eager/compiled mismatch; compiled matches original |
| Joint ApplyRotary + BNSD | 73.433 | 60.570 | +2.10% | Bit-exact in both lanes |
| Joint ApplyRotary + BSND | 71.120 | 59.723 | +3.47% | Bit-exact in both lanes |
| Native SwiGLU only | 76.781 | 60.743 | +1.82% | Fails frozen-reference gate in both lanes |
| Joint ApplyRotary + BSND + SwiGLU | 69.500 | 58.889 | +4.82% | Fails frozen-reference gate in both lanes |

All 15 captures completed with exit code 0, finite/replay checks passed, one
frozen-input hash and physical card were retained, and every profile passed the
text-only attention/projection and kernel-accounting checks. `completed` alone
does not mean numerical adoption is allowed; the explicit gate is shown above.

The closing compiled control averaged **62.763 ms**, **1.44% above** the
opening control. A targeted same-process ABBA repetition follows below to
check the winner against local drift rather than relying on separate-process
means alone.

## Alternated control/candidate repeat

The closing-control drift justified a narrow confirmation run, rather than
repeating the whole matrix. `bench_text_abba.py`, source commit `c7bf9507`,
loads the exact same frozen tensors and existing compiled graph caches into
one process. It checks eager/compiled/frozen parity outside the timer, warms
both paths five times, then repeats **baseline / candidate / candidate /
baseline** three times with 20 clean forwards per block. No profiler is active.

| ABBA cycle | Control ms | Joint ApplyRotary + BSND ms | Latency reduction |
|---|---:|---:|---:|
| 1 | 63.157 | 60.248 | 4.61% |
| 2 | 63.092 | 60.248 | 4.51% |
| 3 | 63.082 | 60.246 | 4.49% |
| All 120 forwards per path | 63.110 | 60.248 | **4.54%** |

All setup and final outputs are bit-exact to the frozen reference. Each cycle
shows a 4.49–4.61% reduction, supporting the gain despite drift in the earlier
separate-process sweep. This alternated comparison is the primary estimate of
the winner's relative speedup; absolute times depend on the run's conditions.
It validates this captured B1/S1274 shape, rather than other sequence lengths,
batches or retrieval quality.

[Exact command](evidence/abba/command.txt), [full timing/parity/cache result](evidence/abba/output/result.json),
and [reservation activity through completion](evidence/abba/reservation_activity.json)
are retained. Both compiled graph files were present before setup and all
cache loads were outside the warmed timing window.


## What changed in the compiled profile

The candidate implementation is in `text_forward_variants.py`; the baseline
uses the original `OptimizedTextStage`. Every layer retains learned Q/K norms,
the original interleaved-MRoPE factors, native GQA, the same causal bool mask,
scale and sparse-mode contract. No checkpoint row permutation is introduced.

| Kernel work per full forward | Original BNSD | Joint ApplyRotary + BSND |
|---|---:|---:|
| Projection matmuls | 144 / 35.469 ms | 144 / 35.523 ms |
| PromptFlashAttention | 36 / 10.847 ms | 36 / 11.999 ms |
| RMSNorm work excluding shared residual cast | 653 / 6.644 ms | 653 / 6.590 ms |
| Rotary | 288 / 2.651 ms | 36 / 0.832 ms |
| Layout and projection splits | 216 / 3.655 ms | 72 / 1.841 ms |
| SiLU × up | 36 / 1.591 ms | 36 / 1.594 ms |
| Residual/DeepStack additions and shared cast | 75 / 0.930 ms | 75 / 0.999 ms |
| Data/input bookkeeping | 1 / 0.006 ms | 1 / 0.005 ms |
| Total | 1,449 / 61.793 ms | 1,053 / 59.384 ms |

The exact source operations and CANN changes are:

- **Q/K rotary, all 36 layers.** Baseline Q/K each split their 128 dimensions
  into halves, negate/concatenate, then compute the two products and sum.
  The 72 SplitVD, 72 Neg, 72 Concat and 72 fused MulMulAdd kernels become
  36 joint `ApplyRotaryPosEmb` calls with Q `[1,1274,32,128]`,
  K `[1,1274,8,128]` and factors `[1,1274,1,128]`.
- **Attention layout, all 36 layers.** BSND removes the Q/K/V-to-BNSD and
  attention-output-to-BSND conversions: all 144 Transpose kernels disappear,
  saving 1.813 ms in the fresh baseline profile. Attention itself grows by
  1.152 ms in the combined candidate. That is why transpose count alone
  cannot predict full-forward benefit.
- **BSND alone** removes the same transposes but is 0.75% slower in the
  complete matrix. Its attention grows from 10.847 to 12.031 ms and other
  kernel costs increase. Joint rotary makes the overall combination useful.
- **Packed gate/up activation.** Native SwiGLU consumes `[1,1274,19456]`
  directly, removing all 36 gate/up SplitVD kernels (1.282 ms) and replacing
  36 SwishMul kernels (1.591 ms) with 36 SwiGlu kernels (1.539 ms).
  The measured whole-forward reduction is 1.82%, with numerical rejection.
  These profile sums are diagnostic and are not predicted wall-time savings.

The remaining 72 splits in the bit-exact winner are exactly 36 packed QKV
splits (`[1,1274,6144]`, 0.558 ms) and 36 packed gate/up splits
(`[1,1274,19456]`, 1.283 ms). Normalization is unchanged in this matrix.

## Numerical findings

Joint ApplyRotary in either attention layout, and BSND alone, produce
**bit-exact** full hidden states in eager and compiled execution for this
captured page.

Separate `RotaryMul` is different: its raw-eager full hidden state differs
from the frozen reference (max abs **0.666016**, mean abs **0.010420**, cosine
**0.9999826**), while its compiled full hidden state is **bit-exact to the
original reference**. Compiled vs that candidate's own eager reference
therefore fails. This is a backend-dependent numerical discrepancy, not a
claim that the compiled output is wrong. It is ineligible under the paired
validation contract; joint ApplyRotary provides a faster consistent candidate.

SwiGLU alone and the combined SwiGLU candidate both differ from the frozen
hidden state (max abs **0.656250**, mean abs **0.010422**, cosine
**0.9999838**). Compiled execution is bit-exact to each candidate's own eager
execution, so the discrepancy already exists in the raw-eager activation
replacement. A cause such as intermediate rounding has not been isolated in
this experiment. The existing tolerance is retained; these numbers do not
establish a retrieval-quality regression or acceptable quality parity.

## Remaining projection cost

The validated winner's four matmuls per layer occupy **35.523 ms**, about
**59.8%** of its profiled kernel sum:

| Source projection | CANN input/weight shapes | Calls | Kernel ms per forward |
|---|---|---:|---:|
| Packed gate/up | `[1274,2560] × [19456,2560]` | 36 | 16.631 |
| MLP down | `[1274,9728] × [2560,9728]` | 36 | 9.369 |
| Packed QKV | `[1274,2560] × [6144,2560]` | 36 | 5.300 |
| Attention output | `[1274,4096] × [2560,4096]` | 36 | 4.223 |

Gate/up plus down account for **26.000 ms**. The next larger opportunity is
therefore in the MLP projections/activation path, with the native SwiGLU
numerical discrepancy as an unresolved gate. The tested rotary/layout change
reduces the smaller surrounding work without changing these matmuls.

## Reproduction and evidence

After pulling the experiment branch into the 910B container, run a fresh matrix
on the explicitly user-reserved card:

```sh
TEXT_PHYSICAL_NPU=0 \
TEXT_VARIANT_ROOT=tmp/21_colqwen3_4b_inference/text_variants_NEW \
  bash 21_colqwen3_4b_inference/run_text_variants.sh
```

`run_text_variants.sh` first sources `npu-setup`, then applies the explicit
physical-card override. It records exact commands, source commit, physical
device, logs and exit codes. The analyzer checks 36 text attention calls and
144 projection matmuls, absence of vision/preparation kernels, identical
frozen inputs, one card, and conservation of CANN counts/timings.

[comparison.json](evidence/comparison.json) retains clean timing distributions,
numerical gates and semantic profile categories. Each variant/lane under
[evidence](evidence/) includes command, exit code, log, result, parsed report,
kernel accounting and **every kernel shape/name group**. The original kernel
CSVs and step traces are mirrored under ignored local `tmp/` and remain on the
server with the full traces, frozen tensor snapshots and isolated graph caches.
[manifest.json](manifest.json) records hashes and original locations.

Raw eager and compiled captures are separate for every candidate. Example
compiled kernel groups: [control](evidence/baseline/torchair/output/profiles/pipe/kernel_shape_groups.json),
[bit-exact winner](evidence/apply_bsnd/torchair/output/profiles/pipe/kernel_shape_groups.json),
[SwiGLU diagnostic](evidence/swiglu/torchair/output/profiles/pipe/kernel_shape_groups.json).

Earlier setup-only failures remain under [startup_evidence](startup_evidence/).
They predate this successful NPU0 matrix and are not inference measurements.
The matrix commit is `3e026452`; the drift-check harness is `c7bf9507`.
