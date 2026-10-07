# Frozen text-forward rotary/layout/SwiGLU experiment

User-requested experiment on 2026-10-07. The candidate implementations are
opt-in; the existing baseline model is retained. Numerical norms and all
learned projection weights remain unchanged in this matrix.

## Workload and measurement

Reuse the exact seven resident tensors and frozen eager hidden-state reference
from [the original isolated text experiment](../text_forward_20261007/README.md):
B1, FP16, S1274, 36 layers, H2560, Q32/KV8, D128. The timed/profiler callable
contains the complete text transformer, three DeepStack additions and final
norm. Vision, preprocessing, model/input transfers, retrieval projection and
compile are outside the window. No generation or KV cache is introduced.

Each variant executes in separate raw-eager and TorchAir processes. Both lanes
use five warmups, 30 clean forwards before profiling, three pipe-profiled
forwards, and 30 clean forwards afterwards. The same automatically selected
physical 910B2 is retained for the entire matrix. A second compiled baseline
at the end checks for timing drift. Each candidate owns an isolated compile
cache keyed by its options and source hash.

Baseline compiled output must remain bit-exact to the frozen eager reference.
Candidates retain the existing elementwise atol=0.002/rtol=0.002 gate, both
against the frozen baseline and against their own raw eager implementation.
`--diagnostic-parity` permits profiling rejected candidates while retaining
`adoption_eligible=false`; it does not relax the acceptance threshold.

## Controlled variants and source operations

| Variant | Attention layout | Rotary | Packed gate/up activation |
|---|---|---|---|
| baseline | BNSD | Existing split/negate/concat | Existing SiLU × up |
| bsnd | BSND | Existing half rotary | Existing SiLU × up |
| rotary_bnsd | BNSD | Separate native RotaryMul Q/K | Existing SiLU × up |
| apply_bnsd | BNSD | Joint native ApplyRotary on BSND Q/K before attention conversion | Existing SiLU × up |
| apply_bsnd | BSND | Joint native ApplyRotary | Existing SiLU × up |
| swiglu | BNSD | Existing half rotary | Native SwiGLU |
| apply_bsnd_swiglu | BSND | Joint native ApplyRotary | Native SwiGLU |

The implemented candidates are in `text_forward_variants.py`; baseline uses
the original `OptimizedTextStage` directly. Q/K learned norms run before rotary
in every variant. The existing prepared interleaved-MRoPE factors are reused;
no checkpoint row permutation or rotary-frequency change is made. Attention
keeps native GQA and the same bool causal mask, scale and sparse-mode contract.

The original compiled profile identifies the target operations across all
36 layers:

| Source operation | Original profiled kernel work per forward |
|---|---:|
| Q/K/V and attention-output layout conversion | 144 Transpose, 1.686 ms |
| Q/K half rotary | 72 Split + 72 Neg + 72 Concat + 72 fused MulMulAdd, 2.694 ms |
| Packed gate/up split | 36 Split, 1.292 ms |
| SiLU(gate) × up | 36 fused SwishMul, 1.651 ms |

These kernel sums are diagnostic targets, not predicted removable wall time.
The 36 QKV splits and four projection matmuls per layer are retained. BSND
tests the attention implementation as well as surrounding transposes; native
rotary/SwiGLU may change arithmetic rounding and must pass the numerical gate.

## Reproduction

After pulling the experiment branch on the 910B container:

```sh
TEXT_VARIANT_ROOT=tmp/21_colqwen3_4b_inference/text_variants_NEW \
  bash 21_colqwen3_4b_inference/run_text_variants.sh
```

The committed runner sources `npu-setup` before enabling strict unset-variable
checking, checks the selected card, copies the existing baseline graph into a
new cache, and runs all variants sequentially. It records exact commands,
source commit, physical device, logs and exit codes. The analyzer checks frozen
input identity, one physical NPU, 36 text attention calls and 144 projection
matmuls, absence of vision/preparation kernels, and conservation of CANN counts
and timings. It exports every kernel shape/name group and numerical gate.

## Execution status

Candidate source is committed and pushed on
`codex/colqwen-warm-forward-profile`, with runner fixes at `6ac2bbb8`.
Python compilation, shell syntax and the four independent profile-accounting
tests pass locally. **No new candidate forward, numerical gate or latency has
been validated on NPU.**

Initial launches stopped during setup. The vendor environment wrapper was
changed to source its scripts before enabling strict unset-variable checking.
A later attempt selected physical NPU3, but failed before timing because
runner-source hashing passed a string to the Path-based helper. That bug was
fixed and the runner now stops immediately when its baseline control fails.
Only that experiment's processes were terminated; failed commands/logs are
retained under [startup_evidence](startup_evidence/), with hashes and original
paths in [manifest.json](manifest.json).

After the fixes, launches stopped at `npu-status: no free NPU found`.
Direct ownership checks identified an RWKV reranker matrix benchmark on
physical NPU1 and a MinerU vision-attention benchmark on physical NPU3.
Those jobs are separate from this experiment. The matrix awaits an exclusive
910B2 slot; no baseline or candidate speedup is inferred from startup logs.
