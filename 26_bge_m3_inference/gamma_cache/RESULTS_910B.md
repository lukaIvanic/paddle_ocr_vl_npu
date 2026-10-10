# 910B gamma/beta cache: completed control, no consistent speedup

The FP16 gamma/beta-to-FP32 casts were hoisted once per core, with an 8-KiB
persistent UB cache and explicit MTE2→V readiness. Numerical results are exact
against unchanged V2. Vector busy time fell, but kernel duration did not improve
consistently. Keep the candidate isolated; it is not a production-default change.

## Contract and implementation

Implementation commit: `2a9c9702`. Upstream ops-nn v9.1.0:
`ceb4536a2bd6fc99b85aec9d0fdcc0f470376292`. The committed patch changes only
`add_layer_norm_static_quant_normal_kernel.h` and V2's three normal FP16 template
instantiations. It preserves reduction arithmetic/order, Mul then Add operand
order, quantization, bias handling, row buffering, GM copies, and the host tiler.
The original FP16 gamma/beta buffers remain allocated. New FP32 buffers never
alias row scratch, and survive every row chunk on a core.

Scope: architecture 2201, FP16, width/stride 1024, normalized output present,
BUFFER_NUM=1, rowStep 1..13. Only V2 keys 1000/1001/1002 opt in. Other contracts
retain the original branch. Tested attributes: static quantization, one FP16
per-tensor multiplication scale, no additional sum output, epsilon 1e-5.

Installed platform UB is 196608 bytes; **runtime tiler logs report 196352 usable
bytes**. The unchanged tiler estimates `18440*r + 4608` bytes and chooses at most
10 rows. Exact candidate allocations are `14336*r + 12320` bytes, plus 2048 for
broadcast bias. At r=10 this is **157728 bytes**, leaving **38624 usable bytes**.
The source patch's 256-KiB example is only a looser upper-bound calculation; it
was an initial hardware-size assumption, corrected in the README after reading
the platform and runtime logs. It is not the capacity used for this result.
For r>=3 the existing tiler estimate exceeds actual candidate broadcast UB by
`4104*r-9760` bytes. r=1/2 use at most 43040 bytes on this device.

Both packages retain the V2 GE/ACLNN/kernel names, as explicitly authorized for
fresh-process comparison. Their vendor identities, install paths, source trees,
Python compiler modules, and graph caches are separate. Only V2 dispatch remains
in each graph package; dependency operator overrides are removed. Baseline and
candidate were never loaded in the same process. Global CANN was not modified.

## Machine, dispatch and validation

2026-10-10, liteserver-c001-4, aarch64, **Ascend910B2 physical NPU 3**, CANN 9.0.1,
Python 3.12.13, torch 2.10.0+cpu, torch-npu 2.10.0.post2, Transformers 5.5.4.
The setup helper selected the same idle NPU for direct and model phases.
No 310P or CPU inference result is claimed.

| Flattened rows × width | AIV blocks | Normal rows/core | RowStep | Last-core rows |
|---|---:|---:|---:|---:|
| 256 × 1024 | 43 | 6 | 6 | 4 |
| 512 × 1024 | 47 | 11 | 10 | 6 |
| 2048 × 1024 | 48 | 43 | 10 | 27 |

Separate debug runs confirm the same baseline/candidate tiling: key **1000**
without bias, **1002** for broadcast bias, **1001** for elementwise bias.
Both report normal tiling and unchanged block/rowStep values. Profiler rows
report AI_VECTOR_CORE; AIC MAC time is zero. Candidate normal-key ELF function
sizes are 16452/16576/16824 bytes versus baseline 15868/15996/16252 bytes.
Loaded library maps, package paths, source/compiler logs and binary hashes are
retained. Debug logging was disabled for performance captures.

All gates passed:

- Direct ACLNN: nine shape/bias cases. Both FP16 and INT8 outputs are bit-exact
  against unchanged V2 in both candidate processes; repeated baseline is also
  exact. CPU FP32 mathematical references pass the existing tolerances.
- Existing composed TorchAir probe: all four cases pass. All four returned
  tensors per case are bit-exact against eager execution.
- Captured real BGE inputs: embedding and residual norm sites 1, 23 and 47;
  FP16/INT8 candidate outputs are exact against the saved baseline inputs/results.
- Full BGE-M3 eager and compiled embeddings: bit-exact across packages for
  B2/S128, B1/S512, B4/S512, in both orders. Calibration scales are saved once
  and reused; checkpoint file hashes are verified. This comparison isolates the
  cache change from the older V2-versus-regular-W8A8 numerical differences.
- Existing compiled-versus-eager graph differences remain 0.0001220703125 maximum
  for B2/S128 and B1/S512, 0.000244140625 for B4/S512, identically in both packages.

## Paired device profiles

Each lane has two fresh-process captures in **baseline → candidate → candidate
→ baseline** order, ten active forwards per shape/capture, after warmup. The
coverage audit assigns every device kernel to a profiler step. Direct calls
produce exactly one V2 kernel. Model forwards retain **48 V2, 48 Quantize and
144 QuantBatchMatmulV3** kernels. Total kernels per forward are 670/646/670 for
B2/S128, B1/S512 and B4/S512 respectively, unchanged across packages.

The table averages the **47 matched residual V2 sites**, excluding embedding:
940 kernel samples per shape/lane. Times are microseconds per invocation.

| Shape | Duration baseline→cache | Change | Vector baseline→cache | Scalar baseline→cache | MTE2 baseline→cache | MTE3 baseline→cache |
|---|---|---:|---|---|---|---|
| B2/S128 | 15.5550 → 17.1955 | +10.55% | 2.3743 → 2.2638 | 1.7475 → 2.1223 | 4.3976 → 4.0704 | 0.8966 → 0.5180 |
| B1/S512 | 19.2815 → 19.1293 | −0.79% | 4.2593 → 4.0072 | 2.9609 → 2.6049 | 3.1542 → 4.4385 | 2.7796 → 1.9666 |
| B4/S512 | 35.0256 → 35.4249 | +1.14% | 16.1440 → 14.9852 | 6.7849 → 7.6425 | 4.8060 → 5.0434 | 4.6541 → 4.9152 |

Per-capture residual-site duration means expose the variation:

| Shape | Baseline first / last | Candidate first / second |
|---|---|---|
| B2/S128 | 15.5717 / 15.5382 | 17.7171 / 16.6739 |
| B1/S512 | 21.0849 / 17.4781 | 19.2997 / 18.9589 |
| B4/S512 | 35.0932 / 34.9581 | 34.2286 / 36.6211 |

The small B1/B4 pooled duration changes are dominated by capture variation.
B2 regresses in both candidate captures. Do not turn 940 correlated site samples
into 940 independent experiments or infer statistical significance from them.

The direct nine-case control also reduces vector busy time in every case.
For broadcast bias at rows 256/512/2048, durations change by −7.02%/−2.51%/+3.11%;
vector times change 2.6510→2.4850 / 4.6290→4.4874 / 17.6080→16.9039 µs.
For no bias, duration changes are −4.30%/+0.62%/−6.53%. Full direct results,
elementwise cases, both orders, shapes and counter sample counts are in JSON.

## Interpretation and limits

The optimization removes two per-row casts: a regular B4 core goes from 86
parameter casts to two. Measured vector work falls, including **7.18%** at the
47 B4 model sites. It does not remove scalar reductions, reciprocal square root,
per-row control or memory transfers. At B4, scalar busy time increases **12.64%**;
the vector/scalar busy fractions move 51.66%/21.63% → 47.37%/24.10%.

The new prologue explicitly waits for parameter MTE2 copies and performs both
casts before row input processing; the original postpones the casts until the
normalization loop. Extra synchronization, altered overlap, UB placement and
runtime branch/code overhead are plausible costs. The counters identify the
tradeoff, but do not distinguish those causes. No additional optimization was
mixed into this control to try to rescue the result.

Pipeline times overlap and must not be summed into kernel latency. Busy ratios
are not peak-chip utilization, HBM bandwidth, or proof of saturation. No separate
clean end-to-end throughput benchmark was added; Python ctypes timing would be
misleading. Model profiles establish integration and device work, not an e2e
speedup claim. Two opposite-order captures on one 910B2 are a bounded experiment,
not a deployment recommendation.

## Reproduction and retained evidence

Start in `/workspace/repos/bgem3-5bb942ae` inside container
`research_vllm_ascend_023_external_workspace` via the supplied direct SSH host.
Use the committed scripts and a fresh destination/run directory:

```bash
source npu-setup
export PATH=/usr/local/python3.12.13/bin:$PATH
bash 26_bge_m3_inference/gamma_cache/build_910b.sh \
  /workspace/operator_sources/bge-ops-nn-v9.1.0 /workspace/operators/bge-gamma-cache-NEW
bash 26_bge_m3_inference/gamma_cache/run_910b.sh \
  /workspace/results/bge-gamma-cache-NEW \
  /workspace/operators/bge-gamma-cache-NEW/graph/vendors/bge_v2_gc_nn
python3 26_bge_m3_inference/gamma_cache/summarize.py \
  --root /workspace/results/bge-gamma-cache-NEW \
  --output /workspace/results/bge-gamma-cache-NEW/summary.json
bash 26_bge_m3_inference/gamma_cache/collect_dispatch_910b.sh \
  /workspace/results/bge-gamma-cache-NEW \
  /workspace/operators/bge-gamma-cache-NEW/graph/vendors/bge_v2_gc_nn
```

Actual build: `/workspace/operators/bge-gamma-cache-2a9c9702`.
Actual run: `/workspace/results/bge_gamma_cache_af6ba196`.
Direct harness commit `af6ba196`; model harness `bca4ce15`; collector `7ab9208b`
(the final baseline process records this last commit, whose only addition is the
collector). No kernel or validation implementation changed between paired runs.
The initial graph command omitted required `--expected-chip`, exited before
compilation, and was corrected/resumed after checking completed direct results.

[Compact evidence](../../tmp/26_bge_m3_inference/gamma_cache_910b_bca4ce15/)
contains commands, exact validation results, profile audit, dispatch excerpts and
hashes. Raw profiles, output tensors, logs and candidate installer are retained
outside Git, locally and remotely; artifact paths/SHA256 are in `artifacts.json`.
The archive excludes only regenerable TorchAir caches.

Candidate installer SHA256:
`bf58c8fee412d745338d5d67c34c9abf20843554d608e52e876885e6201dfbbf`.
Candidate FP16 kernel SHA256:
`7b698a475078ab23e8bbae468356cfda29e603b1636cb1b308cbcc378da896e9`.
Baseline FP16 kernel SHA256:
`1669e050000f1a9907438d4c80f3c879ccb8009653950133a25e1b6f66a36867`.
