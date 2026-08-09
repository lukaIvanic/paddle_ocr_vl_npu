# Custom IncreFA FP16/MHA reproduction on Ascend 910B2

## Result

The fixed-key custom AscendC package reproduces the stock CANN 9.0.0
`IncreFlashAttention` result for the experiment-09 decode contract.

- Validation lane: Blue-zone Ascend 910B2 container
- Physical device: Ascend 910B2 NPU 6, exposed as logical `npu:0`
- Project commit used for the final aggregate: `764e35429b93e26bd3760afb4e8c4a982dc773a0`
- Recovered upstream operator source commit: `afe72144f9f2ac8441929035795db88a111b30c5`
- Operator contract: B1, FP16, BNSD, 16 query heads, 16 stored KV heads,
  head dimension 128, bool attention mask, no actual-sequence-length tensor,
  `num_key_value_heads=0`, `inner_precise=1`
- Fixed tiling key: `11000000000100000`
- Correctness: bit-exact for all saved output elements at KV128, KV512, and
  KV2048 in all eight timed processes

## Non-profiled latency

Each fresh process used 5,000 warm-up calls per KV length followed by nine
blocks of 2,000 calls. Four stock and four custom processes were retained. The
aggregate is the median of the four process-level block medians.

| KV | stock NPU event (us) | custom NPU event (us) | custom / stock | stock host wall (us) | custom host wall (us) | custom / stock |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 49.930 | 50.228 | 1.0060 | 50.013 | 50.303 | 1.0058 |
| 512 | 50.579 | 50.476 | 0.9980 | 50.650 | 50.551 | 0.9981 |
| 2048 | 50.721 | 50.730 | 1.0002 | 50.800 | 50.801 | 1.0000 |

This meets the reproduction performance gate: aggregate custom latency is
within 0.6% of stock for all three target KV lengths, and device-event and
host-wall measurements agree.

## Variance disclosure

No process was removed. `custom_a` entered a slower state at KV512 and KV2048:
its process medians were 63.895 us and 63.877 us. The other custom process
medians were 48.818-51.811 us at KV512 and 49.152-52.085 us at KV2048. Some
stock processes also varied, although less severely. NPU telemetry stayed at
101.6-102.7 W and 48-50 C; the available `npu-smi` fields did not expose a
clock value, so the cause is not proven. The robust aggregate retains this
process and does not present the best run as the baseline.

## Runtime-selection proof

The installed custom object and stock built-in object have different SHA-256
hashes. In a disposable copy of the custom package, removing only the selected
custom object made the call fail with CANN error 161002. The error names the
missing object under the disposable custom package path. This proves that the
custom environment selected the custom package and did not silently fall back
to the built-in object.

`result.json` contains the complete aggregate and all per-process block
distributions. `run.log` is its concise report. The bulky raw tensors and
unsanitized machine logs remain local-only and are not public-repo artifacts.
`command.txt` is the command authority.

## Decision

The FP16/MHA baseline reproduction gate is complete. An AIV-only variant is now
a justified separate experiment. It must retain this custom package as the
correctness and non-profiled performance control.
