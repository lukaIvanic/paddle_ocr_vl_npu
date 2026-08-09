# IncreFA zero-cube hard-sync result

The fixed FP16 MHA flash-decode key now runs without a cube task. The working
task declaration is `KERNEL_TYPE_MIX_AIV_1_0`, which CANN compiles through the
vector channel and schedules at `taskRation=0:1`. The MIX envelope is required
because this kernel uses `SyncAll()`; it provides `intercoreSync=1` without
launching the cube side.

Bare `KERNEL_TYPE_AIV_ONLY` was not valid for this kernel. Both a 48-block and
a one-block launch failed on Ascend 910B2 at vector PC offset `0x42cc` with the
same UB/D-cache bus error (`507035`). The one-block result ruled out multi-core
partitioning and barrier participant count. The equivalent instruction region
works in the validated mixed AIV function. Installed open-source AIV-only
operators with `SyncAll()` use the same `0:1` hard-sync runtime shape.

## Validation

- Project commit: `881d7d3`
- Package source commit: `a63741c`
- Recovered CANN source commit: `afe72144f9f2ac8441929035795db88a111b30c5`
- Hardware: physical Ascend 910B2 NPU 6, exposed as `npu:0`
- Fixed key: `11000000000100000`
- ELF: one `_mix_aiv` function; no `_mix_aic` function
- Selection proof: removing the object from a disposable custom OPP caused
  error `161002` while statting that exact object. There was no stock fallback.
- Stock comparison: bit-exact at KV128, KV512, and KV2048; maximum absolute
  error `0.0`.

## Non-profiled timing

The comparison used the validated `1:2` mixed package as the control and the
new `0:1` hard-sync package as the candidate. The order was ABBA/BABA. Each
implementation ran in four fresh processes. Each process used 5,000 warmups,
then nine blocks of 2,000 calls for every KV length.

| KV | Mixed median of process medians | Zero-cube median | Candidate/control | Latency reduction |
|---:|---:|---:|---:|---:|
| 128 | 50.4425 us | 49.2568 us | 0.9765 | 2.35% |
| 512 | 51.3615 us | 50.0453 us | 0.9744 | 2.56% |
| 2048 | 50.6150 us | 50.6016 us | 0.9997 | 0.03% |

The result supports a small fixed unused-cube launch cost. It does not show that
cube launch was the main KV2048 bottleneck. See `summary.json` for all retained
process medians and exact ratios.
