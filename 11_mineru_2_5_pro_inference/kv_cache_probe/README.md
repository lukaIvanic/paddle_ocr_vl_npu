# MinerU KV-cache layout and storage-format probe

Status, 2026-10-06: CPU packing checks and **910B2 FP16 portable-operation
correctness checks passed**. Genuine format-29 storage was retained and checked,
but IncreFA and FIA rejected it on this installed runtime. The complete private
operation matrix passes native-cache controls with CPU lengths; its NZ variants
fail contract or output validation. **No 310P result or validated ND/NZ speed
ratio exists.**
Use the verified host-master plus `docker exec` route. Cards 4 and 5 are excluded;
Luka authorized MinerU on 7 alongside an idle Clef server. Clef startup is
currently blocked by the other Qwen service's memory use; see the evidence below.
Recheck inventory before every run. See [910B evidence](#910b-evidence).

This probes one-token **text decode attention** with synthetic, identical FP16
Q/K/V and MinerU dimensions: 14 query heads, 2 KV heads, D64, KV4096. Vision
attention has no persistent autoregressive KV cache. This is neither a full
decoder benchmark nor an end-to-end page benchmark.

## Source contract

Audit pinned to vLLM-Ascend commit
`80610e4438dba05011b05f89fc45d91e96992671` (the investigated 0.21.0rc1 source).
Do not silently transfer this conclusion to another release. The allocation
trace corrects the earlier suggestion that the 310P blocked shape alone might
describe its cache: this runner explicitly allocates format **29** as well.

| Part | Verified pinned implementation |
| --- | --- |
| 310P logical K/V shape | Each cache is `[num_blocks, (Hkv * D)/16, block_size, 16]`; for MinerU and block128: `[num_blocks, 8, 128, 16]` |
| Storage descriptor | `torch_npu.empty_with_format(..., acl_format=29)`; `ACL_FORMAT_FRACTAL_NZ=29` |
| Element dtype | `kv_cache_spec.dtype`; this experiment fixes FP16, with no KV quantization |
| Kernel block sizes | Backend lists 128, 64; runner selects the first satisfying `block_size * D <= 128 * 128`, and can split allocator pages into kernel pages |
| Cache write | `DeviceOperator.reshape_and_cache` resolves to `BaseDeviceAdaptor` on 310P, then `torch_npu._npu_reshape_and_cache(..., slot_indices=slot_mapping)` |
| DecodeOnly attention | `torch_npu._npu_paged_attention(..., num_heads=14, num_kv_heads=2, scale_value=1/sqrt(64), block_table=..., context_lens=..., out=...)` |
| Query contract | `[num_tokens, num_heads, head_size]`; one token per request in DecodeOnly |
| Other text phases | Uncached prefill uses `_npu_flash_attention` / `_npu_flash_attention_v3`; cached/mixed prefill uses splitfuse variants. This probe does not measure them. |

Primary source links:

- [310P allocation, lines 95 and 705–784](https://github.com/vllm-project/vllm-ascend/blob/80610e4438dba05011b05f89fc45d91e96992671/vllm_ascend/_310p/model_runner_310p.py#L705)
- [310P shape, block sizes, decode and dispatch](https://github.com/vllm-project/vllm-ascend/blob/80610e4438dba05011b05f89fc45d91e96992671/vllm_ascend/_310p/attention/attention_v1.py#L166)
- [Cache writer and device selection](https://github.com/vllm-project/vllm-ascend/blob/80610e4438dba05011b05f89fc45d91e96992671/vllm_ascend/device/device_op.py#L43)
- [Format constants](https://github.com/vllm-project/vllm-ascend/blob/80610e4438dba05011b05f89fc45d91e96992671/vllm_ascend/utils.py#L54)
- [Common attention query contract](https://github.com/vllm-project/vllm-ascend/blob/80610e4438dba05011b05f89fc45d91e96992671/vllm_ascend/attention/attention_v1.py#L1279)

### The private operation has a chip-specific cache contract

Huawei's [PagedAttention input/output table, CANN 9.0.0-beta.2](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/API/ascendtbapi/ascendtb_01_0197.html)
and [CANN 9.1 basic-function table](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/acce/ascendtb/ascendtb_01_0202.html)
distinguish Atlas A2/A3 products from Atlas inference products. For this probe's
FP16, Hkv2/D64/block128 configuration, the documented contracts specialize to:

| Product family / tested chip | K/V logical shape | Storage |
| --- | --- | --- |
| Atlas A2 / our 910B2 | `[NB, 128, 2, 64]` | ND |
| Atlas inference / target 310P | `[NB, 8, 128, 16]` | NZ |

Both tables put `contextLens` on CPU as int32. These are published contracts,
not documentation retrieved for the exact installed ATB 9.0.0.B160 build. The
910B results nevertheless agree with the documented family distinction: native
ordinary pages pass, whereas the blocked NZ reader fails its head-size check.
The same Python entry point does not select the same supported cache contract
on both chips. No supported ND/NZ comparison on the 910B private operation is
established, and a 310P ND control may also reject by design.

The installed torch-npu reports git revision
`94f8a8e6b523d7ba553e1b80d5b5248478391526`; its `third_party/op-plugin` gitlink
is `dedc316708372c9a8bfde4527abd2a94b74840f5`.
That revision's [PagedAttention wrapper](https://github.com/Ascend/op-plugin/blob/dedc316708372c9a8bfde4527abd2a94b74840f5/op_plugin/ops/atb/PagedAttentionAtb.cpp)
creates `atb::infer::PagedAttentionParam` and calls ATB
`PagedAttentionOperation`. K/V are passed without format conversion.
The [tensor bridge](https://github.com/Ascend/op-plugin/blob/dedc316708372c9a8bfde4527abd2a94b74840f5/op_plugin/utils/custom_functions/atb/AtbCommon.cpp)
copies logical tensor dimensions and selects host/device pointers from the
tensor's device; its [format helper](https://github.com/Ascend/op-plugin/blob/dedc316708372c9a8bfde4527abd2a94b74840f5/op_plugin/utils/custom_functions/atb/Utils.cpp)
normalizes base formats to ND but preserves format 29. The lengths conversion
flag changes storage format, not device placement. There is no ordinary
cache-NZ override in the installed `PagedAttentionParam` header.

Thus this probe really reaches the ATB private reader with genuine NZ; the
reader failure is consistent with its 910B contract, rather than proof of
broken 310P NZ attention. Installed build metadata and pinned source hashes
are preserved in the [contract audit](../../tmp/11_mineru_2_5_pro_inference/atb_contract_audit_20261006T101234Z_64668826/audit.json).

Our current `LocalMinerUStaticCache.allocate` in `local_modeling_mineru.py`
allocates dense BNSD `[B, 2, capacity, 64]` tensors with no explicit NZ cache
conversion. `attend_static_decode` calls IncreFA with a future-slot bool mask,
`actual_seq_lengths=None`, and the production PSE sentinel. Decoder **weights**
in NZ are a separate matter. No controlled MinerU KV-ND-versus-KV-NZ timing was
recovered from the reviewed history; earlier Paddle probes do not establish a
MinerU result.

## Experiment matrix

`probe_attention.py` imports only stdlib in its controller. Workers import only
`torch` and `torch_npu`; there are no model, repository, vLLM or TorchAir imports.
Calling the same torch-npu entry point as vLLM exercises its operation without
starting or modifying vLLM. It does not reproduce the engine scheduler or graphs.

| Operator | Logical layouts, each with descriptor 2 and 29 |
| --- | --- |
| IncreFA | Dense BNSD, current production mask/PSE semantics |
| FIA v1 and v2 | ND pages `[NB, block, Hkv*D]`; blocked pages `[NB, Hkv, D/16, block, 16]`. One-token BSH query avoids assuming NZ BNSD support. |
| `_npu_paged_attention` | 910B-style pages `[NB, block, Hkv, D]`; 310P blocked pages `[NB, Hkv*D/16, block, 16]`. Each also tests synthetic packing and actual `_npu_reshape_and_cache` filling separately. |

The default private-op call keeps `context_lens` on NPU, exactly as the pinned
310P source does. The installed 910B ATB path rejected that metadata with
`tensor.hostData is null`, before attention. `PAGED_LENGTH_DEVICE=cpu` selects a
separately labeled control using CPU int32 lengths with the same NPU query,
K/V, block table and private operation. This is not CPU attention or an
automatic fallback. [Huawei's official 910B test uses CPU lengths](https://github.com/Ascend/op-plugin/blob/d83570a35dfe0d8e9869c3ecfca6647cfccdd9c8/test/test_custom_ops/test_atb_paged_attention.py#L120).
Until this variant runs, those ATB setup errors establish no cache-format
support or performance conclusion.

There are two distinct questions: **logical element order** and **storage
descriptor**. A blocked tensor with descriptor 2 is not evidence that format 29
was exercised. The probe queries the actual NPU format and records it before
and after attention. Ordinary rank-4 ND allocations can normalize to native
NCHW descriptor 0 on torch-npu; this is accepted as the contiguous native control
and still requires a bit-exact KV roundtrip. Requested NZ must remain descriptor
29. If the runtime does not retain an acceptable descriptor,
the case is `format_unavailable` and has no timing.

The FIA five-dimensional blocked contract and the private operation's
four-dimensional format-29 contract are separate cases. Reshaping an NZ tensor
can alter how a backend interprets its storage; the probe does not assume these
contracts are interchangeable. A 910B rejection of the 310P contract is useful
compatibility evidence, not a 310P speed result.

All cases use the same seeded, FP16-rounded Q/K/V for a batch/context/pattern;
the reference computes attention in CPU FP32 from those rounded inputs. The
block table shuffles physical pages; unused cache positions contain poisoned
values. Ragged batches include a full-length row, adjacent length, shorter row,
and length-one row. Packing is checked bit-exactly across NPU format conversion.
The actual cache-writer cases start with separate native allocations and write
all slots through the private writer. Their logical CPU roundtrip is reported
separately because private kernels may address NZ physical storage directly;
attention output must still match the independent reference.

Only passing cases are timed: eager dispatch, warmed operation, NPU events and
synchronized wall time. Allocation, packing, format conversion, bulk cache
writing, and reference validation are outside the attention window. This does
not measure per-token write overhead or compiled decode speed. Report them as
follow-up questions if attention-only results justify further investigation.

Each case has a fresh subprocess, log, JSON and timeout. Rejected contracts are
`operation_error`; hangs are `timeout`; wrong values are `validation_failed`.
Errors are not automatically classified as unsupported by the chip: inspect
their logs. The matrix continues after rejected contracts or wrong values but
stops after a timeout, since terminating a worker might not stop a device-side
hang. It exits 2 for any wrong values and 1 for a timeout or if no case passes.
Exit 0 can still contain rejected cases; review
every status. `same_layout_format_comparisons` includes only validated pairs
with the same operator, logical shape, fill method and fixture hashes.

## 910B run ladder

Tracked source must arrive through a commit/push and source fetch; never edit
tracked files in the container. Use the existing host master via
`bash scripts/blue_zone_exec.sh bash --noprofile --norc -c '<commands>'`, or
pipe a script to `bash scripts/blue_zone_exec.sh bash --noprofile --norc -s`.
Run the following from the
container checkout. Check for tracked changes first and stop if any are present;
never discard existing work. The published probe branch is
`codex/mineru-kv-cache-layout-probe`; select it without creating a work-server
branch:

```bash
cd /workspace/repos/paddle_ocr_vl_npu
git status --short
git fetch origin codex/mineru-kv-cache-layout-probe
git checkout --detach FETCH_HEAD
source npu-setup
export PYTHON=/usr/local/python3.12.13/bin/python3
# Confirm the selected physical card is healthy (OK) and unoccupied first.
CHIP=910B bash 11_mineru_2_5_pro_inference/kv_cache_probe/run_probe.sh
```

Start with the portable operations. Inspect all statuses and correctness before
advancing. Missing FIA v2 on an older runtime is an API availability result;
keep the FIA v1 / IncreFA results. Do not upgrade the environment for this probe.
Then exercise the actual private decode and cache-write entry points:

```bash
CHIP=910B OPERATORS=paged RUN_NAME=paged \
  bash 11_mineru_2_5_pro_inference/kv_cache_probe/run_probe.sh
```

On this installed 910B ATB runtime, run the explicit CPU-length control after
reviewing the default metadata failure:

```bash
CHIP=910B OPERATORS=paged PAGED_LENGTH_DEVICE=cpu RUN_NAME=paged_cpu_lengths \
  bash 11_mineru_2_5_pro_inference/kv_cache_probe/run_probe.sh
```

`FORMATS=2` runs native-storage controls alone when repeating timing after an
occupancy change; the default remains `2,29`. On this shared machine, store
before/after occupancy snapshots and exclude timings that overlap another
workload. The event span includes eager host dispatch and is not pure kernel
time.

If the controls pass, expand the relevant operator's contexts and batches:

```bash
CHIP=910B OPERATORS=increfa,fia RUN_NAME=boundary \
  BATCHES=1,16,32 CONTEXTS=768,1408,2816,4096 PATTERNS=uniform,ragged \
  bash 11_mineru_2_5_pro_inference/kv_cache_probe/run_probe.sh
```

The mask/PSE sentinel preserves the existing workaround at effective lengths
1408 and 2816. A timed-out child is terminated, but a stuck device-side operation
can outlive the host process. Stop the ladder if driver/device errors appear;
do not reset a shared device or kill other users' processes.

The wrapper creates `command.txt`, `exit_code.txt`, `run.log`, `summary.json`,
and per-case JSON/logs under `tmp/11_mineru_2_5_pro_inference/`. Preserve the whole
directory. Local CPU checks use an existing torch installation:

```bash
python3 11_mineru_2_5_pro_inference/kv_cache_probe/probe_attention.py --cpu-self-test
```

## Self-contained 310P handoff

Give the receiving agent [HANDOFF_310P.md](HANDOFF_310P.md) and access to the
published code. That file contains the objective, prior findings, pull-only
constraints, environment discovery and preflight, exact commands, compatibility
branches, stop conditions, correctness/timing gates, and return-report format.
It assumes no conversation history or matching 910B installation. No model
weights, datasets, vLLM engine startup, or TorchAir are required.

## 910B evidence

Environment: physical NPU 7, observed `Ascend910B2`, torch 2.10.0+cpu,
torch-npu 2.10.0, CANN 9.0; Hq14/Hkv2/D64, FP16, block128, capacity4096,
S768, B1/B16 with ragged lengths. Results are one-layer synthetic checks.

- [Portable matrix, source 09e2e2c2](../../tmp/11_mineru_2_5_pro_inference/kv_cache_910B_portable_20261006T092209Z_09e2e2c2/summary.json):
  all 20 cases completed. Eight FIA v1/v2 native-storage cases passed, including
  both ordinary and logical blocked pages; maximum absolute output error was
  0.000131. All ten actual format-29 cases rejected the descriptor: IncreFA
  reports `ERR00007`, FIA v1/v2 CANN reports unsupported dtype/format (161002).
  Two native IncreFA controls were incorrectly skipped because descriptor 2
  normalized to native descriptor 0; this guard was fixed in faa06be5.
  A new Qwen TP4 service was observed on cards 4–7 during this period, so these
  timing samples are **not an isolated benchmark** and need repetition.
- [IncreFA controls, source 5f982e04](../../tmp/11_mineru_2_5_pro_inference/kv_cache_910B_increfa_native_control_20261006T093249Z_5f982e04/summary.json):
  B1/B16 native controls passed, maximum errors 0.0000683 / 0.000131;
  format 29 again rejected. Recorded event medians 0.065 / 0.075 ms include eager
  dispatch; occupancy changed repeatedly on the shared host, so do not use these
  as clean cross-operator speed comparisons.
- [Interrupted private matrix, source 5f982e04](../../tmp/11_mineru_2_5_pro_inference/kv_cache_910B_paged_20261006T093401Z_5f982e04/summary.json):
  exit 143, stopped our controller/worker to release card 7 for Clef.
  The controller recorded two B1 ordinary-page native-storage setup failures;
  a third format-29 packed case also has a worker JSON. ATB logs report missing
  host metadata, not a cache-format rejection. The native writer control had a
  bit-exact logical roundtrip. The subsequent CPU-length matrix is below.
- [Complete private CPU-length matrix, source c3bbae98](../../tmp/11_mineru_2_5_pro_inference/kv_cache_910B_paged_cpu_lengths_20261006T094514Z_c3bbae98/summary.json):
  all 16 cases completed; exit 2 correctly flags two wrong-output variants.
  Four native ordinary-page controls passed (B1/B16, synthetic and private-writer
  filling). Maximum absolute errors were 0.0000683 / 0.000131; recorded event
  medians were 0.063–0.072 ms. Both ordinary-page NZ synthetic cases produced
  wrong attention despite bit-exact KV roundtrips (max errors 13.13 / 15.28), so
  they have no timing. Ten other variants errored. Blocked-page attention,
  including writer-filled NZ, reports `headSize of keyCache and query should be
  same`; the NZ blocked writer itself passed a bit-exact logical roundtrip.
  [Occupancy snapshots](../../tmp/11_mineru_2_5_pro_inference/occupancy_910B_paged_cpu_20261006T094513Z_c3bbae98/before.txt)
  show the Qwen service co-resident on card 7; timings are not isolated kernel
  benchmarks. No validated ND/NZ pair exists.
- [NZ descriptor audit, source 2f466ee5](../../tmp/11_mineru_2_5_pro_inference/kv_cache_910B_nz_descriptor_audit_20261006T095127Z_2f466ee5/summary.json):
  four B1 NZ cases; exit 2. The blocked writer retained descriptor 29 after
  writing and passed bit-exact logical KV roundtrip, then reader setup failed.
  Ordinary-page synthetic NZ retained descriptor 29 before and after attention
  but returned the same wrong values. This confirms genuine NZ storage reached
  attention; no native-storage substitute or writer-induced descriptor change
  explains the result. Failed or incorrect cases were never timed.
- [Initial aborted metadata run, source 264ce6e0](../../tmp/11_mineru_2_5_pro_inference/kv_cache_910B_portable_20261006T092003Z_264ce6e0/summary.json):
  exit 143; five cases failed an unsupported configuration getter before any
  attention call. Fixed in 09e2e2c2. This is not inference validation.

Every linked directory preserves the exact command, exit code, run log and
per-case JSON/logs. No validated same-shape ND/NZ pair was timed, and no NZ
speedup has been demonstrated. The 910B compatibility ladder has established
native controls and concrete NZ failures; further support and speed conclusions
require the supplied 310P handoff on that actual chip.

Clef was stopped initially as authorized. Its requested card-7 restart was
attempted with the existing vLLM-Ascend Clef worker and real BF16 checkpoint.
The ordinary 0.7 memory budget rejected the 8.78 GiB available; a 0.10 budget
passed that gate but model loading ran out of NPU memory. The checkpoint is
17.75 GiB on disk (including skipped visual tensors); the actual allocation
failure proves the available space is insufficient. **Clef is not serving.**
The other service in `zjy-gpqa-back4` was not stopped. No reservation or working
Clef server is claimed from these failed launch attempts.
The [0.7-budget attempt](../../tmp/11_mineru_2_5_pro_inference/clef_idle_card7_verified_env_20261006T093939Z/run.log)
and [0.10-budget allocation failure](../../tmp/11_mineru_2_5_pro_inference/clef_idle_card7_small_budget_20261006T094202Z/run.log)
preserve their actual launch commands as adjacent `command.json` files.
