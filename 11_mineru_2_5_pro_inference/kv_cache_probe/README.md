# MinerU KV-cache layout and storage-format probe

Status, 2026-10-06: implemented; CPU packing checks passed on the Mac. **No NPU
timing or correctness result yet.** The blue-zone gateway is reachable but its
container forward to `127.0.0.1:22021` currently returns connection refused.

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

There are two distinct questions: **logical element order** and **storage
descriptor**. A blocked tensor with descriptor 2 is not evidence that format 29
was exercised. The probe queries the actual NPU format and records it before
and after attention. If the runtime does not retain the requested descriptor,
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

Tracked source must arrive through a commit/push and `git pull`; never edit
tracked files in the container. Once the SSH route is restored, run from the
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

Run only after the 910B ladder has been reviewed. The work-server agent must
pull source and report; it must not edit tracked files, commit, push, or create
branches. Resolve the checkout rather than hardcoding the work-server path:

```bash
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git status --short
git fetch origin codex/mineru-kv-cache-layout-probe
git checkout --detach FETCH_HEAD
git rev-parse HEAD
```

Use the existing working CANN/ATB/torch-npu environment. Select an available
310P according to the server's existing device policy, and set `PYTHON` to its
working torch-npu interpreter. There is no CPU fallback and no BF16 path here.
Do not install packages or modify vLLM. First run only the portable operations:

```bash
CHIP=310P OPERATORS=increfa,fia RUN_NAME=portable \
  BATCHES=1,16 CONTEXTS=768 \
  bash 11_mineru_2_5_pro_inference/kv_cache_probe/run_probe.sh
```

After reviewing statuses/correctness, run `OPERATORS=paged RUN_NAME=paged` on
the same B1/B16 S768 matrix. Preserve unsupported variants instead of changing
their contract until they run. For the private operation, the important case is
`blocked4`, format 29, `fill=writer`; its controls are the same shape/format with
synthetic filling, and the same shape with descriptor 2. If a control rejects
ND on 310P, an ND-versus-NZ speed ratio for that operation is unavailable.

Only if the controls pass, expand `CONTEXTS=768,1408,2816,4096` and
`PATTERNS=uniform,ragged`. If a change is needed, report the failing command,
case JSON/log and minimal proposed source change to Luka; do not apply it.

Return the entire run directory and these fields: commit, hostname, observed
device name, CANN home, torch and torch-npu versions, exact command, every case
status, requested/observed formats, correctness errors, wall/device medians,
and `FORMAT_COMPARISONS`. Explain any writer-logical-roundtrip mismatch
separately from attention correctness. No page/s or end-to-end performance
claim can be derived from this probe.
