# MinerU full-model KV benchmark

This supersedes the isolated eager probe as the performance experiment.
The old `kv_cache_probe/` remains compatibility evidence. Development source
is on `codex/mineru-full-model-kv-benchmark`; 910B validation is in progress.

## Measured workload

The real MinerU2.5-Pro checkpoint reads distinct OmniDocBench crops from an
explicit manifest. Crop categories determine the existing MinerU text/table/
formula prompts. The slow AutoProcessor, min 25088 / max 602112 pixels, complete
vision tower/merger and text prefill produce the actual per-layer KV and first
generated token. Inputs, image hashes, prompt token IDs and model-file identity
are recorded. No random weights, Q/K/V, prompt padding to manufacture context
length, or generated-token replacement is used.

The timed decoder step is one static full TorchAir graph containing token
embedding, all 24 decoder layers, QKV projections, MRoPE, KV writes, attention,
output projection, residuals/norms/MLPs, final norm, real LM head, greedy argmax,
EOS-row masking and per-row position advancement. Each output becomes the next
real input; cache positions and valid lengths advance. A graph rejection is a
failure, with no eager fallback. Vision/text prefill occurs separately because
it is needed once per crop, not on every generated token.

The wall timer encloses the entire generation loop: compiled forwards, sampled
token copies to CPU, host EOS checks and final synchronization. The fixed cohort
drains when all real rows reach EOS or the explicit token cap. Finished slots
still occupy the static graph but contribute zero useful tokens. There is no
hot swapping or speculative generation in this benchmark. Short rows therefore
create visible tail overhead; raw batch token slots are recorded separately.

Primary metric: **useful post-prefill non-EOS output tokens / synchronized
decode-generation wall seconds**. The first token comes from prefill and is
excluded from that numerator. The EOS-producing forward stays in the denominator
but EOS is not a useful token. Completion, per-item token counts and length-cap
hits are reported. This metric is not page throughput or recognition throughput
including prefill. Prefill latency is recorded separately.

Compilation, first compiled call, warmup, input preparation, cache conversion/
reset and correctness generation are outside the timed interval. The checkpoint
and real-prefill bank are shared across variants. Every measurement resets KV
to that bank before the timer. Validated variants alternate forward/reverse
order on successive repeats (AB/BA), with per-trial results and min/median/max
throughput; small differences require inspecting that variation and repeat order.
Do not use a single isolated-kernel median to claim a model speedup.
Each variant has a distinct Python forward code object as well as its own GE
cache directory. Timed generation records Dynamo's new-graph counter and
TorchAir recompilation warnings; either occurring invalidates the measurement.

## Correctness and format gates

The existing production flat decoder runs raw eager from the identical real
prefill to establish each crop's greedy sequence through EOS or the cap. It is
a fidelity control, not a throughput competitor. Every compiled variant must
match that complete sequence before timing, and each timed repeat is checked
again. Token-accounting unit tests separately verify that EOS and inactive
slots cannot inflate the primary metric.

Each K/V allocation records its logical shape and actual storage descriptor.
Prefill conversion must preserve every logical value. Requested NZ must really
be descriptor 29, including after timed generation. Native controls allow
descriptor 0 or 2 as observed on this runtime. Errors, wrong tokens or descriptor
changes invalidate timings for that variant. A compiler may insert internal
TransData or choose a different internal layout; retained external NZ alone
does not prove that attention consumed NZ inside the graph. Inspect the full
model profiler before attributing a speed difference to attention's NZ reads.

| Variant | Attention | K/V logical contract | Requested storage |
| --- | --- | --- | --- |
| `increfa_nd` | Production IncreFA + PSE-sentinel mask | Dense BNSD | Native |
| `increfa_nz` | Same IncreFA | Same dense BNSD | NZ 29 |
| `fia_nd` | Compiled tensor-length FIA | `[NB, block, Hkv, D]`, read as `[NB, block, Hkv*D]` | Native |
| `fia_blocked_nd` | Compiled tensor-length FIA | `[NB, Hkv*D/16, block, 16]`, read as rank 5 | Native |
| `fia_nz` | Same blocked FIA | Same blocked layout | NZ 29 |
| `paged_nd` | Private ATB writer + paged reader | Ordinary 910B paged shape | Native |
| `paged_nz` | Same private APIs | Audited 310P blocked shape | NZ 29 |

FIA gets changing NPU int64 sequence-length tensors through the installed
`torchair.ops` tensor-length interface. Its cache writer is scatter-ND inside
the graph, with metadata shared by all layers. Private paged variants instead
request the actual `_npu_reshape_and_cache` / `_npu_paged_attention` APIs and the
audited NPU-length call contract. Installed TorchAir may have no lowering for
these private ATB APIs; record that failure. Never describe FIA as the private
vLLM decode operator. CPU-only lengths for the private reader are deliberately
not frozen into a graph: that would stop matching advancing real generation.

Only dense IncreFA ND/NZ and blocked FIA ND/NZ are same-contract format pairs.
Comparing dense IncreFA with paged FIA also changes attention and cache-update
implementations. Comparing `fia_nd` with `fia_nz` also changes logical layout.
Those comparisons must be labeled as complete decoder-path comparisons.

## 910B validation ladder

Use the existing host master and `docker exec` route from `CLAUDE.md`, not the
retired direct-container SSH alias. Source arrives by local commit/push and
remote fetch/checkout; never edit tracked files on the validation container.
Verify tracked source is clean before checkout. Use the existing installed
environment, a healthy free card and a fresh per-run compile-cache directory.

From the fetched container checkout:

```bash
source npu-setup
export MODEL=/workspace/models/MinerU2.5-Pro-2605-1.2B
export PYTHON=/usr/local/python3.12.13/bin/python3
CHIP=910B MANIFEST="$PWD/crops/manifest.json" LIMIT=2 BATCH_SIZE=1 \
  MAX_NEW_TOKENS=256 REPEATS=2 VARIANTS=increfa_nd PROFILE=1 \
  RUN_NAME=full_model_control bash 11_mineru_2_5_pro_inference/run_full_model_kv.sh
```

After the full-model control passes, use real mixed crops and larger cohorts:

```bash
CHIP=910B LIMIT=8 BATCH_SIZE=4 MAX_NEW_TOKENS=512 REPEATS=5 PROFILE=1 \
  VARIANTS=increfa_nd,fia_nd,fia_blocked_nd \
  RUN_NAME=full_model_native bash 11_mineru_2_5_pro_inference/run_full_model_kv.sh
```

Then request the actual NZ descriptors, preserving rejected variants:

```bash
CHIP=910B LIMIT=8 BATCH_SIZE=4 MAX_NEW_TOKENS=512 REPEATS=5 \
  VARIANTS=increfa_nd,increfa_nz,fia_blocked_nd,fia_nz \
  RUN_NAME=full_model_formats bash 11_mineru_2_5_pro_inference/run_full_model_kv.sh
```

`PROFILE=1` profiles eight advancing complete decoder forwards outside the
throughput window. Inspect the exported operator/kernel traces to verify all
24 layers, LM head and the selected attention path ran. Exit 2 preserves any
unsupported or incorrect variant; it must not be mistaken for all-pass validation.
The run directory contains command, commit, exit code, occupancy snapshots,
summary, log and optional profiler trace. Timeout defaults to 1800 seconds;
stop all benchmark progression after a timeout or device error.

## 310P transfer gate

Do not reuse the isolated probe's handoff to claim this benchmark ran. First
validate the full-model control, EOS accounting, generated sequences and
compiled forward on 910B, then publish a handoff tied to those results. The
310P environment must be rediscovered: interpreter, model path, CANN/ATB,
torch-npu, TorchAir tensor-length FIA support and operator lowering may differ.
No packages should be installed or tracked source edited on that work server.

An unsupported NZ contract on 910B is not 310P validation. A full-graph compiler
rejection on either chip is a compiler/API compatibility result, not a kernel
speed measurement. Do not freeze lengths, fall back to eager, or substitute
native storage to make a requested NZ graph appear to pass.
