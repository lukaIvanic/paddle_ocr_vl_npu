# MinerU full-model KV benchmark: 310P handoff

**Decode-only investigation. This is not the handoff for the active vision
bottleneck.** See [VISION_310P_TARGET.md](VISION_310P_TARGET.md).

This is a complete brief for an agent with only this repository and no chat
history. The authoring branch is `codex/mineru-full-model-kv-benchmark`.
Read `FULL_MODEL_KV_BENCHMARK.md` for measurement and variant contracts.
The old `kv_cache_probe/probe_attention.py` measured synthetic isolated eager
attention; it cannot establish full-model speed. This task replaces that
performance experiment with real crop generation.

## Objective

Compare cache formats inside the actual FP16 MinerU2.5-Pro decoder on 310P,
using real OmniDocBench text/code/formula/table crops and complete static
TorchAir decoder graphs. Measure useful post-prefill tokens per synchronized
wall second, including KV writes, all 24 decoder layers, LM head, sampling,
CPU token copies and EOS completion handling. Vision and text prefill are real
and separately timed. This is decode throughput, not end-to-end page throughput.

Decoder weights are NZ in every lane. Changing cache storage is a separate
experiment. Dense IncreFA ND/NZ and blocked FIA ND/NZ are same-contract storage
pairs. Comparing dense IncreFA with paged FIA changes both attention and cache
writer/layout; label it as a complete decoder-path comparison.
The private `_npu_reshape_and_cache` / `_npu_paged_attention` variants request
the actual audited vLLM-Ascend APIs. FIA is a different operator. If installed
TorchAir cannot compile the private APIs, record the rejection; do not silently
substitute FIA or an eager call and call it a compiled private-op result.

## Constraints and environment discovery

- Source is pull-only: no tracked edits, commits, pushes, branches, installs,
  shared configuration changes or modifications to vLLM. Runtime files may be
  written under ignored `tmp/`. Report minimal proposed fixes and failing logs.
- Discover this server's existing environment. Do not assume the 910B CANN 9.0,
  torch-npu 2.10, Python path, `npu-setup`, model path or SSH route exists here.
- Use an existing FP16 MinerU2.5-Pro-2605-1.2B checkpoint. No BF16, downloads,
  random weights, synthetic KV, artificial context or CPU inference fallback.
- Select exactly one healthy, unoccupied 310P using the local procedure. Do not
  kill services, reset devices or borrow the 910B sharing authorization.
- Record Python, torch, torch-npu, Transformers, TorchAir, CANN/ATB, device name
  and physical visibility. Identify working torch-npu/TorchAir interpreter and
  existing setup scripts before inference. Missing support is a blocker, not
  authorization to install or upgrade.
- Stop after a timeout or device error and preserve evidence. Operator/compiler
  contract rejection with a healthy device is a compatibility result.

## Obtain the published source

From the existing checkout:

```bash
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git status --short
```

Stop if tracked source is dirty; do not stash or overwrite it. Otherwise:

```bash
git fetch origin codex/mineru-full-model-kv-benchmark
git checkout --detach FETCH_HEAD
git rev-parse HEAD
git merge-base --is-ancestor 97d4e7e9 HEAD
```

Require the ancestor check to exit zero. Record the actual fetched commit.
The repository is private; use existing authorized GitHub credentials. Report
access failure without changing remotes or visibility.

## Preflight and control

Activate the existing working environment; export `PYTHON` as its absolute
interpreter path, `MODEL` as the existing checkpoint directory, and
`ASCEND_RT_VISIBLE_DEVICES` as one selected physical card. Logical `npu:0` is
used by the worker. Preserve `npu-smi info` and the selected runtime versions.
The benchmark imports torch/torch-npu, the repository's owned MinerU model and
helpers, PIL and Transformers for processing; no vLLM import is required.
TorchAir can be top-level or bundled with torch-npu, as locally installed.

Run the token-accounting checks first (these require only stdlib):

```bash
"$PYTHON" -m unittest discover -s 11_mineru_2_5_pro_inference \
  -p test_full_model_kv_metrics.py
```

Then the complete B1 native control:

```bash
CROP_IDS= CHIP=310P MANIFEST="$WORK_SERVER_REPO/crops/manifest.json" LIMIT=2 \
  BATCH_SIZE=1 MAX_NEW_TOKENS=256 REPEATS=2 VARIANTS=increfa_nd PROFILE=1 \
  RUN_NAME=full_model_control bash 11_mineru_2_5_pro_inference/run_full_model_kv.sh
```

Require the actual device to match 310P, fullgraph static TorchAir compile to
pass, complete generated sequences to match the production raw-eager reference,
all timed samples to record zero new graphs/recompilation warnings, and native
cache descriptors to be retained. The reference is generation parity, not an
independent OCR accuracy evaluation. Stop benchmark progression if this control
fails. Read `error` and per-variant traceback; do not route around them.

## Real mixed B4 cohort and format comparisons

Use exactly these distinct repository crops; the explicit list avoids the
manifest prefix, which contains only code/formula:

```bash
export CROP_IDS=hotswap_001_code_txt_p0001_box_id_3,hotswap_002_code_txt_p1474_11,hotswap_003_equation_isolated_p0000_box_id_1,hotswap_004_equation_isolated_p0036_box_id_9,hotswap_038_table_p0010_box_id_1,hotswap_039_table_p0243_box_id_1,hotswap_057_text_block_p0000_box_id_0,hotswap_058_text_block_p0062_box_id_8
CHIP=310P LIMIT=8 BATCH_SIZE=4 MAX_NEW_TOKENS=1024 REPEATS=5 PROFILE=1 \
  VARIANTS=increfa_nd,fia_nd,fia_blocked_nd,increfa_nz,fia_nz,paged_nd,paged_nz \
  RUN_NAME=full_model_mixed bash 11_mineru_2_5_pro_inference/run_full_model_kv.sh
```

Every batch uses the real full vision/text prefill and identical initial KV for
all lanes. Each sampled output becomes the next input; cache positions and
lengths advance. No fixed synthetic QKV, frozen context lengths, post-EOS fill
or inferred full-model speed from isolated attention is allowed. The processor
cap is 602112 pixels / 3072 raw vision tokens; the script verifies actual grids.
Cache capacity is 4096; block size 128. Report length-cap hits explicitly.

Exit 2 means at least one requested variant failed; inspect the summary rather
than discarding passing native controls or claiming everything passed. Descriptor
29 must genuinely survive for an NZ lane, and logical cache conversion must
preserve values. Compilation inside the timer invalidates timings. Candidate
token mismatches do not suppress timings: report normalized token edit distance,
common prefix,
decoded text, output lengths, EOS/cap counts and repeat stability alongside tok/s.
Decide whether drift is material from the actual outputs. Substantially changed
generation lengths mean a different workload, so the ratio is not an equal-work
speedup.
If compile fails because the local tensor-length FIA interface or private ATB
lowering is absent, report that precise version/signature/error. Never freeze
CPU lengths, substitute ND storage, use eager fallback or edit the source.

## Profiler and report

`PROFILE=1` captures eight advancing complete decoder calls per compiled
candidate lane outside throughput. Verify 24 attention layers per call, KV writers, LM head and
sampling. Inspect attention input shapes/formats and TransData kernels. Retained
external NZ alone does not prove the compiled attention kernel consumes NZ: GE
may normalize or convert internally. Report those conversions and distinguish
external storage from actual attention input format.

Send the exact run directory, commit, command, device/environment, exit code,
summary JSON and logs. Include real crop IDs/image hashes/prompt lengths, prefill
seconds, complete token parity, EOS/cap counts, useful tokens and decode seconds
for every repeat, raw slots, median/min/max tok/s, AB/BA order, graph/warning
counters and cache descriptors. Pair ratios only for timing-valid same-contract
ND/NZ lanes, and state whether generated sequences and workload lengths match.
Preserve failed variants and profiler traces. Do not claim
310P parity or speed from the 910B results; chip contracts differ.

The wrapper already writes command, commit, occupancy before/after, exit status,
log, summary and profiles. Compile caches and raw profiler device dumps need
not be relayed; keep processed operator/kernel traces with formats and shapes.

## What 910B established

The [recorded 910B2 results](references/full_model_kv_910B_20261007/RESULTS.md)
verify full-model timing, real mixed input preparation, EOS accounting and
static fullgraph native controls. The compiled NZ candidates were timed with
fidelity reported separately. Their external descriptors remained 29, while
910B profiler attention inputs were ND and added 48 TransData kernels per
decoder step. They also showed substantial repetitive output drift. These
results do not show a native NZ-reading attention hot path or 310P behavior.
Private ATB APIs failed the installed TorchAir writer lowering before reaching
the reader. Re-discover and report the corresponding behavior on this server.
