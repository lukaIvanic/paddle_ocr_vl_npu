# ColQwen3-4B: compiled B1 validation on one 310P

This is a runnable handoff, not a 310P performance report. The 310P machine is
not reachable from the authoring machine. Run there and report directly to Luka
in chat. Keep JSON, tensor captures and logs; do not write a Markdown report.

You need no previous conversation, access to Luka's computer, or access to the
blue-zone server. Everything executable below comes from this GitHub repository:
`https://github.com/lukaIvanic/paddle_ocr_vl_npu.git`. References to 910B results
are comparison facts only; no step fetches a file from that server. Your job is
to discover this machine's existing environment and assets, execute these
committed scripts, and explain the outcome. Do not implement fixes or improvise
replacement code. Read-only discovery and the commands below are authorized.

## Scope and source

Validate the exact FP16/native-weight optimized implementation that completed
full HR on 910B2 at `e46aea34`. This is page/query embedding (prefill), not
generation or decode. B1, original image processing, all 2560 embedding
dimensions, 1110 HR pages, 318 English queries, original FP32 MaxSim scoring.
No quantization, resolution reduction, token truncation, new attention mode,
NZ conversion, CPU model fallback, or eager fallback in the compiled run.

The previous run at `83bd2640` passed the query but failed the first page at
`prepare_text`'s boolean indexed write (`aclnnNonzeroV2` / AICPU `IndexPut`,
507018). The current source replaces that write with `masked_scatter`, the
vision rotary table read with embedding lookup, rotary strided writes with
full-shape selection, and the original owned text model's masked read/write
with dense masked updates. On-device benchmark validity checks also avoid
boolean selection. These are indexing compatibility changes; shapes, token
order, dtype, attention, and dataset are unchanged. They need target validation.
CPU-only position metadata and post-materialization result checks retain CPU
indexing. Historical native-op experiments and the unmodified third-party HF
oracle are not the portable execution path in this brief.

After this failure, use a NEW process and NEW RUN_ROOT, reusing the verified
dataset. Do not continue a process whose NPU stream has already failed. The
source hashes automatically select new compiled cache identities; retain the
old logs. Repeat the contracts and all three smoke cases before full HR.

The source branch is `codex/colqwen-warm-forward-profile` on origin. The server
is pull-only: do not hand-edit tracked source, commit, push, create branches,
patch dependencies, or replace its working CANN/torch/torch-npu stack. A detached
checkout fetched from origin is allowed; it does not create a branch. Preserve
unrelated work. If tracked changes or a checkout conflict exist, stop and report.

Use Bash. First run `pwd`, `hostname`, `uname -a`, `id`, `command -v git`,
`command -v python3`, and `git rev-parse --show-toplevel`. If the last command
fails, locate an existing checkout with a bounded search:

```bash
find /home /root /workspace /opt /data -maxdepth 5 -type d \
  -name paddle_ocr_vl_npu -print 2>/dev/null
```

Enter a returned checkout and verify `git remote -v` identifies the repository
above. If there is no checkout, bootstrap a new detached checkout in an unused
directory below the agent's current writable working directory:

```bash
test ! -e colqwen_310p_validation || exit 1
mkdir colqwen_310p_validation
cd colqwen_310p_validation
git init
git remote add origin https://github.com/lukaIvanic/paddle_ocr_vl_npu.git
git fetch origin codex/colqwen-warm-forward-profile
git checkout --detach FETCH_HEAD
```

This creates no named branch or commit. If GitHub/authentication is unavailable,
report that blocker; do not attempt to reach another machine for source. Once
inside the correct checkout, use the common update sequence:

```bash
export WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git status --short --branch
git diff --quiet && git diff --cached --quiet || exit 1
git fetch origin codex/colqwen-warm-forward-profile
git checkout --detach FETCH_HEAD
git rev-parse HEAD
```

## Discover this machine before selecting paths

Do not assume there is a ColQwen virtual environment or a blue-zone-style
container. Run in the local shell/container that has access to this server's
310P devices. Collect the following read-only inventory:

```bash
command -v npu-setup || true
command -v npu-smi || true
command -v docker || true
df -h
free -h
python3 21_colqwen3_4b_inference/discover_310p.py
```

The committed discovery script needs only standard-library Python (3.8+). It
lists candidate Python environments and their installed-package metadata,
CANN/ATB setup scripts, matching ColQwen checkpoints, and HR dataset roots. It
does not import torch, allocate an NPU, install packages, or alter configuration.
Preserve its printed inventory. A bounded search may report `truncated=true`
or permission errors: that is NOT proof that assets are missing. Repeat with
specific accessible mount points shown by `df -h`, or local paths from the
machine's run commands, using one or more `--root /actual/path` arguments.

If the host has no usable accelerator environment but Docker is available,
inspect `docker ps --format '{{.Names}} {{.Image}}'`. For an existing relevant
310P container, use `docker exec <discovered-name> ...` to run the SAME inventory
inside it. Check that it has the repository and model/dataset mounts. Do not
launch, recreate, restart or install into a container. If none is usable, report
the inventory and blocker. Do not use the blue-zone container name.

Choose an interpreter with the installed torch/torch-npu/TorchAir stack used by
this machine's existing successful 310P work. The inventory's `candidate` path
preserves virtual environments; do not replace it with a resolved system-Python
symlink. Select the ColQwen-compatible HF dependencies listed below. Metadata
discovery is not proof of importability: the preflight tests actual imports.
An absent top-level `torchair` distribution is not by itself a failure: some
stacks bundle it under `torch_npu.dynamo.torchair`, which the preflight checks.
If no compatible interpreter exists, report the candidate/version matrix and
missing or mismatched packages. Do not install packages or create an environment
under this execution-only brief; provisioning must be arranged by Luka.

If `npu-setup` exists, inspect its local contents before sourcing it; use it only
if it sets up this machine's 310P runtime. Otherwise inspect the discovered
`set_env.sh` paths and local successful run commands to identify the existing
matching CANN environment (and ATB environment if that stack uses it). Source
those exact scripts. Do not select the newest-looking version by guess. If the
choice is ambiguous, report the candidates and stop rather than mixing stacks.
Record the setup-script paths and resolved installation versions in your reply.

Run `npu-smi info` after activation. Choose one healthy, idle 310P with enough
free memory and no other process using it. Export its actual physical ID as
`ASCEND_RT_VISIBLE_DEVICES`; the scripts use logical `npu:0`. Never stop another
process. If no card is free, report availability and stop. Do not reuse the
blue-zone NPU-0 reservation or its CANN paths. Before running the next sections,
confirm the selected interpreter, setup scripts, device, model and dataset all
belong to the SAME host/container filesystem.

## Assets and environment

Select existing paths from the inventory; the assignments below are placeholders
to replace with those observed values, not commands to run literally.
The checkpoint is `OpenSearch-AI/Ops-Colqwen3-4B`. Required file SHA256 values,
including both weight shards, are committed in `310p_assets.json`. The dataset
is `vidore/vidore_v3_hr_mteb_format`, revision
`bc7d43d64815ed30f664168c8052106484aba7fd`; the three exact English Parquet files
and their hashes are in `download_hr_reference.py`. Downloading these three
files (440,304,762 bytes, about 440 MB / 420 MiB total) is explicitly authorized
by Luka. If absent, use the download phase below; do not stop merely because
the dataset is missing. Preserve any existing original per-collection ViDoRe
copy. It has different schemas/hashes and is not the input expected by this
runner. Missing model weights remain a provisioning blocker to report.

Use a dedicated ColQwen environment with the server's compatible torch,
torch-npu, TorchAir and torchvision binaries. The validated processor uses
Transformers **4.57.1**, tokenizers **0.22.2**, huggingface-hub **0.36.2**;
PyArrow **21.0.0** was used on 910B. Also required: numpy, Pillow, safetensors
and an importable `pytrec_eval`. The preflight reports actual versions and
requires Transformers 4.57.1. Do not run the existing 910B-specific
`setup_environment.sh` verbatim on 310P. Missing packages or an incompatible
TorchAir API are blockers to report, not permission to modify shared packages.

```bash
export PYTHON_BIN=/absolute/path/to/colqwen/python
export COLQWEN_MODEL=/absolute/path/to/Ops-Colqwen3-4B
# Use an existing verified MTEB copy if found; otherwise this new destination:
export HR_DATASET="$WORK_SERVER_REPO/tmp/datasets/vidore_hr_mteb_bc7d43d"
export RUN_ROOT="$WORK_SERVER_REPO/tmp/21_colqwen3_4b_inference/310p_$(git rev-parse --short HEAD)_$(date -u +%Y%m%dT%H%M%SZ)"
export COLQWEN_CACHE="$WORK_SERVER_REPO/.runtime_cache/21_colqwen3/310p_portable_b1_npu${ASCEND_RT_VISIBLE_DEVICES}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$RUN_ROOT/hf_home" HF_MODULES_CACHE="$RUN_ROOT/hf_modules"
test -x "$PYTHON_BIN" && test -d "$COLQWEN_MODEL" || exit 1
mkdir -p "$RUN_ROOT"

run_phase() {
    local phase="$1"
    shift
    mkdir -p "$RUN_ROOT/$phase"
    { git rev-parse HEAD; hostname; printf 'ASCEND_RT_VISIBLE_DEVICES=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"; printf '%q ' "$@"; printf '\n'; } > "$RUN_ROOT/$phase/command.txt"
    local code
    if "$@" > "$RUN_ROOT/$phase/run.log" 2>&1; then code=0; else code=$?; fi
    printf '%s\n' "$code" > "$RUN_ROOT/$phase/exit_code.txt"
    tail -n 3 "$RUN_ROOT/$phase/run.log"
    return "$code"
}
```

Use the same shell/environment for the commands below. Each phase has an
explicit failure guard. Continue automatically after a successful phase; no
additional approval between successful phases is required by this brief.
These functions/exports are shell state: if your tool starts a fresh shell per
call, source the same setup scripts and repeat the assignments and function
definition in that shell. Keep the SAME recorded RUN_ROOT; do not regenerate it
for every command. Never rerun a completed phase into its existing output
directory. A retry must use a clearly labeled new RUN_ROOT and explain why.
Keep the chosen cache exclusive to this process; do not run concurrent writers
or copy 910B compiled graphs. The local 310P smoke creates its own graphs and
the full run reuses them. HF cache directories above are local writable scratch;
they do not replace the explicit existing model and dataset paths.

## Execute in order

0. **Provision/verify the English dataset:** run this phase even if the target
   directory exists; matching completed files are verified and reused. This
   script uses only Python's standard library, needs no NPU, and does not install
   anything. It downloads directly from the pinned revision on Hugging Face,
   then automatically tries `https://hf-mirror.com` if primary attempts fail.
   Both endpoints must satisfy the SAME pinned sizes and SHA256 hashes.
   Networking is intentional for this phase; subsequent inference stays offline.

```bash
mkdir -p "$RUN_ROOT/download"
{ git rev-parse HEAD; printf '%q ' "$PYTHON_BIN" -u 21_colqwen3_4b_inference/download_hr_reference.py --root "$HR_DATASET"; printf '\n'; } > "$RUN_ROOT/download/command.txt"
set -o pipefail
if "$PYTHON_BIN" -u 21_colqwen3_4b_inference/download_hr_reference.py \
    --root "$HR_DATASET" 2>&1 | tee "$RUN_ROOT/download/run.log"; then
    download_code=0
else
    download_code=$?
fi
printf '%s\n' "$download_code" > "$RUN_ROOT/download/exit_code.txt"
test "$download_code" -eq 0 || exit 1
```

The log is live, not a terminal-only progress bar: flushed JSON every five
seconds includes file, endpoint, attempt, bytes/total, percent, average MB/s,
elapsed seconds and `no_data_s` (time without new bytes). A `connecting` heartbeat
with rising `no_data_s` means no payload has arrived; it is not download progress.
Each socket operation has a 30-second timeout; the parent terminates a worker
after 60 seconds without new bytes (including DNS/connect hangs), or 30 minutes
total per attempt. There are two attempts per file per endpoint. Failures print
`attempt_failed` with the error, and exhaustion exits nonzero. These limits are
downloader defaults chosen for this handoff, not benchmark requirements.

Use a long-lived tool session and inspect `tail -n 5 "$RUN_ROOT/download/run.log"`
every 30–60 seconds if tool output is not streaming. Do not start another writer
to the same dataset directory. After an interrupted run, rerun the same download
command with a new log path: `.part` files resume with HTTP Range; a server that
ignores Range restarts that file safely. Completed files become visible under
their final names only after SHA256 validation. Bad existing final files are
preserved and rejected; select a separate destination rather than overwriting
the original dataset. Success ends with `event=finish, status=verified` and
writes `manifest.json`. If both endpoints fail, report the endpoint errors,
received bytes and log path. No source edits or further approval are required
to download, retry, or proceed after verification.

1. **Preflight:** imports, one-device/chip guard, runtime APIs, exact asset
   hashes and available device memory. No model is loaded here.

```bash
run_phase preflight "$PYTHON_BIN" -u 21_colqwen3_4b_inference/run_portable_smoke.py \
  --phase preflight --model "$COLQWEN_MODEL" --dataset-root "$HR_DATASET" \
  --cache-root "$COLQWEN_CACHE" --output-dir "$RUN_ROOT/preflight/output" || exit 1
```

Before loading the checkpoint, also execute the committed CPU contract tests
with this interpreter (these do not validate NPU kernels):

```bash
run_phase contracts env PYTHONPATH=21_colqwen3_4b_inference "$PYTHON_BIN" -m unittest \
  test_portable_smoke test_hr_evaluation test_optimized_prefill \
  test_prepared_prefill test_indexing_compat || exit 1
```

2. **Real-input smoke:** HR query 0, then distinct full pages 5 and 0. Run
   optimized eager and compiled B1 on identical processed inputs. Five warmups,
   twenty timed complete forwards per case and execution mode. Compilation,
   processing, external transfers, validation and file writes are outside the
   warm forward timing. The same page/query modules and graph names are used
   by the full evaluator. No additional HF reference model is loaded.

```bash
run_phase smoke "$PYTHON_BIN" -u 21_colqwen3_4b_inference/run_portable_smoke.py \
  --model "$COLQWEN_MODEL" --dataset-root "$HR_DATASET" \
  --cache-root "$COLQWEN_CACHE" --output-dir "$RUN_ROOT/smoke/output" || exit 1
```

Check `PORTABLE_SMOKE ... phase=finish ... status=passed`, three completed
cases, recorded memory peaks and own-eager/compiled parity. This integration
check uses `atol=rtol=0.002`; it is not an HF or retrieval-quality criterion.
If it fails, report the actual differences and stop for investigation. Do not
relax it silently. On OOM, report allocation/available-memory data and the first
causal error; do not lower resolution, change dtype, or try quantization.

The model still owns original projection weights alongside packed projections;
do not assume its memory footprint is simply the checkpoint size. The full
evaluation also retains about **6.75 GiB of page embeddings in host RAM**, plus
dataset, model-loading and scoring memory. Check host headroom before that run.

3. **Full HR:** proceed only after the smoke passes on this same chip, runtime,
   source and assets. Reuse these caches. Stay engaged and monitor
   `output/progress.json` and the log until completion.

```bash
run_phase full_hr "$PYTHON_BIN" -u 21_colqwen3_4b_inference/run_hr_evaluation.py \
  --model "$COLQWEN_MODEL" --dataset-root "$HR_DATASET" \
  --workload full --execution torchair --device npu:0 \
  --cache-root "$COLQWEN_CACHE" --output-dir "$RUN_ROOT/full_hr/output" || exit 1
```

Expect `status=completed`, `pages=1110`, `queries=318`, and
`compiled_coverage={"transformer_calls":2538,"all_compiled":true}`. All pages
should have 5040 vision / 1274 real text positions; text attention uses 1280.
All 318 queries share the existing 128-position graph and return only real
query rows. No profiler overhead is enabled. Do not substitute the default
111-page development workload for the full run.

Long-run monitoring is part of the task. Use a long-lived tool session, or the
tool's documented background execution support, so a short tool timeout does
not kill the process. Poll `tail -n 5 "$RUN_ROOT/full_hr/run.log"` and
`cat "$RUN_ROOT/full_hr/output/progress.json"` about every 30–60 seconds.
The evaluator emits heartbeats every five seconds; compilation may take much
longer than one item. The smoke prints phase starts and compile-cache events
but has no heartbeat. Do not label quiet smoke output as a hang or kill it
without evidence. If monitoring becomes unavailable, report the PID/log and
resume monitoring rather than launching a duplicate job. A failed phase's
`exit_code.txt` and first causal traceback take precedence over a partial
throughput line. Stay until completion or a concrete reported blocker.

## Compatibility basis and limits

- [Huawei's requested TorchNPU 26.0 index](https://www.hiascend.com/document/detail/zh/Pytorch/2600/apiref/torchnpuCustomsapi/docs/zh/custom_APIs/torch_npu/torch_npu_list.md).
- [Official pinned PromptFA source](https://github.com/Ascend/op-plugin/blob/6ffd6e22ad4ea0fe6f20d6c8d249b772e92191d4/docs/en/custom_APIs/torch_npu/torch_npu-npu_prompt_flash_attention.md): inference-card FP16; no sequence-length lists or PSE; leave KV-head count at its default by explicitly repeating K/V; sparse mode 0. Vision uses unmasked BNSD, N16/D64. Text uses square bool masks and BSND, N32/D128. Q/K/V are contiguous.
- [Official pinned GELU source](https://github.com/Ascend/op-plugin/blob/6ffd6e22ad4ea0fe6f20d6c8d249b772e92191d4/docs/en/custom_APIs/torch_npu/torch_npu-npu_gelu.md): inference-product FP16/FP32 support and graph mode; keep explicit tanh approximation.
- The conservative 128-aligned masked-attention handling follows the existing
  experiment-13 prefill work. Dummy keys are blocked for real queries; dummy
  queries have a valid self entry. The FP16-mask-pad-then-bool conversion retains
  the observed GE fix. This is not a hidden-width change or token truncation.
- Keep manual FP32-statistics vision LayerNorm with FP16 affine, manual RMSNorm,
  manual rotary, and ordinary SiLU/multiply. Do not import historical native-GQA,
  joint-rotary, native-SwiGLU, or native-RMSNorm variants into this baseline.

These references support the selected API contracts; they do not establish
that the installed 310P software stack compiles the whole model or that it has
sufficient memory. Only the target runs establish that. Harness checks on 910B
use the explicit `--expected-chip 910B` flag; NEVER use that override on 310P or
remove the default chip guard to bypass a discovery/configuration failure.

## Report directly to Luka

Report actual chip/device ID, source commit, runtime versions, asset validation,
memory peaks, eager/compiled warm page timings and pg/s, same-implementation
output differences, compilation/first-use time separately, then full-HR total
time, page-encoding time/pg/s, compiled coverage and all three retrieval metrics.
On failure, give the command, first causal error, stage, and log path; do not
make source changes on this machine. Keep the same RUN_ROOT for later inspection.

The comparison baseline is **910B2**, not an expected 310P speed: at `e46aea34`,
full HR took 453.5 seconds including imports/setup; page encoding 362.8 seconds
(3.060 pg/s); nDCG@10 **66.4901%**, Recall@10 **70.8346%**, MAP@10 **52.0896%**.
Warmed model-forward B1 was about **7.36 pg/s** in the separate benchmark.
Use retrieval metrics to assess target quality; numerical embedding tolerances
do not establish quality equivalence or regression by themselves.
