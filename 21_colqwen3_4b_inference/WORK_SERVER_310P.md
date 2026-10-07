# ColQwen3-4B: compiled B1 validation on one 310P

This is a runnable handoff, not a 310P performance report. The 310P machine is
not reachable from the authoring machine. Run there and report directly to Luka
in chat. Keep JSON, tensor captures and logs; do not write a Markdown report.

## Scope and source

Validate the exact FP16/native-weight optimized implementation that completed
full HR on 910B2 at `e46aea34`. This is page/query embedding (prefill), not
generation or decode. B1, original image processing, all 2560 embedding
dimensions, 1110 HR pages, 318 English queries, original FP32 MaxSim scoring.
No quantization, resolution reduction, token truncation, new attention mode,
NZ conversion, CPU model fallback, or eager fallback in the compiled run.

The source branch is `codex/colqwen-warm-forward-profile` on origin. The server
is pull-only: do not hand-edit tracked source, commit, push, create branches,
patch dependencies, or replace its working CANN/torch/torch-npu stack. A detached
checkout fetched from origin is allowed; it does not create a branch. Preserve
unrelated work. If tracked changes or a checkout conflict exist, stop and report.

```bash
export WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git status --short --branch
git diff --quiet && git diff --cached --quiet || exit 1
git fetch origin codex/colqwen-warm-forward-profile
git checkout --detach FETCH_HEAD
git rev-parse HEAD
```

Activate the existing successful 310P NPU environment. Use `source npu-setup`
only if that is the server's established setup command. Select one healthy,
free 310P; record `npu-smi info`. Never stop another process. Export exactly one
physical ID in `ASCEND_RT_VISIBLE_DEVICES`; the logical device is `npu:0`.
Do not copy the 910B NPU-0 reservation or its CANN paths onto this machine.

## Assets and environment

Resolve existing local paths; the examples below are placeholders to replace.
The checkpoint is `OpenSearch-AI/Ops-Colqwen3-4B`. Required file SHA256 values,
including both weight shards, are committed in `310p_assets.json`. The dataset
is `vidore/vidore_v3_hr_mteb_format`, revision
`bc7d43d64815ed30f664168c8052106484aba7fd`; the three exact English Parquet files
and their hashes are in `download_hr_reference.py`. No substitutes or downloads
are performed by this handoff. If assets are absent, report what is missing and
the paths checked so Luka can arrange provisioning.

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
export HR_DATASET=/absolute/path/to/ViDoRe_v3_hr_mteb_reference
export RUN_ROOT="$WORK_SERVER_REPO/tmp/21_colqwen3_4b_inference/310p_$(git rev-parse --short HEAD)_$(date -u +%Y%m%dT%H%M%SZ)"
export COLQWEN_CACHE="$WORK_SERVER_REPO/.runtime_cache/21_colqwen3/310p_portable_b1"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
test -x "$PYTHON_BIN" && test -d "$COLQWEN_MODEL" && test -d "$HR_DATASET"
mkdir -p "$RUN_ROOT"

run_phase() {
    local phase="$1"
    shift
    mkdir -p "$RUN_ROOT/$phase"
    { git rev-parse HEAD; hostname; printf 'ASCEND_RT_VISIBLE_DEVICES=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"; printf '%q ' "$@"; printf '\n'; } > "$RUN_ROOT/$phase/command.txt"
    "$@" > "$RUN_ROOT/$phase/run.log" 2>&1
    local code=$?
    printf '%s\n' "$code" > "$RUN_ROOT/$phase/exit_code.txt"
    tail -n 3 "$RUN_ROOT/$phase/run.log"
    return "$code"
}
```

Use the same shell/environment for the commands below. Each phase has an
explicit failure guard. Continue automatically after a successful phase; no
additional approval between successful phases is required by this brief.

## Execute in order

1. **Preflight:** imports, one-device/chip guard, runtime APIs, exact asset
   hashes and available device memory. No model is loaded here.

```bash
run_phase preflight "$PYTHON_BIN" -u 21_colqwen3_4b_inference/run_portable_smoke.py \
  --phase preflight --model "$COLQWEN_MODEL" --dataset-root "$HR_DATASET" \
  --cache-root "$COLQWEN_CACHE" --output-dir "$RUN_ROOT/preflight/output" || exit 1
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
All 318 query graphs use the existing 128-position alignment and return only
real query rows. No profiler overhead is enabled. Do not substitute the default
111-page development workload for the full run.

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
sufficient memory. Only the target runs establish that. The successful 910B
smoke of this handoff is explicitly labeled `--expected-chip 910B`.

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
