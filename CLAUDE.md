# CLAUDE.md — paddle_ocr_vl_npu

Current, load-bearing orientation for this repo. `AGENTS.md` holds the deeper
per-experiment history and findings; read it when you need the background behind
a design decision, not to learn where you are.

## What this repo is

A standalone research workspace for **PaddleOCR-VL 1.6** on Ascend NPU. The
recognition VLM (`PaddleOCR-VL-1.6-0.9B`) is a native-resolution vision encoder
plus adaptive MLP projector plus an ERNIE-4.5-0.3B decoder-only multimodal LM.
Visual embeddings replace `<image>` token embeddings before decoder inference;
there is no encoder-decoder cross-attention.

Work is organized as a ladder of numbered experiments, `01_` through `18_`.
**`09_persistent_page_engine/` is the active PaddleOCR-VL engine**; experiments
10–17 are self-contained sibling model/runtime investigations. Read
[09_persistent_page_engine/README.md](09_persistent_page_engine/README.md) before
interpreting any 09 throughput or parity claim.

`18_unirec_paddle_hybrid_pipeline/` owns the continuous hybrid coordinator,
with configurable text/table/formula routing to the existing 09 and 12 engines.
Its README records implementation scope and validation status.

09 owns the full PaddleOCR-VL 1.6 page contract directly — PP-DocLayoutV3
loading and inference, crop/merge policy, prompt routing, page assembly, JSON and
Markdown output. It does **not** import PaddleX or PaddleOCR.

## Lanes

Authoring, 910B validation, and out-of-band 310P validation are separate lanes.

### 1. Local authoring + orchestration (here)

The local Linux or Mac checkout has no accelerator. This lane edits tracked
files, prepares scripts and docs, commits, pushes, and drives the 910B container
through the existing host SSH master and `docker exec`. It
must never present unrun local code as validated inference.

### 2. Blue-zone 910B container

The real validation lane, and reachable from here. **Pull-only for source:** edit
locally → commit → push → `git pull` on the container → run. Never hand-edit
tracked files on the container. If a change is needed, make it locally and push.

Access is through a persistent multiplexed SSH **host** connection followed by
`docker exec -i research_vllm_ascend_021_external_workspace`. Reuse the existing
master; do not create a new connection for every command. The local helper
discovers a live host socket in `/tmp/codex-blue-zone-$(id -u)/`, or accepts
`BLUE_ZONE_CONTROL_PATH` explicitly. On the Mac it also checks the established
`/tmp/codex-blue-zone-scope-%C` socket pattern.

```bash
bash scripts/blue_zone_exec.sh --check
bash scripts/blue_zone_exec.sh bash --noprofile --norc -c \
  'cd /workspace/repos/paddle_ocr_vl_npu && source /usr/local/bin/npu-setup && <command>'
```

The helper requires an existing live master and fails if none is available.
It does not restart containers or replace shared connections.

Current working paths:

| | |
|---|---|
| Host / container | `liteserver-c001-4` / `research_vllm_ascend_021_external_workspace` |
| Accelerator | **Ascend 910B2**; inspect current health and occupancy before runs |
| Checkout | `/workspace/repos/paddle_ocr_vl_npu` |
| Recognizer | `/workspace/models/PaddleOCR-VL-1.6` |
| Layout | `/workspace/models/PP-DocLayoutV3_safetensors` |
| Dataset | `/workspace/datasets/OmniDocBench/` — 1,651 images + `OmniDocBench.json` |
| TorchAir caches | `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/` |

**Always `source npu-setup` first.** It is at `/usr/local/bin/npu-setup`, on
PATH, and must be sourced rather than executed. It sources the installed CANN and ATB,
sets `TORCH_DEVICE_BACKEND_AUTOLOAD=0`, adds the ATB libs and jemalloc preload,
and — importantly on a shared box — calls `npu-status --last-free` to pick a free
device and export it as `ASCEND_RT_VISIBLE_DEVICES`. It prints the physical
device it selected and the interpreter to use.

Verify that the selected physical card reports `OK` health and is unoccupied
before inference. Occupancy-based selection alone does not establish health.
If necessary, choose another healthy, free card from the current inventory;
if none exists, report that blocker rather than sharing an occupied card.

Use `/usr/local/python3.12.13/bin/python3` for model/operator probes; its torch
and torch-npu imports were verified through this route on 2026-10-06 (both
2.10.0, observed device `Ascend910B2`). Full experiment-09 runs use
`/workspace/venvs/vllm_paddle_ocr_pipeline_py312/bin/python`, which supplies
`kornia_rs` for the owned layout frontend. Check the selected interpreter's
installed packages when a run depends on a particular version.

The box is shared with other users. `npu-setup` handles device selection; do not
override it with a hand-picked device unless you have a reason, and never
terminate another user's process. Concurrent runs must use distinct
`--torchair-cache-dir`, `--vision-torchair-cache-dir`, and
`--text-torchair-cache-dir` values; simultaneous writers invalidate each other's
caches.

### The 310P work server (out of band)

Luka's work server has Atlas **310P** devices. It is **not reachable from here**
and cannot push to GitHub — it only pulls. Work with it by writing a
self-contained handoff brief, which Luka carries over; an agent there pulls the
pushed commit, runs, and Luka relays the report back manually. Use the handoff
for the experiment being run rather than an older general smoke runbook.

A brief must assume nothing: it states its own constraints, resolves its own
paths, names every required check, and specifies the exact report format. 310P is
a different chip with different operator constraints than 910B — do not carry a
910B result over to it, or the reverse.

## Evidence conventions

- Runs are recorded under `tmp/<experiment>/<run_name>_<commit>/`, force-added
  past `.gitignore` on purpose: each keeps `command.txt` (git commit, hostname,
  `ASCEND_RT_VISIBLE_DEVICES`, exact command), `exit_code.txt`, `run.log`, and
  the run's output. When you need to know how something was actually invoked,
  read the committed `command.txt` — it is the authority, ahead of any prose.
- Label results by the chip they ran on. A 910B number is not a 310P number.
- Call a smoke test a smoke test. Recognizer-only runs are not proof of full
  page-parser quality or throughput.

## Custom Ascend operators

Read the
[Ascend custom operator handbook](09_persistent_page_engine/custom_ops/ASCEND_CUSTOM_OPERATOR_HANDBOOK.md)
before creating or changing a custom CANN/AscendC operator. It defines the
independent-identity model, eager-first validation ladder, AIV launch and
synchronization gates, TorchAir cache discipline, real-forward adoption gate,
and evidence format derived from the IncreFA work.

## Running things

Current 09 entrypoints:

- `09_persistent_page_engine/scripts/run_offline_e2e.py` — diagnostic page
  assembler over an explicit `--image` list.
- `09_persistent_page_engine/scripts/run_omnidocbench.py` — the full OmniDocBench
  runner and production full-page path.

All experiment CLIs default to `--dtype fp16`. `bf16` is an explicit override;
`fp32` is intentionally not a supported run mode.
