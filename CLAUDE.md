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

Work is organized as a ladder of numbered experiments, `01_` through `09_`.
**`09_persistent_page_engine/` is the active one** — everything earlier is
retained as evidence for how the current design was reached. Read
[09_persistent_page_engine/README.md](09_persistent_page_engine/README.md) before
interpreting any 09 throughput or parity claim.

09 owns the full PaddleOCR-VL 1.6 page contract directly — PP-DocLayoutV3
loading and inference, crop/merge policy, prompt routing, page assembly, JSON and
Markdown output. It does **not** import PaddleX or PaddleOCR (removed in
`61c1418`).

## Lanes

There are two machines, and they are not interchangeable.

### 1. Local authoring + orchestration (here)

Luka's Mac checkout. No accelerator. This lane edits tracked files, prepares
scripts and docs, commits, pushes, and drives the 910B container over SSH. It
must never present unrun local code as validated inference.

### 2. Blue-zone 910B container (`ssh blue_zone_npu_container`)

The real validation lane, and reachable from here. **Pull-only for source:** edit
locally → commit → push → `git pull` on the container → run. Never hand-edit
tracked files on the container. If a change is needed, make it locally and push.

Verified 2026-07-27:

| | |
|---|---|
| Host | `liteserver-c001-4`, aarch64, Linux 5.10 (HCE2) |
| Devices | 8 × **Ascend 910B2**, 64 GB HBM each, `npu-smi` 25.0.rc1.1 |
| CANN | 9.0.0 (`/usr/local/Ascend/cann-9.0.0`, `ascend-toolkit/latest`, `nnal/atb`) |
| Checkout | `/workspace/repos/paddle_ocr_vl_npu` |
| Recognizer | `/workspace/models/PaddleOCR-VL-1.6` |
| Layout | `/workspace/models/PP-DocLayoutV3_safetensors` |
| Dataset | `/workspace/datasets/OmniDocBench/` — 1,651 images + `OmniDocBench.json` |
| TorchAir caches | `/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/` (~13 GB) |

**Always `source npu-setup` first.** It is at `/usr/local/bin/npu-setup`, on
PATH, and must be sourced rather than executed. It sources CANN 9.0.0 and ATB,
sets `TORCH_DEVICE_BACKEND_AUTOLOAD=0`, adds the ATB libs and jemalloc preload,
and — importantly on a shared box — calls `npu-status --last-free` to pick a free
device and export it as `ASCEND_RT_VISIBLE_DEVICES`. It prints the physical
device it selected and the interpreter to use.

Without it, `npu-smi` fails on `libc_sec.so` and `import torch_npu` fails with
"Failed to load the backend extension" — non-interactive SSH does not read
`~/.bashrc`, so nothing is set up for you:

```bash
ssh blue_zone_npu_container 'cd /workspace/repos/paddle_ocr_vl_npu && source npu-setup && <command>'
```

Interpreters — all give torch 2.10.0+cpu / torch_npu 2.10.0 / 8 visible devices:

- `/usr/local/python3.12.13/bin/python3` — transformers 5.5.4, torchvision
  0.25.0, torchair. **This is what the current 09 runs use**, and what
  `npu-setup` prints.
- `/workspace/venvs/paddleocr_vl_baseline_py312` — transformers 5.0.0, no paddlex.
- `/workspace/venvs/vllm_paddle_ocr_pipeline_py312` — transformers 5.5.4 plus
  paddlex 3.7.2. Only needed for legacy PaddleX comparisons; the production path
  no longer imports PaddleX.

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
pushed commit, runs, and Luka relays the report back manually. Existing briefs:
[WORK_SERVER_310P_EAGER_SMOKE.md](WORK_SERVER_310P_EAGER_SMOKE.md) and
[WORK_SERVER_310P_EXP09_LADDER.md](WORK_SERVER_310P_EXP09_LADDER.md).

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

## Running things

Current 09 entrypoints:

- `09_persistent_page_engine/scripts/run_offline_e2e.py` — diagnostic page
  assembler over an explicit `--image` list.
- `09_persistent_page_engine/scripts/run_omnidocbench.py` — the full OmniDocBench
  runner and production full-page path.

All experiment CLIs default to `--dtype fp16`. `bf16` is an explicit override;
`fp32` is intentionally not a supported run mode.
