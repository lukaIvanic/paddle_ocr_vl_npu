# 310P UniRec MSDA node-local shape retry

Run one candidate page only. This retry replaces the registration-order
experiment with a node-local GE contract. Do not run the decomposed baseline,
profiling, recognition, or evaluation yet.

## Why this is different

Huawei's official `ops-nn` source confirms a 310P host-inference defect. The
ACLNN wrapper transforms sampling locations to `[L,B,H,Q,P,2]`, but the graph
infer callback reads dimensions 5 and 1 and creates the wrong 64-element
output. The separate 310P tiler correctly reads `Q` from attention weights.

The converter now publishes the correct internal output descriptor
`[B,H*D,Q]` and sets GE's reserved `_disable_call_shape_inference=true` on
only the six MSDA nodes. CANN's stock tiler and installed 310P kernel remain
unchanged. No host OPP is loaded, so there is no registration race.

A fresh-cache 910B2 control at `2e2234f` passed without any host OPP:

```text
UNIREC_LAYOUT_MSDA_REAL_PHASE_SKIP phase=host_opp_build mode=none
LAYOUT_LAB phase=warmup_call_end call=1/2
LAYOUT_LAB phase=warmup_call_end call=2/2
LAYOUT_LAB page=1/1 wall_ms=90.1 forward_ms=11.9 boxes=6 digest=e0e3510c8327
UNIREC_LAYOUT_MSDA_REAL_PHASE_END phase=forward_candidate status=0 wall_s=80
```

The TorchAir warning that it could not auto-generate an additional symbolic
inference rule is expected. The static output descriptor is already supplied.

## Restrictions

- Pull only. Do not edit tracked files, commit, branch, or push.
- Never use physical NPU 5 or 6.
- Use the existing real `python_nosym` executable. Do not canonicalize it with
  `readlink -f`.
- Reuse the already-passing binding extension SO. Do not rebuild it.
- Use a fresh graph cache.
- Run one candidate page with two warmups.
- Set `MSDA_HOST_OPP_MODE=none`; no user OPP may be built or loaded.
- If it fails, stop after preserving the first GE/TBE error and the node-shape
  markers. Do not launch another NPU run.

## Launch

Source CANN before enabling shell nounset because some `set_env.sh` revisions
read optional unset variables.

```bash
set -eo pipefail
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
git pull --ff-only origin main

source npu-setup
set -u
case ",${ASCEND_RT_VISIBLE_DEVICES:-}," in
  *,5,*|*,6,*) echo "REJECTED_PHYSICAL_DEVICE_5_OR_6" >&2; exit 1 ;;
esac

export PYTHON_BIN="${PYTHON_BIN:?set the existing venv python_nosym executable}"
test -x "$PYTHON_BIN"
test "$(basename "$PYTHON_BIN")" = python_nosym

export MODEL="${MODEL:-$REPO/models/unirec-0.1b}"
export LAYOUT_MODEL="${LAYOUT_MODEL:-$REPO/models/PP-DocLayoutV2_safetensors}"
export OPENOCR_ROOT="${OPENOCR_ROOT:-$REPO/deps/OpenOCR_0d522801}"
export IMAGES_DIR="${IMAGES_DIR:?set the existing OmniDocBench images directory}"
export MSDA_EXTENSION_SO="${MSDA_EXTENSION_SO:?set the successful binding-probe extension SO}"
test -f "$MSDA_EXTENSION_SO"

export MSDA_REBUILD_EXTENSION=0
export MSDA_RUN_MODE=candidate_compile_probe
export MSDA_FORWARD_LIMIT=1
export MSDA_WARMUP_PAGES=2
export MSDA_HOST_OPP_MODE=none

bash 12_unirec_0_1b_inference/run_310p_layout_msda_real_background.sh
```

Immediately give Luka the printed absolute `RUN_LOG` and `TAIL_COMMAND`.
Follow only the owned PID until `exit_code.txt` appears.

## Pass gate

All are required:

1. `UNIREC_LAYOUT_MSDA_REAL_PHASE_SKIP phase=host_opp_build mode=none`.
2. Six `UNIREC_LAYOUT_MSDA_NODE_SHAPE_PIN` lines, each with
   `shape=[1, 256, 300] disable_host_infer=true`.
3. No `UNIREC_LAYOUT_MSDA_HOST_OPP` marker and no host-OPP directory.
4. No `Reshape_86` or 64-versus-76,800 element error.
5. Both warmups and the measured page finish with exit code zero.
6. The owned process exits and the physical NPU returns to clean idle state.

## Required report

Return:

- commit, physical NPU, exact Python path, torch, torch-npu, and CANN;
- all `NODE_SHAPE_PIN` lines and the TorchAir warning, if present;
- final `LAYOUT_LAB page` and `LAYOUT_LAB done` lines;
- first real GE/TBE error and traceback if it fails;
- absolute run root, run log, cache root, extension SO, output JSON, and
  exit-code file;
- confirmation that no host OPP was loaded and the owned NPU was released.

Then stop. Do not change the production default yet.
