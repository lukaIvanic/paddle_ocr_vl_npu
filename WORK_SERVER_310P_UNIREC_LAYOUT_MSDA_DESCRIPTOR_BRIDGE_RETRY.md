# 310P UniRec MSDA descriptor-bridge retry

Run one candidate page only. Do not run a baseline, profiler, recognition, or
evaluation. This retry replaces the ignored
`_disable_call_shape_inference` experiment.

## What the previous failure proved

The previous run printed all six correct
`shape=[1,256,300] disable_host_infer=true` markers, but CANN 9.1 still called
the stock host-inference function and produced the same 64-element output.
Therefore, that reserved attribute does not suppress the later GE
`InferShapePass` for this direct TorchAir custom node on the installed 310P
runtime.

## New mechanism

The actual 310P location buffer remains in the official ACLNN physical order:

```text
[L,B,H,Q,P,2] = [3,1,8,300,4,2]
```

Immediately before the stock GE MSDA node, a same-numel `Reshape` presents the
descriptor as:

```text
[B,H,L,P,2,Q] = [1,8,3,4,2,300]
```

The bytes do not move. This shape makes the broken stock host callback infer
`[B,Q,H*D] = [1,300,256]`, which allocates the correct 76,800 FP32 elements.
The 310P tiler is unaffected: its `isInfBase` path reads Q/H/P from the
attention-weight shape, not the location shape. The device kernel reads the
location pointer as a flat per-level buffer using its tiling values, so it
still sees the official physical byte order.

The kernel physically writes `[B,H*D,Q] = [1,256,300]`. A second same-numel
`Reshape` restores that physical descriptor, then the official output
transpose produces `[1,300,256]`.

No host OPP and no private kernel are used.

## Restrictions

- Pull only. Do not edit tracked files, commit, branch, or push.
- Never use physical NPU 5 or 6.
- Use the existing real `python_nosym` executable. Do not canonicalize it with
  `readlink -f`.
- Reuse the already-passing binding extension SO. Do not rebuild it.
- Use a fresh graph-cache root. Never reuse the failed cache.
- Run one candidate page with two warmups.
- Set `MSDA_HOST_OPP_MODE=none`.
- If it fails, preserve the first real GE/TBE error and stop. Do not launch a
  second NPU run.

## Launch

Source CANN before enabling shell nounset.

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

1. Commit matches the commit named in the Git commit message.
2. Host-OPP build is skipped.
3. Exactly six `UNIREC_LAYOUT_MSDA_DESCRIPTOR_BRIDGE` lines appear, each with:

   ```text
   location_shape=[1, 8, 3, 4, 2, 300]
   internal_output_shape=[1, 256, 300]
   ```

4. No `UNIREC_LAYOUT_MSDA_NODE_SHAPE_PIN` line appears.
5. No `numQueries ... got input 3`, `Reshape_86`, or 64-versus-76,800 error.
6. Both warmups and the one measured page finish with exit code zero.
7. The output tensors are finite, the owned process exits, and the physical
   NPU returns to clean idle state.

## Required report

Return:

- commit, physical NPU, exact Python path, torch, torch-npu, and CANN;
- all six descriptor-bridge lines;
- both warmup-end lines, final `LAYOUT_LAB page`, and `LAYOUT_LAB done`;
- first real GE/TBE error and traceback if it fails;
- absolute run root, run log, cache root, extension SO, output JSON, and
  exit-code file;
- confirmation that no host OPP was loaded and the owned NPU was released.

Then stop. Do not change the production default yet.
