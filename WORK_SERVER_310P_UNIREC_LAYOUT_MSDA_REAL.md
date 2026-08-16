# 310P UniRec real layout native-MSDA A/B

Run the native MSDA operator inside the faithful compiled PP-DocLayoutV2
forward. This is the next step after the standalone binding probe passed. Do
not rerun the standalone microprobe and do not change production defaults.

## What this run does

The owned background runner performs, in one process sequence:

1. exact current-production decomposed layout on the first 128 pages;
2. the same 128 pages with all six MSDA modules replaced by the installed
   `aclnnMultiScaleDeformableAttnFunction` binding;
3. one warmed NPU profile of each compiled graph;
4. page-by-page box, label, order, coordinate, score, and IoU comparison;
5. structural confirmation of `18 GridSample -> 0` and exactly six native
   MSDA calls.

Both graph caches are unique to this run. First-time compilation is visible in
`run.log`. After compilation, each forward lane prints one progress line per
page. Do not infer a stall while the log says `warmup_call_begin`; inspect the
owned PID and current NPU state instead.

## 910B2 reference

Commit `362314e` completed the real A/B on physical NPU 7 with one CPU thread:

```text
pages=128, boxes=988, same-box-count pages=128
labels changed=1, reading-order values changed=2
mean paired IoU=0.999013669, minimum paired IoU=0.941291041
coordinate max abs=2.0 px, coordinate mean abs=0.0458945 px
model forward mean=12.568813 -> 10.808175 ms, 1.162899x
complete sequential layout section=10.600485 -> 10.855643 pages/s, 1.024070x
profile replay=11.725250 -> 10.656558 ms, 1.100285x
kernels=1387 -> 1291
GridSample=18 -> 0
Transpose=125 -> 59
Cast=259 -> 226
native MSDA=6 calls, 0.27954 ms total
```

The one 910B class boundary change was `text -> footer` on
`PPT_linear-algebra primer_page_008.png`. Two order values changed by one on
one book page. Therefore this is a structural and geometry pass, but it is not
yet a production-quality adoption result. The 310P report must expose any such
differences exactly.

## Restrictions

- Pull only. Do not edit tracked files, branch, commit, or push.
- Never use physical NPU 5 or 6.
- Use one free 310P and one CPU thread.
- Reuse the extension SO from the completed binding-probe run. Do not rebuild
  it and do not install or update CANN, torch-npu, or DrivingSDK.
- Do not run recognition prefill, decode, or full OmniDocBench evaluation.
- Follow the owned background PID. Do not terminate unrelated processes.
- Expected wall time is several minutes, mostly two one-time graph compiles.
  If any phase takes more than five minutes without a log change, inspect that
  phase, the owned process tree, and NPU state before continuing.

## Launch

Use the absolute binding run root reported by the already-completed probe:

```bash
set -euo pipefail
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
git pull --ff-only origin main

source npu-setup
case ",${ASCEND_RT_VISIBLE_DEVICES:-}," in
  *,5,*|*,6,*) echo "REJECTED_PHYSICAL_DEVICE_5_OR_6" >&2; exit 1 ;;
esac

export PYTHON_BIN="${PYTHON_BIN:-$REPO/venv/bin/python}"
export MODEL="${MODEL:-$REPO/models/unirec-0.1b}"
export LAYOUT_MODEL="${LAYOUT_MODEL:-$REPO/models/PP-DocLayoutV2_safetensors}"
export OPENOCR_ROOT="${OPENOCR_ROOT:-$REPO/deps/OpenOCR_0d522801}"
export IMAGES_DIR="${IMAGES_DIR:?set the existing OmniDocBench images directory}"
export MSDA_BINDING_RUN_ROOT="${MSDA_BINDING_RUN_ROOT:?set the completed binding run root}"
test "$(tr -d '[:space:]' <"$MSDA_BINDING_RUN_ROOT/exit_code.txt")" = 0
export MSDA_EXTENSION_SO
MSDA_EXTENSION_SO="$(tr -d '[:space:]' <"$MSDA_BINDING_RUN_ROOT/extension_so.txt")"
test -f "$MSDA_EXTENSION_SO"

bash 12_unirec_0_1b_inference/run_310p_layout_msda_real_background.sh
```

Immediately give Luka the printed absolute `RUN_LOG` and `TAIL_COMMAND`. Follow
the run until `exit_code.txt` appears.

## Required report

Return:

1. `UNIREC_LAYOUT_MSDA_REAL_AB ...`;
2. `UNIREC_LAYOUT_MSDA_REAL_PROFILE ...`;
3. the exact `label_mismatches` and `order_mismatches` arrays from
   `comparison_summary.json`;
4. baseline and candidate page/s, model-forward mean/p50/p90, box count, and
   page digest-match count;
5. baseline and candidate profile replay ms, total kernel count, total compute
   ms, and count/time for GridSample, native MSDA, Transpose, Cast,
   ReduceProdD, and Cumsum;
6. physical NPU, commit, Python, torch, torch-npu, process wall time, and the
   absolute run/cache roots;
7. absolute paths to `run.log`, both forward JSONs, both profile summaries,
   `comparison_summary.json`, `exit_code.txt`, and `npu_after.txt`;
8. confirmation that the owned process exited and its NPU was released.

Then stop. Do not make native MSDA the production default. We will decide that
only after reviewing real 310P speed and output drift.
