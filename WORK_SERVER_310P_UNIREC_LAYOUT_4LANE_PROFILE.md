# 310P UniRec layout four-lane forward and profile matrix

Pull the commit containing this brief. Run only PP-DocLayoutV2 on one real
OmniDocBench page. Measure and profile these four exact lanes:

1. eager FP32 body + FP32 reading-order head;
2. eager FP16 body + FP32 reading-order head;
3. TorchAir-compiled FP32 body + FP32 reading-order head;
4. TorchAir-compiled FP16 body + FP32 reading-order head.

The committed background runner performs a production-style forward/output
gate and a clean 20-repeat NPU forward measurement around one profiled replay
for every lane. It then prints a direct 310P-versus-910B2 comparison.

## Restrictions

- Pull only. Do not edit tracked files, create a branch, commit, or push.
- Use one genuinely free physical 310P. Never use physical NPU 5 or 6.
- Use exactly one CPU thread. The runner pins PyTorch intra-op and inter-op to
  one and also pins OpenMP, MKL, OpenBLAS, NumExpr, and vecLib to one.
- Use the exact real page
  `jiaocaineedrop_jiaocai_needrop_en_620.jpg` from the existing
  OmniDocBench images directory.
- Use native depthwise convolutions and native weight formats in all four
  lanes. This matrix isolates only execution mode and body precision.
- Use a new compile-cache root. Do not reuse a cache created by an older source
  revision.
- Run in the background. Immediately give Luka the absolute run-log path and
  the exact `tail -f` command printed by the launcher.
- Preserve the first causal traceback and CANN error. Do not retry with changed
  precision, graph rewrites, thread counts, or JIT settings.
- Do not run page prefill, recognition, decode, OmniDocBench evaluation, or a
  full dataset in this task.
- Numerical differences are evidence, not an automatic failure. Do not invent
  an arbitrary parity tolerance. Report box topology, reading-order signature,
  coordinate drift, score drift, and paired IoU exactly.

## Exact 910B2 reference

These controls ran at source commit `172f209`, physical Ascend 910B2 NPU 7,
CANN 9.0.0, with PyTorch intra-op/inter-op and all CPU libraries pinned to one.
The input was the exact page named above.

Production-style one-page forward gate:

| Lane | Forward |
|---|---:|
| eager FP32 | 56.321 ms |
| eager FP16 body + FP32 head | 56.848 ms |
| compiled FP32 | 17.957 ms |
| compiled FP16 body + FP32 head | 15.613 ms |

Clean event-timed profile controls, averaged before and after profiling:

| Lane | Steady device time | Profile compute | Profile free | Kernels | Cube |
|---|---:|---:|---:|---:|---:|
| eager FP32 | 78.447912 ms | 24.6437 ms | 82.6317 ms | 2,161 | 79.40% |
| eager FP16 body + FP32 head | 76.380231 ms | 23.3988 ms | 81.9559 ms | 2,496 | 77.45% |
| compiled FP32 | 18.164489 ms | 17.9538 ms | 0.1356 ms | 1,559 | 78.19% |
| compiled FP16 body + FP32 head | 14.385262 ms | 14.2638 ms | 0.1266 ms | 1,920 | 71.41% |

The profiler perturbs eager execution, so use the production-style forward as
the real one-page latency and the profiler output for kernel attribution. Do
not present the profiled `Stage` time as production wall time.

Key 910B2 kernel totals, formatted as `count / total ms`:

| Lane | TransData | Cast | Conv2D | Add | Transpose | MatMulV2 | GridSample |
|---|---:|---:|---:|---:|---:|---:|---:|
| eager FP32 | 405 / 5.9125 | 32 / 0.0864 | 123 / 2.7489 | 266 / 2.1561 | 145 / 2.5619 | 145 / 1.5483 | 18 / 0.9974 |
| eager mixed | 405 / 4.6481 | 1 / 0.1931 | 123 / 1.7403 | 266 / 2.0234 | 145 / 2.5615 | 146 / 1.3959 | 18 / 1.0522 |
| compiled FP32 | 378 / 4.2831 | 36 / 0.0716 | 123 / 2.2564 | 137 / 1.4665 | 125 / 1.3534 | 145 / 1.1604 | 18 / 0.9900 |
| compiled mixed | 678 / 2.7972 | 260 / 0.5680 | 123 / 2.2923 | 117 / 0.7257 | 125 / 1.1934 | 146 / 0.9026 | 18 / 0.9744 |

On a separate representative-128 layout-only gate, compiled FP32 kept the box
count on 128/128 pages and the class/label/reading-order signature on 124/128.
Compiled mixed kept the box count on 127/128 and the signature on 106/128. Mean
paired IoU was 0.99613 and 0.99018 respectively. Therefore exact digests are
not required here.

## Resolve and launch

Run from the existing 310P checkout with Bash:

```bash
set -euo pipefail
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
git status --short --branch
git pull --ff-only origin main
git status --short --branch

source npu-setup
case ",${ASCEND_RT_VISIBLE_DEVICES:-}," in
  *,5,*|*,6,*) echo "REJECTED_PHYSICAL_DEVICE_5_OR_6" >&2; exit 1 ;;
esac

export PYTHON_BIN="${PYTHON_BIN:-$REPO/venv/bin/python}"
export MODEL="${MODEL:-$REPO/models/unirec-0.1b}"
export LAYOUT_MODEL="${LAYOUT_MODEL:-$REPO/models/PP-DocLayoutV2_safetensors}"
export OPENOCR_ROOT="${OPENOCR_ROOT:-$REPO/deps/OpenOCR_0d522801}"
export IMAGES_DIR="${IMAGES_DIR:?set this to the existing OmniDocBench images directory}"

test -x "$PYTHON_BIN"
test -f "$MODEL/model.pth"
test -f "$LAYOUT_MODEL/model.safetensors"
test -f "$OPENOCR_ROOT/tools/infer_doc_onnx.py"
test -d "$IMAGES_DIR"

bash 12_unirec_0_1b_inference/run_layout_precision_profile_matrix_background.sh
```

If a project-local default differs, export the correct existing path before
launching. Do not search for or create ONNX exports. Do not download a second
model or dataset merely to match the defaults.

The launcher must immediately print:

```text
UNIREC_310P_LAYOUT_4LANE_STARTED ...
RUN_ROOT=<absolute path>
RUN_LOG=<absolute path>
TAIL_COMMAND=tail -f <absolute path>
EXIT_CODE_FILE=<absolute path>
```

Send `RUN_LOG` and `TAIL_COMMAND` to Luka immediately. The log prints explicit
begin/end markers for all eight phases, so a slow compile or profile parse is
visible without guessing.

## Completion and report

Wait for the owned PID only. Do not infer completion from a quiet terminal.
Require `exit_code.txt` to contain `0` and the final line to contain:

```text
UNIREC_310P_LAYOUT_4LANE_END status=0
```

Return:

1. commit, physical NPU, CPU model, CANN, torch, and torch_npu;
2. all four `UNIREC_310P_LAYOUT_4LANE lane=...` lines;
3. both `UNIREC_310P_LAYOUT_4LANE_PAIR` lines;
4. the final `UNIREC_310P_LAYOUT_4LANE_PROFILE: PASS` line;
5. for every lane, the top 15 operator types with call count and total time;
6. for every lane, the top 15 exact TransData shape/format signatures;
7. absolute paths to `run.log`, `comparison_summary.json`, every
   `profile_suite_summary.json`, every `profile_parse_summary.json`, and the
   new compile-cache root;
8. any warning or fallback concerning internal formats, TorchAir cache loads,
   recompilation, JIT compilation, or profiler parsing.

Then stop. Do not continue into prefill or decode.
