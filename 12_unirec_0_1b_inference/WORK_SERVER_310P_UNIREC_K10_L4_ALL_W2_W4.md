# 310P optimized K10/L4-all W2 and W4 scaling gate

## Objective

Measure clean prefill throughput for W2/T8 and W4/T8 with the new optimized
vision plan. Do not rerun W1 or any old/native baseline.

Both lanes use:

- layout B2/T16, TorchAir FP16 body plus FP32 reading order
- internal layout weights, constant-grouped depthwise, preformatted FrozenBN
- vision lookahead 4 and `310p_k10_l4_all`
- internal vision weights and `constant_grouped_all`
- compact uint8 recognition inputs
- cross-KV 1320, self-KV 2048
- ten compiled vision graphs and zero eager fallback rows
- one physical 310P device

## Work-server setup

Use this commit or a descendant. Do not edit tracked files. The server has four
NPUs and no `npu-setup`; select one free physical device from 0-3 using the
normal work-server method.

Reuse the exact validated paths and cache roots from the completed K10/L4-all
W1/T8 run:

```bash
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
export PYTHON_BIN=/absolute/path/to/python_nosym
export MODEL=/absolute/path/to/unirec-0.1b
export LAYOUT_MODEL=/absolute/path/to/PP-DocLayoutV2_safetensors
export OPENOCR_ROOT=/absolute/path/to/OpenOCR
export IMAGES_DIR=/absolute/path/to/OmniDocBench-v1.6/images
export COMPILE_CACHE=/absolute/path/to/the/same/warmed/recognition/cache
export LAYOUT_CACHE_ROOT=/absolute/path/to/the/same/warmed/layout/cache
export ASCEND_RT_VISIBLE_DEVICES=<one-free-device-from-0-to-3>
export TASKSET_CPUS=<the-same-known-good-64-cpu-mask>
bash 12_unirec_0_1b_inference/run_310p_k10_l4_all_w2_w4_background.sh
```

Do not use `nproc`. Do not apply `readlink -f` to `PYTHON_BIN`. The launcher
checks that the selected taskset exposes at least 64 CPUs. It prints the
absolute log path and `tail -f` command.

The runner executes W2 and then W4 sequentially. It does not overlap the lanes.
It reuses the ten graph caches built by W1. Setup/warmup remains outside the
measured prefill window.

## Required return

After `exit_code.txt` becomes `0`, return only:

- both `UNIREC_310P_K10_L4_ALL_SCALING_RESULT` lines
- `UNIREC_310P_K10_L4_ALL_SCALING_COMPARISON`
- absolute `RUN_ROOT`, `run.log`, and `w2_w4_scaling.json` paths
- peak HBM if it was observed externally

State plainly whether W2 and W4 exceed the corrected W1 reference of about
40 seconds / 3.19 pages/s. Cite the exact prior W1 clean `run_summary.json`,
its measured wall time, and its pages/s. Do not use a trace lane or an
approximate hard-coded value for this comparison.

If W4 fails or OOMs, preserve and report the completed W2 output plus the first
real W4 error. Do not retry with a different configuration.
