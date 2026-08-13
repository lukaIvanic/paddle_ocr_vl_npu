# 310P UniRec compiled grouped-FZ B1 gate

## Goal

Test the exact production-shaped `960x64`, batch-1 UniRec vision encoder with
TorchAir. The optimized graph passes only if:

- its compiled output is bit-exact to the native compiled graph;
- all 22 stage-2/3 5x5 and 7x7 focal Conv2D calls remain;
- logical-weight to `FRACTAL_Z:1` TransData calls are zero; and
- `FRACTAL_Z:1` to grouped-FZ TransData calls are zero.

Do not edit tracked files, commit, push, or create a branch on the work server.
Use one free physical NPU other than 5 or 6. Record the physical NPU and runtime
versions in the report.

## 910B2 reference

Validated on commit `bb14e2b`, physical Ascend 910B2 NPU 7, CANN 9.0.0,
torch/torch_npu 2.10.0:

| lane | warm median | logical to FZ1 | FZ1 to grouped FZ | Conv2D |
|---|---:|---:|---:|---:|
| native compiled | 6.44196 ms | 22 | 22 | 22 |
| grouped-FZ compiled | 4.55282 ms | 0 | 0 | 22 |

The grouped graph was bit-exact to native compiled output (`max_abs=0`,
`mean_abs=0`). A second process reused the same one-OM cache. Its first compiled
call, including cache load/graph setup, was 15.281 s. The steady graph saved
1.88914 ms, or 1.415x.

Directly binding the eager internal-format Parameters is not the solution.
TorchAir discarded those descriptors, restored both 22-count repack families,
and produced incorrect output (`max_abs=3.9375`). The passing lane uses the
group-aware GE Conv2D input descriptor in the existing
`converter_grouped_fz` path.

## Run

Resolve the existing model path instead of assuming a 910B path. The directory
must contain UniRec `config.json` and model weights.

```bash
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git fetch origin main
git checkout --detach origin/main

export PYTHON_BIN="$(command -v python3)"
export MODEL=/absolute/path/to/the/existing/unirec-model-directory
export ASCEND_RT_VISIBLE_DEVICES=0  # one verified-free physical NPU, not 5 or 6

bash 12_unirec_0_1b_inference/run_compiled_grouped_fz_b1_background.sh
```

The launcher prints the absolute `RUN_ROOT`, `RUN_LOG`, and `tail -f` command.
It runs native, grouped, and a grouped warm-cache repeat in one background job.

## Report

Paste these items only. Do not rerun profiling merely to format the report.

1. Commit, physical NPU, CANN version, torch version, and torch_npu version.
2. Both `UNIREC_COMPILED_GROUPED_FZ_VISION_B1` lines.
3. Both `UNIREC_COMPILED_GROUPED_FZ_VISION_KERNELS` lines.
4. From grouped `result.json`:
   - `status`
   - `compiled_first_call_s`
   - `control_after.device_event`
   - `parity.grouped_compiled_vs_native_compiled`
   - aggregate and per-signature `target_operations`
5. OM count under `CACHE_ROOT`, before and after one optional warm-cache repeat.

Stop if compilation fails, output is not exact, either removed TransData family
is nonzero, or the physical Conv2D count is not 22. Preserve the full `run.log`
and report the first causal error. Do not debug or change tracked source.
