# NPU Hot-Swap Bottleneck Matrix

Run this from the work/NPU lane after pulling the latest `main`:

```sh
cd /home/lukaiv/paddle_ocr_vl_npu/04_batched_fixed_cohort_decode
bash run_npu_hotswap_bottleneck_matrix.sh
```

The runner executes the fixed-cohort warmup/reference and the hot-swap
`num-items=8,9,16,32,100` matrix with the same batch size, cache length,
TorchAir cache directory, dtype, EOS mode, and step timing settings. It writes
one JSON file per run under `outputs/hotswap_bottleneck_matrix/` and prints the
same JSON to stdout.

Environment overrides are supported without editing tracked files:

```sh
PYTHON_BIN=/root/miniconda3/envs/paddle_ocr_vl_py310/bin/python \
MODEL=/home/lukaiv/models/paddle_ocr_0_9b_v_1_6 \
DEVICE=npu:0 \
CACHE_LENGTH=1269 \
bash run_npu_hotswap_bottleneck_matrix.sh
```

If `CACHE_LENGTH=1269` is too small, stop and report the exact error. Then run
one discovery command without `CACHE_LENGTH` manually, note the selected
`cache_length` from the JSON, and rerun the whole matrix with that explicit
larger value.

Read the matrix this way:

- `02_fixed_warm_reference` is the speed baseline.
- `03_hotswap_no_replacement_8` isolates scheduler overhead with no real
  replacement.
- `04_hotswap_one_extra_9` isolates the smallest nontrivial replacement case.
- `05_hotswap_one_wave_16` and `06_hotswap_several_waves_32` show replacement
  scaling.
- `07_hotswap_full_100` is the real workload.

For each JSON, paste back `correctness`, `tok_per_s`, `speed_debug`,
`loop.step_timing_summary`, `timing_s`, and `timing_accounting`. Do not write
inline parsing scripts.
