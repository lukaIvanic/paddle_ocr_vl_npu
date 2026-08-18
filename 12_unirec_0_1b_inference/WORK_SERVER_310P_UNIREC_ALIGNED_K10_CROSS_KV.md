# 310P aligned-K10 six-crop cross-KV check

## Goal

Compare the already-written cross-KV tensors for the six aligned-K10 decode
mismatches against the token-exact middle lane. This is a CPU-only artifact
comparison. Do not run inference, load an NPU model, compile a graph, or rerun
prefill/decode.

The six request IDs are:

```text
page_000033_crop_0001
page_000037_crop_0000
page_000046_crop_0000
page_000047_crop_0000
page_000117_crop_0000
page_000119_crop_0002
```

## Run

Pull this commit, then export:

```bash
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git pull --ff-only origin main

export PYTHON_BIN=/absolute/path/to/the/validated/venv/bin/python_nosym
export FACTOR_ROOT=/absolute/path/to/the/completed/factorization/RUN_ROOT
export ALIGNED_ROOT=/absolute/path/to/the/failed/aligned-K10/RUN_ROOT
```

The required artifacts are:

```text
$FACTOR_ROOT/prefill_production_buckets_optimized_weights
$ALIGNED_ROOT/hot_prefill
```

Run:

```bash
bash 12_unirec_0_1b_inference/run_310p_aligned_k10_cross_kv_compare.sh
```

This command must not initialize `torch_npu`. Expected wall time is seconds,
not minutes. If it exceeds 30 seconds, stop it and report the process state and
the sizes of the two `cross_kv.bin` files.

## Report and stop

Paste the complete output plus:

```bash
cat "$ALIGNED_ROOT/six_cross_kv_comparison.json"
```

Do not run an eager or compiled NPU probe yet. The next probe will contain three
same-process lanes: native unpadded raw eager, padded raw eager, and padded
TorchAir compiled. Raw eager alone cannot determine whether TorchAir causes the
failure.
