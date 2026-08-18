# 310P K20 vision-cache builder

## Goal

Compile the new K20 UniRec vision bucket set while the 910B2 practical A/B is
still running. This step builds caches only. Do not run the 128-page benchmark
or decode parity yet.

Commit `6deceef` preserves the exact graph identity of six validated aligned-K10
graphs and adds fourteen new direct-2D graph identities. On a work-server cache
that already passed the aligned-K10 run, the expected state is:

- six legacy graphs present and reused;
- fourteen new graphs missing and compiled once;
- no cache deletion, renaming, or fresh cache parent.

The builder logs registration, first-call begin/end, synchronized wall time,
source hash, slot, OM counts, and NPU memory for every graph.

## Constraints

- Pull only. Do not edit tracked files, commit, push, or create a branch.
- Use one free physical 310P device, 0-3. This server has four NPUs and no
  `npu-setup`.
- Use the venv's real `python_nosym` executable. The launcher deliberately does
  not resolve that symlink through `readlink -f`.
- Reuse the passed aligned-K10 `COMPILE_CACHE`. A fresh cache defeats this test.
- Do not delete a cache after a failure. Preserve the log and inventory.

## Prepare and launch

```bash
set -euo pipefail
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git pull --ff-only origin main
git merge-base --is-ancestor 6deceef HEAD

export PYTHON_BIN=/absolute/path/to/venv/bin/python_nosym
export MODEL=/absolute/path/to/unirec-0.1b
export LAYOUT_MODEL=/absolute/path/to/PP-DocLayoutV2_safetensors
export OPENOCR_ROOT=/absolute/path/to/OpenOCR
export IMAGES_DIR=/absolute/path/to/OmniDocBench/images
export COMPILE_CACHE=/absolute/path/to/the/passed/aligned-K10/cache/parent
export ASCEND_RT_VISIBLE_DEVICES=0
export CPUSET=0-63

bash 12_unirec_0_1b_inference/run_310p_k20_cache_builder_background.sh
```

The physical device is an example, not a reservation. Select a free device
from 0-3. The launcher prints absolute `RUN_ROOT`, `RUN_LOG`, and `PID`. Give
Luka the exact log path immediately so he can use `tail -f`.

## Monitor actively

Inspect progress every 15-30 seconds. Do not wait silently:

```bash
while [[ ! -s "$RUN_ROOT/exit_code.txt" ]]; do
  date -Ins
  ps -p "$(cat "$RUN_ROOT/pid.txt")" -o pid,etime,stat,%cpu,%mem --no-headers || true
  grep -E 'UNIREC_K20_CACHE_PHASE|UNIREC_K20_EXPECTED_COMPILES|warmup_graph_call_(begin|end)|HEARTBEAT|Traceback|ERROR' "$RUN_LOG" | tail -14
  sleep 15
done
```

The preflight should print:

```text
UNIREC_K20_EXPECTED_COMPILES legacy_missing=0 new_missing=14
```

`new_missing` may be smaller if part of K20 was already built. Any nonzero
`legacy_missing` is an invalid cache-parent selection; the runner stops before
compilation. Do not repair it by deleting or changing caches.

For each cold graph, report the last `warmup_graph_call_begin` immediately and
then its `warmup_graph_call_end` duration. This tells us whether the job is
making progress or stuck. A cold graph can take one or more minutes on 310P.
Reused graphs should be much faster.

## Completion report

Exit zero requires `UNIREC_K20_CACHE_BUILDER: PASS` and all twenty target graph
identities present. Paste:

```bash
cat "$RUN_ROOT/final_report.txt"
cat "$RUN_ROOT/process_wall_s.txt"
cat "$RUN_ROOT/cache_before.json"
cat "$RUN_ROOT/cache_after.json"
```

Also report the count of new OM paths:

```bash
comm -13 "$RUN_ROOT/om_before.txt" "$RUN_ROOT/om_after.txt" | tee "$RUN_ROOT/new_oms.txt"
wc -l "$RUN_ROOT/new_oms.txt"
```

Stop after this cache build. Wait for the 910B2 hot K10-vs-K20 result before
running the 310P 128-page performance and token-parity gate.
