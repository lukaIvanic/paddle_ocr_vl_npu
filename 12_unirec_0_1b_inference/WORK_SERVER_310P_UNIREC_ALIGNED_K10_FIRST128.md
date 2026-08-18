# 310P corrected-global-context K10 first-128 gate

## Goal

Validate the direct-2D global-context fix on the first 128 OmniDocBench pages.
The former compiled `1024x704_b1` and `1024x1408_b1` graphs corrupted cross-KV
on 310P because TorchAir mis-lowered the two-stage width-then-height reduction.
The corrected graphs use a direct masked mean in unambiguous `[N,C]` form,
apply GELU there, and expand only for the final broadcast.

Verified 910B2 control at commit `71329c2`, physical NPU 7:

- corrected `1024x704_b1`: finite, 8.22 ms;
- corrected `1024x1408_b1`: finite, 12.35 ms;
- production first-128: 128 pages, 957 crops, zero rejections;
- B128 decode parity: **957/957 token-exact**, 957/957 length-exact;
- identical generated-token sum 40,917; no new runaway output.

This 310P gate separates one-time compilation from hot execution:

1. W1 compiles only the missing corrected graphs. The eight clean legacy graphs
   retain their exact old source hash and cache identity. At most two corrected
   graphs may be missing.
2. W4 runs the full first-128 hot prefill. It must create no OM.
3. The unchanged B128 decoder replays all 957 crops and requires token-exact
   agreement with the canonical 90.13 run.

## Constraints

- Pull only. Do not edit tracked files, commit, push, or create a branch.
- Use one free physical 310P device, 0-3. This server has four NPUs and no
  `npu-setup`.
- Use the venv's real `python_nosym` executable. The launcher intentionally does
  not call `readlink -f` on `PYTHON_BIN`.
- `nproc` returning 1 is not authoritative. The runner uses `taskset` and
  verifies its actual affinity.
- Do not delete or rename any cache. A cache miss is evidence.

## Prepare

```bash
set -euo pipefail
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git pull --ff-only origin main
git rev-parse --short HEAD
git merge-base --is-ancestor c2d40b9 HEAD
```

Export the same validated paths used by the passed canonical/factorization run:

```bash
export PYTHON_BIN=/absolute/path/to/venv/bin/python_nosym
export MODEL=/absolute/path/to/unirec-0.1b
export LAYOUT_MODEL=/absolute/path/to/PP-DocLayoutV2_safetensors
export OPENOCR_ROOT=/absolute/path/to/OpenOCR
export IMAGES_DIR=/absolute/path/to/OmniDocBench/images
export COMPILE_CACHE=/absolute/path/to/existing/production/cache/parent
export CANONICAL_TRACE=/absolute/path/to/canonical/90.13/recognition_trace.jsonl
export UNIREC_PRODUCTION_DECODE_CACHE_PARENT_OVERRIDE=/absolute/path/to/passed/decode/cache/parent
export ASCEND_RT_VISIBLE_DEVICES=0
export CPUSET=0-63
```

The device number is an example, not a reservation. Select a free device from
0-3.

## Launch and monitor

```bash
bash 12_unirec_0_1b_inference/run_310p_aligned_k10_first128_gate_background.sh
```

The launcher immediately prints absolute `RUN_ROOT`, `RUN_LOG`, and `PID`.
Give Luka the `RUN_LOG` path so he can run:

```bash
tail -f "$RUN_LOG"
```

The work agent must also inspect progress every 15-30 seconds. Do not wait
silently. This compact command shows phase boundaries and the latest graph:

```bash
while [[ ! -s "$RUN_ROOT/exit_code.txt" ]]; do
  date -Ins
  ps -p "$(cat "$RUN_ROOT/pid.txt")" -o pid,etime,stat,%cpu,%mem --no-headers || true
  grep -E 'UNIREC_ALIGNED_K10_PHASE|UNIREC_ALIGNED_K10_EXPECTED_COMPILES|warmup_graph_call_(begin|end)|HEARTBEAT|Traceback|ERROR' "$RUN_LOG" | tail -12
  sleep 15
done
```

Expected cold behavior:

- `UNIREC_ALIGNED_K10_EXPECTED_COMPILES count=2` on a server that has not yet
  compiled this fix; zero or one is also valid if a corrected graph already
  exists;
- only `1024x704_b1` and `1024x1408_b1` may be missing;
- compilation may take roughly 1-2 minutes per new shape on 310P. Report the
  measured time instead of waiting without a progress check.

If any legacy graph is missing or the missing count exceeds two, the runner
stops before NPU inference. Paste
`cache_before.json`; do not delete caches or let an unexpectedly cold ten-graph
run consume time.

## Completion

Exit zero requires all of these:

```text
UNIREC_ALIGNED_K10_HOT_OM_INVENTORY_UNCHANGED
UNIREC_FLAT_GLOBAL_K10_FIRST128: PASS
UNIREC_ALIGNED_K10_PARITY exact=957/957 ... mismatches=0
```

Paste back:

```bash
cat "$RUN_ROOT/final_report.txt"
cat "$RUN_ROOT/cache_before.json"
cat "$RUN_ROOT/cache_after_builder.json"
cat "$RUN_ROOT/process_wall_s.txt"
```

Also paste any `hot_om.diff`. A non-empty hot diff means the cache-slot fix did
not work and invalidates the hot timing; do not rerun automatically.
