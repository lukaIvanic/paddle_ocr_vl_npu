# 310P representative-128 optimized-layout B2 comparison

Run one candidate only. Do not rerun the completed eager-FP32 baseline. This
measures the current safe layout optimization stack inside the same
distribution-matched W1/T1 prefill workload used previously.

## Contract

- 128 deterministic distribution-matched pages.
- One worker and one recognition preprocessing thread.
- Layout B2, threshold 0.5.
- Candidate layout: TorchAir FP16 body, FP32 reading-order module,
  `torchair_internal` weights, `constant_grouped` depthwise weights,
  preformatted FrozenBN buffers, and direct softmax.
- Keep decomposed GridSample MSDA. Native MSDA is not part of this test.
- Keep recognition unchanged: native vision weights/depthwise, four-page
  lookahead, cross-KV 1320, self-KV 2048.
- Stop after prefill. Do not run decode, evaluation, profiling, or full 1,651
  pages.
- Use one physical 310P other than 5 or 6.

The exact 910B2 candidate reference is committed at
`12_unirec_0_1b_inference/references/unirec_representative128_layout_optimized_b2_910b_c3559e3.json`:

- Prefill wall: 32.925231 s, 3.887596 pages/s.
- Layout section: 3.339364 s over 64 B2 calls.
- Mean layout call: 52.177559 ms/B2.
- 2,485 crops and 180,336 real source tokens.

These are comparison data, not 310P pass thresholds.

## Locate the completed 310P baseline

The required baseline is the already completed clean lane from the prior
representative-128 W1/T1 cross-chip run. Find it without launching anything:

```bash
REPO="$(git rev-parse --show-toplevel)"
find "$REPO/tmp/12_unirec_0_1b_inference" \
  -path '*/representative128_w1t1_prefill_crosschip_*/clean/output/run_summary.json' \
  -type f -print
```

Select the completed run whose clean lane reported approximately 78.3 seconds,
1.635 pages/s, 64 layout calls, and about 24 seconds of aggregate layout time.
Do not use the trace lane because synchronized trace instrumentation changes
timing. Export the absolute clean `run_summary.json` path:

```bash
export REFERENCE_RUN_SUMMARY=/absolute/path/to/clean/output/run_summary.json
```

The runner validates the complete baseline contract before using it.

## Run

Pull the commit containing this brief. Do not edit tracked files, create a
branch, commit, or push. Preserve the `python_nosym` path: do not replace it
with `readlink -f`, because that resolves out of the validated virtual
environment.

```bash
set -eo pipefail
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
git pull --ff-only origin main

source npu-setup
case ",${ASCEND_RT_VISIBLE_DEVICES:?}," in
  *,5,*|*,6,*) echo "Do not use physical NPU 5 or 6" >&2; exit 1 ;;
esac

export PYTHON_BIN="${PYTHON_BIN:?use the real validated python_nosym executable}"
export MODEL="${MODEL:?OpenDoc unirec-0.1b model.pth directory}"
export LAYOUT_MODEL="${LAYOUT_MODEL:?PP-DocLayoutV2_safetensors directory}"
export OPENOCR_ROOT="${OPENOCR_ROOT:?matching OpenOCR checkout}"
export IMAGES_DIR="${IMAGES_DIR:?OmniDocBench v1.6 images directory}"
export COMPILE_CACHE="${COMPILE_CACHE:?warmed production recognition cache parent}"
export REFERENCE_RUN_SUMMARY="${REFERENCE_RUN_SUMMARY:?completed clean 310P baseline JSON}"
export LAYOUT_CACHE_ROOT="$REPO/.runtime_cache/12_unirec_0_1b_inference/310p_rep128_layout_optimized_b2_$(git rev-parse --short HEAD)_$(date +%Y%m%dT%H%M%S)"

bash 12_unirec_0_1b_inference/run_310p_representative128_layout_optimized_b2_background.sh
```

The launcher returns immediately. Send Luka the printed absolute `RUN_LOG` and
`TAIL_COMMAND` before monitoring. The B2 graph is expected to compile once and
may make setup several minutes long. The measured prefill begins only after
eight warmup pages and excludes setup/compilation.

Wait for `exit_code.txt`. Success requires exit code zero and these lines:

```text
UNIREC_310P_REP128_LAYOUT_REFERENCE: PASS
UNIREC_310P_REP128_LAYOUT_OPTIMIZED_B2: PASS
UNIREC_310P_REP128_LAYOUT_SPEEDUP:
UNIREC_310P_REP128_LAYOUT_VS_910B:
UNIREC_310P_REP128_LAYOUT_WORKER_END status=0
```

Return those lines, setup wall time, absolute run/cache paths, and any warning.
Also state whether crop and real-source-token counts changed versus the 310P
baseline. Stop afterward.
