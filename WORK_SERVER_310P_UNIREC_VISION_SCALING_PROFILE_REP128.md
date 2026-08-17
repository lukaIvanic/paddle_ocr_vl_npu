# 310P UniRec vision scaling, profile, and representative-128 gate

Run the committed background runner once. It executes three ordered gates that
use the same UniRec vision implementation and graph contracts as the completed
910B2 controls.

Do not edit tracked files, create a branch, commit, or push. Do not use physical
NPU 5 or 6. Preserve the validated `python_nosym` path; do not apply
`readlink -f` to the executable because that resolves out of the virtual
environment.

## Questions answered

1. How does the 512x256 compiled graph scale at B1, B4, and B16 on 310P?
2. How much do all-45 grouped focal-depthwise weights plus TorchAir internal
   Conv/Linear weights help at each batch size?
3. At production B16, which kernel types and exact TransData signatures remain?
4. Does the isolated graph improvement reduce the real representative-128 W1/T1
   prefill time when layout is held at the accepted optimized configuration?

## Exact 910B2 controls

The original JSON artifacts are committed, not transcribed:

- `12_unirec_0_1b_inference/references/unirec_vision_512x256_batch_scaling_910b_20260817_native.json`
- `12_unirec_0_1b_inference/references/unirec_vision_512x256_batch_scaling_910b_20260817_optimized.json`

The 910B2 synchronized NPU-event medians were:

| Batch | Native | Optimized | Speedup | Native crops/s | Optimized crops/s |
|---:|---:|---:|---:|---:|---:|
| B1 | 8.240050 ms | 5.358880 ms | 1.5376x | 121.358 | 186.606 |
| B4 | 10.955730 ms | 8.069000 ms | 1.3578x | 365.106 | 495.724 |
| B16 | 19.465710 ms | 16.883830 ms | 1.1529x | 821.958 | 947.652 |

`optimized` means exactly:

- `--focal-depthwise-rewrite constant_grouped_all`: all 45 focal-depthwise
  weights are frozen/prepacked.
- `--weight-format torchair_internal`: persistent internal formats for the
  vision Conv/Linear weights.

These controls are not 310P performance predictions. They only establish the
identical graph contract, correctness checks, and comparison arithmetic.

## Three phases

### Phase 1: six-graph latency matrix

The runner measures native and optimized B1/B4/B16 with 2 warmups and 20
alternating synchronized NPU-event repeats. It saves the native compiled outputs
and reports exact/max/mean differences from the optimized outputs. Measurement
scope is encoder graph compute only: no H2D, preprocessing, layout, text
prefill, or decode.

### Phase 2: production-B16 profiles

The runner profiles only `512x256_b16`, once native and once optimized. Do not
profile B1 and B4. The matrix already measures their latency; B16 is the current
production graph. Compare at minimum:

- total device time and kernel count;
- TransData total time/count;
- MatMulV2, Gelu, Conv2D, LayerNorm/AddLayerNorm totals;
- exact top TransData shape/format signatures;
- cube utilization;
- whether the targeted focal-weight repacks are zero in the optimized profile.

### Phase 3: representative-128 candidate only

Reuse the completed optimized-layout/native-vision run as the baseline. Do not
rerun it. The candidate changes only vision to
`constant_grouped_all + torchair_internal`, retains W1/T1, layout B2 and
threshold 0.5, cross-KV 1320, self-KV 2048, and stops after prefill. It must
retain 2,485 crops, zero rejections, and the same real-source-token count as the
baseline.

## Resolve prior-run paths

Use the successful result from
`WORK_SERVER_310P_UNIREC_REPRESENTATIVE128_LAYOUT_OPTIMIZED_B2.md`:

- `REFERENCE_RUN_SUMMARY` is its
  `output/run_summary.json` (approximately 72 seconds / 1.777 pages/s and
  18.09 seconds layout).
- `LAYOUT_CACHE_ROOT` is the optimized B2 layout cache used by that run.
- `COMPILE_CACHE` is the warmed production recognition cache parent. The matrix,
  profiles, and representative candidate deliberately share it so missing B1/B4
  graphs compile once and B16/text-prefill graphs can be reused.

## Launch

Pull the commit containing this brief, then:

```bash
set -eo pipefail
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
git pull --ff-only origin main

source npu-setup
case ",${ASCEND_RT_VISIBLE_DEVICES:?}," in
  *,5,*|*,6,*) echo "Do not use physical NPU 5 or 6" >&2; exit 1 ;;
esac

export PYTHON_BIN="${PYTHON_BIN:?validated python_nosym executable}"
export MODEL="${MODEL:?OpenDoc unirec-0.1b model.pth directory}"
export LAYOUT_MODEL="${LAYOUT_MODEL:?PP-DocLayoutV2_safetensors directory}"
export OPENOCR_ROOT="${OPENOCR_ROOT:?matching OpenOCR checkout}"
export IMAGES_DIR="${IMAGES_DIR:?OmniDocBench v1.6 images directory}"
export COMPILE_CACHE="${COMPILE_CACHE:?warmed production recognition cache parent}"
export LAYOUT_CACHE_ROOT="${LAYOUT_CACHE_ROOT:?warmed optimized B2 layout cache}"
export REFERENCE_RUN_SUMMARY="${REFERENCE_RUN_SUMMARY:?successful 72-second native-vision baseline run_summary.json}"

bash 12_unirec_0_1b_inference/run_310p_vision_scaling_profile_rep128_background.sh
```

The launcher returns immediately. Send Luka the printed absolute `RUN_LOG` and
`TAIL_COMMAND` before monitoring. Compilation can be quiet for minutes; phase
markers and each completed batch row are printed to `run.log`.

## Completion report

Wait for `exit_code.txt`. Success requires exit zero and:

```text
UNIREC_310P_VISION_REP128_REFERENCE: PASS
UNIREC_310P_VISION_512X256_BATCH batch=1 ...
UNIREC_310P_VISION_512X256_BATCH batch=4 ...
UNIREC_310P_VISION_512X256_BATCH batch=16 ...
UNIREC_310P_VISION_REP128: PASS ...
UNIREC_310P_VISION_OUTPUT .../comparison.json
UNIREC_310P_VISION_WORKER_END status=0
```

Return:

1. The lines above and total process wall time.
2. `comparison.json` and both matrix JSON paths.
3. Both `profile_suite_summary.json` paths.
4. For B1/B4/B16, native and optimized median ms, crops/s, MPix/s,
   optimization speedup, and 310P slowdown versus 910B2.
5. The B16 native-versus-optimized kernel breakdown requested above.
6. Representative baseline/candidate prefill wall, recognition-prefill section,
   crop/token deltas, and speedups.
7. Any compilation, parity, internal-format, or NPU-memory warning.

Stop afterward. Do not run decode, OmniDocBench evaluation, full 1,651 pages,
or additional unrequested profiles.
