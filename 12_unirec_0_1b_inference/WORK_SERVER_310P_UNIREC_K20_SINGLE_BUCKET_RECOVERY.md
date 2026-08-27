# 310P K20 single-bucket cache recovery

## Goal

Restore the accepted K20 cache to an unambiguous 20-of-20 state, then resume
the low-memory full-1651 handoff. The reported `128x1408_b1` graph is probably
not missing. The cache contains the accepted Aug 19 graph identity plus a stray
Aug 25 identity created by failed commit `6e139ea`. The locator rejects the two
matching `compiled_module` paths as ambiguous.

Do not rebuild all K20 graphs. First quarantine only the confirmed stray
identity. If the accepted identity is incomplete, permit exactly one targeted
`128x1408_b1` rebuild. Stop if more than one K20 graph is missing.

## Constraints

- Pull only. Do not edit tracked files, create a branch, commit, or push.
- Use the validated `python_nosym` executable. Do not apply `readlink -f` to
  `PYTHON_BIN`.
- Select one free physical NPU from 0 through 3 only if a rebuild is required.
- Preserve the accepted `58bd81c1...` identity.
- Move the stray identity to a quarantine directory outside `COMPILE_CACHE`.
  Do not delete it.
- Do not touch the other 19 K20 buckets or the compiled layout cache.
- A rebuild is allowed only after the quarantine step and only when the locator
  still reports `128x1408_b1` as the sole missing bucket.
- Hard stop if the builder reports more than one missing graph or creates more
  than one new graph identity.

## Prepare

Reuse the environment recovered by
`WORK_SERVER_310P_UNIREC_LOWMEM_FULL1651_HBM.md` through the point where it
exports `PYTHON_BIN`, `COMPILE_CACHE`, and `LAYOUT_CACHE_ROOT`.

```bash
set -euo pipefail
WORK_SERVER_REPO="$(git rev-parse --show-toplevel)"
cd "$WORK_SERVER_REPO"
git pull --ff-only origin main
git rev-parse HEAD

test "$(basename "$PYTHON_BIN")" = python_nosym
test -x "$PYTHON_BIN"
test -d "$COMPILE_CACHE"
test -d "$LAYOUT_CACHE_ROOT"

RECOVERY_ROOT="$WORK_SERVER_REPO/tmp/12_unirec_0_1b_inference/310p_k20_single_bucket_recovery_$(git rev-parse --short=12 HEAD)_$(date +%Y%m%dT%H%M%S)"
mkdir -p "$RECOVERY_ROOT"
export RECOVERY_ROOT

set +e
"$PYTHON_BIN" 12_unirec_0_1b_inference/locate_unirec_production_caches.py \
  --search-root "$COMPILE_CACHE" \
  --search-root "$LAYOUT_CACHE_ROOT" \
  --output "$RECOVERY_ROOT/locator_before.json" \
  | tee "$RECOVERY_ROOT/locator_before.log"
LOCATOR_BEFORE_STATUS="${PIPESTATUS[0]}"
set -e
test "$LOCATOR_BEFORE_STATUS" = 2
grep -q 'buckets=19/20 missing=128x1408_b1' "$RECOVERY_ROOT/locator_before.log"
```

## Verify and quarantine the stray identity

The checks below require exactly one matching bucket directory, one accepted
identity, and one stray identity. They record whether the accepted graph is
complete before anything moves.

```bash
mapfile -t BUCKET_DIRS < <(
  find "$COMPILE_CACHE" -maxdepth 1 -type d \
    -name 'vision_full_bucket_128x1408_b1_float16_src38b70231a30c_dwconstant_grouped_all*wtorchair_internal*' \
    -print
)
test "${#BUCKET_DIRS[@]}" = 1
BUCKET_DIR="${BUCKET_DIRS[0]}"

mapfile -t ACCEPTED_DIRS < <(
  find "$BUCKET_DIR" -maxdepth 1 -type d -name '*_gecache_58bd81c1*' -print
)
mapfile -t STRAY_DIRS < <(
  find "$BUCKET_DIR" -maxdepth 1 -type d -name '*_gecache_84df4019*' -print
)
test "${#ACCEPTED_DIRS[@]}" = 1
test "${#STRAY_DIRS[@]}" = 1
ACCEPTED_DIR="${ACCEPTED_DIRS[0]}"
STRAY_DIR="${STRAY_DIRS[0]}"

ACCEPTED_METHOD_DIR="$ACCEPTED_DIR/_forward_flat_bucket_slot_6"
if [[ -d "$ACCEPTED_METHOD_DIR" ]]; then
  ACCEPTED_MODULE_COUNT="$(
    find "$ACCEPTED_METHOD_DIR" -maxdepth 1 -type f -name compiled_module \
      | wc -l | tr -d '[:space:]'
  )"
  ACCEPTED_OM_COUNT="$(
    find "$ACCEPTED_METHOD_DIR" -maxdepth 1 -type f -name '*.om' \
      | wc -l | tr -d '[:space:]'
  )"
else
  ACCEPTED_MODULE_COUNT=0
  ACCEPTED_OM_COUNT=0
fi
case "$ACCEPTED_MODULE_COUNT:$ACCEPTED_OM_COUNT" in
  1:0) ACCEPTED_COMPLETE=0 ;;
  1:*) ACCEPTED_COMPLETE=1 ;;
  0:*) ACCEPTED_COMPLETE=0 ;;
  *) printf 'Unexpected accepted identity counts: modules=%s oms=%s\n' \
       "$ACCEPTED_MODULE_COUNT" "$ACCEPTED_OM_COUNT" >&2; exit 1 ;;
esac
export ACCEPTED_COMPLETE

{
  printf 'bucket_dir=%s\naccepted_dir=%s\nstray_dir=%s\naccepted_complete=%s\naccepted_modules=%s\naccepted_oms=%s\n' \
    "$BUCKET_DIR" "$ACCEPTED_DIR" "$STRAY_DIR" "$ACCEPTED_COMPLETE" \
    "$ACCEPTED_MODULE_COUNT" "$ACCEPTED_OM_COUNT"
  find "$ACCEPTED_DIR" "$STRAY_DIR" -maxdepth 2 \
    \( -name compiled_module -o -name '*.om' \) \
    -printf '%p %s %TY-%Tm-%TdT%TH:%TM:%TS\n' | sort
} | tee "$RECOVERY_ROOT/identity_inventory_before.txt"

QUARANTINE_ROOT="$(dirname "$COMPILE_CACHE")/quarantine_310p_k20_128x1408_$(date +%Y%m%dT%H%M%S)"
mkdir -p "$QUARANTINE_ROOT"
mv "$STRAY_DIR" "$QUARANTINE_ROOT/"
printf 'quarantine_root=%s\n' "$QUARANTINE_ROOT" \
  | tee "$RECOVERY_ROOT/quarantine.txt"
```

This move is recoverable. The quarantined identity remains on disk but is no
longer under the K20 locator root.

## Check for 20-of-20 without compilation

```bash
set +e
"$PYTHON_BIN" 12_unirec_0_1b_inference/locate_unirec_production_caches.py \
  --search-root "$COMPILE_CACHE" \
  --search-root "$LAYOUT_CACHE_ROOT" \
  --output "$RECOVERY_ROOT/locator_after.json" \
  | tee "$RECOVERY_ROOT/locator_after.log"
LOCATOR_AFTER_STATUS="${PIPESTATUS[0]}"
set -e

if [[ "$LOCATOR_AFTER_STATUS" = 0 ]]; then
  test "$ACCEPTED_COMPLETE" = 1
  grep -q 'buckets=20/20 missing=none' "$RECOVERY_ROOT/locator_after.log"
  grep -q '^UNIREC_K20_COMPILE_CACHE=' "$RECOVERY_ROOT/locator_after.log"
  grep -q 'UNIREC_PRODUCTION_CACHE_LOCATOR: PASS' "$RECOVERY_ROOT/locator_after.log"
  printf 'UNIREC_310P_K20_SINGLE_BUCKET_RECOVERY: PASS compilation=0\n' \
    | tee "$RECOVERY_ROOT/final_report.txt"
elif [[ "$LOCATOR_AFTER_STATUS" = 2 ]]; then
  test "$ACCEPTED_COMPLETE" = 0
  grep -q 'buckets=19/20 missing=128x1408_b1' "$RECOVERY_ROOT/locator_after.log"
  printf 'UNIREC_310P_K20_SINGLE_BUCKET_RECOVERY: NEEDS_ONE_GRAPH_BUILD\n' \
    | tee "$RECOVERY_ROOT/final_report.txt"
else
  printf 'Unexpected locator status after quarantine: %s\n' \
    "$LOCATOR_AFTER_STATUS" >&2
  exit 1
fi
```

If this reports `PASS compilation=0`, do not run the cache builder. Resume
`WORK_SERVER_310P_UNIREC_LOWMEM_FULL1651_HBM.md` from `Launch in the
background`. The existing K20 graph did not need recompilation.

## Fallback if the accepted graph is incomplete

Do not use this fallback for any other condition. Run it only when the previous
section reports `NEEDS_ONE_GRAPH_BUILD`, which means the accepted identity was
incomplete and the post-quarantine locator reports only `128x1408_b1` missing.
Run the committed K20 builder once with the same `COMPILE_CACHE`.

Before launch, export the model, layout, OpenOCR, images, device, and CPU values
required by `WORK_SERVER_310P_UNIREC_K20_CACHE_BUILDER.md`. Then:

```bash
export ALLOW_FULL_K20_REBUILD=0
LAUNCH_OUTPUT="$(
  bash 12_unirec_0_1b_inference/run_310p_k20_cache_builder_background.sh
)"
printf '%s\n' "$LAUNCH_OUTPUT"
BUILD_ROOT="$(printf '%s\n' "$LAUNCH_OUTPUT" | sed -n 's/^RUN_ROOT=//p')"
BUILD_LOG="$(printf '%s\n' "$LAUNCH_OUTPUT" | sed -n 's/^RUN_LOG=//p')"
test -n "$BUILD_ROOT" && test -n "$BUILD_LOG"
printf 'For Luka: tail -f %q\n' "$BUILD_LOG"
```

The preflight must print exactly:

```text
UNIREC_K20_EXPECTED_COMPILES legacy_missing=0 new_missing=1
```

Stop immediately if either number differs. Monitor the single cold call. At
completion, require `UNIREC_K20_CACHE_BUILDER: PASS`, then rerun the locator and
require 20-of-20. Compare `om_before.txt` and `om_after.txt`. There may be at
most one new graph identity. Do not repeat the builder after a failure.

## Report

Report:

- commit and absolute `RECOVERY_ROOT`;
- accepted and stray identity paths;
- quarantine path;
- locator before and after summaries;
- whether compilation occurred;
- if compilation occurred, the builder root, exact missing counts, cold-call
  duration, and new OM paths;
- confirmation that no other K20 bucket and no layout cache changed.

Then continue the low-memory full-1651 handoff. Do not launch another baseline.
