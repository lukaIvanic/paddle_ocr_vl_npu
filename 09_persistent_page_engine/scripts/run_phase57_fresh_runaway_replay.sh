#!/usr/bin/env bash
# Replay four representative Phase-57 runaways in fresh one-page processes.
set -euo pipefail

REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
source npu-setup

PYTHON_BIN=/usr/local/python3.12.13/bin/python
E2E="$REPO/09_persistent_page_engine/scripts/run_omnidocbench.py"
REFERENCE="$REPO/tmp/09_persistent_page_engine/910b_phase57_authority_898ced7/phase57_910b_authority.gdatlas.zip"
MODEL=/home/lukaiv/models/PaddleOCR-VL-1.6
LAYOUT_MODEL=/home/lukaiv/models/PP-DocLayoutV3_safetensors
DATASET_JSON=/home/lukaiv/datasets/OmniDocBench/OmniDocBench.json
IMAGES_DIR=/home/lukaiv/datasets/OmniDocBench/images

DECODE_CACHE=.runtime_cache/310p_phase57_decode_b64_k4096_pse
VISION_CACHE=.runtime_cache/310p_phase52_v16_vision_4352_nz
BATCHED_CACHE=.runtime_cache/310p_phase57_vision_batched_unused
TEXT_CACHE=.runtime_cache/310p_phase52_v16_text
PACKED_CACHE=.runtime_cache/310p_phase52_v16_text_packed

for path in "$REFERENCE" "$MODEL/model.safetensors" "$DATASET_JSON"; do
  test -s "$path"
done
for path in "$LAYOUT_MODEL" "$IMAGES_DIR" "$DECODE_CACHE" "$VISION_CACHE" "$TEXT_CACHE" "$PACKED_CACHE"; do
  test -e "$path"
done

FULL="$($PYTHON_BIN - <<'PY'
import json
from pathlib import Path
roots=[]
for path in Path("tmp/09_persistent_page_engine").glob("310p_phase57_cap4096_b64_pse_*/full"):
    summary=path/"output/run_summary.json"
    if summary.is_file():
        data=json.loads(summary.read_text())
        if data.get("result_count")==data.get("prediction_count")==1651:
            roots.append(path)
if not roots:
    raise SystemExit("no completed Phase-57 full run")
print(max(roots,key=lambda p:p.stat().st_mtime))
PY
)"
test -s "$FULL/output/recognition_trace.jsonl"

SHORT="$(git rev-parse --short HEAD)"
ROOT="${FULL%/full}/fresh_runaway_replay_${SHORT}"
test ! -e "$ROOT"
mkdir -p "$ROOT"

"$PYTHON_BIN" - "$REFERENCE" "$FULL/output" "$ROOT/cases.json" <<'PY'
import json, sys, unicodedata, re, zipfile
from pathlib import Path
bundle, candidate_root, output = map(Path, sys.argv[1:])
with zipfile.ZipFile(bundle) as archive, archive.open("recognition_trace.jsonl") as handle:
    reference=[json.loads(line) for line in handle if line.strip()]
candidate=[json.loads(line) for line in (candidate_root/"recognition_trace.jsonl").read_text().splitlines() if line.strip()]
summary=json.loads((candidate_root/"run_summary.json").read_text())
images=[str(value) for value in summary["images"]]
page_index={name:index for index,name in enumerate(images)}
page_request_counts={}
for row in candidate:
    page_request_counts[str(row["source_image_name"])]=page_request_counts.get(str(row["source_image_name"]),0)+1
def key(row): return str(row["source_image_name"]),int(row["block_index"])
ref={key(row):row for row in reference}
cand={key(row):row for row in candidate}
fields=("prompt","input_tokens","projected_image_tokens","crop_size","min_pixels","max_pixels")
rows=[]
for stable in sorted(ref.keys()&cand.keys()):
    left,right=ref[stable],cand[stable]
    if str(right.get("stop_reason")) not in {"kv_cache_full","repetition"}:
        continue
    if any(left.get(field)!=right.get(field) for field in fields):
        continue
    left_tokens=list(left.get("token_ids") or ())
    right_tokens=list(right.get("token_ids") or ())
    delta=len(right_tokens)-len(left_tokens)
    if delta < 128:
        continue
    if page_request_counts[stable[0]] > 64:
        continue
    rows.append({
        "source_image_name":stable[0],"page_index":page_index[stable[0]],
        "block_index":stable[1],"label":right.get("label"),
        "candidate_stop":right.get("stop_reason"),"reference_stop":left.get("stop_reason"),
        "reference_tokens":len(left_tokens),"candidate_tokens":len(right_tokens),
        "token_delta":delta,"page_request_count":page_request_counts[stable[0]],
    })
selected=[]
for stop in ("kv_cache_full","repetition"):
    seen_pages=set()
    candidates=sorted((row for row in rows if row["candidate_stop"]==stop),key=lambda row:(row["token_delta"],row["candidate_tokens"]),reverse=True)
    for row in candidates:
        if row["source_image_name"] in seen_pages:
            continue
        selected.append(row);seen_pages.add(row["source_image_name"])
        if len(seen_pages)==2: break
    if len(seen_pages)!=2:
        raise SystemExit(f"could not select two {stop} cases")
output.write_text(json.dumps({"selection":"two largest metadata-exact unique-page cases per stop reason","cases":selected},indent=2,ensure_ascii=False)+"\n")
print("PHASE57_FRESH_REPLAY selection",json.dumps(selected,separators=(",",":"),ensure_ascii=False),flush=True)
PY

PRODUCTION_ARGS=(
  "$E2E"
  --dataset-json "$DATASET_JSON" --images-dir "$IMAGES_DIR"
  --layout-model "$LAYOUT_MODEL" --recognizer-model "$MODEL"
  --batch-size 64 --cache-length 4096 --max-new-tokens 4096
  --preprocessor-min-pixels 28224 --preprocessor-max-pixels 802816
  --text-crop-scale 0.5
  --decode-backend torchair
  --decode-optimization combined_apply_pse_sentinel
  --torchair-cache-dir "$DECODE_CACHE"
  --vision-backend torchair --vision-attention prompt_flash_attention
  --vision-buckets 256,384,512,640,768,1408,1920,2048,2944,4096
  --vision-torchair-cache-dir "$VISION_CACHE"
  --vision-batched-cache-dir "$BATCHED_CACHE"
  --vision-promptfa-align-128
  --vision-mlp-intermediate-size 4352
  --vision-linear-weight-format fractal_nz
  --vision-padding bucket --vision-packing greedy
  --vision-pack-target 768 --vision-router-lookahead 32
  --text-buckets 1152
  --text-packing production_group
  --text-pack-buckets 128,256,384,512,768,1024
  --text-pack-max-members 32
  --text-torchair-cache-dir "$TEXT_CACHE"
  --text-packed-cache-dir "$PACKED_CACHE"
  --layout-device npu --no-layout-graph-capture
  --preprocess-all-pages-first --layout-workers 8 --no-timeline
  --recognition-input-fingerprints
)

cache_inventory() {
  for cache in "$DECODE_CACHE" "$VISION_CACHE" "$BATCHED_CACHE" "$TEXT_CACHE" "$PACKED_CACHE"; do
    printf '%s\t%s\t%s\n' "$cache" "$(find "$cache" -type f 2>/dev/null | wc -l)" "$(du -sb "$cache" 2>/dev/null | cut -f1)"
  done
}
cache_inventory >"$ROOT/cache_before.tsv"

while IFS=$'\t' read -r case_index page_index block_index; do
  lane="$ROOT/case_${case_index}"
  mkdir -p "$lane"
  printf '%q ' "$PYTHON_BIN" "${PRODUCTION_ARGS[@]}" --offset "$page_index" --limit 1 --output-dir "$lane/output" >"$lane/command.sh"
  printf '\n' >>"$lane/command.sh"
  printf 'PHASE57_FRESH_REPLAY case=%s/4 page_index=%s block=%s state=begin\n' "$((case_index+1))" "$page_index" "$block_index" | tee -a "$ROOT/progress.log"
  SECONDS=0
  set +e
  set -o pipefail
  PYTHONUNBUFFERED=1 timeout --signal=TERM --kill-after=30s 600 \
    "$PYTHON_BIN" "${PRODUCTION_ARGS[@]}" \
    --offset "$page_index" --limit 1 --output-dir "$lane/output" \
    2>&1 | tee "$lane/run.log"
  ec="${PIPESTATUS[0]}"
  set -e
  printf '%s\n' "$ec" >"$lane/exit_code.txt"
  printf '%s\n' "$SECONDS" >"$lane/wall_s.txt"
  test "$ec" -eq 0
  printf 'PHASE57_FRESH_REPLAY case=%s/4 state=pass wall_s=%s\n' "$((case_index+1))" "$SECONDS" | tee -a "$ROOT/progress.log"
done < <("$PYTHON_BIN" - "$ROOT/cases.json" <<'PY'
import json,sys
for index,row in enumerate(json.load(open(sys.argv[1]))["cases"]):
    print(index,row["page_index"],row["block_index"],sep="\t")
PY
)

cache_inventory >"$ROOT/cache_after.tsv"
diff -u "$ROOT/cache_before.tsv" "$ROOT/cache_after.tsv" >"$ROOT/cache_diff.txt" || true
test ! -s "$ROOT/cache_diff.txt"

"$PYTHON_BIN" - "$REFERENCE" "$FULL/output" "$ROOT" <<'PY' | tee "$ROOT/final_sentence.txt"
import json,sys,zipfile
from pathlib import Path
bundle,full_output,root=map(Path,sys.argv[1:])
selection=json.loads((root/"cases.json").read_text())
with zipfile.ZipFile(bundle) as archive,archive.open("recognition_trace.jsonl") as handle:
    reference=[json.loads(line) for line in handle if line.strip()]
full=[json.loads(line) for line in (full_output/"recognition_trace.jsonl").read_text().splitlines() if line.strip()]
def key(row): return str(row["source_image_name"]),int(row["block_index"])
ref={key(row):row for row in reference};production={key(row):row for row in full}
results=[]
for index,case in enumerate(selection["cases"]):
    rows=[json.loads(line) for line in (root/f"case_{index}/output/recognition_trace.jsonl").read_text().splitlines() if line.strip()]
    stable=(case["source_image_name"],int(case["block_index"]))
    isolated={key(row):row for row in rows}.get(stable)
    if isolated is None: raise SystemExit(f"target missing in isolated run: {stable}")
    prod=production[stable];authority=ref[stable]
    isolated_tokens=list(isolated.get("token_ids") or ())
    prod_tokens=list(prod.get("token_ids") or ())
    ref_tokens=list(authority.get("token_ids") or ())
    runaway=isolated.get("stop_reason") in {"kv_cache_full","repetition"} and len(isolated_tokens)>=len(ref_tokens)+128
    if isolated_tokens==prod_tokens and isolated.get("stop_reason")==prod.get("stop_reason"):
        classification="FRESH_REPRODUCED_EXACT"
    elif runaway:
        classification="FRESH_RUNAWAY_DIFFERENT_TRAJECTORY"
    elif isolated.get("stop_reason")=="eos" and len(isolated_tokens)<len(prod_tokens):
        classification="PRODUCTION_CONTEXT_DEPENDENT_CLEAN_IN_FRESH_PROCESS"
    else:
        classification="FRESH_DIFFERENT_NON_RUNAWAY"
    results.append({
        **case,"classification":classification,
        "isolated_tokens":len(isolated_tokens),"isolated_stop":isolated.get("stop_reason"),
        "isolated_equals_production_tokens":isolated_tokens==prod_tokens,
        "isolated_equals_reference_tokens":isolated_tokens==ref_tokens,
        "isolated_input_fingerprints":isolated.get("input_fingerprints"),
        "isolated_vision_route":isolated.get("vision"),
        "isolated_text_route":isolated.get("text_prefill"),
    })
report={"classification_counts":{},"cases":results}
for row in results: report["classification_counts"][row["classification"]]=report["classification_counts"].get(row["classification"],0)+1
(root/"report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
print("PHASE57_FRESH_RUNAWAY_REPLAY PASS classifications="+json.dumps(report["classification_counts"],separators=(",",":"))+f" report={root/'report.json'}")
PY
