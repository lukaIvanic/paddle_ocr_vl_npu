# Work-server 310P eager validation handoff

This is the second one-way execution brief for the AI agent operating on
Luka's work server. Read `AGENTS.md` and
`WORK_SERVER_310P_EAGER_SMOKE.md` first. The first five-phase smoke has already
been reported as passing.

The work-server checkout remains a pull-only NPU validation lane. Do not edit
tracked files, commit, push, or create branches.

## Objective

Answer two questions that the tiny first smoke did not:

1. Is the complete OmniDocBench v1.6 dataset required by Experiment 09 present
   and readable?
2. Can the eager Experiment 09 diagnostic pipeline process a small,
   shape-diverse cross-page workload with all detected regions, rather than
   only two regions from one page?

This remains a compatibility test, not a throughput benchmark.

## Constraints

- Discover and reuse the exact Python, NPU activation, recognizer model, layout
  model, dataset JSON, and images directory that passed the first handoff.
- Do not assume Blue Zone paths.
- Run `git pull --ff-only origin main` and record the new Git commit.
- Use only a free Atlas 310P device. Do not terminate other users' processes.
- Do not install or replace packages.
- Do not invoke TorchAir, `torch.compile`, cached graphs, or performance
  compilation.
- Keep vision attention on the manual implementation.
- Do not reduce the recognizer's `min_pixels`.
- Do not cap the number of regions. Every eligible region detected on each
  selected page must enter recognition.
- Put all logs and outputs below
  `tmp/09_persistent_page_engine/310p_eager_validation/`.

Resolve the repository and create the output directory before redirecting any
logs:

```sh
REPO="$(git rev-parse --show-toplevel)"
OUTPUT_ROOT="$REPO/tmp/09_persistent_page_engine/310p_eager_validation"
mkdir -p "$OUTPUT_ROOT"
```

## Phase 1: audit the entire dataset

Run the committed audit using the already-discovered Python and dataset paths:

```sh
"$PYTHON_BIN" \
  "$REPO/09_persistent_page_engine/scripts/audit_omnidocbench_dataset.py" \
  --dataset-json "$DATASET_JSON" \
  --images-dir "$IMAGES_DIR" \
  --output "$OUTPUT_ROOT/dataset_audit.json" \
  >"$OUTPUT_ROOT/dataset_audit.log" 2>&1
```

The audit follows the same image-path contract as Experiment 09: each
annotation's `page_info.image_path` basename must resolve beneath
`IMAGES_DIR`. It validates:

- exactly 1,651 annotation entries;
- valid image-path schema;
- no duplicated referenced basenames;
- every referenced image exists and is non-empty;
- every referenced image can be fully decoded by OpenCV as a three-channel
  image;
- a deterministic eight-page sample distributed from the first through the
  last annotation.

Dataset completeness passes only when `dataset_audit.json` has
`"valid": true`. Existence of the JSON file and one working page is not enough.

If the audit fails for missing, zero-byte, or unreadable images, follow the
repair protocol below. Do not start inference until the post-repair audit
passes. For any other failure, stop and report all count fields and the first
listed failures.

### Phase 1A: diagnose and repair unreadable images

The first execution of this handoff found eight existing files for which
`cv2.imdecode` returned `None`. Their names contain literal `%20` sequences and
their annotated sources include `docstructbench_llm-raw-scihub` and
`the-eye`.

Do not rename the files and do not replace `%20` with spaces. The audit opens
each path through `Path` and `np.fromfile`; no URL decoding occurs at that
point. Because bytes were read successfully, a literal `%20` in the local
filename is not by itself a decoding failure.

Use `dataset_audit.json` as the authoritative list of failures. Do not hard-code
or guess the eight names. For each unreadable file, record:

- annotation index and exact basename;
- absolute path and byte size;
- `sha256sum`;
- `file` output;
- the first 64 bytes and last 64 bytes in hexadecimal;
- whether the first bytes resemble JPEG (`ff d8 ff`), PNG
  (`89 50 4e 47 0d 0a 1a 0a`), HTML, JSON, or a Git LFS pointer;
- whether the final bytes contain the JPEG end marker `ff d9`;
- results from both Pillow `Image.verify()` and a separate reopen plus
  `Image.load()`;
- results from OpenCV `imdecode` using `IMREAD_UNCHANGED` and `IMREAD_COLOR`;
- annotation `page_info.width` and `page_info.height`, when present.

Write the complete per-file diagnostics to:

```text
$OUTPUT_ROOT/corrupt_image_diagnostics.json
$OUTPUT_ROOT/corrupt_image_diagnostics.log
```

Classify every file as one of:

1. truncated JPEG;
2. HTML/error response saved as an image;
3. Git LFS/Xet pointer rather than image content;
4. another image format with a misleading extension;
5. malformed image that another decoder can read but OpenCV cannot;
6. unknown.

Search the existing dataset download cache, source archives, and other sensible
mounted dataset roots for byte-distinct copies with the exact same basename.
Do not scan all of `/`. A candidate is usable only if OpenCV decodes it,
its dimensions agree with the annotation, and it is not another pointer/error
file.

If no valid local copy exists, download only the eight exact files from the
official OpenDataLab Hugging Face dataset:

```text
repository: opendatalab/OmniDocBench
repository type: dataset
file: images/<exact literal basename>
```

List the repository files through `HfApi` at one recorded revision before
downloading. Prefer an exact `images/<local basename>` match. If no exact match
exists, check whether exactly one remote basename becomes equal after URL
decoding with `urllib.parse.unquote`; verify that mapping against the official
`OmniDocBench.json` from the same revision. Stop if the mapping is absent or
ambiguous.

Use `huggingface_hub.hf_hub_download` with the selected repository filename
string, not a hand-assembled `curl` URL. This matters because the percent sign
in a literal `%20` filename would need to be URL-escaped as `%25` in a raw URL,
whereas `hf_hub_download` handles repository paths correctly. Record both the
remote filename and the resolved dataset revision SHA returned by
`HfApi().dataset_info(...)`. Preserve the local basename expected by the local
annotation even if the canonical remote path uses spaces.

Download into an isolated cache or `$OUTPUT_ROOT/replacement_candidates/`.
Never download directly over the live dataset files.

For each downloaded candidate:

1. confirm the exact basename;
2. record size and SHA-256;
3. decode it with the same OpenCV operation used by the audit;
4. verify its decoded dimensions against the matching annotation;
5. confirm it comes from the recorded OpenDataLab dataset revision.

Reject any candidate that fails one of those checks. Do not repair an image by
re-encoding the existing malformed bytes: that would create a locally modified
benchmark input rather than restore the canonical file.

Once all eight candidates pass, create
`$OUTPUT_ROOT/corrupt_originals/` and copy the original eight files there,
preserving their exact basenames and metadata. Replace the live files
atomically: copy each verified candidate to a temporary sibling path and rename
that path over the corrupt file only after the copy and SHA-256 check succeed.
Do not modify `OmniDocBench.json`.

Write a replacement manifest containing, for every file:

- basename;
- annotation index;
- diagnosis;
- original size and SHA-256;
- replacement size and SHA-256;
- replacement source repository, revision, and remote filename;
- decoded width and height;
- backup path;
- final live path.

Save it as:

```text
$OUTPUT_ROOT/corrupt_image_replacements.json
```

Rerun the entire 1,651-image audit from Phase 1, writing the new result to:

```text
$OUTPUT_ROOT/dataset_audit_after_repair.json
$OUTPUT_ROOT/dataset_audit_after_repair.log
```

Continue to Phase 2 only if the new report has `"valid": true`, exactly 1,651
decoded images, and zero missing, zero-byte, or unreadable images.

If a canonical file cannot be downloaded because of the work-server proxy,
leave the original untouched and report the exact network failure. If the
official candidate is itself unreadable or has dimensions inconsistent with
the local v1.6 annotation, stop and report the evidence rather than inventing a
conversion.

## Phase 2: select the exact eight pages

Use the latest successful audit's `uniform_sample` verbatim:
`dataset_audit_after_repair.json` when repairs were needed, otherwise
`dataset_audit.json`. It should contain these annotation indices when the
dataset has 1,651 pages:

```text
0, 236, 471, 707, 943, 1179, 1414, 1650
```

Verify each `absolute_path` still exists. Record the indices, filenames, byte
sizes, and dimensions in the final report.

Do not substitute easier pages and do not use the first eight consecutive
pages. The purpose of this selection is inexpensive coverage across the whole
dataset ordering, not statistical performance measurement.

## Phase 3: eight-page eager Experiment 09 run

Invoke `09_persistent_page_engine/scripts/run_offline_e2e.py` once, passing one
`--image` argument for each of the eight selected paths. Construct the argument
array programmatically from `dataset_audit.json`; do not manually retype or
silently reorder the paths.

The effective command must be equivalent to:

```sh
"$PYTHON_BIN" \
  "$REPO/09_persistent_page_engine/scripts/run_offline_e2e.py" \
  --image "<sample index 0 path>" \
  --image "<sample index 236 path>" \
  --image "<sample index 471 path>" \
  --image "<sample index 707 path>" \
  --image "<sample index 943 path>" \
  --image "<sample index 1179 path>" \
  --image "<sample index 1414 path>" \
  --image "<sample index 1650 path>" \
  --layout-model "$LAYOUT_MODEL" \
  --recognizer-model "$RECOGNIZER_MODEL" \
  --dtype fp16 \
  --decode-backend raw_eager \
  --vision-backend raw_eager \
  --vision-attention manual \
  --vision-padding none \
  --text-backend raw_eager \
  --text-padding none \
  --batch-size 1 \
  --cache-length 4096 \
  --max-new-tokens 32 \
  --no-save-annotated \
  --output-dir "$OUTPUT_ROOT/eight_pages"
```

There must be no `--max-regions` argument. The 32-token cap intentionally
truncates many OCR outputs; that is acceptable because this is a repeated
execution and integration check.

Before running, save the fully expanded shell-safe command to
`$OUTPUT_ROOT/eight_pages_command.txt`. Capture complete stdout/stderr in
`$OUTPUT_ROOT/eight_pages.log` and preserve the exit code.

This diagnostic runner accepts multiple `--image` arguments and uses one
run-scoped `ContinuousRecognizer`, so the run exercises:

- eight real layout calls;
- varied page and crop dimensions;
- every eligible detected region;
- cross-page request production and completion;
- repeated eager vision and text prefill;
- repeated eager static-KV decode and native torch-npu operations;
- B=1 decode-slot replacement as requests finish.

## Required checks

The run passes only if all of the following are true:

- process exit code is zero;
- `run.json` exists and parses;
- `aggregate.pages == 8`;
- `aggregate.partial_pages == 0`;
- `aggregate.layout_regions > 0`;
- `aggregate.recognized_regions > 0`;
- every selected input page appears exactly once in the results;
- all eight pages appear in `aggregate.page_completion_order`;
- `configuration.decode_backend == "raw_eager"`;
- `configuration.vision_backend == "raw_eager"`;
- `configuration.text_backend == "raw_eager"`;
- `configuration.vision_prefill.padding == "none"`;
- `configuration.text_prefill.padding == "none"`;
- no TorchAir cache or compilation activity appears in the log;
- no CPU or CUDA fallback occurs;
- accounting fields are internally consistent;
- at least one page contains multiple recognized regions.

Treat generated-text differences or 32-token truncation as diagnostics, not as
failures. A crash, missing page, missing region result, invalid accounting,
native-op failure, or fallback is a failure.

If the run reports that 4,096 cache positions are insufficient for a specific
prompt plus 32 generated tokens, report that crop and required capacity. Do not
raise the cache again or skip the crop in this pass.

## Phase 4: complete-output eager reference page

The 32-token eight-page run validates repeated execution but deliberately
truncates long regions. Finish the eager validation with one page whose regions
are allowed to run naturally to EOS.

Use this exact OmniDocBench page:

```text
PPT_The Right Moves_page_024.png
```

Resolve it beneath the audited `IMAGES_DIR`; do not use a similarly named copy
from elsewhere. Run:

```sh
"$PYTHON_BIN" \
  "$REPO/09_persistent_page_engine/scripts/run_offline_e2e.py" \
  --image "$IMAGES_DIR/PPT_The Right Moves_page_024.png" \
  --layout-model "$LAYOUT_MODEL" \
  --recognizer-model "$RECOGNIZER_MODEL" \
  --dtype fp16 \
  --decode-backend raw_eager \
  --vision-backend raw_eager \
  --vision-attention manual \
  --vision-padding none \
  --text-backend raw_eager \
  --text-padding none \
  --batch-size 1 \
  --cache-length 4096 \
  --max-new-tokens 2808 \
  --no-save-annotated \
  --output-dir "$OUTPUT_ROOT/full_output_page"
```

There must be no `--max-regions` argument. Save the fully expanded command to
`$OUTPUT_ROOT/full_output_page_command.txt` and the complete log to
`$OUTPUT_ROOT/full_output_page.log`.

The 2,808-token cap is the Experiment 09 full-output ceiling that still fits
the 4,096-position cache for the retained workload. It is intentionally much
higher than this page needs. The success condition is natural EOS completion,
not reaching the cap.

### 910B eager reference

This exact command was validated on a 910B at Git commit `556d5dd`, using
torch `2.10.0+cpu` and torch-npu `2.10.0`. The observed configuration was:

```text
decode_backend=raw_eager
vision_backend=raw_eager
vision_attention=manual
vision_padding=none
text_backend=raw_eager
text_padding=none
batch_size=1
cache_length=4096
max_new_tokens=2808
decode_attention=increfa
decode_cache_update=npu_scatter
```

The reference completed five layout/recognition regions, all by EOS:

| Region | Label | Input tokens | Generated incl. EOS | Stop | Compact token-ID SHA-256 |
| --- | --- | ---: | ---: | --- | --- |
| 001 | `paragraph_title` | 165 | 7 | `eos` | `1fb3a2ba8476f9ef2066d97f95f6a381045e330157f95eae0c91951cfc9f4093` |
| 002 | `text` | 293 | 14 | `eos` | `9c5236da4551449b97f089ca8091be435fe5e2a90f9631d07ea9febb00e33bd0` |
| 003 | `text` | 895 | 42 | `eos` | `11a29aa8660a05402678aec730d59c0957fb585c9c83a88227f0248e6c892533` |
| 004 | `text` | 385 | 15 | `eos` | `da323dee18d6dececca5e525b6387ad2509af1a9fd9dfb19b6b9894a005bdb40` |
| 005 | `number` | 167 | 3 | `eos` | `4f90f056e19dff68308f911795c3c87c5b7c2ce0571a6a8ba5ac976593e21d72` |

The token-ID hash is over
`json.dumps(token_ids, separators=(",", ":"))` encoded as UTF-8, with no
trailing newline. The reference aggregate was:

```text
pages=1
layout_regions=5
recognized_regions=5
partial_pages=0
stop_reason_counts={"eos": 5}
generated_tokens_including_eos=81
decode_graph_calls=81
hot_swap_decode_admissions=4
```

The assembled page text was:

```text
Subject Support con't.

- ESL subject section courses are credit-granting courses.

These courses focus on the subject content while placing emphasis on subject-related vocabulary, language structures and cultural background in order to support students who are acquiring English at the same time that they are learning the subject.

• Often these subject specific courses are aligned with specific ESL levels.
```

Including its final newline, that page text has SHA-256:

```text
ee0eba01a6fa41ca94b5b87a721ed10db8694c2f5f29b203a83e2f3bb355ced9
```

### Complete-output success checks

The 310P run passes this phase only if:

- exit code is zero and `run.json` parses;
- exactly one page completes with `partial_pages == 0`;
- all detected eligible regions have a result;
- every recognized region has `stop_reason == "eos"`;
- no region reaches `length_cap`;
- output text is non-empty and page assembly succeeds;
- configuration remains raw eager/manual/unpadded exactly as specified;
- decode reports `increfa` and `npu_scatter`;
- no TorchAir compilation or CPU/CUDA fallback occurs.

Compare the region count, token IDs, texts, and assembled-page hash with the
910B reference. Exact parity is strong evidence and should be reported.
However, a token/text mismatch alone is a diagnostic rather than an automatic
compatibility failure if every structural and EOS check above passes. Report
the first differing region and both values. Do not hide a mismatch and do not
reclassify a crash, length cap, missing region, or fallback as numeric drift.

## Phase 5: 310P characterization matrix

Phases 1–4 establish compatibility. This phase characterizes the first useful
optimized paths in one iteration:

1. eager PromptFlashAttention;
2. isolated layout frontend throughput with one versus two workers;
3. cold TorchAir decode graph creation and persisted warm replay;
4. B1 versus B4 decode batching;
5. eager-B4 versus compiled-B4 correctness and speed.

Vision and text prefill remain `raw_eager` throughout this phase. Only the
decode backend is compiled. This separation is mandatory: do not enable
TorchAir vision or text-prefill buckets yet.

Create:

```sh
CHAR_ROOT="$OUTPUT_ROOT/characterization"
GIT_COMMIT="$(git rev-parse HEAD)"
mkdir -p "$CHAR_ROOT"
```

Record NPU identity, free HBM, Python/package/CANN versions, Git commit, all
expanded commands, and exit codes. Sample NPU memory/utilization while each run
is active if the server already provides a safe read-only monitoring command;
do not introduce new synchronization or profiler instrumentation into the
timed process.

### Phase 5A: eager PromptFlashAttention full-output gate

Repeat the Phase 4 complete-output page with only:

```text
--vision-attention prompt_flash_attention
--vision-promptfa-align-128
```

Keep decode, vision execution, and text execution `raw_eager`; keep B1,
KV4096, no padding, and the 2,808-token ceiling. Write it below:

```text
$CHAR_ROOT/full_output_promptfa_eager_b1/
```

On the 910B reference, PromptFA was byte-identical to manual attention across
all five regions. It preserved five EOS stops and 81 generated tokens while
changing:

| Metric | Manual eager | PromptFA eager | PromptFA aligned repeat |
| --- | ---: | ---: | ---: |
| Page wall | 2.7916 s | 2.5904 s | 2.5243 s |
| Vision device total | 0.4943 s | 0.3266 s | 0.3107 s |
| Real / physical vision tokens | 7,360 / 7,360 | 7,360 / 7,360 | 7,360 / 7,552 |

For 310P, all Phase 4 structural/EOS checks remain mandatory. Compare the
complete-output tokens and text against the manual Phase 4 result. Exact parity
is expected and is the PromptFA correctness gate.

The aligned 910B lane was byte-identical to both the unaligned PromptFA and
manual reference across every token, text result, and EOS stop. It rounded each
physical crop length to a 128-token multiple with 97.46% aggregate useful
vision tokens. The integration retains the minimal BNSD 310P call contract:
no `actual_seq_lengths`, no `actual_seq_lengths_kv`, and no explicit
`num_key_value_heads`.

If this gate fails or crashes, preserve the first causal native-op traceback
and skip PromptFA-based lanes below. Continue the layout comparison and use
manual vision attention for the decode-graph lanes.

If an earlier Phase 5 attempt failed with
`attention mask must be NULL, when Qs, Kvs is unAlign`, pull the commit that
introduces `--vision-promptfa-align-128` and rerun Phase 5A plus only the
previously skipped PromptFA lanes. Do not rerun successful manual-attention
or layout lanes.

### Phase 5B: isolated 32-page layout comparison

Run the committed owned-layout lab on the first 32 OmniDocBench pages with one
worker:

```sh
"$PYTHON_BIN" \
  "$REPO/09_persistent_page_engine/scripts/layout_owned_lab.py" \
  --dataset-json "$DATASET_JSON" \
  --images-dir "$IMAGES_DIR" \
  --layout-model "$LAYOUT_MODEL" \
  --limit 32 \
  --workers 1 \
  --no-timeline \
  --output-dir "$CHAR_ROOT/layout_w1"
```

Then run the exact same pages with the decode-prefetch/two-worker mode and
compare its request manifest:

```sh
"$PYTHON_BIN" \
  "$REPO/09_persistent_page_engine/scripts/layout_owned_lab.py" \
  --dataset-json "$DATASET_JSON" \
  --images-dir "$IMAGES_DIR" \
  --layout-model "$LAYOUT_MODEL" \
  --limit 32 \
  --workers 2 \
  --no-timeline \
  --reference-requests "$CHAR_ROOT/layout_w1/requests.jsonl" \
  --output-dir "$CHAR_ROOT/layout_w2"
```

The `workers=2` mode does not batch pages through one detector call. It
prefetches the next page's CPU file read/image decode while the current page
continues through the sequential layout path.

Require:

- 32 pages and 510 requests in both lanes;
- identical page, request, label, and reading-order structure;
- either byte-identical `requests.jsonl`, or only coordinate/crop-size
  differences of at most one pixel from parallel Kornia resize rounding;
- report `reference_comparison.exact` honestly and list every tolerated
  one-pixel difference rather than relabeling it exact;
- no PaddleX dependency;
- the same NPU graph/mask layout route;
- no inference or OCR recognizer execution.

Report `frontend_wall_s`, pages/s, seconds/page, setup time, request count, and
all `stage_totals_s`. In particular separate image decode, model device time,
postprocessing, page preparation, and wait/D2H fields.

The 910B reference was:

| Layout lane | Wall | Pages/s | Seconds/page | Requests |
| --- | ---: | ---: | ---: | ---: |
| W1 serial | 4.7166 s | 6.7845 | 0.14739 | 510 |
| W2 decode prefetch | 3.1290 s | 10.2268 | 0.09778 | 510 |

The W2 request manifest was byte-identical to W1. These numbers are comparison
context, not 310P pass thresholds.

### Phase 5C: B1 decode graph creation and persisted replay

This is the first TorchAir test. Use the complete-output reference page so
every request terminates naturally and can be compared with Phase 4.

Choose a new 310P-specific cache root that has never been used on another
hardware target, for example:

```sh
B1_CACHE="$REPO/.runtime_cache/310p_decode_b1_k4096_${GIT_COMMIT}"
```

Do not reuse or copy 910B caches. Confirm the path does not exist before the
cold run. If it exists from a previous failed attempt, preserve and report it;
choose a new suffixed path rather than deleting evidence.

Run the Phase 4 command twice with these changes:

```text
--decode-backend torchair
--torchair-cache-dir "$B1_CACHE"
```

Keep manual vision attention, B1, eager/unpadded vision, eager/unpadded text,
KV4096, and the 2,808-token ceiling. Write the first process to
`$CHAR_ROOT/decode_b1_cold/` and the second process to
`$CHAR_ROOT/decode_b1_warm/`, reusing the exact same cache.

Require:

- both runs satisfy all Phase 4 structural and EOS checks;
- cold and warm compiled results have exact token/text parity with manual
  eager Phase 4;
- cold and warm compiled results are byte-identical to each other;
- `configuration.decode_backend == "torchair"`;
- vision and text remain `raw_eager` with no padding;
- `configuration.decode_attention == "increfa"`;
- `configuration.decode_cache_update == "npu_scatter"`;
- the warm process reuses the same shape cache;
- no unexpected second compilation occurs during inference.

Report separately:

- total setup;
- `compile_wrapper`;
- `compile_first_call`;
- inference `run_wall_s`;
- `continuous_decode_wall_s`;
- raw and effective decode token/s;
- cache directory, size, and file count;
- token/text hashes.

The 910B B1 reference was:

| B1 lane | First graph call | Page wall | Decode wall | Raw slots/s |
| --- | ---: | ---: | ---: | ---: |
| Raw eager | none | 2.7916 s | 1.2766 s | 63.45 |
| TorchAir cold | 13.1517 s | 2.0828 s | 0.5582 s | 145.12 |
| TorchAir warm | 0.2239 s | 2.1201 s | 0.5919 s | 136.84 |

All three produced identical tokens and text. Compile/load setup is outside the
reported inference wall and must not be mixed into steady replay throughput.

If the cold B1 graph fails, do not attempt B4 compilation. Preserve the cache,
complete traceback, and compiler logs. Continue only with eager lanes.

### Phase 5D: matched eight-page performance matrix

Use the exact same eight pages, order, all-region policy, KV4096, and 32-token
cap from Phase 3. Reuse the successful manual-eager B1 Phase 3 result as the
baseline; do not rerun it merely to populate a table.

Run these additional lanes:

| Lane | Vision attention | Decode backend | Batch | Decode cache |
| --- | --- | --- | ---: | --- |
| PFA eager B1 | PromptFA | `raw_eager` | 1 | none |
| PFA eager B4 | PromptFA | `raw_eager` | 4 | none |
| PFA TorchAir B4 cold | PromptFA | `torchair` | 4 | new B4 cache |
| PFA TorchAir B4 warm | PromptFA | `torchair` | 4 | same B4 cache |

Every PromptFA lane must also include:

```text
--vision-promptfa-align-128
```

If Phase 5A rejected PromptFA, replace PromptFA with manual attention in all
four lanes and label the matrix accordingly.

Use a new hardware-specific B4 cache:

```sh
B4_CACHE="$REPO/.runtime_cache/310p_decode_b4_k4096_${GIT_COMMIT}"
```

Apply the same cold-cache preservation rule as B1. The cold and warm B4 runs
must use exactly the same cache path and all other arguments.

For every lane, record:

- setup and compile/load timings;
- E2E wall, pages/s, and regions/s;
- layout-region, recognized-region, partial-page, and stop-reason counts;
- summed vision-prefill and text-prefill device seconds;
- real and physical vision/text tokens and corresponding device token/s;
- decode wall, graph calls, raw/effective tokens, raw/effective token/s;
- active-slot fraction, idle slots, look-ahead slots, and hot-swap admissions;
- output fingerprint over ordered request ID, token IDs, text, and stop reason;
- peak observed HBM if available without perturbing execution.

### Matrix correctness comparisons

Do not use one global token-parity rule:

1. Manual-eager B1 versus PromptFA-eager B1 must be compared directly. On
   910B they were byte-identical across all 164 regions.
2. PromptFA-eager B4 versus PromptFA-TorchAir B4 cold must be byte-identical.
3. PromptFA-TorchAir B4 cold versus warm must be byte-identical.
4. B1 versus B4 may differ numerically. Report the number and first example,
   but do not fail B4 solely for differing tokens if the two B4 backends agree.

This distinction is evidence-backed: on 910B, manual B1 and PromptFA B1 shared
one exact fingerprint, while eager B4, cold compiled B4, and warm compiled B4
shared a second exact fingerprint. The difference therefore came from batch
execution numerics, not PromptFA and not TorchAir lowering.

All lanes must still have identical page counts, region counts, partial-page
counts, and stop-reason counts. A crash, missing region, inconsistent
accounting, B4 eager/compiled mismatch, cold/warm mismatch, or fallback is a
failure.

### 910B eight-page reference

The exact same 164-region workload produced:

| Lane | Wall | Vision device | Text device | Decode wall | Raw decode/s | Effective decode/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Manual eager B1 | 56.7277 s | 7.2434 s | 5.8241 s | 40.5095 s | 64.36 | 60.31 |
| PFA eager B1 | 55.9988 s | 7.4866 s | 5.7222 s | 39.6292 s | 65.78 | 61.65 |
| PFA eager B4 | 27.5907 s | 7.4739 s | 5.7885 s | 11.1628 s | 235.78 | 218.85 |
| PFA TorchAir B4 cold | 19.2247 s | 8.0451 s | 6.1260 s | 1.7411 s | 1511.66 | 1403.11 |
| PFA TorchAir B4 warm | 17.8821 s | 7.4108 s | 5.6450 s | 1.6395 s | 1605.36 | 1490.08 |

The B4 scheduler executed 658 decode calls with 99.05% active-slot
utilization, 160 hot-swap admissions, 25 idle slots, and 164 completion
look-ahead slots. PromptFA was essentially neutral on this mixed corpus;
batching and decode graph replay produced the material gains.

Do not use the 910B timings as 310P thresholds. Use them to ensure the same
metric definitions and to identify whether the 310P bottleneck shape differs.

## Scope of the conclusion

If the dataset audit, eight-page run, and complete-output page all pass, it is
fair to conclude:

- the complete OmniDocBench v1.6 image set needed by this checkout is present
  and OpenCV-readable;
- both local models and their support files load;
- Experiment 09's eager diagnostic path repeatedly works across diverse real
  pages on this 310P software stack;
- at least one real page completes naturally by EOS without a short diagnostic
  token cap;
- the exercised native NPU operations are callable in real inference.

If the applicable Phase 5 gates also pass, it is additionally fair to
conclude:

- the committed eager PromptFlashAttention call is compatible with this 310P
  stack;
- the layout decode-prefetch mode preserves exact request output;
- B1 and B4 TorchAir decode graphs can be created and replayed from persisted
  hardware-specific caches;
- compiled B4 agrees with eager B4 for the tested workload;
- the reported 310P timing matrix is a valid relative characterization of
  these specific execution modes.

It is still **not** fair to conclude:

- OCR accuracy matches the reference implementation;
- the entire 1,651-page pipeline has completed inference;
- TorchAir vision or text-prefill compilation works;
- a cache created on 310P is portable to another hardware/software stack;
- throughput is representative or optimized.

## Required report

Write `$OUTPUT_ROOT/agent_report.md` and end your response to Luka with the same
concise block:

```text
310P EAGER VALIDATION: PASS | PARTIAL | FAIL

Git commit:
Host / NPU:
Python:
torch / torch_npu / CANN:

Initial dataset audit: PASS | FAIL
Annotations:
Referenced unique images:
Decoded images:
Missing / zero-byte / unreadable:
Dataset bytes:
Audit wall time:

Unreadable-image diagnoses:
Canonical source repository / revision:
Files replaced:
Original -> replacement SHA-256:
Original backups:
Post-repair dataset audit: PASS | NOT NEEDED | FAIL
Post-repair decoded images:
Replacement manifest:

Uniform indices:
Uniform filenames:

Eight-page eager run: PASS | SKIPPED | FAIL
Pages completed:
Layout regions:
Recognized regions:
Partial pages:
Run wall time:
Decode iterations / admissions:
Configuration backends:

Complete-output eager page: PASS | SKIPPED | FAIL
Regions / EOS stops / length caps:
Generated tokens including EOS:
Decode calls / hot-swap admissions:
Region token-ID parity with 910B:
Region text parity with 910B:
Assembled page SHA-256 / parity:
First mismatch:
Complete-output run.json:

PromptFA full-output gate: PASS | SKIPPED | FAIL
PromptFA B1 parity:

Layout W1: wall / pages-s / seconds-page / requests:
Layout W2: wall / pages-s / seconds-page / requests:
Layout manifest parity:
Layout stage totals:

B1 graph cold: compile-first / wall / decode wall / raw-effective tok-s:
B1 graph warm: compile-first / wall / decode wall / raw-effective tok-s:
B1 eager-compiled parity:
B1 cache path / size:

Eight-page matrix:
manual eager B1:
PFA eager B1:
PFA eager B4:
PFA TorchAir B4 cold:
PFA TorchAir B4 warm:
B1 PromptFA parity:
B4 eager-compiled parity:
B4 cold-warm parity:
B1-B4 first difference:
B4 cache path / size:

First generated token IDs/text:
First blocker or important warning:
Exact command record:
run.json:
dataset_audit.json:
Log paths:
```

Report exact observed fields. Do not summarize a missing value as though it
passed, and do not turn this compatibility smoke into a performance claim.
