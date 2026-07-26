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

If the audit fails, do not start inference. Report all count fields and the
first listed failures. Do not download or replace files in this pass.

## Phase 2: select the exact eight pages

Use `dataset_audit.json`'s `uniform_sample` verbatim. It should contain these
annotation indices when the dataset has 1,651 pages:

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

## Scope of the conclusion

If both phases pass, it is fair to conclude:

- the complete OmniDocBench v1.6 image set needed by this checkout is present
  and OpenCV-readable;
- both local models and their support files load;
- Experiment 09's eager diagnostic path repeatedly works across diverse real
  pages on this 310P software stack;
- the exercised native NPU operations are callable in real inference.

It is **not** yet fair to conclude:

- OCR accuracy matches the reference implementation;
- the entire 1,651-page pipeline has completed inference;
- PromptFlashAttention works;
- TorchAir compilation works;
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

Dataset audit: PASS | FAIL
Annotations:
Referenced unique images:
Decoded images:
Missing / zero-byte / unreadable:
Dataset bytes:
Audit wall time:

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

First generated token IDs/text:
First blocker or important warning:
Exact command record:
run.json:
dataset_audit.json:
Log paths:
```

Report exact observed fields. Do not summarize a missing value as though it
passed, and do not turn this compatibility smoke into a performance claim.
