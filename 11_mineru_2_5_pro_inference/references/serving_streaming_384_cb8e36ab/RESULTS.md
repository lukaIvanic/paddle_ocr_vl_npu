# MinerU bounded streaming validation

Runtime commit `cb8e36ab`; physical Ascend 910B2 NPU4; first 384 OmniDocBench
v1.6 pages. The model, processor, B32/KV4096 configuration, vision lookahead 32,
CPU preparation depth 64, FP16 and compiled kernels match the reference.

| Metric | Unchanged trace anchor | Streaming |
| --- | ---: | ---: |
| Pipeline wall, seconds | 551.485 | 363.509 |
| Pages/s, including trace and writer drain | 0.69630 | 1.05637 |
| Average active decode slots | 25.72% | 96.57% |
| Decode device time, seconds | 227.081 | 61.833 |
| Completed pages | 384 | 384 |
| Request traces | 5486 | 5486 |
| Length-capped crops | 18 | 18 |

The previous untraced baseline was 540.304 s, or 0.71071 pages/s. Streaming is
48.6% faster in pages/s than that run. Setup and warmup are outside all pipeline
timings. The streaming run's first page was written after 13.057 s. It kept at
most 32 live pages, 64 queued CPU preparations, 63 generation states and two
pending page writes. No decode row stayed empty while prepared work was ready.

## Output audit

- All 384 layout sequences are token-exact.
- 5099/5102 recognition sequences are token-exact.
- 383/384 Markdown pages are byte-identical.
- One short Chinese text region substitutes `座` for `度` twice. No accuracy
  preference is claimed for either output.
- The two other raw differences are random table-image labels. Their final
  Markdown and block JSON are byte-identical, and raw text matches after
  bijective placeholder renaming.
- No request/page is missing or extra. All trace/token counters reconcile.
- All 18 pre-existing length-capped crops retain identical token sequences.

This is output-regression validation, not a new official accuracy evaluation.
The 8-page smoke also passed: all 146 request inputs matched; two recognition
sequences differed only in LaTeX presentation; no length caps occurred there.

## Table placeholder audit

The installed helper is
`/workspace/venvs/mineru_pro_vllm_py312/lib/python3.12/site-packages/mineru_vl_utils/post_process/table_image_processor.py`.
Its SHA256 was
`288013481da8215dacfea43102ce997c8f2d815ebdd813d89b14beabd5777b32`.
`_generate_uid` at lines 172-173 uses `random.choices` to make four-character
labels. `mask_and_encode_table_image` draws those labels into the image and
records their original embedded-image data. Postprocessing replaces the labels
using that map. The refactor did not change this helper.

The affected requests are:

- `page-d29fe4d2-832a-4ad6-ac80-dc01cf8b0e16.png:recognition:2`
- `page-d5f79be0-5d57-4849-9897-6106dd32117a.png:recognition:4`

The comparison remains strict by default. Its explicit
`--allow-table-image-placeholders` option accepts only an image-hash change for
a table with identical geometry, prompt IDs and other inputs, EOS completion,
bijectively renamed labels, and byte-identical final Markdown AND block JSON.
It never excuses other input changes or changed table content.

## Reproduction and contents

Verify `SHA256SUMS`, then extract `streaming.tar.gz` into an empty directory.
The archive includes output token traces, predictions, block JSON, run metadata,
command, source commit, exit code, run log and `comparison.json`.

Run the comparison against the extracted anchor:

```sh
python3 11_mineru_2_5_pro_inference/compare_generation_traces.py \
  /path/to/anchor/output /path/to/streaming/output \
  --allow-table-image-placeholders --output /path/to/comparison.json
```

The two input paths must be different extracted directories. The comparison
prints all exceptions and returns nonzero for unexpected input changes, missing
or empty pages, missing crops on unchanged layouts, new length stops, or token
accounting mismatches.
