# Experiment 5 Queue Benchmark Flow

`bench_recognizer_queue.py` measures a recognizer-only serving shape for real
OmniDocBench crops that are already selected in `crops/hotswap_100_manifest.json`.
It does not run page layout detection. It does include crop file read/decode,
local PaddleOCR-VL preprocessing, native-resolution vision encoding, adaptive
MLP projection, text prefill, static-cache decode, output postprocess, and
correctness validation.

The queue benchmark supports fixed decode cohorts through `ACTIVE_BATCH_SIZE`.
It stays padding-free: if the final cohort has fewer than `ACTIVE_BATCH_SIZE`
items, the script compiles/wraps and decodes that smaller real cohort shape
instead of adding fake rows.

## Data Flow

```mermaid
flowchart LR
    A["Manifest rows<br/>crop path, prompt, metadata"] --> B["CPU input build"]
    B --> B1["Image read/decode<br/>PIL RGB crop"]
    B1 --> B2["Resize + normalize + patchify<br/>pixel_values: [N, 3, 14, 14]<br/>image_grid_thw: [1, 3]"]
    B2 --> B3["Tokenizer + image-token prompt<br/>input_ids: [1, T]<br/>attention_mask: [1, T]"]
    B3 --> C["Cache preflight<br/>required = T + max_new_tokens - 1"]
    C -->|overflow| CERR["Stop with JSON error<br/>cache_length_too_small"]
    C -->|fits| D["Model setup"]
    D --> D1["Load local model fp16"]
    D1 --> D2["Decode weight format<br/>NPU: FRACTAL_NZ<br/>CUDA: skipped"]
    D2 --> D3["Compile/wrap decode<br/>NPU: TorchAir cache_compile<br/>CUDA: raw eager or torch.compile"]
    D3 --> E["Ready bank build<br/>one crop at a time"]
    E --> E1["Device transfer"]
    E1 --> E2["Vision embeddings + 27-layer encoder<br/>hidden: [N, 1152]"]
    E2 --> E3["Post LN + adaptive MLP projector<br/>image_embeds: [N/4, 1024]"]
    E3 --> E4["Text token embeddings + image embed scatter<br/>inputs_embeds: [1, T, 1024]"]
    E4 --> E5["mRoPE indices + static KV cache alloc"]
    E5 --> E6["Text prefill writes KV cache<br/>per layer K/V: [1, 2, cache_length, 128]"]
    E6 --> E7["LM head argmax<br/>ReadyItem: cache, rope_deltas, cache_position, next_token"]
    E7 --> F["Decode queue<br/>fixed real cohorts"]
    F --> F1["Batched static decode calls<br/>next_token[B,1] -> logits -> argmax"]
    F1 --> F2["EOS policy<br/>none or overlap_event_flags"]
    F2 --> F3["Device token tensors kept until postprocess"]
    F3 --> G["Postprocess"]
    G --> G1["Copy tokens to CPU"]
    G1 --> G2["Trim at EOS + tokenizer.decode"]
    G2 --> H["Validation"]
    H --> H1["Direct local static generation per item"]
    H1 --> H2["Compare trimmed token IDs<br/>check invalid IDs"]
    H2 --> I["JSON report + printed summary"]
```

## Code Flow

```mermaid
sequenceDiagram
    participant CLI as parse_args/main
    participant Inputs as build_queue_inputs
    participant Model as LocalPaddleOCRVL
    participant Ready as build_ready_item
    participant Decode as decode_ready_item
    participant Output as materialize/validate

    CLI->>Inputs: load manifest and selected crop rows
    Inputs->>Inputs: preprocess_crop_timed for each crop
    Inputs->>Inputs: build_inputs tokenizer prompt for each crop
    Inputs-->>CLI: list[QueueInput], input timing summary
    CLI->>CLI: prompt_token_summary cache preflight
    alt any item needs more than cache_length
        CLI-->>CLI: print valid JSON error and exit
    else all items fit
        CLI->>Model: from_pretrained(dtype, device)
        CLI->>Model: cast_decode_linear_weights_to_nz
        CLI->>Model: make_flat_static_decode_module
        CLI->>Decode: compile_decode_module and one warm call
        loop selected QueueInput rows
            CLI->>Ready: build_ready_item
            Ready->>Model: vision embeddings, encoder, post LN
            Ready->>Model: adaptive MLP projector
            Ready->>Model: text embedding + image scatter
            Ready->>Model: mRoPE + static cache allocate
            Ready->>Model: text prefill + lm_head argmax
            Ready-->>CLI: ReadyItem
        end
        loop ReadyItem rows
            CLI->>Decode: decode_ready_item
            Decode->>Decode: call compiled/raw static decode by real fixed cohort until EOS or cap
            Decode-->>CLI: DecodedDeviceItem
        end
        CLI->>Output: materialize_decoded_item for each item
        CLI->>Output: validate_outputs against generate_ids_static
        Output-->>CLI: correctness and text samples
        CLI-->>CLI: print JSON and runner summary
    end
```

## Main Data Objects

`QueueInput` is host-side prepared data for one crop. It contains the manifest
entry, absolute crop path, prompt, `input_ids`, `attention_mask`, `pixel_values`,
`image_grid_thw`, and CPU input-build timings. The important shapes are:

- `pixel_values`: `[vision_tokens, 3, 14, 14]`
- `image_grid_thw`: `[1, 3]`, with `[grid_t, grid_h, grid_w]`
- `input_ids`: `[1, prompt_tokens]`
- `attention_mask`: `[1, prompt_tokens]`

`ReadyItem` is device-side state after vision/projector/prefill. It contains the
per-item static cache, `rope_deltas`, `next_cache_position`, first `next_token`,
stage timings, `vision_tokens`, and `projected_image_tokens`. For this model,
projected image tokens are normally `vision_tokens / 4` because the adaptive MLP
connector performs a 2x2 spatial merge.

`DecodedDeviceItem` is the device-resident decode result before CPU
materialization. It keeps a list of one-token tensors so the measured
`decode_queue` phase does not include tokenizer decode or bulk CPU token copies.

`ReadyCohort` is a padding-free batch of `ReadyItem` rows. It concatenates
single-item K/V cache rows, `rope_deltas`, `next_cache_position`, and
`next_token` across batch dimension. The last cohort can be smaller than
`ACTIVE_BATCH_SIZE`.

`DecodedItem` is the CPU-side output after materialization. It contains raw token
IDs, EOS-trimmed token IDs, generated text, EOS/length-cap flags, and
postprocess timing.

## Filters And Hard Checks

The benchmark deliberately fails early instead of silently changing the serving
shape:

- `num_items`, `max_new_tokens`, and `cache_length` must be positive.
- `active_batch_size` must be positive.
- Every selected crop file must exist.
- `cache_length` must cover `input_tokens + max_new_tokens - 1` for every item.
- Image-token count and projected-image-embedding count must match exactly.
- Validation is required by default (`validation_items=-1`) and compares every
  queued output against direct local static generation.
- Token IDs are checked for invalid values outside the model vocabulary.

There is no hidden text padding path in this benchmark. Different crop/prompt
lengths become different ready states, and the decode queue groups those ready
states into fixed real cohorts.

## Timing Buckets

`setup_timing_s` is reported but excluded from throughput. It includes model
load, device-specific decode weight format handling, compile/wrapper creation,
and the first decode call that may trigger compilation or cache load. On NPU,
weight format handling preconverts decoder linear weights to FRACTAL_NZ. On
CUDA, the same step is skipped and reported as such.

`phase_timing_s.input_build_wall` is host-side crop processing and prompt
construction for all selected crops.

`phase_timing_s.ready_bank_build` is sequential per-crop device transfer,
native-resolution vision encoder, adaptive MLP projector, text prefill, and
first-token LM head.

`phase_timing_s.decode_queue` is only the fixed-cohort text decode loop over
already-ready states.

`pipeline_stage_timing_summary_s` is a clearer set of aliases over the existing
timers:

- `vision_prefill`: vision prepare, native-resolution visual encoder, and
  adaptive MLP projector.
- `text_prefill`: text token embedding, image embed scatter, mRoPE/static-cache
  setup, text decoder prefill, first-token LM head, and argmax.
- `text_decode`: the fixed-cohort decode queue.

`phase_timing_s.decode_output_postprocess` is token tensor materialization,
EOS trimming, and tokenizer decode.

`phase_timing_s.validation` is direct static generation used for correctness.
It is not included in throughput.

## Device Differences

On NPU, the decode path uses TorchAir cache compile, IncreFlashAttention,
`torch_npu.scatter_update_` for decode KV writes, and optional overlapped EOS
flag copies on a second NPU stream. Decoder linear weights are preconverted to
FRACTAL_NZ before compile.

On CUDA, TorchAir and `torch_npu` operators are not used. The same local model
falls back to manual attention and per-row KV cache writes. Use `raw_eager` for
the closest correctness smoke, or a regular `torch.compile` backend such as
`inductor` for CUDA compiler smoke. CUDA numbers are useful for debugging the
Python/data flow, but they are not Ascend throughput evidence.
