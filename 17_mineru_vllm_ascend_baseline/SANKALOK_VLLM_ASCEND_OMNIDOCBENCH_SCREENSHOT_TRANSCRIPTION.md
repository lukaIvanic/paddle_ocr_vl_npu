# Experiment 17 source: Sankalok MinerU vLLM-Ascend OmniDocBench screenshots

Transcribed from seven photographs supplied by Luka on 2026-08-27. This file
records the visible text and code. It does not validate the reported run, code,
software versions, dataset contents, or performance. No command shown in the
screenshots was executed while preparing this transcription.

The photographs have blur, perspective distortion, and overlapping regions.
Line wrapping and indentation in the Python listing were reconstructed from the
visible code. The throughput summary below is a claim shown in the screenshots,
not a result reproduced in this repository.

## Reported configuration and results

> Here's the full picture of Sankalok's vLLM-ascend offline settings for
> MinerU2.5-Pro on OmniDocBench:

### Hardware

- Chip: Atlas 310P3 (4 chips on the box, but TP=1, single chip used)
- CANN: 8.0.0
- torch_npu: 2.10.0
- vLLM: 0.21.0 with vllm-ascend plugin

### Model

- MinerU2.5-Pro-2605-1.2B, resolved as `Qwen2VLForConditionalGeneration`
- Loaded as `float16`, no quantization
- Weight layout: `FRACTAL_NZ` (310P-specific)

### Compiled and asynchronous mode

The screenshot labels this as the main result at approximately 0.16 pages/s.

| Setting | Reported value |
|---|---|
| Engine | `AsyncLLM` (v1 async engine) |
| `tensor_parallel_size` | `1` |
| `max_model_len` | `8192` |
| `gpu_memory_utilization` | `0.9` |
| `dtype` | `float16` |
| `max_num_seqs` | `512` |
| `max_num_batched_tokens` | `16384` |
| `enforce_eager` | `False` (compilation on) |
| `enable_prefix_caching` | `True` |
| `enable_chunked_prefill` | `True` (reported as auto-enabled) |

Reported Ascend compilation configuration:

```text
enable_npugraph_ex: True
enable_static_kernel: True
fuse_norm_quant: False
```

Reported CUDAGraph configuration:

```text
cudagraph_mode: FULL_DECODE_ONLY
cudagraph_capture_sizes: [1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 28, 32]
```

Reported Hugging Face overrides:

```text
tie_word_embeddings: True (both top-level and text_config)
```

Reported pipeline:

```text
MinerUClient(backend="vllm-async-engine", batch_size=0, image_analysis=True)
concurrent_two_step_extract(images) - async concurrent two-step (layout then OCR)
Custom MinerULogitsProcessor (no-repeat-ngram)
```

Reported result:

```text
981 pages in 6144 s -> 0.1597 pages/s (layout + OCR end to end)
```

### Eager-mode baseline

The screenshot describes the eager lane as using the same model, dtype, and
tensor-parallel setting, with these differences:

- `enforce_eager=True`, with no compilation, CUDAGraph, or static kernel
- `enable_prefix_caching=False`
- Synchronous `LLM` engine
- Page-by-page sequential `two_step_extract`
- No `max_num_seqs` or `max_num_batched_tokens` tuning

Reported result:

```text
981 pages in 32998 s -> 0.0298 pages/s
```

### Reported explanation of the speedup

The screenshot attributes the approximate 5x eager-to-compiled speedup to:

1. Static-kernel compilation plus `npugraph_ex`, with precompiled decode graphs
   for batch sizes 1 through 32.
2. An asynchronous concurrent pipeline, which overlaps pages in the engine.
3. Prefix caching for layout and OCR prompt prefixes across pages.

The screenshot also states:

> The script (`mineru.py:84`) currently has `images = images[:10]`, which would
> only run 10 pages, but the logged run processed all 981. That line was added
> later for testing.

This statement is important because the photographs do not include the original
981-page command, log, file list, output count, or timing artifact.

## Transcribed `mineru.py`

The terminal command shown in the photograph reads
`/home/sankalok/July2026/mineru_offline/mineru.py`, prints a separator, and then
reads `remove_md.py`.

```python
import os
import time
from tqdm import tqdm
from pathlib import Path

import torch
import torch_npu

# from vllm import LLM
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.engine.arg_utils import AsyncEngineArgs

from PIL import Image
from mineru_vl_utils import MinerUClient
from mineru_vl_utils import MinerULogitsProcessor  # if vllm>=0.10.1
from mineru_vl_utils.post_process import json2md


if __name__ == "__main__":
    setting = "decode compile -- async"
    # setting = "eager only"

    dataset_path = "/home/sankalok/July2026/datasets/OmniDocBenchV1.0/"
    dataset_files = [
        os.path.join(dataset_path, file)
        for file in os.listdir(dataset_path)
        if os.path.isfile(os.path.join(dataset_path, file))
    ]

    results_dir = "/home/sankalok/July2026/mineru_offline/results/"
    os.makedirs(results_dir, exist_ok=True)

    results_dir_path = Path(results_dir)

    # llm = LLM(
    #     model="/home/sankalok/models/MinerU2.5-Pro-2605-1.2B/",
    #     trust_remote_code=True,
    #     tensor_parallel_size=1,
    #     logits_processors=[MinerULogitsProcessor],
    #     max_model_len=8192,
    #     gpu_memory_utilization=0.9,
    #     dtype="float16",
    #     # enforce_eager=True,
    #     additional_config={
    #         "ascend_compilation_config": {
    #             "fuse_norm_quant": False,
    #         }
    #     },
    #     compilation_config={
    #         "cudagraph_mode": "FULL_DECODE_ONLY",
    #         "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32],
    #     },
    #     hf_overrides={
    #         "tie_word_embeddings": True,
    #         "text_config": {
    #             "tie_word_embeddings": True
    #         }
    #     },
    #     enable_prefix_caching=False,  # if vllm>=0.10.1
    # )

    llm = AsyncLLM.from_engine_args(
        AsyncEngineArgs(
            model="/home/sankalok/models/MinerU2.5-Pro-2605-1.2B/",
            trust_remote_code=True,
            tensor_parallel_size=1,
            logits_processors=[MinerULogitsProcessor],
            max_model_len=8192,
            gpu_memory_utilization=0.9,
            dtype="float16",
            max_num_seqs=512,
            max_num_batched_tokens=16384,
            disable_log_stats=False,
            # enforce_eager=True,
            additional_config={
                "ascend_compilation_config": {
                    "fuse_norm_quant": False,
                    "enable_npugraph_ex": True,
                    "enable_static_kernel": True,
                }
            },
            compilation_config={
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [
                    1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 28, 32,
                ],
            },
            hf_overrides={
                "tie_word_embeddings": True,
                "text_config": {
                    "tie_word_embeddings": True
                }
            },
            enable_prefix_caching=True,  # if vllm>=0.10.1
        )
    )

    # client = MinerUClient(
    #     backend="vllm-engine",
    #     vllm_llm=llm,
    #     batch_size=0,
    #     image_analysis=False,  # default False, set True to enable image/chart analysis
    # )

    start_time = time.time()

    client = MinerUClient(
        backend="vllm-async-engine",
        vllm_async_llm=llm,
        batch_size=0,
        image_analysis=True,  # default False, set True to enable image/chart analysis
    )

    images = [
        Image.open(file)
        for file in tqdm(dataset_files, desc="Extracting Images")
    ]
    images = images[:10]
    content_lists = client.concurrent_two_step_extract(images)

    iteration = 0

    for i in tqdm(range(len(content_lists)), desc="json2md"):
        markdown_result = json2md(content_lists[i])

        result_path_filename = dataset_files[i].replace(".jpg", ".md")
        result_path_filename = result_path_filename.split("/")[-1]
        with open(
            results_dir_path / result_path_filename,
            "w",
            encoding="utf-8",
        ) as f:
            f.write(markdown_result)
        iteration += 1

    end_time = time.time()
    total_time = end_time - start_time

    print("\n" * 5)
    print("------------------------------------------------------------")
    print(f"Setting:         {setting}")
    print(f"Total Time:      {total_time} seconds.")
    print(f"Total Pages:     {iteration} pages")
    print(f"Avg Page/Second: {iteration / total_time} page/second.")
    print("------------------------------------------------------------")

    # iteration = 1
    # total_time = 0.0
    # total_files = len(dataset_files)
    #
    # for file in tqdm(dataset_files, desc="OmniDocBench v1.0 Inference"):
    #     start_time = time.time()
    #     content_list = client.two_step_extract(Image.open(file))
    #     markdown_result = json2md(content_list)
    #     end_time = time.time()
    #     time_taken = end_time - start_time
    #
    #     total_time += time_taken
    #     print(
    #         f"Iteration: {iteration} | File: {file} | "
    #         f"Time Taken: {time_taken} seconds."
    #     )
    #     iteration += 1
    #
    #     result_path_filename = file.replace(".jpg", ".md")
    #     result_path_filename = result_path_filename.split("/")[-1]
    #     with open(
    #         results_dir_path / result_path_filename,
    #         "w",
    #         encoding="utf-8",
    #     ) as f:
    #         f.write(markdown_result)
    #
    # print("\n" * 5)
    # print("------------------------------------------------------------")
    # print(f"Setting:         {setting}")
    # print(f"Total Time:      {total_time} seconds.")
    # print(f"Avg Page/Second: {iteration / total_time} page/second.")
    # print("------------------------------------------------------------")
```

## Transcribed `remove_md.py`

The directory name is blurred in this part of the photograph. It appears to be
`July2026`, consistent with the other visible paths.

```python
import os
import glob

# specify the directory
dir_path = "/home/sankalok/July2026/datasets/OmniDocBenchV1.0/"

# get all .md files in the directory
md_files = glob.glob(os.path.join(dir_path, "*.md"))

# iterate over the files and remove them
for md_file in md_files:
    os.remove(md_file)
```

## Reproduction gaps visible in the screenshots

- The screenshots do not show the exact vLLM-Ascend plugin version or commit.
- The code uses `os.listdir()` without sorting or filtering by image extension.
- The current code limits the image list to 10 items. The claimed 981-page run
  therefore came from a different revision or runtime edit.
- The screenshots do not show warmup treatment, exact timing boundaries, NPU
  selection, engine logs, error counts, generated-file counts, output hashes,
  or OmniDocBench accuracy scoring.
- The reported eager and compiled lanes change execution mode, engine type,
  concurrency, prefix caching, and tuning together. The 5x difference is not an
  isolated measurement of compilation alone.
- The code uses `image_analysis=True`, so the reported number is intended as
  full two-step layout plus OCR throughput, not recognizer-only throughput.
