# GLM-OCR Decode Speedup Roadmap

This folder is a portable, self-contained version of the five decode speedup steps used in the presentation. It does not import from `experiments/` or from an external live GLM-OCR runner checkout. The minimal live GLM-OCR runner/model code used by these experiments is vendored under `speedup_roadmap/common/`.

## Dependencies

Use an Ascend Python environment with:

- CANN and driver sourced for the target NPU.
- `torch`, `torch_npu`, and TorchAir support through `torch_npu` / `torchair`.
- `safetensors`, `transformers`, `Pillow`, and `numpy`.
- GLM-OCR model directory containing `config.json`, a single `model.safetensors`, tokenizer/processor files, and GLM4V image-processor files.
- A test image reachable by `--image-path`.
- Optional profiling: `torch_npu.profiler` and CANN profiling tools.

The current 910B baseline used:

```text
torch==2.6.0+cpu
torch_npu==2.6.0.post5
transformers==4.57.6
safetensors==0.7.0
Pillow==11.3.0
numpy==1.26.4
```

Other versions can work, but if a new server fails early, compare these first.

On the current 910B remote baseline:

```sh
cd /research/aoe_research
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64:$LD_LIBRARY_PATH
source .venv/bin/activate
```

On a new server, the equivalent environment setup must make both CANN toolkit libraries and driver libraries visible. If Python fails with `libascend_hal.so: cannot open shared object file`, the driver `LD_LIBRARY_PATH` is missing.

Default model/image paths are set for the current remote portable GLM-OCR bundle:

```text
--model-dir       /root/autodl-tmp/glm_ocr_portable_bundle/models/GLM-OCR
--image-path      /root/autodl-tmp/glm_ocr_portable_bundle/data/OmniDocBench/images/book_zh_GB12082006_extracted_page_8.png
```

Override the model and image paths when copying this folder to a new server.

## Transfer Checklist

Copy these things to the new server:

```text
speedup_roadmap/
GLM-OCR model directory
one test image
```

Keep the folder name as `speedup_roadmap`. The scripts are designed to run as:

```sh
python speedup_roadmap/01_compile_baseline.py ...
```

from the parent directory. If the folder lives somewhere unusual, set:

```sh
export PYTHONPATH=/path/to/parent:$PYTHONPATH
```

Do not copy `experiments/`; this roadmap does not use it. Do not copy the old live GLM-OCR runner either; the minimal live runner is already vendored in:

```text
speedup_roadmap/common/run_local_glm_ocr.py
speedup_roadmap/common/local_modeling_glm_ocr.py
```

Important model-file assumption: the roadmap currently loads exactly:

```text
<model-dir>/model.safetensors
```

If the model is sharded into multiple `.safetensors` files, either merge it first or adapt the loader.

## First Smoke Checks

Before running the ladder, confirm the imports and NPU environment:

```sh
python - <<'PY'
import torch
import torch_npu
from speedup_roadmap.common.roadmap import import_torchair, load_live_modules

print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
print("npu available", torch_npu.npu.is_available())
print("device", torch_npu.npu.get_device_name(0))

torchair, CompilerConfig = import_torchair()
print("torchair", getattr(torchair, "__file__", torchair))
print("CompilerConfig", CompilerConfig)

_, _, live_items = load_live_modules()
print("runner module", live_items[1].__module__)
PY
```

Expected: `runner module` should be `speedup_roadmap.common.run_local_glm_ocr`. If it points to another checkout, `PYTHONPATH` or the working directory is wrong.

Check model/image paths:

```sh
test -f /path/to/GLM-OCR/config.json
test -f /path/to/GLM-OCR/model.safetensors
test -f /path/to/test.png
```

## Run The Ladder

Run from the repository root, or from a copied folder whose parent is on `PYTHONPATH`.

```sh
python speedup_roadmap/01_compile_baseline.py
python speedup_roadmap/02_overlap_eos_item.py
python speedup_roadmap/03a_mrope_once.py
python speedup_roadmap/03b_qkv_rmsnorm_rotary.py
python speedup_roadmap/03c_half_layout_rotary.py
```

For a copied server, pass paths explicitly:

```sh
MODEL=/path/to/GLM-OCR
IMAGE=/path/to/test.png

python speedup_roadmap/01_compile_baseline.py --model-dir "$MODEL" --image-path "$IMAGE"
python speedup_roadmap/02_overlap_eos_item.py --model-dir "$MODEL" --image-path "$IMAGE"
python speedup_roadmap/03a_mrope_once.py --model-dir "$MODEL" --image-path "$IMAGE"
python speedup_roadmap/03b_qkv_rmsnorm_rotary.py --model-dir "$MODEL" --image-path "$IMAGE"
python speedup_roadmap/03c_half_layout_rotary.py --model-dir "$MODEL" --image-path "$IMAGE"
```

Each script writes a manifest under:

```text
artifacts/speedup_roadmap/<step>/manifest.json
```

The expected speed progression on the current 910B baseline is roughly:

```text
01 eager live decode                         ~24 decode tok/s
01 flat compiled decode                      ~250 decode tok/s
02 overlapped EOS .item()                    ~350 decode tok/s
03a mRoPE once per decode step               ~425-432 decode tok/s
03b fused QKV + npu_rms_norm + no-slice rot  ~460-482 decode tok/s
03c half-layout + npu_rotary_mul half        ~500-507 decode tok/s
```

Exact tok/s can move a few percent across reruns or machines. Correctness should stay exact for the reported comparisons.

## Profiling

Profiling is off by default. Add `--profile` to any step:

```sh
python speedup_roadmap/03c_half_layout_rotary.py --profile --profile-steps 8 --profile-metric pipe
```

Keep `--profile-steps` at 8 or lower. Larger profiles become bulky quickly.

Useful common arguments:

```sh
--device 0
--cache-len 1024
--decode-steps 32
--warmup 1
--repeats 5
--model-dir /path/to/GLM-OCR
--image-path /path/to/test.png
```

Step 02 defaults to `--decode-steps 128` because it measures useful EOS stopping behavior. The other steps default to 32 decode steps.

## What Each Step Tests

- `01_compile_baseline.py`: compares the vendored live eager decode path against a flat Tensor-only compiled decode boundary.
- `02_overlap_eos_item.py`: keeps the compiled decode path but overlaps the EOS `.item()`/CPU synchronization so useful throughput stays near fixed-step speed.
- `03a_mrope_once.py`: computes multimodal RoPE factor selection once per decode step instead of once per layer.
- `03b_qkv_rmsnorm_rotary.py`: adds fused QKV projection, `torch_npu.npu_rms_norm`, and the no-full-width-slice rotary path.
- `03c_half_layout_rotary.py`: converts Q/K to half layout and uses `torch_npu.npu_rotary_mul(..., rotary_mode="half")`.

## Common Failure Points

- `ModuleNotFoundError: No module named 'torchair'`: this is okay if `torch_npu.dynamo.torchair` exists. The roadmap helper tries both import paths.
- `libascend_hal.so` missing: source CANN and export the driver library directories in `LD_LIBRARY_PATH`.
- `ImportError` around `Glm4vImageProcessor`: use a Transformers version with GLM4V support. The known-good baseline used `transformers==4.57.6`.
- `FileNotFoundError: model.safetensors`: the model directory is wrong or the weights are sharded. This roadmap expects one `model.safetensors`.
- NPU unavailable: do not fall back to CPU. Fix CANN/driver/`torch_npu` first.
- 310P vs 910B: correctness should still be checked, but tok/s and available operator/compiler behavior may differ.
