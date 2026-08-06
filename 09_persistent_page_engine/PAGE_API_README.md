# Full-page OCR API benchmark client

Use this client when the full-page OCR API is already running:

- `scripts/run_omnidocbench_page_api_eval.py`

This is the only Python file from this repository that the product system
needs. It contains the HTTP client, robust matching runner, Page-TEDS runner,
and CDM runner. It does not import or execute another file from this
repository.

The command sends all 1,651 OmniDocBench pages to the API. It saves the page
Markdown, runs the official matching metrics, and runs Page-TEDS. Formula CDM
is off by default because it needs a large system environment.

## One-time evaluator setup

Download [`OmniDocBench.json`](https://huggingface.co/datasets/MinerU25Pro-NIPS26/OmniDocBench-v1.6/resolve/main/OmniDocBench.json?download=true)
and the [`images/` directory](https://huggingface.co/datasets/MinerU25Pro-NIPS26/OmniDocBench-v1.6/tree/main/images).

Validated dataset fingerprints:

- `OmniDocBench.json` SHA-256: `a45cd84b04ad8b793e775089640e6b681209abea33ead54c1828ddca35fae496`
- Referenced images: `1,651`
- Referenced-images aggregate SHA-256: `58feeb96c60fcfab12ba4348c4e093ceaf1b707658dbfd0e08c24d7821d4c221`

Clone the unmodified upstream evaluator and select the validated commit:

```sh
git clone https://github.com/opendatalab/OmniDocBench.git /workspace/repos/OmniDocBench_eval
git -C /workspace/repos/OmniDocBench_eval checkout 2b161d010d2e3aff77a0edef359ea3a6411d23cd
```

Create a Python 3.10 environment and install the public evaluator. Its pinned
`pyproject.toml` installs all Python packages from PyPI.

```sh
python3.10 -m venv /workspace/venvs/omnidocbench_py310
/workspace/venvs/omnidocbench_py310/bin/python -m pip install --upgrade pip
/workspace/venvs/omnidocbench_py310/bin/python -m pip install \
  --no-build-isolation /workspace/repos/OmniDocBench_eval
```

Set two paths for the commands below:

```sh
export OMNIDOCBENCH_EVALUATOR_ROOT=/workspace/repos/OmniDocBench_eval
export OMNIDOCBENCH_EVAL_PYTHON=/workspace/venvs/omnidocbench_py310/bin/python
```

## Run the full benchmark

```sh
"$OMNIDOCBENCH_EVAL_PYTHON" \
  09_persistent_page_engine/scripts/run_omnidocbench_page_api_eval.py \
  --api-url http://API_HOST:8766/v1/pages \
  --dataset-json /path/to/OmniDocBench.json \
  --images-dir /path/to/images \
  --evaluator-root "$OMNIDOCBENCH_EVALUATOR_ROOT" \
  --output-dir output/full_page_api
```

The benchmark uses 64 independent HTTP requests. The server schedules layout,
prefill, and decode work. The server stays ready after the benchmark. You can
send more pages or run another benchmark without restarting it.

The script checks the dataset and evaluator before inference. A dataset mismatch
prints a red warning and continues. An evaluator commit or source mismatch stops
the run unless `--allow-evaluator-mismatch` is supplied.

Expected 910B2 reference:

- Throughput: approximately `1.951 pages/s`
- Text-block Edit distance: approximately `0.0507`
- Official text score: approximately `0.9493`
- Display-formula Edit distance: approximately `0.0903`
- Table Page-TEDS: approximately `0.9444`
- TEDS timeouts/errors: `0`

## Optional formula CDM and Overall score

Supply `--do-cdm` only when Formula Page-CDM and the official three-part
Overall score are required. Without this option, the script does not check for
or use TeX Live, ImageMagick, or Ghostscript.

CDM requires these public system tools on `PATH`:

- [TeX Live](https://tug.org/texlive/) with `pdflatex`, `kpsewhich`,
  `CJK.sty`, and `c70gkai.fd`;
- [ImageMagick 7](https://imagemagick.org/script/download.php) with PDF
  support and the `magick` command;
- [Ghostscript](https://ghostscript.com/releases/gsdnld.html) with the `gs`
  command.

The validated CDM environment used TeX Live 2025, ImageMagick 7.1.1-47, and
Ghostscript 9.55.0. When it is available, run:

```sh
"$OMNIDOCBENCH_EVAL_PYTHON" \
  09_persistent_page_engine/scripts/run_omnidocbench_page_api_eval.py \
  --do-cdm \
  --api-url http://API_HOST:8766/v1/pages \
  --dataset-json /path/to/OmniDocBench.json \
  --images-dir /path/to/images \
  --evaluator-root "$OMNIDOCBENCH_EVALUATOR_ROOT" \
  --output-dir output/full_page_api_with_cdm
```

Expected CDM reference:

- Formula Page-CDM: approximately `0.9741`
- Official Overall: approximately `95.5933%`
- CDM timeouts/errors: `0`

Small numerical differences across devices are normal. The official Overall is:

```text
mean(1 - text-block Edit distance, formula Page-CDM, table Page-TEDS)
```

Do not use sample-CDM in this formula.

## Rescore without rerunning OCR

If evaluation was interrupted, keep the same output directory and run:

```sh
"$OMNIDOCBENCH_EVAL_PYTHON" \
  09_persistent_page_engine/scripts/run_omnidocbench_page_api_eval.py \
  --score-only \
  --dataset-json /path/to/OmniDocBench.json \
  --images-dir /path/to/images \
  --evaluator-root "$OMNIDOCBENCH_EVALUATOR_ROOT" \
  --output-dir output/full_page_api
```

Add `--do-cdm` to this command only when the CDM environment is available.

The final files are:

- `output/full_page_api/benchmark_summary.md`
- `output/full_page_api/benchmark_summary.json`
- `output/full_page_api/generation/predictions/`
- `output/full_page_api/evaluation/`

The evaluator speedups are embedded in the single client script. The pinned
upstream OmniDocBench source remains unchanged.
