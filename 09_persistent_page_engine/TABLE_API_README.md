# Table OCR client

The crop OCR API must already be running.

For OmniDocBench, download [`OmniDocBench.json`](https://huggingface.co/datasets/MinerU25Pro-NIPS26/OmniDocBench-v1.6/resolve/main/OmniDocBench.json?download=true) and the [`images/` directory](https://huggingface.co/datasets/MinerU25Pro-NIPS26/OmniDocBench-v1.6/tree/main/images). The expected JSON SHA-256 is `a45cd84b04ad8b793e775089640e6b681209abea33ead54c1828ddca35fae496`. The expected aggregate hash for the 1,651 images is `58feeb96c60fcfab12ba4348c4e093ceaf1b707658dbfd0e08c24d7821d4c221`.

Clone the [OmniDocBench evaluator](https://github.com/opendatalab/OmniDocBench) and use commit `2b161d010d2e3aff77a0edef359ea3a6411d23cd`.

```sh
python3 -m pip install Pillow==10.4.0 apted==1.0.3 lxml==4.9.1 tqdm==4.67.1 Levenshtein==0.25.1 rapidfuzz==3.14.5 beautifulsoup4==4.11.1 pylatexenc==2.10 pandas==2.0.3 evaluate==0.4.3 numpy==1.24.4 matplotlib==3.7.5 datasets==5.0.0 huggingface-hub==1.22.0
```

Run the OmniDocBench table test:

```sh
python3 09_persistent_page_engine/scripts/run_omnidocbench_table_api.py \
  --omnidocbench \
  --dataset-json /path/to/OmniDocBench.json \
  --images-dir /path/to/images \
  --evaluator-root /path/to/OmniDocBench \
  --api-url http://API_HOST:8765/v1/ocr \
  --output-dir output/omnidocbench_tables
```

Run one or more table crops:

```sh
python3 09_persistent_page_engine/scripts/run_omnidocbench_table_api.py \
  --images table1.png table2.jpg \
  --crop-type table \
  --api-url http://API_HOST:8765/v1/ocr \
  --output-dir output/tables
```
