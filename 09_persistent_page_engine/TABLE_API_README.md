# Table OCR client

Use these three files:

- `scripts/run_omnidocbench_table_api.py`
- `TABLE_API_README.md`
- `example_table.png`

The crop OCR API must already be running.

For OmniDocBench, download [`OmniDocBench.json`](https://huggingface.co/datasets/MinerU25Pro-NIPS26/OmniDocBench-v1.6/resolve/main/OmniDocBench.json?download=true) and the [`images/` directory](https://huggingface.co/datasets/MinerU25Pro-NIPS26/OmniDocBench-v1.6/tree/main/images). The client checks the inputs before inference. It prints a warning and continues if they do not match the expected 1,651-image dataset.

Expected dataset fingerprints:

- `OmniDocBench.json` SHA-256: `a45cd84b04ad8b793e775089640e6b681209abea33ead54c1828ddca35fae496`
- 1,651 referenced images
- Referenced-images aggregate SHA-256: `58feeb96c60fcfab12ba4348c4e093ceaf1b707658dbfd0e08c24d7821d4c221`

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

Expected reference result:

- 665 tables on 458 pages
- Page-TEDS: approximately `0.954`
- Sample TEDS: approximately `0.949`
- Page structure-only TEDS: approximately `0.978`
- TEDS timeouts: `0`
- TEDS errors: `0`

Small numerical differences are normal. A timeout receives a score of zero and
makes the reported score too low. If the summary reports a timeout, follow the
printed `--score-only` command with a larger timeout. Do not rerun OCR.

Run the included table example:

```sh
python3 09_persistent_page_engine/scripts/run_omnidocbench_table_api.py \
  --images 09_persistent_page_engine/example_table.png \
  --crop-type table \
  --api-url http://API_HOST:8765/v1/ocr \
  --output-dir output/example_table
```

The expected result is a seven-row HTML table: two header rows and five process
rows. It must contain the headings
`Available`, `Processes`, `Allocation`, and `Max`; the process rows `P0` through
`P4`; and the values shown in `example_table.png`. The first cell of the final
four rows must use `rowspan="4" colspan="4"`. The client saves the exact model
output in `output/example_table/results.md` and prints it in the terminal.

Expected model output (whitespace can differ):

```html
<table><tr><td colspan="4">Available</td><td rowspan="2">Processes</td><td colspan="4">Allocation</td><td colspan="4">Max</td></tr><tr><td>A</td><td>B</td><td>C</td><td>D</td><td>A</td><td>B</td><td>C</td><td>D</td><td>A</td><td>B</td><td>C</td><td>D</td></tr><tr><td>1</td><td>5</td><td>2</td><td>0</td><td>P0</td><td>0</td><td>0</td><td>1</td><td>2</td><td>0</td><td>0</td><td>1</td><td>2</td></tr><tr><td rowspan="4" colspan="4"></td><td>P1</td><td>1</td><td>0</td><td>0</td><td>0</td><td>1</td><td>7</td><td>5</td><td>0</td></tr><tr><td>P2</td><td>1</td><td>3</td><td>5</td><td>4</td><td>2</td><td>3</td><td>5</td><td>6</td></tr><tr><td>P3</td><td>0</td><td>6</td><td>3</td><td>2</td><td>0</td><td>6</td><td>5</td><td>2</td></tr><tr><td>P4</td><td>0</td><td>0</td><td>1</td><td>4</td><td>0</td><td>6</td><td>5</td><td>6</td></tr></table>
```

For another table crop, replace `example_table.png` with any PNG or JPEG file.
