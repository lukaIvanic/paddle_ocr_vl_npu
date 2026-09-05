"""CPU-only complete preprocessor comparison; no model/NPU allocation or timing claim."""
import argparse
import io
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "09_persistent_page_engine"), str(ROOT / "09_persistent_page_engine/scripts"),
                str(ROOT / "09_persistent_page_engine/tests")]
from PIL import Image
import torch
import table_closed_loop_api_client as client
from paddleocr_vl.model import preprocessing as p
from test_preprocessing_normalization_lookup import reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = p.apply_pixel_overrides(p.load_preprocessor_config(args.model), min_pixels=28224, max_pixels=802816)
    selected = client.select_tables(SimpleNamespace(source_jsonl=client.load.DEFAULT_SOURCE,
                                   set="random", count=100, shuffle_seed=1))[:8]
    payloads = client.load.prepare_http_payloads(selected, args.images_dir)
    candidate = p._normalize_image_array
    results = []
    for item in selected:
        with Image.open(io.BytesIO(payloads[item["request_id"]])) as opened:
            image = opened.convert("RGB")
        with patch.object(p, "_normalize_image_array", reference):
            expected, grid = p.preprocess_pil_image(image, cfg)
        actual, actual_grid = p.preprocess_pil_image(image, cfg)
        assert torch.equal(expected, actual) and torch.equal(grid, actual_grid)
        times = {"reference": [], "lookup": []}
        # Interleave paired complete preprocessing calls; first two pairs warm
        # CPU allocation/Pillow paths. Payload preparation is outside this CPU probe.
        for iteration in range(7):
            order = [("reference", reference), ("lookup", candidate)]
            if iteration % 2:
                order.reverse()
            for label, fn in order:
                with patch.object(p, "_normalize_image_array", fn):
                    started = time.perf_counter()
                    p.preprocess_pil_image(image, cfg)
                    elapsed = time.perf_counter() - started
                if iteration >= 2:
                    times[label].append(elapsed)
        row = {"request_id": item["request_id"], "size": image.size, "grid": grid.tolist(),
               "bit_exact_float32": True, "samples_s": times,
               "median_s": {k: statistics.median(v) for k, v in times.items()}}
        results.append(row)
        print(json.dumps(row), flush=True)
    report = {"scope": "CPU only: existing Pillow resize + normalization + patchification; not serving E2E",
              "warmup_pairs": 2, "measured_pairs": 5, "rows": results}
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
