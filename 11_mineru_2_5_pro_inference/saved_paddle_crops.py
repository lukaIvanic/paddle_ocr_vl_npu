"""Freeze PP-DocLayoutV3 geometry as recognition inputs, never its OCR content.

This tests final saved Paddle regions, not byte-identical historical Paddle
recognizer inputs (which could be merged/resized). PNGs retain original pixels,
with the same white polygon mask as experiment 09. No GT geometry is consulted.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path


LABEL_MAP = {
    "text": "text", "abstract": "text", "content": "text", "vertical_text": "text",
    "doc_title": "title", "paragraph_title": "title", "display_formula": "equation",
    "inline_formula": "equation", "table": "table", "number": "page_number",
    "header": "header", "footer": "footer", "aside_text": "aside_text",
    "formula_number": "formula_number", "reference_content": "ref_text",
    "footnote": "page_footnote", "algorithm": "algorithm",
    # Paddle captions do not identify table-vs-image ownership. Keep as text;
    # do not infer ownership from geometry or ground truth in this first test.
    "figure_title": "text", "vision_footnote": "text",
    "image": "image", "header_image": "image", "footer_image": "image",
    "seal": "image", "chart": "chart",
}
SKIP_TYPES = {"image", "chart"}  # Matched to current MinerU image_analysis=False.


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def geometry_record(block, width, height, index):
    """Explicit allowlist prevents saved Paddle text or GT leaking into inputs."""
    label = block["block_label"]
    kind = LABEL_MAP[label]  # Unknown labels must fail, not silently disappear.
    bbox = list(block["block_bbox"])
    if len(bbox) != 4 or any(int(x) != x for x in bbox):
        raise ValueError(f"non-integral bbox: {bbox}")
    x1, y1, x2, y2 = bbox = list(map(int, bbox))
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"invalid bbox: {bbox} in {width}x{height}")
    return {"index": index, "source_label": label, "type": kind,
            "bbox_pixels": bbox, "bbox": [x1 / width, y1 / height, x2 / width, y2 / height],
            "polygon_points": block["block_polygon_points"],
            "source_block_id": block["block_id"], "source_order": block.get("block_order"),
            "source_group_id": block.get("group_id"), "recognize": kind not in SKIP_TYPES}


def export(regions, dataset_json, images_dir, output, workers=8):
    import numpy as np
    from PIL import Image
    module_path = Path(__file__).resolve().parents[1] / "09_persistent_page_engine/pipeline/layout_postprocess.py"
    spec = importlib.util.spec_from_file_location("saved_paddle_geometry", module_path)
    geometry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geometry)
    rows = [json.loads(line) for line in regions.read_text().splitlines() if line.strip()]
    by_name = {row["image_name"]: row for row in rows}
    dataset = json.loads(dataset_json.read_text())
    names = [Path(row["page_info"]["image_path"]).name for row in dataset]
    if len(by_name) != len(rows) or len(set(names)) != len(names) or set(names) != set(by_name):
        raise ValueError("saved geometry must cover dataset exactly once")
    output.mkdir(parents=True, exist_ok=False)

    def one(item):
        page_index, name = item
        row = by_name[name]
        path = images_dir / name
        with Image.open(path) as source:
            image = source.convert("RGB")
        if image.size != (row["width"], row["height"]):
            raise ValueError(f"page dimension mismatch: {name}")
        pixels = np.asarray(image)
        records = []
        for index, block in enumerate(row["parsing_res_list"]):
            record = geometry_record(block, *image.size, index)
            if record["recognize"]:
                crop = geometry.crop_layout_regions(pixels, [{
                    "coordinate": record["bbox_pixels"], "label": record["source_label"],
                    "polygon_points": record["polygon_points"]}])[0]["img"]
                relative = f"crops/{page_index:04d}/{index:04d}.png"
                target = output / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(crop).save(target)
                record.update(crop=relative, crop_sha256=sha256(target),
                              crop_size=[int(crop.shape[1]), int(crop.shape[0])])
            records.append(record)
        return {"image_name": name, "dataset_index": page_index,
                "image_sha256": sha256(path), "width": image.width, "height": image.height,
                "blocks": records}

    pages = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for page in executor.map(one, enumerate(names)):
            pages.append(page)
            if len(pages) % 100 == 0:
                print(f"[crop-export] {len(pages)}/{len(names)}", flush=True)
    manifest = {"schema_version": 1, "layout_source": "PP-DocLayoutV3_saved_final_regions",
                "regions_path": str(regions.resolve()), "regions_sha256": sha256(regions),
                "dataset_sha256": sha256(dataset_json), "label_map": LABEL_MAP,
                "crop_policy": "original RGB pixels; experiment09 white polygon mask; no crop merging/rescaling",
                "order_policy": "saved parsing_res_list order, not a new bbox sort",
                "saved_ocr_content_used": False, "image_analysis": False,
                "label_counts": dict(Counter(b["source_label"] for p in pages for b in p["blocks"])),
                "recognition_crops": sum(b["recognize"] for p in pages for b in p["blocks"]),
                "pages": pages}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    # Selection is by filename only; annotations are copied for the evaluator,
    # never loaded by the recognition source.
    smoke_names = set(names[:2]) | {
        "notes_f7f010b78016aeebd76e56d9283eb67f_72.jpg",
        "jiaocaineedrop_jiaocai_needrop_en_349.jpg",
        "scihub_fmicb.2019.01892.pdf_2.jpg",
        "book_en_[搬书匠#893][Pyomo—Optimization Modeling in Python].2012.英文版_page_016.png",
    }
    for label in ("table", "display_formula"):
        smoke_names.add(next(p["image_name"] for p in pages if any(b["source_label"] == label for b in p["blocks"])))
    missing = smoke_names - set(names)
    if missing:
        raise ValueError(f"smoke pages missing: {missing}")
    smoke = [row for row, name in zip(dataset, names) if name in smoke_names]
    (output / "smoke_dataset.json").write_text(json.dumps(smoke, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "pages"}, indent=2), flush=True)
    print(f"EXPORTED pages={len(pages)} smoke_pages={len(smoke)} manifest={output / 'manifest.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regions", type=Path, required=True)
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    export(args.regions, args.dataset_json, args.images_dir, args.output, args.workers)


if __name__ == "__main__":
    main()
