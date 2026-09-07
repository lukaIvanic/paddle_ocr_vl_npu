"""Freeze exact live crop pixels and baseline raw outputs for selective cap A/B."""
import argparse
from collections import Counter, defaultdict
import inspect
import json
from pathlib import Path
import sys

from generation_trace import image_fingerprint
from saved_paddle_crops import sha256, SKIP_TYPES


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def selected(row, image_token_id, max_pixels):
    if row["phase"] != "recognition":
        raise ValueError("expected a recognition-only live baseline")
    return row["prompt_token_ids"].count(image_token_id) * 4 * 196 > max_pixels


def baseline_helper():
    from mineru_vl_utils import MinerUClient
    from mineru_vl_utils.mineru_client import MinerUClientHelper
    defaults = {k: p.default for k, p in inspect.signature(MinerUClient.__init__).parameters.items()}
    values = dict(backend="transformers", prompts={}, sampling_params={}, image_analysis=False)
    for name, param in inspect.signature(MinerUClientHelper).parameters.items():
        if name not in values:
            value = defaults.get(name, param.default)
            if value is inspect.Parameter.empty:
                raise ValueError(f"cannot establish helper default {name}")
            values[name] = value
    return MinerUClientHelper(**values)


def prepare(reference_run, output, max_pixels):
    from PIL import Image
    from mineru_vl_utils.structs import ContentBlock
    from mineru_vl_utils.post_process import json2md
    sys.path.append(str(Path(__file__).resolve().parents[1] / "09_persistent_page_engine"))
    from pipeline.layout_frontend import _decode_rgb
    from pipeline.layout_postprocess import crop_layout_regions
    reference_run, output = reference_run.resolve(), output.resolve()
    summary_file = reference_run / "output/run_summary_shard_00.json"
    summary = json.loads(summary_file.read_text())
    if summary["completed"] != 1651 or summary["failed"] or summary["layout_backend"] != "pp-doclayout-v3":
        raise ValueError("expected successful full live-layout reference")
    if not summary["processor_min_pixels"] <= max_pixels < summary["processor_max_pixels"]:
        raise ValueError("invalid lower cap")
    model = Path(summary["model"])
    for filename, digest in summary["model_hashes"].items():
        path = Path(summary["dataset_json"]) if filename == "dataset_json" else model / filename
        if sha256(path) != digest:
            raise ValueError(f"reference asset hash mismatch: {path}")
    config = json.loads((model / "config.json").read_text())
    processor = json.loads((model / "preprocessor_config.json").read_text())
    if (processor["patch_size"], processor["merge_size"]) != (14, 2):
        raise ValueError("unexpected model image patch geometry")
    trace = reference_run / "output/generation_trace.jsonl"
    rows = defaultdict(dict)
    for line in trace.open():
        row = json.loads(line)
        name, index = row["page"], row["block_index"]
        if index in rows[name]:
            raise ValueError("duplicate baseline request")
        rows[name][index] = dict(row, rerun=selected(row, config["image_token_id"], max_pixels))
    dataset = json.loads(Path(summary["dataset_json"]).read_text())
    output.mkdir(parents=True, exist_ok=False)
    helper = baseline_helper()
    pages, selected_records, reconstruction_mismatches = [], [], []
    for page_index, data in enumerate(dataset):
        name = Path(data["page_info"]["image_path"]).name
        geometry_path = reference_run / "output/layout_regions" / f"{Path(name).stem}.json"
        geometry = json.loads(geometry_path.read_text())
        if geometry["image_name"] != name:
            raise ValueError("geometry page mismatch")
        if set(rows.get(name, {})) != {b["index"] for b in geometry["blocks"] if b["recognize"]}:
            raise ValueError("trace/geometry membership mismatch")
        pixels = None
        blocks, seed = [], []
        for index, original in enumerate(geometry["blocks"]):
            record = dict(original)
            if index != record["index"] or record["recognize"] != (record["type"] not in SKIP_TYPES):
                raise ValueError("invalid live geometry")
            row = rows.get(name, {}).get(index)
            record.update(rerun=bool(row and row["rerun"]), baseline_raw_text=row["raw_text"] if row else None)
            seed.append(ContentBlock(type=record["type"], bbox=record["bbox"], angle=None,
                                     content=record["baseline_raw_text"]))
            if row:
                seed[-1].scored = None
                if (row["block_type"], row["bbox"]) != (record["type"], record["bbox"]):
                    raise ValueError("trace/geometry identity mismatch")
            if record["rerun"]:
                if pixels is None:
                    pixels, _ = _decode_rgb(Path(summary["images_dir"]) / name)
                    if list(pixels.shape[:2]) != [geometry["height"], geometry["width"]]:
                        raise ValueError("page size mismatch")
                crop = Image.fromarray(crop_layout_regions(pixels, [{
                    "coordinate": record["bbox_pixels"], "polygon_points": record["polygon_points"],
                    "label": record["source_label"]}])[0]["img"])
                if image_fingerprint(crop) != record["crop_pixel_sha256"]:
                    raise ValueError(f"regenerated crop differs from baseline: {name}:{index}")
                if image_fingerprint(helper.resize_by_need(crop)) != row["image_sha256"]:
                    raise ValueError("helper image mismatch")
                relative = f"crops/{page_index:04d}/{index:04d}.png"
                target = output / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                crop.save(target)
                record.update(crop=relative, crop_sha256=sha256(target), recognition_image_sha256=row["image_sha256"])
                selected_records.append({k: row[k] for k in ("request_id", "page", "block_index", "block_type")}
                                        | {"original_vision_tokens": row["prompt_token_ids"].count(config["image_token_id"])*4})
            blocks.append(record)
        reconstructed = json2md(helper.post_process(seed))
        if reconstructed != (reference_run / "output/predictions" / f"{Path(name).stem}.md").read_text():
            reconstruction_mismatches.append(name)
        pages.append({"image_name": name, "geometry_sha256": sha256(geometry_path), "blocks": blocks})
        if (page_index + 1) % 100 == 0:
            print(f"FREEZE {page_index+1}/1651 selected={len(selected_records)}", flush=True)
    if reconstruction_mismatches:
        dump(output / "reconstruction_mismatches.json", reconstruction_mismatches)
        raise ValueError("baseline raw-output reconstruction must be Markdown-exact")
    if not selected_records:
        raise ValueError("no affected crops")
    manifest = {"schema": "mineru_selective_crop_replay_v1", "reference_run": str(reference_run),
                "reference_summary_sha256": sha256(summary_file), "reference_trace_sha256": sha256(trace),
                "candidate_max_pixels": max_pixels, "baseline_max_pixels": summary["processor_max_pixels"],
                "baseline_reconstruction_exact_pages": len(pages), "pages": pages}
    dump(output / "manifest.json", manifest)
    selected_pages = {r["page"] for r in selected_records}
    dump(output / "selection.json", {"selected_crops": len(selected_records), "selected_pages": len(selected_pages),
         "by_type": dict(Counter(r["block_type"] for r in selected_records)), "crops": selected_records})
    smoke_names = [p["image_name"] for p in pages if p["image_name"] in selected_pages][:2]
    dump(output / "smoke_dataset.json", [r for r in dataset if Path(r["page_info"]["image_path"]).name in smoke_names])
    print(f"READY crops={len(selected_records)} pages={len(selected_pages)} reconstruction_exact={len(pages)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-pixels", type=int, default=602112)
    args = parser.parse_args()
    prepare(args.reference_run, args.output, args.max_pixels)
