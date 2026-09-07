"""Audit live layout geometry/crop pixels against the frozen hybrid experiment."""
import argparse
import json
from pathlib import Path

from generation_trace import image_fingerprint


def compare(live_output, saved_manifest, saved_output):
    from PIL import Image
    saved = {p["image_name"]: p for p in json.loads(saved_manifest.read_text())["pages"]}
    pages = []
    for file in sorted((live_output / "layout_regions").glob("*.json")):
        live = json.loads(file.read_text())
        name = live["image_name"]
        expected = [b for b in saved[name]["blocks"] if b["recognize"]]
        actual = [b for b in live["blocks"] if b["recognize"]]
        geometry = lambda rows: [(r["source_label"], r["bbox_pixels"]) for r in rows]
        same_geometry = geometry(expected) == geometry(actual)
        same_pixels = False
        if same_geometry:
            pixel_matches = []
            for a, b in zip(actual, expected):
                with Image.open(saved_manifest.parent / b["crop"]) as crop:
                    pixel_matches.append(a["crop_pixel_sha256"] == image_fingerprint(crop.convert("RGB")))
            same_pixels = all(pixel_matches)
        prediction = Path(name).stem + ".md"
        text_equal = (live_output / "predictions" / prediction).read_bytes() == (saved_output / "predictions" / prediction).read_bytes()
        pages.append({"image": name, "live_crops": len(actual), "saved_crops": len(expected),
                      "ordered_geometry_exact": same_geometry, "crop_pixels_exact": same_pixels,
                      "markdown_exact": text_equal})
    summary = json.loads((live_output / "run_summary_shard_00.json").read_text())
    if len(pages) != summary["completed"] or summary["failed"]:
        raise ValueError("incomplete live page evidence")
    return {"pages": pages, "page_count": len(pages),
            "ordered_geometry_exact_pages": sum(p["ordered_geometry_exact"] for p in pages),
            "crop_pixels_exact_pages": sum(p["crop_pixels_exact"] for p in pages),
            "markdown_exact_pages": sum(p["markdown_exact"] for p in pages),
            "layout_calls": summary["streaming"]["layout_calls"],
            "recognition_requests": summary["streaming"]["requests_completed_by_phase"],
            "pipeline_wall_s": summary["pipeline_wall_s"]}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--live-output", type=Path, required=True)
    p.add_argument("--saved-manifest", type=Path, required=True)
    p.add_argument("--saved-output", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = compare(args.live_output, args.saved_manifest, args.saved_output)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
