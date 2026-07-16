"""Sequential full-page orchestration for Experiment 08."""

from __future__ import annotations

import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from engine import SequentialRecognizer
from layout import PPDocLayoutV3Runtime
from schema import PageResult, SkippedRegion, per_second


IMAGE_LABELS = {"image", "header_image", "footer_image"}
MARKDOWN_IGNORE_LABELS = {
    "number",
    "footnote",
    "header",
    "header_image",
    "footer",
    "footer_image",
    "aside_text",
}


def recognition_prompt(label: str) -> str:
    if label == "table":
        return "Table Recognition:"
    if label == "chart":
        return "Chart Recognition:"
    if "formula" in label and label != "formula_number":
        return "Formula Recognition:"
    if label == "spotting":
        return "Spotting:"
    if label == "seal":
        return "Seal Recognition:"
    return "OCR:"


class OfflinePagePipeline:
    def __init__(
        self,
        *,
        layout: PPDocLayoutV3Runtime,
        recognizer: SequentialRecognizer,
        recognize_chart: bool = False,
        recognize_seal: bool = False,
        recognize_image: bool = False,
        max_regions: int | None = None,
        artifact_dir: Path | None = None,
        save_crops: bool = False,
        save_annotated: bool = True,
    ):
        self.layout = layout
        self.recognizer = recognizer
        self.recognize_chart = bool(recognize_chart)
        self.recognize_seal = bool(recognize_seal)
        self.recognize_image = bool(recognize_image)
        self.max_regions = max_regions
        self.artifact_dir = artifact_dir
        self.save_crops = bool(save_crops)
        self.save_annotated = bool(save_annotated)

    def _skip_reason(self, label: str) -> str | None:
        if label in IMAGE_LABELS and not self.recognize_image:
            return "official_default_skips_image_blocks"
        if label == "chart" and not self.recognize_chart:
            return "official_default_disables_chart_recognition"
        if label == "seal" and not self.recognize_seal:
            return "official_default_disables_seal_recognition"
        return None

    def run_page(self, image_path: Path, page_index: int) -> PageResult:
        page_started = time.perf_counter()
        image_path = image_path.expanduser().resolve()
        started = time.perf_counter()
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        image_load_s = time.perf_counter() - started

        page_id = f"page_{page_index:04d}_{image_path.stem}"
        layout_regions, layout_timing = self.layout.predict(image)
        recognized = []
        skipped = []
        crop_total_s = 0.0
        recognition_started = time.perf_counter()
        recognized_count = 0

        crop_dir = None
        if self.artifact_dir is not None and self.save_crops:
            crop_dir = self.artifact_dir / "crops" / page_id
            crop_dir.mkdir(parents=True, exist_ok=True)

        for region in layout_regions:
            skip_reason = self._skip_reason(region.label)
            if skip_reason is None and self.max_regions is not None and recognized_count >= self.max_regions:
                skip_reason = "max_regions_debug_limit"
            if skip_reason is not None:
                skipped.append(
                    SkippedRegion(
                        layout_order=region.order,
                        label=region.label,
                        reason=skip_reason,
                        box=region.box,
                    )
                )
                continue

            started = time.perf_counter()
            left = max(0, min(image.width, math.floor(region.box.x0)))
            top = max(0, min(image.height, math.floor(region.box.y0)))
            right = max(0, min(image.width, math.ceil(region.box.x1)))
            bottom = max(0, min(image.height, math.ceil(region.box.y1)))
            if right <= left or bottom <= top:
                skipped.append(
                    SkippedRegion(
                        layout_order=region.order,
                        label=region.label,
                        reason="empty_clamped_crop",
                        box=region.box,
                    )
                )
                continue
            crop = image.crop((left, top, right, bottom))
            crop_total_s += time.perf_counter() - started
            if crop_dir is not None:
                crop.save(crop_dir / f"{region.order:03d}_{region.label}.png")

            result = self.recognizer.recognize(
                request_id=f"{page_id}_region_{region.order:03d}",
                layout_order=region.order,
                label=region.label,
                prompt=recognition_prompt(region.label),
                box=region.box,
                crop=crop,
            )
            recognized.append(result)
            recognized_count += 1

        recognition_wall_s = time.perf_counter() - recognition_started
        started = time.perf_counter()
        reading_order_text = "\n\n".join(
            result.text.strip()
            for result in recognized
            if result.label not in MARKDOWN_IGNORE_LABELS and result.text.strip()
        )
        postprocess_s = time.perf_counter() - started

        if self.artifact_dir is not None:
            page_dir = self.artifact_dir / "pages"
            page_dir.mkdir(parents=True, exist_ok=True)
            (page_dir / f"{page_id}.txt").write_text(reading_order_text + "\n", encoding="utf-8")
            if self.save_annotated:
                annotated = image.copy()
                draw = ImageDraw.Draw(annotated)
                for region in layout_regions:
                    draw.rectangle(region.box.as_list(), outline="red", width=3)
                    draw.text(
                        (region.box.x0 + 3, region.box.y0 + 3),
                        f"{region.order} {region.label} {region.score:.2f}",
                        fill="red",
                        stroke_width=2,
                        stroke_fill="white",
                    )
                annotated.save(page_dir / f"{page_id}_layout.png")

        generated_tokens = sum(item.generated_tokens_including_eos for item in recognized)
        decode_tokens = sum(item.decode_tokens_after_prefill_including_eos for item in recognized)
        decode_wall = sum(item.timing_s["compiled_decode_wall"] for item in recognized)
        page_total_s = time.perf_counter() - page_started
        timing = {
            "image_load": float(image_load_s),
            **layout_timing,
            "crop_extraction": float(crop_total_s),
            "sequential_recognition_wall": float(recognition_wall_s),
            "reading_order_text_postprocess": float(postprocess_s),
            "page_total": float(page_total_s),
        }
        partial = any(item.reason == "max_regions_debug_limit" for item in skipped)
        return PageResult(
            page_id=page_id,
            image_path=image_path,
            image_size=image.size,
            layout_regions=layout_regions,
            recognized_regions=recognized,
            skipped_regions=skipped,
            reading_order_text=reading_order_text,
            timing_s=timing,
            rates={
                "layout_regions_per_s": per_second(len(layout_regions), layout_timing["layout_inference"]),
                "recognition_regions_per_s": per_second(len(recognized), recognition_wall_s),
                "decode_effective_tok_per_s": per_second(decode_tokens, decode_wall),
                "page_output_tok_per_s": per_second(generated_tokens, page_total_s),
            },
            partial=partial,
        )


def aggregate_pages(pages: list[PageResult]) -> dict[str, Any]:
    recognized = [region for page in pages for region in page.recognized_regions]
    layout_regions = [region for page in pages for region in page.layout_regions]
    skipped = [region for page in pages for region in page.skipped_regions]
    page_wall = sum(page.timing_s["page_total"] for page in pages)
    decode_wall = sum(region.timing_s["compiled_decode_wall"] for region in recognized)
    decode_tokens = sum(region.decode_tokens_after_prefill_including_eos for region in recognized)
    output_tokens = sum(region.generated_tokens_including_eos for region in recognized)
    return {
        "pages": len(pages),
        "partial_pages": sum(1 for page in pages if page.partial),
        "layout_regions": len(layout_regions),
        "recognized_regions": len(recognized),
        "skipped_regions": len(skipped),
        "layout_label_counts": dict(sorted(Counter(region.label for region in layout_regions).items())),
        "stop_reason_counts": dict(sorted(Counter(region.stop_reason for region in recognized).items())),
        "generated_tokens_including_eos": output_tokens,
        "decode_tokens_after_prefill_including_eos": decode_tokens,
        "sum_page_wall_s": float(page_wall),
        "sum_compiled_decode_wall_s": float(decode_wall),
        "rates": {
            "pages_per_s": per_second(len(pages), page_wall),
            "regions_per_s": per_second(len(recognized), page_wall),
            "decode_effective_tok_per_s": per_second(decode_tokens, decode_wall),
            "e2e_output_tok_per_s": per_second(output_tokens, page_wall),
        },
    }
