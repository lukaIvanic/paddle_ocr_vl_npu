"""Full-page orchestration with sequential prefill and continuous decode."""

from __future__ import annotations

import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from engine import ContinuousRecognizer, RecognitionInput
from layout import PPDocLayoutV3Runtime
from schema import ContinuousDecodeResult, PageResult, SkippedRegion, per_second


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
        recognizer: ContinuousRecognizer,
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
        requests: list[RecognitionInput] = []
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

            requests.append(
                RecognitionInput(
                    request_id=f"{page_id}_region_{region.order:03d}",
                    layout_order=region.order,
                    label=region.label,
                    prompt=recognition_prompt(region.label),
                    box=region.box,
                    crop=crop,
                )
            )
            recognized_count += 1

        if requests:
            recognized, decode_schedule = self.recognizer.recognize_many(
                requests,
                schedule_id=f"{page_id}_continuous_decode",
            )
        else:
            decode_schedule = ContinuousDecodeResult(
                schedule_id=f"{page_id}_continuous_decode",
                batch_size=self.recognizer.batch_size,
                requests=0,
                graph_calls=0,
                initial_admissions=0,
                hot_swap_admissions=0,
                prefill_only_completions=0,
                raw_decode_token_slots=0,
                active_decode_token_slots=0,
                effective_decode_tokens=0,
                idle_decode_token_slots=0,
                lookahead_decode_token_slots=0,
                kv_prefix_bytes_copied=0,
                initial_kv_prefix_bytes_copied=0,
                hot_swap_kv_prefix_bytes_copied=0,
                timing_s={
                    "continuous_decode_wall": 0.0,
                    "decode_model_and_argmax_device": 0.0,
                    "slot_admission_device": 0.0,
                    "slot_admission_enqueue_wall": 0.0,
                    "d2h_wait_wall": 0.0,
                    "retire_and_refill_host_wall": 0.0,
                },
                rates={
                    "raw_decode_tok_per_s": None,
                    "effective_decode_tok_per_s": None,
                    "effective_fraction": None,
                    "active_slot_fraction": None,
                    "effective_device_tok_per_s": None,
                },
            )

        recognition_wall_s = time.perf_counter() - recognition_started
        started = time.perf_counter()
        reading_order_text = "\n\n".join(
            result.text.strip()
            for result in recognized
            if result.label not in MARKDOWN_IGNORE_LABELS and result.text.strip()
        )
        postprocess_s = time.perf_counter() - started
        page_pipeline_s = time.perf_counter() - page_started

        artifact_started = time.perf_counter()
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
        artifact_write_s = time.perf_counter() - artifact_started

        generated_tokens = sum(item.generated_tokens_including_eos for item in recognized)
        raw_decode_slots = decode_schedule.raw_decode_token_slots
        effective_decode_tokens = decode_schedule.effective_decode_tokens
        decode_wall = decode_schedule.timing_s["continuous_decode_wall"]
        prefill_wall = sum(item.timing_s["prefill_request_total"] for item in recognized)
        page_total_including_artifacts_s = time.perf_counter() - page_started
        timing = {
            "image_load": float(image_load_s),
            **layout_timing,
            "crop_extraction": float(crop_total_s),
            "recognition_wall": float(recognition_wall_s),
            "sequential_prefill_wall_sum": float(prefill_wall),
            "continuous_decode_wall": float(decode_wall),
            "reading_order_text_postprocess": float(postprocess_s),
            "page_total": float(page_pipeline_s),
            "artifact_write": float(artifact_write_s),
            "page_total_including_artifacts": float(page_total_including_artifacts_s),
        }
        partial = any(item.reason == "max_regions_debug_limit" for item in skipped)
        return PageResult(
            page_id=page_id,
            image_path=image_path,
            image_size=image.size,
            layout_regions=layout_regions,
            recognized_regions=recognized,
            decode_schedule=decode_schedule,
            skipped_regions=skipped,
            reading_order_text=reading_order_text,
            timing_s=timing,
            rates={
                "layout_regions_per_s": per_second(len(layout_regions), layout_timing["layout_inference"]),
                "recognition_regions_per_s": per_second(len(recognized), recognition_wall_s),
                "raw_decode_tok_per_s": per_second(raw_decode_slots, decode_wall),
                "effective_decode_tok_per_s": per_second(effective_decode_tokens, decode_wall),
                "page_output_tok_per_s": per_second(generated_tokens, page_pipeline_s),
            },
            partial=partial,
        )


def aggregate_pages(pages: list[PageResult]) -> dict[str, Any]:
    recognized = [region for page in pages for region in page.recognized_regions]
    layout_regions = [region for page in pages for region in page.layout_regions]
    skipped = [region for page in pages for region in page.skipped_regions]
    decode_schedules = [page.decode_schedule for page in pages]
    page_wall = sum(page.timing_s["page_total"] for page in pages)
    decode_wall = sum(
        schedule.timing_s["continuous_decode_wall"]
        for schedule in decode_schedules
    )
    raw_decode_slots = sum(schedule.raw_decode_token_slots for schedule in decode_schedules)
    active_decode_slots = sum(schedule.active_decode_token_slots for schedule in decode_schedules)
    effective_decode_tokens = sum(schedule.effective_decode_tokens for schedule in decode_schedules)
    idle_decode_slots = sum(schedule.idle_decode_token_slots for schedule in decode_schedules)
    lookahead_decode_slots = sum(
        schedule.lookahead_decode_token_slots for schedule in decode_schedules
    )
    initial_admissions = sum(schedule.initial_admissions for schedule in decode_schedules)
    hot_swap_admissions = sum(schedule.hot_swap_admissions for schedule in decode_schedules)
    kv_prefix_bytes = sum(schedule.kv_prefix_bytes_copied for schedule in decode_schedules)
    output_tokens = sum(region.generated_tokens_including_eos for region in recognized)
    return {
        "pages": len(pages),
        "partial_pages": sum(1 for page in pages if page.partial),
        "layout_regions": len(layout_regions),
        "recognized_regions": len(recognized),
        "skipped_regions": len(skipped),
        "decode_schedules": len(decode_schedules),
        "decode_graph_calls": sum(schedule.graph_calls for schedule in decode_schedules),
        "initial_decode_admissions": initial_admissions,
        "hot_swap_decode_admissions": hot_swap_admissions,
        "layout_label_counts": dict(sorted(Counter(region.label for region in layout_regions).items())),
        "stop_reason_counts": dict(sorted(Counter(region.stop_reason for region in recognized).items())),
        "generated_tokens_including_eos": output_tokens,
        "raw_decode_token_slots": raw_decode_slots,
        "active_decode_token_slots": active_decode_slots,
        "effective_decode_tokens": effective_decode_tokens,
        "idle_decode_token_slots": idle_decode_slots,
        "lookahead_decode_token_slots": lookahead_decode_slots,
        "kv_prefix_bytes_copied": kv_prefix_bytes,
        "sum_page_wall_s": float(page_wall),
        "sum_continuous_decode_wall_s": float(decode_wall),
        "rates": {
            "pages_per_s": per_second(len(pages), page_wall),
            "regions_per_s": per_second(len(recognized), page_wall),
            "raw_decode_tok_per_s": per_second(raw_decode_slots, decode_wall),
            "effective_decode_tok_per_s": per_second(effective_decode_tokens, decode_wall),
            "effective_fraction": (
                float(effective_decode_tokens) / float(raw_decode_slots)
                if raw_decode_slots > 0
                else None
            ),
            "active_slot_fraction": (
                float(active_decode_slots) / float(raw_decode_slots)
                if raw_decode_slots > 0
                else None
            ),
            "e2e_output_tok_per_s": per_second(output_tokens, page_wall),
        },
    }
