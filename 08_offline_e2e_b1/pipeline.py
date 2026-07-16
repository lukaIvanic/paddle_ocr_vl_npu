"""Lazy multi-page orchestration with run-scoped continuous decode."""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from PIL import Image, ImageDraw

from engine import ContinuousRecognizer, RecognitionInput
from layout import PPDocLayoutV3Runtime
from schema import (
    ContinuousDecodeResult,
    PageResult,
    RecognitionResult,
    SkippedRegion,
    per_second,
)


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


@dataclass
class _PageWork:
    page_index: int
    page_id: str
    image_path: Path
    image_size: tuple[int, int]
    image: Image.Image | None
    layout_regions: list[Any]
    skipped_regions: list[SkippedRegion]
    requests: list[RecognitionInput]
    request_count: int
    page_started: float
    recognition_started: float
    run_started: float
    image_load_s: float
    layout_timing_s: dict[str, float]
    crop_extraction_s: float
    recognized_regions: list[RecognitionResult] = field(default_factory=list)
    result: PageResult | None = None


@dataclass(frozen=True)
class PipelineRunResult:
    pages: list[PageResult]
    decode_schedule: ContinuousDecodeResult
    run_wall_s: float
    completion_order: list[str]


class OfflinePagePipeline:
    """Feed pages lazily into one run-scoped recognition scheduler."""

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

    def _prepare_page(
        self,
        image_path: Path,
        page_index: int,
        *,
        run_started: float,
    ) -> _PageWork:
        page_started = time.perf_counter()
        image_path = image_path.expanduser().resolve()
        started = time.perf_counter()
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        image_load_s = time.perf_counter() - started

        page_id = f"page_{page_index:04d}_{image_path.stem}"
        layout_regions, layout_timing = self.layout.predict(image)
        recognition_started = time.perf_counter()
        skipped: list[SkippedRegion] = []
        requests: list[RecognitionInput] = []
        crop_total_s = 0.0
        recognized_count = 0

        crop_dir = None
        if self.artifact_dir is not None and self.save_crops:
            crop_dir = self.artifact_dir / "crops" / page_id
            crop_dir.mkdir(parents=True, exist_ok=True)

        for region in layout_regions:
            skip_reason = self._skip_reason(region.label)
            if (
                skip_reason is None
                and self.max_regions is not None
                and recognized_count >= self.max_regions
            ):
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

        return _PageWork(
            page_index=page_index,
            page_id=page_id,
            image_path=image_path,
            image_size=image.size,
            image=image,
            layout_regions=layout_regions,
            skipped_regions=skipped,
            requests=requests,
            request_count=len(requests),
            page_started=page_started,
            recognition_started=recognition_started,
            run_started=run_started,
            image_load_s=float(image_load_s),
            layout_timing_s=layout_timing,
            crop_extraction_s=float(crop_total_s),
        )

    def _finalize_page(
        self,
        work: _PageWork,
        *,
        schedule_id: str,
    ) -> PageResult:
        if work.result is not None:
            raise RuntimeError(f"page {work.page_id} was finalized twice")
        if len(work.recognized_regions) != work.request_count:
            raise RuntimeError(
                f"page {work.page_id} completed {len(work.recognized_regions)} "
                f"of {work.request_count} recognition requests"
            )

        recognized = sorted(
            work.recognized_regions,
            key=lambda result: result.layout_order,
        )
        recognition_finished = time.perf_counter()
        started = time.perf_counter()
        reading_order_text = "\n\n".join(
            result.text.strip()
            for result in recognized
            if result.label not in MARKDOWN_IGNORE_LABELS and result.text.strip()
        )
        postprocess_s = time.perf_counter() - started
        page_pipeline_s = time.perf_counter() - work.page_started
        submission_to_completion_s = time.perf_counter() - work.run_started

        artifact_started = time.perf_counter()
        image = work.image
        if self.artifact_dir is not None:
            page_dir = self.artifact_dir / "pages"
            page_dir.mkdir(parents=True, exist_ok=True)
            (page_dir / f"{work.page_id}.txt").write_text(
                reading_order_text + "\n",
                encoding="utf-8",
            )
            if self.save_annotated:
                if image is None:
                    raise RuntimeError(f"page {work.page_id} released its image too early")
                annotated = image.copy()
                draw = ImageDraw.Draw(annotated)
                for region in work.layout_regions:
                    draw.rectangle(region.box.as_list(), outline="red", width=3)
                    draw.text(
                        (region.box.x0 + 3, region.box.y0 + 3),
                        f"{region.order} {region.label} {region.score:.2f}",
                        fill="red",
                        stroke_width=2,
                        stroke_fill="white",
                    )
                annotated.save(page_dir / f"{work.page_id}_layout.png")
        artifact_write_s = time.perf_counter() - artifact_started

        generated_tokens = sum(
            item.generated_tokens_including_eos for item in recognized
        )
        prefill_wall = sum(
            item.timing_s["prefill_request_total"] for item in recognized
        )
        timing = {
            "image_load": work.image_load_s,
            **work.layout_timing_s,
            "crop_extraction": work.crop_extraction_s,
            "recognition_wall": float(
                recognition_finished - work.recognition_started
            ),
            "sequential_prefill_wall_sum": float(prefill_wall),
            "reading_order_text_postprocess": float(postprocess_s),
            "page_total": float(page_pipeline_s),
            "run_submission_to_page_completion": float(
                submission_to_completion_s
            ),
            "artifact_write": float(artifact_write_s),
            "page_total_including_artifacts": float(
                time.perf_counter() - work.page_started
            ),
        }
        partial = any(
            item.reason == "max_regions_debug_limit"
            for item in work.skipped_regions
        )
        result = PageResult(
            page_id=work.page_id,
            image_path=work.image_path,
            image_size=work.image_size,
            layout_regions=work.layout_regions,
            recognized_regions=recognized,
            decode_schedule_id=schedule_id,
            skipped_regions=work.skipped_regions,
            reading_order_text=reading_order_text,
            timing_s=timing,
            rates={
                "layout_regions_per_s": per_second(
                    len(work.layout_regions),
                    work.layout_timing_s["layout_inference"],
                ),
                "recognition_regions_per_s": per_second(
                    len(recognized),
                    timing["recognition_wall"],
                ),
                "page_output_tok_per_s": per_second(
                    generated_tokens,
                    page_pipeline_s,
                ),
            },
            partial=partial,
        )
        work.result = result
        work.image = None
        work.requests.clear()
        return result

    def run_pages(
        self,
        image_paths: Iterable[Path],
        *,
        on_page_completed: Callable[[PageResult], None] | None = None,
        schedule_id: str = "run_cross_page_continuous_decode",
    ) -> PipelineRunResult:
        paths = list(image_paths)
        if not paths:
            raise ValueError("run_pages requires at least one image")

        run_started = time.perf_counter()
        works: dict[int, _PageWork] = {}
        request_to_page: dict[str, _PageWork] = {}
        completion_order: list[str] = []

        def emit_completed_page(work: _PageWork) -> None:
            result = self._finalize_page(work, schedule_id=schedule_id)
            completion_order.append(result.page_id)
            if on_page_completed is not None:
                on_page_completed(result)

        def request_source() -> Iterable[RecognitionInput]:
            for page_index, path in enumerate(paths):
                work = self._prepare_page(
                    path,
                    page_index,
                    run_started=run_started,
                )
                works[page_index] = work
                if work.request_count == 0:
                    emit_completed_page(work)
                    continue
                for request in work.requests:
                    request_to_page[request.request_id] = work
                    yield request
                work.requests.clear()

        def accept_result(result: RecognitionResult) -> None:
            try:
                work = request_to_page.pop(result.request_id)
            except KeyError as exc:
                raise RuntimeError(
                    f"recognizer completed unknown request {result.request_id}"
                ) from exc
            work.recognized_regions.append(result)
            if len(work.recognized_regions) == work.request_count:
                emit_completed_page(work)

        _results, decode_schedule = self.recognizer.recognize_stream(
            request_source(),
            schedule_id=schedule_id,
            on_result=accept_result,
        )
        run_wall_s = time.perf_counter() - run_started

        if request_to_page:
            raise AssertionError(
                f"{len(request_to_page)} recognition requests never completed"
            )
        unfinished = [
            work.page_id for work in works.values() if work.result is None
        ]
        if unfinished:
            raise AssertionError(f"pages never emitted: {unfinished}")
        pages = [works[index].result for index in range(len(paths))]
        if any(page is None for page in pages):
            raise AssertionError("page result ordering contains an empty entry")
        return PipelineRunResult(
            pages=[page for page in pages if page is not None],
            decode_schedule=decode_schedule,
            run_wall_s=float(run_wall_s),
            completion_order=completion_order,
        )

    def run_page(self, image_path: Path, page_index: int = 0) -> PageResult:
        """Compatibility wrapper; scheduling remains run-scoped by default."""

        if page_index != 0:
            raise ValueError("run_page only supports page_index=0; use run_pages")
        return self.run_pages([image_path]).pages[0]


def aggregate_pages(
    pages: list[PageResult],
    decode_schedule: ContinuousDecodeResult,
    *,
    run_wall_s: float,
) -> dict[str, Any]:
    recognized = [region for page in pages for region in page.recognized_regions]
    layout_regions = [region for page in pages for region in page.layout_regions]
    skipped = [region for page in pages for region in page.skipped_regions]
    output_tokens = sum(
        region.generated_tokens_including_eos for region in recognized
    )
    real_vision_tokens = sum(
        int(region.vision.get("real_vision_tokens", 0)) for region in recognized
    )
    physical_vision_tokens = sum(
        int(region.vision.get("physical_vision_tokens", 0)) for region in recognized
    )
    page_latencies = [page.timing_s["page_total"] for page in pages]
    return {
        "pages": len(pages),
        "partial_pages": sum(1 for page in pages if page.partial),
        "layout_regions": len(layout_regions),
        "recognized_regions": len(recognized),
        "skipped_regions": len(skipped),
        "decode_schedules": 1,
        "decode_graph_calls": decode_schedule.graph_calls,
        "initial_decode_admissions": decode_schedule.initial_admissions,
        "hot_swap_decode_admissions": decode_schedule.hot_swap_admissions,
        "ready_buffer_capacity": decode_schedule.ready_buffer_capacity,
        "ready_buffer_low_watermark": (
            decode_schedule.ready_buffer_low_watermark
        ),
        "max_ready_queue_depth": decode_schedule.max_ready_queue_depth,
        "ready_source_refill_count": decode_schedule.ready_source_refill_count,
        "layout_label_counts": dict(
            sorted(Counter(region.label for region in layout_regions).items())
        ),
        "stop_reason_counts": dict(
            sorted(Counter(region.stop_reason for region in recognized).items())
        ),
        "generated_tokens_including_eos": output_tokens,
        "vision_execution_counts": dict(
            sorted(Counter(str(region.vision.get("execution", "unknown")) for region in recognized).items())
        ),
        "vision_bucket_counts": dict(
            sorted(
                Counter(
                    str(region.vision["bucket"])
                    for region in recognized
                    if region.vision.get("bucket") is not None
                ).items(),
                key=lambda item: int(item[0]),
            )
        ),
        "real_vision_tokens": int(real_vision_tokens),
        "physical_vision_tokens": int(physical_vision_tokens),
        "vision_useful_token_fraction": (
            float(real_vision_tokens) / float(physical_vision_tokens)
            if physical_vision_tokens > 0
            else None
        ),
        "raw_decode_token_slots": decode_schedule.raw_decode_token_slots,
        "active_decode_token_slots": decode_schedule.active_decode_token_slots,
        "effective_decode_tokens": decode_schedule.effective_decode_tokens,
        "idle_decode_token_slots": decode_schedule.idle_decode_token_slots,
        "lookahead_decode_token_slots": decode_schedule.lookahead_decode_token_slots,
        "kv_prefix_bytes_copied": decode_schedule.kv_prefix_bytes_copied,
        "run_wall_s": float(run_wall_s),
        "sum_page_latency_s": float(sum(page_latencies)),
        "mean_page_latency_s": (
            float(sum(page_latencies)) / len(page_latencies)
            if page_latencies
            else None
        ),
        "continuous_decode_wall_s": float(
            decode_schedule.timing_s["continuous_decode_wall"]
        ),
        "run_scoped_scheduler_wall_s": float(
            decode_schedule.timing_s["run_scoped_scheduler_wall"]
        ),
        "rates": {
            "pages_per_s": per_second(len(pages), run_wall_s),
            "regions_per_s": per_second(len(recognized), run_wall_s),
            "raw_decode_tok_per_s": decode_schedule.rates[
                "raw_decode_tok_per_s"
            ],
            "effective_decode_tok_per_s": decode_schedule.rates[
                "effective_decode_tok_per_s"
            ],
            "effective_fraction": decode_schedule.rates["effective_fraction"],
            "active_slot_fraction": decode_schedule.rates[
                "active_slot_fraction"
            ],
            "e2e_output_tok_per_s": per_second(output_tokens, run_wall_s),
        },
    }
