"""Feed PaddleX-prepared page crops into one continuous recognition run."""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from PIL import Image

from paddleocr_vl.serving.engine import ContinuousRecognizer
from paddleocr_vl.serving.types import (
    ContinuousDecodeResult,
    RecognitionRequest,
    RecognitionResult,
)


PROMPT_LABELS = {
    "OCR:": "text",
    "Table Recognition:": "table",
    "Formula Recognition:": "formula",
    "Chart Recognition:": "chart",
    "Spotting:": "spotting",
    "Seal Recognition:": "seal",
}


@dataclass
class _PageState:
    ordinal: int
    blocks: list[Any]
    batches: dict[tuple[int, int], dict[str, Any]]
    block_profiles: dict[tuple[int, int], tuple[int, int]]
    drop_figures: set[Any]
    visible_image_labels: list[str]
    skip_special_tokens: bool
    remaining: int
    result: Any | None = None


@dataclass(frozen=True)
class _RequestOwner:
    page: _PageState
    pixel_profile: tuple[int, int]
    profile_index: int
    request_index: int
    block_index: int


@dataclass(frozen=True)
class PaddleXPageRunResult:
    schedule: ContinuousDecodeResult
    summary: dict[str, Any]
    completion_order: list[str]


class PaddleXPageBridge:
    """Reuse PaddleX page logic without its synchronous VLM batch drain."""

    def __init__(
        self,
        pipeline: Any,
        recognizer: ContinuousRecognizer,
        *,
        trace_path: Path,
        min_pixels: int | None,
    ) -> None:
        self.pipeline = pipeline
        self.recognizer = recognizer
        self.trace_path = trace_path.expanduser().resolve()
        self.min_pixels = min_pixels

        from paddlex.inference.pipelines.paddleocr_vl.pipeline import IMAGE_LABELS

        self.image_labels = list(IMAGE_LABELS)
        required = (
            "_paddleocr_vl_prepare_page_core",
            "_paddleocr_vl_prepare_page_serial_benchmarked",
            "_paddleocr_vl_layout_prep_parallel_pages",
            "_paddleocr_vl_aggregate_vlm_batches",
            "_paddleocr_vl_assemble_parsing_results",
        )
        missing = [name for name in required if not hasattr(pipeline, name)]
        if missing:
            raise RuntimeError(f"installed PaddleX lacks required helpers: {missing}")

    def _capture_pages(
        self,
        images: list[Any],
        layout_results: list[Any],
        imgs_in_doc: list[Any],
        *,
        first_ordinal: int,
        use_chart_recognition: bool,
        use_seal_recognition: bool,
        use_ocr_for_image_block: bool,
        vlm_kwargs: dict[str, Any] | None,
        merge_layout_blocks: bool,
        layout_shape_mode: str,
    ) -> list[_PageState]:
        kwargs = dict(vlm_kwargs or {})
        default_min = int(kwargs.pop("min_pixels", None) or 112896)
        default_max = int(kwargs.pop("max_pixels", None) or 1003520)
        visible_image_labels = self.image_labels + ["seal"]
        image_labels = [] if use_ocr_for_image_block else self.image_labels.copy()
        if not use_chart_recognition:
            image_labels.append("chart")
            visible_image_labels.append("chart")
        if not use_seal_recognition:
            image_labels.append("seal")

        config = {
            "layout_shape_mode": layout_shape_mode,
            "merge_layout_blocks": merge_layout_blocks,
            "image_labels": image_labels,
            "use_chart_recognition": use_chart_recognition,
            "use_seal_recognition": use_seal_recognition,
        }
        for label in ("ocr", "table", "chart", "formula", "seal"):
            config[f"{label}_min_pixels"] = int(
                kwargs.pop(f"{label}_min_pixels", default_min)
            )
            config[f"{label}_max_pixels"] = int(
                kwargs.pop(f"{label}_max_pixels", default_max)
            )

        payloads = [
            (0, image, layout_result, page_imgs, config)
            for image, layout_result, page_imgs in zip(
                images, layout_results, imgs_in_doc
            )
        ]
        workers = min(int(self.pipeline.layout_prep_cpu_workers), len(payloads))
        if len(payloads) > 1 and workers > 1:
            prepared = self.pipeline._paddleocr_vl_layout_prep_parallel_pages(
                payloads, workers
            )
        else:
            prepared = [
                self.pipeline._paddleocr_vl_prepare_page_serial_benchmarked(payload)
                for payload in payloads
            ]

        pages: list[_PageState] = []
        group_has_spotting = any(item[3] for item in prepared)
        for offset, page in enumerate(prepared):
            blocks, _has_spotting, drops, batches, profiles = (
                self.pipeline._paddleocr_vl_aggregate_vlm_batches([page])
            )
            remaining = 0
            for batch in batches.values():
                count = len(batch["images"])
                batch["vlm_results"] = [None] * count
                remaining += count
            pages.append(
                _PageState(
                    ordinal=first_ordinal + offset,
                    blocks=blocks,
                    batches=batches,
                    block_profiles=profiles,
                    drop_figures=drops,
                    visible_image_labels=visible_image_labels,
                    skip_special_tokens=not group_has_spotting,
                    remaining=remaining,
                )
            )
        return pages

    def _finish_page(self, page: _PageState) -> Any:
        if page.result is None:
            raise RuntimeError(f"page {page.ordinal} has no PaddleX result template")
        parsing, tables, spotting = (
            self.pipeline._paddleocr_vl_assemble_parsing_results(
                page.blocks,
                page.batches,
                page.block_profiles,
                page.drop_figures,
                page.visible_image_labels,
            )
        )
        page.result["parsing_res_list"] = parsing[0]
        page.result["table_res_list"] = tables[0]
        page.result["spotting_res"] = spotting[0]
        return page.result

    @staticmethod
    def _trace(result: RecognitionResult, owner: _RequestOwner) -> dict[str, Any]:
        return {
            "global_request_index": owner.request_index,
            "page_input_index": owner.page.ordinal,
            "block_index": owner.block_index,
            "request_id": result.request_id,
            "label": PROMPT_LABELS.get(result.prompt, "unknown"),
            "prompt": result.prompt,
            "crop_size": list(result.crop_size),
            "min_pixels": owner.pixel_profile[0],
            "max_pixels": owner.pixel_profile[1],
            "input_tokens": result.input_tokens,
            "projected_image_tokens": result.projected_image_tokens,
            "token_ids": result.token_ids,
            "text": result.text,
            "stop_reason": result.stop_reason,
            "generated_tokens_including_eos": result.generated_tokens_including_eos,
            "decode_tokens_after_prefill_including_eos": (
                result.decode_tokens_after_prefill_including_eos
            ),
            "vision": result.vision,
            "text_prefill": result.text_prefill,
            "timing_s": result.timing_s,
            "device_stage_s": result.device_stage_s,
        }

    def _summarize(
        self,
        results: list[RecognitionResult],
        schedule: ContinuousDecodeResult,
        wall_s: float,
        profiles: Counter[str],
    ) -> dict[str, Any]:
        stage_s: defaultdict[str, float] = defaultdict(float)
        for result in results:
            for name, seconds in result.device_stage_s.items():
                stage_s[name] += float(seconds)
        total = lambda field: sum(int(getattr(result, field)) for result in results)
        nested = lambda section, field: sum(
            int(getattr(result, section).get(field, 0)) for result in results
        )
        generated = total("generated_tokens_including_eos")
        real_vision = nested("vision", "real_vision_tokens")
        physical_vision = nested("vision", "physical_vision_tokens")
        real_text = nested("text_prefill", "real_text_tokens")
        physical_text = nested("text_prefill", "physical_text_tokens")
        raw_slots = schedule.raw_decode_token_slots
        return {
            "schedules": 1,
            "requests": len(results),
            "wall_s": wall_s,
            "generated_tokens_including_eos": generated,
            "decode_tokens_after_prefill_including_eos": total(
                "decode_tokens_after_prefill_including_eos"
            ),
            "input_tokens": total("input_tokens"),
            "projected_image_tokens": total("projected_image_tokens"),
            "real_vision_tokens": real_vision,
            "physical_vision_tokens": physical_vision,
            "real_text_tokens": real_text,
            "physical_text_tokens": physical_text,
            "decode_graph_calls": schedule.graph_calls,
            "raw_decode_token_slots": raw_slots,
            "active_decode_token_slots": schedule.active_decode_token_slots,
            "effective_decode_tokens": schedule.effective_decode_tokens,
            "idle_decode_token_slots": schedule.idle_decode_token_slots,
            "lookahead_decode_token_slots": schedule.lookahead_decode_token_slots,
            "decode_wall_s": schedule.timing_s["continuous_decode_wall"],
            "run_scoped_scheduler_wall_s": schedule.timing_s[
                "run_scoped_scheduler_wall"
            ],
            "kv_prefix_bytes_copied": schedule.kv_prefix_bytes_copied,
            "stop_reason_counts": dict(
                sorted(Counter(result.stop_reason for result in results).items())
            ),
            "pixel_profile_request_counts": dict(sorted(profiles.items())),
            "device_stage_s": dict(sorted(stage_s.items())),
            "run_output_tok_per_s": generated / wall_s if wall_s else None,
            "decode_useful_token_fraction": (
                schedule.effective_decode_tokens / raw_slots if raw_slots else None
            ),
            "vision_useful_token_fraction": (
                real_vision / physical_vision if physical_vision else None
            ),
            "text_useful_token_fraction": (
                real_text / physical_text if physical_text else None
            ),
            "trace_path": str(self.trace_path),
        }

    def run(
        self,
        image_paths: Iterable[str],
        *,
        emit_page: Callable[[Any], None],
        schedule_id: str = "paddlex_cross_page",
    ) -> PaddleXPageRunResult:
        paths = list(image_paths)
        if not paths:
            raise ValueError("PaddleXPageBridge.run requires at least one page")

        pending: deque[_PageState] = deque()
        owners: dict[str, _RequestOwner] = {}
        results: list[RecognitionResult] = []
        completion_order: list[str] = []
        profiles: Counter[str] = Counter()
        completed_pages = 0
        next_page = 0
        next_request = 0
        original_get_results = self.pipeline.get_layout_parsing_results

        def capture_layout_results(
            images: list[Any],
            layout_det_results: list[Any],
            imgs_in_doc: list[Any],
            use_chart_recognition: bool = False,
            use_seal_recognition: bool = False,
            use_ocr_for_image_block: bool = False,
            vlm_kwargs: dict[str, Any] | None = None,
            merge_layout_blocks: bool = True,
            layout_shape_mode: str = "auto",
        ) -> tuple[list[list], list[list], list[dict], list[Any]]:
            nonlocal next_page
            captured = self._capture_pages(
                images,
                layout_det_results,
                imgs_in_doc,
                first_ordinal=next_page,
                use_chart_recognition=use_chart_recognition,
                use_seal_recognition=use_seal_recognition,
                use_ocr_for_image_block=use_ocr_for_image_block,
                vlm_kwargs=vlm_kwargs,
                merge_layout_blocks=merge_layout_blocks,
                layout_shape_mode=layout_shape_mode,
            )
            pending.extend(captured)
            next_page += len(captured)
            count = len(captured)
            return (
                [[] for _ in range(count)],
                [[] for _ in range(count)],
                [{} for _ in range(count)],
                imgs_in_doc,
            )

        def complete_page(page: _PageState) -> None:
            nonlocal completed_pages
            result = self._finish_page(page)
            emit_page(result)
            completion_order.append(Path(str(result["input_path"])).name)
            completed_pages += 1

        def request_source() -> Iterable[RecognitionRequest]:
            nonlocal next_request
            for template in self.pipeline.predict(
                paths,
                use_queues=False,
                min_pixels=self.min_pixels,
                max_new_tokens=self.recognizer.max_new_tokens,
            ):
                if not pending:
                    raise RuntimeError("PaddleX emitted a page without captured state")
                page = pending.popleft()
                page.result = template
                if page.remaining == 0:
                    complete_page(page)
                    continue
                for profile, batch in page.batches.items():
                    for profile_index, (image, prompt, block_id) in enumerate(
                        zip(batch["images"], batch["queries"], batch["vlm_block_ids"])
                    ):
                        if prompt not in PROMPT_LABELS:
                            raise ValueError(f"unsupported PaddleX prompt: {prompt!r}")
                        request_id = (
                            f"page_{page.ordinal:06d}_block_{int(block_id[1]):06d}"
                        )
                        owners[request_id] = _RequestOwner(
                            page,
                            (int(profile[0]), int(profile[1])),
                            profile_index,
                            next_request,
                            int(block_id[1]),
                        )
                        next_request += 1
                        profiles[f"{profile[0]}:{profile[1]}"] += 1
                        yield RecognitionRequest(
                            request_id=request_id,
                            crop=Image.fromarray(
                                np.ascontiguousarray(image[:, :, ::-1])
                            ),
                            prompt=prompt,
                            skip_special_tokens=page.skip_special_tokens,
                            min_pixels=int(profile[0]),
                            max_pixels=int(profile[1]),
                        )
            if pending:
                raise RuntimeError(f"{len(pending)} captured PaddleX pages were not emitted")

        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        self.pipeline.get_layout_parsing_results = capture_layout_results
        try:
            with self.trace_path.open("w", encoding="utf-8") as trace:
                def accept_result(result: RecognitionResult) -> None:
                    owner = owners.pop(result.request_id)
                    batch = owner.page.batches[owner.pixel_profile]
                    if batch["vlm_results"][owner.profile_index] is not None:
                        raise RuntimeError(f"duplicate result {result.request_id}")
                    batch["vlm_results"][owner.profile_index] = {"result": result.text}
                    owner.page.remaining -= 1
                    results.append(result)
                    trace.write(
                        json.dumps(
                            self._trace(result, owner),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    trace.flush()
                    if owner.page.remaining == 0:
                        complete_page(owner.page)

                schedule = self.recognizer.run(
                    request_source(),
                    schedule_id=schedule_id,
                    emit_result=accept_result,
                )
        finally:
            self.pipeline.get_layout_parsing_results = original_get_results

        wall_s = time.perf_counter() - started
        if owners or completed_pages != len(paths) or len(results) != schedule.requests:
            raise AssertionError(
                "cross-page accounting mismatch: "
                f"owners={len(owners)} pages={completed_pages}/{len(paths)} "
                f"results={len(results)}/{schedule.requests}"
            )
        return PaddleXPageRunResult(
            schedule=schedule,
            summary=self._summarize(results, schedule, wall_s, profiles),
            completion_order=completion_order,
        )
