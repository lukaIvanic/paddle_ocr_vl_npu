"""PaddleX VL-recognition adapter backed by Experiment 08.

The official PaddleX v1.6 pipeline remains responsible for layout filtering,
crop preparation, block merging, table/formula handling, parsing-result
assembly, and Markdown conversion.  This adapter replaces only
``pipeline.vl_rec_model`` so the exact same prepared crops and prompts flow
through the local optimized recognizer.
"""

from __future__ import annotations

import json
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
from PIL import Image

from engine import ContinuousRecognizer, RecognitionInput
from schema import Box


PROMPT_LABELS = {
    "OCR:": "text",
    "Table Recognition:": "table",
    "Formula Recognition:": "formula",
    "Chart Recognition:": "chart",
    "Spotting:": "spotting",
    "Seal Recognition:": "seal",
}


class PaddleXContinuousRecognizerAdapter:
    """Expose PaddleX's model ``predict`` contract over ContinuousRecognizer."""

    def __init__(
        self,
        recognizer: ContinuousRecognizer,
        *,
        batch_size: int,
        trace_path: Path,
    ):
        self.recognizer = recognizer
        self.batch_sampler = SimpleNamespace(batch_size=int(batch_size))
        self.trace_path = trace_path.expanduser().resolve()
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self._trace = self.trace_path.open("w", encoding="utf-8")
        self._lock = threading.Lock()
        self._closed = False
        self._batch_index = 0
        self._request_index = 0
        self._summary: dict[str, Any] = {
            "batches": 0,
            "requests": 0,
            "wall_s": 0.0,
            "generated_tokens_including_eos": 0,
            "decode_tokens_after_prefill_including_eos": 0,
            "input_tokens": 0,
            "projected_image_tokens": 0,
            "real_vision_tokens": 0,
            "physical_vision_tokens": 0,
            "real_text_tokens": 0,
            "physical_text_tokens": 0,
            "decode_graph_calls": 0,
            "decode_wall_s": 0.0,
            "run_scoped_scheduler_wall_s": 0.0,
            "kv_prefix_bytes_copied": 0,
        }
        self._stop_reasons: Counter[str] = Counter()
        self._vision_routes: Counter[str] = Counter()
        self._text_routes: Counter[str] = Counter()
        self._device_stage_s: defaultdict[str, float] = defaultdict(float)
        self._pixel_profiles: Counter[str] = Counter()

    def predict(
        self,
        inputs: Iterable[dict[str, Any]],
        *,
        skip_special_tokens: bool = True,
        **kwargs: Any,
    ) -> Iterable[dict[str, str]]:
        items = list(inputs)
        if not items:
            return iter(())
        max_new_tokens = kwargs.get("max_new_tokens")
        if max_new_tokens is None:
            max_new_tokens = self.recognizer.max_new_tokens
        if int(max_new_tokens) != self.recognizer.max_new_tokens:
            raise ValueError(
                "PaddleX max_new_tokens must match the recognizer runtime: "
                f"request={int(max_new_tokens)} runtime={self.recognizer.max_new_tokens}"
            )
        min_pixels = int(
            kwargs.get("min_pixels")
            if kwargs.get("min_pixels") is not None
            else self.recognizer.model_preprocessor_min_pixels
        )
        max_pixels = int(
            kwargs.get("max_pixels")
            if kwargs.get("max_pixels") is not None
            else self.recognizer.preprocessor_config["max_pixels"]
        )
        if min_pixels <= 0 or max_pixels < min_pixels:
            raise ValueError(
                f"invalid PaddleX pixel profile min={min_pixels} max={max_pixels}"
            )

        with self._lock:
            if self._closed:
                raise RuntimeError("PaddleX recognizer adapter is closed")
            batch_index = self._batch_index
            self._batch_index += 1
            first_request_index = self._request_index
            self._request_index += len(items)
            preprocessor_config = dict(self.recognizer.preprocessor_config)
            preprocessor_config["min_pixels"] = min_pixels
            preprocessor_config["max_pixels"] = max_pixels
            self.recognizer.preprocessor_config = preprocessor_config

            requests = []
            for item_index, item in enumerate(items):
                image = item.get("image")
                query = str(item.get("query", ""))
                if not isinstance(image, np.ndarray) or image.ndim != 3:
                    raise TypeError(
                        "PaddleX VL inputs must contain a BGR numpy image, got "
                        f"{type(image).__name__}"
                    )
                if query not in PROMPT_LABELS:
                    raise ValueError(f"unsupported PaddleX recognition query: {query!r}")
                height, width = image.shape[:2]
                crop = Image.fromarray(
                    np.ascontiguousarray(image[:, :, ::-1]),
                    mode="RGB",
                )
                requests.append(
                    RecognitionInput(
                        request_id=(
                            f"paddlex_batch_{batch_index:06d}_item_{item_index:06d}"
                        ),
                        layout_order=item_index,
                        label=PROMPT_LABELS[query],
                        prompt=query,
                        box=Box(0.0, 0.0, float(width), float(height)),
                        crop=crop,
                        skip_special_tokens=bool(skip_special_tokens),
                    )
                )

            started = time.perf_counter()
            results, schedule = self.recognizer.recognize_stream(
                requests,
                schedule_id=f"paddlex_batch_{batch_index:06d}",
            )
            batch_wall_s = time.perf_counter() - started
            result_by_id = {result.request_id: result for result in results}
            ordered_results = [
                result_by_id[request.request_id] for request in requests
            ]
            self._record_batch(
                batch_index=batch_index,
                first_request_index=first_request_index,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                batch_wall_s=batch_wall_s,
                results=ordered_results,
                schedule=schedule,
            )
            return iter({"result": result.text} for result in ordered_results)

    def _record_batch(
        self,
        *,
        batch_index: int,
        first_request_index: int,
        min_pixels: int,
        max_pixels: int,
        batch_wall_s: float,
        results: list[Any],
        schedule: Any,
    ) -> None:
        self._summary["batches"] += 1
        self._summary["requests"] += len(results)
        self._summary["wall_s"] += float(batch_wall_s)
        self._summary["decode_graph_calls"] += int(schedule.graph_calls)
        self._summary["decode_wall_s"] += float(
            schedule.timing_s["continuous_decode_wall"]
        )
        self._summary["run_scoped_scheduler_wall_s"] += float(
            schedule.timing_s["run_scoped_scheduler_wall"]
        )
        self._summary["kv_prefix_bytes_copied"] += int(
            schedule.kv_prefix_bytes_copied
        )
        self._pixel_profiles[f"{min_pixels}:{max_pixels}"] += len(results)

        for item_index, result in enumerate(results):
            self._summary["generated_tokens_including_eos"] += int(
                result.generated_tokens_including_eos
            )
            self._summary["decode_tokens_after_prefill_including_eos"] += int(
                result.decode_tokens_after_prefill_including_eos
            )
            self._summary["input_tokens"] += int(result.input_tokens)
            self._summary["projected_image_tokens"] += int(
                result.projected_image_tokens
            )
            self._summary["real_vision_tokens"] += int(
                result.vision.get("real_vision_tokens", 0)
            )
            self._summary["physical_vision_tokens"] += int(
                result.vision.get("physical_vision_tokens", 0)
            )
            self._summary["real_text_tokens"] += int(
                result.text_prefill.get("real_text_tokens", 0)
            )
            self._summary["physical_text_tokens"] += int(
                result.text_prefill.get("physical_text_tokens", 0)
            )
            self._stop_reasons[result.stop_reason] += 1
            self._vision_routes[str(result.vision.get("execution", "unknown"))] += 1
            self._text_routes[str(result.text_prefill.get("execution", "unknown"))] += 1
            for name, seconds in result.device_stage_s.items():
                self._device_stage_s[name] += float(seconds)
            trace_record = {
                "global_request_index": first_request_index + item_index,
                "batch_index": batch_index,
                "batch_item_index": item_index,
                "request_id": result.request_id,
                "label": result.label,
                "prompt": result.prompt,
                "crop_size": list(result.crop_size),
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
                "input_tokens": result.input_tokens,
                "projected_image_tokens": result.projected_image_tokens,
                "token_ids": result.token_ids,
                "text": result.text,
                "stop_reason": result.stop_reason,
                "generated_tokens_including_eos": (
                    result.generated_tokens_including_eos
                ),
                "decode_tokens_after_prefill_including_eos": (
                    result.decode_tokens_after_prefill_including_eos
                ),
                "vision": result.vision,
                "text_prefill": result.text_prefill,
                "timing_s": result.timing_s,
                "device_stage_s": result.device_stage_s,
            }
            self._trace.write(
                json.dumps(trace_record, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
        self._trace.flush()

    def summary(self) -> dict[str, Any]:
        data = dict(self._summary)
        wall_s = float(data["wall_s"])
        output_tokens = int(data["generated_tokens_including_eos"])
        physical_vision_tokens = int(data["physical_vision_tokens"])
        physical_text_tokens = int(data["physical_text_tokens"])
        data.update(
            {
                "stop_reason_counts": dict(sorted(self._stop_reasons.items())),
                "vision_execution_counts": dict(sorted(self._vision_routes.items())),
                "text_execution_counts": dict(sorted(self._text_routes.items())),
                "pixel_profile_request_counts": dict(
                    sorted(self._pixel_profiles.items())
                ),
                "device_stage_s": dict(sorted(self._device_stage_s.items())),
                "output_tok_per_s": (
                    output_tokens / wall_s if wall_s > 0 else None
                ),
                "vision_useful_token_fraction": (
                    int(data["real_vision_tokens"]) / physical_vision_tokens
                    if physical_vision_tokens > 0
                    else None
                ),
                "text_useful_token_fraction": (
                    int(data["real_text_tokens"]) / physical_text_tokens
                    if physical_text_tokens > 0
                    else None
                ),
                "trace_path": str(self.trace_path),
            }
        )
        return data

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._trace.flush()
                self._trace.close()
                self._closed = True
