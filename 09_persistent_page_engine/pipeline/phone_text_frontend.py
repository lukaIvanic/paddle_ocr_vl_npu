"""Deterministic frontend for the fixed 1280x1920 phone-text workload."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from paddleocr_vl.serving.types import RecognitionRequest


@dataclass(frozen=True)
class _DecodedPhonePage:
    ordinal: int
    image_path: Path
    image: np.ndarray
    decode_timing: dict[str, float | int]
    page_started_s: float


@dataclass(frozen=True)
class _PreprocessedPhonePage:
    decoded: _DecodedPhonePage
    boxes: tuple[tuple[int, tuple[int, int, int, int]], ...]
    segmentation_s: float


@dataclass(frozen=True)
class _DetectedPhonePage:
    preprocessed: _PreprocessedPhonePage


@dataclass(frozen=True)
class _PreparedPhonePage:
    ordinal: int
    image_path: Path
    image_size: tuple[int, int]
    blocks: list[dict[str, Any]]
    requests: list[RecognitionRequest]
    request_block_indices: list[int]
    figure_token_maps: dict[int, dict[str, str]]
    dropped_figure_paths: set[str]
    document_images: list[dict[str, Any]]
    timing_s: dict[str, float]
    statistics: dict[str, Any]


class PhoneText20Frontend:
    """Create tight OCR crops from 20 fixed line slots without a layout model.

    The product contract is deliberately narrow: a 1280x1920 light screenshot,
    up to 20 single-line text rows, and no OCR for the status bar or title.  The
    row positions are fixed.  A grayscale threshold only trims each slot to its
    actual ink bounds; it does not infer document semantics.
    """

    expected_size = (1280, 1920)
    line_count = 20
    body_x = (40, 1240)
    first_slot_y = 190
    slot_stride = 80
    slot_height = 78
    ink_threshold = 160
    pad_x = 20
    pad_y = 8

    def __init__(self) -> None:
        self.setup_s = 0.0
        self.graph_capture = False

    def decode_page(self, image_path: Path, ordinal: int) -> _DecodedPhonePage:
        path = image_path.expanduser().resolve()
        page_started_s = time.perf_counter()
        read_started = time.perf_counter()
        compressed = path.read_bytes()
        read_finished = time.perf_counter()
        with Image.open(path) as source:
            image = np.asarray(source.convert("RGB"), dtype=np.uint8).copy()
        decode_finished = time.perf_counter()
        height, width = image.shape[:2]
        if (width, height) != self.expected_size:
            raise ValueError(
                "phone_text_20 requires an exact 1280x1920 image; "
                f"got {width}x{height} for {path}"
            )
        return _DecodedPhonePage(
            ordinal=int(ordinal),
            image_path=path,
            image=image,
            decode_timing={
                "compressed_bytes": len(compressed),
                "file_read_s": read_finished - read_started,
                "image_decode_s": decode_finished - read_finished,
            },
            page_started_s=page_started_s,
        )

    def _line_boxes(
        self,
        image: np.ndarray,
    ) -> tuple[tuple[int, tuple[int, int, int, int]], ...]:
        x0, x1 = self.body_x
        boxes: list[tuple[int, tuple[int, int, int, int]]] = []
        for line_index in range(self.line_count):
            slot_y0 = self.first_slot_y + line_index * self.slot_stride
            slot_y1 = slot_y0 + self.slot_height
            slot = image[slot_y0:slot_y1, x0:x1]
            # Fixed-point BT.601 luminance.  The threshold excludes the faint
            # row separators while retaining antialiased black text.
            rgb = slot.astype(np.uint16)
            gray = (
                77 * rgb[:, :, 0]
                + 150 * rgb[:, :, 1]
                + 29 * rgb[:, :, 2]
            ) >> 8
            ys, xs = np.nonzero(gray < self.ink_threshold)
            if not len(xs):
                continue
            ink_x0 = max(x0, x0 + int(xs.min()) - self.pad_x)
            ink_x1 = min(x1, x0 + int(xs.max()) + 1 + self.pad_x)
            ink_y0 = max(slot_y0, slot_y0 + int(ys.min()) - self.pad_y)
            ink_y1 = min(slot_y1, slot_y0 + int(ys.max()) + 1 + self.pad_y)
            boxes.append(
                (line_index, (ink_x0, ink_y0, ink_x1, ink_y1))
            )
        return tuple(boxes)

    def preprocess_decoded_page(
        self,
        decoded: _DecodedPhonePage,
    ) -> _PreprocessedPhonePage:
        started = time.perf_counter()
        boxes = self._line_boxes(decoded.image)
        return _PreprocessedPhonePage(
            decoded=decoded,
            boxes=boxes,
            segmentation_s=time.perf_counter() - started,
        )

    def transfer_preprocessed_page(
        self,
        preprocessed: _PreprocessedPhonePage,
    ) -> _PreprocessedPhonePage:
        return preprocessed

    def detect_transferred_page(
        self,
        transferred: _PreprocessedPhonePage,
    ) -> _DetectedPhonePage:
        return _DetectedPhonePage(preprocessed=transferred)

    def prepare_detected_page(
        self,
        detected: _DetectedPhonePage,
        *,
        min_pixels: int | None = None,
        max_pixels: int = 1_003_520,
        text_max_pixels: int | None = None,
        table_max_pixels: int | None = None,
        text_crop_scale: float = 1.0,
    ) -> _PreparedPhonePage:
        del table_max_pixels
        if not 0.0 < text_crop_scale <= 1.0:
            raise ValueError(
                "text_crop_scale must be in (0, 1], got "
                f"{text_crop_scale}"
            )
        preprocessed = detected.preprocessed
        decoded = preprocessed.decoded
        image = decoded.image
        effective_min_pixels = int(min_pixels or 112_896)
        effective_max_pixels = int(text_max_pixels or max_pixels)
        if effective_max_pixels < effective_min_pixels:
            raise ValueError(
                "text max_pixels must not be smaller than min_pixels: "
                f"min={effective_min_pixels} max={effective_max_pixels}"
            )

        preparation_started = time.perf_counter()
        blocks: list[dict[str, Any]] = []
        requests: list[RecognitionRequest] = []
        request_block_indices: list[int] = []
        for line_index, box in preprocessed.boxes:
            x0, y0, x1, y1 = box
            crop_rgb = np.ascontiguousarray(image[y0:y1, x0:x1])
            crop = Image.fromarray(crop_rgb)
            source_crop_size = tuple(int(value) for value in crop.size)
            if text_crop_scale != 1.0:
                scaled_size = tuple(
                    max(1, round(value * text_crop_scale))
                    for value in source_crop_size
                )
                crop = crop.resize(scaled_size, resample=Image.Resampling.BICUBIC)
            block_index = len(blocks)
            blocks.append(
                {
                    "label": "text",
                    "score": 1.0,
                    "box": [x0, y0, x1, y1],
                    "coordinate": [x0, y0, x1, y1],
                    "polygon_points": None,
                    "group_id": line_index,
                    "img": None,
                }
            )
            requests.append(
                RecognitionRequest(
                    request_id=(
                        f"page_{decoded.ordinal:06d}_line_{line_index:02d}"
                    ),
                    crop=crop,
                    prompt="OCR:",
                    skip_special_tokens=True,
                    min_pixels=effective_min_pixels,
                    max_pixels=effective_max_pixels,
                    source_crop_size=source_crop_size,
                )
            )
            request_block_indices.append(block_index)

        preparation_s = time.perf_counter() - preparation_started
        decode = decoded.decode_timing
        page_total_s = time.perf_counter() - decoded.page_started_s
        return _PreparedPhonePage(
            ordinal=decoded.ordinal,
            image_path=decoded.image_path,
            image_size=self.expected_size,
            blocks=blocks,
            requests=requests,
            request_block_indices=request_block_indices,
            figure_token_maps={},
            dropped_figure_paths=set(),
            document_images=[],
            timing_s={
                "file_read_s": float(decode["file_read_s"]),
                "image_decode_s": float(decode["image_decode_s"]),
                "phone_line_segmentation_s": preprocessed.segmentation_s,
                "layout_model_submit_s": 0.0,
                "layout_postprocess_s": 0.0,
                "page_preparation_s": preparation_s,
                "page_total_s": page_total_s,
            },
            statistics={
                "frontend": "phone_text_20",
                "use_layout_detection": False,
                "configured_line_slots": self.line_count,
                "nonempty_line_slots": len(blocks),
                "raw_layout_boxes": 0,
                "filtered_layout_boxes": 0,
                "merged_blocks": len(blocks),
                "requests": len(requests),
            },
        )
