"""Self-contained PP-DocLayoutV3 to PaddleOCR-VL request frontend."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import torch
import yaml
from kornia_rs.image import Image as KorniaImage
from PIL import Image
from torchvision.io import ImageReadMode, decode_image

from paddleocr_vl.serving.types import RecognitionRequest
from utils.timeline import TimelineRecorder

from .layout_postprocess import (
    IMAGE_LABELS,
    LayoutPostprocessor,
    crop_formula_margin,
    crop_layout_regions,
    merge_blocks,
)
from .layout_output import (
    gather_document_images,
    tokenize_table_figures,
)
from .layout_model_runtime import (
    _MaskRectangleFastPath,
    _NpuGraphCoreForward,
    _decoder_forward_final_heads_only,
    _extract_custom_vertices_vectorized,
    _extract_polygon_points_by_masks_owned,
    _install_pp_doclayout_v3_npu_indexput_compat,
    _mask_to_box_capture_friendly,
    _post_process_selected_masks_only,
)


@dataclass(frozen=True)
class PreparedLayoutPage:
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


@dataclass(frozen=True)
class DecodedLayoutPage:
    ordinal: int
    image_path: Path
    image: np.ndarray
    decode_timing: dict[str, float | int]
    page_started_s: float
    page_started_ns: int


@dataclass(frozen=True)
class DetectedLayoutPage:
    """A decoded page whose device-backed layout detection has completed."""

    decoded: DecodedLayoutPage
    boxes: list[dict[str, Any]]
    detect_timing: dict[str, float]


def _load_layout_labels(model_dir: Path) -> list[str]:
    metadata_path = model_dir / "inference.yml"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"PP-DocLayoutV3 metadata is missing: {metadata_path}"
        )
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    labels = metadata.get("label_list")
    if not isinstance(labels, list) or not labels:
        raise ValueError(f"{metadata_path} has no label_list")
    return [str(label) for label in labels]


def _find_decoder(model: Any) -> Any:
    decoder = getattr(model, "decoder", None)
    if decoder is None:
        decoder = getattr(
            getattr(getattr(model, "model", None), "decoder", None),
            "decoder",
            None,
        )
    if decoder is None:
        decoder = getattr(getattr(model, "model", None), "decoder", None)
    if decoder is None or not hasattr(decoder, "layers"):
        raise RuntimeError("PP-DocLayoutV3 decoder was not found")
    return decoder


def _normalization_divisor(processor: Any) -> torch.Tensor:
    mean, std, do_rescale = processor._fuse_mean_std_and_rescale_factor(
        do_normalize=processor.do_normalize,
        image_mean=processor.image_mean,
        image_std=processor.image_std,
        do_rescale=processor.do_rescale,
        rescale_factor=processor.rescale_factor,
        device=None,
    )
    mean_tensor = torch.as_tensor(mean, dtype=torch.float32)
    std_tensor = torch.as_tensor(std, dtype=torch.float32)
    if (
        not processor.do_normalize
        or do_rescale
        or mean_tensor.ndim != 1
        or mean_tensor.shape != std_tensor.shape
        or not torch.equal(mean_tensor, torch.zeros_like(mean_tensor))
    ):
        raise RuntimeError(
            "owned layout preprocessing requires fused zero-mean normalization"
        )
    return std_tensor.reshape(-1, 1, 1)


def _decode_rgb(path: Path) -> tuple[np.ndarray, dict[str, float | int]]:
    started = time.perf_counter()
    compressed = path.read_bytes()
    read_finished = time.perf_counter()
    if compressed.startswith(b"\x89PNG\r\n\x1a\n"):
        image = KorniaImage.decode(compressed, "RGB").data
    else:
        encoded = torch.frombuffer(bytearray(compressed), dtype=torch.uint8)
        image = (
            decode_image(encoded, mode=ImageReadMode.RGB)
            .permute(1, 2, 0)
            .numpy()
        )
    decode_finished = time.perf_counter()
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise RuntimeError(
            f"TorchVision returned an unsupported image for {path}: "
            f"shape={image.shape}, dtype={image.dtype}"
        )
    return image, {
        "compressed_bytes": len(compressed),
        "file_read_s": read_finished - started,
        "image_decode_s": decode_finished - read_finished,
    }


def _prompt_for_label(label: str) -> str:
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


def _rgb_to_pil(image: np.ndarray) -> Image.Image:
    """Copy an RGB NumPy buffer into an independent RGB image."""

    if not image.flags.c_contiguous:
        image = np.ascontiguousarray(image)
    height, width = image.shape[:2]
    return Image.frombytes(
        "RGB",
        (width, height),
        image,
        "raw",
        "RGB",
        0,
        1,
    )


class OwnedLayoutFrontend:
    """Own the exact sequential page-to-recognition-request contract."""

    def __init__(
        self,
        model_dir: Path,
        device: torch.device,
        *,
        threshold: float = 0.3,
        timeline: TimelineRecorder | None = None,
        graph_capture: bool = True,
        device_stage_timing: bool = False,
        npu_indexput_compat: bool = True,
        model_backend: str = "transformers",
        model_dtype: torch.dtype | None = None,
    ) -> None:
        from transformers import AutoImageProcessor

        if model_backend not in {"transformers", "owned"}:
            raise ValueError(
                f"unsupported layout model backend: {model_backend!r}"
            )
        if model_backend == "owned" and graph_capture:
            raise ValueError(
                "the owned PP-DocLayoutV3 model is eager-only; disable "
                "layout graph capture"
            )

        # The frontend submits one 800x800 detector resize at a time. Letting
        # PyTorch fan each resize across all 192 host CPUs costs more in thread
        # scheduling than it saves in image work.
        torch.set_num_threads(min(64, len(os.sched_getaffinity(0))))
        self.model_dir = model_dir.expanduser().resolve()
        self.model_backend = model_backend
        self.device = device
        self.timeline = timeline
        self.device_stage_timing = bool(
            device_stage_timing and self.device.type == "npu"
        )
        self.graph_capture = bool(
            graph_capture and self.device.type == "npu"
        )
        self.labels = _load_layout_labels(self.model_dir)

        setup_started = time.perf_counter()
        self.processor = AutoImageProcessor.from_pretrained(self.model_dir)
        self._normalization_divisor = _normalization_divisor(self.processor)
        if self.model_backend == "transformers":
            from transformers import AutoModelForObjectDetection

            self.model = AutoModelForObjectDetection.from_pretrained(
                self.model_dir
            )
        else:
            from .owned_layout_model import (
                OwnedPPDocLayoutV3ForObjectDetection,
            )

            self.model = (
                OwnedPPDocLayoutV3ForObjectDetection.from_pretrained(
                    self.model_dir
                )
            )
        self.model.eval().to(
            device=self.device,
            dtype=model_dtype,
        )
        self.model_dtype = next(self.model.parameters()).dtype
        self.npu_indexput_compat = bool(
            npu_indexput_compat
            and self.device.type == "npu"
            and self.model_backend == "transformers"
        )
        if self.npu_indexput_compat:
            _install_pp_doclayout_v3_npu_indexput_compat(self.model)

        if self.model_backend == "transformers":
            decoder = _find_decoder(self.model)
            decoder.forward = MethodType(
                _decoder_forward_final_heads_only,
                decoder,
            )
            decoder._layout_emit_masks = True
        self.processor.extract_custom_vertices = MethodType(
            _extract_custom_vertices_vectorized,
            self.processor,
        )
        layout_mask_guard_state = getattr(
            type(self.processor)._extract_polygon_points_by_masks,
            "_layout_mask_guard_state",
            None,
        )
        if layout_mask_guard_state is not None:
            self.processor._layout_mask_guard_state = layout_mask_guard_state
        self.processor._extract_polygon_points_by_masks = MethodType(
            _extract_polygon_points_by_masks_owned,
            self.processor,
        )
        self.mask_fast_path = _MaskRectangleFastPath(
            self.processor._extract_polygon_points_by_masks
        )
        self.processor._extract_polygon_points_by_masks = self.mask_fast_path
        if self.graph_capture:
            from transformers.models.pp_doclayout_v3 import (
                modeling_pp_doclayout_v3,
            )

            modeling_pp_doclayout_v3.mask_to_box_coordinate = (
                _mask_to_box_capture_friendly
            )
            self.model.forward = _NpuGraphCoreForward(self.model.forward)
            capture_height = int(self.processor.size["height"])
            capture_width = int(self.processor.size["width"])
            capture_input = torch.zeros(
                (1, 3, capture_height, capture_width),
                device=self.device,
                dtype=self.model_dtype,
            )
            self.model(pixel_values=capture_input)
        self.postprocessor = LayoutPostprocessor(
            self.labels,
            threshold=threshold,
        )
        self._crop_executor = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="layout-crop",
        )
        if self.device.type == "npu":
            torch.npu.synchronize()
        self.setup_s = time.perf_counter() - setup_started

    def _span(
        self,
        row: str,
        name: str,
        started_ns: int,
        *,
        flow_id: str,
        event_type: str = "work",
        args: dict[str, Any] | None = None,
    ) -> None:
        if self.timeline is not None:
            self.timeline.record_span(
                row,
                name,
                started_ns,
                time.perf_counter_ns(),
                flow_id=flow_id,
                event_type=event_type,
                args=args or {},
            )

    def _prepare_pixel_values(self, image_rgb: np.ndarray) -> torch.Tensor:
        """Run the processor's exact singleton math without batch plumbing."""

        pixel_values = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0)
        pixel_values = self.processor.resize(
            image=pixel_values,
            size=self.processor.size,
            resample=self.processor.resample,
            antialias=False,
        )
        pixel_values = pixel_values.to(dtype=torch.float32)
        pixel_values.div_(self._normalization_divisor)
        return pixel_values.to(
            device=self.device,
            dtype=self.model_dtype,
        )

    @torch.inference_mode()
    def _detect(
        self,
        image_rgb: np.ndarray,
        *,
        flow_id: str,
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        timing: dict[str, float] = {}
        height, width = image_rgb.shape[:2]

        started = time.perf_counter()
        started_ns = time.perf_counter_ns()
        inputs = {"pixel_values": self._prepare_pixel_values(image_rgb)}
        timing["layout_preprocess_h2d_s"] = time.perf_counter() - started
        self._span(
            "Layout detection",
            "Owned layout preprocessing and H2D",
            started_ns,
            flow_id=flow_id,
        )

        started = time.perf_counter()
        started_ns = time.perf_counter_ns()
        device_events: dict[str, Any] | None = None
        if self.device_stage_timing:
            import torch_npu

            device_events = {
                name: torch_npu.npu.Event(enable_timing=True)
                for name in (
                    "model_start",
                    "model_end",
                    "metadata_end",
                    "mask_start",
                    "mask_end",
                )
            }
            device_events["model_start"].record()
        outputs = self.model(**inputs)
        if device_events is not None:
            device_events["model_end"].record()
        timing["layout_model_submit_s"] = time.perf_counter() - started
        self._span(
            "Layout detection",
            "Owned PP-DocLayoutV3 inference",
            started_ns,
            flow_id=flow_id,
        )

        started = time.perf_counter()
        started_ns = time.perf_counter_ns()
        predictions = _post_process_selected_masks_only(
            self.processor,
            outputs,
            threshold=self.postprocessor.threshold,
            target_sizes=[[height, width]],
            timing=timing,
            device_timing_events=device_events,
        )
        if device_events is not None:
            timing["layout_model_device_s"] = (
                float(
                    device_events["model_start"].elapsed_time(
                        device_events["model_end"]
                    )
                )
                / 1000.0
            )
            timing["layout_device_metadata_postprocess_s"] = (
                float(
                    device_events["model_end"].elapsed_time(
                        device_events["metadata_end"]
                    )
                )
                / 1000.0
            )
            timing["layout_device_mask_postprocess_s"] = (
                float(
                    device_events["mask_start"].elapsed_time(
                        device_events["mask_end"]
                    )
                )
                / 1000.0
            )
        postprocessor_started = time.perf_counter()
        boxes = self.postprocessor(
            predictions[0],
            (width, height),
            timing=timing if self.device_stage_timing else None,
        )
        timing["layout_structural_postprocess_cpu_s"] = (
            time.perf_counter() - postprocessor_started
        )
        timing["layout_postprocess_s"] = time.perf_counter() - started
        self._span(
            "Layout postprocess",
            "Owned detector postprocessing",
            started_ns,
            flow_id=flow_id,
            args={"boxes": len(boxes)},
        )
        return boxes, timing

    def decode_page(
        self,
        image_path: Path,
        ordinal: int,
    ) -> DecodedLayoutPage:
        path = image_path.expanduser().resolve()
        page_started_s = time.perf_counter()
        page_started_ns = time.perf_counter_ns()
        decode_started_ns = time.perf_counter_ns()
        image, decode = _decode_rgb(path)
        self._span(
            "Page input",
            "Owned page read and decode",
            decode_started_ns,
            flow_id=f"page:{ordinal}",
            event_type="io",
            args={
                "bytes": decode["compressed_bytes"],
                "width": image.shape[1],
                "height": image.shape[0],
            },
        )
        return DecodedLayoutPage(
            ordinal=ordinal,
            image_path=path,
            image=image,
            decode_timing=decode,
            page_started_s=page_started_s,
            page_started_ns=page_started_ns,
        )

    def prepare_page(
        self,
        image_path: Path,
        ordinal: int,
        *,
        min_pixels: int | None = None,
        max_pixels: int = 1_003_520,
        text_max_pixels: int | None = None,
        text_crop_scale: float = 1.0,
    ) -> PreparedLayoutPage:
        return self.prepare_decoded_page(
            self.decode_page(image_path, ordinal),
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            text_max_pixels=text_max_pixels,
            text_crop_scale=text_crop_scale,
        )

    def prepare_decoded_page(
        self,
        decoded: DecodedLayoutPage,
        *,
        min_pixels: int | None = None,
        max_pixels: int = 1_003_520,
        text_max_pixels: int | None = None,
        text_crop_scale: float = 1.0,
    ) -> PreparedLayoutPage:
        return self.prepare_detected_page(
            self.detect_decoded_page(decoded),
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            text_max_pixels=text_max_pixels,
            text_crop_scale=text_crop_scale,
        )

    def detect_decoded_page(
        self,
        decoded: DecodedLayoutPage,
    ) -> DetectedLayoutPage:
        boxes, detect_timing = self._detect(
            decoded.image,
            flow_id=f"page:{decoded.ordinal}",
        )
        return DetectedLayoutPage(
            decoded=decoded,
            boxes=boxes,
            detect_timing=detect_timing,
        )

    def prepare_detected_page(
        self,
        detected: DetectedLayoutPage,
        *,
        min_pixels: int | None = None,
        max_pixels: int = 1_003_520,
        text_max_pixels: int | None = None,
        text_crop_scale: float = 1.0,
    ) -> PreparedLayoutPage:
        decoded = detected.decoded
        ordinal = decoded.ordinal
        path = decoded.image_path
        image = decoded.image
        decode = decoded.decode_timing
        flow_id = f"page:{ordinal}"

        boxes = detected.boxes
        detect_timing = detected.detect_timing
        preparation_started = time.perf_counter()
        preparation_started_ns = time.perf_counter_ns()
        document_images = gather_document_images(image, boxes)
        blocks = crop_layout_regions(
            image,
            boxes,
            executor=self._crop_executor if len(boxes) > 1 else None,
        )
        image_labels = IMAGE_LABELS + ["chart", "seal"]
        blocks = merge_blocks(
            blocks,
            non_merge_labels=image_labels + ["table"],
        )

        effective_min_pixels = int(min_pixels or 112_896)
        effective_text_max_pixels = int(text_max_pixels or max_pixels)
        if effective_text_max_pixels < effective_min_pixels:
            raise ValueError(
                "text max_pixels must not be smaller than min_pixels: "
                f"min={effective_min_pixels} max={effective_text_max_pixels}"
            )
        if not 0.0 < text_crop_scale <= 1.0:
            raise ValueError(
                "text_crop_scale must be in (0, 1], got "
                f"{text_crop_scale}"
            )
        request_specs: list[tuple[int, np.ndarray, str]] = []
        figure_token_maps: dict[int, dict[str, str]] = {}
        dropped_figure_paths: set[str] = set()
        for block_index, block in enumerate(blocks):
            block_image = block["img"]
            label = block["label"]
            if label in image_labels or block_image is None:
                continue
            prompt = _prompt_for_label(label)
            if label == "table":
                (
                    block_image,
                    figure_token_map,
                    dropped,
                ) = tokenize_table_figures(
                    block_image,
                    block["box"],
                    document_images,
                )
                figure_token_maps[block_index] = figure_token_map
                dropped_figure_paths.update(dropped)
            elif "formula" in label and label != "formula_number":
                cropped = crop_formula_margin(block_image)
                crop_height, crop_width = cropped.shape[:2]
                if crop_height > 2 and crop_width > 2:
                    block_image = cropped
            request_specs.append((block_index, block_image, prompt))

        if len(request_specs) > 1:
            crops = list(
                self._crop_executor.map(
                    _rgb_to_pil,
                    (
                        block_image
                        for _, block_image, _ in request_specs
                    ),
                )
            )
        else:
            crops = [
                _rgb_to_pil(block_image)
                for _, block_image, _ in request_specs
            ]

        requests: list[RecognitionRequest] = []
        request_block_indices: list[int] = []
        for (block_index, _, prompt), crop in zip(
            request_specs,
            crops,
        ):
            source_crop_size = tuple(int(value) for value in crop.size)
            if prompt == "OCR:" and text_crop_scale != 1.0:
                scaled_size = tuple(
                    max(1, round(value * text_crop_scale))
                    for value in source_crop_size
                )
                crop = crop.resize(
                    scaled_size,
                    resample=Image.Resampling.BICUBIC,
                )
            requests.append(
                RecognitionRequest(
                    request_id=(
                        f"page_{ordinal:06d}_block_{block_index:06d}"
                    ),
                    crop=crop,
                    prompt=prompt,
                    skip_special_tokens=True,
                    min_pixels=effective_min_pixels,
                    max_pixels=(
                        effective_text_max_pixels
                        if prompt == "OCR:"
                        else int(max_pixels)
                    ),
                    source_crop_size=source_crop_size,
                )
            )
            request_block_indices.append(block_index)

        # Requests and visible page artifacts now own independent RGB images.
        # Release non-visible NumPy crops before layout-first mode queues the
        # prepared page for recognition.
        for block in blocks:
            if block["label"] not in image_labels:
                block["img"] = None
        preparation_s = time.perf_counter() - preparation_started
        self._span(
            "Crop / page preparation",
            "Owned filter, crop, merge, and request build",
            preparation_started_ns,
            flow_id=flow_id,
            args={"requests": len(requests)},
        )

        timing = {
            "file_read_s": float(decode["file_read_s"]),
            "image_decode_s": float(decode["image_decode_s"]),
            **detect_timing,
            "page_preparation_s": preparation_s,
            "page_total_s": time.perf_counter() - decoded.page_started_s,
        }
        self._span(
            "Page input",
            "Owned page frontend",
            decoded.page_started_ns,
            flow_id=flow_id,
            event_type="scope",
            args={"requests": len(requests)},
        )
        return PreparedLayoutPage(
            ordinal=ordinal,
            image_path=path,
            image_size=(image.shape[1], image.shape[0]),
            blocks=blocks,
            requests=requests,
            request_block_indices=request_block_indices,
            figure_token_maps=figure_token_maps,
            dropped_figure_paths=dropped_figure_paths,
            document_images=document_images,
            timing_s=timing,
            statistics={
                "raw_layout_boxes": len(boxes),
                "filtered_layout_boxes": len(boxes),
                "merged_blocks": len(blocks),
                "requests": len(requests),
            },
        )
