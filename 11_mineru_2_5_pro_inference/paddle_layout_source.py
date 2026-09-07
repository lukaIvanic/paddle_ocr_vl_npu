"""Live PP-DocLayoutV3 frontend for MinerU's persistent recognition queue.

CPU workers prepare detector inputs and crops. Only the consuming inference
thread executes the detector, sharing the NPU with recognition. A bounded page
window applies backpressure to file iterators and PageInbox alike.
"""
from collections import Counter
from pathlib import Path
import sys
import threading
import time

from generation_trace import image_fingerprint
from phase_logging import log_phase
from saved_paddle_crops import LABEL_MAP, SKIP_TYPES
from streaming_pipeline import MinerUPageSource, PageState


def load_paddle_frontend(model, *, graph_capture=False):
    # Fail during setup, before loading either layout tensors or submitting pages.
    import kornia_rs
    import shapely.geometry
    import torch
    root = Path(__file__).resolve().parents[1] / "09_persistent_page_engine"
    if str(root) not in sys.path:
        sys.path.append(str(root))
    from pipeline.layout_frontend import OwnedLayoutFrontend
    frontend = OwnedLayoutFrontend(Path(model), torch.device("npu:0"), graph_capture=graph_capture)
    unknown = set(frontend.labels) - set(LABEL_MAP) - {"reference"}
    if unknown:
        frontend.close()
        raise ValueError(f"layout checkpoint has unmapped labels: {sorted(unknown)}")
    return frontend


class PaddleRegionFrontend:
    """Bridge live geometry to the crop policy frozen in the saved-layout test."""
    def __init__(self, frontend):
        self.frontend = frontend

    def prepare(self, name, loader, ordinal):
        import numpy as np
        from pipeline.layout_frontend import DecodedLayoutPage
        started = time.perf_counter()
        value = loader()
        if isinstance(value, (str, Path)):
            return self.frontend.preprocess_decoded_page(self.frontend.decode_page(Path(value), ordinal))
        image = value.convert("RGB")
        decoded = DecodedLayoutPage(ordinal, Path(name), np.asarray(image),
            {"image_load_decode_s": time.perf_counter() - started}, started, time.perf_counter_ns())
        return self.frontend.preprocess_decoded_page(decoded)

    def detect(self, prepared):
        return self.frontend.detect_preprocessed_page(prepared)

    def crops(self, detected, client):
        from PIL import Image
        from mineru_vl_utils.structs import ContentBlock
        from pipeline.layout_postprocess import crop_layout_regions, merge_blocks, IMAGE_LABELS
        boxes = self.frontend.postprocess_detected_regions(detected)
        pixels = detected.decoded.image
        height, width = pixels.shape[:2]
        # Preserve Paddle's post-merge ordering/group metadata without joining
        # crop pixels or suppressing subsequent members of a merged group.
        regions = merge_blocks(crop_layout_regions(pixels, boxes),
            non_merge_labels=IMAGE_LABELS + ["chart", "seal", "table"], merge_images=False)
        blocks, images, prompts, params, indices, metadata = [], [], [], [], [], []
        for index, region in enumerate(regions):
            label = region["label"]
            kind = LABEL_MAP[label]
            x1, y1, x2, y2 = map(int, region["box"])
            bbox = [x1 / width, y1 / height, x2 / width, y2 / height]
            blocks.append(ContentBlock(type=kind, bbox=bbox, angle=None, content=None))
            record = {"index": index, "source_label": label, "type": kind,
                      "bbox_pixels": [x1, y1, x2, y2], "bbox": bbox,
                      "polygon_points": [list(map(float, point)) for point in region["polygon_points"]],
                      "source_group_id": region.get("group_id"), "recognize": kind not in SKIP_TYPES}
            if record["recognize"]:
                crop = Image.fromarray(region["img"])
                record.update(crop_size=list(crop.size), crop_pixel_sha256=image_fingerprint(crop))
                images.append(client.helper.resize_by_need(crop))
                prompts.append(client.prompts.get(kind) or client.prompts["[default]"])
                params.append(client.sampling_params.get(kind) or client.sampling_params.get("[default]"))
                indices.append(index)
            metadata.append(record)
        return blocks, (images, prompts, params, indices), {
            "width": width, "height": height, "blocks": metadata,
            "layout_timing_s": detected.detect_timing}


class PaddleLayoutPageSource(MinerUPageSource):
    def __init__(self, client, pages, *, layout_frontend, on_layout=None, **kwargs):
        self.layout_frontend = layout_frontend
        self.on_layout = on_layout
        self.layout_owner = threading.get_ident()
        self.layout_calls = 0
        self.layout_wall_s = 0.0
        self.layout_timing = Counter()
        self.layout_region_counts = Counter()
        super().__init__(client, pages, **kwargs)

    def _fill_pages(self):
        while not self.input_exhausted and len(self.pages) < self.page_window:
            if self.inbox is not None:
                item = self.inbox.pull(block=False)
                if item is None:
                    self.input_exhausted = self.inbox.closed
                    break
                name, loader = item
            else:
                try:
                    name, loader = next(self.pages_iter)
                except StopIteration:
                    self.input_exhausted = True
                    break
            if name in self.seen_pages:
                raise ValueError(f"duplicate input page: {name}")
            self.seen_pages.add(name)
            self.pages[name] = PageState(name)
            self.front_jobs.append(("paddle_input", name, self.frontend.submit(
                self.layout_frontend.prepare, name, loader, len(self.seen_pages) - 1)))
            self.max_pages = max(self.max_pages, len(self.pages))

    def _harvest_frontend(self, *, block):
        if self.raw:
            return
        # Transform only the next queued frontend job. Prioritizing its crops
        # prevents draining all page-window detections before recognition starts.
        while self.front_jobs:
            kind, name, future = self.front_jobs[0]
            if not future.done() and not block:
                break
            if kind == "paddle_input":
                if threading.get_ident() != self.layout_owner:
                    raise RuntimeError("layout inference must run on the NPU-owning thread")
                started = time.perf_counter()
                prepared = future.result()
                self.cpu_wait_s += time.perf_counter() - started
                self.front_jobs.popleft()
                started = time.perf_counter()
                log_phase("paddle_layout", "start", page=name)
                detected = self.layout_frontend.detect(prepared)
                elapsed = time.perf_counter() - started
                self.layout_wall_s += elapsed
                self.layout_calls += 1
                log_phase("paddle_layout", "finish", page=name, elapsed_s=elapsed)
                self.front_jobs.appendleft(("paddle_crops", name, self.frontend.submit(
                    self.layout_frontend.crops, detected, self.client)))
                block = True
                continue
            if kind != "paddle_crops":
                raise RuntimeError(f"unexpected frontend job: {kind}")
            started = time.perf_counter()
            blocks, prepared, metadata = future.result()
            self.cpu_wait_s += time.perf_counter() - started
            self.front_jobs.popleft()
            self.layout_timing.update(metadata["layout_timing_s"])
            self.layout_region_counts.update(row["source_label"] for row in metadata["blocks"])
            if self.on_layout is not None:
                self.on_layout(name, metadata)
            page = self.pages[name]
            page.blocks = blocks
            images, prompts, params, indices = prepared
            if not len(images) == len(prompts) == len(params) == len(indices) or len(set(indices)) != len(indices):
                raise ValueError("invalid live crop batch")
            page.remaining = set(indices)
            from generation_trace import request_identity
            for image, prompt, param, index in zip(images, prompts, params, indices):
                self.raw.append((request_identity(name, "recognition", index, blocks[index]), image, prompt, param))
            if not indices:
                self._finish_page(name)
            # Return after one page; inherited _pump dispatches CPU preparation.
            if self.raw:
                break
            block = False

    def metadata(self):
        return {**super().metadata(), "layout_source": "PP-DocLayoutV3_live",
                "layout_calls": self.layout_calls, "layout_host_wall_s": self.layout_wall_s,
                "layout_timing_s": dict(self.layout_timing), "layout_region_counts": dict(self.layout_region_counts),
                "requests_admitted_by_phase": dict(self.phase_admitted),
                "requests_completed_by_phase": dict(self.phase_completed)}
