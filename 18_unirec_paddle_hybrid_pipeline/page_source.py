"""Paddle-owned layout/crop/assembly contract; recognition routing only is new."""
from collections import Counter, deque
from dataclasses import replace
from pathlib import Path
from threading import Condition
import time


class PageInbox:
    """Independent page arrivals; close is distinct from temporary emptiness."""
    def __init__(self):
        self.items = deque()
        self.closed = False
        self.condition = Condition()
        self.wakeup = lambda: None

    def submit(self, path):
        with self.condition:
            if self.closed:
                raise RuntimeError("page input is closed")
            self.items.append(Path(path))
            self.condition.notify_all()
            self.wakeup()

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
            self.wakeup()

    def wait(self):
        with self.condition:
            while not self.items and not self.closed:
                self.condition.wait()


class PageSource:
    def set_wakeup(self, notify):
        self.inbox.wakeup = notify
        self.preparation.notify = notify

    def __init__(self, inbox, frontend, routing, emit_page, emit_trace):
        self.inbox, self.frontend, self.routing = inbox, frontend, routing
        self.emit_page, self.emit_trace = emit_page, emit_trace
        self.pages = {}
        self.owners = {}
        self.ordinal = 0
        self.completed = 0
        from layout_preparation import LayoutPreparation
        self.preparation = LayoutPreparation(self.prepare_input, self.detect, self.prepare_crops)
        self.frontend_stage_s = Counter()

    def prepare_input(self, path, ordinal):
        import torch
        with torch.inference_mode():
            return self.frontend.preprocess_decoded_page(self.frontend.decode_page(path, ordinal))

    def detect(self, inputs):
        import torch
        detected = self.frontend.detect_preprocessed_page(inputs)
        # All selected metadata/masks are already CPU-owned. Fence any remaining
        # layout work before another model gets the shared compute owner.
        torch.npu.synchronize()
        return detected

    def prepare_crops(self, detected):
        import torch
        with torch.inference_mode():
            prepared = self.frontend.prepare_detected_page(
                detected, min_pixels=28224, max_pixels=802816, text_crop_scale=1.0,
            )
            # Preserve the previous routing-specific resize, now on CPU worker.
            for i, request in enumerate(prepared.requests):
                if self.routing.model_for(request.prompt) == "paddle" and request.prompt == "OCR:":
                    from PIL import Image
                    size = tuple(max(1, round(v * 0.5)) for v in request.crop.size)
                    prepared.requests[i] = replace(request, crop=request.crop.resize(size, Image.Resampling.BICUBIC))
        return prepared, detected.decoded.page_started_s

    def pump(self):
        self.preparation.check_errors()
        with self.inbox.condition:
            if self.preparation.input_future is None and self.inbox.items:
                self.preparation.submit(self.inbox.items.popleft(), self.ordinal)
                self.ordinal += 1

    @property
    def can_advance(self):
        return self.preparation.available

    @property
    def has_pending(self):
        return bool(self.inbox.items) or self.preparation.pending

    @property
    def exhausted(self):
        return self.inbox.closed and not self.has_pending

    def wait(self):
        self.inbox.wait()

    def advance(self, adapters):
        result = self.preparation.advance()
        if result is None:
            return
        prepared, started = result
        self.frontend_stage_s.update(prepared.timing_s)
        state = {"prepared": prepared, "recognition": {}, "remaining": len(prepared.requests), "started": started}
        self.pages[prepared.ordinal] = state
        routed = {name: [] for name in adapters}
        for request, index in zip(prepared.requests, prepared.request_block_indices, strict=True):
            model = self.routing.model_for(request.prompt)
            self.owners[request.request_id] = (prepared.ordinal, index, request.prompt)
            routed[model].append(request)
        for model, requests in routed.items():
            adapters[model].enqueue_page(requests)
        prepared.requests.clear()
        prepared.request_block_indices.clear()
        if not state["remaining"]:
            self.finish(prepared.ordinal)

    def summary(self):
        return {**self.preparation.summary(), "frontend_stage_s": dict(self.frontend_stage_s)}

    def close(self):
        self.preparation.close()

    def complete(self, request_id, text, token_ids, stop_reason, model):
        ordinal, index, prompt = self.owners.pop(request_id)
        state = self.pages[ordinal]
        state["recognition"][index] = text
        state["remaining"] -= 1
        self.emit_trace({
            "request_id": request_id,
            "page": state["prepared"].image_path.name,
            "block_index": index,
            "prompt": prompt,
            "model": model,
            "text": text,
            "token_ids": token_ids,
            "stop_reason": stop_reason,
        })
        if not state["remaining"]:
            self.finish(ordinal)

    def finish(self, ordinal):
        from pipeline.layout_output import OwnedPageResult, assemble_page_blocks
        state = self.pages.pop(ordinal)
        p = state["prepared"]
        result = OwnedPageResult(
            input_path=p.image_path,
            width=p.image_size[0],
            height=p.image_size[1],
            blocks=assemble_page_blocks(
                p.blocks, state["recognition"],
                figure_token_maps=p.figure_token_maps,
                dropped_figure_paths=p.dropped_figure_paths,
            ),
            document_images=p.document_images,
        )
        self.emit_page(result)
        self.completed += 1
        print(f"HYBRID page_finish ordinal={ordinal} completed={self.completed} page_wall_s={time.perf_counter()-state['started']:.3f}", flush=True)
