"""Paddle-owned layout/crop/assembly contract; recognition routing only is new."""
from collections import deque
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

    def submit(self, path):
        with self.condition:
            if self.closed:
                raise RuntimeError("page input is closed")
            self.items.append(Path(path))
            self.condition.notify_all()

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()

    def wait(self):
        with self.condition:
            while not self.items and not self.closed:
                self.condition.wait()


class PageSource:
    def __init__(self, inbox, frontend, routing, emit_page, emit_trace):
        self.inbox, self.frontend, self.routing = inbox, frontend, routing
        self.emit_page, self.emit_trace = emit_page, emit_trace
        self.pages = {}
        self.owners = {}
        self.ordinal = 0
        self.completed = 0

    @property
    def has_pending(self):
        return bool(self.inbox.items)

    @property
    def exhausted(self):
        return self.inbox.closed and not self.has_pending

    def wait(self):
        self.inbox.wait()

    def advance(self, adapters):
        with self.inbox.condition:
            path = self.inbox.items.popleft()
        started = time.perf_counter()
        prepared = self.frontend.prepare_page(
            path, self.ordinal, min_pixels=28224, max_pixels=802816,
            text_crop_scale=1.0,
        )
        self.ordinal += 1
        state = {"prepared": prepared, "recognition": {}, "remaining": len(prepared.requests), "started": started}
        self.pages[prepared.ordinal] = state
        for request, index in zip(prepared.requests, prepared.request_block_indices, strict=True):
            model = self.routing.model_for(request.prompt)
            if model == "paddle" and request.prompt == "OCR:":
                from PIL import Image
                size = tuple(max(1, round(v * 0.5)) for v in request.crop.size)
                request = replace(request, crop=request.crop.resize(size, Image.Resampling.BICUBIC))
            self.owners[request.request_id] = (prepared.ordinal, index, request.prompt)
            adapters[model].pending.append(request)
        prepared.requests.clear()
        prepared.request_block_indices.clear()
        if not state["remaining"]:
            self.finish(prepared.ordinal)

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
