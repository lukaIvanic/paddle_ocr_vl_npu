"""Bounded page frontend and completion routing for one persistent MinerU model.

Only CPU work runs in executors. The caller owns all NPU operations. Pages are
collection boundaries, never barriers between independent model requests.
"""
from __future__ import annotations

from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable
from threading import Condition
import time

from generation_trace import image_fingerprint, request_identity
from phase_logging import log_phase


@dataclass
class PageState:
    name: str
    image: Any = None
    blocks: Any = None
    remaining: set[int] = field(default_factory=set)


class PageInbox:
    """Thread-safe bounded live input. close_input is distinct from an empty queue."""
    def __init__(self, capacity: int = 32):
        if capacity < 1:
            raise ValueError("input capacity must be positive")
        self.capacity = capacity
        self.items = deque()
        self.condition = Condition()
        self.input_closed = False
        self.error = None

    def submit(self, name, loader):
        with self.condition:
            while len(self.items) >= self.capacity and not self.input_closed and self.error is None:
                self.condition.wait()
            if self.error is not None:
                raise RuntimeError("page consumer failed") from self.error
            if self.input_closed:
                raise RuntimeError("page input is closed")
            self.items.append((name, loader))
            self.condition.notify_all()

    def close_input(self):
        with self.condition:
            self.input_closed = True
            self.condition.notify_all()

    @property
    def closed(self):
        with self.condition:
            return self.input_closed and not self.items

    def pull(self, *, block=False):
        with self.condition:
            while block and not self.items and not self.input_closed and self.error is None:
                self.condition.wait()
            if self.error is not None:
                raise RuntimeError("page input failed") from self.error
            if not self.items:
                return None
            item = self.items.popleft()
            self.condition.notify_all()
            return item

    def wait_for_work(self):
        with self.condition:
            while not self.items and not self.input_closed and self.error is None:
                self.condition.wait()
            if self.error is not None:
                raise RuntimeError("page consumer failed") from self.error

    def fail(self, error):
        with self.condition:
            self.error = error
            self.condition.notify_all()


class BoundedWriter:
    """One ordered writer with bounded backpressure and mandatory final drain."""
    def __init__(self, write: Callable, capacity: int = 8):
        if capacity < 1:
            raise ValueError("writer capacity must be positive")
        self.write = write
        self.capacity = capacity
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mineru-writer")
        self.pending = deque()
        self.max_pending = 0

    def submit(self, *args):
        while self.pending and (self.pending[0].done() or len(self.pending) >= self.capacity):
            self.pending.popleft().result()
        self.pending.append(self.executor.submit(self.write, *args))
        self.max_pending = max(self.max_pending, len(self.pending))

    def close(self):
        try:
            while self.pending:
                self.pending.popleft().result()
        finally:
            self.executor.shutdown(wait=True, cancel_futures=True)


class MinerUPageSource:
    """Open request source fed by bounded lazy pages and layout completions.

    pull() may return None while layouts are in flight. That is not EOF.
    This source waits only on submitted CPU jobs, never on its own NPU work.
    """
    def __init__(self, client, pages: Iterable[tuple[str, Callable]], *,
                 on_page: Callable, page_window: int = 32, prepare_depth: int = 64,
                 trace=None):
        if page_window < 1 or prepare_depth < 1:
            raise ValueError("page window and preparation depth must be positive")
        if client.helper.enable_cross_page_table_merge:
            raise ValueError("streaming does not support cross-page table merge")
        self.client = client
        self.adapter = client.client
        self.inbox = pages if isinstance(pages, PageInbox) else None
        self.pages_iter = None if self.inbox is not None else iter(pages)
        self.page_window = page_window
        self.prepare_depth = prepare_depth
        self.on_page = on_page
        self.trace = trace
        self.pages: dict[str, PageState] = {}
        self.seen_pages: set[str] = set()
        self.input_exhausted = False
        self.frontend = ThreadPoolExecutor(max_workers=2, thread_name_prefix="mineru-pages")
        self.prepare = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mineru-prepare")
        self.front_jobs = deque()
        self.raw = deque()
        self.cpu_jobs = deque()
        self.inflight: dict[int, dict] = {}
        self.next_id = 0
        self.completed_pages = 0
        self.max_pages = 0
        self.max_prepare = 0
        self.cpu_wait_s = 0.0
        self.cpu_prepare_worker_s = 0.0
        self.cpu_mrope_prepare_s = 0.0
        self.request_h2d_submit_s = 0.0
        self.phase_admitted = Counter()
        self.phase_completed = Counter()
        self.phase_started = set()
        self.close_logged = False
        log_phase(
            "page_pipeline",
            "start",
            page_window=int(self.page_window),
            prepare_depth=int(self.prepare_depth),
            live_input=bool(self.inbox is not None),
        )
        try:
            self._fill_pages()
        except BaseException:
            self.close()
            raise

    def _layout_job(self, name, loader):
        image = loader()
        layout_image = self.client.helper.prepare_for_layout(image)
        prompt = self.client.prompts.get("[layout]") or self.client.prompts["[default]"]
        params = self.client.sampling_params.get("[layout]") or self.client.sampling_params.get("[default]")
        return image, (request_identity(name, "layout"), layout_image, prompt, params)

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
            self.front_jobs.append(("layout", name, self.frontend.submit(self._layout_job, name, loader)))
            self.max_pages = max(self.max_pages, len(self.pages))

    def _expand(self, image, text):
        blocks = self.client.helper.parse_layout_output(text)
        prepared = self.client.helper.prepare_for_extract(image, blocks)
        return blocks, prepared

    def _finish_page(self, name):
        page = self.pages[name]
        if page.remaining:
            raise RuntimeError(f"page has unfinished blocks: {name}")
        processed = self.client.helper.post_process(page.blocks)
        self.on_page(name, processed)
        del self.pages[name]
        self.completed_pages += 1
        self._fill_pages()

    def _harvest_frontend(self, *, block: bool):
        while self.front_jobs:
            kind, name, future = self.front_jobs[0]
            if not future.done() and not block:
                break
            started = time.perf_counter()
            result = future.result()
            self.cpu_wait_s += time.perf_counter() - started
            self.front_jobs.popleft()
            page = self.pages[name]
            if kind == "layout":
                page.image, raw_request = result
                self.raw.append(raw_request)
            else:
                page.blocks, prepared = result
                images, prompts, params, indices = prepared
                if not len(images) == len(prompts) == len(params) == len(indices):
                    raise ValueError("crop helper returned inconsistent lengths")
                page.remaining = set(indices)
                if len(page.remaining) != len(indices):
                    raise ValueError("crop helper returned duplicate block indices")
                for image, prompt, param, index in zip(images, prompts, params, indices):
                    self.raw.append((request_identity(name, "recognition", index, page.blocks[index]),
                                     image, prompt, param))
                if not indices:
                    self._finish_page(name)
            block = False

    def _prepare_cpu(self, raw_request):
        context, image, prompt, params = raw_request
        chat = self.adapter.processor.apply_chat_template(
            self.adapter.build_messages(prompt, has_image=True), tokenize=False,
            add_generation_prompt=True)
        cpu = self.adapter._prepare_cpu_inputs(image, chat)
        record = dict(context, schema_version=1, chat_prompt=chat)
        if self.trace is not None:
            record.update(image_sha256=image_fingerprint(image),
                          prompt_token_ids=cpu[0].input_ids[0].tolist())
        return record, params, cpu

    def _pump(self, *, block: bool):
        self._fill_pages()
        self._harvest_frontend(block=block and not self.cpu_jobs and not self.raw)
        while self.raw and len(self.cpu_jobs) < self.prepare_depth:
            self.cpu_jobs.append(self.prepare.submit(self._prepare_cpu, self.raw.popleft()))
            self.max_prepare = max(self.max_prepare, len(self.cpu_jobs))

    @property
    def closed(self):
        return self.input_exhausted and not self.pages

    @property
    def pending_count(self):
        return len(self.raw) + len(self.cpu_jobs) + len(self.front_jobs)

    @property
    def upstream_exhausted(self):
        return (self.input_exhausted and self.pending_count == 0
                and not any(x["phase"] == "layout" for x in self.inflight.values()))

    def pull(self, *, block: bool):
        while True:
            self._pump(block=block)
            if self.cpu_jobs:
                break
            if not block or not self.front_jobs:
                return None
        future = self.cpu_jobs[0]
        if not block and not future.done():
            return None
        started = time.perf_counter()
        record, params, cpu = future.result()
        self.cpu_wait_s += time.perf_counter() - started
        self.cpu_jobs.popleft()
        inputs, position_ids, rope_deltas, worker_s, mrope_s = cpu
        self.cpu_prepare_worker_s += worker_s
        self.cpu_mrope_prepare_s += mrope_s
        started = time.perf_counter()
        request = self.adapter._finish_generation(inputs, params, position_ids, rope_deltas)
        self.request_h2d_submit_s += time.perf_counter() - started
        record["max_new_tokens"] = request.max_new_tokens
        index = self.next_id
        self.next_id += 1
        self.inflight[index] = record
        phase = str(record["phase"])
        self.phase_admitted[phase] += 1
        if phase not in self.phase_started:
            self.phase_started.add(phase)
            log_phase(
                f"request_phase_{phase}",
                "start",
                first_request_id=record.get("request_id"),
                first_page=record.get("page"),
            )
        self._pump(block=False)
        return index, request

    def complete(self, index: int, ids: list[int]):
        record = self.inflight.pop(index)
        phase = str(record["phase"])
        self.phase_completed[phase] += 1
        if self.phase_completed[phase] == 1:
            log_phase(
                f"request_phase_{phase}",
                "first_finish",
                first_request_id=record.get("request_id"),
                first_page=record.get("page"),
                generated_tokens=len(ids),
            )
        filtered = [token for token in ids if token not in self.adapter.skip_token_ids]
        text = self.adapter.processor.batch_decode(
            [filtered], skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
        if self.trace is not None:
            self.trace.write(record, ids, text)
        name = record["page"]
        page = self.pages[name]
        if record["phase"] == "layout":
            self.front_jobs.append(("recognition", name,
                                    self.frontend.submit(self._expand, page.image, text)))
        else:
            index = record["block_index"]
            if index not in page.remaining:
                raise RuntimeError(f"duplicate or unexpected block completion: {record['request_id']}")
            page.blocks[index].content = text
            page.blocks[index].scored = None
            page.remaining.remove(index)
            if not page.remaining:
                self._finish_page(name)

    def close(self):
        if not self.close_logged:
            self.close_logged = True
            clean_finish = bool(self.closed and not self.inflight)
            for phase in sorted(self.phase_started):
                log_phase(
                    f"request_phase_{phase}",
                    "finish" if clean_finish else "stop",
                    admitted=int(self.phase_admitted[phase]),
                    completed=int(self.phase_completed[phase]),
                )
            log_phase(
                "page_pipeline",
                "finish" if clean_finish else "stop",
                pages_completed=int(self.completed_pages),
                pages_seen=len(self.seen_pages),
                remaining_inflight=len(self.inflight),
            )
        if self.inbox is not None and not self.closed:
            self.inbox.fail(RuntimeError("page stream stopped before input drained"))
        self.frontend.shutdown(wait=True, cancel_futures=True)
        self.prepare.shutdown(wait=True, cancel_futures=True)

    def wait_for_work(self):
        if self.inbox is not None and not self.input_exhausted:
            self.inbox.wait_for_work()
            self._fill_pages()
        elif not self.closed and self.pending_count == 0:
            raise RuntimeError("page source stalled without runnable work")

    def metadata(self):
        return {"pages_completed": self.completed_pages, "page_window": self.page_window,
                "max_live_pages": self.max_pages, "prepare_depth": self.prepare_depth,
                "max_cpu_preparation_queue": self.max_prepare,
                "cpu_prepare_wait_s": self.cpu_wait_s,
                "cpu_prepare_worker_s": self.cpu_prepare_worker_s,
                "cpu_mrope_prepare_s": self.cpu_mrope_prepare_s,
                "request_h2d_submit_s": self.request_h2d_submit_s,
                "remaining_inflight": len(self.inflight), "source_closed": self.closed}
