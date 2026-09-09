"""Stream UniRec vision -> page-local text packs -> continuous decode.

The coordinator supplies cross-page CPU work. Three persistent stage threads
overlap only inside serve(); no stage can submit after serve() has returned.
Vision uses experiment12's owner and publishes completed key groups early,
as its standalone service does. Page assembly stays on the coordinator.
"""
from collections import deque
from queue import Queue, Empty
from types import SimpleNamespace

from .unirec import UniRecAdapter
from streamed_execution import StreamedExecution


class StreamedUniRecAdapter(UniRecAdapter):
    streamed = True

    def __init__(self, *args, cpu_workers=4, cpu_threads=8, **kwargs):
        self.completions = Queue()
        super().__init__(*args, **kwargs)
        from unirec_cpu_pool import UniRecCpuPool
        self.cpu.close()
        self.cpu = UniRecCpuPool(self.runner.processor, self.capacity, cpu_workers, cpu_threads)
        self.steps.close()  # Only primed at the initial pre-NPU yield.
        self.steps = self.decoder.iter_run(self.source, on_complete=self.complete, cooperative_refill=True,
                                          on_step=lambda row: self.timing.unirec_step(row))
        next(self.steps)
        import torch
        self.text_stream = torch.npu.Stream(device=self.runner.device)
        self.decode_stream = torch.npu.Stream(device=self.runner.device)
        self.execution = StreamedExecution()
        self.chunks = deque()
        self.text_groups = deque()
        self.external_pending = True
        self.input_closed = False
        self.next_source = 0
        self.buffer_capacity = self.capacity  # Standalone vision_record_budget=128.
        self.buffer_rows = 0
        self.buffer_high_water = 0
        self.text_reserved = 0
        self.window_count = 0
        self.window_rows = []
        self.pages_per_window = []
        self.decode_initialized = False

    def set_supply(self, pending, closed):
        self.external_pending, self.input_closed = pending, closed

    def set_upstream(self, pending, *, closed=False):
        # Coordinator pending includes CPU work; staged vision/text also count.
        remaining = self.external_pending or bool(self.pending) or self.buffer_rows > 0
        super().set_upstream(remaining, closed=self.input_closed and not remaining)

    @property
    def prefill_available(self):
        if self.chunks or self.text_groups:
            return True
        count = sum(len(page) for page in self.pending)
        if not count:
            return False
        # Accumulate across pages while upstream can replenish this window.
        if self.external_pending and count < self.buffer_capacity:
            return False
        return self.cpu.ready(self.planned_prefill_requests())

    def planned_prefill_requests(self):
        # Timing snapshots must describe the actual cross-page prerequisite,
        # not only the first page inherited from the serial adapter.
        target = min(sum(len(page) for page in self.pending), self.buffer_capacity)
        ready = []
        for page in self.pending:
            ready.extend(list(page)[:target-len(ready)])
            if len(ready) == target:
                break
        return ready

    def _ingest(self):
        from run_opendoc_batched_unirec import iter_greedy_text_packs
        from vision_full_batch import PreprocessedVisionInput
        capacity = self.buffer_capacity - self.buffer_rows
        while capacity and self.pending:
            page = self.pending[0]
            count = min(len(page), capacity, self.ready_capacity)
            requests = list(page)[:count]
            if not self.cpu.ready(requests):
                break
            prepared = self.cpu.take(requests)
            for _ in requests:
                page.popleft()
            if not page:
                self.pending.popleft()
            chunk = SimpleNamespace(requests=requests, inputs=[], encoded={}, groups=None, submitted=False)
            crops = []
            for request, (pixels, image_size) in zip(requests, prepared, strict=True):
                index = self.next_source
                self.next_source += 1
                chunk.inputs.append(PreprocessedVisionInput(index, pixels, image_size, request.request_id))
                crops.append(SimpleNamespace(image_size=image_size, source_index=index, request=request))
            chunk.groups = []
            for packed, group in iter_greedy_text_packs(crops, runner=self.runner):
                if not packed:
                    raise RuntimeError("UniRec crop exceeds production packed text contract")
                chunk.groups.append(group)
            self.chunks.append(chunk)
            self.buffer_rows += count
            self.buffer_high_water = max(self.buffer_high_water, self.buffer_rows)
            self.prefill_request_counts[count] += 1
            capacity -= count

    def _vision_job(self, chunks):
        import torch
        import torch_npu
        torch_npu.npu.set_device(self.runner.device)
        records = [dict(source_index=item.source_index,
                        processed_image_size=(item.processed_width, item.processed_height),
                        preprocessed_input=item) for chunk in chunks for item in chunk.inputs]
        with torch.inference_mode():
            _, report = self.vision.owner.encode(records, retain_outputs=False, retain_loaded_graphs=True,
                on_encoded_batch=lambda batch: self.execution.publish("vision", batch))
        return report

    def _text_job(self, group, values):
        import torch
        import torch_npu
        torch_npu.npu.set_device(self.runner.device)
        with torch.inference_mode(), torch.npu.stream(self.text_stream):
            for hidden, _ in values:
                hidden.record_stream(self.text_stream)
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            start.record()
            exports = self.export_prefill_group(values, record_ready_event=True)
            end.record()
        # Do not synchronize globally: decode consumes each export's ready_event.
        return group, exports, start, end

    def _decode_job(self, max_steps):
        import torch
        import torch_npu
        torch_npu.npu.set_device(self.runner.device)
        with torch.inference_mode(), torch.npu.stream(self.decode_stream):
            start = self.graph_calls
            while True:
                try:
                    state = next(self.steps)
                except StopIteration as result:
                    return True, result.value
                if state['graph_calls'] - start >= max_steps:
                    return False, state
                if state['active'] < self.capacity and not self.ready_count and self.source.upstream_pending:
                    return False, state

    def complete(self, result):
        # Decoder thread owns token history; only owner thread publishes pages.
        self.completions.put(result)

    def _publish_completions(self):
        while True:
            try:
                result = self.completions.get_nowait()
            except Empty:
                break
            super().complete(result)

    def _accept_event(self, kind, stage, value, text_events):
        if kind == "publish":
            for item in value:
                chunk = next(c for c in self.chunks if any(i.source_index == item.source_index for i in c.inputs))
                chunk.encoded[item.source_index] = item
                if len(chunk.encoded) == len(chunk.inputs):
                    for group in chunk.groups:
                        self.text_groups.append((chunk, group))
            return
        if stage == "vision":
            self.vision.record_report(value)
        elif stage == "text":
            group, exports, start, end = value
            for crop, item in zip(group, exports, strict=True):
                self.publish_prefill(crop.request, item)
            self.text_reserved -= len(group)
            self.buffer_rows -= len(group)
            text_events.append((start, end))
            used = {c.source_index for c in group}
            for chunk in list(self.chunks):
                for index in used:
                    chunk.encoded.pop(index, None)
                chunk.groups = [g for g in chunk.groups if g is not group]
                if not chunk.groups:
                    self.chunks.remove(chunk)
        else:
            finished, state = value
            if finished:
                self.done, self.summary = True, state
                self.active = 0
            else:
                self.active, self.graph_calls = state["active"], state["graph_calls"]
                self.decode_initialized = True
            self._publish_completions()

    def serve(self, decode_steps):
        import torch
        start_steps = self.graph_calls
        text_events = []
        self._ingest()
        try:
            while True:
                self.set_upstream(False)
                stop_submitting = self.graph_calls - start_steps >= decode_steps or self.done
                if not stop_submitting:
                    waiting = [chunk for chunk in self.chunks if not chunk.submitted]
                    if waiting and "vision" not in self.execution.running:
                        for chunk in waiting:
                            chunk.submitted = True
                        self.window_count += 1
                        self.window_rows.append(sum(len(c.inputs) for c in waiting))
                        self.pages_per_window.append(len(waiting))
                        self.execution.submit("vision", self._vision_job, waiting)
                    if self.text_groups and "text" not in self.execution.running:
                        chunk, group = self.text_groups[0]
                        if len(group) <= self.ready_capacity - self.ready_count - self.text_reserved:
                            self.text_groups.popleft()
                            values = [(chunk.encoded[c.source_index].hidden_states, chunk.encoded[c.source_index].prep) for c in group]
                            self.text_reserved += len(group)
                            self.execution.submit("text", self._text_job, group, values)
                    if "decode" not in self.execution.running:
                        can_decode = (self.ready_count > 0 or self.active >= self.capacity or
                                      (not self.source.upstream_pending and (self.active or self.source.close_requested)))
                        if can_decode:
                            self.execution.submit("decode", self._decode_job,
                                                  decode_steps - (self.graph_calls - start_steps))
                if not self.execution.running:
                    break
                self._accept_event(*self.execution.receive(), text_events)
        finally:
            # Errors cannot leave another stage running while Paddle/layout starts
            # or model teardown begins. Futures never block on downstream puts.
            for future in tuple(self.execution.running.values()):
                try:
                    future.result()
                except BaseException:
                    pass  # Original failure propagates; join remaining workers.
            with self.timing.scope("unirec.stream_yield_fence"):
                torch.npu.synchronize()
        for start, end in text_events:
            self.prefill_device_s["text_prefill_envelope"] += start.elapsed_time(end)/1000
        self._publish_completions()

    def stream_summary(self):
        from hybrid_timing import distribution
        return {**self.execution.summary(), "vision_windows": self.window_count,
                "vision_window_rows": distribution(self.window_rows) if self.window_rows else None,
                "vision_window_page_fragments": distribution(self.pages_per_window) if self.pages_per_window else None,
                "vision_text_buffer_capacity": self.buffer_capacity,
                "vision_text_buffer_high_water_rows": self.buffer_high_water,
                "vision_text_buffer_final_rows": self.buffer_rows}

    def close(self):
        # Generator-local inference mode is entered on this persistent thread.
        self.execution.pools["decode"].submit(self.steps.close).result()
        self.execution.close()
        super().close()
