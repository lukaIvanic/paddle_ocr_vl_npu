"""MinerU preparation and ready leases behind experiment 18's adapter contract."""
from collections import deque
from adapters.base import Adapter
from hybrid_timing import timed_method
from hybrid_routing import MINERU_TASKS


class ReadySource:
    prefilled = True

    def __init__(self, complete):
        self.items = deque()
        self.complete = complete
        self.upstream = True
        self.input_closed = False

    @property
    def upstream_exhausted(self):
        return not self.upstream and not self.items

    @property
    def closed(self):
        return self.input_closed and self.upstream_exhausted

    def pull(self, *, block=False):
        if block:
            raise RuntimeError("A shared-owner ready source must not block")
        return self.items.popleft() if self.items else None


class MinerUAdapter(Adapter):
    def __init__(self, engine, client, helper, emit, *, ready_capacity=None, cpu_capacity=64):
        import torch
        from streaming_decode import iter_decode_stream
        super().__init__(engine.batch_size, emit, ready_capacity=ready_capacity)
        self.engine, self.client, self.helper = engine, client, helper
        self.source = ReadySource(self.complete)
        self.ready_arena = engine.model.allocate_static_cache(
            batch_size=self.ready_capacity, cache_length=engine.cache_length,
            device=engine.model.device, dtype=engine.model.dtype, init_mode="zeros")
        self.free_rows = deque(range(self.ready_capacity))
        self.records = {}
        self.next_index = 0
        self.ready_high_water = 0
        self.steps = iter_decode_stream(engine, self.source, cooperative=True)
        with torch.inference_mode():
            next(self.steps)
        self.start_cpu_preparation(self.prepare_cpu, cpu_capacity, "hybrid-mineru-cpu")

    @property
    def ready_count(self):
        return self.ready_capacity - len(self.free_rows)

    def set_upstream(self, pending, *, closed=False):
        self.source.upstream = pending
        self.source.input_closed = closed

    def ready_storage_summary(self):
        return dict(capacity=self.ready_capacity, live=self.ready_count,
                    high_water=self.ready_high_water, cache_length=self.engine.cache_length,
                    allocated_bytes=sum(t.numel() * t.element_size() for t in self.ready_arena.flat_tensors()),
                    active_cache_bytes=sum(t.numel() * t.element_size() for t in self.engine._arena_for_batch().flat_tensors()))

    def prepare_cpu(self, request):
        import torch
        kind = MINERU_TASKS[request.prompt]
        prompt = self.helper.prompts.get(kind) or self.helper.prompts["[default]"]
        params = self.helper.sampling_params.get(kind) or self.helper.sampling_params.get("[default]")
        with torch.inference_mode():
            image = self.helper.resize_by_need(request.crop.convert("RGB"))
            chat = self.client.processor.apply_chat_template(
                self.client.build_messages(prompt, has_image=True), tokenize=False,
                add_generation_prompt=True)
            cpu = self.client._prepare_cpu_inputs(image, chat)
        return kind, params, cpu

    def prefill(self):
        import torch
        from fixed_batch_engine import PrefilledGeneration
        requests, prepared = self.take_prepared_requests()
        entries = []
        with torch.inference_mode(), self.timing.scope("mineru.prefill_inputs_h2d"):
            for request, (kind, params, cpu) in zip(requests, prepared, strict=True):
                inputs, position_ids, rope_deltas, _worker_s, _mrope_s = cpu
                value = self.client._finish_generation(inputs, params, position_ids, rope_deltas)
                index = self.next_index
                self.next_index += 1
                self.records[index] = (request.request_id, kind, value.max_new_tokens)
                entries.append((self.free_rows.popleft(), index, value))
        self.ready_high_water = max(self.ready_high_water, self.ready_count)
        with torch.inference_mode():
            with self.timing.scope("mineru.vision_prefill"):
                _, metrics = self.engine._prepare_vision_window([(i, r) for _, i, r in entries])
                self.prefill_device_s.update(metrics)
            with self.timing.scope("mineru.text_prefill"):
                states, _, metrics = self.engine._prefill_slots(self.ready_arena, entries)
                self.prefill_device_s.update(metrics)
                self.prefill_tokens.update(vision_real=metrics["raw_vision_tokens"],
                                           text_input=metrics["text_prefill_tokens"],
                                           text_physical=metrics["physical_text_prefill_tokens"])
            for row, index, value in entries:
                lease = PrefilledGeneration(
                    cache=self.engine._slot_view(self.ready_arena, row), state=states[row],
                    prompt_length=int(value.input_ids.shape[1]), max_new_tokens=value.max_new_tokens,
                    release=lambda row=row: self.free_rows.append(row))
                self.source.items.append((index, lease))
            with self.timing.scope("mineru.prefill_yield_fence"):
                torch.npu.synchronize()

    @timed_method("mineru.completion_conversion")
    def complete(self, index, ids):
        from mineru_vl_utils.structs import ContentBlock
        request_id, kind, limit = self.records.pop(index)
        filtered = [token for token in ids if token not in self.client.skip_token_ids]
        raw = self.client.processor.batch_decode(
            [filtered], skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
        # Convert MinerU's recognition format with its existing postprocessor;
        # experiment 09 still owns layout geometry and page assembly.
        blocks = self.helper.post_process([ContentBlock(kind, [0, 0, 1, 1], content=raw)])
        if len(blocks) != 1:
            raise RuntimeError("MinerU single-crop postprocessing changed block count")
        text = blocks[0].content or ""
        reason = "eos" if ids[-1] == self.engine.eos_token_id else "length"
        self.emit(request_id, text, ids, reason, "mineru")

    def close(self):
        self.cpu.close()
        self.steps.close()
        self.source.items.clear()
        self.records.clear()
