"""Paddle execution delegates to experiment 09's existing model and arena."""
from collections import deque
from .base import Adapter


class ReadySource:
    def __init__(self):
        self.items = deque()
        self.closed = False

    def pull(self, *, block=False):
        return self.items.popleft() if self.items else None

    def pull_for_decode_slots(self, *, block, available_slots):
        return self.pull() if available_slots > 0 else None


class PaddleAdapter(Adapter):
    def __init__(self, recognizer, emit):
        super().__init__(recognizer.batch_size, emit, ready_capacity=recognizer.ready_buffer_capacity)
        self.recognizer = recognizer
        self.source = ReadySource()
        recognizer._begin_decode_schedule()
        self.steps = recognizer.decode_scheduler.iter_run_stream(
            self.source, on_completion=self.complete,
            ready_buffer_capacity=recognizer.ready_buffer_capacity,
            ready_buffer_low_watermark=recognizer.ready_buffer_low_watermark,
        )
        import torch
        with torch.inference_mode():
            next(self.steps)

    @property
    def ready_count(self):
        return len(self.source.items)

    def set_upstream(self, pending, *, closed=False):
        self.source.closed = closed and not pending

    def prefill(self):
        import torch
        requests = self.take_prefill_requests()
        with torch.inference_mode():
            # Reuse the production grouping implementation, including its
            # vision pack target and text-pack membership limits.
            for group in self.recognizer._iter_packed_prefill_groups(requests):
                staged = self.recognizer._stage_prefill_group(group)
                inflight = self.recognizer._enqueue_staged_prefill_group(staged)
                for item in self.recognizer._finalize_prefill_group(inflight):
                    self.prefill_device_s.update(item.device_stage_s)
                    self.prefill_tokens.update({
                        "vision_real": item.vision["real_vision_tokens"],
                        "vision_physical": item.vision["physical_vision_tokens"],
                        "text_input": item.input_tokens,
                    })
                    self.source.items.append(self.recognizer._ready_from_prefilled(item))
        torch.npu.synchronize()

    def complete(self, completion):
        result = self.recognizer._result_from_completion(completion, schedule_id="hybrid:paddle")
        self.emit(result.request_id, result.text, result.token_ids, result.stop_reason, "paddle")

    def close(self):
        self.steps.close()
