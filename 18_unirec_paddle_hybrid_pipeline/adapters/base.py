from collections import Counter, deque


class Adapter:
    def __init__(self, capacity, emit):
        self.capacity = capacity
        self.emit = emit
        self.pending = deque()
        self.active = 0
        self.graph_calls = 0
        self.done = False
        self.summary = None
        self.prefill_tokens = Counter()
        self.prefill_device_s = Counter()

    @property
    def occupied(self):
        return self.active + self.ready_count

    @property
    def free(self):
        return self.capacity - self.occupied

    def advance(self, count):
        import torch
        start = self.graph_calls
        with torch.inference_mode():
            while not self.done:
                try:
                    state = next(self.steps)
                except StopIteration as finished:
                    self.done = True
                    self.summary = finished.value
                    break
                self.active = state["active"]
                self.graph_calls = state["graph_calls"]
                if self.graph_calls - start >= count or self.occupied < self.capacity:
                    break
        # Cross-model compute is deliberately serialized. This also completes
        # token transfers; generator-local pending-copy/slot epoch ownership
        # remains entirely in the original engine.
        torch.npu.synchronize()
