from collections import Counter, deque


class Adapter:
    def __init__(self, capacity, emit, *, ready_capacity=None):
        self.capacity = capacity
        self.ready_capacity = capacity if ready_capacity is None else ready_capacity
        self.emit = emit
        self.pending = deque()
        self.active = 0
        self.graph_calls = 0
        self.done = False
        self.summary = None
        self.prefill_tokens = Counter()
        self.prefill_device_s = Counter()
        self.prefill_request_counts = Counter()

    @property
    def occupied(self):
        return self.active + self.ready_count

    @property
    def free(self):
        return max(0, self.capacity - self.occupied)

    def enqueue_page(self, requests):
        if requests:
            self.pending.append(deque(requests))

    def take_prefill_requests(self):
        """Page-local production bounded by ready storage, not decode vacancies.

        A large page resumes on a later refill. Both model adapters use this
        contract; each retains its own vision/text grouping within the chunk.
        No blocking queue put is needed on the shared NPU-owning thread.
        """
        page = self.pending[0]
        count = min(len(page), self.ready_capacity - self.ready_count)
        if count <= 0:
            raise RuntimeError("prefill requested without ready-storage capacity")
        requests = [page.popleft() for _ in range(count)]
        if not page:
            self.pending.popleft()
        self.prefill_request_counts[count] += 1
        return requests

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
