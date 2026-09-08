"""One persistent CPU worker; bounded submitted + completed request storage."""
from concurrent.futures import ThreadPoolExecutor
from threading import get_ident
import time


class CpuPreparation:
    def __init__(self, prepare, capacity, name):
        self.prepare = prepare
        self.capacity = capacity
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)
        self.futures = {}
        self.submitted = 0
        self.consumed = 0
        self.high_water = 0
        self.service_s = 0.0
        self.worker_thread_ids = set()
        self.notify = lambda: None

    def _run(self, request):
        started = time.perf_counter()
        value = self.prepare(request)
        return value, time.perf_counter() - started, get_ident()

    def pump(self, pages, notify):
        self.notify = notify
        # Surface a worker failure even while another model is decoding.
        for future in self.futures.values():
            if future.done() and future.exception() is not None:
                future.result()
        for page in pages:
            for request in page:
                key = request.request_id
                if key in self.futures:
                    continue
                if len(self.futures) >= self.capacity:
                    return
                future = self.executor.submit(self._run, request)
                self.futures[key] = future
                self.submitted += 1
                self.high_water = max(self.high_water, len(self.futures))
                future.add_done_callback(lambda _future: self.notify())

    def ready(self, requests):
        return all(r.request_id in self.futures and self.futures[r.request_id].done()
                   for r in requests)

    def take(self, requests):
        if not self.ready(requests):
            raise RuntimeError("CPU preparation consumed before completion")
        values = []
        for request in requests:
            value, seconds, thread_id = self.futures.pop(request.request_id).result()
            values.append(value)
            self.service_s += seconds
            self.worker_thread_ids.add(thread_id)
            self.consumed += 1
        return values

    def summary(self):
        return dict(capacity=self.capacity, submitted=self.submitted, consumed=self.consumed,
                    high_water_requests=self.high_water, service_wall_s=self.service_s,
                    worker_threads_observed=len(self.worker_thread_ids))

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.futures.clear()
