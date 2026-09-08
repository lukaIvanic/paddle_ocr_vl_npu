"""Two bounded CPU stages around coordinator-owned layout detection."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from threading import get_ident
import time


class LayoutPreparation:
    def __init__(self, prepare_input, detect, prepare_crops):
        self.prepare_input, self.detect, self.prepare_crops = prepare_input, detect, prepare_crops
        self.input_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hybrid-page-input")
        self.crop_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hybrid-page-crops")
        self.input_future = self.crop_future = None
        self.notify = lambda: None
        self.counts = Counter()
        self.service_s = Counter()
        self.threads = {"input": set(), "crops": set()}
        self.high_water = 0

    @staticmethod
    def timed(function, *args):
        started = time.perf_counter()
        return function(*args), time.perf_counter() - started, get_ident()

    @property
    def pending(self):
        return self.input_future is not None or self.crop_future is not None

    @property
    def available(self):
        if self.crop_future is not None:
            return self.crop_future.done()
        return self.input_future is not None and self.input_future.done()

    def check_errors(self):
        for future in (self.input_future, self.crop_future):
            if future is not None and future.done() and future.exception() is not None:
                future.result()

    def submit(self, path, ordinal):
        if self.input_future is not None:
            raise RuntimeError("layout input slot is occupied")
        self.input_future = self.input_worker.submit(self.timed, self.prepare_input, path, ordinal)
        self.input_future.add_done_callback(lambda _: self.notify())
        self.counts["submitted"] += 1
        self.high_water = max(self.high_water, 1 + (self.crop_future is not None))

    def consume(self, future, stage):
        value, seconds, thread = future.result()
        self.counts[stage] += 1
        self.service_s[stage] += seconds
        self.threads[stage].add(thread)
        return value

    def advance(self):
        if not self.available:
            raise RuntimeError("layout stage selected before CPU completion")
        if self.crop_future is not None:
            prepared = self.consume(self.crop_future, "crops")
            self.crop_future = None
            return prepared
        inputs = self.consume(self.input_future, "input")
        self.input_future = None
        started = time.perf_counter()
        detected = self.detect(inputs)  # Only called on the coordinator.
        self.service_s["detect_owner"] += time.perf_counter() - started
        self.counts["detected"] += 1
        self.crop_future = self.crop_worker.submit(self.timed, self.prepare_crops, detected)
        self.crop_future.add_done_callback(lambda _: self.notify())
        return None

    def summary(self):
        return {"input_capacity": 1, "crop_capacity": 1,
                "high_water_pages": self.high_water, "counts": dict(self.counts),
                "service_wall_s": dict(self.service_s),
                "worker_threads_observed": {k: len(v) for k, v in self.threads.items()}}

    def close(self):
        self.input_worker.shutdown(wait=True, cancel_futures=True)
        self.crop_worker.shutdown(wait=True, cancel_futures=True)
        self.input_future = self.crop_future = None
