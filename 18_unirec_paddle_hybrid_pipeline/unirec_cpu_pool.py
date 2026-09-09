"""UniRec-style persistent CPU processes with a resize thread pool per process.

Paddle still owns layout/crop geometry. Only already-routed RGB crop resizing
crosses this boundary; model objects and NPU tensors never enter a CPU worker.
"""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, Future
import multiprocessing as mp
import os
import threading
import time

from cpu_preparation import CpuPreparation


def _initialize(threads):
    global _resize_threads
    _resize_threads = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="unirec-resize")


def _resize(job):
    import numpy as np
    from PIL import Image
    key, crop, size = job
    start = time.perf_counter_ns()
    image = crop.convert("RGB")
    value = (np.ascontiguousarray(np.asarray(image.resize(size, Image.Resampling.BICUBIC))), image.size)
    return key, value, start, time.perf_counter_ns(), (os.getpid(), threading.get_ident())


def _prepare(jobs):
    return list(_resize_threads.map(_resize, jobs))


class UniRecCpuPool(CpuPreparation):
    def __init__(self, processor, capacity=128, workers=4, threads=8):
        from hybrid_timing import NO_TIMING
        self.processor, self.capacity = processor, capacity
        self.workers, self.threads = workers, threads
        self.executor = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                            initializer=_initialize, initargs=(threads,))
        self.futures = {}
        self.submitted = self.consumed = self.high_water = 0
        self.service_s = 0.0
        self.worker_thread_ids = set()
        self.notify = lambda: None
        self.timing = NO_TIMING
        self.name = "hybrid-unirec-cpu"
        self.batches = []

    def pump(self, pages, notify):
        self.notify = notify
        for future in self.futures.values():
            if future.done() and future.exception() is not None:
                future.result()
        for page in pages:
            jobs = []
            for r in page:
                if r.request_id in self.futures:
                    continue
                if len(self.futures) >= self.capacity:
                    break
                self.futures[r.request_id] = Future()
                size = self.processor.get_processed_size(*r.crop.size)
                jobs.append((r.request_id, r.crop, size))
                self.submitted += 1
            if jobs:
                queued = time.perf_counter_ns()
                proxies = {key: self.futures[key] for key, _, _ in jobs}
                batch = self.executor.submit(_prepare, jobs)
                self.batches.append(batch)
                def finished(future, proxies=proxies, queued=queued):
                    try:
                        for key, value, start, end, identity in future.result():
                            if self.timing.enabled:
                                self.timing.trace.record_span("CPU queue", self.name, queued, start,
                                                              flow_id=key, track="queue", lane=self.name)
                                self.timing.trace.record_span("CPU service", self.name, start, end, flow_id=key)
                            proxies[key].set_result((value, (end-start)/1e9, identity, end))
                    except BaseException as exc:
                        for proxy in proxies.values():
                            if not proxy.done():
                                proxy.set_exception(exc)
                    finally:
                        self.notify()
                batch.add_done_callback(finished)
            if len(self.futures) >= self.capacity:
                break
        self.high_water = max(self.high_water, len(self.futures))
        self.batches = [f for f in self.batches if not f.done()]

    def take(self, requests):
        if not self.ready(requests):
            raise RuntimeError("CPU preparation consumed before completion")
        values = []
        for r in requests:
            value, seconds, identity, end = self.futures.pop(r.request_id).result()
            if self.timing.enabled:
                self.timing.trace.record_span("CPU ready residence", self.name, end, time.perf_counter_ns(),
                                              flow_id=r.request_id, track="queue", lane=self.name)
            self.service_s += seconds
            self.worker_thread_ids.add(identity)
            self.consumed += 1
            values.append(value)
        return values

    def summary(self):
        return {**super().summary(), "configured_processes": self.workers,
                "configured_threads_per_process": self.threads,
                "worker_processes_observed": len({pid for pid, _ in self.worker_thread_ids})}
