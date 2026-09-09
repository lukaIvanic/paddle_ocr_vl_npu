"""Persistent stage threads, joined at the shared NPU ownership boundary."""
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import time


class StreamedExecution:
    """One in-flight operation per stage, with early producer publication.

    The owner alone mutates scheduling state. Workers publish messages; they
    never wait on downstream queue capacity or touch page output bookkeeping.
    stop() stops NEW submissions, not running work. All futures are joined
    before the caller's device fence and cross-model handoff.
    """
    def __init__(self, stages=("vision", "text", "decode")):
        self.pools = {s: ThreadPoolExecutor(max_workers=1, thread_name_prefix="unirec-" + s)
                      for s in stages}
        self.events = Queue()
        self.running = {}
        self.spans = {s: [] for s in stages}

    def publish(self, stage, value):
        self.events.put(("publish", stage, value))

    def submit(self, stage, function, *args):
        if stage in self.running:
            raise RuntimeError("Stage already running: " + stage)
        def work():
            start = time.perf_counter_ns()
            try:
                return function(*args)
            finally:
                self.spans[stage].append((start, time.perf_counter_ns()))
        future = self.pools[stage].submit(work)
        self.running[stage] = future
        future.add_done_callback(lambda f: self.events.put(("finish", stage, f)))

    def receive(self, *, block=True):
        kind, stage, value = self.events.get(block=block)
        if kind == "finish":
            del self.running[stage]
            value = value.result()  # Fail directly; no retry or fallback.
        return kind, stage, value

    def close(self):
        for pool in self.pools.values():
            pool.shutdown(wait=True, cancel_futures=True)

    def summary(self):
        from hybrid_timing import distribution
        events = []
        for stage, spans in self.spans.items():
            for start, end in spans:
                events.extend(((start, 1), (end, -1)))
        active = 0
        last = None
        union = overlap = 0
        for instant, delta in sorted(events):
            if last is not None:
                union += (instant-last) if active else 0
                overlap += (instant-last) if active > 1 else 0
            active += delta
            last = instant
        return dict(stage_host_envelopes={k: distribution([(b-a)/1e9 for a,b in v]) if v else None
                                         for k,v in self.spans.items()},
                    stage_host_union_s=union/1e9, multi_stage_host_overlap_s=overlap/1e9,
                    timing_basis="Overlapping worker submit-through-return envelopes, NOT kernel-active time; never add to owner wall")
