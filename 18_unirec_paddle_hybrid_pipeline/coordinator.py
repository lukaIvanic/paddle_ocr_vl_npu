"""One compute owner. Full decode first, alternating ties, no timers."""
from collections import Counter
import time
from threading import Event
from hybrid_timing import NO_TIMING


class Coordinator:
    def __init__(self, adapters, pages, *, decode_steps=32, timing=NO_TIMING):
        if decode_steps < 1:
            raise ValueError("decode_steps must be positive")
        self.adapters = adapters
        self.pages = pages
        self.decode_steps = decode_steps
        self.last_model = None
        self.calls = Counter()
        self.wall_s = Counter()
        self.wakeup = Event()
        self.pages.set_wakeup(self.wakeup.set)
        self.timing = timing

    def wait_snapshot(self):
        blockers, state = [], {}
        for name, adapter in self.adapters.items():
            if adapter.done:
                continue
            requests = adapter.planned_prefill_requests() if adapter.free > 0 else []
            missing = [r.request_id for r in requests
                       if r.request_id not in adapter.cpu.futures or not adapter.cpu.futures[r.request_id].done()]
            state[name] = dict(active=adapter.active, ready=adapter.ready_count,
                               pending_pages=len(adapter.pending), missing_cpu_requests=missing)
            if missing:
                blockers.append(name + ".cpu_preparation")
            elif requests or adapter.occupied >= adapter.capacity:
                blockers.append("ready_before_wait")
        prep = self.pages.preparation
        future = prep.crop_future if prep.crop_future is not None else prep.input_future
        if future is not None:
            blockers.append("ready_before_wait" if future.done() else
                            "page.crops" if prep.crop_future is not None else "page.input")
        if not blockers:
            blockers.append("awaiting_input" if not self.pages.exhausted else "unclassified")
        if "ready_before_wait" in blockers:
            blockers = ["ready_before_wait"]
        return dict(blockers=sorted(set(blockers)), state=state)

    def choose(self, names):
        return next((name for name in names if name != self.last_model), names[0])

    def action(self):
        name, phase = self._action() or (None, None)
        if name is not None and getattr(self.adapters[name], "streamed", False):
            phase = "stream"
        return (name, phase) if phase is not None else None

    def _action(self):
        live = [name for name, adapter in self.adapters.items() if not adapter.done]
        if not live:
            return None
        full = [name for name in live if self.adapters[name].occupied >= self.adapters[name].capacity]
        if full:
            return self.choose(full), "decode"
        # A ready reservoir smaller than the active batch must be admitted
        # before another prefill can run. Engine iterator boundaries return
        # control after that admission without requiring a full ready cohort.
        admit = [name for name in live
                 if getattr(self.adapters[name], "ready_capacity", self.adapters[name].capacity) < self.adapters[name].capacity
                 and self.adapters[name].ready_count >= self.adapters[name].ready_capacity]
        if admit:
            return self.choose(admit), "decode"
        prefills = [name for name in live if self.adapters[name].prefill_available and self.adapters[name].free > 0]
        if prefills:
            return self.choose(prefills), "prefill"
        if self.pages.has_pending:
            return None, "layout" if self.pages.can_advance else "wait"
        # No future pages: partial decode is eligible only after that model's
        # pending crops have entered its ready queue. A finished source can
        # still own active decode slots.
        draining = [name for name in live if not self.adapters[name].pending
                    and (self.adapters[name].occupied or self.pages.exhausted)]
        return (self.choose(draining), "decode") if draining else (None, "wait")

    def run(self):
        with self.timing.scope("pipeline"):
            return self._run()

    def _run(self):
        started = time.perf_counter()
        while True:
            with self.timing.scope("control.pump_and_choose"):
                self.wakeup.clear()
                self.pages.pump()
                for adapter in self.adapters.values():
                    if hasattr(adapter, "set_supply"):
                        adapter.set_supply(self.pages.has_pending, self.pages.exhausted)
                    adapter.pump_preparation(self.wakeup.set)
                action = self.action()
            if action is None:
                break
            name, phase = action
            before = time.perf_counter()
            key = f"{name or 'shared'}.{phase}"
            snapshot = self.wait_snapshot() if phase == "wait" and self.timing.enabled else None
            with self.timing.scope(key, args=snapshot):
                if phase == "layout":
                    self.pages.advance(self.adapters)
                elif phase == "wait":
                    self.wakeup.wait()
                else:
                    adapter = self.adapters[name]
                    adapter.set_upstream(self.pages.has_pending or bool(adapter.pending), closed=self.pages.exhausted)
                    if phase == "stream":
                        adapter.serve(self.decode_steps)
                    elif phase == "prefill":
                        adapter.prefill()
                    else:
                        adapter.advance(self.decode_steps)
                    self.last_model = name
            self.calls[key] += 1
            self.wall_s[key] += time.perf_counter() - before
        return {"wall_s": time.perf_counter() - started, "calls": dict(self.calls), "action_wall_s": dict(self.wall_s)}
