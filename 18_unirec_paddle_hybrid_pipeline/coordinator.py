"""One compute owner. Full decode first, alternating ties, no timers."""
from collections import Counter
import time
from threading import Event


class Coordinator:
    def __init__(self, adapters, pages, *, decode_steps=32):
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

    def choose(self, names):
        return next((name for name in names if name != self.last_model), names[0])

    def action(self):
        live = [name for name, adapter in self.adapters.items() if not adapter.done]
        if not live:
            return None
        full = [name for name in live if self.adapters[name].occupied >= self.adapters[name].capacity]
        if full:
            return self.choose(full), "decode"
        prefills = [name for name in live if self.adapters[name].prefill_available and self.adapters[name].free > 0]
        if prefills:
            return self.choose(prefills), "prefill"
        if self.pages.has_pending:
            return None, "layout"
        # No future pages: partial decode is eligible only after that model's
        # pending crops have entered its ready queue. A finished source can
        # still own active decode slots.
        draining = [name for name in live if not self.adapters[name].pending
                    and (self.adapters[name].occupied or self.pages.exhausted)]
        return (self.choose(draining), "decode") if draining else (None, "wait")

    def run(self):
        started = time.perf_counter()
        while True:
            self.wakeup.clear()
            for adapter in self.adapters.values():
                adapter.pump_preparation(self.wakeup.set)
            action = self.action()
            if action is None:
                break
            name, phase = action
            before = time.perf_counter()
            if phase == "layout":
                self.pages.advance(self.adapters)
            elif phase == "wait":
                self.wakeup.wait()
            else:
                adapter = self.adapters[name]
                adapter.set_upstream(self.pages.has_pending or bool(adapter.pending), closed=self.pages.exhausted)
                if phase == "prefill":
                    adapter.prefill()
                else:
                    adapter.advance(self.decode_steps)
                self.last_model = name
            key = f"{name or 'shared'}.{phase}"
            self.calls[key] += 1
            self.wall_s[key] += time.perf_counter() - before
        return {"wall_s": time.perf_counter() - started, "calls": dict(self.calls), "action_wall_s": dict(self.wall_s)}
