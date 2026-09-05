"""DIAGNOSTIC ONLY: saved-length admission oracle around the real crop API.

No model math/output changes. Never use this adapter for acceptance gates.
ORACLE_ADMISSION_RESULTS is a completed client results.jsonl label source.
ORACLE_ADMISSION_ENABLE=0 records the same diagnostic configuration, ungated.
The only schedule change is deferring an already CPU-prepared head request.
"""
import hashlib
from collections import deque
import json
import os
from pathlib import Path
import re
import sys
import time
import statistics

REPO = next(p for p in Path(__file__).resolve().parents if (p / "CLAUDE.md").is_file())
sys.path.insert(0, str(REPO / "09_persistent_page_engine/scripts"))
sys.path.insert(0, str(REPO / "09_persistent_page_engine"))

DEADLINE_S = 2.0
STEP_S = 0.00135  # Approximate warm B4 cadence; not a future measured latency.
PREFILL_S = 0.1  # Conservative cost of one incoming request's prefill.
SHORT_TOKENS = 256


class DecodeCadence:
    """Past real iteration cadence, excluding prefill and idle intervals."""
    def __init__(self):
        self.samples = deque(maxlen=64)
        self.last = None
        self.interrupted = True
        self.step_s = STEP_S
        self.count = 0

    def step(self, now):
        if self.last is not None and not self.interrupted:
            self.samples.append(now-self.last)
            self.count += 1
            if len(self.samples) >= 16 and self.count % 16 == 0:
                self.step_s = statistics.median(self.samples)
        self.last = now
        self.interrupted = False


def protection(new_tokens, new_age, active, step_s=STEP_S):
    """Return threatened running IDs; active entries are (id, age, remaining)."""
    if new_tokens is None or new_tokens >= SHORT_TOKENS:
        return []
    slack = DEADLINE_S - new_age - PREFILL_S - new_tokens * step_s
    return [key for key, age, remaining in active
            if remaining is not None and remaining > 0
            and remaining * step_s < slack
            and age + remaining * step_s < DEADLINE_S
            and age + remaining * step_s + PREFILL_S >= DEADLINE_S]


def install():
    import serve_crop_ocr_api as api
    original_worker = api._worker_main

    def worker(jobs, results, config):
        from paddleocr_vl.serving.engine import ContinuousRecognizer, _OpenPrefillSource
        from paddleocr_vl.serving.continuous_decode import DecodeArena
        if not config["request_scheduling_metrics"] or config["max_prefill_interruptions"] is not None:
            raise ValueError("oracle diagnostic requires scheduling metrics and no interruption cap")
        label_path = Path(os.environ["ORACLE_ADMISSION_RESULTS"]).resolve()
        raw = label_path.read_bytes()
        lengths = {}
        for record in map(json.loads, raw.splitlines()):
            response = record["service_result"]["response"]
            if record["status"] != "ok" or response["stop_reason"] != "eos":
                raise ValueError("oracle source must contain complete native streams")
            lengths[record["request_id"]] = len(response["token_ids"])
        enabled = os.environ.get("ORACLE_ADMISSION_ENABLE", "1") == "1"
        cadence = DecodeCadence()
        original_step = DecodeArena.step

        def step(self, *args, **kwargs):
            cadence.step(time.perf_counter())
            return original_step(self, *args, **kwargs)

        DecodeArena.step = step
        original_configuration = ContinuousRecognizer.configuration

        def configuration(self):
            value = original_configuration(self)
            value["diagnostic_admission_oracle"] = {
                "non_qualifying": True, "enabled": enabled,
                "source": str(label_path), "sha256": hashlib.sha256(raw).hexdigest(),
                "deadline_s": DEADLINE_S, "step_s": STEP_S,
                "prefill_s": PREFILL_S, "short_tokens": SHORT_TOKENS,
                "cadence": "median_last64_uninterrupted_intervals_update_every16",
            }
            return value

        def estimate(request_id):
            # Explicit benchmark-ID lookup: forbidden outside this diagnostic.
            match = re.search(r"page_[^:]+", request_id)
            return lengths.get(match[0]) if match else None

        original_pull = _OpenPrefillSource.pull
        blocked = {}
        logged = set()

        def pull(self, *, block):
            # Start/finish CPU work independently; never stall active decode on it.
            self._submit_available(block_for_first=block and not self.pending)
            if self.pending and self.pending[0][1].done():
                key, future = self.pending[0]
                if not future.cancelled() and future.exception() is None:
                    now = time.perf_counter()
                    metrics = self.scheduling_metrics.requests
                    active = []
                    for state in self.recognizer.decode_scheduler.arena.slots:
                        if state is None or state.first_decode_launched_at is None:
                            continue
                        running = state.ready.request_id
                        total = estimate(running)
                        active.append((running, now - metrics[running].started_at,
                                       None if total is None else total - len(state.token_ids)))
                    protected = protection(estimate(key), now - metrics[key].started_at, active, cadence.step_s)
                    if key not in logged:
                        logged.add(key)
                        print("ORACLE_DECISION " + json.dumps({"request_id": key, "new_tokens": estimate(key), "new_age_s": now-metrics[key].started_at, "active": active, "step_s": cadence.step_s, "protected": protected, "enabled": enabled}), flush=True)
                    if enabled and protected:
                        if key not in blocked:
                            blocked[key] = now
                            print("ORACLE_DEFER " + json.dumps({"request_id": key, "protected": protected, "age_s": now-metrics[key].started_at}), flush=True)
                        return None
                if key in blocked:
                    print("ORACLE_RELEASE " + json.dumps({"request_id": key, "deferred_s": time.perf_counter()-blocked.pop(key)}), flush=True)
            ready = original_pull(self, block=block)
            if ready is not None or block:
                cadence.interrupted = True
            return ready

        ContinuousRecognizer.configuration = configuration
        _OpenPrefillSource.pull = pull
        try:
            original_worker(jobs, results, config)
        finally:
            print("ORACLE_CADENCE " + json.dumps({"intervals": cadence.count, "final_step_s": cadence.step_s}), flush=True)

    # Spawn targets need a module-level callable, not an unpicklable closure.
    return api, worker


api, _wrapped_worker = install()


def oracle_worker(jobs, results, config):
    _wrapped_worker(jobs, results, config)


api._worker_main = oracle_worker
if __name__ == "__main__":
    api.main()
