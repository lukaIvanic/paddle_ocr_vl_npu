"""Observed owner-wall accounting; overlapping CPU service is kept separate."""
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
import threading
import time


def distribution(values):
    values = sorted(values)
    def percentile(q):
        index = (len(values) - 1) * q
        lo = int(index)
        return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (index - lo)
    return dict(count=len(values), total_s=sum(values), mean_s=sum(values)/len(values),
                p50_s=percentile(.5), p99_s=percentile(.99), max_s=values[-1])


class PipelineTiming:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.owner = threading.get_ident()
        self.local = threading.local()
        self.trace = None
        self.step_samples = defaultdict(list)
        if enabled:
            from utils.timeline import TimelineRecorder
            self.trace = TimelineRecorder()

    @contextmanager
    def scope(self, label, *, flow_id=None, args=None):
        if not self.enabled:
            yield
            return
        stack = getattr(self.local, "stack", None)
        if stack is None:
            self.local.stack = stack = []
        frame = [time.perf_counter_ns(), 0]
        stack.append(frame)
        try:
            yield
        finally:
            end = time.perf_counter_ns()
            stack.pop()
            duration = end - frame[0]
            if stack:
                stack[-1][1] += duration
            self.trace.record_span(
                "Hybrid owner" if threading.get_ident() == self.owner else "Hybrid worker",
                label, frame[0], end, flow_id=flow_id, event_type="scope",
                args={**(args or {}), "exclusive_ns": duration - frame[1]},
            )

    def instrument(self, obj, method, label):
        if not self.enabled:
            return
        original = getattr(obj, method)
        @wraps(original)
        def measured(*args, **kwargs):
            with self.scope(label):
                return original(*args, **kwargs)
        setattr(obj, method, measured)

    def submit(self, executor, function, *args, stage, flow_id):
        if not self.enabled:
            return executor.submit(function, *args)
        job = dict(stage=stage, flow_id=flow_id, queued=time.perf_counter_ns())
        def work():
            start = time.perf_counter_ns()
            self.trace.record_span("CPU queue", stage, job["queued"], start,
                                   flow_id=flow_id, track="queue", lane=stage)
            try:
                return function(*args)
            finally:
                job["finished"] = end = time.perf_counter_ns()
                self.trace.record_span("CPU service", stage, start, end, flow_id=flow_id)
        future = executor.submit(work)
        future.hybrid_timing_job = job
        return future

    def consume(self, future):
        if self.enabled:
            job = future.hybrid_timing_job
            self.trace.record_span("CPU ready residence", job["stage"], job["finished"],
                                   time.perf_counter_ns(), flow_id=job["flow_id"],
                                   track="queue", lane=job["stage"])

    def summary(self):
        if not self.enabled:
            return {"enabled": False}
        inclusive, exclusive, workers, queues, ready, waits, stages, delivery = (defaultdict(list) for _ in range(8))
        for event in self.trace.events():
            name, row = event["name"], event["row"]
            seconds = event["duration_ns"] / 1e9
            if row == "Hybrid owner":
                inclusive[name].append(seconds)
                exclusive[name].append(event["args"]["exclusive_ns"] / 1e9)
                if name == "shared.wait":
                    key = " + ".join(event["args"]["blockers"])
                    waits[key].append(seconds)
            elif row == "CPU service": workers[name].append(seconds)
            elif row == "CPU queue": queues[name].append(seconds)
            elif row == "CPU ready residence": ready[name].append(seconds)
            elif row == "Hybrid worker": stages[name].append(seconds)
            elif row == "CPU result delivery": delivery[name].append(seconds)
        measured = sum(inclusive.get("pipeline", []))
        partition = sum(sum(values) for values in exclusive.values())
        if abs(partition - measured) > 1e-6 or any(value < 0 for values in exclusive.values() for value in values):
            raise RuntimeError("Owner timing scopes do not form a nonnegative pipeline partition")
        return {
            "enabled": True,
            "semantics": {
                "owner_exclusive": "Non-overlapping host elapsed intervals; includes NPU execution/waits within each scope, not CPU-active time.",
                "owner_inclusive": "Nested scopes; never add these totals together.",
                "cpu_service": "Overlapping worker elapsed, not additive to pipeline wall or proof of hidden CPU work.",
                "wait_blocker_sets": "Prerequisites observed unresolved at wait entry; combined sets count each wait once. Not per-worker causal savings; includes wake/scheduling latency.",
                "device_events": "Existing event envelopes may include transfers and host submission gaps; not kernel-active time.",
            },
            "pipeline_scope_s": measured,
            "owner_partition_sum_s": partition,
            "owner_partition_error_s": partition - measured,
            "owner_inclusive": {k: distribution(v) for k,v in inclusive.items()},
            "owner_exclusive": {k: distribution(v) for k,v in exclusive.items()},
            "wait_blocker_sets": {k: distribution(v) for k,v in waits.items()},
            "exposed_cpu_dependency_wait_s": sum(sum(v) for k,v in waits.items()
                                                  if k not in ("ready_before_wait", "awaiting_input", "unclassified")),
            "cpu_service": {k: distribution(v) for k,v in workers.items()},
            "cpu_queue_residence": {k: distribution(v) for k,v in queues.items()},
            "cpu_ready_residence": {k: distribution(v) for k,v in ready.items()},
            "overlapping_worker_scopes": {k: distribution(v) for k,v in stages.items()},
            "cpu_result_delivery": {k: distribution(v) for k,v in delivery.items()},
            "unirec_step_diagnostics": {k: distribution(v) for k,v in self.step_samples.items()},
            "unirec_step_diagnostics_note": "Existing host timers, nested/non-additive: decode_step contains submission/token wait; scheduler contains retirement/admission/completion callbacks. These are not device-only or exclusive CPU work.",
        }

    def unirec_step(self, record):
        if self.enabled:
            for key in ("input_build_s", "graph_submit_s", "token_select_d2h_wait_s",
                        "decode_step_s", "scheduler_s", "iteration_wall_s"):
                self.step_samples[key].append(record[key])


NO_TIMING = PipelineTiming(enabled=False)


def timed_method(label):
    def decorate(function):
        @wraps(function)
        def measured(self, *args, **kwargs):
            with self.timing.scope(label):
                return function(self, *args, **kwargs)
        return measured
    return decorate


def install(timing, adapters, source):
    source.preparation.timing = timing
    source.frontend.timeline = timing.trace
    for method, label in (("detect", "layout.detect_transfer_and_fence"),):
        # LayoutPreparation stores bound callables, so instrument that callable.
        timing.instrument(source.preparation, method, label)
    timing.instrument(source, "advance", "layout.advance_or_publish")
    timing.instrument(source, "finish", "output.page_assembly_and_emit")
    timing.instrument(source, "emit_page", "output.page_files")
    timing.instrument(source, "emit_trace", "output.crop_trace")
    for name, adapter in adapters.items():
        adapter.timing = adapter.cpu.timing = timing
        adapter.timing_model = name
        # complete() is decorated in the adapter: the iterator already captured
        # that bound method before this timing object was installed.
        timing.instrument(adapter, "emit", f"{name}.publish_completion")
        if name == "unirec":
            timing.instrument(adapter.vision, "encode", "unirec.vision_prefill_envelope")
            if getattr(adapter, "streamed", False):
                timing.instrument(adapter, "_vision_job", "unirec.streamed_vision_worker")
                timing.instrument(adapter, "_text_job", "unirec.streamed_text_worker")
                timing.instrument(adapter, "_decode_job", "unirec.streamed_decode_worker")
            timing.instrument(adapter.runner, "prefill_encoder_hidden_states_packed_for_cohort",
                              "unirec.text_prefill_envelope")
        elif name == "paddle":
            r = adapter.recognizer
            r.timeline = timing.trace
            r.decode_scheduler.timeline = timing.trace
            for method, label in (("_stage_prefill_group", "paddle.prefill_stage_inputs"),
                                  ("_enqueue_staged_prefill_group", "paddle.prefill_enqueue"),
                                  ("_finalize_prefill_group", "paddle.prefill_finalize"),
                                  ("_result_from_completion", "paddle.result_conversion")):
                timing.instrument(r, method, label)
