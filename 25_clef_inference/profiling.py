"""Clef observation, anchored on experiment 21: deferred events, no stage barriers."""
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from functools import wraps
import json
import threading
import time


class Journal:
    def __init__(self, root, profile=False):
        self.root, self.profile = root, profile
        self.events = (root / 'events.jsonl').open('w', buffering=65536)
        self.items = (root / 'items.jsonl').open('w')
        self.start = time.perf_counter()
        self.current = None
        self.stack = []
        self.pending = []
        self.pool = []
        self.rows = []
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.worker = threading.Thread(target=self.heartbeat, daemon=True)
        self.worker.start()

    def emit(self, event, **fields):
        value = dict(event=event, elapsed_s=time.perf_counter()-self.start, **fields)
        with self.lock:
            self.events.write(json.dumps(value)+'\n')
            if event not in ('section', 'device_resolution'):
                self.events.flush()
                print('CLEF_BENCH '+json.dumps(value), flush=True)

    def heartbeat(self):
        while not self.done.wait(5):
            row = self.current
            self.emit('heartbeat', completed=len(self.rows),
                      active=None if row is None else {'phase': row['phase'], 'index': row['index']},
                      section=self.stack[-1] if self.stack else None)

    @contextmanager
    def section(self, name, device=False):
        if self.current is None:
            yield
            return
        import torch
        row = self.current
        depth = len(self.stack)
        self.stack.append(name)
        events = None
        if device:
            events = self.pool.pop() if self.pool else (
                torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True))
            events[0].record()
        began = time.perf_counter()
        context = torch.profiler.record_function('clef/'+name) if self.profile else nullcontext()
        try:
            with context:
                yield
        finally:
            if events:
                events[1].record()
            stats = {'name': name, 'depth': depth, 'host_s': time.perf_counter()-began}
            if events:
                stats['device_status'] = 'pending'
                self.pending.append((events, stats, row['phase'], row['index']))
            row['sections'].append(stats)
            self.stack.pop()
            self.emit('section', phase=row['phase'], index=row['index'], **stats)

    def resolve(self):
        pending = []
        for events, stats, phase, index in self.pending:
            if events[1].query():
                stats.update(device_status='complete', device_interval_ms=float(events[0].elapsed_time(events[1])))
                self.emit('device_resolution', phase=phase, index=index, **stats)
                self.pool.append(events)
            else:
                pending.append((events, stats, phase, index))
        self.pending = pending

    @contextmanager
    def item(self, row):
        if self.current is not None:
            raise RuntimeError('Items cannot nest')
        row['sections'] = []
        self.current = row
        began = time.perf_counter()
        self.emit('item_start', phase=row['phase'], index=row['index'])
        try:
            yield row
        finally:
            row['wall_s'] = time.perf_counter()-began
            self.resolve()  # Never synchronizes; caller naturally materializes its output.
            row['unattributed_host_s'] = row['wall_s']-sum(s['host_s'] for s in row['sections'] if s['depth']==0)
            self.rows.append(row)
            self.items.write(json.dumps(row)+'\n')
            self.items.flush()
            self.emit('item_finish', item=row)
            self.current = None

    def close(self):
        self.done.set()
        self.worker.join()
        self.resolve()
        self.emit('observer_finish', pending_device_events=len(self.pending))
        self.events.close()
        self.items.close()


@contextmanager
def observe_model(model, journal):
    """External wrappers call original methods unchanged and restore on exit."""
    import torch
    import local_modeling_clef as modeling
    restored = []

    def install(obj, attribute, label, device=False, trace_only=False):
        original = getattr(obj, attribute)
        had_own = attribute in vars(obj)
        @wraps(original)
        def wrapped(*args, **kwargs):
            context = torch.profiler.record_function('clef/'+label) if trace_only else journal.section(label, device=device)
            with context:
                return original(*args, **kwargs)
        setattr(obj, attribute, wrapped)
        restored.append((obj, attribute, original, had_own))

    try:
        install(model.backbone, 'forward', 'backbone', device=True)
        install(model.head, 'forward', 'decision_head', device=True)
        if journal.profile:
            for name, module in model.named_modules():
                if isinstance(module, (modeling.GatedDeltaNet, modeling.FullAttention, modeling.MLP,
                                       modeling.EvidenceRoutingLayer)):
                    install(module, 'forward', name, trace_only=True)
            install(modeling, 'chunk_gated_delta_rule', 'gdn_scan', trace_only=True)
        yield
    finally:
        for obj, attribute, original, had_own in reversed(restored):
            if had_own:
                setattr(obj, attribute, original)
            else:
                delattr(obj, attribute)


class PipelineProfiler:
    """Observe one actual item per phase; preceding item warms the profiler."""
    def __init__(self, root, items_per_phase, phases, capture_index, enabled):
        self.capture = None
        if enabled:
            import torch_npu.profiler as p
            starts = {phase*items_per_phase+capture_index for phase in range(phases)}
            def schedule(step):
                if step in starts:
                    return p.ProfilerAction.RECORD_AND_SAVE
                if step+1 in starts:
                    return p.ProfilerAction.WARMUP
                return p.ProfilerAction.NONE
            self.capture = p.profile(activities=[p.ProfilerActivity.CPU,p.ProfilerActivity.NPU],
                schedule=schedule, record_shapes=True, with_stack=False, profile_memory=False,
                on_trace_ready=p.tensorboard_trace_handler(str(root/'profiler'), analyse_flag=True))
            self.capture.start()

    def step(self):
        if self.capture:
            self.capture.step()

    def close(self):
        if self.capture:
            self.capture.stop()
            self.capture = None


def distribution(values):
    ordered = sorted(values)
    if not ordered:
        return {'count': 0}
    def quantile(fraction):
        position = (len(ordered)-1)*fraction
        lo = int(position)
        hi = min(lo+1, len(ordered)-1)
        return ordered[lo]+(ordered[hi]-ordered[lo])*(position-lo)
    return {'count': len(ordered), 'sum': sum(ordered), 'mean': sum(ordered)/len(ordered),
            'p50': quantile(.5), 'p90': quantile(.9), 'max': ordered[-1]}


def summarize(rows):
    phases, sections = defaultdict(list), defaultdict(list)
    for row in rows:
        phases[row['phase']].append(row)
        for section in row['sections']:
            sections[(row['phase'], section['name'])].append(section)
    return {'phases': {name: {'wall_s': distribution([r['wall_s'] for r in items]),
                'subsequent_items_wall_s': distribution([r['wall_s'] for r in items if not r['first_use']]),
                'unattributed_host_s': distribution([r['unattributed_host_s'] for r in items])}
                for name, items in phases.items()},
            'sections': [{'phase': phase, 'name': name,
                'host_s': distribution([s['host_s'] for s in items]),
                'device_interval_ms': distribution([s['device_interval_ms'] for s in items if 'device_interval_ms' in s])}
                for (phase, name), items in sections.items()]}

