"""Single default observer: host spans + deferred events, never timing barriers."""
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import faulthandler
import json
import threading
import time
import torch


class Journal:
    def __init__(self, root, profile=False):
        self.root, self.profile = root, profile
        self.start = time.monotonic()
        self.events = (root/'events.jsonl').open('w', buffering=65536)
        self.items = (root/'items.jsonl').open('w', buffering=65536)
        self.lock = threading.Lock()
        self.active = None
        self.progress = {}
        self.pending = []
        self.write_s = 0.0
        self.done = threading.Event()
        self.worker = threading.Thread(target=self.heartbeat, daemon=True)
        self.worker.start()

    def emit(self, phase, **data):
        row = dict(phase=phase, utc=datetime.now(timezone.utc).isoformat(),
                   run_elapsed_s=time.monotonic()-self.start, **data)
        begin = time.perf_counter()
        with self.lock:
            self.events.write(json.dumps(row)+'\n')
            if phase not in ('section_start','section_finish'):
                print('HR_EVAL '+json.dumps(row), flush=True)
                self.events.flush()
        self.write_s += time.perf_counter()-begin

    def heartbeat(self):
        last_dump = 0.0
        while not self.done.wait(5):
            active = self.active
            age = time.monotonic()-active['start'] if active else None
            self.emit('heartbeat', active=active, active_elapsed_s=age, progress=self.progress)
            if age is not None and age > 120 and time.monotonic()-last_dump > 120:
                faulthandler.dump_traceback()
                last_dump = time.monotonic()

    @contextmanager
    def section(self, record, name, tokens=0, device=False, route=None):
        if name in record['sections']:
            raise ValueError(f'Duplicate section: {name}')
        self.active = dict(kind=record['kind'],id=record['id'],section=name,start=time.monotonic())
        self.emit('section_start',kind=record['kind'],id=record['id'],section=name,tokens=tokens,route=route)
        begin = time.perf_counter()
        a = b = None
        if device:
            a, b = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            a.record()
        context = torch.profiler.record_function('colqwen.'+name) if self.profile else nullcontext()
        try:
            with context:
                yield
        finally:
            if b is not None:
                b.record()
            stats = dict(host_s=time.perf_counter()-begin,tokens=tokens)
            if route:
                stats['route'] = route
            if a is not None:
                stats['device_status'] = 'pending'
                self.pending.append((a,b,stats))
            record['sections'][name] = stats
            self.emit('section_finish',kind=record['kind'],id=record['id'],section=name,**stats)
            self.active = None

    def resolve(self):
        """Called after natural materialization; never waits for incomplete events."""
        pending = []
        for a,b,stats in self.pending:
            if b.query():
                ms = float(a.elapsed_time(b))
                stats.update(device_interval_ms=ms,device_status='complete',
                             device_tok_s=stats['tokens']*1000/ms if ms and stats['tokens'] else None)
            else:
                pending.append((a,b,stats))
        self.pending = pending

    def complete(self, row, completed, total, window):
        self.resolve()
        elapsed = time.perf_counter()-window
        self.progress = dict(kind=row['kind'],completed=completed,total=total,
            elapsed_s=elapsed,items_per_s=completed/elapsed,
            eta_s=(total-completed)*elapsed/completed)
        row['unattributed_host_s'] = row['wall_s']-sum(s['host_s'] for s in row['sections'].values())
        self.items.write(json.dumps(row)+'\n')
        self.items.flush()
        # All item fields (including section timing and token counts) are immediate.
        self.emit('item_finish',item=row,progress=self.progress)
        (self.root/'progress.json').write_text(json.dumps(self.progress)+'\n')

    def close(self):
        self.done.set(); self.worker.join(timeout=2)
        self.resolve()
        self.emit('observer_finish',pending_device_events=len(self.pending),write_s=self.write_s)
        self.events.close(); self.items.close()


class PipelineProfiler:
    """Capture actual query/page/scoring items, never replay a subsection."""
    def __init__(self, root, queries, pages, enabled):
        self.capture = None
        self.root = root
        if enabled:
            import torch_npu.profiler as p
            # One warmup and one captured real item at each workflow boundary.
            starts = {0,queries,queries+pages}
            def schedule(step):
                if step in starts:
                    return p.ProfilerAction.WARMUP
                if step-1 in starts:
                    return p.ProfilerAction.RECORD_AND_SAVE
                return p.ProfilerAction.NONE
            self.capture = p.profile(activities=[p.ProfilerActivity.CPU,p.ProfilerActivity.NPU],
                schedule=schedule, record_shapes=True, with_stack=False, profile_memory=False,
                on_trace_ready=p.tensorboard_trace_handler(str(root/'profiler'),analyse_flag=True))
            self.capture.start()

    def step(self):
        if self.capture:
            self.capture.step()

    def close(self):
        if self.capture:
            self.capture.stop()
            self.capture = None
