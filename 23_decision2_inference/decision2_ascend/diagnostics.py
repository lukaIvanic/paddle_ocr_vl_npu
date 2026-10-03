"""Opt-in real-serving attribution, not a replay or isolated head benchmark.

Stream-event intervals include dispatch gaps: they are NOT summed kernel busy
time. Resolve events only after the existing vLLM pooling output synchronization.
No extra device synchronization is introduced by this instrumentation.
"""
from contextlib import contextmanager, nullcontext
import json
import os
from pathlib import Path
import time

import torch

ROOT = os.environ.get('EOS_DIAGNOSTICS_DIR')
ACTIVE = False
ROW = {}
EVENTS = {}
INDEX = 0


@contextmanager
def _stage(name, stream):
    start = time.perf_counter()
    pair = None
    if stream:
        pair = [torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)]
        pair[0].record()
    with torch.profiler.record_function('decision2/' + name):
        yield
    if pair:
        pair[1].record()
        EVENTS[name] = pair
    ROW.setdefault('host_ms', {}).setdefault(name, []).append((time.perf_counter()-start)*1000)


def stage(name, stream=False):
    return _stage(name, stream) if ROOT and ACTIVE else nullcontext()


def metadata(lengths, options):
    if ROOT and ACTIVE:
        ROW.update(lengths=lengths, options=options, batch_size=len(lengths))


def install():
    if not ROOT:
        return
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    if getattr(NPUModelRunner, '_decision2_diagnostics', False):
        return
    folder = Path(ROOT)
    folder.mkdir(parents=True, exist_ok=True)
    log = (folder / f'stages_{os.getpid()}.jsonl').open('a', buffering=1)
    execute = NPUModelRunner.execute_model
    pool = NPUModelRunner._pool

    def wrapped_pool(self, *args, **kwargs):
        with stage('pool_and_output'):
            return pool(self, *args, **kwargs)

    def wrapped_execute(self, *args, **kwargs):
        global ACTIVE, ROW, EVENTS, INDEX
        INDEX += 1
        ACTIVE = True
        ROW = {'step':INDEX, 'time':time.time(), 'pid':os.getpid()}
        EVENTS = {}
        try:
            with stage('execute'):
                result = execute(self, *args, **kwargs)
            ROW['stream_elapsed_ms'] = {}
            for name,(start,end) in EVENTS.items():
                # _pool already synchronizes normal non-CUDA output delivery.
                # Never force a sync if this installed path changes.
                ROW['stream_elapsed_ms'][name] = start.elapsed_time(end) if end.query() else None
            log.write(json.dumps(ROW)+'\n')
            return result
        finally:
            ACTIVE = False

    NPUModelRunner.execute_model = wrapped_execute
    NPUModelRunner._pool = wrapped_pool
    NPUModelRunner._decision2_diagnostics = True
