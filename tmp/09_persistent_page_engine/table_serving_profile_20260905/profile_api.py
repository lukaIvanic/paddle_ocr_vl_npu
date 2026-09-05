"""Diagnostic-only launcher for the unchanged crop serving API.

TABLE_SERVING_PROFILE_ROOT selects an output directory. Capture starts after
32 actual B2 steps, then uses 5 profiler warmups and 20 active iterations.
No synthetic inputs, altered model calls, or manual per-step synchronization.
Profiled client timing must not be used as a performance/goal result.
"""
import json
import os
from pathlib import Path
import sys
import time

REPO = next(p for p in Path(__file__).resolve().parents if (p / "CLAUDE.md").is_file())
sys.path.insert(0, str(REPO / "09_persistent_page_engine/scripts"))
sys.path.insert(0, str(REPO / "09_persistent_page_engine"))
import serve_crop_ocr_api as api

original_worker = api._worker_main


def profiled_worker(jobs, results, config):
    import torch
    import torch_npu.profiler as prof
    from paddleocr_vl.serving.engine import ContinuousRecognizer

    original_serve = ContinuousRecognizer.serve

    def serve(self, *args, **kwargs):
        destination = Path(os.environ["TABLE_SERVING_PROFILE_ROOT"]).resolve()
        destination.mkdir(parents=True, exist_ok=False)
        arena = self.decode_arena
        original_step = arena.step
        profiler = None
        eligible = captured = 0
        observations = []

        def step(*step_args, **step_kwargs):
            nonlocal profiler, eligible, captured
            if profiler is None:
                eligible += int(arena.num_active == 2)
                if eligible <= 32:
                    return original_step(*step_args, **step_kwargs)
                profiler = prof.profile(
                    activities=[prof.ProfilerActivity.CPU, prof.ProfilerActivity.NPU],
                    schedule=prof.schedule(wait=0, warmup=5, active=20, repeat=1),
                    on_trace_ready=prof.tensorboard_trace_handler(str(destination), analyse_flag=True),
                    record_shapes=True, profile_memory=False, with_stack=False,
                    experimental_config=prof._ExperimentalConfig(
                        profiler_level=prof.ProfilerLevel.Level1,
                        aic_metrics=prof.AiCMetrics.PipeUtilization,
                        export_type=prof.ExportType.Text, data_simplification=False,
                    ),
                )
                profiler.start()
                print("SERVING_PROFILE start: actual B2; warmup=5 active=20", flush=True)
            observations.append({
                "index": captured, "profiler_warmup": captured < 5,
                "iteration": step_kwargs.get("iteration"), "host_epoch_s": time.time(),
                "active_slots": arena.num_active,
                "positions": [None if slot is None else slot.ready.prompt_length + slot.iterations_launched
                              for slot in arena.slots],
                "request_ids": [None if slot is None else slot.ready.request_id for slot in arena.slots],
            })
            with torch.profiler.record_function("serving.decode_step"):
                result = original_step(*step_args, **step_kwargs)
            profiler.step()
            captured += 1
            if captured == 25:
                profiler.stop()
                arena.step = original_step
                (destination / "capture.json").write_text(json.dumps({
                    "actual_b2_steps_before_profiler": 32, "warmups": 5, "repeats": 20,
                    "configuration": config, "observations": observations,
                    "profiled_latency_is_not_a_goal_result": True,
                }, indent=2) + "\n")
                print("SERVING_PROFILE done: restored ordinary loop", flush=True)
            return result

        arena.step = step
        try:
            return original_serve(self, *args, **kwargs)
        finally:
            arena.step = original_step
            if profiler is not None and captured < 25:
                profiler.stop()

    ContinuousRecognizer.serve = serve
    original_worker(jobs, results, config)


api._worker_main = profiled_worker
if __name__ == "__main__":
    api.main()
