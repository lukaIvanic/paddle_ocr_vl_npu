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

    if os.environ.get("TABLE_SERVING_PROFILE_COMPONENT") == "embedding":
        from paddleocr_vl.model.vision_prefill import PaddleOCRVisionEmbeddings
        original_forward = PaddleOCRVisionEmbeddings.forward
        calls = 0
        observations = []
        destination = Path(os.environ["TABLE_SERVING_PROFILE_ROOT"]).resolve()
        destination.mkdir(parents=True, exist_ok=False)

        def forward(module, pixels, image_grid_thw):
            nonlocal calls
            index = calls
            calls += 1
            if index == 0 and module.linear_patch_projection:
                patches = pixels.reshape(-1, *pixels.shape[-3:])
                module.linear_patch_projection = False
                try:
                    reference = module.project_patches(patches).detach().float().cpu()
                finally:
                    module.linear_patch_projection = True
                candidate = module.project_patches(patches).detach().float().cpu()
                delta = candidate - reference
                comparison = {
                    "scope": "same real input, original Conv2D versus linear patch projection",
                    "shape": list(reference.shape),
                    "all_finite": bool(torch.isfinite(candidate).all()),
                    "max_abs": float(delta.abs().max()),
                    "mean_abs": float(delta.abs().mean()),
                    "different_fraction": float((delta != 0).float().mean()),
                    "relative_l2": float(delta.norm()/reference.norm().clamp_min(1e-12)),
                }
                (destination / "patch_comparison.json").write_text(json.dumps(comparison, indent=2)+"\n")
                print("SERVING_PROFILE patch comparison " + json.dumps(comparison), flush=True)
            if index < 5 or index >= 8:
                return original_forward(module, pixels, image_grid_thw)
            # Five complete real requests precede three separately captured
            # forwards. Stop at each forward so inter-request decoder work does
            # not inflate the capture. Profiler stop synchronizes: diagnostic only.
            capture_dir = destination / f"embedding_{index-5}"
            with prof.profile(
                activities=[prof.ProfilerActivity.CPU, prof.ProfilerActivity.NPU],
                on_trace_ready=prof.tensorboard_trace_handler(str(capture_dir), analyse_flag=True),
                record_shapes=True, profile_memory=False, with_stack=False,
                experimental_config=prof._ExperimentalConfig(
                    profiler_level=prof.ProfilerLevel.Level1,
                    aic_metrics=prof.AiCMetrics.PipeUtilization,
                    export_type=prof.ExportType.Text, data_simplification=False,
                ),
            ):
                with torch.profiler.record_function("serving.vision_embeddings"):
                    result = original_forward(module, pixels, image_grid_thw)
            observations.append({"request_index": index, "pixel_shape": list(pixels.shape),
                                 "grid": image_grid_thw.tolist(), "host_epoch_s": time.time()})
            (destination / "capture.json").write_text(json.dumps({
                "component": "real vision patch and position embeddings",
                "full_request_warmups": 5, "profiled_forwards": len(observations),
                "configuration": config, "observations": observations,
                "profiled_latency_is_not_a_goal_result": True,
            }, indent=2)+"\n")
            print(f"SERVING_PROFILE embedding {index-4}/3 done", flush=True)
            return result

        PaddleOCRVisionEmbeddings.forward = forward
        try:
            return original_worker(jobs, results, config)
        finally:
            PaddleOCRVisionEmbeddings.forward = original_forward

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
