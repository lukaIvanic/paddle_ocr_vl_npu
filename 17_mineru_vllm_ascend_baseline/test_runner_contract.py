#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("run_omnidocbench.py")
SPEC = importlib.util.spec_from_file_location("experiment17_runner", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


class RunnerContractTest(unittest.TestCase):
    def test_atomic_json_write_supports_maximum_length_final_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / ("x" * 250 + ".json")
            RUNNER.write_json(path, {"status": "ok"})
            self.assertEqual(json.loads(path.read_text()), {"status": "ok"})
            self.assertEqual(list(path.parent.glob(".tmp-*")), [])

    def test_compiled_async_preset_matches_reference(self) -> None:
        spec = RUNNER.preset_spec(RUNNER.MODE_COMPILED_ASYNC)
        self.assertEqual(spec["engine"], "AsyncLLM")
        self.assertEqual(spec["max_num_seqs"], 512)
        self.assertEqual(spec["max_num_batched_tokens"], 16384)
        self.assertTrue(spec["enable_prefix_caching"])
        self.assertTrue(spec["enable_chunked_prefill"])
        self.assertTrue(spec["enable_npugraph_ex"])
        self.assertTrue(spec["enable_static_kernel"])
        self.assertEqual(spec["cudagraph_capture_sizes"], RUNNER.CAPTURE_SIZES)
        self.assertEqual(
            spec["compile_cache_dir"], str(RUNNER.DEFAULT_COMPILE_CACHE_DIR)
        )
        self.assertTrue(spec["image_analysis"])

    def test_eager_sync_preset_omits_tuned_scheduler_limits(self) -> None:
        spec = RUNNER.preset_spec(RUNNER.MODE_EAGER_SYNC)
        self.assertEqual(spec["engine"], "LLM")
        self.assertTrue(spec["enforce_eager"])
        self.assertFalse(spec["enable_prefix_caching"])
        self.assertIsNone(spec["max_num_seqs"])
        self.assertIsNone(spec["max_num_batched_tokens"])

    def test_engine_kwargs_have_required_compatibility_overrides(self) -> None:
        sentinel = object()
        kwargs = RUNNER.build_engine_kwargs(
            RUNNER.MODE_COMPILED_ASYNC,
            Path("/model"),
            sentinel,
        )
        self.assertEqual(kwargs["tensor_parallel_size"], 1)
        self.assertEqual(kwargs["dtype"], "float16")
        self.assertTrue(kwargs["hf_overrides"]["tie_word_embeddings"])
        self.assertTrue(
            kwargs["hf_overrides"]["text_config"]["tie_word_embeddings"]
        )
        ascend = kwargs["additional_config"]["ascend_compilation_config"]
        self.assertFalse(ascend["fuse_norm_quant"])
        self.assertTrue(ascend["enable_npugraph_ex"])
        self.assertTrue(ascend["enable_static_kernel"])
        self.assertEqual(
            kwargs["compilation_config"]["cache_dir"],
            str(RUNNER.DEFAULT_COMPILE_CACHE_DIR),
        )

    def test_dataset_selection_is_ordered_and_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            images.mkdir()
            for name in ("a.png", "b.png", "c.png"):
                (images / name).write_bytes(b"image")
            dataset = [
                {"page_info": {"image_path": f"nested/{name}"}}
                for name in ("a.png", "b.png", "c.png")
            ]
            dataset_path = root / "dataset.json"
            dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
            pages, source = RUNNER.load_input_pages(
                dataset_json=dataset_path,
                image_list=None,
                images_dir=images,
                offset=1,
                limit=2,
            )
            self.assertEqual([page.image_name for page in pages], ["b.png", "c.png"])
            self.assertTrue(source.startswith("dataset_json:"))


if __name__ == "__main__":
    unittest.main()
