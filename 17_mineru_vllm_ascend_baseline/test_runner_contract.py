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

    def test_compiled_async_preset_uses_static_kernel_off_default(self) -> None:
        spec = RUNNER.preset_spec(RUNNER.MODE_COMPILED_ASYNC)
        self.assertEqual(spec["engine"], "AsyncLLM")
        self.assertEqual(spec["max_num_seqs"], 512)
        self.assertEqual(spec["max_num_batched_tokens"], 16384)
        self.assertTrue(spec["enable_prefix_caching"])
        self.assertTrue(spec["enable_chunked_prefill"])
        self.assertTrue(spec["enable_npugraph_ex"])
        self.assertFalse(spec["enable_static_kernel"])
        self.assertEqual(spec["cudagraph_capture_sizes"], RUNNER.CAPTURE_SIZES)
        self.assertEqual(
            spec["compile_cache_dir"],
            str(RUNNER.DEFAULT_STATIC_OFF_COMPILE_CACHE_DIR),
        )
        self.assertTrue(spec["image_analysis"])

    def test_static_kernel_ab_changes_only_flag_and_compile_cache(self) -> None:
        enabled = RUNNER.preset_spec(
            RUNNER.MODE_COMPILED_ASYNC,
            enable_static_kernel=True,
        )
        disabled = RUNNER.preset_spec(
            RUNNER.MODE_COMPILED_ASYNC,
            enable_static_kernel=False,
        )
        self.assertTrue(enabled["enable_static_kernel"])
        self.assertFalse(disabled["enable_static_kernel"])
        self.assertNotEqual(
            enabled["compile_cache_dir"],
            disabled["compile_cache_dir"],
        )
        ignored = {"enable_static_kernel", "compile_cache_dir"}
        self.assertEqual(
            {key: value for key, value in enabled.items() if key not in ignored},
            {key: value for key, value in disabled.items() if key not in ignored},
        )

    def test_eager_sync_preset_omits_tuned_scheduler_limits(self) -> None:
        spec = RUNNER.preset_spec(RUNNER.MODE_EAGER_SYNC, block_size=128)
        self.assertEqual(spec["engine"], "LLM")
        self.assertTrue(spec["enforce_eager"])
        self.assertFalse(spec["enable_prefix_caching"])
        self.assertIsNone(spec["max_num_seqs"])
        self.assertIsNone(spec["max_num_batched_tokens"])
        self.assertEqual(spec["block_size"], 128)

    def test_eager_async_keeps_async_scheduler_without_graphs(self) -> None:
        spec = RUNNER.preset_spec(RUNNER.MODE_EAGER_ASYNC, block_size=128)
        self.assertEqual(spec["engine"], "AsyncLLM")
        self.assertTrue(spec["enforce_eager"])
        self.assertTrue(spec["enable_prefix_caching"])
        self.assertTrue(spec["enable_chunked_prefill"])
        self.assertFalse(spec["enable_npugraph_ex"])
        self.assertIsNone(spec["cudagraph_mode"])
        self.assertEqual(spec["block_size"], 128)

    def test_aclgraph_async_disables_npugraph_and_static_kernels(self) -> None:
        spec = RUNNER.preset_spec(RUNNER.MODE_ACLGRAPH_ASYNC, block_size=128)
        self.assertEqual(spec["engine"], "AsyncLLM")
        self.assertFalse(spec["enforce_eager"])
        self.assertFalse(spec["enable_npugraph_ex"])
        self.assertFalse(spec["enable_static_kernel"])
        self.assertEqual(spec["cudagraph_mode"], "FULL_DECODE_ONLY")
        self.assertEqual(
            spec["compile_cache_dir"],
            str(RUNNER.DEFAULT_ACLGRAPH_NO_NPUGRAPH_CACHE_DIR),
        )
        self.assertEqual(spec["block_size"], 128)

    def test_310p_engine_kwargs_apply_block_size_to_all_diagnostic_modes(self) -> None:
        for mode in (
            RUNNER.MODE_EAGER_SYNC,
            RUNNER.MODE_EAGER_ASYNC,
            RUNNER.MODE_ACLGRAPH_ASYNC,
        ):
            with self.subTest(mode=mode):
                kwargs = RUNNER.build_engine_kwargs(
                    mode,
                    Path("/model"),
                    object(),
                    block_size=128,
                )
                self.assertEqual(kwargs["block_size"], 128)

    def test_aclgraph_engine_kwargs_disable_npugraph(self) -> None:
        kwargs = RUNNER.build_engine_kwargs(
            RUNNER.MODE_ACLGRAPH_ASYNC,
            Path("/model"),
            object(),
            block_size=128,
        )
        ascend = kwargs["additional_config"]["ascend_compilation_config"]
        self.assertFalse(ascend["enable_npugraph_ex"])
        self.assertFalse(ascend["enable_static_kernel"])
        self.assertEqual(
            kwargs["compilation_config"]["cudagraph_mode"],
            "FULL_DECODE_ONLY",
        )

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
        self.assertFalse(ascend["enable_static_kernel"])
        self.assertEqual(
            kwargs["compilation_config"]["cache_dir"],
            str(RUNNER.DEFAULT_STATIC_OFF_COMPILE_CACHE_DIR),
        )

    def test_static_kernel_on_engine_kwargs_use_reference_cache(self) -> None:
        kwargs = RUNNER.build_engine_kwargs(
            RUNNER.MODE_COMPILED_ASYNC,
            Path("/model"),
            object(),
            enable_static_kernel=True,
        )
        ascend = kwargs["additional_config"]["ascend_compilation_config"]
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
