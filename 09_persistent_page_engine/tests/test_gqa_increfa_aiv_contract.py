"""CPU-side contract tests for the experimental GQA AIV graph operator."""

from __future__ import annotations

import inspect
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import torch


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from paddleocr_vl.model.gqa_increfa_aiv import (
    MIN_KV_LENGTH_FOR_48_CORES,
    gqa_incre_flash_attention_aiv,
)
from paddleocr_vl.model import text_decode


class GqaIncrefaAivContractTest(unittest.TestCase):
    def test_mixed24_prep_matches_outer_superkernel_geometry(self) -> None:
        kernel = (
            EXPERIMENT_ROOT
            / "custom_ops"
            / "paddle_decode_kv_scatter_query"
            / "op_kernel"
            / "paddle_decode_kv_scatter_query_v4.cpp"
        ).read_text(encoding="utf-8")

        v4_start = kernel.index(
            'void paddle_decode_kv_scatter_query_v4('
        )
        mixed_start = kernel.index(
            'void paddle_decode_kv_scatter_query_mixed24('
        )
        v4_kernel = kernel[v4_start:mixed_start]
        mixed_kernel = kernel[mixed_start:]

        self.assertIn("GetBlockIdx() == 0", v4_kernel)
        self.assertIn(
            "kernel.Process();\n        PipeBarrier<PIPE_ALL>();",
            v4_kernel,
        )
        self.assertNotIn("SyncAll<true>();", v4_kernel)
        self.assertIn("KERNEL_TYPE_MIX_AIC_1_1", mixed_kernel)
        self.assertIn("if (g_coreType == AIC)", mixed_kernel)
        self.assertIn("GetBlockIdx() == 0", mixed_kernel)
        self.assertIn("SyncAll<true>();", mixed_kernel)
        self.assertGreater(
            mixed_kernel.index("SyncAll<true>();"),
            mixed_kernel.index("PipeBarrier<PIPE_ALL>();"),
        )

        host = (
            EXPERIMENT_ROOT
            / "custom_ops"
            / "paddle_decode_kv_scatter_query"
            / "op_host"
            / "paddle_decode_kv_scatter_query.cpp"
        ).read_text(encoding="utf-8")
        converter = (
            EXPERIMENT_ROOT
            / "paddleocr_vl"
            / "model"
            / "decode_kv_scatter_query.py"
        ).read_text(encoding="utf-8")
        self.assertIn("context->SetBlockDim(24);", host)
        self.assertIn("PaddleDecodeKvScatterQueryMixed24", host)
        self.assertIn('GE_OP_NAME_MIXED24 = "PaddleDecodeKvScatterQueryMixed24"', converter)

    def test_attention_overlay_uses_superkernel_safe_plain_kv_abi(self) -> None:
        operator_root = (
            EXPERIMENT_ROOT
            / "custom_ops"
            / "paddle_gqa_increfa_aiv"
        )
        op_def = (
            operator_root
            / "source_overlay_decode_attention"
            / "op_host"
            / "paddle_decode_gqa_attention_aiv_def.cpp"
        ).read_text(encoding="utf-8")
        converter = (
            EXPERIMENT_ROOT
            / "paddleocr_vl"
            / "model"
            / "decode_gqa_attention_aiv.py"
        ).read_text(encoding="utf-8")
        build_script = (operator_root / "build.sh").read_text(encoding="utf-8")
        kernel_entry = (
            operator_root
            / "source_overlay_decode_attention"
            / "op_kernel"
            / "paddle_decode_gqa_attention_aiv.cpp"
        ).read_text(encoding="utf-8")

        self.assertIn('Input("key").ParamType(REQUIRED)', op_def)
        self.assertIn('Input("value").ParamType(REQUIRED)', op_def)
        self.assertNotIn('Input("pse_shift")', op_def)
        self.assertIn('"key": key', converter)
        self.assertIn('"value": value', converter)
        self.assertIn("0014-superkernel-plain-kv-attention.patch", build_script)
        self.assertIn("0015-decoder-fixed-no-optional-inputs.patch", build_script)
        self.assertIn("PADDLE_DECODE_GQA_PLAIN_KV", kernel_entry)
        self.assertNotIn("uint8_t *pseShift", kernel_entry)

    def test_attention_only_mixed24_matches_outer_decoder_geometry(self) -> None:
        operator_root = (
            EXPERIMENT_ROOT / "custom_ops" / "paddle_gqa_increfa_aiv"
        )
        build_script = (operator_root / "build.sh").read_text(encoding="utf-8")
        geometry_patch = (
            operator_root / "patches" / "0021-attention-only-mixed24.patch"
        ).read_text(encoding="utf-8")
        kernel = (
            operator_root
            / "source_overlay_decode_attention"
            / "op_kernel"
            / "paddle_decode_gqa_attention_aiv.cpp"
        ).read_text(encoding="utf-8")
        converter = (
            EXPERIMENT_ROOT
            / "paddleocr_vl"
            / "model"
            / "decode_gqa_attention_mixed24.py"
        ).read_text(encoding="utf-8")

        self.assertIn("decode_attention_only_mixed24)", build_script)
        self.assertIn("0021-attention-only-mixed24.patch", build_script)
        self.assertIn('EXPECTED_TASK_RATIO="1:1"', build_script)
        self.assertIn("KERNEL_TYPE_MIX_AIC_1_1", geometry_patch)
        self.assertIn("launchAivNum = 24U", geometry_patch)
        self.assertIn("launchAicNum = 24U", geometry_patch)
        self.assertIn("if (GetBlockIdx() >= 16U)", kernel)
        self.assertIn('GE_OP_NAME = "PaddleDecodeGqaAttentionMixed24"', converter)

        optimization = text_decode.resolve_decode_optimization(
            "paddle_decoder_megakernel_b1_attention_mixed24"
        )
        self.assertTrue(optimization.super_kernel_scope)
        self.assertTrue(optimization.ascendc_decode_gqa_attention_mixed24)
        self.assertFalse(optimization.ascendc_decode_gqa_attention)
        self.assertFalse(optimization.ascendc_decode_gqa)
        self.assertFalse(optimization.ascendc_packed_qkv_rope_gqa_mixed24)
        self.assertEqual(
            text_decode.decode_attention_label(
                SimpleNamespace(type="npu"), optimization
            ),
            "paddle_decode_gqa_attention_mixed24",
        )

    def test_fused_decode_overlay_has_compact_balanced_plain_kv_abi(self) -> None:
        operator_root = (
            EXPERIMENT_ROOT
            / "custom_ops"
            / "paddle_gqa_increfa_aiv"
        )
        overlay = operator_root / "source_overlay_decode_fused_plain"
        op_def = (
            overlay
            / "op_host"
            / "paddle_decode_gqa_incre_flash_attention_aiv_def.cpp"
        ).read_text(encoding="utf-8")
        kernel = (
            overlay
            / "op_kernel"
            / "paddle_decode_gqa_incre_flash_attention_aiv.cpp"
        ).read_text(encoding="utf-8")
        converter = (
            EXPERIMENT_ROOT
            / "paddleocr_vl"
            / "model"
            / "decode_gqa_increfa_aiv.py"
        ).read_text(encoding="utf-8")
        build_script = (operator_root / "build.sh").read_text(encoding="utf-8")

        self.assertIn('Input("key").ParamType(REQUIRED)', op_def)
        self.assertIn('Input("value").ParamType(REQUIRED)', op_def)
        self.assertNotIn('Input("pse_shift")', op_def)
        self.assertNotIn('Input("key_cache_ref")', op_def)
        self.assertIn("PADDLE_DECODE_GQA_PLAIN_KV", kernel)
        self.assertIn("if (GetBlockIdx() == 0)", kernel)
        idle_guard = "if (GetBlockIdx() >= kAivCoreCount)"
        self.assertIn(idle_guard, kernel)
        self.assertLess(
            kernel.index(idle_guard),
            kernel.index("TPipe fusedPipe;"),
        )
        self.assertNotIn("SyncAll();", kernel)
        self.assertIn("SyncAll(syncGlobal, syncLocal, kAivCoreCount)", kernel)
        sync_call = kernel.index(
            "SyncAll(syncGlobal, syncLocal, kAivCoreCount)"
        )
        reset_call = kernel.index("fusedPipe.Reset();")
        self.assertIn(
            "PipeBarrier<PIPE_ALL>();",
            kernel[sync_call:reset_call],
        )
        self.assertNotIn("syncPipe.Destroy();", kernel)
        self.assertIn("&fusedPipe);", kernel)
        self.assertNotIn("GetUserWorkspace(workspace)", kernel)
        self.assertIn(
            "reinterpret_cast<__gm__ int32_t *>(attentionOut)", kernel
        )
        self.assertIn("kSyncStorageBytes", kernel)
        self.assertNotIn("attentionWorkspace", kernel)
        self.assertIn("FetchEventID(HardEvent::V_MTE3)", kernel)
        self.assertNotIn("SetWaitFlag", kernel)
        self.assertIn('"key": key_cache', converter)
        self.assertIn('mapping[1] = 1', converter)
        self.assertIn('decode_fused_plain)', build_script)
        self.assertIn("source_overlay_decode_fused_plain", build_script)
        fixed_abi_patch = (
            operator_root / "patches" / "0015-decoder-fixed-no-optional-inputs.patch"
        ).read_text(encoding="utf-8")
        self.assertIn("+    ifaContext.deqScale1.tensor = nullptr;", fixed_abi_patch)
        self.assertIn("+    ifaContext.blockTable.tensor = nullptr;", fixed_abi_patch)
        self.assertIn("+    ifaContext.kvPaddingSize.desc = nullptr;", fixed_abi_patch)
        self.assertNotIn("0016-decoder-soft-sync-workspace.patch", build_script)
        self.assertIn("0017-decoder-reuse-attention-tpipe.patch", build_script)
        tpipe_reuse_patch = (
            operator_root / "patches" / "0017-decoder-reuse-attention-tpipe.patch"
        ).read_text(encoding="utf-8")
        self.assertIn("+    TPipe &tPipe = *externalPipe;", tpipe_reuse_patch)
        probe = (
            EXPERIMENT_ROOT
            / "scripts"
            / "probes"
            / "compare_paddle_decode_gqa_increfa_aiv.py"
        ).read_text(encoding="utf-8")
        self.assertIn("--super-kernel-options", probe)
        self.assertIn("split-mode=4", probe)

    def test_nonsplit_megakernel_uses_separate_prep_and_16_aiv_blocks(self) -> None:
        optimization = text_decode.resolve_decode_optimization(
            "paddle_decoder_megakernel_b1_nonsplit_gqa"
        )

        self.assertTrue(optimization.super_kernel_scope)
        self.assertTrue(optimization.ascendc_kv_scatter_query)
        self.assertFalse(optimization.ascendc_decode_gqa)
        self.assertTrue(optimization.ascendc_decode_gqa_attention)
        self.assertEqual(optimization.gqa_aiv_vector_core_count, 16)
        self.assertIn(
            "strict-scope-check=abort",
            optimization.super_kernel_options,
        )

    def test_mixed_superkernel_gqa_is_an_independent_operator(self) -> None:
        operator_root = (
            EXPERIMENT_ROOT
            / "custom_ops"
            / "paddle_gqa_increfa_aiv"
        )
        build_script = (operator_root / "build.sh").read_text(encoding="utf-8")
        mixed_patch = (
            operator_root
            / "patches"
            / "0018-decoder-mixed-task-geometry.patch"
        ).read_text(encoding="utf-8")
        converter = (
            EXPERIMENT_ROOT
            / "paddleocr_vl"
            / "model"
            / "decode_gqa_increfa_mixed.py"
        ).read_text(encoding="utf-8")
        probe = (
            EXPERIMENT_ROOT
            / "scripts"
            / "probes"
            / "compare_paddle_decode_matmul_qkv_gqa_mixed_boundary.py"
        ).read_text(encoding="utf-8")

        self.assertIn("decode_fused_plain_mixed_superkernel)", build_script)
        self.assertIn("paddle_decode_kv_gqa_mixed", build_script)
        self.assertIn(
            "paddle_decode_gqa_incre_flash_attention_mixed",
            build_script,
        )
        self.assertIn("0018-decoder-mixed-task-geometry.patch", build_script)
        self.assertIn("renamed_path=", build_script)
        self.assertIn("KERNEL_TYPE_MIX_AIC_1_1", mixed_patch)
        self.assertIn("launchAicNum = launchAivNum", mixed_patch)
        self.assertIn(
            "SyncAll<false>(syncGlobal, syncLocal, kAivCoreCount)",
            mixed_patch,
        )
        self.assertIn("if (g_coreType == AIC)", mixed_patch)
        self.assertIn(
            'paddleocr_vl::decode_gqa_incre_flash_attention_mixed',
            converter,
        )
        self.assertIn(
            'GE_OP_NAME = "PaddleDecodeGqaIncreFlashAttentionMixed"',
            converter,
        )
        self.assertIn("decode_gqa_increfa_mixed", probe)
        self.assertNotIn("decode_gqa_increfa_aiv import", probe)

        mixed_optimization = text_decode.resolve_decode_optimization(
            "paddle_decoder_megakernel_b1_fused_gqa_mixed"
        )
        self.assertTrue(mixed_optimization.super_kernel_scope)
        self.assertTrue(mixed_optimization.ascendc_decode_gqa_mixed)
        self.assertFalse(mixed_optimization.ascendc_decode_gqa)
        self.assertIn(
            "strict-scope-check=abort",
            mixed_optimization.super_kernel_options,
        )
        self.assertIn("early-start=0", mixed_optimization.super_kernel_options)
        self.assertEqual(
            text_decode.decode_attention_label(
                SimpleNamespace(type="npu"), mixed_optimization
            ),
            "paddle_decode_gqa_increfa_mixed",
        )
        self.assertIn(
            "register_decode_gqa_increfa_mixed_converter",
            (EXPERIMENT_ROOT / "paddleocr_vl" / "model" / "text_decode.py")
            .read_text(encoding="utf-8"),
        )

    def test_mixed24_matches_decoder_geometry_before_hardware_barrier(self) -> None:
        operator_root = (
            EXPERIMENT_ROOT
            / "custom_ops"
            / "paddle_gqa_increfa_aiv"
        )
        build_script = (operator_root / "build.sh").read_text(encoding="utf-8")
        mixed24_patch = (
            operator_root
            / "patches"
            / "0019-decoder-24-aiv-hardware-barrier.patch"
        ).read_text(encoding="utf-8")
        converter = (
            EXPERIMENT_ROOT
            / "paddleocr_vl"
            / "model"
            / "decode_gqa_increfa_mixed.py"
        ).read_text(encoding="utf-8")
        boundary_probe = (
            EXPERIMENT_ROOT
            / "scripts"
            / "probes"
            / "compare_paddle_decode_matmul_qkv_gqa_mixed_boundary.py"
        ).read_text(encoding="utf-8")
        added_lines = "\n".join(
            line[1:]
            for line in mixed24_patch.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        )

        self.assertIn("decode_fused_plain_mixed_superkernel24", build_script)
        self.assertIn("paddle_decode_kv_gqa_mixed24", build_script)
        self.assertIn("0019-decoder-24-aiv-hardware-barrier.patch", build_script)
        self.assertIn("constexpr uint32_t launchAivNum = 24U", mixed24_patch)
        self.assertIn("GQA mixed24 block dim", mixed24_patch)
        self.assertIn("SyncAll<true>();", mixed24_patch)
        barrier = mixed24_patch.index("SyncAll<true>();")
        idle_return = mixed24_patch.index(
            "if (GetBlockIdx() >= kAivCoreCount)", barrier
        )
        self.assertLess(barrier, idle_return)
        self.assertNotIn("SyncAll<false>", added_lines)
        self.assertNotIn("kSyncStorageBytes", added_lines)
        self.assertIn(
            'GE_OP_NAME_MIXED24 = "PaddleDecodeGqaIncreFlashAttentionMixed24"',
            converter,
        )
        self.assertIn('choices=("mixed", "mixed24")', boundary_probe)

        optimization = text_decode.resolve_decode_optimization(
            "paddle_decoder_megakernel_b1_fused_gqa_mixed24"
        )
        self.assertTrue(optimization.super_kernel_scope)
        self.assertTrue(optimization.ascendc_decode_gqa_mixed24)
        self.assertFalse(optimization.ascendc_decode_gqa_mixed)
        self.assertFalse(optimization.ascendc_decode_gqa)
        self.assertEqual(optimization.gqa_aiv_vector_core_count, 16)
        self.assertIn("feed-sync-all=0", optimization.super_kernel_options)
        self.assertIn("early-start=0", optimization.super_kernel_options)
        self.assertEqual(
            text_decode.decode_attention_label(
                SimpleNamespace(type="npu"), optimization
            ),
            "paddle_decode_gqa_increfa_mixed24",
        )

    def test_packed_qkv_rope_operator_removes_the_split_pipe_boundary(self) -> None:
        operator_root = (
            EXPERIMENT_ROOT / "custom_ops" / "paddle_gqa_increfa_aiv"
        )
        overlay = operator_root / "source_overlay_decode_packed_qkv_rope"
        kernel = (
            overlay
            / "op_kernel"
            / "paddle_decode_gqa_incre_flash_attention_aiv.cpp"
        ).read_text(encoding="utf-8")
        op_def = (
            overlay
            / "op_host"
            / "paddle_decode_gqa_incre_flash_attention_aiv_def.cpp"
        ).read_text(encoding="utf-8")
        build_script = (operator_root / "build.sh").read_text(encoding="utf-8")
        geometry_patch = (
            operator_root / "patches" / "0020-packed-qkv-rope-mixed24.patch"
        ).read_text(encoding="utf-8")
        converter = (
            EXPERIMENT_ROOT
            / "paddleocr_vl"
            / "model"
            / "decode_packed_qkv_rope_gqa_mixed24.py"
        ).read_text(encoding="utf-8")

        self.assertIn("decode_packed_qkv_rope_mixed24)", build_script)
        self.assertIn("PADDLE_GQA_PREFLIGHT_ONLY", build_script)
        self.assertIn("PREFLIGHT_SUMMARY", build_script)
        self.assertIn('if [[ "$BUILD_SOURCE_ROOT" != /* ]]', build_script)
        self.assertEqual(
            (overlay / "CMakeLists.txt").read_text(encoding="utf-8").strip(),
            "add_subdirectory(op_host)",
        )
        self.assertIn("source_overlay_decode_packed_qkv_rope", build_script)
        self.assertIn("0020-packed-qkv-rope-mixed24.patch", build_script)
        self.assertIn('Input("query").ParamType(REQUIRED)', op_def)
        self.assertIn('Input("factor_lut").ParamType(REQUIRED)', op_def)
        self.assertIn('Input("rope_delta").ParamType(REQUIRED)', op_def)
        self.assertNotIn('Input("key_state")', op_def)
        self.assertNotIn('Input("value_state")', op_def)
        self.assertIn("qkvGm[kQueryElements]", kernel)
        self.assertIn("RotateHalf(", kernel)
        self.assertIn("cosineInputQueue", kernel)
        self.assertIn("sineInputQueue", kernel)
        self.assertNotIn("factorInputQueue", kernel)
        self.assertIn("packedInputQueue", kernel)
        self.assertNotIn("maskOutputQueue", kernel)
        self.assertIn("SyncAll<true>();", kernel)
        self.assertGreaterEqual(kernel.count("PipeBarrier<PIPE_V>();"), 6)
        self.assertNotIn("SyncAll<false>", kernel)
        self.assertIn("incre_flash_attention_FIAS_arch32", kernel)
        self.assertIn("kPackedQueryShape", geometry_patch)
        self.assertIn("launchAivNum = 24U", geometry_patch)
        self.assertNotIn('mapping[3] = 3', converter)
        self.assertIn('mapping[4] = 0', converter)
        self.assertIn('"query": qkv', converter)
        self.assertIn("rope_delta.shape != (1,)", converter)
        self.assertIn(
            "rope_deltas = rope_deltas.reshape(-1)",
            inspect.getsource(text_decode),
        )

        optimization = text_decode.resolve_decode_optimization(
            "paddle_decoder_megakernel_b1_packed_qkv_rope_gqa_mixed24"
        )
        self.assertTrue(optimization.super_kernel_scope)
        self.assertTrue(optimization.ascendc_packed_qkv_rope_gqa_mixed24)
        self.assertFalse(optimization.ascendc_qkv_split)
        self.assertFalse(optimization.ascendc_rope_lookup)
        self.assertEqual(
            text_decode.decode_attention_label(
                SimpleNamespace(type="npu"), optimization
            ),
            "paddle_decode_packed_qkv_rope_gqa_mixed24",
        )

    def test_rejects_unsafe_48_core_short_partition(self) -> None:
        kv_length = MIN_KV_LENGTH_FOR_48_CORES - 1
        query = torch.empty((1, 16, 1, 128), dtype=torch.float16)
        key = torch.empty((1, 2, kv_length, 128), dtype=torch.float16)
        value = torch.empty_like(key)
        mask = torch.empty((1, 1, 1, kv_length), dtype=torch.bool)

        with self.assertRaisesRegex(ValueError, "requires KV length >= 1536"):
            gqa_incre_flash_attention_aiv(
                query,
                key,
                value,
                mask,
                num_heads=16,
                num_key_value_heads=2,
                scale_value=128**-0.5,
                vector_core_count=48,
            )

    def test_qkv_split_completes_fused_producer_before_destroy(self) -> None:
        kernel = (
            EXPERIMENT_ROOT
            / "custom_ops"
            / "paddle_decode_qkv_split"
            / "op_kernel"
            / "paddle_decode_qkv_split_v4.cpp"
        ).read_text(encoding="utf-8")
        process = kernel.index("kernel.Process();")
        barrier = kernel.index("PipeBarrier<PIPE_ALL>();", process)
        destroy = kernel.index("pipe.Destroy();", barrier)
        self.assertLess(process, barrier)
        self.assertLess(barrier, destroy)

    def test_fused_decode_attention_uses_static_b1_shape(self) -> None:
        query = torch.empty((1, 16, 1, 128), dtype=torch.float16)
        key_state = torch.empty((1, 2, 1, 128), dtype=torch.float16)
        value_state = torch.empty_like(key_state)
        attention = SimpleNamespace(
            num_heads=16,
            num_key_value_heads=2,
            head_dim=128,
            scaling=128**-0.5,
            o_proj=object(),
        )
        optimization = text_decode.resolve_decode_optimization(
            "paddle_decoder_megakernel_b1_fused_gqa"
        )

        with (
            mock.patch.object(
                text_decode,
                "_project_decode_qkv",
                return_value=(query, key_state, value_state),
            ),
            mock.patch.object(
                text_decode,
                "_apply_decode_rotary",
                return_value=(query, key_state),
            ),
            mock.patch.object(
                text_decode,
                "decode_gqa_incre_flash_attention_aiv",
                return_value=query,
            ),
            mock.patch.object(
                text_decode,
                "_linear_tokenwise",
                side_effect=lambda _linear, tensor: tensor,
            ),
        ):
            output = text_decode._decode_attention(
                attention,
                torch.empty((1, 1, 1024), dtype=torch.float16),
                (torch.empty(0), torch.empty(0)),
                None,
                torch.empty((1, 2, 1024, 128), dtype=torch.float16),
                torch.empty((1, 2, 1024, 128), dtype=torch.float16),
                torch.zeros((1,), dtype=torch.int64),
                torch.empty((1, 1, 1, 1024), dtype=torch.bool),
                None,
                None,
                optimization,
            )

        self.assertEqual(output.shape, (1, 1, 2048))


if __name__ == "__main__":
    unittest.main()
