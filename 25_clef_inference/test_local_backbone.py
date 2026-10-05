"""Small independent-oracle checks on NPU; Transformers is test-only.

Run in the existing Clef environment after sourcing npu-setup. These checks
cover chunk boundaries and a two-layer hybrid forward, not model quality.
The real checkpoint is checked separately by run_local_smoke.py.
"""
import unittest
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from modeling_backbone import TextBackbone, chunk_gated_delta_rule


class BackboneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.npu.is_available():
            raise RuntimeError("NPU required; do not report skipped checks as validation")
        torch.npu.set_device(0)
        torch.set_num_threads(8)

    @torch.inference_mode()
    def test_chunk_scan_against_token_recurrence(self):
        # Independent scalar-token recurrence tests causal masking and chunk carry.
        for length in (1, 63, 64, 65, 129):
            with self.subTest(length=length):
                torch.manual_seed(length)
                q, k, v = [torch.randn(1, length, 2, 8, device="npu:0") for _ in range(3)]
                g = -torch.rand(1, length, 2, device="npu:0")
                beta = torch.rand(1, length, 2, device="npu:0")
                actual = chunk_gated_delta_rule(q, k, v, g, beta)
                q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * 8**-0.5
                k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
                state = torch.zeros(1, 2, 8, 8, device="npu:0")
                expected = []
                for t in range(length):
                    state = state * g[:, t].exp()[..., None, None]
                    residual = v[:, t] - (k[:, t, :, :, None] * state).sum(-2)
                    state = state + k[:, t, :, :, None] * (beta[:, t, :, None] * residual)[..., None, :]
                    expected.append((q[:, t, :, :, None] * state).sum(-2))
                torch.testing.assert_close(actual, torch.stack(expected, dim=1), atol=2e-6, rtol=2e-5)

    @torch.inference_mode()
    def test_hybrid_backbone_against_transformers(self):
        config = Qwen3_5TextConfig(
            vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            layer_types=["linear_attention", "full_attention"], num_attention_heads=4,
            num_key_value_heads=2, head_dim=16, linear_num_key_heads=2,
            linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
            linear_conv_kernel_dim=4, rms_norm_eps=1e-6,
            rope_parameters=dict(rope_type="default", rope_theta=10000000,
                                 partial_rotary_factor=0.5, mrope_section=[1, 1, 2]))
        config._attn_implementation = "eager"
        torch.manual_seed(7)
        reference = Qwen3_5TextModel(config).to(device="npu:0", dtype=torch.bfloat16).eval()
        local = TextBackbone(SimpleNamespace(**config.to_dict())).to(device="npu:0", dtype=torch.bfloat16).eval()
        local.load_state_dict(reference.state_dict(), strict=True)
        # .to(bfloat16) casts buffers too; the release loader initializes RoPE in FP32.
        inv, _ = reference.rotary_emb.compute_default_rope_parameters(config)
        reference.rotary_emb.inv_freq = inv.to("npu:0")
        local.inv_freq = inv.to("npu:0")
        for length in (1, 65, 129):
            with self.subTest(length=length):
                ids = torch.randint(0, config.vocab_size, (1, length), device="npu:0")
                expected = reference(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state
                torch.testing.assert_close(local(ids), expected, atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, "one nonempty"):
            local(torch.zeros(2, 4, dtype=torch.long, device="npu:0"))


if __name__ == "__main__":
    unittest.main()
