#!/usr/bin/env python3

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from modeling_qwen3_moe_pipeline import (
    Qwen3MoeConfig,
    selected_expert_bmm,
)


class SelectedExpertBmmTest(unittest.TestCase):
    def test_matches_direct_selected_expert_math(self) -> None:
        torch.manual_seed(7)
        tokens = 3
        hidden = 8
        intermediate = 5
        experts = 7
        top_k = 3
        hidden_states = torch.randn(tokens, hidden)
        selected = torch.tensor([[0, 2, 6], [4, 1, 3], [2, 5, 0]])
        weights = torch.softmax(torch.randn(tokens, top_k), dim=-1)
        gate_up = torch.randn(experts, 2 * intermediate, hidden)
        down = torch.randn(experts, hidden, intermediate)

        actual = selected_expert_bmm(
            hidden_states, selected, weights, gate_up, down
        )
        expected_rows = []
        for token_index in range(tokens):
            output = torch.zeros(hidden)
            for slot in range(top_k):
                expert_index = int(selected[token_index, slot])
                gate, up = F.linear(
                    hidden_states[token_index], gate_up[expert_index]
                ).chunk(2, dim=-1)
                expert_output = F.linear(F.silu(gate) * up, down[expert_index])
                output += weights[token_index, slot] * expert_output
            expected_rows.append(output)
        expected = torch.stack(expected_rows)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_expected_qwen3_30b_contract(self) -> None:
        config = Qwen3MoeConfig(
            vocab_size=151936,
            hidden_size=2048,
            intermediate_size=6144,
            moe_intermediate_size=768,
            num_hidden_layers=48,
            num_attention_heads=32,
            num_key_value_heads=4,
            head_dim=128,
            num_experts=128,
            num_experts_per_tok=8,
            norm_topk_prob=True,
            rms_norm_eps=1e-6,
            rope_theta=1_000_000.0,
            max_position_embeddings=40_960,
            tie_word_embeddings=False,
        )
        config.validate_qwen3_30b_a3b()


if __name__ == "__main__":
    unittest.main()
