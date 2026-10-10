"""Small architecture/loader checks; these do not establish NPU support."""
import json
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import XLMRobertaConfig, XLMRobertaModel

from modeling_bge_m3 import BGEM3, Config, load_model


class ReferenceTest(unittest.TestCase):
    def test_encoder_pooling_and_checkpoint(self):
        torch.manual_seed(17)
        hf_config = XLMRobertaConfig(vocab_size=64, hidden_size=32, intermediate_size=48,
                                    num_hidden_layers=2, num_attention_heads=4,
                                    max_position_embeddings=64, type_vocab_size=1,
                                    hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
        hf_config._attn_implementation = "eager"
        reference = XLMRobertaModel(hf_config).eval()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text(hf_config.to_json_string())
            torch.save(reference.state_dict(), root / "pytorch_model.bin")
            local = load_model(root, torch.device("cpu"), torch.float32)
        ids = torch.tensor([[0, 7, 8, 2, 1, 1], [0, 9, 10, 11, 12, 2]])
        mask = ids.ne(1).long()
        with torch.inference_mode():
            expected_hidden = reference(ids, attention_mask=mask).last_hidden_state
            actual_hidden = local.forward_hidden_states(ids, mask)
            torch.testing.assert_close(actual_hidden, expected_hidden, atol=2e-6, rtol=2e-5)
            expected = torch.nn.functional.normalize(expected_hidden[:, 0], dim=-1)
            torch.testing.assert_close(local(ids, mask), expected, atol=2e-6, rtol=2e-5)
            # Padding and batching must not change the first sentence's embedding.
            single = local(ids[:1, :4], mask[:1, :4])
            torch.testing.assert_close(single[0], local(ids, mask)[0], atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(local(ids, mask).norm(dim=-1), torch.ones(2))

    def test_reject_decoder(self):
        raw = XLMRobertaConfig(is_decoder=True).to_dict()
        with self.assertRaises(ValueError):
            Config.from_dict(raw)


if __name__ == "__main__":
    unittest.main()
