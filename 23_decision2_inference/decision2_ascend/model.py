"""Eos head over Ascend's registered Qwen3.5 text core. TP1, full prefill only.

The original head class and weights come from the checksum-verified Eos bundle.
No LM head, vision tower, CUDA kernels, or native Transformers forward is used.
"""
import json
from pathlib import Path
import sys

import torch
from torch import nn
from safetensors.torch import load_file
from vllm.model_executor.layers.pooler.abstract import Pooler
from vllm.model_executor.models.interfaces import IsHybrid, HasInnerState, SupportsMRoPE
from vllm.model_executor.models.interfaces_base import default_pooling_type
from vllm.model_executor.models.qwen3_5 import Qwen3_5Model, Qwen3_5ForConditionalGeneration


class DecisionPooler(Pooler):
    def __init__(self, head):
        super().__init__()
        self.head = head

    def get_supported_tasks(self):
        return {"classify"}

    def forward(self, hidden_states, pooling_metadata):
        cursor = pooling_metadata.get_pooling_cursor()
        if cursor.is_partial_prefill():
            raise RuntimeError("Decision2 initial adapter requires unchunked full prefill")
        outputs = []
        for i, params in enumerate(pooling_metadata.pooling_params):
            meta = (params.extra_kwargs or {}).get("decision2")
            if meta is None:
                # vLLM 0.21 _dummy_pooler_run_task supplies CPU zero tokens and
                # no extra_kwargs. Real execution does not request CPU tokens;
                # the HTTP boundary rejects missing metadata before scheduling.
                dummy = pooling_metadata.prompt_token_ids_cpu
                if dummy is None or bool(torch.any(dummy != 0)):
                    raise ValueError("Missing Decision2 option positions")
                length = int(cursor.prompt_lens_cpu[i])
                meta = {"candidate_positions": [0] * 255, "query_position": length - 1,
                        "token_count": length}
            if meta["token_count"] != int(cursor.prompt_lens_cpu[i]):
                raise ValueError("Server changed the token sequence length")
            positions = torch.tensor([*meta["candidate_positions"], meta["query_position"]],
                                     dtype=torch.long, device=hidden_states.device)
            selected = hidden_states.index_select(0, positions + cursor.first_token_indices_gpu[i])
            logits = self.head(selected[:-1].unsqueeze(0), selected[-1].unsqueeze(0))[0]
            outputs.append(logits)
        return outputs


@default_pooling_type(seq_pooling_type="LAST", tok_pooling_type="ALL")
class Decision2EosForPooling(nn.Module, IsHybrid, HasInnerState, SupportsMRoPE):
    is_pooling_model = True
    get_mamba_state_dtype_from_config = Qwen3_5ForConditionalGeneration.get_mamba_state_dtype_from_config
    get_mamba_state_shape_from_config = Qwen3_5ForConditionalGeneration.get_mamba_state_shape_from_config
    get_mamba_state_copy_func = Qwen3_5ForConditionalGeneration.get_mamba_state_copy_func

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        if vllm_config.parallel_config.tensor_parallel_size != 1:
            raise ValueError("Initial Decision2 adapter supports TP1 only")
        if vllm_config.scheduler_config.enable_chunked_prefill or vllm_config.cache_config.enable_prefix_caching:
            raise ValueError("Initial Decision2 adapter requires chunking and prefix caching disabled")
        config = vllm_config.model_config.hf_text_config
        self.bundle = Path(config.decision2_bundle)
        sys.path.insert(0, str(self.bundle))
        from decision2._vendor.dev2model.decision_model import CandidateHead
        self.model = Qwen3_5Model(vllm_config=vllm_config, prefix="model")
        self.pooler = DecisionPooler(CandidateHead(config.hidden_size).float())
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def embed_input_ids(self, input_ids):
        return self.model.embed_input_ids(input_ids)

    def get_mrope_input_positions(self, input_tokens, mm_features):
        if mm_features:
            raise ValueError("Eos accepts text only")
        # Text occupies identical positions on the three M-RoPE axes, matching
        # Qwen3.5's native text-only position_ids expansion. No vision config.
        return torch.arange(len(input_tokens), dtype=torch.long).expand(3, -1).clone(), 0

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def load_weights(self, weights):
        loaded = {"model." + n for n in self.model.load_weights(weights)}
        self.pooler.head.load_state_dict(load_file(str(self.bundle / "decision_head.safetensors")), strict=True)
        loaded.update("pooler.head." + n for n, _ in self.pooler.head.named_parameters())
        missing = set(dict(self.named_parameters())) - loaded
        if missing:
            raise RuntimeError(f"Uninitialized parameters: {sorted(missing)}")
        print(json.dumps({"event": "decision2_weights_loaded", "parameters": len(loaded),
                          "head_dtype": str(next(self.pooler.head.parameters()).dtype),
                          "core": type(self.model).__name__}), flush=True)
        return loaded
