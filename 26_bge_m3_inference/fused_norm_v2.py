"""Unchanged upstream V2, exposed eagerly and in GE; experimental BGE wiring."""
import ctypes as C
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

_API = None


def initialize(library):
    """Call before importing torch_npu/TorchAir on the CANN 9.0.1 test host."""
    global _API
    library = Path(library)
    es = library.resolve().parents[2] / f"op_proto/es/lib/linux/{os.uname().machine}/libes_nn.so"
    if es.exists():
        C.CDLL("libes_nn.so", mode=C.RTLD_GLOBAL)
        C.CDLL(str(es), mode=C.RTLD_GLOBAL)
    metadata = library.resolve().parents[2] / "libbge_v2_graph_infer.so"
    if metadata.exists():
        C.CDLL(str(metadata), mode=C.RTLD_GLOBAL)
    import torch_npu
    from test_add_layer_norm_quant_v2 import V2
    _API = V2(library)
    from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
    from torchair import ge

    @register_fx_node_ge_converter(torch.ops.bge_m3.norm_quant_v2.default)
    def converter(x1, x2, gamma, beta, scale, bias, eps, meta_outputs=None):
        outputs = ge.custom_op("AddLayerNormQuantV2", inputs={
            "x1": x1, "x2": x2, "gamma": gamma, "beta": beta, "bias": bias,
            "scales1": scale, "scales2": None, "zero_points1": None, "zero_points2": None,
        }, attrs={"quant_mode": ge.attr.Str("static"), "epsilon": ge.attr.Float(eps),
                  "additional_output": ge.attr.Bool(False), "div_mode": ge.attr.Bool(False)},
            outputs=["y1", "y2", "x", "layernorm_res", "out_scales1", "out_scales2"])
        return outputs[3], outputs[0]


@torch.library.custom_op("bge_m3::norm_quant_v2", mutates_args=())
def norm_quant(x1: torch.Tensor, x2: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
               scale: torch.Tensor, bias: torch.Tensor | None, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    if _API is None:
        raise RuntimeError("Call initialize(op_api) first")
    return _API(x1, x2, gamma, beta, scale, bias, eps)


@norm_quant.register_fake
def fake(x1, x2, gamma, beta, scale, bias, eps):
    return torch.empty_like(x1), torch.empty_like(x1, dtype=torch.int8)


class FusedBGEM3(nn.Module):
    """48 norm->quant edges, keeping FP16 residuals and all original W8A8 scales."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        for i, layer in enumerate(model.encoder.layer):
            for name, linear in (("qkv", layer.attention.self.qkv.projections[0]),
                                 ("ffn", layer.intermediate.dense)):
                self.register_buffer(f"{name}_scale_{i}", linear.input_scale.reciprocal().half())

    def normalize(self, x, residual, norm, scale, bias=None):
        shape = x.shape
        y, q = norm_quant(x.reshape(-1, shape[-1]), residual.reshape(-1, shape[-1]),
                          norm.weight.reshape(1, -1), norm.bias.reshape(1, -1), scale,
                          None if bias is None else bias.reshape(1, -1), norm.eps)
        return y.reshape(shape), q

    @staticmethod
    def project(linear, x):
        # Feed bias to V2, as GE already fuses this bias into baseline AddLayerNorm.
        import torch_npu
        weight = linear.weight_q.T if linear.graph_transpose else linear.weight_q
        return torch_npu.npu_quant_matmul(linear.quantize(x), weight,
                                          scale=linear.deq_scale, bias=None,
                                          output_dtype=torch.float16).reshape(*x.shape[:-1], linear.out_features)

    @staticmethod
    def attention(attention, q_int8, shape, bias):
        b, s, h = shape
        q, k, v = [p.matmul(q_int8).view(b, s, attention.heads, attention.head_dim)
                    .transpose(1, 2).reshape(b * attention.heads, s, attention.head_dim)
                    for p in attention.qkv.projections]
        scores = torch.bmm(q, k.transpose(1, 2)) * (attention.head_dim ** -0.5)
        probs = torch.softmax(scores.view(b, attention.heads, s, s) + bias, dim=-1)
        context = torch.bmm(probs.view(b * attention.heads, s, s), v)
        return context.view(b, attention.heads, s, attention.head_dim).transpose(1, 2).contiguous().view(b, s, h)

    def forward(self, input_ids, attention_mask):
        emb = self.model.embeddings
        mask = input_ids.ne(emb.padding_idx).int()
        positions = (mask.cumsum(dim=1).to(mask.dtype) * mask).long() + emb.padding_idx
        x = emb.word_embeddings(input_ids) + emb.token_type_embeddings(torch.zeros_like(input_ids))
        hidden, quant = self.normalize(x, emb.position_embeddings(positions), emb.LayerNorm, self.qkv_scale_0)
        bias = (1.0 - attention_mask[:, None, None, :].to(hidden.dtype)) * torch.finfo(hidden.dtype).min
        for i, layer in enumerate(self.model.encoder.layer):
            context = self.attention(layer.attention.self, quant, hidden.shape, bias)
            out = layer.attention.output
            hidden, quant = self.normalize(self.project(out.dense, context), hidden, out.LayerNorm,
                                            getattr(self, f"ffn_scale_{i}"), out.dense.bias)
            intermediate = F.gelu(layer.intermediate.dense.matmul(quant).reshape(*hidden.shape[:-1], -1))
            if i + 1 == len(self.model.encoder.layer):
                hidden = layer.output(intermediate, hidden)  # No quantized consumer after final norm.
            else:
                out = layer.output
                hidden, quant = self.normalize(self.project(out.dense, intermediate), hidden, out.LayerNorm,
                                                getattr(self, f"qkv_scale_{i+1}"), out.dense.bias)
        return F.normalize(hidden[:, 0], p=2, dim=-1)
