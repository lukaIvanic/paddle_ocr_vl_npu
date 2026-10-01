"""Full production vision stack with first-block diagnostic outputs, real inputs."""
import argparse
import json
from pathlib import Path
import time
import traceback

import torch
from torch import nn

from local_modeling_colqwen3 import LocalColQwen3, rotate_half
from prepared_prefill import (PreparedVisionStage, prepare_inputs, linear, bmm_attention,
                              gelu_tanh, StageCompiler)
from bench_prepared_prefill import emit, load_case, timed, compare
from run_hf_baseline import sha256


class DiagnosticVisionStage(PreparedVisionStage):
    labels = ('final', 'tap5', 'tap11', 'tap17', 'norm1', 'qkv', 'q_rotary',
              'k_rotary', 'attention', 'out_projection', 'attention_residual',
              'norm2', 'mlp_fc1', 'gelu', 'mlp_fc2', 'block0',
              'qk_sample', 'scaled_qk_sample', 'masked_qk_sample',
              'softmax_fp32_sample', 'softmax_fp16_sample')

    def forward(self, hidden, cos, sin, mask):
        length = hidden.shape[0]
        cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
        taps, diagnostics, attention_samples = [], [], []
        for index, block in enumerate(self.blocks):
            attn = block.attn
            normalized = block.norm1(hidden)
            qkv = linear(attn.qkv, normalized)
            q, k, v = qkv.reshape(length, 3, attn.heads, -1).permute(1, 0, 2, 3).unbind(0)
            qr = (q.float()*cos + rotate_half(q.float())*sin).to(q.dtype)
            kr = (k.float()*cos + rotate_half(k.float())*sin).to(k.dtype)
            q, k, v = [a.transpose(0, 1).unsqueeze(0).contiguous() for a in (qr, kr, v)]
            if index == 0:
                # Same production math; expose every 64th query row across all
                # heads/keys to bound diagnostic output memory on real pages.
                # Extra outputs can inhibit fusion: compare ordinary graph too.
                heads, dim = q.shape[1], q.shape[-1]
                q3, k3, v3 = [a.reshape(heads, length, dim) for a in (q, k, v)]
                qk = torch.bmm(q3, k3.transpose(1, 2)).reshape(1, heads, length, length)
                scaled = qk * attn.scale
                masked = scaled + mask
                probabilities32 = torch.softmax(masked, dim=-1, dtype=torch.float32)
                probabilities16 = probabilities32.to(q.dtype)
                out = torch.bmm(probabilities16.reshape(heads, length, length), v3)
                out = out.reshape(1, heads, length, dim).transpose(1, 2).contiguous().reshape(length, -1)
                attention_samples = [a[:, :, ::64, :].contiguous() for a in
                                     (qk, scaled, masked, probabilities32, probabilities16)]
            else:
                out = bmm_attention(q, k, v, attn.scale, mask).reshape(length, -1)
            projected = linear(attn.proj, out)
            residual = hidden + projected
            norm2 = block.norm2(residual)
            fc1 = linear(block.mlp.linear_fc1, norm2)
            gelu = gelu_tanh(fc1)
            fc2 = linear(block.mlp.linear_fc2, gelu)
            hidden = residual + fc2
            if index == 0:
                diagnostics = [normalized, qkv, qr, kr, out, projected, residual, norm2, fc1, gelu, fc2, hidden]
            if index in self.tap_indices:
                taps.append(hidden)
        return (hidden, taps[0], taps[1], taps[2], *diagnostics, *attention_samples)


@torch.inference_mode()
def run(args, result):
    import torch_npu  # noqa: F401
    torch.npu.set_device('npu:0')
    torch.npu.set_compile_mode(jit_compile=False)
    emit('model_load_start')
    model = LocalColQwen3.from_pretrained(args.model)
    batch, _ = load_case(args.anchor, 0, 'npu:0')
    prepared = prepare_inputs(model, batch)
    stage = DiagnosticVisionStage(model).eval()
    reference, _ = timed(stage, prepared.vision_args)
    production, _ = timed(PreparedVisionStage(model).eval(), prepared.vision_args)
    result['diagnostic_eager_vs_production'] = compare(reference[:4], production)
    assert all(p['exact'] for p in result['diagnostic_eager_vs_production']['parts'])
    compiler = StageCompiler(args.model, args.cache_root, emit)
    compiler.identity['diagnostic_source'] = sha256(Path(__file__))
    call = compiler.get('vision_diagnostic', stage, prepared.vision_args)
    emit('diagnostic_graph_start')
    candidate, elapsed = timed(call, prepared.vision_args)
    result['first_call_s'] = elapsed
    result['outputs'] = {label: compare(a,b) for label,a,b in zip(stage.labels,candidate,reference)}
    result['cache_records'] = compiler.records
    result['status'] = 'completed_diagnostic'
    emit('diagnostic_graph_finish', seconds=elapsed, outputs=result['outputs'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--anchor', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--cache-root', type=Path, default=Path('.runtime_cache/21_colqwen3/prepared'))
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = {'status': 'started', 'scope': 'diagnostic full vision graph; output taps may change fusion'}
    try:
        run(args, result)
    except Exception:
        result.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        (args.output_dir/'result.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
