"""Short real-input HF/custom backward, score replay and optimizer controls."""
import argparse
import gc
import json
from pathlib import Path

from distill_runtime import Runtime, read, save
from margin_distillation import margin_loss_and_score_gradient


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    runtime = Runtime(args.model)
    torch = runtime.torch
    import torch_npu
    from transformers import AutoModelForCausalLM
    data = read(args.dataset)
    rows = runtime.records(data['validation'], 'control')
    # Real examples with unequal lengths, small enough for HF eager backward.
    group = next(g for g in data['validation'] if
                 max(len(r['ids']) for r in rows if r['group_id'] == g['id']) <= 512)
    selected = [r for r in rows if r['group_id'] == group['id']][:4]
    assert len({len(r['ids']) for r in selected}) > 1
    target = torch.tensor([2., -.5, 1., -2.], device=runtime.device)
    keys = ['embed_tokens.weight', 'layers.0.self_attn.q_proj.weight',
            'layers.27.self_attn.q_proj.weight']
    result = {'passed': False, 'group_id': group['id'], 'lengths': [len(r['ids']) for r in selected],
              'dtype': 'FP32 parameters, BF16 autocast', 'runs': {}}
    baseline = None
    for mode in ['hf', 'eager', 'fusion_attention', 'replay']:
        if mode == 'hf':
            model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32,
                attn_implementation='eager', local_files_only=True).to(runtime.device).eval()
        else:
            model = runtime.load(args.model, 'fusion_attention' if mode == 'replay' else mode).eval()
        named = dict(model.named_parameters())
        available = [k for k in keys if k in named or 'model.' + k in named]
        assert len(available) >= 2
        if mode == 'replay':
            with torch.no_grad():
                scores = runtime.logits(model, selected)
                loss, gradient = margin_loss_and_score_gradient(scores, target)
            # Same padded shape as joint forward to isolate graph replay from
            # shape-sensitive BF16 arithmetic. Zero other rows' score gradients.
            for i in range(len(selected)):
                z = runtime.logits(model, selected)
                (z[i] * gradient[i]).backward()
        else:
            scores = runtime.logits(model, selected, hf=mode == 'hf')
            loss, _ = margin_loss_and_score_gradient(scores, target)
            loss.backward()
        grads = {k: named[k if k in named else 'model.'+k].grad.detach().float().cpu() for k in available}
        # Keep representative full projection gradients and only meaningful
        # shared embedding rows to avoid zero rows dominating agreement.
        ek = 'embed_tokens.weight'
        if ek in grads:
            active = torch.unique(torch.tensor([i for r in selected for i in r['ids']] + runtime.answers.cpu().tolist()))
            grads[ek] = grads[ek][active]
        metrics = {'loss': loss.item(), 'scores': scores.detach().cpu().tolist(), 'gradients': {}}
        if baseline is not None:
            for k, g in grads.items():
                b = baseline[k]
                metrics['gradients'][k] = {
                    'cosine': torch.nn.functional.cosine_similarity(g.flatten(), b.flatten(), dim=0).item(),
                    'relative_l2': ((g-b).norm() / b.norm().clamp_min(1e-20)).item(),
                    'max_absolute_difference': (g-b).abs().max().item()}
        result['runs'][mode] = metrics
        if mode == 'hf':
            baseline = grads
        if mode == 'fusion_attention':
            fusion_grads = grads
            parameter = named['layers.0.self_attn.q_proj.weight']
            start = parameter.detach().clone()
            gradient = parameter.grad.detach().clone()
        if mode == 'replay':
            result['replay_relative_l2'] = {k: ((g-fusion_grads[k]).norm()/fusion_grads[k].norm().clamp_min(1e-20)).item()
                                          for k, g in grads.items()}
        del model, named, grads
        gc.collect()
        torch.npu.empty_cache()
        print('BACKWARD_CONTROL', mode, json.dumps(metrics), flush=True)
    a = torch.nn.Parameter(start.clone()); b = torch.nn.Parameter(start.clone())
    ordinary = torch.optim.AdamW([a], lr=1e-6, betas=(.9,.999), eps=1e-8, weight_decay=0)
    fused = torch_npu.optim.NpuFusedAdamW([b], lr=1e-6, betas=(.9,.999), eps=1e-8, weight_decay=0)
    a.grad = gradient.clone(); b.grad = gradient.clone()
    ordinary.step(); fused.step()
    result['optimizer_max_parameter_difference'] = (a-b).abs().max().item()
    result['optimizer_max_update'] = (a-start).abs().max().item()
    # Loose enough for BF16 rounding, strict enough to catch disconnected or
    # reversed gradients. Retain values for interpretation, not just pass/fail.
    result['passed'] = (all(v['cosine'] > .98 and v['relative_l2'] < .25
        for mode in ['eager', 'fusion_attention'] for v in result['runs'][mode]['gradients'].values())
        and max(result['replay_relative_l2'].values()) < 1e-4
        and result['optimizer_max_parameter_difference'] < 1e-7)
    save(args.output, result)
    assert result['passed'], json.dumps(result)


if __name__ == '__main__':
    main()
