"""Fresh-process V2 package control; exact parity and direct device profiles."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from fused_norm_v2 import initialize, FusedBGEM3


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(actual, expected):
    rows = []
    for a, b in zip(actual, expected, strict=True):
        a, b = a.cpu(), b.cpu()
        d = (a.float() - b.float()).abs()
        rows.append({'shape': list(a.shape), 'dtype': str(a.dtype),
                     'exact': torch.equal(a, b), 'max_abs': float(d.max()),
                     'different_elements': int(torch.count_nonzero(d))})
    assert all(r['exact'] for r in rows), rows
    return rows


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--op-api', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--reference', type=Path)
    p.add_argument('--phase', choices=['direct', 'model'], required=True)
    p.add_argument('--model-dir', default='/workspace/model_downloads/bge-m3')
    p.add_argument('--steps', type=int, default=10)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    initialize(args.op_api)
    import torch_npu
    from test_add_layer_norm_quant_v2 import V2
    from profile_w8a8 import capture
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    assert '910B' in torch.npu.get_device_name(0).upper()
    torch.manual_seed(17)
    result = {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'device': torch.npu.get_device_name(0), 'physical_npu': os.getenv('ASCEND_RT_VISIBLE_DEVICES'),
              'custom_opp': os.getenv('ASCEND_CUSTOM_OPP_PATH'), 'library': str(args.op_api),
              'library_sha256': sha(args.op_api), 'phase': args.phase, 'cases': [], 'passed': False}
    def save():
        (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    def retain(label, tensors):
        cpu = [x.cpu() for x in tensors]
        torch.save(cpu, args.output / (label + '.pt'))
        if args.reference:
            return compare(cpu, torch.load(args.reference / (label + '.pt'), weights_only=True))
        return None
    op = V2(args.op_api)
    if args.phase == 'direct':
        for rows in (256, 512, 2048):
            x1, x2 = [torch.randn(rows, 1024).half().npu() for _ in range(2)]
            gamma = (1 + .1 * torch.randn(1, 1024)).half().npu()
            beta, bias = [(.1 * torch.randn(1, 1024)).half().npu() for _ in range(2)]
            elementwise = (.1 * torch.randn(rows, 1024)).half().npu()
            scale = torch.tensor([16.], dtype=torch.float16, device='npu')
            for kind, b in [('none', None), ('broadcast', bias), ('elementwise', elementwise)]:
                label = f'rows{rows}_{kind}'
                inputs = (x1, x2, gamma, beta, scale, b)
                outputs = op(*inputs)
                summed = x1.float().cpu() + x2.float().cpu()
                if b is not None:
                    # The upstream elementwise branch computes (x1+bias)+x2.
                    summed = x1.float().cpu() + b.float().cpu() + x2.float().cpu() if kind == 'elementwise' else summed + b.float().cpu()
                ref = torch.nn.functional.layer_norm(summed, (1024,), gamma.float().cpu().flatten(), beta.float().cpu().flatten())
                torch.testing.assert_close(outputs[0].float().cpu(), ref, atol=.004, rtol=.002)
                qerr = (outputs[1].cpu().int() - (ref*16).round().clamp(-128,127).int()).abs().max().item()
                assert qerr <= 1
                parity = retain(label, outputs)
                profile = capture(lambda: op(*inputs), args.output / 'profiles' / label, label, steps=args.steps)
                row = {'label': label, 'shape': [rows,1024], 'bias': kind, 'baseline_parity': parity,
                       'math_norm_max_abs': float((outputs[0].float().cpu()-ref).abs().max()),
                       'math_int8_max_abs': qerr, 'profile': profile}
                result['cases'].append(row); save()
                print('DIRECT', json.dumps(row), flush=True)
    else:
        from benchmark_w8a8 import compiled_entrypoint
        from download_model import verify
        from run_embedder import Runner
        from w8a8 import calibrate, convert
        from validate_910b import CASES
        root = Path(__file__).resolve().parents[1]
        release = json.loads((root / 'release.json').read_text())
        assert all(verify(Path(args.model_dir)/n, v) for n,v in release['files'].items())
        result['checkpoint_revision'] = release['revision']
        texts = json.loads((root / 'quantization_texts.json').read_text())
        runner = Runner(args.model_dir, max_length=256)
        if args.reference:
            scales = json.loads((args.reference / 'scales.json').read_text())
        else:
            scales = calibrate(runner.model, [runner.tokenize(texts['calibration'][i:i+4]) for i in range(0,12,4)])
        (args.output/'scales.json').write_text(json.dumps(scales, indent=2)+'\n')
        convert(runner.model, scales, 'full_w8a8')
        model = FusedBGEM3(runner.model).eval()
        # Record actual inputs/results at embedding, early, middle and late norms.
        captured = []
        original = model.normalize
        call_index = 0
        def record(x, residual, norm, scale, bias=None):
            nonlocal call_index
            out = original(x, residual, norm, scale, bias)
            if call_index in (0,1,23,47):
                inputs = [x.reshape(-1,1024), residual.reshape(-1,1024), norm.weight.reshape(1,-1),
                          norm.bias.reshape(1,-1), scale, None if bias is None else bias.reshape(1,-1)]
                captured.append({'site': call_index, 'inputs': [None if t is None else t.cpu() for t in inputs],
                                 'eps': norm.eps, 'outputs': [t.reshape(-1,1024).cpu() for t in out]})
            call_index += 1
            return out
        model.normalize = record
        runner.max_length = 128
        model(**runner.tokenize(CASES[0]['texts']))
        model.normalize = original
        torch.save(captured, args.output/'real_inputs.pt')
        if args.reference:
            real_rows=[]
            for c in torch.load(args.reference/'real_inputs.pt', weights_only=True):
                actual = op(*[None if t is None else t.npu() for t in c['inputs']], eps=c['eps'])
                real_rows.append({'site': c['site'], 'parity': compare(actual,c['outputs'])})
            result['real_input_baseline_parity']=real_rows
            print('REAL_INPUT_PARITY',json.dumps(real_rows),flush=True)
        cases=CASES + [{'name':'batch4_full512','length':512,'texts':[t*100 for t in texts['held_out'][:4]]}]
        for case in cases:
            runner.max_length=case['length']; tokens=runner.tokenize(case['texts'])
            eager=model(**tokens).cpu()
            fn=compiled_entrypoint(model,case['name'],args.output/'cache')
            compiled=fn(**tokens).cpu()
            assert torch.isfinite(compiled).all()
            parity=retain(case['name'],[eager,compiled])
            row={'label':case['name'],'shape':list(tokens['input_ids'].shape), 'baseline_parity':parity,
                 'compiled_eager_max_abs':float((compiled-eager).abs().max()),
                 'profile':capture(lambda:fn(**tokens),args.output/'profiles'/case['name'],case['name'],steps=args.steps)}
            result['cases'].append(row);save();print('MODEL',json.dumps(row),flush=True)
    result['passed']=True;save();print('PASS',str(args.output),flush=True)

if __name__ == '__main__':
    main()
