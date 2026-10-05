"""Isolate scan-local rounding and downstream amplification in the worst pair."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo / '25_clef_inference'))
from run_local_smoke import NoTransformers, encode_record
from run_reranking_smoke import digest, request_for, save
sys.meta_path.insert(0, NoTransformers())
import torch
import torch_npu  # noqa: F401
from torch.nn import functional as F
from torch_npu.npu.npu_config import _CubeMathType
from tokenizers import Tokenizer
import local_modeling_clef as modeling

torch.npu.set_option({'ACL_PRECISION_MODE': 'must_keep_origin_dtype'})
torch.npu.matmul.allow_hf32 = False
torch.npu.conv.allow_hf32 = False
torch.npu.matmul.cube_math_type = _CubeMathType.KEEP_DTYPE
torch.set_float32_matmul_precision('highest')
if not torch.npu.is_available() or '910B' not in torch.npu.get_device_name(0):
    raise RuntimeError('910B required; no CPU fallback')
torch.npu.set_device(0)
torch.set_num_threads(8)
if torch.npu.mem_get_info()[0] < 28 * 1024**3:
    raise RuntimeError('Less than 28 GiB free for model plus diagnostic tensors')

fixture_path = repo / 'tmp/25_clef_inference/reranking_lengths_44ca1ad4/fixture.json'
reference_path = repo / 'tmp/25_clef_inference/gdn_boundary_40_ccf188d6/result.json'
fixture = json.loads(fixture_path.read_text())
reference = json.loads(reference_path.read_text())
assert reference['status'] == 'completed' and reference['fixture_sha256'] == digest(fixture_path)
worst = max(reference['rows'], key=lambda r: abs(r['modes']['whole']['score'] - r['modes']['document_boundary']['score']))
pair = next(p for p in fixture['pairs'] if all(p[k] == worst[k] for k in ('task', 'qid', 'did')))
cut = pair['prefix_tokens']
tokenizer = Tokenizer.from_file(str(args.model / 'tokenizer.json'))
assert digest(args.model / 'tokenizer.json') == fixture['tokenizer_sha256']
tokenizer.no_padding()
tokenizer.no_truncation()
record = encode_record(tokenizer, request_for(pair), max_length=pair['input_tokens'])
assert hashlib.sha256(json.dumps(record.input_ids).encode()).hexdigest() == pair['input_sha256']

source_path = repo / '25_clef_inference/local_modeling_clef.py'
source = source_path.read_text()
fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'chunk_gated_delta_rule')
variant = ast.unparse(fn)
replacements = {
    'chunk_size=64):': 'chunk_size=64, initial_state=None):',
    'padding = -length % chunk_size': 'normalized_qk = (query.transpose(1, 2).clone(), key.transpose(1, 2).clone())\n    padding = -length % chunk_size',
    'state = torch.zeros(batch, heads, key_dim, value_dim, dtype=values.dtype, device=values.device)':
        'state = torch.zeros(batch, heads, key_dim, value_dim, dtype=values.dtype, device=values.device) if initial_state is None else initial_state',
    'return output.transpose(1, 2).to(dtype, memory_format=torch.contiguous_format)':
        'return output.transpose(1, 2).to(dtype, memory_format=torch.contiguous_format), state, output.transpose(1, 2).contiguous(), normalized_qk',
}
for old, new in replacements.items():
    if variant.count(old) != 1:
        raise RuntimeError('Owned scan changed; inspect derivation: ' + old)
    variant = variant.replace(old, new, 1)
namespace = {'torch': torch, 'F': F}
exec(compile(variant, '<GDN scan diagnostic>', 'exec'), namespace)
scan = namespace['chunk_gated_delta_rule']
original = modeling.chunk_gated_delta_rule
active = {'mode': 'warmup', 'index': 0, 'selected': set(), 'restore': False, 'trace': False}
baseline = {}
baseline_scans = {}


def errors(value, ref):
    diff = value.float() - ref.float()
    return {'max_abs': float(diff.abs().max()),
            'relative_rms': float(diff.square().mean().sqrt() / ref.float().square().mean().sqrt().clamp_min(1e-30)),
            'changed_count': int((value != ref).sum()), 'elements': value.numel()}


def traced_scan(q, k, v, g, beta, chunk_size=64):
    index = active['index']
    active['index'] += 1
    tensors = q, k, v, g, beta
    if active['mode'] == 'baseline':
        output = original(*tensors, chunk_size)
        baseline_scans[index] = {'inputs': tuple(t.clone() for t in tensors), 'output': output.clone()}
        return output
    if index not in active['selected']:
        return original(*tensors, chunk_size)
    prefix, state, raw_prefix, prefix_qk = scan(*(t[:, :cut] for t in tensors), chunk_size=chunk_size)
    suffix, _, raw_suffix, suffix_qk = scan(*(t[:, cut:] for t in tensors), initial_state=state, chunk_size=chunk_size)
    split = torch.cat((prefix, suffix), dim=1)
    if active['trace'] or active['restore']:
        whole, _, raw_whole, whole_qk = scan(*tensors, chunk_size=chunk_size)
    if active['trace']:
        raw_split = torch.cat((raw_prefix, raw_suffix), dim=1)
        entry = {'recurrent_index': index, 'backbone_layer': layer_numbers[index],
                 'inputs_vs_baseline': {name: errors(t, b) for name, t, b in zip(('q', 'k', 'v', 'g', 'beta'), tensors, baseline_scans[index]['inputs'])},
                 'normalized_qk_same_input': {name: errors(torch.cat((p, s), dim=1), w) for name, p, s, w in zip(('q', 'k'), prefix_qk, suffix_qk, whole_qk)},
                 'local_raw_fp32': errors(raw_split, raw_whole),
                 'local_bf16_prefix': errors(prefix, whole[:, :cut]),
                 'local_bf16_suffix': errors(suffix, whole[:, cut:]),
                 'output_vs_baseline': errors(split, baseline_scans[index]['output']),
                 'rounding_examples': []}
        # Inspect already-computed NPU values on CPU; no CPU model execution.
        changed = (split != whole).cpu().nonzero()[:3].tolist()
        for coordinate in changed:
            loc = tuple(coordinate)
            entry['rounding_examples'].append({'index': coordinate,
                'whole_fp32': float(raw_whole[loc]), 'split_fp32': float(raw_split[loc]),
                'whole_bf16': float(whole[loc]), 'split_bf16': float(split[loc])})
        active['local_scans'].append(entry)
    return whole if active['restore'] else split


def capture(name):
    def hook(module, inputs, output):
        if active['mode'] == 'baseline':
            baseline[name] = output.detach().clone()
        elif active['trace']:
            active['activations'].append({'module': name, 'error': errors(output, baseline[name])})
    return hook


result = {'status': 'running', 'pair': {k: pair[k] for k in ('task', 'qid', 'did', 'input_tokens', 'prefix_tokens', 'input_sha256')},
          'scope': 'GDN same-input local errors, downstream propagation, and single-layer/cumulative interventions',
          'fixture_sha256': digest(fixture_path), 'reference_sha256': digest(reference_path),
          'model_source_sha256': digest(source_path), 'probe_sha256': digest(Path(__file__)),
          'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
          'physical_npu': os.environ.get('ASCEND_RT_VISIBLE_DEVICES'), 'dtype': 'bfloat16',
          'modes': {}, 'single_layer_splits': [], 'cumulative_splits': []}
handles = []


def run(mode, selected=(), trace=False, restore=False):
    active.update(mode=mode, index=0, selected=set(selected), restore=restore, trace=trace,
                  local_scans=[], activations=[])
    logits = model(ids, record)[0]
    assert active['index'] == 24
    assert logits.dtype == torch.bfloat16 and torch.isfinite(logits).all()
    probabilities = dict(zip(record.questions[0].option_ids, logits.float().softmax(-1).cpu().tolist()))
    value = {'logits': logits.float().cpu().tolist(), 'score': probabilities['true'],
             'split_recurrent_layers': sorted(selected)}
    if trace:
        value.update(local_scans=active['local_scans'], activations=active['activations'])
    print(json.dumps({'mode': mode, 'logits': value['logits'], 'score': value['score']}), flush=True)
    return value


try:
    model = modeling.load_model(args.model, 'npu:0', progress=lambda step: print(step, flush=True))
    layer_numbers = [i for i, layer in enumerate(model.backbone.layers) if layer.kind == 'linear_attention']
    assert len(layer_numbers) == 24
    result['gdn_backbone_layers'] = layer_numbers
    # Record input-side projections/convolution, scan post-processing, decoder
    # outputs (including MLP/residual), and head stages.
    for name, module in model.named_modules():
        if ((name.startswith('backbone.layers.') and name.count('.') == 2)
            or ('.linear_attn.' in name and name.rsplit('.', 1)[-1] in ('in_proj_qkv', 'in_proj_z', 'in_proj_b', 'in_proj_a', 'conv1d', 'norm', 'out_proj'))
            or name in ('backbone.norm', 'head.hidden_norm', 'head.memory_projection', 'head.question_projection',
                        'head.global_projection', 'head.field_norm', 'head.option_norm', 'head.residual_scorer')
            or (name.startswith(('head.evidence_layers.', 'head.layers.')) and name.count('.') == 2)):
            handles.append(module.register_forward_hook(capture(name)))
    ids = torch.tensor([record.input_ids], device='npu:0', dtype=torch.long)
    with torch.inference_mode():
        model(ids, record)
        modeling.chunk_gated_delta_rule = traced_scan
        result['modes']['baseline'] = run('baseline')
        assert result['modes']['baseline']['logits'] == worst['modes']['whole']['logits']
        result['modes']['all_split'] = run('all_split', range(24), trace=True)
        assert result['modes']['all_split']['logits'] == worst['modes']['document_boundary']['logits']
        save(args.output, result)
        result['modes']['restore_scan_outputs'] = run('restore_scan_outputs', range(24), trace=True, restore=True)
        assert result['modes']['restore_scan_outputs']['logits'] == result['modes']['baseline']['logits']
        assert all(a['error']['changed_count'] == 0 for a in result['modes']['restore_scan_outputs']['activations'])
        save(args.output, result)
        for index in range(24):
            value = run('single_' + str(index), [index])
            value['backbone_layer'] = layer_numbers[index]
            result['single_layer_splits'].append(value)
            save(args.output, result)
        for count in range(1, 25):
            value = run('first_' + str(count), range(count))
            result['cumulative_splits'].append(value)
            save(args.output, result)
        result['modes']['baseline_repeat'] = run('baseline_repeat', trace=True)
        assert result['modes']['baseline_repeat']['logits'] == result['modes']['baseline']['logits']
        assert all(a['error']['changed_count'] == 0 for a in result['modes']['baseline_repeat']['activations'])
    result['transformers_imported'] = any(n == 'transformers' or n.startswith('transformers.') for n in sys.modules)
    assert not result['transformers_imported']
    result['status'] = 'completed'
except BaseException as exc:
    result.update(status='failed', error=str(exc))
    raise
finally:
    modeling.chunk_gated_delta_rule = original
    for handle in handles:
        handle.remove()
    save(args.output, result)
