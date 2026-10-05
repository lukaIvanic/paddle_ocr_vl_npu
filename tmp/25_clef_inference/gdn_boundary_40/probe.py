"""Real-checkpoint GDN boundary check; not complete document-cache reuse.

All full-sequence projection, convolution, attention and head shapes remain
unchanged. Only the recurrent scan splits, carrying its FP32 state unchanged.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--fixture', type=Path, required=True)
parser.add_argument('--model', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--max-length', type=int, required=True)
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
from tokenizers import Tokenizer
import local_modeling_clef as modeling

if not torch.npu.is_available() or '910B' not in torch.npu.get_device_name(0):
    raise RuntimeError('910B required; no CPU fallback')
torch.npu.set_device(0)
torch.set_num_threads(8)
free, _ = torch.npu.mem_get_info()
if free < 26 * 1024**3:
    raise RuntimeError('Less than 26 GiB free before model load')
source_path = repo / '25_clef_inference/local_modeling_clef.py'
source = source_path.read_text()
fn = next(n for n in ast.parse(source).body
          if isinstance(n, ast.FunctionDef) and n.name == 'chunk_gated_delta_rule')
variant = ast.unparse(fn)
# Derive the scan from the actual model: only expose initial/final state.
replacements = {
    'chunk_size=64):': 'chunk_size=64, initial_state=None):',
    'state = torch.zeros(batch, heads, key_dim, value_dim, dtype=values.dtype, device=values.device)':
        'state = torch.zeros(batch, heads, key_dim, value_dim, dtype=values.dtype, device=values.device) if initial_state is None else initial_state',
    'return output.transpose(1, 2).to(dtype, memory_format=torch.contiguous_format)':
        'return output.transpose(1, 2).to(dtype, memory_format=torch.contiguous_format), state',
}
for old, new in replacements.items():
    if variant.count(old) != 1:
        raise RuntimeError('Owned scan changed; inspect state-probe derivation')
    variant = variant.replace(old, new, 1)
namespace = {'torch': torch, 'F': F}
exec(compile(variant, '<GDN initial/final state probe>', 'exec'), namespace)
scan = namespace['chunk_gated_delta_rule']
original = modeling.chunk_gated_delta_rule
active = {'cut': 0, 'calls': 0}


def split_scan(q, k, v, g, beta, chunk_size=64):
    active['calls'] += 1
    cut = active['cut']
    if cut == 0:
        return original(q, k, v, g, beta, chunk_size)
    tensors = q, k, v, g, beta
    prefix, state = scan(*(t[:, :cut] for t in tensors), chunk_size=chunk_size)
    if state.dtype != torch.float32:
        raise AssertionError('Recurrent state must remain FP32')
    suffix, _ = scan(*(t[:, cut:] for t in tensors), chunk_size=chunk_size,
                     initial_state=state)
    return torch.cat((prefix, suffix), dim=1)


fixture = json.loads(args.fixture.read_text())
if len(fixture['pairs']) != 40 or len(fixture['groups']) != 10:
    raise ValueError('Expected frozen ten-query, forty-pair fixture')
if digest(args.model / 'tokenizer.json') != fixture['tokenizer_sha256']:
    raise ValueError('Tokenizer differs from frozen fixture')
tokenizer = Tokenizer.from_file(str(args.model / 'tokenizer.json'))
tokenizer.no_padding()
tokenizer.no_truncation()
requests = [request_for(pair) for pair in fixture['pairs']]
records = [encode_record(tokenizer, request, max_length=args.max_length) for request in requests]
for pair, record in zip(fixture['pairs'], records):
    if len(record.input_ids) != pair['input_tokens'] or hashlib.sha256(json.dumps(record.input_ids).encode()).hexdigest() != pair['input_sha256']:
        raise ValueError('Encoded fixture changed')
result = {'status': 'running', 'scope': 'real_checkpoint_recurrence_boundary_only',
    'complete_document_cache_reuse': False, 'fixture_sha256': digest(args.fixture),
    'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
    'model_source_sha256': digest(source_path), 'probe_sha256': digest(Path(__file__)),
    'derived_scan_sha256': hashlib.sha256(variant.encode()).hexdigest(),
    'physical_npu': os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
    'hostname': platform.node(), 'device_name': torch.npu.get_device_name(0),
    'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
    'dtype': 'bfloat16', 'recurrent_state_dtype': 'float32',
    'max_length': args.max_length, 'rows': []}
save(args.output, result)


def summarize():
    rows = result['rows']
    comparisons = {}
    for mode in ('document_boundary', 'aligned_boundary', 'whole_repeat'):
        diffs = sorted(abs(r['modes'][mode]['score'] - r['modes']['whole']['score']) for r in rows)
        comparisons[mode] = {
            'score_abs_diff_mean': sum(diffs)/len(diffs),
            'score_abs_diff_p50': diffs[(len(diffs)-1)//2],
            'score_abs_diff_p90': diffs[math.ceil(.9*len(diffs))-1],
            'score_abs_diff_p95': diffs[math.ceil(.95*len(diffs))-1],
            'score_abs_diff_max': diffs[-1],
            'logit_abs_diff_max': max(abs(a-b) for r in rows for a,b in zip(r['modes'][mode]['logits'], r['modes']['whole']['logits'])),
            'identical_logit_pairs': sum(r['modes'][mode]['logits'] == r['modes']['whole']['logits'] for r in rows),
        }
    rankings = {}
    for key, group in fixture['groups'].items():
        selected = [r for r in rows if (r['task'], r['qid']) == (group['task'], group['qid'])]
        orders = {mode: [r['did'] for r in sorted(selected, key=lambda r: (-r['modes'][mode]['score'], r['did']))]
                  for mode in ('whole', 'document_boundary', 'aligned_boundary', 'whole_repeat')}
        strict_reversals, ties_changed = [], []
        for i, a in enumerate(selected):
            for b in selected[i+1:]:
                before = a['modes']['whole']['score'] - b['modes']['whole']['score']
                after = a['modes']['document_boundary']['score'] - b['modes']['document_boundary']['score']
                if before * after < 0:
                    strict_reversals.append([a['did'], b['did']])
                if (before == 0) != (after == 0):
                    ties_changed.append([a['did'], b['did']])
        rankings[key] = {'orders': orders, 'document_boundary_order_unchanged': orders['whole'] == orders['document_boundary'],
                         'strict_reversals': strict_reversals, 'ties_changed': ties_changed}
    return {'comparisons': comparisons, 'rankings': rankings,
            'unchanged_document_boundary_rankings': sum(v['document_boundary_order_unchanged'] for v in rankings.values()),
            'queries': len(rankings)}


import math
try:
    model = modeling.load_model(args.model, 'npu:0', progress=lambda step: print(step, flush=True))
    expected_layers = sum(isinstance(m, modeling.GatedDeltaNet) for m in model.modules())
    if expected_layers != 24:
        raise AssertionError('Expected all 24 GDN layers')
    with torch.inference_mode():
        warm = torch.tensor([records[0].input_ids], device='npu:0', dtype=torch.long)
        model(warm, records[0])
        del warm
        for index, (pair, record) in enumerate(zip(fixture['pairs'], records)):
            ids = torch.tensor([record.input_ids], device='npu:0', dtype=torch.long)
            row = {k: pair[k] for k in ('task', 'qid', 'did', 'document_tokens', 'query_tokens', 'prefix_tokens', 'input_tokens', 'input_sha256')}
            row['modes'] = {}
            result['rows'].append(row)
            for mode in ('whole', 'document_boundary', 'aligned_boundary', 'whole_repeat'):
                active.update(cut=pair['prefix_tokens'] if mode == 'document_boundary' else pair['prefix_tokens']//64*64, calls=0)
                modeling.chunk_gated_delta_rule = original if mode in ('whole', 'whole_repeat') else split_scan
                torch.npu.synchronize()
                start = time.perf_counter()
                logits = model(ids, record)[0]
                torch.npu.synchronize()
                elapsed = time.perf_counter() - start
                if logits.dtype != torch.bfloat16 or logits.shape != (2,) or not torch.isfinite(logits).all():
                    raise ValueError('Invalid BF16 noul logits')
                if mode in ('document_boundary', 'aligned_boundary') and active['calls'] != expected_layers:
                    raise AssertionError('Did not split all recurrent layers')
                probabilities = dict(zip(record.questions[0].option_ids, logits.float().softmax(-1).cpu().tolist()))
                row['modes'][mode] = {'logits': logits.float().cpu().tolist(), 'score': probabilities['true'],
                                      'model_s': elapsed, 'split_layer_calls': active['calls'],
                                      'cut': active['cut'] if mode in ('document_boundary', 'aligned_boundary') else None}
                save(args.output, result)
                print(json.dumps({'pair': index+1, 'of': 40, 'mode': mode, **row['modes'][mode]}), flush=True)
                del logits
            del ids
    result['summary'] = summarize()
    result['transformers_imported'] = any(n == 'transformers' or n.startswith('transformers.') for n in sys.modules)
    if result['transformers_imported']:
        raise AssertionError('Transformers imported')
    result['status'] = 'completed'
    print(json.dumps(result['summary']), flush=True)
except BaseException as exc:
    result.update(status='failed', error=str(exc))
    raise
finally:
    modeling.chunk_gated_delta_rule = original
    save(args.output, result)
