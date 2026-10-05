"""Roundtrip all FP32 document caches through BF16 disk/RAM storage."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
for name in ('model', 'reference', 'fixture', 'cache-dir', 'output'):
    parser.add_argument('--' + name, type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / '25_clef_inference'))
from run_local_smoke import NoTransformers, encode_record
from run_reranking_smoke import cache_cast, digest, request_for, save, sync_saved
sys.meta_path.insert(0, NoTransformers())
import torch
import torch_npu
from torch_npu.npu.npu_config import _CubeMathType
from tokenizers import Tokenizer
from local_modeling_clef import DocumentCache, load_model

torch.npu.set_option({'ACL_PRECISION_MODE': 'must_keep_origin_dtype'})
torch.npu.matmul.allow_hf32 = False
torch.npu.conv.allow_hf32 = False
torch.npu.matmul.cube_math_type = _CubeMathType.KEEP_DTYPE
torch.set_float32_matmul_precision('highest')
torch.npu.set_device(0)
torch.set_num_threads(8)
assert '910B' in torch.npu.get_device_name(0)
assert torch.npu.mem_get_info()[0] >= 50 * 1024**3
ref = json.loads(args.reference.read_text())
fixture = json.loads(args.fixture.read_text())
assert ref['status'] == 'completed' and ref['dtype'] == 'float32'
assert ref['fixture_sha256'] == digest(args.fixture)
tokenizer = Tokenizer.from_file(str(args.model / 'tokenizer.json'))
tokenizer.no_padding()
tokenizer.no_truncation()
model = load_model(args.model, 'npu:0', progress=lambda x: print(x, flush=True)).float()
torch.npu.empty_cache()
assert model.cache_identity == ref['model_cache_identity']
result = {'status': 'running', 'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
          'reference_sha256': digest(args.reference), 'fixture_sha256': digest(args.fixture),
          'physical_device': os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
          'storage_dtype': 'bfloat16', 'compute_dtype': 'float32', 'rows': []}
try:
    with torch.inference_mode():
        for i, (pair, prepared, original) in enumerate(zip(fixture['pairs'], ref['prepared'], ref['rows'])):
            cpu = DocumentCache.load(prepared['path'])
            assert all(t.dtype == torch.float32 for t in cpu.tensors().values())
            packed = cache_cast(cpu, torch.bfloat16)
            assert packed.nbytes * 2 == cpu.nbytes
            path = args.cache_dir / (packed.key + '.safetensors')
            packed.save(path)
            sync_saved(path)
            loaded = DocumentCache.load(path)
            assert all(t.dtype == torch.bfloat16 and torch.equal(t, packed.tensors()[n]) for n,t in loaded.tensors().items())
            del packed, cpu
            torch.npu.synchronize()
            start = time.perf_counter()
            # Transfer reduced tensors first; expand on NPU, never in host RAM.
            cache = cache_cast(loaded.to('npu:0'), torch.float32)
            torch.npu.synchronize()
            upload_expand_s = time.perf_counter() - start
            assert all(t.dtype == torch.float32 and t.device.type == 'npu' for t in cache.tensors().values())
            record = encode_record(tokenizer, request_for(pair), ref['max_length'])
            ids = torch.tensor([record.input_ids], dtype=torch.long, device='npu:0')
            logits = model(ids, record, cache)[0]
            probs = dict(zip(record.questions[0].option_ids, logits.softmax(-1).cpu().tolist()))
            values = logits.cpu().tolist()
            assert torch.isfinite(logits).all()
            result['rows'].append({'task': pair['task'], 'qid': pair['qid'], 'did': pair['did'],
                'score': probs['true'], 'logits': values,
                'score_abs_diff': abs(probs['true'] - original['cached']['score']),
                'logit_abs_diff': max(abs(a-b) for a,b in zip(values, original['cached']['logits'])),
                'cache_bytes': loaded.nbytes, 'upload_expand_s': upload_expand_s})
            save(args.output, result)
            print(json.dumps({'completed': i+1, **result['rows'][-1]}), flush=True)
            del loaded, cache, ids, logits
    groups = sorted({(r['task'], r['qid']) for r in result['rows']})
    ranks = {}
    for task, qid in groups:
        reduced = sorted((r for r in result['rows'] if (r['task'],r['qid'])==(task,qid)), key=lambda r:(-r['score'],r['did']))
        baseline = sorted((r for r in ref['rows'] if (r['task'],r['qid'])==(task,qid)), key=lambda r:(-r['cached']['score'],r['did']))
        ranks[task+'/'+qid] = {'bf16_storage': [r['did'] for r in reduced], 'fp32_storage': [r['did'] for r in baseline]}
    result.update(status='completed', rankings=ranks,
        max_score_abs_diff=max(r['score_abs_diff'] for r in result['rows']),
        max_logit_abs_diff=max(r['logit_abs_diff'] for r in result['rows']),
        unchanged_rankings=sum(r['bf16_storage']==r['fp32_storage'] for r in ranks.values()))
except BaseException as exc:
    result.update(status='failed', error=str(exc))
    raise
finally:
    save(args.output, result)
print(json.dumps({k:v for k,v in result.items() if k not in ('rows','rankings')}), flush=True)
