"""Fresh-process cache preload and scoring; no document precomputation."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model', type=Path, required=True)
parser.add_argument('--result', type=Path, required=True)
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
reference = json.loads(args.result.read_text())
assert reference['status'] == 'completed'
assert '910B' in torch.npu.get_device_name(0)
required_gib = 50 if reference['dtype'] == 'float32' else 28
assert torch.npu.mem_get_info()[0] >= required_gib * 1024**3
fixture_path = repo / 'tmp/25_clef_inference/reranking_lengths_44ca1ad4/fixture.json'
assert digest(fixture_path) == reference['fixture_sha256']
fixture = json.loads(fixture_path.read_text())
tokenizer = Tokenizer.from_file(str(args.model / 'tokenizer.json'))
tokenizer.no_padding()
tokenizer.no_truncation()
started = time.perf_counter()
ram = [DocumentCache.load(p['path']) for p in reference['prepared']]
preload_s = time.perf_counter() - started
assert all(t.device.type == 'cpu' for c in ram for t in c.tensors().values())
model = load_model(args.model, 'npu:0', progress=lambda step: print(step, flush=True))
if reference['dtype'] == 'float32':
    model.float()
    torch.npu.empty_cache()
assert all(c.model_identity == model.cache_identity for c in ram)


def forbid_preparation(*args, **kwargs):
    raise AssertionError('Document was recomputed during startup/reuse check')


model.prepare_document = forbid_preparation
seen = []
handle = model.backbone.embed_tokens.register_forward_pre_hook(lambda module, inputs: seen.append(inputs[0].shape[1]))
indices = sorted({0, max(range(len(reference['rows'])), key=lambda i: reference['rows'][i]['input_tokens']),
                  next(i for i, row in enumerate(reference['rows']) if row['task'] == 'MedicalRetrieval' and row['did'] == '80953')})
output = {'status': 'running', 'source_result_sha256': digest(args.result), 'physical_npu': os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
          'dtype': reference['dtype'], 'preloaded_documents': len(ram), 'ram_bytes': sum(c.nbytes for c in ram),
          'preload_s': preload_s, 'disk_read_scope': 'filesystem cache may be warm', 'rows': []}
try:
    with torch.inference_mode():
        for index in indices:
            record = encode_record(tokenizer, request_for(fixture['pairs'][index]), reference['max_length'])
            ids = torch.tensor([record.input_ids], dtype=torch.long, device='npu:0')
            cache = ram[index].to('npu:0')
            logits = model(ids, record, cache)[0].float().cpu().tolist()
            assert logits == reference['rows'][index]['cached']['logits']
            assert seen[-1] == len(record.input_ids) - len(cache.prefix_ids)
            output['rows'].append({'index': index, 'cached_logits_exact': True,
                                   'backbone_tokens_executed': seen[-1], 'input_tokens': len(record.input_ids)})
            del ids, cache
    output['status'] = 'completed'
finally:
    handle.remove()
    save(args.output, output)
print(json.dumps(output), flush=True)
