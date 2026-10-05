"""Real B1 Clef requests with ordinary timing and optional Ascend traces.

FP32 execution, BF16 document-cache storage. The model source is never edited
or copied. Profiler runs are diagnostics, not clean latency measurements.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from profiling import Journal, PipelineProfiler, observe_model, summarize
from run_local_smoke import NoTransformers, encode_document_prefix, encode_record
from run_reranking_smoke import cache_cast, digest, request_for, save

HERE = Path(__file__).resolve().parent
DEFAULT_FIXTURE = HERE.parent / 'tmp/25_clef_inference/reranking_lengths_44ca1ad4/fixture.json'


def main(observer_factory=Journal):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument('--mode', choices=('all', 'prepare', 'uncached', 'cached'), default='all')
    parser.add_argument('--cache-dir', type=Path, help='Existing all-BF16 cache files; required for cached-only runs')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--profile-index', type=int, default=1, help='Zero-based real item captured per phase, after a warmup item')
    args = parser.parse_args()
    if args.mode == 'cached' and args.cache_dir is None:
        parser.error('cached-only mode requires --cache-dir')
    fixture = json.loads(args.fixture.read_text())
    pairs = fixture['pairs']
    if len(pairs) != fixture['expected_pairs'] or fixture['truncation'] != 'none':
        raise ValueError('Fixture contract changed')
    if args.profile and not 1 <= args.profile_index < len(pairs):
        parser.error('--profile-index must have a preceding item and be within the fixture')
    if digest(args.model/'tokenizer.json') != fixture['tokenizer_sha256']:
        raise ValueError('Fixture tokenizer changed')
    args.output_dir.mkdir(parents=True, exist_ok=False)
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
    if not torch.npu.is_available() or '910B' not in torch.npu.get_device_name(0):
        raise RuntimeError('910B required; no CPU model fallback')
    torch.npu.set_device(0)
    torch.set_num_threads(8)
    if torch.npu.mem_get_info()[0] < 50*1024**3:
        raise RuntimeError('Insufficient free NPU memory for FP32 model')
    journal = observer_factory(args.output_dir, profile=args.profile)
    profiler = None
    result = {'status': 'running', 'profile': args.profile, 'mode': args.mode,
        'fixture_sha256': digest(args.fixture), 'model_source_sha256': digest(HERE/'local_modeling_clef.py'),
        'git_commit': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'hostname': platform.node(), 'physical_device': os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
        'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
        'compute_dtype': 'float32', 'cache_storage_dtype': 'bfloat16',
        'timing_contract': 'Whole-item wall time; host submission spans and deferred stream intervals are separate. Nested sections overlap. Profiler runs include instrumentation/export overhead.',
        'workload_scope': fixture['quality_scope'], 'scores': {}, 'phase_memory': {}}
    save(args.output_dir/'command.json', {'argv': sys.argv})
    save(args.output_dir/'workload.json', {'fixture_sha256': digest(args.fixture),
        'pairs': [{k:p[k] for k in ('task','qid','did','input_tokens','prefix_tokens')} for p in pairs]})
    try:
        with journal.item({'phase': 'setup', 'index': -1, 'first_use': True}):
            with journal.section('model_load'):
                model = load_model(args.model, 'npu:0', progress=lambda name: journal.emit('loading', name=name)).float()
                torch.npu.synchronize()  # Setup completion only, never between inference stages.
                torch.npu.empty_cache()
            result['model_cache_identity'] = model.cache_identity
            tokenizer = Tokenizer.from_file(str(args.model/'tokenizer.json'))
            tokenizer.no_padding()
            tokenizer.no_truncation()
            ram = {}
            if args.cache_dir is not None:
                with journal.section('disk_to_ram_preload'):
                    for pair in pairs:
                        prefix = encode_document_prefix(tokenizer, pair['document'])
                        key = hashlib.sha256(json.dumps([model.cache_identity, 'torch.bfloat16', prefix]).encode()).hexdigest()
                        cache = DocumentCache.load(args.cache_dir/(key+'.safetensors'))
                        if cache.prefix_ids != prefix or cache.model_identity != model.cache_identity:
                            raise ValueError('Incompatible saved cache')
                        if any(t.dtype != torch.bfloat16 for t in cache.tensors().values()):
                            raise ValueError('Every stored cache tensor must be BF16')
                        ram[prefix] = cache
                result['ram_preloaded_bytes'] = sum(c.nbytes for c in ram.values())
        phases = ['prepare','uncached','cached'] if args.mode=='all' else [args.mode]
        profiler = PipelineProfiler(args.output_dir, len(pairs), len(phases), args.profile_index, args.profile)
        instrument = observe_model(model, journal) if getattr(journal, 'instrument_model', True) else nullcontext()
        with torch.inference_mode(), instrument:
            for phase in phases:
                torch.npu.reset_peak_memory_stats()
                phase_start = time.perf_counter()
                for index, pair in enumerate(pairs):
                    row = {'phase': phase, 'index': index, 'first_use': index==0,
                           'task': pair['task'], 'qid': pair['qid'], 'did': pair['did'],
                           'input_tokens': pair['input_tokens'], 'prefix_tokens': pair['prefix_tokens'],
                           'suffix_tokens': pair['input_tokens']-pair['prefix_tokens']}
                    with journal.item(row):
                        with journal.section('encode'):
                            request = request_for(pair)
                            prefix = encode_document_prefix(tokenizer, pair['document'])
                            if len(prefix)!=pair['prefix_tokens']:
                                raise ValueError('Frozen prefix length changed')
                            if phase!='prepare':
                                record = encode_record(tokenizer, request, max_length=pair['input_tokens'])
                                if digest_tokens(record.input_ids) != pair['input_sha256']:
                                    raise ValueError('Frozen token sequence changed')
                        if phase=='prepare':
                            with journal.section('document_forward', device=True):
                                cache = model.prepare_document(prefix)
                            with journal.section('cache_to_bf16', device=True):
                                packed = cache_cast(cache, torch.bfloat16)
                            with journal.section('cache_materialize_wait'):
                                cpu = packed.to('cpu')
                            row['cache_bytes'] = cpu.nbytes
                            # RAM-only preparation benchmark: file writing is a separate task.
                            ram[prefix] = cpu
                            del cache, packed, cpu
                        else:
                            with journal.section('input_transfer', device=True):
                                ids = torch.tensor([record.input_ids], dtype=torch.long, device='npu:0')
                            cache = None
                            if phase=='cached':
                                with journal.section('cache_transfer', device=True):
                                    packed = ram[prefix].to('npu:0')
                                with journal.section('cache_expand_fp32', device=True):
                                    cache = cache_cast(packed, torch.float32)
                                del packed
                            with journal.section('model_forward', device=True):
                                logits = model(ids, record, cache=cache)[0]
                            with journal.section('probabilities', device=True):
                                probabilities = logits.float().softmax(-1)
                            with journal.section('output_materialize_wait'):
                                values = logits.float().cpu().tolist()
                                probs = probabilities.cpu().tolist()
                            with journal.section('validate'):
                                if len(values)!=2 or not all(math.isfinite(x) for x in values+probs):
                                    raise ValueError('Invalid logits/probabilities')
                                score = dict(zip(record.questions[0].option_ids, probs))['true']
                                row.update(logits=values, score=score)
                            del ids, cache, logits, probabilities
                    profiler.step()  # Trace export is outside the item's latency.
                result['phase_memory'][phase] = {
                    'allocated_bytes': torch.npu.memory_allocated(), 'reserved_bytes': torch.npu.memory_reserved(),
                    'peak_allocated_bytes': torch.npu.max_memory_allocated(),
                    'peak_reserved_bytes': torch.npu.max_memory_reserved(),
                    'phase_wall_s_including_observation': time.perf_counter()-phase_start}
        if digest(HERE/'local_modeling_clef.py') != result['model_source_sha256']:
            raise AssertionError('Modeling source changed')
        result['status'] = 'completed'
    except BaseException as exc:
        result.update(status='failed', error=repr(exc))
        raise
    finally:
        if profiler:
            profiler.close()
        journal.close()
        result['summary'] = summarize(journal.rows)
        result['pending_device_events'] = len(journal.pending)
        for phase in ('uncached','cached'):
            result['scores'][phase] = [{k:r[k] for k in ('task','qid','did','logits','score')}
                                      for r in journal.rows if r['phase']==phase and 'logits' in r]
        result['total_s'] = time.perf_counter()-journal.start
        result['transformers_imported'] = any(n=='transformers' or n.startswith('transformers.') for n in sys.modules)
        if result['status']=='completed' and (journal.pending or result['transformers_imported']):
            result.update(status='failed', error='Unresolved device events or unexpected Transformers import')
        save(args.output_dir/'result.json', result)
    if result['status']!='completed':
        raise RuntimeError(result.get('error', 'Benchmark failed'))


def digest_tokens(ids):
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


if __name__=='__main__':
    main()
