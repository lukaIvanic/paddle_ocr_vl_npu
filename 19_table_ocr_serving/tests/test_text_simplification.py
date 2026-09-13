"""CPU control-flow/arithmetic tests, NOT NPU kernel or performance validation.

Compare retained contracts with the validated step-2 source. NPU calls are
simulated on tiny CPU tensors so their order/arguments and KV writes can be
checked without creating a device context or compiling any graph.
"""
from __future__ import annotations

import ast
import copy
from dataclasses import dataclass, replace
import hashlib
import inspect
import io
import json
import os
import queue
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from fixed_architecture_reference import PaddleOCRTextConfig, PaddleOCRVLConfig, text_model, text_call, cache as allocate_cache, FreezeArchitecture

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / '19_table_ocr_serving'
sys.path.insert(0, str(EXPERIMENT))
# Production imports torch_npu once. CPU tests explicitly provide a test double;
# this is not a CPU fallback in the runtime.
sys.modules.setdefault('torch_npu', types.ModuleType('torch_npu'))
import p06_text_prefill_and_decode as current

PIN = 'dc755584'
PATH = '19_table_ocr_serving/text_prefill_and_decode.py'
OLD = subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{PIN}:{PATH}'], text=True)
NEW = (EXPERIMENT / 'p06_text_prefill_and_decode.py').read_text()
NAMES = ('combined_apply_complete_layer_prefetch1_rope_lut_packed_mlp',)
FUNCTIONS = {
    'LocalPaddleOCRVLStaticCache', '_linear_tokenwise', '_packed_linear',
    'prepare_decode_optimization_modules', 'prepare_decode_projections', 'prepare_decode_compact_lm_head',
    'prepare_decode_weight_prefetch', 'prepare_decode_rope_factor_lut',
    'build_static_decode_bool_mask', 'update_decode_kv_cache_',
    '_prepare_multimodal_rotary_factors', '_lookup_scalar_rotary_factors',
    '_project_decode_qkv', '_apply_decode_rotary', '_decode_rms_norm',
    '_decode_add_rms_norm', '_decode_add_with_optional_rms_norm', '_decode_mlp',
    '_decode_attention', 'run_text_decode_transformer', 'TextDecodeStage',
}


def without_methods(source, removals):
    """Allow explicit legacy-method deletions, preserving all remaining source."""
    lines = source.splitlines(True)
    spans = []
    for cls in ast.parse(source).body:
        if isinstance(cls, ast.ClassDef) and cls.name in removals:
            for fn in cls.body:
                if isinstance(fn, ast.FunctionDef) and fn.name in removals[cls.name]:
                    start = min([fn.lineno] + [d.lineno for d in fn.decorator_list]) - 1
                    end = fn.end_lineno
                    while end < len(lines) and not lines[end].strip():
                        end += 1
                    spans.append((start, end))
    for start, end in sorted(spans, reverse=True):
        del lines[start:end]
    return ''.join(lines)


class ChooseSimulatedNPU(ast.NodeTransformer):
    def visit_Attribute(self, node):
        if node.attr == 'type' and isinstance(node.value, ast.Attribute) and node.value.attr == 'device':
            return ast.copy_location(ast.Constant('npu'), node)
        return self.generic_visit(node)


def load_decode(source, name):
    tree = ast.parse(source)
    begin = next((i for i, n in enumerate(tree.body) if isinstance(n, ast.ClassDef) and n.name == 'DecodeOptimizationConfig'), None)
    end = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.ClassDef) and n.name == 'LocalPaddleOCRVLStaticCache')
    module = types.ModuleType(name)
    module.__dict__.update({k:v for k,v in vars(current).items() if k.startswith('TEXT_')})
    module.__dict__.update(torch=torch, nn=torch.nn, dataclass=dataclass, replace=replace,
                           PaddleOCRTextConfig=PaddleOCRTextConfig)
    sys.modules[name] = module
    nodes = tree.body[begin:end] if begin is not None else []
    nodes += [ChooseSimulatedNPU().visit(copy.deepcopy(n)) for n in tree.body[end:]
              if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in FUNCTIONS]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), name, 'exec'), module.__dict__)
    return module


def signature(value):
    if isinstance(value, torch.Tensor):
        data = value.detach().contiguous().numpy().tobytes()
        return (tuple(value.shape), tuple(value.stride()), str(value.dtype), hashlib.sha256(data).hexdigest())
    if isinstance(value, (list, tuple)):
        return tuple(signature(x) for x in value)
    if isinstance(value, dict):
        return tuple((k, signature(v)) for k, v in sorted(value.items()))
    return value


class SimulatedNPU(types.ModuleType):
    def __init__(self):
        super().__init__('torch_npu')
        self.events = []

    def record(self, operation, *args, **kwargs):
        self.events.append((operation, signature(args), signature(kwargs)))

    def npu_prefetch(self, *args):
        self.record('prefetch', *args)

    def npu_rms_norm(self, x, weight, eps):
        self.record('rms_norm', x, weight, eps)
        rstd = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
        return (x.float() * rstd).to(x.dtype) * weight, rstd

    def npu_add_rms_norm(self, x, residual, weight, eps):
        self.record('add_rms_norm', x, residual, weight, eps)
        summed = x + residual
        rstd = torch.rsqrt(summed.float().square().mean(-1, keepdim=True) + eps)
        return (summed.float() * rstd).to(x.dtype) * weight, rstd, summed

    def npu_apply_rotary_pos_emb(self, q, k, cos, sin, **kwargs):
        self.record('apply_rotary', q, k, cos, sin, **kwargs)
        def rotate(x):
            first, second = x.chunk(2, -1)
            return x * cos + torch.cat((-second, first), -1) * sin
        return rotate(q), rotate(k)

    def npu_swiglu(self, x, dim):
        self.record('swiglu', x, dim)
        gate, up = x.chunk(2, dim)
        return torch.nn.functional.silu(gate) * up

    def scatter_update_(self, cache, positions, values, axis):
        self.record('scatter', cache, positions, values, axis)
        for b, position in enumerate(positions.tolist()):
            cache[b, :, position:position + 1, :] = values[b]
        return cache

    def npu_incre_flash_attention(self, q, k, v, **kwargs):
        self.record('increfa', q, k, v, **kwargs)
        assert kwargs['input_layout'] == 'BNSD'
        assert kwargs['pse_shift'] is None and kwargs['actual_seq_lengths'] is None
        assert 'inner_precise' not in kwargs
        groups = kwargs['num_heads'] // kwargs['num_key_value_heads']
        k, v = k.repeat_interleave(groups, 1), v.repeat_interleave(groups, 1)
        scores = (q @ k.transpose(-1, -2)) * kwargs['scale_value']
        scores = scores.masked_fill(kwargs['atten_mask'], torch.finfo(scores.dtype).min)
        return torch.softmax(scores.float(), -1).to(q.dtype) @ v


def exercise(module, contract, batch, compact):
    torch.manual_seed(1729)
    config = PaddleOCRTextConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=32, rope_parameters={'rope_theta': 500000., 'mrope_section': [1, 1, 2]})
    model = torch.nn.Module()
    model.config = types.SimpleNamespace(text_config=config)
    # This prefill/model-class implementation is independently checked unchanged.
    model.model = text_model(current, config)
    model.lm_head = torch.nn.Linear(32, 64, bias=False)
    legacy = hasattr(module, 'resolve_decode_optimization')
    if legacy:
        module.prepare_decode_optimization_modules(model, contract)
    else:
        module.prepare_decode_projections(model)
    if compact:
        module.prepare_decode_compact_lm_head(model, (3, 11, 18, 24, 32, 41, 52, 63))
    args = (model, contract) if legacy else (model,)
    module.prepare_decode_rope_factor_lut(*args, cache_length=32, dtype=torch.float32)
    module.prepare_decode_weight_prefetch(*args)
    cache = allocate_cache(module, config, batch_size=batch, cache_length=32,
        device=torch.device('cpu'), dtype=torch.float32)
    for tensor in cache.flat_tensors():
        tensor.copy_(torch.randn_like(tensor) * .1)
    inputs = torch.arange(batch).view(-1, 1) + 4
    positions = torch.arange(batch) + 3
    deltas = torch.ones(batch, 1, dtype=torch.int64)
    fake = SimulatedNPU()
    module.torch_npu = fake
    for name, child in model.named_modules():
        if isinstance(child, torch.nn.Linear):
            child.register_forward_pre_hook(lambda layer, args, name=name: fake.record('linear:' + name, args, layer.weight))
    stage = text_call(module, config, module.TextDecodeStage, *args, **({'cache_length': 32} if legacy else {}))
    outputs = []
    with patch.dict(sys.modules, {'torch_npu': fake}), torch.inference_mode():
        for _ in range(2):
            out = stage(inputs, positions, deltas, *cache.flat_tensors())
            outputs.append(out.clone())
            inputs = out if compact or not legacy else out[:, -1].argmax(-1).view(-1, 1)
            positions = positions + 1
    return outputs, tuple(x.clone() for x in cache.flat_tensors()), fake.events


class TextSimplificationTests(unittest.TestCase):
    def test_nz_conversion_aborts_instead_of_falling_back(self):
        class Linear:
            def __init__(self, fmt=2, device='npu'):
                self.weight = types.SimpleNamespace(
                    device=types.SimpleNamespace(type=device),
                    data=types.SimpleNamespace(format=fmt))

        for mode in ('success', 'already_nz', 'wrong_format', 'cast_error',
                     'partial_failure', 'non_npu'):
            with self.subTest(mode=mode):
                first = Linear(29 if mode == 'already_nz' else 2,
                               'cpu' if mode == 'non_npu' else 'npu')
                head = Linear()
                model = types.SimpleNamespace(
                    model=types.SimpleNamespace(named_modules=lambda: [('projection', first)]),
                    lm_head=head)
                calls = []

                def cast(data, fmt):
                    calls.append(fmt)
                    if mode == 'cast_error' or (mode == 'partial_failure' and len(calls) == 2):
                        raise ValueError('simulated cast failure')
                    return types.SimpleNamespace(format=2 if mode == 'wrong_format' else fmt)

                fake = types.SimpleNamespace(
                    get_npu_format=lambda w: w.data.format, npu_format_cast=cast)
                with patch.object(current.nn, 'Linear', Linear), patch.object(current, 'torch_npu', fake):
                    if mode in ('success', 'already_nz'):
                        report = current.cast_decode_linear_weights_to_nz(model)
                        self.assertTrue(report['all_after_are_nz'])
                        self.assertEqual(first.weight.data.format, 29)
                        self.assertEqual(head.weight.data.format, 29)
                        self.assertEqual(len(calls), 1 if mode == 'already_nz' else 2)
                    else:
                        name = 'lm_head' if mode == 'partial_failure' else 'projection'
                        with self.assertRaisesRegex(RuntimeError, name) as raised:
                            current.cast_decode_linear_weights_to_nz(model)
                        if mode in ('cast_error', 'partial_failure'):
                            self.assertIsInstance(raised.exception.__cause__, ValueError)
                        self.assertEqual(len(calls), 0 if mode == 'non_npu' else 2 if mode == 'partial_failure' else 1)

    def test_dead_decode_plumbing_removed(self):
        self.assertNotIn('cache_length', inspect.signature(current.TextDecodeStage).parameters)
        self.assertNotIn('use_scatter_pa', inspect.signature(current.update_decode_kv_cache_).parameters)
        parameters = inspect.signature(current._decode_attention).parameters
        for name in ('position_embeddings', 'pse_shift', 'actual_seq_lengths'):
            self.assertNotIn(name, parameters)
        for name in ('_decode_prefetch_current_mlp', '_decode_prefetch_next_attention'):
            self.assertNotIn(name, NEW)
        fn = next(n for n in ast.parse(NEW).body if isinstance(n, ast.FunctionDef) and n.name == '_decode_attention')
        call = next(n for n in ast.walk(fn) if isinstance(n, ast.Call)
                    and ast.unparse(n.func) == 'torch_npu.npu_incre_flash_attention')
        kwargs = {kw.arg: kw.value for kw in call.keywords}
        self.assertIsNone(ast.literal_eval(kwargs['pse_shift']))
        self.assertIsNone(ast.literal_eval(kwargs['actual_seq_lengths']))
        runtime = ast.parse((EXPERIMENT / 'p02_serving_runtime.py').read_text())
        configuration = next(n for n in ast.walk(runtime) if isinstance(n, ast.FunctionDef) and n.name == 'configuration')
        result = next(n.value for n in configuration.body if isinstance(n, ast.Return))
        metadata = {ast.literal_eval(k): v for k, v in zip(result.keys, result.values)}
        self.assertNotIn('decode_attention', metadata)
        self.assertNotIn('decode_cache_update', metadata)
        for name in ('DECODE_ATTENTION', 'DECODE_CACHE_UPDATE',
                     'decode_attention_label', 'decode_cache_update_label',
                     'decode_native_fallback', 'decode_mixed_format'):
            self.assertNotIn(name, NEW)
        update = ast.parse(inspect.getsource(current.update_decode_kv_cache_))
        self.assertFalse(any(isinstance(n, ast.If) for n in ast.walk(update)))
        self.assertEqual(ast.literal_eval(metadata['vision_prompt_fa_layout']), 'bnsd')

    def test_fixed_settings_against_saved_readiness(self):
        import p02_serving_runtime as runtime
        import p05_vision_prefill as vision_prefill
        recorded = json.loads((ROOT / 'tmp/19_table_ocr_serving/step2_b8qps6_dc755584_20260910_cached/b8/ready.json').read_text())['configuration']
        # The fixed serving settings are module constants; compare them with the
        # readiness record of the validated run. No checkpoint or device is opened.
        self.assertEqual(runtime.CACHE_LENGTH, recorded['cache_length'])
        self.assertEqual(runtime.MAX_NEW_TOKENS, recorded['max_new_tokens'])
        self.assertEqual(str(runtime.DTYPE), recorded['dtype'])
        self.assertEqual(recorded['decode_backend'], 'torchair')
        self.assertFalse(recorded['compact_decode_control'])
        self.assertEqual(list(vision_prefill.VISION_BUCKETS), recorded['vision_prefill']['buckets'])
        self.assertEqual(list(current.TEXT_PREFILL_BUCKETS), recorded['text_prefill']['buckets'])
        self.assertEqual(vision_prefill.VISION_SEQUENCE_ALIGNMENT, recorded['vision_prefill']['sequence_alignment'])
        padding_source = ast.parse(inspect.getsource(vision_prefill.prepare_vision_mlp_intermediate))
        target = next(n.value for n in ast.walk(padding_source) if isinstance(n,ast.Assign)
                      and any(isinstance(t,ast.Name) and t.id=='target' for t in n.targets))
        self.assertEqual(ast.literal_eval(target), recorded['vision_prefill']['mlp_intermediate_size'])
        for kind in ('min', 'max'):
            self.assertEqual(getattr(runtime, kind.upper() + '_PIXELS'), recorded['preprocessor']['effective_' + kind + '_pixels'])
        _, vocab = current.load_decode_vocab_token_ids(runtime.DECODE_VOCAB_TOKEN_IDS_PATH, full_vocab_size=103424)
        # Vocabulary intentionally changed after the mechanical-copy anchor.
        # Keep all the original runtime checks above, but anchor this decision
        # to the completed 60k NPU run, not the old 16k readiness record.
        head_record = json.loads((ROOT / 'tmp/19_table_ocr_serving/lm_head_60416_20260911/b8_expanded_measured/b8/ready.json').read_text())['configuration']['decode_vocab']
        self.assertEqual(vocab['token_ids_sha256'], head_record['token_ids_sha256'])
        self.assertEqual(vocab['selected_vocab_size'], head_record['selected_vocab_size'])
        # This is now the sole implementation, anchored by the combined run.
        preprocessing_record = json.loads((ROOT / 'tmp/19_table_ocr_serving/preprocess_options_20260911/b8_both_measured/b8/ready.json').read_text())['configuration']['preprocessing_benchmark']
        self.assertEqual(preprocessing_record, {'resize_backend': 'kornia_rs', 'compact_uint8': True})
        from p03_crop_processing import preprocess_pil_image
        parameters = inspect.signature(preprocess_pil_image).parameters
        self.assertEqual(set(parameters), {'image'})
        for name in ('compact_uint8_preprocess', 'image_resize_backend'):
            self.assertNotIn(name, inspect.signature(runtime.ContinuousRecognizer).parameters)

    def test_http_worker_cli_and_request_wiring(self):
        import pickle
        import p01_serve as serve
        import p02_serving_runtime as runtime
        explicit_paths = {
            '--model-path': '/chosen/model',
            '--graph-cache-directory': '/chosen/graphs',
            '--log-folder': '/chosen/logs',
        }
        path_args = [value for pair in explicit_paths.items() for value in pair]
        with patch.object(sys, 'argv', ['p01_serve.py', *path_args]):
            args = serve.parse_args()
        self.assertIsInstance(args, serve.ServeConfig)
        self.assertEqual(pickle.loads(pickle.dumps(args)), args)
        self.assertEqual(args, serve.ServeConfig(
            model_path=Path('/chosen/model'), graph_cache_directory=Path('/chosen/graphs'),
            log_folder=Path('/chosen/logs')))
        for flag, value in explicit_paths.items():
            self.assertEqual(getattr(args, flag[2:].replace('-', '_')), Path(value))
            remaining = [item for pair in explicit_paths.items() if pair[0] != flag for item in pair]
            with patch.object(sys, 'argv', ['p01_serve.py', *remaining]), \
                 patch('sys.stderr', new=io.StringIO()) as error, self.assertRaises(SystemExit) as exit:
                serve.parse_args()
            self.assertEqual(exit.exception.code, 2)
            self.assertIn(flag, error.getvalue())
        for name in ('HERE', 'EXPERIMENT_ROOT', 'REPO_ROOT'):
            self.assertFalse(hasattr(serve, name))
        expected_args = {'host', 'port', 'request_timeout_s', 'max_image_bytes', 'queue_capacity', 'run_eagerly',
            'full_decode_lm_head',
            'model_path', 'device', 'decode_batch_size', 'metrics_level',
            'graph_cache_directory', 'log_folder'}
        self.assertEqual(set(vars(args)), expected_args)
        self.assertEqual(serve.PROMPTS, {'table': 'Table Recognition:'})
        self.assertEqual(args.metrics_level, 'scheduling')
        self.assertFalse(args.full_decode_lm_head)
        with patch.object(sys, 'argv', ['p01_serve.py', *path_args, '--expanded-decode-lm-head']), \
             patch('sys.stderr',new=io.StringIO()), self.assertRaises(SystemExit):
            serve.parse_args()
        with patch.object(sys, 'argv', ['p01_serve.py', *path_args, '--full-decode-lm-head']):
            self.assertTrue(serve.parse_args().full_decode_lm_head)
        for level in ('basic', 'scheduling', 'detailed'):
            with patch.object(sys, 'argv', ['p01_serve.py', *path_args, '--metrics-level', level]):
                self.assertEqual(serve.parse_args().metrics_level, level)
        for flags in (['--metrics-level', 'unknown'], ['--decode-device-timing'],
                      ['--request-scheduling-metrics'], ['--torchair-cache-dir', '/old/cache']):
            with patch.object(sys, 'argv', ['p01_serve.py', *path_args, *flags]), \
                 patch('sys.stderr', new=io.StringIO()), self.assertRaises(SystemExit):
                serve.parse_args()
        with patch.object(sys, 'argv', ['p01_serve.py', *path_args,
                '--host', '0.0.0.0', '--port', '9001', '--queue-capacity', '10',
                '--request-timeout-s', '30', '--max-image-bytes', '1234',
                '--device', 'npu:1', '--decode-batch-size', '8', '--run-eagerly',
                '--metrics-level', 'basic']):
            overridden = serve.parse_args()
        self.assertEqual(overridden, serve.ServeConfig(
            model_path=args.model_path, graph_cache_directory=args.graph_cache_directory,
            log_folder=args.log_folder,
            host='0.0.0.0', port=9001, queue_capacity=10, request_timeout_s=30,
            max_image_bytes=1234, device='npu:1', decode_batch_size=8, run_eagerly=True,
            metrics_level='basic'))
        seen = []
        test = self
        constructor_signature = inspect.signature(runtime.ContinuousRecognizer)
        serve_signature = inspect.signature(runtime.ContinuousRecognizer.serve)
        @dataclass
        class Result:
            request_id: str
            text: str
        @dataclass
        class Summary:
            requests: int
        class FakeRecognizer:
            preprocessor_min_pixels_override = 28224
            preprocessor_max_pixels_override = 802816
            def __init__(self, **kwargs):
                constructor_signature.bind(**kwargs)
                seen.append(kwargs)
            def configuration(self):
                return {}
            def serve(self, source, **kwargs):
                serve_signature.bind(self, source, **kwargs)
                test.assertEqual(kwargs['collect_scheduling_metrics'], level != 'basic')
                request = source.pull(block=False)
                test.assertFalse(hasattr(request, 'min_pixels'))
                test.assertFalse(hasattr(request, 'max_pixels'))
                test.assertEqual(request.prompt, 'Table Recognition:')
                kwargs['emit_result'](Result(request.request_id, '<table></table>'))
                test.assertIsNone(source.pull(block=False))
                test.assertTrue(source.closed)
                return Summary(1)
        for level in ('basic', 'scheduling', 'detailed'):
            jobs, results = queue.Queue(), queue.Queue()
            jobs.put(dict(request_id='test', image_bytes=b'not-decoded-in-this-test', prompt='Table Recognition:',
                          crop_type='table', submitted_monotonic_s=0.0))
            jobs.put(None)
            cfg = serve.ServeConfig(model_path=Path('/unused'), decode_batch_size=8,
                       full_decode_lm_head=True, metrics_level=level,
                       graph_cache_directory=Path('/unused/graphs'), log_folder=Path('/unused/logs'))
            with patch.object(runtime, 'ContinuousRecognizer', FakeRecognizer), \
                 patch.object(serve, '_freeze_setup_gc', return_value={'enabled': True}) as freeze, \
                 patch('sys.stdout', new=io.StringIO()):
                serve.run_inference_process(jobs, results, cfg)
            messages = []
            while not results.empty(): messages.append(results.get())
            self.assertEqual([m['kind'] for m in messages], ['ready', 'result', 'service_summary'], messages)
            self.assertTrue(messages[1]['ok'])
            self.assertEqual(messages[0]['configuration']['metrics_level'], level)
            self.assertFalse(seen[-1]['eager'])
            self.assertTrue(seen[-1]['full_decode_lm_head'])
            self.assertEqual(seen[-1]['decode_device_timing'], level == 'detailed')
            self.assertEqual(seen[-1]['model'], '/unused')
            self.assertEqual(seen[-1]['graph_cache_directory'], Path('/unused/graphs'))
            self.assertNotIn('torchair_cache_dir', seen[-1])
            freeze.assert_called_once()

    def test_full_head_setup_and_cache_separation(self):
        import p02_serving_runtime as runtime
        cache_keys = []
        for full in (False,True):
            model = torch.nn.Module()
            model.lm_head = torch.nn.Linear(4,64,bias=False)
            owner = types.SimpleNamespace(model=model,full_decode_lm_head=full)
            metadata = dict(enabled=True,selected_vocab_size=3,token_ids_sha256='abc123456789ffff')
            load = unittest.mock.Mock(return_value=((3,11,63),metadata))
            with torch.inference_mode(), patch.object(runtime,'load_decode_vocab_token_ids',load):
                cache_keys.append(runtime.ContinuousRecognizer._prepare_decode_lm_head(owner))
            self.assertEqual(owner.decode_vocab['enabled'],not full)
            if full:
                load.assert_not_called()
                self.assertFalse(hasattr(model,'decode_lm_head'))
                self.assertFalse(hasattr(model,'decode_token_id_map'))
                self.assertEqual(owner.decode_vocab['selected_vocab_size'],64)
            else:
                load.assert_called_once_with(runtime.DECODE_VOCAB_TOKEN_IDS_PATH,full_vocab_size=64)
                self.assertTrue(torch.equal(model.decode_lm_head.weight,model.lm_head.weight[[3,11,63]]))
                self.assertEqual(model.decode_token_id_map.tolist(),[3,11,63])
        self.assertEqual(cache_keys,['selected_vocab_3_abc123456789','full_vocab_64'])
        # The decode graph cache is keyed by the selected head.
        tree = ast.parse((EXPERIMENT / 'p02_serving_runtime.py').read_text())
        cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name == 'ContinuousRecognizer')
        init = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name == '__init__')
        stages = next(n for n in ast.walk(init) if isinstance(n,ast.Call)
                      and ast.unparse(n.func) == 'self.model.make_inference_stages')
        head = next(k.value for k in stages.keywords if k.arg == 'decode_head_cache_key')
        self.assertEqual(ast.unparse(head), 'decode_head_cache_key')
        cache_root = next(k.value for k in stages.keywords if k.arg == 'graph_cache_directory')
        self.assertEqual(ast.unparse(cache_root), 'graph_cache_directory')

    @classmethod
    def setUpClass(cls):
        cls.old = load_decode(OLD, '_step2_decode_control')
        cls.new = load_decode(NEW, '_simplified_decode_control')
        torch.set_num_threads(1)

    def test_retained_contracts_and_rejection(self):
        for name in ('DecodeOptimizationConfig', 'DECODE_OPTIMIZATION_PRESETS', 'resolve_decode_optimization', 'decode_optimization_names'):
            self.assertFalse(hasattr(self.new, name), name)
        import inspect
        self.assertNotIn('optimization', inspect.signature(self.new.TextDecodeStage).parameters)
        self.assertNotIn('init_mode', inspect.signature(self.new.LocalPaddleOCRVLStaticCache.allocate).parameters)

    def test_outputs_kv_and_operation_trace(self):
        for contract in NAMES:
            for batch in (1, 2, 8):
                for compact in (False, True):
                    with self.subTest(contract=contract, batch=batch, compact=compact):
                        a, ka, events_a = exercise(self.old, contract, batch, compact)
                        b, kb, events_b = exercise(self.new, contract, batch, compact)
                        self.assertEqual(events_a, events_b)
                        if not compact:
                            # The legacy full head returned logits. The new
                            # serving boundary returns their greedy native IDs.
                            a = [torch.argmax(x[:, -1, :].float(), dim=-1, keepdim=True) for x in a]
                        self.assertTrue(all(x.shape == (batch, 1) and x.dtype == torch.int64 for x in b))
                        self.assertTrue(all(torch.equal(x, y) for x, y in zip(a, b)))
                        self.assertTrue(all(torch.equal(x, y) for x, y in zip(ka, kb)))

    def test_prefill_softmax_contract(self):
        self.assertFalse(hasattr(current, 'get_text_softmax_dtype_mode'))
        marker = '# ---- Relocated text-prefill implementation (unchanged computation) ----'
        old = types.ModuleType('_step2_prefill')
        old.__dict__.update(current.__dict__)
        old.PaddleOCRTextConfig = PaddleOCRTextConfig
        old.__dict__.update(os=os, TEXT_SOFTMAX_DTYPE_ENV='PADDLE_OCR_VL_TEXT_SOFTMAX_DTYPE',
                            SOFTMAX_DTYPE_CHOICES=('fp32', 'model'))
        nodes = [n for n in ast.parse(OLD[OLD.index(marker):]).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), 'old_prefill', 'exec'), old.__dict__)
        for dtype in (torch.float16, torch.float32):
            results = []
            for module in (old, current):
                torch.manual_seed(1729)
                cfg = PaddleOCRTextConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                    max_position_embeddings=32, rope_parameters={'rope_theta': 500000., 'mrope_section': [1, 1, 2]})
                model = torch.nn.Module()
                model.config = types.SimpleNamespace(text_config=cfg)
                model.model = text_model(module, cfg).to(dtype)
                stage = text_call(module, cfg, module.TextPrefillStage, model)
                cache = allocate_cache(current, cfg, batch_size=1, cache_length=32,
                    device=torch.device('cpu'), dtype=dtype)
                hidden = torch.randn(1, 8, 32).to(dtype)
                mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]])
                positions = torch.arange(8).view(1, 1, 8).expand(3, 1, 8)
                with patch.dict(os.environ, {'PADDLE_OCR_VL_TEXT_SOFTMAX_DTYPE': 'fp32'}), torch.inference_mode():
                    out = stage(hidden, mask, positions, torch.tensor([4]), *cache.flat_tensors())
                results.append((signature(out), signature(cache.flat_tensors())))
            self.assertEqual(*results)

    def test_model_composition_and_serving_wiring(self):
        import p04_paddle_ocr_vl_1_6_modeling as modeling
        path = '19_table_ocr_serving/paddle_ocr_vl_1_6_modeling.py'
        source = subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{PIN}:{path}'], text=True)
        removed = {'get_image_features', 'build_inputs_embeds', 'forward_static_prefill',
            'forward_static_decode', 'forward', 'generate_ids', 'generate_ids_static'}
        expected = without_methods(source, {'LocalPaddleOCRVLForConditionalGeneration': removed})
        expected = expected.replace('        vision_attention: str,\n', '').replace('            attention_impl=vision_attention,\n', '')
        # The image-loop counter is positive at every check; only its dead
        # fallback and redundant bookkeeping are removed.
        expected = expected.replace('                remain_images = image_nums\n', '')
        expected = expected.replace(' if remain_images > 0 else len(input_tokens) + 1', '')
        expected = expected.replace('                    remain_images -= 1\n', '')
        def defs(src):
            return {n.name: ast.get_source_segment(src, n) for n in ast.parse(src).body
                    if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        old, new = defs(expected), defs((EXPERIMENT / 'p04_paddle_ocr_vl_1_6_modeling.py').read_text())
        self.assertEqual(old.keys() - {'LocalModelOutput', 'LocalStaticModelOutput', '_resolve_model_dir'}, new.keys() - {'model_source_hash'})
        self.assertNotIn('_resolve_model_dir', new)
        for name in new.keys() - {'LocalPaddleOCRVLForConditionalGeneration', 'model_source_hash'}:
            self.assertEqual(old[name], new[name], name)
        def methods(src):
            cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == 'LocalPaddleOCRVLForConditionalGeneration')
            return {n.name: ast.get_source_segment(src,n) for n in cls.body if isinstance(n,ast.FunctionDef)}
        a,b = methods(expected),methods((EXPERIMENT/'p04_paddle_ocr_vl_1_6_modeling.py').read_text())
        local_loader = a['from_pretrained'].replace(
            'model_id_or_path: str | Path = "PaddlePaddle/PaddleOCR-VL-1.6"',
            'model_dir: str | Path').replace(
            'model_dir = _resolve_model_dir(model_id_or_path)',
            'model_dir = Path(model_dir).expanduser()')
        local_loader = local_loader.replace('        config = PaddleOCRVLConfig.from_model_dir(model_dir)\n        model = cls(config)', '        model = cls()')
        self.assertEqual(local_loader, b['from_pretrained'])
        for name in a.keys()-{'make_inference_stages','allocate_static_cache','from_pretrained'}:
            expected_method = FreezeArchitecture().visit(ast.parse(a[name]))
            self.assertEqual(ast.dump(expected_method), ast.dump(ast.parse(b[name])), name)
        for name in removed:
            self.assertNotIn(name, vars(modeling.LocalPaddleOCRVLForConditionalGeneration))
        # Constructor source above is identical, preserving checkpoint hierarchy.
        cfg = PaddleOCRVLConfig.from_dict({
            'vision_config': {'hidden_size': 144, 'num_attention_heads': 2, 'num_hidden_layers': 2, 'intermediate_size': 272},
            'text_config': {'hidden_size': 32, 'num_attention_heads': 4, 'num_key_value_heads': 2,
                'head_dim': 8, 'num_hidden_layers': 2, 'intermediate_size': 64, 'vocab_size': 64,
                'rope_parameters': {'rope_theta': 500000., 'mrope_section': [1, 1, 2]}}})
        old_class = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'LocalPaddleOCRVLForConditionalGeneration')
        scope = dict(vars(modeling))
        exec(compile(ast.Module(body=[old_class], type_ignores=[]), 'old_model', 'exec'), scope)
        a = scope['LocalPaddleOCRVLForConditionalGeneration'].__new__(scope['LocalPaddleOCRVLForConditionalGeneration'])
        torch.nn.Module.__init__(a)
        a.config = cfg
        b = modeling.LocalPaddleOCRVLForConditionalGeneration.__new__(modeling.LocalPaddleOCRVLForConditionalGeneration)
        torch.nn.Module.__init__(b)
        # Compare actual CPU position construction, including generic cases
        # beyond the serving endpoint's one-image request contract.
        merge = cfg.vision_config.spatial_merge_size
        for image_count in (0, 1, 2):
            tokens = [7]
            for _ in range(image_count):
                tokens += [cfg.vision_start_token_id] + [cfg.image_token_id] * 4 + [8]
            ids = torch.tensor([tokens])
            grid = torch.tensor([[1, 2 * merge, 2 * merge]] * image_count).reshape(-1, 3)
            for masked in (False, True):
                inputs = torch.cat((torch.zeros((1, 2), dtype=ids.dtype), ids), dim=1) if masked else ids
                mask = torch.cat((torch.zeros((1, 2), dtype=ids.dtype), torch.ones_like(ids)), dim=1) if masked else None
                for images in (grid, None) if image_count == 0 else (grid,):
                    with self.subTest(image_count=image_count, masked=masked, grid_present=images is not None):
                        before = a.get_rope_index(inputs, images, mask)
                        after = b.get_rope_index(inputs, images, mask)
                        for x, y in zip(before, after):
                            self.assertTrue(torch.equal(x, y))


    def test_zero_cache_matches_reference(self):
        cfg=PaddleOCRTextConfig(num_hidden_layers=2,num_key_value_heads=2,head_dim=8)
        for batch in (1,2,8):
            kwargs=dict(batch_size=batch,cache_length=32,device=torch.device('cpu'),dtype=torch.float16)
            old=self.old.LocalPaddleOCRVLStaticCache.allocate(cfg,init_mode='zeros',**kwargs)
            new=allocate_cache(self.new, cfg,**kwargs)
            self.assertEqual(signature(old.flat_tensors()),signature(new.flat_tensors()))

    def test_fixed_text_buckets_and_preparation(self):
        # Preserve every route, including actual overflow, not just the corpus.
        old = types.ModuleType('_prefill_routing_control')
        old.__dict__.update(current.__dict__)
        old.PaddleOCRTextConfig = PaddleOCRTextConfig
        source = subprocess.check_output(['git', '-C', str(ROOT), 'show', f'0976fa33:{PATH}'], text=True)
        nodes = [n for n in ast.parse(source).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                 and n.name in {'select_text_bucket', 'prepare_text_prefill', 'TextPrefillRuntime'}]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'routing_control', 'exec'), old.__dict__)
        a = old.TextPrefillRuntime.__new__(old.TextPrefillRuntime)
        a.buckets = (128, 256, 512, 1024, 1152)
        a.padding, a.backend = 'bucket', 'torchair'
        b = current.TextPrefillRuntime.__new__(current.TextPrefillRuntime)
        b.eager = False
        for length in range(1, 4097):
            self.assertEqual(a.route(length), b.route(length))
        for length in (1, 128, 129, 300, 700, 1152, 1153):
            route = b.route(length)
            torch.manual_seed(1)
            inputs = (torch.randn(1, length, 4), torch.ones(1, length, dtype=torch.int64),
                      torch.arange(length).view(1, 1, length).expand(3, 1, length))
            before = old.prepare_text_prefill(*inputs, physical_seq_len=route['physical_text_tokens'], execution=route['execution'])
            after = current.prepare_text_prefill(*inputs, physical_seq_len=route['physical_text_tokens'], execution=route['execution'])
            self.assertEqual(signature(vars(before)), signature(vars(after)))

    def test_text_prefill_constructor_call_order(self):
        # Exercise setup using CPU allocations and a fake compiler, without
        # altering the production imports or rebuilding any graph.
        source = subprocess.check_output(['git', '-C', str(ROOT), 'show', f'0976fa33:{PATH}'], text=True)
        old = types.ModuleType('_prefill_setup_control')
        old.__dict__.update(current.__dict__)
        # The old cache naming helpers are reference-only; production removed them.
        helpers = subprocess.check_output(['git', '-C', str(ROOT), 'show',
            '564da03f:19_table_ocr_serving/_support/model/compile_utils.py'], text=True)
        exec(compile(helpers, 'historical_compile_helpers', 'exec'), old.__dict__)
        old.PaddleOCRTextConfig = PaddleOCRTextConfig
        names = {'parse_text_buckets', 'TextPrefillRuntime', 'text_cache_dir_for_bucket', 'text_source_hash'}
        nodes = [ChooseSimulatedNPU().visit(n) for n in ast.parse(source).body
                 if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
        old.TEXT_PADDING_CHOICES = ('auto', 'none', 'bucket')
        old.TEXT_BACKEND_CHOICES = ('raw_eager', 'torchair')
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), 'setup_control', 'exec'), old.__dict__)
        cfg = PaddleOCRTextConfig(hidden_size=32, num_hidden_layers=2, num_key_value_heads=2, head_dim=8)
        model = torch.nn.Module()
        model.config = types.SimpleNamespace(text_config=cfg)
        model.model = torch.nn.Module()
        traces = []
        with tempfile.TemporaryDirectory() as tmp:
            for module in (old, current):
                events = []
                def allocate(**kwargs):
                    events.append(('allocate', kwargs))
                    return allocate_cache(current, cfg, **kwargs)
                model.allocate_static_cache = allocate
                def compile_graph(fn, **kwargs):
                    events.append(('compile', fn.__func__.__name__, {k:v for k,v in kwargs.items() if k not in ('config', 'cache_dir')}))
                    def run(*inputs):
                        events.append(('warm', signature(inputs)))
                    return run
                compiler = types.ModuleType('torchair')
                compiler.__path__ = []
                compiler.CompilerConfig = dict
                compiler.inference = types.ModuleType('torchair.inference')
                compiler.inference.cache_compile = compile_graph
                extra = dict(backend='torchair', buckets=(128,256,512,1024,1152), dtype=torch.float16,
                             linear_weight_format='decode_nz', padding='bucket', model_dir=Path(tmp),
                             cache_root=Path(tmp)) if module is old else dict(
                             graph_directories={b:Path(tmp)/str(b) for b in current.TEXT_PREFILL_BUCKETS})
                with patch.object(module, 'import_torchair', return_value=(compiler, dict), create=True), \
                     patch.dict(sys.modules, {'torchair':compiler, 'torchair.inference':compiler.inference}), \
                     patch.object(module, 'synchronize', side_effect=lambda device: events.append(('sync', str(device))), create=True), \
                     patch.object(module, 'torch_npu', types.SimpleNamespace(npu=types.SimpleNamespace(
                         synchronize=lambda device:events.append(('sync',str(device)))))):
                    text_call(module, cfg, module.TextPrefillRuntime, model, cache_length=4096,
                                             device=torch.device('cpu'), **extra)
                traces.append(events)
        self.assertEqual(*traces)

    def test_single_import_and_cache_identity(self):
        tree = ast.parse(NEW)
        imports = [n for n in ast.walk(tree) if isinstance(n, ast.Import)
                   and any(a.name == 'torch_npu' for a in n.names)]
        self.assertEqual(len(imports), 1)
        self.assertIn(imports[0], tree.body)
        for name in ('text_source_hash', 'short_file_hash', 'torchair_cache_dir_for_shape',
                     'text_cache_dir_for_bucket', 'import_torchair'):
            self.assertFalse(hasattr(current, name))
        self.assertNotIn('huggingface_hub', (EXPERIMENT / 'p04_paddle_ocr_vl_1_6_modeling.py').read_text())


if __name__ == '__main__':
    unittest.main(verbosity=2)
