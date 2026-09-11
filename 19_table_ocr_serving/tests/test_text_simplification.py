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
import re
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / '19_table_ocr_serving'
sys.path.insert(0, str(EXPERIMENT))
# Production imports torch_npu once. CPU tests explicitly provide a test double;
# this is not a CPU fallback in the runtime.
sys.modules.setdefault('torch_npu', types.ModuleType('torch_npu'))
import text_prefill_and_decode as current

PIN = 'dc755584'
PATH = '19_table_ocr_serving/text_prefill_and_decode.py'
OLD = subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{PIN}:{PATH}'], text=True)
NEW = (ROOT / PATH).read_text()
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


class LockedServingContract(ast.NodeTransformer):
    """Specialize the reference only at explicit, benchmarked application choices.

    No operator semantics or timing/copy behavior is normalized away. The removed
    allocations below belonged exclusively to the unused token-policy machinery.
    """
    removed = {'token_selection', 'preferred_token_id', 'alternate_preferred_token_id',
        'cell_start_token_ids', 'math_close_token_id', 'decode_token_id_map',
        'compact_step_control', 'max_prefill_interruptions', 'token_selection_policy_active',
        'token_selection_policy_mask', 'token_selection_override_used', 'token_selection_math_open'}

    def visit_arguments(self, n):
        pairs = list(zip(n.args, [None] * (len(n.args) - len(n.defaults)) + n.defaults))
        pairs = [(a, d) for a, d in pairs if a.arg not in self.removed | {'prefill_policy_mask'}]
        n.args = [a for a, _ in pairs]
        n.defaults = [d for _, d in pairs if d is not None]
        pairs = [(a, d) for a, d in zip(n.kwonlyargs, n.kw_defaults) if a.arg not in self.removed]
        n.kwonlyargs = [a for a, _ in pairs]
        n.kw_defaults = [d for _, d in pairs]
        return n

    def visit_FunctionDef(self, n):
        if n.name == 'execute':
            # The compact decoder returns native IDs. Everything after the old
            # early return was the unreachable full-logit policy implementation.
            for i, stmt in enumerate(n.body):
                if isinstance(stmt, ast.If) and ast.unparse(stmt.test) == 'self.decode_token_id_map is not None':
                    n.body = n.body[:i] + stmt.body
                    break
        return self.generic_visit(n)

    def visit_ImportFrom(self, n):
        return None if n.module == '_support.model.token_selection' else n

    def visit_If(self, n):
        condition = ast.unparse(n.test)
        if condition in {
            'self.compact_step_control',
            'self.compact_step_control and token_selection != TOKEN_SELECTION_GREEDY',
            'decode_token_id_map is not None and self.token_selection != TOKEN_SELECTION_GREEDY',
            'self.max_prefill_interruptions is not None',
            'max_prefill_interruptions is not None and max_prefill_interruptions < 0',
        }:
            return [self.visit(stmt) for stmt in n.orelse]
        return self.generic_visit(n)

    def visit_IfExp(self, n):
        if (isinstance(n.test, ast.Compare) and ast.unparse(n.test.left) == 'self.token_selection'
                and isinstance(n.test.ops[0], ast.In)):
            self.assert_non_greedy(n.test.comparators[0])
            return self.visit(n.orelse)
        return self.generic_visit(n)

    @staticmethod
    def assert_non_greedy(n):
        assert isinstance(n, ast.Tuple)
        assert all(isinstance(e, ast.Name) and e.id.startswith('TOKEN_SELECTION_')
                   and e.id != 'TOKEN_SELECTION_GREEDY' for e in n.elts)

    def visit_Assign(self, n):
        if any(ast.unparse(t) in {'self.' + x for x in self.removed} | {'prefill_policy_mask'} for t in n.targets):
            return None
        if any(ast.unparse(t) == 'active' for t in n.targets) and isinstance(n.value, ast.Tuple) and not n.value.elts:
            return None
        return self.generic_visit(n)

    def visit_AnnAssign(self, n):
        if ast.unparse(n.target) in {'token_selection_policy_active', 'prefill_interruptions'}:
            return None
        return self.generic_visit(n)

    def visit_For(self, n):
        if ast.unparse(n.iter) == 'active' and ast.unparse(n.target) == 'state':
            assert ast.unparse(n.body[0]) == 'state.prefill_interruptions += 1'
            return None
        return self.generic_visit(n)

    def visit_Expr(self, n):
        if isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute):
            owner = ast.unparse(n.value.func.value)
            if any(owner.startswith('self.' + x) for x in
                   ('token_selection_policy_mask', 'token_selection_override_used', 'token_selection_math_open')):
                return None
        return self.generic_visit(n)

    def visit_Call(self, n):
        n.keywords = [kw for kw in n.keywords if kw.arg not in self.removed]
        return self.generic_visit(n)

    def visit_Attribute(self, n):
        n.attr = {'_vision_packing_stats':'_vision_prefill_stats',
                  '_text_packing_stats':'_text_prefill_stats'}.get(n.attr,n.attr)
        if n.attr == 'logical_tensors':
            n.attr = 'flat_tensors'
        return self.generic_visit(n)

    def visit_Pass(self, n):
        return None


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
    module.__dict__.update(torch=torch, nn=torch.nn, dataclass=dataclass, replace=replace,
                           PaddleOCRTextConfig=current.PaddleOCRTextConfig)
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
    config = current.PaddleOCRTextConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=32, rope_parameters={'rope_theta': 500000., 'mrope_section': [1, 1, 2]})
    model = torch.nn.Module()
    model.config = types.SimpleNamespace(text_config=config)
    # This prefill/model-class implementation is independently checked unchanged.
    model.model = current.PaddleOCRTextModel(config)
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
    cache = module.LocalPaddleOCRVLStaticCache.allocate(config, batch_size=batch, cache_length=32,
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
    stage = module.TextDecodeStage(*args, **({'cache_length': 32} if legacy else {}))
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
        runtime = ast.parse((EXPERIMENT / 'serving_runtime.py').read_text())
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
        import serving_runtime as runtime
        recorded = json.loads((ROOT / 'tmp/19_table_ocr_serving/step2_b8qps6_dc755584_20260910_cached/b8/ready.json').read_text())['configuration']
        tree = ast.parse((EXPERIMENT / 'serving_runtime.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ContinuousRecognizer')
        init = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__'))
        # Execute real setup decisions up to tokenizer loading, with a fake NPU
        # facade. No checkpoint, tensor, compiler or device is opened.
        cutoff = next(i for i, n in enumerate(init.body) if isinstance(n, ast.Assign)
                      and ast.unparse(n.targets[0]) == 'self.preprocessing_tokenizer')
        init.body = init.body[:cutoff]
        init.decorator_list = []
        scope = dict(vars(runtime))
        scope.update(torch=types.SimpleNamespace(float16=torch.float16,
            device=lambda name: types.SimpleNamespace(type=name.split(':')[0]),
            npu=types.SimpleNamespace(config=types.SimpleNamespace(), is_available=lambda: True,
                                      set_compile_mode=lambda **kwargs: None)),
            _resolve_model_dir=lambda model: Path(model), _emit_setup_progress=lambda *args: None,
            load_preprocessor_config=lambda _: {'min_pixels': 112896, 'max_pixels': 1003520})
        exec(compile(ast.fix_missing_locations(ast.Module(body=[init], type_ignores=[])), '<setup-contract>', 'exec'), scope)
        instance = types.SimpleNamespace()
        with patch.dict(sys.modules, {'torch_npu': types.ModuleType('torch_npu')}):
            scope['__init__'](instance, model='/unused', batch_size=8, torchair_cache_dir=Path('/unused'),
                              decode_device_timing=False)
        for name in ('decode_backend', 'cache_length', 'max_new_tokens', 'batch_size',
                     'compact_decode_control'):
            self.assertEqual(getattr(instance, name), recorded[name], name)
        self.assertEqual(str(instance.dtype), recorded['dtype'])
        self.assertEqual(list(instance.vision_buckets), recorded['vision_prefill']['buckets'])
        self.assertEqual(list(instance.text_buckets), recorded['text_prefill']['buckets'])
        self.assertEqual(instance.vision_seq_alignment, recorded['vision_prefill']['sequence_alignment'])
        import vision_prefill
        padding_source = ast.parse(inspect.getsource(vision_prefill.prepare_vision_mlp_intermediate))
        target = next(n.value for n in ast.walk(padding_source) if isinstance(n,ast.Assign)
                      and any(isinstance(t,ast.Name) and t.id=='target' for t in n.targets))
        self.assertEqual(ast.literal_eval(target), recorded['vision_prefill']['mlp_intermediate_size'])
        for kind in ('min', 'max'):
            self.assertEqual(instance.preprocessor_config[kind + '_pixels'], recorded['preprocessor']['effective_' + kind + '_pixels'])
        _, vocab = current.load_decode_vocab_token_ids(instance.decode_vocab_token_ids_path, full_vocab_size=103424)
        # Vocabulary intentionally changed after the mechanical-copy anchor.
        # Keep all the original runtime checks above, but anchor this decision
        # to the completed 60k NPU run, not the old 16k readiness record.
        head_record = json.loads((ROOT / 'tmp/19_table_ocr_serving/lm_head_60416_20260911/b8_expanded_measured/b8/ready.json').read_text())['configuration']['decode_vocab']
        self.assertEqual(vocab['token_ids_sha256'], head_record['token_ids_sha256'])
        self.assertEqual(vocab['selected_vocab_size'], head_record['selected_vocab_size'])

    def test_http_worker_cli_and_request_wiring(self):
        import serve
        import serving_runtime as runtime
        with patch.object(sys, 'argv', ['serve.py']):
            args = serve.parse_args()
        expected_args = {'host', 'port', 'request_timeout_s', 'max_image_bytes', 'queue_capacity', 'eager',
            'full_decode_lm_head',
            'model', 'device', 'decode_batch_size', 'no_decode_device_timing', 'request_scheduling_metrics',
            'torchair_cache_dir', 'vision_torchair_cache_dir', 'text_torchair_cache_dir', 'service_summary_output'}
        self.assertEqual(set(vars(args)), expected_args)
        self.assertEqual(serve.PROMPTS, {'table': 'Table Recognition:'})
        self.assertTrue(args.no_decode_device_timing)
        self.assertTrue(args.request_scheduling_metrics)
        self.assertFalse(args.full_decode_lm_head)
        with patch.object(sys, 'argv', ['serve.py', '--expanded-decode-lm-head']), \
             patch('sys.stderr',new=io.StringIO()), self.assertRaises(SystemExit):
            serve.parse_args()
        with patch.object(sys, 'argv', ['serve.py', '--full-decode-lm-head']):
            self.assertTrue(serve.parse_args().full_decode_lm_head)
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
                request = source.pull(block=False)
                test.assertEqual((request.min_pixels, request.max_pixels), (28224, 802816))
                test.assertEqual(request.prompt, 'Table Recognition:')
                kwargs['emit_result'](Result(request.request_id, '<table></table>'))
                test.assertIsNone(source.pull(block=False))
                test.assertTrue(source.closed)
                return Summary(1)
        jobs, results = queue.Queue(), queue.Queue()
        jobs.put(dict(request_id='test', image_bytes=b'not-decoded-in-this-test', prompt='Table Recognition:',
                      crop_type='table', submitted_monotonic_s=0.0))
        jobs.put(None)
        cfg = dict(model='/unused', device='npu:0', decode_batch_size=8, decode_device_timing=False, eager=False,
                   full_decode_lm_head=True,
                   request_scheduling_metrics=True, torchair_cache_dir='/unused/decode',
                   vision_torchair_cache_dir='/unused/vision', text_torchair_cache_dir='/unused/text')
        with patch.object(runtime, 'ContinuousRecognizer', FakeRecognizer), \
             patch.object(serve, '_freeze_setup_gc', return_value={'enabled': True}) as freeze, \
             patch('sys.stdout', new=io.StringIO()):
            serve._worker_main(jobs, results, cfg)
        messages = []
        while not results.empty(): messages.append(results.get())
        self.assertEqual([m['kind'] for m in messages], ['ready', 'result', 'service_summary'], messages)
        self.assertTrue(messages[1]['ok'])
        self.assertFalse(seen[0]['eager'])
        self.assertTrue(seen[0]['full_decode_lm_head'])
        freeze.assert_called_once()

    def test_full_head_setup_and_cache_separation(self):
        import serving_runtime as runtime
        tree = ast.parse((EXPERIMENT / 'serving_runtime.py').read_text())
        cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name == 'ContinuousRecognizer')
        init = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name == '__init__')
        start = next(i for i,n in enumerate(init.body) if isinstance(n,ast.Assign)
                     and ast.unparse(n.targets[0]) == 'full_vocab_size')
        # Execute the actual head-selection setup, not a duplicated policy.
        selection = ast.Module(body=init.body[start:start+2],type_ignores=[])
        cache_keys = []
        for full in (False,True):
            model = torch.nn.Module()
            model.lm_head = torch.nn.Linear(4,64,bias=False)
            owner = types.SimpleNamespace(model=model,full_decode_lm_head=full,
                decode_vocab_token_ids_path=Path('/unused/vocab.json'))
            metadata = dict(enabled=True,selected_vocab_size=3,token_ids_sha256='abc123456789ffff')
            scope = dict(vars(runtime),self=owner)
            load = unittest.mock.Mock(return_value=((3,11,63),metadata))
            scope['load_decode_vocab_token_ids'] = load
            with torch.inference_mode():
                exec(compile(selection,'head_selection','exec'),scope)
            cache_keys.append(scope['decode_head_cache_key'])
            self.assertEqual(owner.decode_vocab['enabled'],not full)
            if full:
                load.assert_not_called()
                self.assertFalse(hasattr(model,'decode_lm_head'))
                self.assertFalse(hasattr(model,'decode_token_id_map'))
                self.assertEqual(owner.decode_vocab['selected_vocab_size'],64)
            else:
                load.assert_called_once()
                self.assertTrue(torch.equal(model.decode_lm_head.weight,model.lm_head.weight[[3,11,63]]))
                self.assertEqual(model.decode_token_id_map.tolist(),[3,11,63])
        self.assertEqual(cache_keys,['selected_vocab_3_abc123456789','full_vocab_64'])
        stages = next(n for n in ast.walk(init) if isinstance(n,ast.Call)
                      and ast.unparse(n.func) == 'self.model.make_inference_stages')
        cache_root = next(k.value for k in stages.keywords if k.arg == 'decode_cache_root')
        self.assertEqual(ast.unparse(cache_root),'torchair_cache_dir / decode_head_cache_key')

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

    def test_protected_source(self):
        def definitions(source):
            return {n.name: ast.get_source_segment(source, n) for n in ast.parse(source).body
                    if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        old, new = definitions(OLD), definitions(NEW)
        # NZ setup now intentionally rejects failed conversions; its success and
        # failure contracts are tested separately instead of source equality.
        for name in ('load_decode_vocab_token_ids', 'prepare_decode_compact_lm_head'):
            expected = old[name].replace('optimization: str | DecodeOptimizationConfig = "baseline"',
                'optimization: str | DecodeOptimizationConfig = "' + NAMES[0] + '"')
            self.assertEqual(expected, new[name], name)
        marker = '# ---- Relocated text-prefill implementation (unchanged computation) ----'
        # Explicit mechanical inlining whitelist; no other prefill changes.
        before = OLD[OLD.index(marker):]
        # Prefill definitions now live in purpose-led sections, not one suffix.
        after = NEW
        expected = without_methods(before, {
            'PaddleOCRAttention': {'attend', 'forward', 'forward_prefill_static'},
            'PaddleOCRDecoderLayer': {'forward', 'forward_prefill_static'},
            'PaddleOCRTextModel': {'forward', 'forward_prefill_static'}})
        expected = expected.replace('import os\n', '')
        expected = re.sub(r'^TEXT_SOFTMAX_DTYPE_ENV = .*\n|^SOFTMAX_DTYPE_CHOICES = .*\n', '', expected, flags=re.M)
        for name in ('get_text_softmax_dtype_mode', 'attention_softmax'):
            expected = re.sub(r'^def ' + name + r'\([\s\S]*?(?=^def |^class |^@)', '', expected, flags=re.M)
        expected = re.sub(r'        probs = attention_softmax\([\s\S]*?        \)\n',
                          '        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)\n', expected)
        expected = re.sub(r'        probabilities = attention_softmax\([\s\S]*?        \)\n',
                          '        probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)\n', expected)
        expected = expected.replace('        self.softmax_dtype_mode = get_text_softmax_dtype_mode()\n', '')
        expected = expected.replace('get_text_softmax_dtype_mode()', "'fp32'")
        expected = expected.replace('f"softmax{cache_key_part(\'fp32\')}"', '"softmaxfp32"')
        expected = expected[expected.index('DEFAULT_TEXT_BUCKETS'):]
        expected = expected.replace('                init_mode="zeros",\n', '')
        # These explicitly changed definitions have focused behavior tests below.
        changed = {'_activation', 'build_causal_mask', 'PaddleOCRMLP',
                   'parse_text_buckets', 'select_text_bucket', 'prepare_text_prefill',
                   'text_cache_dir_for_bucket', 'TextPrefillRuntime'}
        def unchanged_defs(source):
            definitions = {}
            for n in ast.parse(source).body:
                if not isinstance(n, (ast.FunctionDef, ast.ClassDef)) or n.name in changed:
                    continue
                if isinstance(n, ast.ClassDef) and n.name == 'TextPrefillStage':
                    # Only declaration order changed: forward now introduces
                    # the computation before _attention and __init__.
                    methods = [m for m in n.body if isinstance(m, ast.FunctionDef)]
                    n.body = [m for m in n.body if not isinstance(m, ast.FunctionDef)]
                    n.body += sorted(methods, key=lambda m: m.name)
                definitions[n.name] = ast.dump(n)
            return definitions
        expected_defs, actual_defs = unchanged_defs(expected), unchanged_defs(after)
        self.assertEqual(expected_defs, {name: actual_defs[name] for name in expected_defs})
        for relative in ('crop_processing.py', '_support/serving/continuous_decode.py',
                         '_support/model/compile_utils.py'):
            previous = subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{PIN}:19_table_ocr_serving/{relative}'])
            if relative == 'crop_processing.py':
                # Only the unused minimum-only compatibility wrapper is removed.
                removed = b'def apply_min_pixels_override(cfg: dict, min_pixels: int | None) -> dict:\n    """Backward-compatible wrapper for callers overriding only ``min_pixels``."""\n    return apply_pixel_overrides(cfg, min_pixels=min_pixels)\n\n\n'
                self.assertEqual(previous.count(removed), 1)
                previous = previous.replace(removed, b'')
            if relative == '_support/serving/continuous_decode.py':
                expected = LockedServingContract().visit(ast.parse(previous))
                actual = ast.parse((EXPERIMENT / relative).read_text())
                self.assertEqual(ast.dump(expected), ast.dump(actual), relative)
                continue
            actual = (EXPERIMENT / relative).read_bytes()
            if relative == 'crop_processing.py':
                # Preprocessing is unchanged; only formatting functions/imports
                # were moved here, with exact-source tests in the crop suite.
                def defs(src):
                    return {n.name: ast.dump(n) for n in ast.parse(src).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
                expected_defs, actual_defs = defs(previous), defs(actual)
                self.assertEqual(expected_defs, {k:actual_defs[k] for k in expected_defs})
            else:
                self.assertEqual(previous, actual, relative)

    def test_prefill_softmax_contract(self):
        self.assertFalse(hasattr(current, 'get_text_softmax_dtype_mode'))
        marker = '# ---- Relocated text-prefill implementation (unchanged computation) ----'
        old = types.ModuleType('_step2_prefill')
        old.__dict__.update(current.__dict__)
        old.__dict__.update(os=os, TEXT_SOFTMAX_DTYPE_ENV='PADDLE_OCR_VL_TEXT_SOFTMAX_DTYPE',
                            SOFTMAX_DTYPE_CHOICES=('fp32', 'model'))
        nodes = [n for n in ast.parse(OLD[OLD.index(marker):]).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), 'old_prefill', 'exec'), old.__dict__)
        for dtype in (torch.float16, torch.float32):
            results = []
            for module in (old, current):
                torch.manual_seed(1729)
                cfg = current.PaddleOCRTextConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                    max_position_embeddings=32, rope_parameters={'rope_theta': 500000., 'mrope_section': [1, 1, 2]})
                model = torch.nn.Module()
                model.config = types.SimpleNamespace(text_config=cfg)
                model.model = module.PaddleOCRTextModel(cfg).to(dtype)
                stage = module.TextPrefillStage(model)
                cache = current.LocalPaddleOCRVLStaticCache.allocate(cfg, batch_size=1, cache_length=32,
                    device=torch.device('cpu'), dtype=dtype)
                hidden = torch.randn(1, 8, 32).to(dtype)
                mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]])
                positions = torch.arange(8).view(1, 1, 8).expand(3, 1, 8)
                with patch.dict(os.environ, {'PADDLE_OCR_VL_TEXT_SOFTMAX_DTYPE': 'fp32'}), torch.inference_mode():
                    out = stage(hidden, mask, positions, torch.tensor([4]), *cache.flat_tensors())
                results.append((signature(out), signature(cache.flat_tensors())))
            self.assertEqual(*results)

    def test_model_composition_and_serving_wiring(self):
        import paddle_ocr_vl_1_6_modeling as modeling
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
        old, new = defs(expected), defs((ROOT / path).read_text())
        self.assertEqual(old.keys() - {'LocalModelOutput', 'LocalStaticModelOutput', '_resolve_model_dir'}, new.keys())
        self.assertNotIn('_resolve_model_dir', new)
        for name in new.keys() - {'LocalPaddleOCRVLForConditionalGeneration'}:
            self.assertEqual(old[name], new[name], name)
        def methods(src):
            cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == 'LocalPaddleOCRVLForConditionalGeneration')
            return {n.name: ast.get_source_segment(src,n) for n in cls.body if isinstance(n,ast.FunctionDef)}
        a,b = methods(expected),methods((ROOT/path).read_text())
        local_loader = a['from_pretrained'].replace(
            'model_id_or_path: str | Path = "PaddlePaddle/PaddleOCR-VL-1.6"',
            'model_dir: str | Path').replace(
            'model_dir = _resolve_model_dir(model_id_or_path)',
            'model_dir = Path(model_dir).expanduser()')
        self.assertEqual(local_loader, b['from_pretrained'])
        for name in a.keys()-{'make_inference_stages','allocate_static_cache','from_pretrained'}:
            self.assertEqual(a[name],b[name],name)
        for name in removed:
            self.assertNotIn(name, vars(modeling.LocalPaddleOCRVLForConditionalGeneration))
        # Constructor source above is identical, preserving checkpoint hierarchy.
        cfg = modeling.PaddleOCRVLConfig.from_dict({
            'vision_config': {'hidden_size': 144, 'num_attention_heads': 2, 'num_hidden_layers': 2, 'intermediate_size': 272},
            'text_config': {'hidden_size': 32, 'num_attention_heads': 4, 'num_key_value_heads': 2,
                'head_dim': 8, 'num_hidden_layers': 2, 'intermediate_size': 64, 'vocab_size': 64,
                'rope_parameters': {'rope_theta': 500000., 'mrope_section': [1, 1, 2]}}})
        old_class = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'LocalPaddleOCRVLForConditionalGeneration')
        scope = dict(vars(modeling))
        exec(compile(ast.Module(body=[old_class], type_ignores=[]), 'old_model', 'exec'), scope)
        torch.manual_seed(42)
        a = scope['LocalPaddleOCRVLForConditionalGeneration'](cfg)
        torch.manual_seed(42)
        b = modeling.LocalPaddleOCRVLForConditionalGeneration(cfg)
        self.assertEqual(signature(a.state_dict()), signature(b.state_dict()))
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


    def test_compiler_wrapper_unchanged(self):
        def active_cache_block(source):
            fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'compile_text_decode_stage')
            branches = [n for n in fn.body if isinstance(n, ast.If) and ast.unparse(n.test) == "backend_name == 'torchair'"]
            body = branches[0].body if branches else fn.body
            index = next(i for i, n in enumerate(body) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'shape_cache_dir')
            class WithoutSelection(ast.NodeTransformer):
                def visit_Call(self,n):
                    # Removed fixed arguments only; retain all graph-compile flags.
                    if ast.unparse(n.func) == 'torchair_cache_dir_for_shape':
                        n.keywords=[kw for kw in n.keywords if kw.arg not in ('optimization','dtype','linear_weight_format')]
                    else:
                        n.keywords=[kw for kw in n.keywords if kw.arg!='optimization']
                    n.args=[a for a in n.args if not(isinstance(a,ast.Name) and a.id=='optimization')]
                    return self.generic_visit(n)
                def visit_Dict(self,n):
                    # Only removed descriptive fields are normalized. Compiler
                    # invocation, wrapper construction and ordering stay exact.
                    pairs=[(k,v) for k,v in zip(n.keys,n.values) if not(isinstance(k,ast.Constant) and k.value in ('decode_optimization','decode_optimization_config','decode_attention','decode_cache_update','dtype','linear_weight_format'))]
                    n.keys=[k for k,v in pairs];n.values=[v for k,v in pairs]
                    return self.generic_visit(n)
                def visit_IfExp(self,n):
                    if ast.unparse(n.test) == 'model_dir is not None':
                        return self.visit(n.body)
                    return self.generic_visit(n)
            return ast.dump(WithoutSelection().visit(ast.Module(body=body[index:], type_ignores=[])))
        self.assertEqual(active_cache_block(OLD), active_cache_block(NEW))

    def test_independent_prefill_execution_unchanged(self):
        path='19_table_ocr_serving/serving_runtime.py'
        old_source=subprocess.check_output(['git','-C',str(ROOT),'show',f'{PIN}:{path}'],text=True)
        new_source=(ROOT/path).read_text()
        def classes(src):
            return {n.name:n for n in ast.parse(src).body if isinstance(n,ast.ClassDef)}
        old_classes,new_classes=classes(old_source),classes(new_source)
        # Arrival/admission/CPU lookahead behavior is not rewritten by this cleanup.
        class CropHandoff(ast.NodeTransformer):
            def visit_If(self, n):
                # Additive CPU-readiness observations are independently tested.
                # Remove ONLY the two new metric calls for this scheduling AST
                # comparison; future polling, waiting and prefill stay checked.
                if (ast.unparse(n.test) == 'self.scheduling_metrics is not None'
                    and len(n.body)==1 and isinstance(n.body[0],ast.Expr)
                    and isinstance(n.body[0].value,ast.Call)
                    and ast.unparse(n.body[0].value.func) in
                        {'self.scheduling_metrics.cpu_prefill_eligible',
                         'self.scheduling_metrics.cpu_prepared'}):
                    return None
                return self.generic_visit(n)
            def visit_FunctionDef(self, n):
                n = self.generic_visit(n)
                if n.name == 'pull':
                    # The prefill handoff itself is tested end-to-end in the
                    # singleton parity test. Check the admission loop around it.
                    for loop in ast.walk(n):
                        if not isinstance(loop, ast.While): continue
                        start = next((i for i,x in enumerate(loop.body) if isinstance(x, ast.Assign)
                            and isinstance(x.targets[0], ast.Name) and x.targets[0].id in ('group','crop')), None)
                        if start is not None:
                            end = next(i for i in range(start,len(loop.body)) if isinstance(loop.body[i],ast.If)
                                and ast.unparse(loop.body[i].test)=='self.scheduling_metrics is not None')
                            loop.body[start:end] = ast.parse('finalized = crop_handoff(prepared, consumer_wait_s)').body
                return n
            def visit_Subscript(self,n):
                if ast.unparse(n)=='finalized[0]': return ast.Name(id='finalized',ctx=ast.Load())
                return self.generic_visit(n)
        def method_order_independent(cls):
            # The request-facing methods moved ahead of setup; retain all
            # class fields, decorators, signatures, and complete method bodies.
            methods = [m for m in cls.body if isinstance(m,ast.FunctionDef)]
            cls.body = [m for m in cls.body if not isinstance(m,ast.FunctionDef)]
            cls.body += sorted(methods,key=lambda m:m.name)
            return ast.dump(cls)
        self.assertEqual(method_order_independent(CropHandoff().visit(LockedServingContract().visit(old_classes['_OpenPrefillSource']))),
                         method_order_independent(CropHandoff().visit(new_classes['_OpenPrefillSource'])))
        def methods(cls):
            return {n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)}
        old,new=methods(old_classes['ContinuousRecognizer']),methods(new_classes['ContinuousRecognizer'])
        facts={
            'group.profiled_route is not None':False,
            'batch_size == 1 and len(group.members) == 1':True,
            'batch_size > 1':False,
            'batch_size == 1':True,
            'len(group.members) > 1':False,
            'len(group.members) == 1':True,
            'self.batched_text_prefill is not None':False,
            'self.packed_text_prefill is not None':False,
            "self.vision_packing != 'off' or self.text_packing != 'off'":False,
        }
        class SelectedBranch(LockedServingContract):
            def visit_If(self,n):
                choice=facts.get(ast.unparse(n.test))
                n=super().visit_If(n)
                return (n.body if choice else n.orelse) if choice is not None else n
            def visit_IfExp(self,n):
                choice=facts.get(ast.unparse(n.test))
                n=super().visit_IfExp(n)
                return (n.body if choice else n.orelse) if choice is not None else n
            def visit_Pass(self,n):return None
        for name in ('serve','_iter_cpu_prepared','_decode_ready_source','_result_from_completion'):
            a=SelectedBranch().visit(copy.deepcopy(old[name]))
            b=SelectedBranch().visit(copy.deepcopy(new[name]))
            self.assertEqual(ast.dump(a),ast.dump(b),name)

    def test_zero_cache_matches_reference(self):
        cfg=current.PaddleOCRTextConfig(num_hidden_layers=2,num_key_value_heads=2,head_dim=8)
        for batch in (1,2,8):
            kwargs=dict(batch_size=batch,cache_length=32,device=torch.device('cpu'),dtype=torch.float16)
            old=self.old.LocalPaddleOCRVLStaticCache.allocate(cfg,init_mode='zeros',**kwargs)
            new=self.new.LocalPaddleOCRVLStaticCache.allocate(cfg,**kwargs)
            self.assertEqual(signature(old.flat_tensors()),signature(new.flat_tensors()))

    def test_fixed_text_buckets_and_preparation(self):
        # Preserve every route, including actual overflow, not just the corpus.
        old = types.ModuleType('_prefill_routing_control')
        old.__dict__.update(current.__dict__)
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
        names = {'parse_text_buckets', 'TextPrefillRuntime', 'text_cache_dir_for_bucket'}
        nodes = [ChooseSimulatedNPU().visit(n) for n in ast.parse(source).body
                 if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
        old.TEXT_PADDING_CHOICES = ('auto', 'none', 'bucket')
        old.TEXT_BACKEND_CHOICES = ('raw_eager', 'torchair')
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), 'setup_control', 'exec'), old.__dict__)
        cfg = current.PaddleOCRTextConfig(hidden_size=32, num_hidden_layers=2, num_key_value_heads=2, head_dim=8)
        model = torch.nn.Module()
        model.config = types.SimpleNamespace(text_config=cfg)
        model.model = torch.nn.Module()
        traces = []
        with tempfile.TemporaryDirectory() as tmp:
            for module in (old, current):
                events = []
                def allocate(**kwargs):
                    events.append(('allocate', kwargs))
                    return current.LocalPaddleOCRVLStaticCache.allocate(cfg, **kwargs)
                model.allocate_static_cache = allocate
                def compile_graph(fn, **kwargs):
                    events.append(('compile', fn.__func__.__name__, {k:v for k,v in kwargs.items() if k not in ('config', 'cache_dir')}))
                    def run(*inputs):
                        events.append(('warm', signature(inputs)))
                    return run
                compiler = types.SimpleNamespace(inference=types.SimpleNamespace(cache_compile=compile_graph))
                extra = dict(backend='torchair', buckets=(128,256,512,1024,1152), dtype=torch.float16,
                             linear_weight_format='decode_nz', padding='bucket') if module is old else {}
                with patch.object(module, 'import_torchair', return_value=(compiler, dict)), \
                     patch.object(module, 'synchronize', side_effect=lambda device: events.append(('sync', str(device)))):
                    module.TextPrefillRuntime(model, cache_root=Path(tmp), cache_length=4096,
                                             device=torch.device('cpu'), model_dir=Path(tmp), **extra)
                traces.append(events)
        self.assertEqual(*traces)

    def test_single_import_and_cache_identity(self):
        tree = ast.parse(NEW)
        imports = [n for n in ast.walk(tree) if isinstance(n, ast.Import)
                   and any(a.name == 'torch_npu' for a in n.names)]
        self.assertEqual(len(imports), 1)
        self.assertIn(imports[0], tree.body)
        self.assertEqual(current.decode_source_hash(), current.short_file_hash(EXPERIMENT / 'text_prefill_and_decode.py'))
        for function in (current.torchair_cache_dir_for_shape, current.text_cache_dir_for_bucket):
            self.assertNotIn('dtype', inspect.signature(function).parameters)
            self.assertNotIn('linear_weight_format', inspect.signature(function).parameters)
        self.assertNotIn('huggingface_hub', (EXPERIMENT / 'paddle_ocr_vl_1_6_modeling.py').read_text())


if __name__ == '__main__':
    unittest.main(verbosity=2)
