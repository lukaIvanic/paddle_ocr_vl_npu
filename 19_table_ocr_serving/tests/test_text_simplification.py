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
import types
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / '19_table_ocr_serving'
sys.path.insert(0, str(EXPERIMENT))
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
    for name, child in model.named_modules():
        if isinstance(child, torch.nn.Linear):
            child.register_forward_pre_hook(lambda layer, args, name=name: fake.record('linear:' + name, args, layer.weight))
    stage = module.TextDecodeStage(*args, **({'cache_length': 32} if legacy else {}))
    outputs = []
    with patch.dict(sys.modules, {'torch_npu': fake}), torch.inference_mode():
        for _ in range(2):
            out = stage(inputs, positions, deltas, *cache.flat_tensors())
            outputs.append(out.clone())
            inputs = out if compact else out[:, -1].argmax(-1).view(-1, 1)
            positions = positions + 1
    return outputs, tuple(x.clone() for x in cache.flat_tensors()), fake.events


class TextSimplificationTests(unittest.TestCase):
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
        self.assertEqual(ast.unparse(metadata['decode_attention']), 'DECODE_ATTENTION')
        self.assertEqual(ast.unparse(metadata['decode_cache_update']), 'DECODE_CACHE_UPDATE')
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
                              vision_linear_weight_format='fractal_nz', decode_device_timing=False)
        for name in ('decode_backend', 'cache_length', 'max_new_tokens', 'batch_size',
                     'compact_decode_control', 'vision_linear_patch_projection', 'vision_attention_weight_padding'):
            self.assertEqual(getattr(instance, name), recorded[name], name)
        self.assertEqual(str(instance.dtype), recorded['dtype'])
        self.assertEqual(list(instance.vision_buckets), recorded['vision_prefill']['buckets'])
        self.assertEqual(list(instance.text_buckets), recorded['text_prefill']['buckets'])
        self.assertEqual(instance.vision_seq_alignment, recorded['vision_prefill']['sequence_alignment'])
        self.assertEqual(instance.vision_mlp_intermediate_size_requested, recorded['vision_prefill']['mlp_intermediate_size'])
        for kind in ('min', 'max'):
            self.assertEqual(instance.preprocessor_config[kind + '_pixels'], recorded['preprocessor']['effective_' + kind + '_pixels'])
        _, vocab = current.load_decode_vocab_token_ids(instance.decode_vocab_token_ids_path, full_vocab_size=103424)
        self.assertEqual(vocab['token_ids_sha256'], recorded['decode_vocab']['token_ids_sha256'])
        self.assertEqual(vocab['selected_vocab_size'], recorded['decode_vocab']['selected_vocab_size'])

    def test_http_worker_cli_and_request_wiring(self):
        import serve
        import serving_runtime as runtime
        with patch.object(sys, 'argv', ['serve.py']):
            args = serve.parse_args()
        expected_args = {'host', 'port', 'request_timeout_s', 'max_image_bytes', 'queue_capacity',
            'model', 'device', 'decode_batch_size', 'no_decode_device_timing', 'request_scheduling_metrics',
            'torchair_cache_dir', 'vision_torchair_cache_dir', 'text_torchair_cache_dir', 'service_summary_output'}
        self.assertEqual(set(vars(args)), expected_args)
        self.assertEqual(serve.PROMPTS, {'table': 'Table Recognition:'})
        self.assertTrue(args.no_decode_device_timing)
        self.assertTrue(args.request_scheduling_metrics)
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
        cfg = dict(model='/unused', device='npu:0', decode_batch_size=8, decode_device_timing=False,
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
        self.assertEqual(seen[0]['vision_linear_weight_format'], 'fractal_nz')
        freeze.assert_called_once()

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
                        self.assertTrue(all(torch.equal(x, y) for x, y in zip(a, b)))
                        self.assertTrue(all(torch.equal(x, y) for x, y in zip(ka, kb)))

    def test_protected_source(self):
        def definitions(source):
            return {n.name: ast.get_source_segment(source, n) for n in ast.parse(source).body
                    if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        old, new = definitions(OLD), definitions(NEW)
        for name in ('decode_source_hash', 'load_decode_vocab_token_ids', 'prepare_decode_compact_lm_head', 'cast_decode_linear_weights_to_nz'):
            expected = old[name].replace('optimization: str | DecodeOptimizationConfig = "baseline"',
                'optimization: str | DecodeOptimizationConfig = "' + NAMES[0] + '"')
            self.assertEqual(expected, new[name], name)
        marker = '# ---- Relocated text-prefill implementation (unchanged computation) ----'
        # Explicit mechanical inlining whitelist; no other prefill changes.
        before = OLD[OLD.index(marker):]
        after = NEW[NEW.index('DEFAULT_TEXT_BUCKETS'):]
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
        self.assertEqual(ast.dump(ast.parse(expected)), ast.dump(ast.parse(after)))
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
            self.assertEqual(previous, (EXPERIMENT / relative).read_bytes(), relative)

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
        self.assertEqual(old.keys() - {'LocalModelOutput', 'LocalStaticModelOutput'}, new.keys())
        for name in new.keys() - {'LocalPaddleOCRVLForConditionalGeneration'}:
            self.assertEqual(old[name], new[name], name)
        def methods(src):
            cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == 'LocalPaddleOCRVLForConditionalGeneration')
            return {n.name: ast.get_source_segment(src,n) for n in cls.body if isinstance(n,ast.FunctionDef)}
        a,b = methods(expected),methods((ROOT/path).read_text())
        for name in a.keys()-{'make_inference_stages','allocate_static_cache'}:
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
            body = next(n for n in fn.body if isinstance(n, ast.If) and ast.unparse(n.test) == "backend_name == 'torchair'").body
            index = next(i for i, n in enumerate(body) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'shape_cache_dir')
            class WithoutSelection(ast.NodeTransformer):
                def visit_Call(self,n):
                    n.keywords=[kw for kw in n.keywords if kw.arg!='optimization']
                    n.args=[a for a in n.args if not(isinstance(a,ast.Name) and a.id=='optimization')]
                    return self.generic_visit(n)
                def visit_Dict(self,n):
                    pairs=[(k,v) for k,v in zip(n.keys,n.values) if not(isinstance(k,ast.Constant) and k.value in ('decode_optimization','decode_optimization_config'))]
                    n.keys=[k for k,v in pairs];n.values=[v for k,v in pairs]
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
        self.assertEqual(ast.dump(LockedServingContract().visit(old_classes['_OpenPrefillSource'])),ast.dump(new_classes['_OpenPrefillSource']))
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
        for name in ('serve','_enqueue_staged_prefill_group','_stage_prefill_group',
                     '_finalize_prefill_group','_iter_cpu_prepared','_iter_single_prefill_groups',
                     '_prepared_group','_decode_ready_source','prefill_prepared_one','_result_from_completion'):
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


if __name__ == '__main__':
    unittest.main(verbosity=2)
