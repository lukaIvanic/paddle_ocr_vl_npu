"""CPU arithmetic/call-contract checks, not Ascend kernel or speed validation.

The old and cleaned vision modules execute the same two-layer D72 model.
Only the NPU-device guard is replaced in memory; PromptFA is simulated on CPU.
All other definitions (including preparation and compilation) are source-checked.
"""
from __future__ import annotations

import ast
import inspect
import copy
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

import torch
from fixed_architecture_reference import PaddleOCRVisionConfig, vision_model, FreezeArchitecture

from test_text_simplification import ChooseSimulatedNPU, signature, without_methods

ROOT = Path(__file__).resolve().parents[2]
PATH = '19_table_ocr_serving/vision_prefill.py'
PIN = 'dc755584'
OLD = subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{PIN}:{PATH}'], text=True)
NEW = (ROOT / PATH).read_text()
REMOVED_DEFINITIONS = {'get_vision_attention_impl', 'get_vision_prompt_fa_layout',
    'get_vision_prompt_fa_mask_sparse_mode', 'get_vision_softmax_dtype_mode', 'attention_softmax',
    'PreparedPackedVisionPrefill', 'prepare_packed_vision_prefill', 'rotate_half', 'apply_rotary_pos_emb_vision'}
REMOVED_DEFINITIONS |= {'_activation', 'prompt_flash_attention_call_head_dim', 'parse_vision_buckets', 'align_vision_buckets'}
CHANGED_DEFINITIONS = {'vision_prompt_flash_attention_bnsd', 'PaddleOCRVisionAttention',
    'VisionPrefillStage', 'vision_cache_dir_for_bucket', 'VisionPrefillRuntime', 'PaddleOCRVisionEmbeddings'}
CHANGED_DEFINITIONS |= {'prepare_vision_mlp_intermediate', 'prepare_vision_linear_weight_format',
                       'PaddleOCRVisionMLP', 'align_vision_seq_len', 'select_vision_bucket',
                       'prepare_vision_attention_weight_padding'}


class ResolveVisionMetadata(ast.NodeTransformer):
    """Normalize only the deleted selector and its recorded constant values."""
    def visit_FunctionDef(self, n):
        if n.name == 'prepare_packed':
            return None
        pairs = [(a, d) for a, d in zip(n.args.kwonlyargs, n.args.kw_defaults) if a.arg != 'attention_impl']
        n.args.kwonlyargs = [a for a, _ in pairs]
        n.args.kw_defaults = [d for _, d in pairs]
        return self.generic_visit(n)

    def visit_Assign(self, n):
        if any(ast.unparse(t) == 'self.attention_impl' for t in n.targets):
            return None
        return self.generic_visit(n)

    def visit_If(self, n):
        if 'self.attention_impl' in ast.unparse(n.test):
            return []  # Only the removed implementation validation guards.
        return self.generic_visit(n)

    def visit_Attribute(self, n):
        if ast.unparse(n) == 'self.attention_impl':
            return ast.copy_location(ast.Constant('prompt_flash_attention'), n)
        return self.generic_visit(n)

    def visit_Name(self, n):
        if n.id == 'attention_impl':
            return ast.copy_location(ast.Constant('prompt_flash_attention'), n)
        return n

    def visit_Call(self, n):
        literals = {'get_vision_prompt_fa_layout': 'bnsd',
            'get_vision_prompt_fa_mask_sparse_mode': 1, 'get_vision_softmax_dtype_mode': 'fp32'}
        if isinstance(n.func, ast.Name) and n.func.id in literals:
            return ast.copy_location(ast.Constant(literals[n.func.id]), n)
        n.keywords = [kw for kw in n.keywords if kw.arg != 'attention_impl']
        n = self.generic_visit(n)
        if isinstance(n.func, ast.Name) and n.func.id == 'cache_key_part' and len(n.args) == 1 and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str):
            return n.args[0]
        return n

    def visit_IfExp(self, n):
        n = self.generic_visit(n)
        if isinstance(n.test, ast.Compare) and all(isinstance(x, ast.Constant) for x in [n.test.left, *n.test.comparators]):
            test = ast.fix_missing_locations(ast.Expression(n.test))
            return n.body if eval(compile(test, '<literal>', 'eval'), {'__builtins__': {}}) else n.orelse
        return n

    def visit_JoinedStr(self, n):
        n = self.generic_visit(n)
        values = []
        for v in n.values:
            if isinstance(v, ast.FormattedValue) and isinstance(v.value, ast.Constant) and v.format_spec is None and v.conversion == -1:
                v = ast.Constant(str(v.value.value))
            if values and isinstance(values[-1], ast.Constant) and isinstance(v, ast.Constant):
                values[-1].value += v.value
            else:
                values.append(v)
        return values[0] if len(values) == 1 and isinstance(values[0], ast.Constant) else ast.JoinedStr(values=values)


def load_vision(source, name):
    tree = ast.parse(source)
    for i, node in enumerate(tree.body):
        if isinstance(node, ast.FunctionDef) and node.name == 'vision_prompt_flash_attention_bnsd':
            tree.body[i] = ChooseSimulatedNPU().visit(copy.deepcopy(node))
    module = types.ModuleType(name)
    module.__file__ = str(ROOT / PATH)
    sys.modules[name] = module
    exec(compile(ast.fix_missing_locations(tree), module.__file__, 'exec'), module.__dict__)
    return module


class SimulatedPromptFA(types.ModuleType):
    def __init__(self):
        super().__init__('torch_npu')
        self.events = []

    def record(self, name, *args, **kwargs):
        self.events.append((name, signature(args), signature(kwargs)))

    def npu_prompt_flash_attention(self, q, k, v, **kwargs):
        self.record('promptfa', q, k, v, **kwargs)
        assert kwargs['input_layout'] == 'BNSD'
        assert kwargs['num_heads'] == q.shape[1]
        assert kwargs['sparse_mode'] == (1 if 'atten_mask' in kwargs else 0)
        assert 'inner_precise' not in kwargs
        scores = (q @ k.transpose(-1, -2)) * kwargs['scale_value']
        if 'atten_mask' in kwargs:
            scores = scores.masked_fill(kwargs['atten_mask'], torch.finfo(scores.dtype).min)
        return torch.softmax(scores.float(), -1).to(q.dtype) @ v


def exercise(module, mode, batch, seq, real, dtype):
    torch.manual_seed(1729)
    cfg = PaddleOCRVisionConfig(hidden_size=144, num_attention_heads=2,
        intermediate_size=272, num_hidden_layers=2, image_size=28)
    model = torch.nn.Module()
    model.visual = vision_model(module, cfg).to(dtype=dtype)
    if mode == 'weight_padded':
        module.prepare_vision_attention_weight_padding(model)
        kwargs = {'target_intermediate_size':4352} if 'target_intermediate_size' in inspect.signature(module.prepare_vision_mlp_intermediate).parameters else {}
        module.prepare_vision_mlp_intermediate(model, **kwargs)
    stage = (module.VisionPrefillStage(model, attention_impl='prompt_flash_attention')
             if module.__name__ == '_step2_vision_control' else module.VisionPrefillStage(model))
    hidden = torch.randn(batch, seq, 144).to(dtype)
    hidden[:, real:] = 0
    angles = torch.randn(seq, 36).repeat(1, 2)
    mask = (torch.arange(seq) >= real).view(1, 1, 1, seq)
    fake = SimulatedPromptFA()
    module.torch_npu = fake
    for name, child in model.named_modules():
        if isinstance(child, (torch.nn.Linear, torch.nn.LayerNorm)):
            child.register_forward_pre_hook(
                lambda layer, args, name=name: fake.record(name, args, layer.weight))
    with patch.dict(sys.modules, {'torch_npu': fake}), torch.inference_mode():
        out = stage(hidden, angles.cos(), angles.sin(), mask)
    return out, fake.events, model.state_dict()


class VisionSimplificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = load_vision(OLD, '_step2_vision_control')
        cls.new = load_vision(NEW, '_cleaned_vision_control')
        torch.set_num_threads(1)

    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_stage_outputs_weights_and_call_order(self):
        for mode in ('weight_padded',):
            for batch, seq, real in ((1, 8, 8), (1, 16, 9), (2, 16, 13)):
                for dtype in (torch.float32, torch.float16):
                    with self.subTest(mode=mode, batch=batch, seq=seq, real=real, dtype=dtype):
                        a, calls_a, weights_a = exercise(self.old, mode, batch, seq, real, dtype)
                        b, calls_b, weights_b = exercise(self.new, mode, batch, seq, real, dtype)
                        self.assertTrue(torch.equal(a, b))
                        self.assertEqual(calls_a, calls_b)
                        self.assertEqual(weights_a.keys(), weights_b.keys())
                        for name in weights_a:
                            self.assertTrue(torch.equal(weights_a[name], weights_b[name]), name)

    def test_legacy_forward_removed(self):
        for name in ('PaddleOCRVisionAttention', 'PaddleOCRVisionEncoderLayer',
                     'PaddleOCRVisionEncoder', 'PaddleOCRVisionTransformer', 'PaddleOCRVisionModel'):
            self.assertNotIn('forward', vars(getattr(self.new, name)), name)

    def test_required_mask_and_no_choice_plumbing(self):
        for name in REMOVED_DEFINITIONS:
            self.assertFalse(hasattr(self.new, name), name)
        q = torch.randn(1, 2, 8, 80)
        with self.assertRaises(TypeError):
            self.new.vision_prompt_flash_attention_bnsd(q, q, q, num_heads=2, scale=72**-.5)
        for real, physical in ((8, 8), (8, 16)):
            cfg = PaddleOCRVisionConfig(hidden_size=144, num_attention_heads=2,
                intermediate_size=272, num_hidden_layers=2, image_size=28)
            model = types.SimpleNamespace(visual=vision_model(self.new, cfg))
            hidden = torch.randn(real, 144)
            grid = torch.tensor([[1, 2, real // 2]])
            a = self.old.prepare_vision_prefill(model, hidden, grid, physical_seq_len=physical, execution='test')
            b = self.new.prepare_vision_prefill(model, hidden, grid, physical_seq_len=physical, execution='test')
            self.assertEqual(signature(vars(a)), signature(vars(b)))
            self.assertIsInstance(b.attention_mask, torch.Tensor)
            self.assertEqual(bool(b.attention_mask.any()), real != physical)

    def test_all_other_source_unchanged(self):
        def definitions(source):
            return {n.name: ast.get_source_segment(source, n) for n in ast.parse(source).body
                    if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        retained_old = without_methods(OLD, {name: {'forward'} for name in (
            'PaddleOCRVisionAttention', 'PaddleOCRVisionEncoderLayer',
            'PaddleOCRVisionEncoder', 'PaddleOCRVisionTransformer', 'PaddleOCRVisionModel')})
        old, new = definitions(retained_old), definitions(NEW)
        self.assertEqual(old.keys() - REMOVED_DEFINITIONS, new.keys())
        for name in old.keys() - CHANGED_DEFINITIONS - REMOVED_DEFINITIONS:
            if name == 'PaddleOCRProjector':
                # forward now introduces the operation before its constructor.
                def method_sources(src):
                    return {m.name: ast.get_source_segment(src,m)
                            for m in ast.parse(src).body[0].body
                            if isinstance(m,ast.FunctionDef)}
                a = ast.unparse(ast.fix_missing_locations(FreezeArchitecture('vision').visit(ast.parse(old[name]))))
                b = ast.unparse(ast.parse(new[name]))
                self.assertEqual(method_sources(a),method_sources(b))
                continue
            self.assertEqual(ast.dump(FreezeArchitecture('vision').visit(ast.parse(old[name]))), ast.dump(ast.parse(new[name])), name)
        # The selected padded-head computation and full layer loop stay literal.
        def methods(source):
            return {n.name: ast.get_source_segment(source, n) for n in ast.parse(source).body[0].body
                    if isinstance(n, ast.FunctionDef)}
        a, b = methods(old['VisionPrefillStage']), methods(new['VisionPrefillStage'])
        class WeightPadded(ast.NodeTransformer):
            def visit_Call(self, n):
                if ast.unparse(n.func) == '_activation':
                    return ast.Call(func=ast.Attribute(value=ast.Name(id='F',ctx=ast.Load()),attr='gelu',ctx=ast.Load()),
                        args=[n.args[1]],keywords=[ast.keyword(arg='approximate',value=ast.Constant('tanh'))])
                return self.generic_visit(n)
            def visit_If(self, n):
                if ast.unparse(n.test) == 'self.weight_padded_attention':
                    return [self.visit(x) for x in n.body]
                return self.generic_visit(n)
            def visit_IfExp(self, n):
                if ast.unparse(n.test) == 'self.weight_padded_attention':
                    return self.visit(n.body)
                return self.generic_visit(n)
            def visit_Attribute(self, n):
                if n.attr == '_weight_padded_attention':
                    n.attr = '_attention'
                return self.generic_visit(n)
        for name in ('_weight_padded_attention', 'forward'):
            expected = a[name].replace('def _weight_padded_attention(', 'def _attention(')
            expected = WeightPadded().visit(ast.parse(expected))
            actual = ast.parse(b['_attention' if name == '_weight_padded_attention' else name])
            self.assertEqual(ast.dump(expected), ast.dump(actual), name)
        # Only the linear-patch option was enabled in the serving anchor.
        class LinearPatch(ast.NodeTransformer):
            def visit_Assign(self, n):
                if any(ast.unparse(t) == 'self.linear_patch_projection' for t in n.targets):
                    return None
                return self.generic_visit(n)
            def visit_If(self, n):
                if ast.unparse(n.test) == 'not self.linear_patch_projection':
                    return []
                if n.body and isinstance(n.body[0],ast.Raise) and 'linear patch projection requires' in ast.unparse(n.body[0]):
                    return []
                return self.generic_visit(n)
        def method_order_independent(tree):
            cls = tree.body[0]
            methods = [m for m in cls.body if isinstance(m,ast.FunctionDef)]
            cls.body = [m for m in cls.body if not isinstance(m,ast.FunctionDef)]
            cls.body += sorted(methods,key=lambda m:m.name)
            return ast.dump(tree)
        self.assertEqual(method_order_independent(FreezeArchitecture('vision').visit(LinearPatch().visit(ast.parse(old['PaddleOCRVisionEmbeddings'])))),
                         method_order_independent(ast.parse(new['PaddleOCRVisionEmbeddings'])))
        # Same graph compiler call; only fixed constructor/cache selectors were
        # removed. Constructor behavior is also exercised with a fake compiler.
        def compiler_calls(source):
            return [ast.dump(n) for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Call)
                    and ast.unparse(n.func)=='torchair.inference.cache_compile']
        self.assertEqual(compiler_calls(old['VisionPrefillRuntime']),compiler_calls(new['VisionPrefillRuntime']))

    def test_vision_bucket_routes_and_fixed_cache_inputs(self):
        for name in ('dtype','head_dim','mlp_intermediate_size','linear_weight_format','weight_padded_attention'):
            self.assertNotIn(name, inspect.signature(self.new.vision_cache_dir_for_bucket).parameters)
        for eager in (False, True):
            old = self.old.VisionPrefillRuntime.__new__(self.old.VisionPrefillRuntime)
            old.padding = 'bucket'
            old.backend = 'raw_eager' if eager else 'torchair'
            old.seq_alignment = 128
            old.buckets = (256,384,512,640,768,1408,1920,2048,2944,4096)
            new = self.new.VisionPrefillRuntime.__new__(self.new.VisionPrefillRuntime)
            new.eager = eager
            for length in range(1, 8193):
                self.assertEqual(old.route(length),new.route(length))


if __name__ == '__main__':
    unittest.main(verbosity=2)
