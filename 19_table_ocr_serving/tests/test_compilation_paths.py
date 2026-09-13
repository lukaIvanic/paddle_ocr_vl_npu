"""CPU setup tests: exact paths and unchanged compiler/warmup calls, no NPU."""
from __future__ import annotations
import ast
from contextlib import ExitStack
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from test_text_simplification import ROOT, EXPERIMENT, signature
import p04_paddle_ocr_vl_1_6_modeling as modeling
import p06_text_prefill_and_decode as text
import p05_vision_prefill as vision


class CompilationPathTests(unittest.TestCase):
    def test_model_setup_owns_final_paths_and_stage_order(self):
        calls = []
        def runtime(name):
            def create(model, **kwargs):
                calls.append((name, kwargs))
                return types.SimpleNamespace(setup_timing_s={'compile_wrapper':0., 'compile_first_call':0.})
            return create

        for head in ('selected_vocab_60416_abc', 'full_vocab_103424'):
            for batch in (2, 8):
                calls.clear()
                with patch.object(modeling, 'model_source_hash', return_value='source123'), \
                     patch.object(modeling, 'torch_npu', types.SimpleNamespace(npu=types.SimpleNamespace(synchronize=lambda device:None))), \
                     patch.object(modeling, 'VisionPrefillRuntime', runtime('vision')), \
                     patch.object(modeling, 'TextPrefillRuntime', runtime('prefill')), \
                     patch.object(modeling, 'TextDecodeRuntime', runtime('decode')):
                    modeling.LocalPaddleOCRVLForConditionalGeneration.make_inference_stages(
                        object(), graph_cache_directory=Path('/cache'), decode_head_cache_key=head,
                        batch_size=batch, cache_length=4096, device=torch.device('cpu'), eager=True)
                self.assertEqual([name for name, _ in calls], ['vision', 'prefill', 'decode'])
                self.assertEqual(calls[0][1]['graph_directories'],
                    {b:Path('/cache/source123/vision_prefill')/f'seq{b}' for b in vision.VISION_BUCKETS})
                self.assertEqual(calls[1][1]['graph_directories'],
                    {b:Path('/cache/source123/text_prefill')/f'seq{b}_kv4096' for b in text.TEXT_PREFILL_BUCKETS})
                self.assertEqual(calls[2][1]['graph_directory'], Path('/cache/source123/decode')/head/f'b{batch}_kv4096')

    def test_source_fingerprint_covers_all_three_files(self):
        filenames = ('p04_paddle_ocr_vl_1_6_modeling.py', 'p05_vision_prefill.py', 'p06_text_prefill_and_decode.py')
        original = {name:(EXPERIMENT/name).read_bytes() for name in filenames}
        expected = hashlib.sha256()
        for name in filenames:
            expected.update(name.encode()); expected.update(original[name])
        baseline = modeling.model_source_hash()
        self.assertEqual(baseline, expected.hexdigest()[:12])
        for changed in filenames:
            def read(path):
                return original[path.name] + (b'\n# edited\n' if path.name == changed else b'')
            with patch.object(Path, 'read_bytes', read):
                self.assertNotEqual(modeling.model_source_hash(), baseline)
        with patch.object(Path, 'read_bytes', side_effect=FileNotFoundError):
            with self.assertRaises(FileNotFoundError):
                modeling.model_source_hash()
        self.assertFalse((EXPERIMENT/'_support/model/compile_utils.py').exists())

    def test_vision_and_decode_setup_matches_previous_calls(self):
        # The previous setup is replayed with the same stage modules and tiny
        # CPU tensors. Only directory names and removed metadata may differ.
        for module, names, runtime_name in (
            (vision, {'VisionPrefillRuntime', 'vision_cache_dir_for_bucket', 'vision_source_hash'}, 'VisionPrefillRuntime'),
            (text, {'TextDecodeRuntime', 'compile_text_decode_stage', 'torchair_cache_dir_for_shape', 'text_source_hash'}, 'TextDecodeRuntime'),
        ):
            with self.subTest(runtime=runtime_name), tempfile.TemporaryDirectory() as tmp:
                old = types.ModuleType('historical_setup')
                old.__dict__.update(vars(module))
                helpers = subprocess.check_output(['git','-C',str(ROOT),'show',
                    '564da03f:19_table_ocr_serving/_support/model/compile_utils.py'],text=True)
                exec(compile(helpers, 'old_helpers', 'exec'), old.__dict__)
                historical_filename = 'vision_prefill.py' if module is vision else 'text_prefill_and_decode.py'
                source = subprocess.check_output(['git','-C',str(ROOT),'show',
                    f'564da03f:19_table_ocr_serving/{historical_filename}'],text=True)
                nodes = [n for n in ast.parse(source).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
                exec(compile(ast.Module(body=nodes,type_ignores=[]),'old_setup','exec'),old.__dict__)
                traces = []
                for selected in (old, module):
                    events = []
                    paths = []
                    def compile_graph(fn, **kwargs):
                        paths.append(kwargs['cache_dir'])
                        events.append(('compile',fn.__func__.__name__,{k:v for k,v in kwargs.items() if k not in ('config','cache_dir')}))
                        self.assertIsInstance(kwargs['config'], dict)
                        def run(*inputs):
                            events.append(('warm',signature(inputs)))
                        return run
                    compiler = types.ModuleType('torchair'); compiler.__path__ = []
                    compiler.CompilerConfig = dict
                    compiler.inference = types.ModuleType('torchair.inference')
                    compiler.inference.cache_compile = compile_graph
                    model = torch.nn.Module(); model.model = torch.nn.Module()
                    model.visual = types.SimpleNamespace(vision_model=torch.nn.Module())
                    def allocate(**kwargs):
                        events.append(('allocate',kwargs))
                        shape = (kwargs['batch_size'],1,kwargs['cache_length'],1)
                        return text.LocalPaddleOCRVLStaticCache((torch.zeros(shape),),(torch.zeros(shape),),kwargs['cache_length'])
                    model.allocate_static_cache = allocate
                    with ExitStack() as stack:
                        stack.enter_context(patch.dict(sys.modules,{'torchair':compiler,'torchair.inference':compiler.inference}))
                        stack.enter_context(patch.object(selected,'import_torchair',return_value=(compiler,dict),create=True))
                        sync = lambda device:events.append(('sync',str(device)))
                        stack.enter_context(patch.object(selected,'synchronize',side_effect=sync,create=True))
                        stack.enter_context(patch.object(selected,'torch_npu',types.SimpleNamespace(npu=types.SimpleNamespace(synchronize=sync))))
                        kwargs = dict(device=torch.device('cpu'))
                        if module is vision:
                            stack.enter_context(patch.multiple(selected,VISION_BUCKETS=(4,8),VISION_HIDDEN_SIZE=8,VISION_HEADS=2))
                            dirs = {b:Path(tmp)/f'exact-vision-{b}' for b in (4,8)}
                            kwargs.update(cache_root=Path(tmp)) if selected is old else kwargs.update(graph_directories=dirs)
                            expected_paths = [str(p) for p in dirs.values()]
                        else:
                            for name in ('prepare_decode_rope_factor_lut','prepare_decode_weight_prefetch'):
                                stack.enter_context(patch.object(selected,name,side_effect=lambda *a,_name=name,**k:events.append((_name,k))))
                            kwargs.update(batch_size=2,cache_length=16)
                            directory = Path(tmp)/'exact-decode'
                            kwargs.update(cache_root=Path(tmp)) if selected is old else kwargs.update(graph_directory=directory)
                            expected_paths = [str(directory)]
                        getattr(selected,runtime_name)(model,**kwargs)
                    if selected is module:
                        self.assertEqual(paths,expected_paths)
                    traces.append(events)
                self.assertEqual(*traces)


if __name__ == '__main__':
    unittest.main()
