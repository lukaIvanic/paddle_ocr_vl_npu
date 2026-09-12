"""CPU-only singleton prefill contract tests; no Ascend speed/kernel claim."""
from __future__ import annotations
import ast
from contextlib import contextmanager
import html
import re
from collections import Counter
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

import torch
from test_text_simplification import ROOT, EXPERIMENT, signature
import serving_runtime as current
import crop_processing


def reference_runtime():
    source = subprocess.check_output(['git','-C',str(ROOT),'show',
        '0976fa33:19_table_ocr_serving/serving_runtime.py'],text=True)
    ns = dict(vars(current))
    names = {'_InFlightPrefillMember','_TextPrefillInputMember','_TextPackTrace',
             '_InFlightPrefillGroup','_PreparedPrefillGroup','_StagedPrefillGroup',
             'ContinuousRecognizer'}
    nodes = [n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name in names]
    module = types.ModuleType('_old_crop_contract')
    module.__dict__.update(ns)
    # Annotations resolve lazily, as in the original module.
    nodes.insert(0,ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0))
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])), 'old_crop', 'exec'),module.__dict__)
    return module


class CropPipelineTests(unittest.TestCase):
    def test_legacy_metric_schema_without_packing_state(self):
        source = subprocess.check_output(['git','-C',str(ROOT),'show',
            '0976fa33:19_table_ocr_serving/serving_runtime.py'],text=True)
        ns=dict(vars(current))
        nodes=[n for n in ast.parse(source).body if isinstance(n,ast.ClassDef)
               and n.name in ('_VisionPackingRunStats','_TextPackingRunStats')]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'old_stats','exec'),ns)
        a,b=ns['_VisionPackingRunStats']('off',1920,32),current._VisionPrefillStats()
        c,d=ns['_TextPackingRunStats']('off',(128,256,512,1024)),current._TextPrefillStats()
        self.assertEqual(a.summary(),b.summary())
        self.assertEqual(c.summary(),d.summary())
        for execution in ('compiled','eager_overflow','eager_padded'):
            route=dict(execution=execution,real_vision_tokens=100,physical_vision_tokens=256,bucket=256)
            a.record(crops=1,route=route); b.record(route)
            c.groups+=1; c.crops+=1; c.fallback_crops+=1; d.record()
            self.assertEqual(a.summary(),b.summary())
            self.assertEqual(c.summary(),d.summary())

    def test_eager_opt_out_uses_same_stages_without_compiler(self):
        import text_prefill_and_decode as text
        import vision_prefill as vision
        cfg = types.SimpleNamespace(text_config=types.SimpleNamespace(num_hidden_layers=2),
            vision_config=types.SimpleNamespace(hidden_size=144,num_attention_heads=2))
        model = torch.nn.Module()
        model.config=cfg
        model.model=torch.nn.Module()
        model.visual=types.SimpleNamespace(vision_model=torch.nn.Module())
        kwargs=dict(cache_root=Path('/unused'),device=torch.device('cpu'),eager=True)
        with patch.object(text,'import_torchair',side_effect=AssertionError('compiler called')), \
             patch.object(vision,'import_torchair',side_effect=AssertionError('compiler called')):
            t=text.TextPrefillRuntime(model,cache_length=4096,**kwargs)
            v=vision.VisionPrefillRuntime(model,**kwargs)
            stage=text.TextDecodeStage(model)
            fn,metadata=text.compile_text_decode_stage(stage,batch_size=2,cache_length=4096,**kwargs)
        self.assertIs(fn,stage)
        self.assertFalse(metadata['enabled'])
        self.assertEqual(t.route(300)['physical_text_tokens'],512)
        self.assertEqual(v.route(300)['physical_vision_tokens'],384)
        self.assertEqual(t.route(300)['execution'],'eager_padded')
        self.assertEqual(v.route(300)['execution'],'eager_padded')
        self.assertEqual(t.compiled,{})
        self.assertEqual(v.compiled,{})

    def test_html_conversion_unchanged_and_cleanup_removed(self):
        source = subprocess.check_output(['git','-C',str(ROOT),'show',
            '0976fa33:19_table_ocr_serving/_support/pipeline/layout_output.py'],text=True)
        section = source[source.index('def _shortest_repeating_substring('):source.index('def untokenize_table_figures(')]
        now = (EXPERIMENT/'crop_processing.py').read_text()
        # Parser stays verbatim; the renamed/consolidated converter is checked
        # against historical execution, including malformed marker sequences.
        def definitions(src):
            return {n.name: ast.get_source_segment(src,n) for n in ast.parse(src).body
                    if isinstance(n,ast.FunctionDef)}
        expected, actual = definitions(section), definitions(now)
        retained = {'_parse_otsl_rows'}
        self.assertEqual({name:expected[name] for name in retained},
                         {name:actual[name] for name in retained})
        for name in ('truncate_repetitive_content', '_repeating_suffix', '_shortest_repeating_substring'):
            self.assertNotIn(name, actual)
        regex = next(n for n in ast.parse(section).body if isinstance(n,ast.Assign))
        self.assertIn(ast.get_source_segment(section,regex),now)
        self.assertFalse((EXPERIMENT/'_support/pipeline/layout_output.py').exists())
        ns = dict(html=html,re=re,Counter=Counter,Any=object)
        exec(section,ns)
        import random
        rng = random.Random(19)
        fragments = ('<fcel>', '<ecel>', '<lcel>', '<ucel>', '<xcel>', '<nl>',
                     '汉字', r'\(x\)', ' $2 & <tag> "quoted" ', '\n\t', "'", '&amp;')
        for _ in range(1000):
            content = ''.join(rng.choices(fragments, k=rng.randrange(0,100)))
            self.assertEqual(ns['convert_otsl_to_html'](content),
                             crop_processing.convert_otsl_to_html(content), content)
        self.assertNotIn('normalize_recognition_text', actual)
        # Exercise the actual HTTP result callback, without starting a server.
        serve_source = (EXPERIMENT/'serve.py').read_text()
        emit = next(n for n in ast.walk(ast.parse(serve_source))
                    if isinstance(n,ast.FunctionDef) and n.name == 'emit_result')
        emitted = []
        jobs = {}
        scope = dict(Any=object, asdict=dict, request_jobs=jobs,
            convert_otsl_to_html=crop_processing.convert_otsl_to_html,
            time=types.SimpleNamespace(perf_counter=lambda: 1.),
            results=types.SimpleNamespace(put=emitted.append))
        exec(compile(ast.Module(body=[emit],type_ignores=[]), '<http-result>', 'exec'), scope)
        for text in ('','<fcel>A & B<lcel><nl><ucel><xcel><nl>',
                     '<ucel>orphan<nl><fcel>row<ecel>',r'\(x$y\)',
                     ('repeat me\n'*600), 'abc12345'*1000,
                     '  text  with\tspaces\n\n\n next line \n',
                     r'<fcel>$ per oz<fcel>\(x\)<nl>'):
            jobs['test'] = dict(crop_type='table', submitted_monotonic_s=0.)
            class Result(dict):
                request_id = 'test'
            scope['emit_result'](Result(request_id='test', text=text, token_ids=[10,2]))
            payload = emitted[-1]['payload']
            self.assertEqual(payload['text'], ns['convert_otsl_to_html'](text) or text)
            self.assertEqual(payload['raw_text'], text)
            self.assertEqual(payload['token_ids'], [10,2])

    def test_prefill_transfer_compute_and_result_parity(self):
        old = reference_runtime()
        for timeline_enabled in (False,True):
            reports=[]
            for module in (old,current):
                trace=[]
                class Event:
                    def __init__(self,name): self.name=name
                    def synchronize(self): trace.append(('sync',self.name))
                class Stream:
                    def __init__(self,name): self.name=name; self.serial=0
                    def wait_event(self,e): trace.append(('wait',self.name,e.name))
                    def record_event(self):
                        self.serial+=1
                        name=f'{self.name}:{self.serial}'
                        trace.append(('event',name))
                        return Event(name)
                transfer,compute=Stream('transfer'),Stream('compute')
                @contextmanager
                def stream(s):
                    trace.append(('enter',s.name)); yield
                    trace.append(('exit',s.name))
                fake=types.SimpleNamespace(npu=types.SimpleNamespace(stream=stream,current_stream=lambda:compute))
                class DeviceTimeline:
                    def __init__(self,device): self.spans={}
                    def measure(self,key,fn):
                        stage=key.removeprefix('member:0:').removeprefix('group:')
                        trace.append(('stage',stage))
                        out=fn()
                        if isinstance(out,torch.Tensor): trace.append(('tensor',stage,signature(out)))
                        self.spans[key]=dict(seconds=.001,start_ns=0,end_ns=1000000,clock='fake')
                        return out
                    def resolve_spans(self): trace.append(('resolve',)); return self.spans
                engine=module.ContinuousRecognizer.__new__(module.ContinuousRecognizer)
                engine.device=torch.device('cpu')
                engine.prefill_transfer_stream=transfer
                engine.prefill_host_tokens=torch.zeros(32,dtype=torch.int64)
                engine._vision_pack_sequence=engine._prefill_sequence=0
                engine._vision_packing_stats=types.SimpleNamespace(record=lambda **kw:None)
                engine._text_packing_stats=types.SimpleNamespace(groups=0,crops=0,fallback_crops=0)
                engine._vision_prefill_stats=current._VisionPrefillStats()
                engine._text_prefill_stats=current._TextPrefillStats()
                engine.diagnostic_prefill_kv_request_ids=set()
                engine.compact_rescale_factor=1/255
                engine.compact_image_mean=0.5
                engine.compact_image_std=0.5
                engine.timeline=types.SimpleNamespace(record_span=lambda *a,**k:None,
                    record_span_seconds=lambda *a,**k:None) if timeline_enabled else None
                torch.manual_seed(7)
                embedding=torch.nn.Embedding(16,4)
                head=torch.nn.Linear(4,16)
                cache=types.SimpleNamespace(key=torch.zeros(4,4))
                lease=types.SimpleNamespace(cache=cache,slot_index=2,generation=3,release=lambda:None)
                engine.prefill_cache_pool=types.SimpleNamespace(acquire=lambda:lease)
                engine.model=types.SimpleNamespace(
                    config=types.SimpleNamespace(image_token_id=5),
                    model=types.SimpleNamespace(embed_tokens=embedding),lm_head=head,
                    visual=types.SimpleNamespace(dtype=torch.float32,vision_model=types.SimpleNamespace(
                        embeddings=lambda pixels,image_grid_thw:pixels.squeeze(0))),
                    mlp_AR=lambda features,grid:features[:2])
                engine.vision_prefill=types.SimpleNamespace(
                    route=lambda n:dict(execution='compiled',real_vision_tokens=n,physical_vision_tokens=8,
                        padding_vision_tokens=8-n,useful_token_fraction=n/8,bucket=8),
                    prepare=lambda hidden,grid,route:hidden,run_prepared=lambda hidden:hidden)
                def text_run(prepared,cache):
                    cache.key.copy_(prepared[0])
                    return prepared.sum(dim=1,keepdim=True)
                engine.text_prefill=types.SimpleNamespace(
                    route=lambda n:dict(execution='compiled',real_text_tokens=n,physical_text_tokens=128,
                        padding_text_tokens=128-n,useful_token_fraction=n/128,bucket=128),
                    prepare=lambda embeds,mask,pos,route:embeds,run_prepared=text_run)
                prepared=current.CpuPreparedRecognition(
                    request_id='any-crop',prompt='Table Recognition:',crop_size=(42,28),skip_special_tokens=False,
                    pixel_values=torch.arange(16).reshape(4,4).to(torch.uint8),
                    image_grid_thw=torch.tensor([[1,2,2]]),input_ids=torch.tensor([[2,5,5,8]]),
                    attention_mask=torch.ones(1,4,dtype=torch.int64),position_ids=torch.zeros(3,1,4,dtype=torch.int64),
                    rope_deltas=torch.zeros(1,1,dtype=torch.int64),image_token_count=2,
                    timing_s=dict(cpu_image_and_prompt_preprocess=.01,cpu_mrope_index=.01,cpu_pin_memory=.01),
                    request_started=0,preparation_finished=0)
                with patch.dict(sys.modules,{'torch_npu':fake}), patch.dict(module.__dict__, {'IMAGE_TOKEN_ID': 5}), patch.object(module,'DeviceTimeline',DeviceTimeline):
                    if module is old:
                        crop=engine._prepared_group([(prepared,0.0)])
                        staged=engine._stage_prefill_group(crop)
                        inflight=engine._enqueue_staged_prefill_group(staged)
                        result=engine._finalize_prefill_group(inflight)[0]
                    else:
                        crop=engine._prepared_crop(prepared,0.0)
                        staged=engine._stage_crop(crop)
                        inflight=engine._enqueue_crop(staged)
                        result=engine._finalize_crop(inflight)
                fields=('request_id','prompt','crop_size','skip_special_tokens','rope_deltas',
                        'next_cache_position','next_token','first_token','input_tokens','projected_image_tokens',
                        'device_stage_s','input_fingerprints')
                reports.append((trace,signature(cache.key),signature({k:getattr(result,k) for k in fields})))
            self.assertEqual(*reports)


if __name__=='__main__': unittest.main()
