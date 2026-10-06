"""B1 real-pair padding, static TorchAir parity, warm timing and CPU/NPU profiles."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import numpy as np
import torch
from local_modeling_rwkv_embedding import Embedding
from local_modeling_rwkv_reranker import Reranker
from probe_wkv7 import load_bridge
from run_cpu_reference import CReference, sha256
from run_nanoscidocs import save
from run_nanoscidocs_reranker import DATA_SHA256, CANDIDATES_SHA256
from run_reranker_smoke import PREFIX, SUFFIX, require


class Scorer(torch.nn.Module):
    def __init__(self, backbone, reranker):
        super().__init__()
        self.backbone, self.reranker = backbone, reranker

    def forward(self, ids):
        _, state, _ = self.backbone.encode_states(ids)
        return self.reranker(state[1])


def measure(call, repeats):
    for _ in range(2):
        call()
    torch.npu.synchronize()
    samples = []
    for _ in range(repeats):
        before = time.perf_counter()
        call()
        torch.npu.synchronize()
        samples.append(time.perf_counter() - before)
    return {'samples_seconds': samples, 'median_seconds': statistics.median(samples),
            'min_seconds': min(samples), 'p95_seconds': float(np.percentile(samples, 95))}


def profile(call, output, label):
    import torch_npu.profiler as prof
    call(); torch.npu.synchronize()
    with prof.profile(activities=[prof.ProfilerActivity.CPU, prof.ProfilerActivity.NPU],
            schedule=prof.schedule(wait=0, warmup=0, active=1, repeat=1),
            experimental_config=prof._ExperimentalConfig(profiler_level=prof.ProfilerLevel.Level1,
                aic_metrics=prof.AiCMetrics.PipeUtilization, export_type=prof.ExportType.Text),
            on_trace_ready=prof.tensorboard_trace_handler(str(output), analyse_flag=True),
            record_shapes=True, profile_memory=False, with_stack=True) as capture:
        with torch.profiler.record_function(label):
            for _ in range(2):
                call()
        torch.npu.synchronize(); capture.step()
    parser_path = Path(__file__).resolve().parents[1]/'05_full_recognizer_optimizations/parse_npu_profile.py'
    spec = importlib.util.spec_from_file_location('profile_parser', parser_path)
    parser = importlib.util.module_from_spec(spec); spec.loader.exec_module(parser)
    runs = [parser.parse_run(p, topn=15, skip_trace=False) for p in parser.find_run_roots(output)]
    assert runs and all('kernel_details' in r for r in runs), 'Missing analyzed NPU trace'
    save(output/'summary.json', {'parser_sha256': sha256(parser_path), 'profile_iterations': 2, 'runs': runs})
    return {'path': str(output), 'summary_sha256': sha256(output/'summary.json'),
            'iterations': 2, 'kernel_totals': [r['kernel_details'] for r in runs]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint','reranker','runtime','build-root','data','candidates','baseline-result','smoke-result','cache-root','output']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--bucket', type=int, choices=(256,512,1024,2048), required=True)
    p.add_argument('--repeats', type=int, default=10)
    p.add_argument('--profile', action='store_true')
    p.add_argument('--expect-cache', action='store_true')
    p.add_argument('--allow-shared-device', action='store_true')
    args = p.parse_args()
    if not 3 <= args.repeats <= 100: p.error('Use repeats 3..100')
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).parent; start = time.perf_counter(); tokenizer = None
    report = {'source_commit': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
              'source_sha256': {n:sha256(root/n) for n in ['run_reranker_buckets.py','local_modeling_rwkv_embedding.py','local_modeling_rwkv_reranker.py']},
              'batch_size': 1, 'bucket': args.bucket, 'physical_npu': os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
              'dense_dtype': 'fp16', 'state_pointwise_dtype': 'float32', 'document_caching': False,
              'padding': 'left zero to bucket; one terminal EOS; last 2048 for raw overflow',
              'compile_parity_atol': .02, 'compile_parity_rtol': .005, 'cases': [], 'all_checks_passed': False}
    try:
        import torch_npu
        import torchair
        from torchair.configs.compiler_config import CompilerConfig
        from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
        from torchair import ge
        torch.set_num_threads(4)
        assert sha256(args.data) == DATA_SHA256 and sha256(args.candidates) == CANDIDATES_SHA256
        baseline = json.loads(args.baseline_result.read_text()); smoke = json.loads(args.smoke_result.read_text())
        assert baseline['all_checks_passed'] and smoke['all_checks_passed']
        for name in ['local_modeling_rwkv_embedding.py','local_modeling_rwkv_reranker.py']:
            assert report['source_sha256'][name] == smoke['source_sha256'][name]
            # The benchmark predates the compile-only DELETE_DEREF cleanup guard.
            expected = ('14af5fb43576f0b0b505d8fcd20ba1adc255ded9b485c0a845ff8d48c6924f24'
                        if name == 'local_modeling_rwkv_embedding.py' else report['source_sha256'][name])
            assert baseline['source_sha256'][name] == expected
        report['input_sha256'] = {n:sha256(getattr(args,n)) for n in ['checkpoint','reranker','data','candidates','baseline_result','smoke_result']}
        data = json.loads(args.data.read_text()); candidates = json.loads(args.candidates.read_text())
        old_scores = json.loads((args.baseline_result.parent/'pair_scores.json').read_text())
        assert sha256(args.baseline_result.parent/'pair_scores.json') == baseline['artifacts']['pair_scores.json']['sha256']
        corpus = {x['_id']:x['text'].strip() for x in data['corpus']}; queries = {x['_id']:x['text'] for x in data['queries']}
        tokenizer = CReference(args.runtime, 4)
        lower = {256:0,512:256,1024:512,2048:1024}[args.bucket]
        eligible = []
        for qid, query in queries.items():
            for doc in candidates[qid]:
                did = doc['document_id']
                ids = tokenizer.tokenize(PREFIX.format(document=corpus[did])+SUFFIX.format(query=query)).tolist()+[65535]
                ids = ids[-2048:]
                if lower < len(ids) <= args.bucket:
                    eligible.append({'query_id':qid, 'document_id':did, 'input_ids':ids, 'b2_logit':old_scores[qid][did]})
        assert len(eligible) >= 2
        # Include maximum padding, nearest bucket boundary, and the closest B2-scored pair within one query.
        selected = [min(eligible,key=lambda x:len(x['input_ids'])), max(eligible,key=lambda x:len(x['input_ids']))]
        close = []
        for qid in queries:
            ordered = sorted([x for x in eligible if x['query_id']==qid],key=lambda x:x['b2_logit'])
            close.extend((abs(a['b2_logit']-b['b2_logit']),a,b) for a,b in zip(ordered,ordered[1:]))
        pair = min(close,key=lambda x:x[0]) if close else None
        if pair: selected.extend(pair[1:])
        selected = list({(x['query_id'],x['document_id']):x for x in selected}.values())
        report['selection'] = {'eligible_pairs':len(eligible),'selection_basis':'minimum/maximum raw length and closest same-query B2 logits; not a benchmark sample',
                               'close_pair':None if pair is None else [[x['query_id'],x['document_id']] for x in pair[1:]]}
        save(args.output/'inputs.json', selected)
        load_bridge(root/'wkv7_npu', args.build_root)
        status = subprocess.check_output(['/usr/local/bin/npu-status'], text=True)
        chosen = next((x for x in status.splitlines() if x.startswith(f"NPU {report['physical_npu']}: ")), '')
        assert 'Health=OK' in chosen and (args.allow_shared_device or ': free ' in chosen)
        torch.npu.set_device(0); torch.npu.set_compile_mode(jit_compile=False); torch.npu.config.allow_internal_format=False
        free,total = torch.npu.mem_get_info(); assert free >= 2*1024**3
        report.update(device=torch.npu.get_device_name(0),device_status=status,torch=torch.__version__,
                      torch_npu=torch_npu.__version__,torchair=getattr(torchair,'__version__','unknown'),hbm_before={'free_bytes':free,'total_bytes':total})
        backbone = Embedding(args.checkpoint,'npu:0',torch.float16)
        reranker = Reranker(args.reranker,'npu:0',torch.float16); model = Scorer(backbone,reranker).eval()
        @register_fx_node_ge_converter(torch.ops.rwkv_reference.wkv7.default)
        def convert(k,v,w,r,a,b,hi,meta_outputs=None):
            return ge.custom_op('RwkvReferenceWkv7',inputs=dict(k=k,v=v,w=w,r=r,a=a,b=b,hi=hi),attrs={},outputs=['o','ho'])
        key = hashlib.sha256(json.dumps({'sources':report['source_sha256'],'weights':[report['input_sha256'][n] for n in ['checkpoint','reranker']],
              'runtime':[report[n] for n in ['torch','torch_npu','torchair','device']], 'precision':'fp16_fp32_native'},sort_keys=True).encode()).hexdigest()[:16]
        cache = args.cache_root/f'{key}_b1_t{args.bucket}'
        before_files = {str(f.relative_to(cache)):f.stat().st_size for f in cache.rglob('*') if f.is_file()} if cache.exists() else {}
        if args.expect_cache: assert before_files, 'Expected persistent cache is empty'
        report['compile'] = {'cache_dir':str(cache),'cache_files_before':before_files,'dynamic':False,'fullgraph':True,'ge_cache':True}
        forward = torchair.inference.cache_compile(model.forward,config=CompilerConfig(),dynamic=False,
                                                  fullgraph=True,cache_dir=str(cache),ge_cache=True)
        with torch.inference_mode():
            ni = None
            for case in selected:
                raw = torch.tensor([case['input_ids']],dtype=torch.long,device='npu')
                padded = torch.tensor([[0]*(args.bucket-raw.shape[1])+case['input_ids']],dtype=torch.long,device='npu')
                unpadded = model(raw); eager = model(padded); torch.npu.synchronize()
                print('COMPILE_START',json.dumps({'bucket':args.bucket,'cache_expected':args.expect_cache,'raw_tokens':raw.shape[1]}),flush=True)
                before = time.perf_counter(); actual = forward(padded); torch.npu.synchronize(); first = time.perf_counter()-before
                parity = require(actual,eager,.02,.005)
                assert torch.equal(forward(padded),actual), 'Repeated compiled scoring changed'
                a,b = float(unpadded.item()),float(eager.item())
                row = {k:case[k] for k in ['query_id','document_id','b2_logit']}
                row.update(raw_tokens=raw.shape[1],padding_tokens=args.bucket-raw.shape[1],raw_logit=a,padded_logit=b,
                           padding_logit_delta=b-a,compiled_logit=float(actual.item()),compiled_parity=parity,
                           first_call_seconds=first,eager=measure(lambda:model(padded),args.repeats),
                           compiled=measure(lambda:forward(padded),args.repeats))
                report['cases'].append(row); save(args.output/'result.json',report)
                print('CASE',json.dumps(row),flush=True)
                ni = padded
            _,state,_ = backbone.encode_states(ni)
            report['stage_timings'] = {'backbone_eager':measure(lambda:backbone.encode_states(ni),args.repeats),
                                      'reranker_eager':measure(lambda:reranker(state[1]),args.repeats),
                                      'timing_scope':'prepared NPU inputs; synchronized warm calls; profiles excluded; last selected pair'}
            if pair:
                matched = [next(x for x in report['cases'] if (x['query_id'],x['document_id'])==(c['query_id'],c['document_id'])) for c in pair[1:]]
                report['close_pair_order'] = {k:float(np.sign(matched[0][k]-matched[1][k])) for k in ['b2_logit','raw_logit','padded_logit','compiled_logit']}
            if args.profile:
                def stages():
                    with torch.profiler.record_function('rwkv.backbone'):
                        _,s,_ = backbone.encode_states(ni)
                    with torch.profiler.record_function('rwkv.reranker'):
                        return reranker(s[1])
                report['profiles'] = {'eager':profile(stages,args.output/'profile_eager','rwkv.eager_score'),
                                      'compiled':profile(lambda:forward(ni),args.output/'profile_compiled','rwkv.compiled_score')}
            report['compile']['cache_files_after'] = {str(f.relative_to(cache)):f.stat().st_size for f in cache.rglob('*') if f.is_file()}
            assert report['compile']['cache_files_after']
            report['all_checks_passed'] = True
            print('SUMMARY',json.dumps({'bucket':args.bucket,'passed':True,'maximum_padding_logit_delta':max(abs(x['padding_logit_delta']) for x in report['cases']),
                  'eager_ms':[1000*x['eager']['median_seconds'] for x in report['cases']],
                  'compiled_ms':[1000*x['compiled']['median_seconds'] for x in report['cases']],
                  'close_pair_order':report.get('close_pair_order'),'stages':report['stage_timings']}),flush=True)
    except Exception as e:
        report['error'] = f'{type(e).__name__}: {e}'; raise
    finally:
        if tokenizer is not None: tokenizer.close()
        report['total_seconds'] = time.perf_counter()-start
        save(args.output/'result.json',report)

if __name__ == '__main__': main()
