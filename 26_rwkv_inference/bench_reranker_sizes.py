"""Background 90M regression gate, then 317M/1.3B eager and TorchAir speed probes."""
import argparse
import gc
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import torch
from local_modeling_rwkv_embedding import Embedding
from local_modeling_rwkv_reranker import Reranker, CHECKPOINT_SHA256 as TINY_RANKER_SHA
from run_cpu_reference import CHECKPOINT_SHA256 as TINY_EMBED_SHA, sha256
from run_nanoscidocs import save
from run_reranker_smoke import cpu_models, metrics, require
from probe_reranker_endpoint import Backbone, compiled, measure
from probe_wkv7 import load_bridge
from wkv7_endpoint import load_endpoint, register_converter

SIZES = {'tiny': ('rwkv0b1-emb-curriculum.pth', 'rwkv0b1-reranker.pth', 12, 768),
         'base': ('rwkv0b4-emb-curriculum.pth', 'rwkv0b3-reranker.pth', 24, 1024),
         'large': ('rwkv1b4-emb-curriculum.pth', 'rwkv1b3-reranker.pth', 24, 2048)}


def require_state(actual, expected, dtype):
    """FP16 state gate uses aggregate error; scores keep their original gate."""
    if dtype != 'fp16':
        return require(actual, expected, .02, .005)
    check = metrics(actual, expected)
    check['normalized_rmse_limit'] = .002
    assert check['finite'] and check['normalized_rmse'] <= .002, check
    return check


def pinned_pair(args):
    manifest = json.loads((Path(__file__).parent/'data/large_checkpoints.json').read_text())
    hashes = {f['rfilename']: f['lfs']['sha256'] for f in manifest['files']}
    hashes.update({'rwkv0b1-emb-curriculum.pth': TINY_EMBED_SHA, 'rwkv0b1-reranker.pth': TINY_RANKER_SHA})
    emb, rank, depth, width = SIZES[args.size]
    return args.models/emb, args.models/rank, hashes[emb], hashes[rank], depth, width


def cpu_gate(args, model, ranker, tokens):
    """Sixteen real input tokens; independent upstream FP32 math, not a CPU fallback."""
    ids = tokens[:8] + tokens[-8:]
    emb_path, rank_path, *_ = pinned_pair(args)
    oracle_args = SimpleNamespace(checkpoint=emb_path, reranker=rank_path, upstream=args.upstream)
    before = time.perf_counter()
    cpu, cpu_ranker, indices = cpu_models(oracle_args)
    state = cpu.generate_zero_state(1)
    cpu_hidden = cpu.forward_seq_batch([ids], state, True)
    expected = cpu_ranker(torch.stack([state[1][i] for i in indices])).reshape(-1)
    hidden, actual_state, _ = model.encode_states(torch.tensor([ids], dtype=torch.long, device='npu'))
    actual = ranker(actual_state[1]); torch.npu.synchronize()
    state_checks = [metrics(a, b) for a, b in zip(actual_state, state)]
    hidden_check = metrics(hidden, cpu_hidden)
    assert all(c['finite'] and c['normalized_rmse'] <= .02 for c in state_checks+[hidden_check])
    logit_check = require(actual, expected, .05, .01)
    row = dict(input_ids=ids, states=state_checks, hidden=hidden_check, logits=logit_check,
               cpu_logits=expected.tolist(), npu_logits=actual.cpu().tolist(), seconds=time.perf_counter()-before)
    del cpu, cpu_ranker, state, cpu_hidden, expected, hidden, actual_state, actual
    gc.collect()
    print('CPU_GATE', json.dumps(row), flush=True)
    return row


def profile_pipeline(args, report, cases, static_tokens, encode, head, ids, lengths):
    """Warm real-batch pipeline and device-input forward traces for each backend."""
    from run_reranker_buckets import profile
    from run_cpu_reference import CReference
    from run_reranker_smoke import PREFIX, SUFFIX
    from run_nanoscidocs_reranker import DATA_SHA256
    report['profile_source_sha256']={n:sha256(Path(__file__).parent/n) for n in ['run_reranker_buckets.py','run_cpu_reference.py','run_nanoscidocs_reranker.py']}
    report['profile_parser_sha256']=sha256(Path(__file__).parents[1]/'05_full_recognizer_optimizations/parse_npu_profile.py')
    assert sha256(args.profile_data)==DATA_SHA256
    before=time.perf_counter()
    data=json.loads(args.profile_data.read_text())
    corpus={r['_id']:r['text'].strip() for r in data['corpus']}
    queries={r['_id']:r['text'] for r in data['queries']}
    fixture=args.output/'pipeline_pairs.json'
    save(fixture,[dict(query=queries[c['query_id']],document=corpus[c['document_id']]) for c in cases])
    backend=report['profiling_backend']
    tokenizer=CReference(args.runtime,4)
    report['pipeline_setup_seconds']=time.perf_counter()-before
    def prepare():
        with torch.profiler.record_function('rwkv.disk_read_pair'):
            pairs=json.loads(fixture.read_text())
        with torch.profiler.record_function('rwkv.cpu_tokenize'):
            rows=[(tokenizer.tokenize(PREFIX.format(**pair)+SUFFIX.format(**pair)).tolist()+[65535])[-2048:] for pair in pairs]
        with torch.profiler.record_function('rwkv.cpu_batch_prepare'):
            assert all(len(tokens)<=static_tokens for tokens in rows)
            cpu_ids=torch.tensor([tokens+[0]*(static_tokens-len(tokens)) for tokens in rows],dtype=torch.long)
            cpu_lengths=torch.tensor([len(tokens) for tokens in rows],dtype=torch.int32)
        return cpu_ids,cpu_lengths
    def forward(ni,lens):
        with torch.profiler.record_function('rwkv.'+backend+'_backbone'):
            states=encode(ni,lens)
        with torch.profiler.record_function('rwkv.'+backend+'_head'):
            return head(states[1])
    def pipeline():
        cpu_ids,cpu_lengths=prepare()
        with torch.profiler.record_function('rwkv.h2d'):
            ni=cpu_ids.to('npu');lens=cpu_lengths.to('npu')
        logits=forward(ni,lens)
        with torch.profiler.record_function('rwkv.d2h_scores'):
            values=logits.cpu().tolist()
        with torch.profiler.record_function('rwkv.disk_write_scores'):
            save(args.output/'pipeline_score.json',values)
        return values
    try:
        ci,cl=prepare()
        assert ci.tolist()==ids.cpu().tolist() and cl.tolist()==lengths.cpu().tolist(), 'Real-text tokenizer differs from validated input'
        expected=head(encode(ids,lengths)[1]).cpu().tolist()
        assert pipeline()==expected
        report['timings'][backend+'_pipeline_total']=measure(pipeline,args.repeats,len(cases))
        report['pipeline_scope']='Warm filesystem read of a real NanoSCIDOCS batch, CPU tokenization/batch construction, synchronous pageable H2D, uncached selected-backend backbone/head, D2H scores, JSON write without fsync. Excludes startup, dataset indexing, tokenizer initialization, compilation and profiling. Device-event elapsed can include CPU gaps.'
        report.setdefault('profiles',{})
        for label,call in [('forward',lambda:forward(ids,lengths)),('pipeline',pipeline)]:
            print('PROFILE_START',label,flush=True)
            key=backend+'_'+label
            report['profiles'][key]=profile(call,args.output/('profile_'+key),'rwkv.'+key,
                warmup_iterations=args.profile_warmup,active_iterations=args.profile_active)
            save(args.output/'result.json',report)
            print('PROFILE_DONE',label,flush=True)
        assert pipeline()==expected, 'Profiling changed scores'
    finally:tokenizer.close()


def worker(args):
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).parent; start = time.perf_counter()
    report = dict(size=args.size, dtype=args.dtype, batch_size=1 if args.gate_only else args.batch_size, gate_only=args.gate_only, bucket=args.bucket,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_sha256={n: sha256(root/n) for n in ['bench_reranker_sizes.py', 'local_modeling_rwkv_embedding.py',
            'local_modeling_rwkv_reranker.py', 'probe_reranker_endpoint.py', 'run_reranker_smoke.py', 'wkv7_endpoint.py']},
        physical_npu=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'), shared_device=args.allow_shared_device, all_checks_passed=False,
        backend=args.backend, scope='Prepared device inputs; uncached backbone plus state-readout head. Synchronized steady calls exclude CPU checks, compile, input transfers and tokenization.',
        recurrence=args.recurrence, matrix_chunk_size=args.matrix_chunk_size,
        matrix_compute_dtype=args.matrix_compute_dtype, state_gate='FP16 state normalized RMSE <=0.002; FP32 allclose atol0.02/rtol0.005; all scores allclose atol0.02/rtol0.005; not a full-suite accuracy claim', timings={})
    try:
        import torch_npu
        torch.set_num_threads(4)
        status = subprocess.check_output(['/usr/local/bin/npu-status'], text=True)
        selected = next(s for s in status.splitlines() if s.startswith('NPU '+report['physical_npu']+': '))
        assert 'Health=OK' in selected and (args.allow_shared_device or ': free ' in selected), selected
        report['npu_status_before']=selected
        torch.npu.set_device(0); torch.npu.set_compile_mode(jit_compile=False)
        torch.npu.config.allow_internal_format=False; torch.npu.matmul.allow_hf32=False
        free, total = torch.npu.mem_get_info()
        if args.allow_shared_device:
            assert free > 6*1024**3, 'Shared probe requires at least 6 GiB free'
            budget=min(free-2*1024**3,6*1024**3)
            torch.npu.set_per_process_memory_fraction(budget/total,0)
            report['allocator_budget_bytes']=budget
        else:assert free > 24*1024**3
        report.update(device=torch.npu.get_device_name(0), torch=torch.__version__, torch_npu=torch_npu.__version__,
                      hbm_before=dict(free_bytes=free, total_bytes=total), state_dtype='fp32')
        load_bridge(root/'wkv7_npu', args.reference_build); load_endpoint(args.build_root); register_converter()
        emb, rank, emb_sha, rank_sha, depth, width = pinned_pair(args)
        dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}[args.dtype]
        before=time.perf_counter()
        model = Embedding(emb, 'npu:0', dtype, expected_sha256=emb_sha)
        ranker = Reranker(rank, 'npu:0', dtype, expected_sha256=rank_sha)
        torch.npu.synchronize()
        report['checkpoint_hash_load_convert_h2d_seconds']=time.perf_counter()-before
        assert (model.depth, model.width, ranker.depth, ranker.width) == (depth, width, depth, width)
        assert all(0 <= i < model.depth for i in ranker.layer_indices)
        report.update(checkpoint_sha256=emb_sha, reranker_sha256=rank_sha,
                      architecture=dict(backbone_layers=depth, reranker_layers=ranker.depth, width=width,
                                        heads=model.heads, head_size=64, selected_layers=ranker.layer_indices))
        case_dir = args.cases_root/f'reranker_b1_t{args.bucket}_cold_ea9aa394/probe'
        assert json.loads((case_dir/'result.json').read_text())['all_checks_passed']
        cases = json.loads((case_dir/'inputs.json').read_text())[:args.batch_size]
        assert len(cases)==args.batch_size and all(16 <= len(c['input_ids']) <= args.bucket and c['input_ids'][-1]==65535 for c in cases)
        report['cases_sha256'] = sha256(case_dir/'inputs.json')
        with torch.inference_mode():
            report['cpu_gate'] = cpu_gate(args, model, ranker, cases[0]['input_ids'])
            if args.gate_only:
                report['all_checks_passed']=True
                return
            backbone = Backbone(model).eval()
            lengths = torch.tensor([len(c['input_ids']) for c in cases], dtype=torch.int32, device='npu')
            static_tokens=len(cases[0]['input_ids']) if args.exact_input_shape else args.bucket
            report.update(static_tokens=static_tokens,exact_input_shape=args.exact_input_shape)
            ids = torch.tensor([c['input_ids']+[0]*(static_tokens-len(c['input_ids'])) for c in cases], dtype=torch.long, device='npu')
            save(args.output/'inputs.json', [dict(query_id=c['query_id'], document_id=c['document_id'],
                input_ids=c['input_ids'], valid_length=len(c['input_ids']), bucket=args.bucket,static_tokens=static_tokens) for c in cases])
            report['valid_tokens'] = lengths.cpu().tolist()
            eager = backbone(ids, lengths); expected = ranker(eager[1])
            singles = []
            report['padding_state_checks'] = []
            for i,c in enumerate(cases):
                _, state, _ = model.encode_states(torch.tensor([c['input_ids']], dtype=torch.long, device='npu'))
                for j, x in enumerate(state):
                    batched = eager[j][:,:,i:i+1] if j==0 else eager[j][:,i:i+1]
                    report['padding_state_checks'].append(dict(row=i, state=j, comparison=metrics(batched,x)))
                    save(args.output/'result.json',report)
                    require_state(batched, x, args.dtype)
                singles.append(ranker(state[1]))
            report['padding_logit_diagnostics'] = dict(comparison=metrics(expected,torch.cat(singles)), padded=expected.cpu().tolist(), unpadded=torch.cat(singles).cpu().tolist())
            save(args.output/'result.json',report)
            report['batch_and_right_padding_vs_single'] = require(expected, torch.cat(singles), .02, .005)
            del singles, state
            assert torch.equal(expected, ranker(backbone(ids,lengths)[1]))
            report['raw_checks_passed']=True
            if args.recurrence == 'matrix':
                from matrix_recurrence import MatrixRecurrence
                report['matrix_source_sha256']={n:sha256(root/n) for n in
                    ['matrix_recurrence.py','wkv7_matrix/rwkv7_chunk_scan.py','wkv7_matrix/provenance.json']}
                matrix_dtype={'fp32':torch.float32,'fp16':torch.float16,'bf16':torch.bfloat16}[args.matrix_compute_dtype]
                for block in model.blocks:
                    block.matrix_recurrence=MatrixRecurrence(args.matrix_chunk_size,matrix_dtype)
                candidate=backbone(ids,lengths);candidate_logits=ranker(candidate[1])
                report['matrix_vs_vector_state_diagnostics']=[metrics(a,b) for a,b in zip(candidate,eager)]
                report['matrix_vs_vector_logit_diagnostics']=dict(comparison=metrics(candidate_logits,expected),actual=candidate_logits.cpu().tolist(),expected=expected.cpu().tolist())
                save(args.output/'result.json',report)
                report['matrix_vs_vector_states']=[require_state(a,b,args.matrix_compute_dtype) for a,b in zip(candidate,eager)]
                report['matrix_vs_vector_logits']=require(candidate_logits,expected,.02,.005)
                report['matrix_logits']=candidate_logits.cpu().tolist()
                report['vector_logits']=expected.cpu().tolist()
                # Full-model continuation on the first real document/query pair;
                # correctness only, outside every benchmark/profiler window.
                real=torch.tensor([cases[0]['input_ids']],device='npu',dtype=torch.long)
                split=(real.shape[1]//2//args.matrix_chunk_size)*args.matrix_chunk_size
                if split:
                    _,prefix,_=model.encode_states(real[:,:split])
                    _,continued,_=model.encode_states(real[:,split:],state=prefix)
                    _,whole,_=model.encode_states(real)
                    report['matrix_full_model_continuation']=[require_state(a,b,args.matrix_compute_dtype) for a,b in zip(continued,whole)]
                    require(ranker(continued[1]),ranker(whole[1]),.02,.005)
                    del prefix,continued,whole
                eager,expected=candidate,candidate_logits
                del real
                assert torch.equal(expected,ranker(backbone(ids,lengths)[1]))
            if args.diagnose_matrix_compile:
                assert args.recurrence == 'matrix'
                from matrix_recurrence import DiagnosticScoring
                full = DiagnosticScoring(backbone, ranker)
                expected_diagnostic = full(ids, lengths)
                names = [name for name, _ in full.selected.diagnostic_tensors]
                print('DIAGNOSTIC_COMPILE_START', flush=True)
                compiled_full = compiled(full.forward, args.output/'diagnostic_full_cache')
                actual_diagnostic = compiled_full(ids, lengths)
                torch.npu.synchronize()
                rows = []
                for name, actual_tensor, expected_tensor in zip(names, actual_diagnostic[2], expected_diagnostic[2]):
                    rows.append(dict(name=name, shape=list(actual_tensor.shape),
                        eager_nonfinite=int((~torch.isfinite(expected_tensor)).sum().cpu()),
                        compiled_nonfinite=int((~torch.isfinite(actual_tensor)).sum().cpu()),
                        comparison=metrics(actual_tensor, expected_tensor)))
                report['matrix_compile_diagnostic'] = dict(scope='Complete real-input backbone and head with first-layer diagnostic outputs; compiler configuration unchanged. Instrumented correctness only, no speed result.',
                    tensors=rows, logits=metrics(actual_diagnostic[0], expected_diagnostic[0]),
                    states=[metrics(a,b) for a,b in zip(actual_diagnostic[1], expected_diagnostic[1])])
                save(args.output/'matrix_compile_diagnostic.json', report['matrix_compile_diagnostic'])
                report['all_checks_passed'] = all(row['comparison']['finite'] and row['comparison']['normalized_rmse'] <= .02 for row in rows)
                return
            report['timings']['eager_total'] = measure(lambda: ranker(backbone(ids,lengths)[1]), args.repeats, args.batch_size)
            print('EAGER_TIMING', json.dumps(report['timings']['eager_total']), flush=True)
            save(args.output/'result.json', report)
            if args.profile:
                report['profiling_backend']='raw_eager'
                profile_pipeline(args,report,cases,static_tokens,backbone,ranker,ids,lengths)
            if args.backend=='raw_eager':
                report['all_checks_passed']=True
                return
            if args.warm_cache_from:
                prior=json.loads((args.warm_cache_from/'result.json').read_text())
                assert prior['all_checks_passed'] and prior['dtype']==args.dtype and prior['batch_size']==args.batch_size and prior['static_tokens']==static_tokens
                assert prior['checkpoint_sha256']==emb_sha and prior['reranker_sha256']==rank_sha
                for n in ['local_modeling_rwkv_embedding.py','local_modeling_rwkv_reranker.py','probe_reranker_endpoint.py','wkv7_endpoint.py']:
                    assert prior['source_sha256'][n]==report['source_sha256'][n]
                before=time.perf_counter()
                for n in ['backbone_cache','head_cache']:shutil.copytree(args.warm_cache_from/n,args.output/n)
                report['copy_private_graph_cache_seconds']=time.perf_counter()-before
            before=time.perf_counter();print('COMPILE_START',args.size,args.dtype,args.bucket,flush=True)
            encode=compiled(backbone.forward,args.output/'backbone_cache'); head=compiled(ranker.forward,args.output/'head_cache')
            actual=encode(ids,lengths); logits=head(actual[1]); torch.npu.synchronize()
            report['compile_and_first_call_seconds']=time.perf_counter()-before
            report['compiled_state_diagnostics']=[metrics(a,b) for a,b in zip(actual,eager)]
            report['compiled_logit_diagnostics']=dict(metrics(logits,expected),actual=logits.cpu().tolist(),expected=expected.cpu().tolist())
            report['compiled_layer_diagnostics']=[dict(layer=i,
                shifts=metrics(actual[0][i],eager[0][i]),
                matrix=metrics(actual[1][i],eager[1][i])) for i in range(depth)]
            save(args.output/'result.json',report)
            if args.profile_invalid_compile and not report['compiled_logit_diagnostics']['finite']:
                from run_reranker_buckets import profile
                report['invalid_compiled_profile']=profile(lambda:head(encode(ids,lengths)[1]),
                    args.output/'profile_invalid_torchair_forward','rwkv.invalid_torchair_forward',
                    warmup_iterations=args.profile_warmup,active_iterations=args.profile_active)
                report['invalid_compiled_profile']['valid_for_speed_comparison']=False
                save(args.output/'result.json',report)
            report['compiled_states_vs_eager']=[require_state(a,b,args.dtype) for a,b in zip(actual,eager)]
            report['compiled_logits_vs_eager']=require(logits,expected,.02,.005)
            assert torch.equal(head(encode(ids,lengths)[1]),logits)
            fixed=actual[1].clone().contiguous()
            report['timings']['torchair_total']=measure(lambda:head(encode(ids,lengths)[1]),args.repeats,args.batch_size)
            # Candidate comparison measures complete scoring, never isolated kernels.
            if args.recurrence == 'vector':
                report['timings']['torchair_head_only']=measure(lambda:head(fixed),args.repeats,args.batch_size)
            assert torch.equal(head(encode(ids,lengths)[1]),logits)
            if args.profile:
                report['profiling_backend']='torchair'
                profile_pipeline(args,report,cases,static_tokens,encode,head,ids,lengths)
            report['all_checks_passed']=True
            print('TIMINGS',json.dumps(report['timings']),flush=True)
    except Exception as e:
        report['error']=f'{type(e).__name__}: {e}'
        raise
    finally:
        report['total_seconds']=time.perf_counter()-start
        if hasattr(torch, 'npu') and torch.npu.is_initialized():
            report['peak_hbm_bytes']=dict(allocated=torch.npu.max_memory_allocated(),reserved=torch.npu.max_memory_reserved())
        save(args.output/'result.json',report)
        print('WORKER_RESULT',json.dumps({k:v for k,v in report.items() if k not in ['source_sha256','timings']}),flush=True)


def free_device(args):
    deadline=time.monotonic()+args.idle_wait_seconds
    while time.monotonic()<deadline:
        status=subprocess.check_output(['/usr/local/bin/npu-status'],text=True)
        for device in args.devices:
            if any(s.startswith(f'NPU {device}: free ') and 'Health=OK' in s for s in status.splitlines()):return device
        print('WAITING_FOR_IDLE_NPU',flush=True);time.sleep(60)
    raise RuntimeError('No idle NPU within the configured wait period')


def coordinate(args):
    args.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    report=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                status='waiting_for_downloads',runs=[],all_checks_passed=False)
    def persist():save(args.output/'result.json',report)
    persist();deadline=time.monotonic()+1800
    try:
        while True:
            if args.download_status.exists():
                download=json.loads(args.download_status.read_text())
                if download.get('error'):raise RuntimeError('Checkpoint download failed: '+download['error'])
                if download.get('complete'):break
            if time.monotonic()>deadline:raise TimeoutError('Checkpoint downloads exceeded 30 minutes')
            time.sleep(10)
        report['download_status_sha256']=sha256(args.download_status)
        matrix=[('tiny','fp32',512,True)]+[(size,dtype,T,False)
            for size in ['base','large'] for dtype in ['fp32','fp16'] for T in [512,2048]]
        for size,dtype,T,gate in matrix:
            name=f'{size}_{dtype}_b{args.batch_size}_t{T}'
            report.update(status='waiting_for_idle_npu',active=name);persist()
            device=free_device(args);out=args.output/name
            argv=[sys.executable,'-u',str(Path(__file__).resolve()),*sys.argv[1:],
                  '--worker','--size',size,'--dtype',dtype,'--bucket',str(T)]
            ix=argv.index('--output');argv[ix+1]=str(out)
            if gate:argv.append('--gate-only')
            env=os.environ.copy();env['ASCEND_RT_VISIBLE_DEVICES']=str(device)
            report.update(status='running',active=name,physical_npu=device);persist()
            save(args.output/(name+'_command.json'),dict(argv=argv,physical_npu=device))
            print('JOB_START',name,device,flush=True)
            with (args.output/(name+'.log')).open('w') as log:
                child=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:exit_code=child.wait(timeout=1800)
                except subprocess.TimeoutExpired:
                    import signal
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=30)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    exit_code=124
            row=json.loads((out/'result.json').read_text()) if (out/'result.json').exists() else dict(error='No worker report')
            row.update(run=name,exit_code=exit_code);report['runs'].append(row);persist()
            print('JOB_DONE',name,exit_code,flush=True)
            if gate and (exit_code or not row.get('all_checks_passed')):raise RuntimeError('Tiny-model regression gate failed')
        passed=all(r.get('all_checks_passed') and r['exit_code']==0 for r in report['runs'])
        report.update(status='complete' if passed else 'complete_with_errors',all_checks_passed=passed)
        if not passed:raise RuntimeError('One or more speed probes failed; inspect per-run results')
    except Exception as e:report.update(status='failed',error=f'{type(e).__name__}: {e}');raise
    finally:
        report['total_seconds']=time.perf_counter()-start;persist()
        print('SUMMARY',json.dumps({k:v for k,v in report.items() if k!='runs'}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['models','cases-root','upstream','reference-build','build-root','download-status','output']:
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--devices',nargs='+',type=int,default=[7,6,4,3,2,1,0])
    p.add_argument('--repeats',type=int,default=10)
    p.add_argument('--recurrence',choices=['vector','matrix'],default='vector')
    p.add_argument('--matrix-chunk-size',type=int,choices=[16,32,64,128],default=64)
    p.add_argument('--matrix-compute-dtype',choices=['fp32','fp16','bf16'],default='fp32')
    p.add_argument('--diagnose-matrix-compile',action='store_true',help='Full-scoring numerical diagnosis with first-layer outputs, no timing')
    p.add_argument('--profile-invalid-compile',action='store_true',help='Diagnostic full-forward capture; invalid output is never accepted for speed')
    p.add_argument('--profile',action='store_true',help='Explicit B1/B4 worker: eager and optional compiled CPU/NPU traces with shapes')
    p.add_argument('--profile-data',type=Path)
    p.add_argument('--profile-warmup',type=int,default=5)
    p.add_argument('--profile-active',type=int,default=2)
    p.add_argument('--runtime',type=Path)
    p.add_argument('--warm-cache-from',type=Path)
    p.add_argument('--backend',choices=['torchair','raw_eager'],default='torchair')
    p.add_argument('--batch-size',type=int,choices=[1,4],default=4)
    p.add_argument('--exact-input-shape',action='store_true',help='B1 worker: compile the original token length without bucket padding')
    p.add_argument('--idle-wait-seconds',type=int,default=21600)
    p.add_argument('--allow-shared-device',action='store_true',help='Explicit middle/largest-reduced-precision worker probe; reserve 2 GiB headroom and cap allocator at 6 GiB')
    p.add_argument('--worker',action='store_true');p.add_argument('--gate-only',action='store_true')
    p.add_argument('--size',choices=list(SIZES),default='base')
    p.add_argument('--dtype',choices=['fp32','fp16','bf16'],default='fp32')
    p.add_argument('--bucket',type=int,choices=[512,2048],default=512)
    args=p.parse_args()
    if not 0<args.idle_wait_seconds<=86400 or not 3<=args.repeats<=100 or not set(args.devices).issubset({0,1,2,3,4,6,7}):p.error('Use 3..100 repeats and healthy idle devices 0/1/2/3/4/6/7')
    if args.exact_input_shape and (not args.worker or args.batch_size!=1):
        p.error('Exact input shape requires an explicit B1 worker')
    if args.batch_size!=4 and not args.worker:
        p.error('B1 is an explicit worker probe; the background matrix uses B4')
    if args.allow_shared_device and (not args.worker or args.size=='tiny' or (args.size=='large' and args.dtype=='fp32')):
        p.error('Shared-device mode requires an explicit middle or largest-FP16/BF16 worker')
    if args.profile and (not args.worker or args.gate_only or not args.profile_data or not args.runtime):
        p.error('Profiling requires explicit B1/B4 worker, --profile-data and --runtime')
    if not 3<=args.profile_warmup<=20 or not 2<=args.profile_active<=5:
        p.error('Use profile warmup 3..20 and active 2..5')
    if args.warm_cache_from and not args.worker:p.error('Warm cache is for an explicit worker')
    if args.recurrence == 'matrix' and (not args.worker or args.gate_only or args.warm_cache_from):
        p.error('Matrix path requires an explicit full-model worker and fresh private caches')
    worker(args) if args.worker else coordinate(args)


if __name__=='__main__':main()
