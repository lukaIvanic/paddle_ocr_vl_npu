"""Background 90M regression gate, then 317M/1.3B eager and TorchAir speed probes."""
import argparse
import gc
import json
import os
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


def worker(args):
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).parent; start = time.perf_counter()
    report = dict(size=args.size, dtype=args.dtype, batch_size=1 if args.gate_only else 4, gate_only=args.gate_only, bucket=args.bucket,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_sha256={n: sha256(root/n) for n in ['bench_reranker_sizes.py', 'local_modeling_rwkv_embedding.py',
            'local_modeling_rwkv_reranker.py', 'probe_reranker_endpoint.py', 'run_reranker_smoke.py', 'wkv7_endpoint.py']},
        physical_npu=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'), shared_device=args.allow_shared_device, all_checks_passed=False,
        scope='Prepared device inputs; uncached backbone plus state-readout head. Synchronized steady calls exclude CPU checks, compile, input transfers and tokenization.',
        timings={})
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
        dtype = {'fp16': torch.float16, 'fp32': torch.float32}[args.dtype]
        model = Embedding(emb, 'npu:0', dtype, expected_sha256=emb_sha)
        ranker = Reranker(rank, 'npu:0', dtype, expected_sha256=rank_sha)
        assert (model.depth, model.width, ranker.depth, ranker.width) == (depth, width, depth, width)
        assert all(0 <= i < model.depth for i in ranker.layer_indices)
        report.update(checkpoint_sha256=emb_sha, reranker_sha256=rank_sha,
                      architecture=dict(backbone_layers=depth, reranker_layers=ranker.depth, width=width,
                                        heads=model.heads, head_size=64, selected_layers=ranker.layer_indices))
        case_dir = args.cases_root/f'reranker_b1_t{args.bucket}_cold_ea9aa394/probe'
        assert json.loads((case_dir/'result.json').read_text())['all_checks_passed']
        cases = json.loads((case_dir/'inputs.json').read_text())[:4]
        assert len(cases)==4 and all(16 <= len(c['input_ids']) <= args.bucket and c['input_ids'][-1]==65535 for c in cases)
        report['cases_sha256'] = sha256(case_dir/'inputs.json')
        with torch.inference_mode():
            report['cpu_gate'] = cpu_gate(args, model, ranker, cases[0]['input_ids'])
            if args.gate_only:
                report['all_checks_passed']=True
                return
            backbone = Backbone(model).eval()
            lengths = torch.tensor([len(c['input_ids']) for c in cases], dtype=torch.int32, device='npu')
            ids = torch.tensor([c['input_ids']+[0]*(args.bucket-len(c['input_ids'])) for c in cases], dtype=torch.long, device='npu')
            save(args.output/'inputs.json', [dict(query_id=c['query_id'], document_id=c['document_id'],
                input_ids=c['input_ids'], valid_length=len(c['input_ids']), bucket=args.bucket) for c in cases])
            report['valid_tokens'] = lengths.cpu().tolist()
            eager = backbone(ids, lengths); expected = ranker(eager[1])
            singles = []
            for i,c in enumerate(cases):
                _, state, _ = model.encode_states(torch.tensor([c['input_ids']], dtype=torch.long, device='npu'))
                for j, x in enumerate(state):
                    batched = eager[j][:,:,i:i+1] if j==0 else eager[j][:,i:i+1]
                    require(batched, x, .02, .005)
                singles.append(ranker(state[1]))
            report['batch_and_right_padding_vs_single'] = require(expected, torch.cat(singles), .02, .005)
            del singles, state
            assert torch.equal(expected, ranker(backbone(ids,lengths)[1]))
            report['raw_checks_passed']=True
            report['timings']['eager_total'] = measure(lambda: ranker(backbone(ids,lengths)[1]), args.repeats, 4)
            print('EAGER_TIMING', json.dumps(report['timings']['eager_total']), flush=True)
            save(args.output/'result.json', report)
            before=time.perf_counter();print('COMPILE_START',args.size,args.dtype,args.bucket,flush=True)
            encode=compiled(backbone.forward,args.output/'backbone_cache'); head=compiled(ranker.forward,args.output/'head_cache')
            actual=encode(ids,lengths); logits=head(actual[1]); torch.npu.synchronize()
            report['compile_and_first_call_seconds']=time.perf_counter()-before
            report['compiled_states_vs_eager']=[require(a,b,.02,.005) for a,b in zip(actual,eager)]
            report['compiled_logits_vs_eager']=require(logits,expected,.02,.005)
            assert torch.equal(head(encode(ids,lengths)[1]),logits)
            fixed=actual[1].clone().contiguous()
            report['timings']['torchair_total']=measure(lambda:head(encode(ids,lengths)[1]),args.repeats,4)
            report['timings']['torchair_head_only']=measure(lambda:head(fixed),args.repeats,4)
            assert torch.equal(head(encode(ids,lengths)[1]),logits)
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
            name=f'{size}_{dtype}_b4_t{T}'
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
    p.add_argument('--idle-wait-seconds',type=int,default=21600)
    p.add_argument('--allow-shared-device',action='store_true',help='Explicit middle-model worker probe only; reserve 2 GiB headroom and cap allocator at 6 GiB')
    p.add_argument('--worker',action='store_true');p.add_argument('--gate-only',action='store_true')
    p.add_argument('--size',choices=list(SIZES),default='base')
    p.add_argument('--dtype',choices=['fp32','fp16'],default='fp32')
    p.add_argument('--bucket',type=int,choices=[512,2048],default=512)
    args=p.parse_args()
    if not 0<args.idle_wait_seconds<=86400 or not 3<=args.repeats<=100 or not set(args.devices).issubset({0,1,2,3,4,6,7}):p.error('Use 3..100 repeats and healthy idle devices 0/1/2/3/4/6/7')
    if args.allow_shared_device and (not args.worker or args.size!='base'):
        p.error('Shared-device mode is limited to an explicit middle-model worker')
    worker(args) if args.worker else coordinate(args)


if __name__=='__main__':main()
