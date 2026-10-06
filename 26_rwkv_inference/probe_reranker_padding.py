"""Diagnose padding with released CPU math; do not change the inference path."""
import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import torch
from local_modeling_rwkv_embedding import Embedding
from local_modeling_rwkv_reranker import Reranker
from probe_wkv7 import load_bridge
from run_cpu_reference import CReference, sha256
from run_nanoscidocs import save
from run_reranker_smoke import PREFIX, SUFFIX, cpu_models, metrics, require

WRAPPER_SHA256 = 'a6a00f711c95a3a1d927c7d3c4bfc2436bae1d6ba06bf4b76d7d1e373c850123'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint','reranker','upstream','runtime','build-root','cases-root','data','output']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--allow-shared-device', action='store_true')
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).parent; start = time.perf_counter(); tokenizer = None
    report = {'source_commit': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
              'source_sha256': {n:sha256(root/n) for n in ['probe_reranker_padding.py','local_modeling_rwkv_embedding.py','local_modeling_rwkv_reranker.py','run_reranker_smoke.py']},
              'physical_npu': os.environ.get('ASCEND_RT_VISIBLE_DEVICES'), 'backend': 'raw_eager',
              'cpu_reference': 'Hash-pinned released math, FP32; explicit recurrence replaces CUDA',
              'npu_precision': 'FP16 projections, FP32 state/pointwise', 'cases': [],
              'thresholds': {'cpu_npu_atol': .02, 'cpu_npu_rtol': .005,
                             'split_atol': .002, 'split_rtol': .001}, 'all_checks_passed': False}
    try:
        import torch_npu
        torch.set_num_threads(4)
        wrapper = args.upstream/'wrapper.py'; assert sha256(wrapper) == WRAPPER_SHA256
        tree = ast.parse(wrapper.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TokenizerWrapper')
        ns = {'torch': torch, 'EOS_INDEX': 65535}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(wrapper), 'exec'), ns)
        tokenizer = CReference(args.runtime, 4)
        released = ns['TokenizerWrapper'](SimpleNamespace(encode=lambda text: tokenizer.tokenize(text).tolist()), PREFIX+SUFFIX)
        report['wrapper_sha256'] = WRAPPER_SHA256
        report['upstream_predict_default_batch_size'] = 32  # Wrapper default; MTEB may override it.
        data = json.loads(args.data.read_text())
        corpus = {x['_id']:x['text'].strip() for x in data['corpus']}
        queries = {x['_id']:x['text'] for x in data['queries']}
        groups = []
        for bucket in [512,2048]:
            folder = args.cases_root/f'reranker_b1_t{bucket}_cold_ea9aa394/probe'
            old = json.loads((folder/'result.json').read_text()); assert old['all_checks_passed']
            for name in ['local_modeling_rwkv_embedding.py','local_modeling_rwkv_reranker.py']:
                assert report['source_sha256'][name] == old['source_sha256'][name]
            selected = old['selection']['close_pair']; inputs = json.loads((folder/'inputs.json').read_text())
            cases = [next(x for x in inputs if [x['query_id'],x['document_id']] == key) for key in selected]
            texts = [(queries[x['query_id']], corpus[x['document_id']]) for x in cases]
            for x, text in zip(cases,texts):
                assert released([text]).tolist() == [x['input_ids']], 'Prompt/token/EOS mismatch'
            rows = released(texts).tolist()
            assert rows == [[0]*(max(len(x['input_ids']) for x in cases)-len(x['input_ids']))+x['input_ids'] for x in cases]
            groups.append((bucket,cases,rows))
        save(args.output/'inputs.json', [{'bucket':b,'cases':cases,'upstream_batch_rows':rows} for b,cases,rows in groups])
        report['tokenizer_wrapper_parity'] = True
        report['input_sha256'] = {n:sha256(getattr(args,n)) for n in ['checkpoint','reranker','data']}
        load_bridge(root/'wkv7_npu', args.build_root)
        status = subprocess.check_output(['/usr/local/bin/npu-status'], text=True)
        selected = next((x for x in status.splitlines() if x.startswith(f"NPU {report['physical_npu']}: ")), '')
        assert 'Health=OK' in selected and (args.allow_shared_device or ': free ' in selected)
        torch.npu.set_device(0); torch.npu.set_compile_mode(jit_compile=False); torch.npu.config.allow_internal_format=False
        free,total = torch.npu.mem_get_info(); assert free >= 2*1024**3
        report.update(device=torch.npu.get_device_name(0), torch=torch.__version__, torch_npu=torch_npu.__version__,
                      hbm_before={'free_bytes':free,'total_bytes':total})
        embed = Embedding(args.checkpoint,'npu:0',torch.float16)
        ranker = Reranker(args.reranker,'npu:0',torch.float16)
        cpu, cpu_ranker, indices = cpu_models(args)
        def cpu_score(ids):
            state = cpu.generate_zero_state(1)
            cpu.forward_seq_batch([ids],state,True)
            return cpu_ranker(torch.stack([state[1][i] for i in indices])).reshape(-1), state
        def npu_score(ids):
            _,state,_ = embed.encode_states(torch.tensor([ids],dtype=torch.long,device='npu'))
            return ranker(state[1]), state
        with torch.inference_mode():
            e = embed.get('emb.weight')[0]; norm = embed.norm(e, 'blocks.0.ln0')
            report['pad_embedding'] = {'token_id':0, 'raw_l2':float(e.norm().item()),
                                       'after_ln0_l2':float(norm.norm().item()), 'raw_all_zero':bool((e == 0).all().item())}
            for bucket,cases,batch_rows in groups:
                for case, companion in zip(cases,batch_rows):
                    ids = case['input_ids']; count = bucket-len(ids); padded = [0]*count+ids
                    print('CPU_START',json.dumps({'bucket':bucket,'tokens':len(ids),'padding':count}),flush=True)
                    ca, cs = cpu_score(ids); cb, ps = cpu_score(padded)
                    na, ns = npu_score(ids); nb, bs = npu_score(padded)
                    row = {k:case[k] for k in ['query_id','document_id']}
                    row.update(bucket=bucket,raw_tokens=len(ids),padding_tokens=count,
                        cpu_raw_logit=float(ca.item()),cpu_padded_logit=float(cb.item()),
                        npu_raw_logit=float(na.item()),npu_padded_logit=float(nb.item()),
                        cpu_padding_delta=float((cb-ca).item()),npu_padding_delta=float((nb-na).item()),
                        raw_cpu_npu=require(na,ca,.02,.005),padded_cpu_npu=require(nb,cb,.02,.005),
                        cpu_raw_vs_padded_state=[metrics(a,b) for a,b in zip(ps,cs)],
                        npu_raw_vs_padded_state=[metrics(a,b) for a,b in zip(bs,ns)])
                    # A zero-only prefix must be tested without adding EOS to it.
                    _,prefix,_ = embed.encode_states(torch.zeros((1,count),dtype=torch.long,device='npu'))
                    row['zero_prefix'] = {'shift_l2':float(prefix[0].norm().item()),
                                          'matrix_l2':float(prefix[1].norm().item())}
                    assert row['zero_prefix']['matrix_l2'] > 0, 'Unexpected neutral zero-prefix'
                    _,continued,_ = embed.encode_states(torch.tensor([ids],dtype=torch.long,device='npu'),prefix)
                    row['prefix_continuation_logit'] = float(ranker(continued[1]).item())
                    row['prefix_continuation_vs_full'] = require(ranker(continued[1]),nb,.002,.001)
                    row['companion_padding_tokens'] = len(companion)-len(ids)
                    row['companion_logit'] = float(npu_score(companion)[0].item())
                    row['right_padding_logit'] = float(npu_score(ids+[0]*count)[0].item())
                    report['cases'].append(row); save(args.output/'result.json',report)
                    print('CASE',json.dumps(row),flush=True)
            sweep_case = groups[0][1][1]; ids = sweep_case['input_ids']
            report['padding_count_sweep'] = [{'padding_tokens':n,'logit':float(npu_score([0]*n+ids)[0].item())}
                                             for n in [0,1,2,8,32,64,128,206]]
            report['pair_order'] = []
            for bucket,_,_ in groups:
                a,b = [x for x in report['cases'] if x['bucket'] == bucket]
                report['pair_order'].append({'bucket':bucket, **{k:a[k]-b[k] for k in
                    ['cpu_raw_logit','cpu_padded_logit','npu_raw_logit','npu_padded_logit','companion_logit']}})
            report['all_checks_passed'] = True
            print('SUMMARY',json.dumps({k:report[k] for k in ['all_checks_passed','pad_embedding','pair_order','padding_count_sweep']}),flush=True)
    finally:
        if tokenizer is not None: tokenizer.close()
        report['total_seconds'] = time.perf_counter()-start
        save(args.output/'result.json',report)

if __name__ == '__main__': main()
