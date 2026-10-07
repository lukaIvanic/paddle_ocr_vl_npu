"""Independent vector launch/precision probes; original packages stay untouched.

Identity ledger: aiv-fp32 -> rwkv_aiv32 / RwkvAiv32Wkv7;
aiv-fp16 -> rwkv_aiv16 / RwkvAiv16Wkv7. Each namespace is also its
vendor/kernel prefix; public ACLNN uses aclnn + the GE name.
Both use the endpoint contract, including the one-token readout head.
FP16 means FP16 inputs, state, products, sums, exp and reductions, not just I/O.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from wkv7_endpoint import prepare as prepare_endpoint, replace

IDENTITIES = {'aiv-fp32': ('rwkv_aiv32', 'RwkvAiv32Wkv7'),
              'aiv-fp16': ('rwkv_aiv16', 'RwkvAiv16Wkv7')}


def prepare(build, variant):
    namespace, ge_name = IDENTITIES[variant]
    source = prepare_endpoint(build)
    base_manifest = json.loads((source/'manifest.json').read_text())
    manifest = dict(variant=variant, endpoint_base=base_manifest,
        identities=dict(pytorch=namespace+'::wkv7', ge=ge_name,
                        aclnn='aclnn'+ge_name, kernel=namespace+'_wkv7', vendor=namespace),
        contract='Contiguous [B,H,T<=2048,64], state [B,H,64,64], int32 lengths[B]; endpoint semantics; Ascend910B2',
        precision='all recurrence tensors and arithmetic '+('FP16' if variant=='aiv-fp16' else 'FP32'),
        launch='GetCoreNumAiv() instead of generic GetCoreNum(); same token/head algorithm',
        generated_sha256={})
    for name in base_manifest['generated_sha256']:
        old = source/name
        text = old.read_text().replace('RwkvEndpointWkv7',ge_name).replace('rwkv_endpoint',namespace)
        if name == 'op_host/rwkv_endpoint_wkv7.cpp':
            text = text.replace('ascendcPlatform.GetCoreNum()', 'ascendcPlatform.GetCoreNumAiv()')
            if variant == 'aiv-fp16':text=text.replace('ge::DT_FLOAT}', 'ge::DT_FLOAT16}')
        if variant == 'aiv-fp16' and name == 'bridge.cpp':
            text=text.replace('at::kFloat', 'at::kHalf').replace('FP32','FP16')
        if variant == 'aiv-fp16' and name == 'op_kernel/rwkv_endpoint_wkv7.cpp':
            text=text.replace('float','half').replace('0.0f','half(0.0f)')
            start=text.index('            // compute astate')
            end=text.index('\n        }',start)
            # 32-byte blocks hold 16 half elements. Each state row is four
            # blocks. Brcb expands each of 64 scalars into ONE 16-half block;
            # srcBlkStride=0 then repeats it across the four blocks of a row.
            text=text[:start]+HALF_UPDATE+text[end:]
        target=source/name.replace('rwkv_endpoint',namespace)
        if target != old:old.unlink()
        target.write_text(text)
        manifest['generated_sha256'][str(target.relative_to(source))]=hashlib.sha256(target.read_bytes()).hexdigest()
    (source/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return source


HALF_UPDATE = r'''            Mul(stateLocal[offset2], abLocal[t * HEAD_SIZE], stateLocal[offset0],
                64, 64, {1,1,1,4,0,4});
            PipeBarrier<PIPE_V>();
            WholeReduceSum(stateLocal[offset1], stateLocal[offset2], 64, 64, 1, 1, 4);
            PipeBarrier<PIPE_V>();
            Brcb(broadLocal0, stateLocal[offset1], 8, {1,8});
            PipeBarrier<PIPE_V>();
            Mul(stateLocal[offset2], abLocal[tileLength * HEAD_SIZE + t * HEAD_SIZE],
                broadLocal0, 64, 64, {1,1,0,4,0,1});
            PipeBarrier<PIPE_V>();
            Brcb(broadLocal0, vLocal[t * HEAD_SIZE], 8, {1,8});
            PipeBarrier<PIPE_V>();
            Mul(stateLocal[offset1], kLocal[t * HEAD_SIZE], broadLocal0,
                64, 64, {1,1,0,4,0,1});
            PipeBarrier<PIPE_V>();
            Mul(broadLocal0, stateLocal[offset0], wLocal[t * HEAD_SIZE],
                64, 64, {1,1,1,4,4,0});
            PipeBarrier<PIPE_V>();
            Add(stateLocal[offset1], broadLocal0, stateLocal[offset1], HEAD_ELEMENTS);
            PipeBarrier<PIPE_V>();
            Add(stateLocal[offset0], stateLocal[offset1], stateLocal[offset2], HEAD_ELEMENTS);
            PipeBarrier<PIPE_V>();
            Mul(stateLocal[offset2], rLocal[t * HEAD_SIZE], stateLocal[offset0],
                64, 64, {1,1,1,4,0,4});
            PipeBarrier<PIPE_V>();
            WholeReduceSum(oOutLocal[t * HEAD_SIZE], stateLocal[offset2], 64, 64, 1, 1, 4);'''


def load_variant(build, variant):
    import torch
    import torch_npu
    from torch.utils.cpp_extension import load
    from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
    from torchair import ge
    namespace, ge_name=IDENTITIES[variant]
    package=Path(torch_npu.__file__).parent
    folder=build/'bridge';folder.mkdir(exist_ok=True)
    load(name=namespace+'_wkv7_bridge',sources=[str(Path(str(build)+'_source')/'bridge.cpp')],
        build_directory=str(folder),is_python_module=False,
        extra_include_paths=[str(package/'include'),str(package/'include/third_party/acl/inc'),
            str(package/'include/third_party/op-plugin'),str(package/'include/third_party/op-plugin/op_plugin/include')],
        extra_cflags=['-O3','-std=c++17'],extra_ldflags=[f'-L{package / "lib"}','-ltorch_npu'],verbose=True)
    op=getattr(torch.ops,namespace).wkv7.default
    @register_fx_node_ge_converter(op)
    def convert(k,v,w,r,a,b,hi,lengths,meta_outputs=None):
        return ge.custom_op(ge_name,inputs=dict(k=k,v=v,w=w,r=r,a=a,b=b,hi=hi,lengths=lengths),
                            attrs={},outputs=['o','ho'])
    return op


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build-root',type=Path,required=True)
    p.add_argument('--variant',choices=IDENTITIES,required=True)
    args=p.parse_args()
    assert not args.build_root.exists(), 'Use a fresh build directory'
    source=prepare(args.build_root,args.variant)
    subprocess.run(['bash',str(source/'build.sh'),str(args.build_root)],check=True)
