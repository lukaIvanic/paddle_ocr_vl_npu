"""Build a hash-pinned, independently named endpoint-state WKV7 variant.

The generated source is a reviewable overlay of the existing imported operator.
Only identity, one device lengths input, and trailing recurrence updates change.
Never replace the existing package or operator; use a fresh external build root.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def replace(text, old, new, count=1):
    assert text.count(old) == count, (old, text.count(old), count)
    return text.replace(old, new)


def prepare(build):
    root = Path(__file__).parent
    pin = json.loads((root/'data/wkv7_endpoint_sources.json').read_text())
    source = Path(str(build)+'_source')
    source.mkdir(exist_ok=False)
    manifest = {'base':pin, 'generated_sha256':{}}
    for name, digest in pin['base_sources'].items():
        raw = (root/'wkv7_npu'/name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == digest, name
        text = raw.decode().replace('RwkvReferenceWkv7','RwkvEndpointWkv7').replace('rwkv_reference','rwkv_endpoint')
        if name == 'bridge.cpp':
            text = replace(text, 'const at::Tensor& hi)', 'const at::Tensor& hi, const at::Tensor& lengths)', 2)
            text = replace(text, 'auto out =', '''TORCH_CHECK(lengths.device() == k.device() && lengths.scalar_type() == at::kInt &&
                lengths.is_contiguous() && lengths.dim() == 1 && lengths.size(0) == k.size(0),
                "Expected contiguous int32 lengths[B] on the same NPU; values must be 0..T");
    auto out =''')
            text = replace(text, 'b, hi, out, ht);', 'b, hi, lengths, out, ht);')
            text = replace(text, 'Tensor hi) ->', 'Tensor hi, Tensor lengths) ->')
        elif name == 'op_host/rwkv_reference_wkv7.cpp':
            text = replace(text, 'this->Output("o")', '''this->Input("lengths")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("o")''')
        elif name == 'op_kernel/rwkv_reference_wkv7.cpp':
            text = replace(text, 'GM_ADDR h0, GM_ADDR o,  GM_ADDR ht)', 'GM_ADDR h0, GM_ADDR lengths, GM_ADDR o, GM_ADDR ht)')
            text = replace(text, 'uint32_t uh_offset =', '''this->headOffset = headOffset;
        lengthsGm.SetGlobalBuffer((__gm__ int32_t *)lengths, totalHeads / HEAD_NUMS);
        uint32_t uh_offset =''')
            text = replace(text, 'LocalTensor<float> oOutLocal = outQueueO.AllocTensor<float>();\n        for', '''LocalTensor<float> oOutLocal = outQueueO.AllocTensor<float>();
        // Fixed padded strides; the device length is data, not a tiling/graph key.
        const uint32_t valid = lengthsGm.GetValue((headOffset + progress_h) / HEAD_NUMS);
        const uint32_t start = progress_tile * tileLength;
        const uint32_t steps = valid <= start ? 0 :
            (valid - start < currentTileLength ? valid - start : currentTileLength);
        Duplicate(oOutLocal, 0.0f, currentTileLength * HEAD_SIZE);
        PipeBarrier<PIPE_V>();
        for''')
            text = replace(text, 't < currentTileLength; t++', 't < steps; t++')
            text = replace(text, 'TPipe pipe;', 'TPipe pipe;\n    GlobalTensor<int32_t> lengthsGm;\n    uint32_t headOffset;')
            text = replace(text, 'GM_ADDR a, GM_ADDR b, GM_ADDR h0, GM_ADDR o, GM_ADDR ht,',
                           'GM_ADDR a, GM_ADDR b, GM_ADDR h0, GM_ADDR lengths, GM_ADDR o, GM_ADDR ht,')
            text = replace(text, 'k, v, w, r, a, b, h0, o, ht);', 'k, v, w, r, a, b, h0, lengths, o, ht);')
        target = source/name.replace('rwkv_reference','rwkv_endpoint')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        manifest['generated_sha256'][str(target.relative_to(source))] = hashlib.sha256(target.read_bytes()).hexdigest()
    (source/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return source


def load_endpoint(build):
    import torch_npu
    from torch.utils.cpp_extension import load
    package = Path(torch_npu.__file__).parent
    folder = build/'bridge'; folder.mkdir(exist_ok=True)
    return load(name='rwkv_endpoint_wkv7_bridge', sources=[str(Path(str(build)+'_source')/'bridge.cpp')],
        build_directory=str(folder), is_python_module=False,
        extra_include_paths=[str(package/'include'),str(package/'include/third_party/acl/inc'),
            str(package/'include/third_party/op-plugin'),str(package/'include/third_party/op-plugin/op_plugin/include')],
        extra_cflags=['-O3','-std=c++17'],extra_ldflags=[f'-L{package / "lib"}','-ltorch_npu'],verbose=True)


def register_converter():
    import torch
    from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
    from torchair import ge
    @register_fx_node_ge_converter(torch.ops.rwkv_reference.wkv7.default)
    def reference_convert(k,v,w,r,a,b,hi,meta_outputs=None):
        return ge.custom_op('RwkvReferenceWkv7',inputs=dict(k=k,v=v,w=w,r=r,a=a,b=b,hi=hi),
                            attrs={},outputs=['o','ho'])
    @register_fx_node_ge_converter(torch.ops.rwkv_endpoint.wkv7.default)
    def convert(k,v,w,r,a,b,hi,lengths,meta_outputs=None):
        return ge.custom_op('RwkvEndpointWkv7',inputs=dict(k=k,v=v,w=w,r=r,a=a,b=b,hi=hi,lengths=lengths),
                            attrs={},outputs=['o','ho'])


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build-root',type=Path,required=True)
    args=p.parse_args()
    assert not args.build_root.exists(), 'Use a fresh build directory'
    source=prepare(args.build_root)
    subprocess.run(['bash',str(source/'build.sh'),str(args.build_root)],check=True)
