"""Posthoc attribution for this frozen B1 5040/1274-token pipe capture.

Uses exported CANN shapes, dtypes and fused names, cross-checked against source.
Not a general profiler parser: shape-specific rules deliberately fail the count
checks if used on a different workload. Shared residual-add/FP32-cast kernels
remain separate from normalization. No model execution or new timing occurs.
"""
import json
from pathlib import Path


def category(g):
    t, names = g['type'], g['names']
    shape = g['input_shapes'].split(';')[0]
    dtype = g['input_dtypes']
    if t.startswith('MatMul') or 'FlashAttention' in t:
        return 'matrix_attention'
    if (t == 'Square'
        or names == ['AddRsqrt'] and shape.startswith('1,1274,')
        or names == ['MulCast'] and shape.startswith('1,1274,')
        or t == 'Mul' and shape in ('2560', '128')
        or t == 'Cast' and shape.startswith('1,1274,') and dtype == 'FLOAT16'):
        return 'text_rmsnorm'
    if (t == 'ReduceMeanD' and shape == '5040,1024'
        or names == ['SubPow/Square']
        or names == ['MulCast'] and shape == '5040,1024'
        or names == ['MulAdd'] and shape == '5040,1024'
        or names == ['AddRsqrt'] and shape == '5040,1'
        or t == 'Cast' and shape == '5040,1024' and dtype == 'FLOAT16'):
        return 'vision_manual_layernorm'
    if (t in ('Neg', 'ConcatV2D')
        or t == 'SplitVD' and shape not in ('1,1274,19456', '1,1274,6144')
        or names == ['MulMulAdd']
        or t in ('Mul', 'Cast', 'Add') and shape == '5040,16,64'):
        return 'rotary'
    if t in ('Transpose', 'Unpack', 'SplitVD'):
        return 'layout_qkv_unpack_projection_split'
    if t in ('GeluV2', 'Gelu') or names == ['SwishMul']:
        return 'mlp_activations'
    if t == 'MaskedScatter':
        return 'image_embedding_masked_scatter'
    if t == 'Add':
        return 'residual_add_shared_norm_cast_and_prepare_add'
    return 'remaining_preparation_merger_norm_and_finish'


def main():
    root = Path(__file__).parent
    capture = json.loads((root / 'kernel_shape_groups.json').read_text())['torchair']
    groups = capture['groups']
    assert sum(g['count_per_forward'] for g in groups) == 2515
    assert sum(g['count_per_forward'] for g in groups if g['type'] == 'Square') == 145
    assert sum(g['count_per_forward'] for g in groups if g['type'] == 'ReduceMeanD') == 96
    buckets = {}
    for g in groups:
        buckets.setdefault(category(g), []).append(g)
    rows = [dict(category=k, ms_per_forward=sum(g['ms_per_forward'] for g in v),
                 kernels_per_forward=sum(g['count_per_forward'] for g in v), groups=v)
            for k, v in buckets.items()]
    rows.sort(key=lambda r: -r['ms_per_forward'])
    total = sum(r['ms_per_forward'] for r in rows)
    other = sum(r['ms_per_forward'] for r in rows if r['category'] != 'matrix_attention')
    assert abs(total - 122.70710466666665) < 1e-8
    assert abs(other - 33.62716233333333) < 1e-8
    result = dict(source=capture['source'], profile_steps=3,
                  scope='Summed device kernel durations, not clean wall latency; '
                        'semantic attribution inferred from fused names, shapes, dtypes and owned source.',
                  shared_boundary='AddCast kernels perform both residual addition and the FP32 cast '
                                  'for a subsequent norm; assigned to shared residual/cast work.',
                  total_ms_per_forward=total, other_ms_per_forward=other, categories=rows)
    (root / 'other_kernel_attribution.json').write_text(json.dumps(result, indent=2) + '\n')
    for r in rows:
        print(f"{r['category']:52s} {r['ms_per_forward']:8.4f} ms "
              f"{r['kernels_per_forward']:5.0f} kernels/forward")


if __name__ == '__main__':
    main()
