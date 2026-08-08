# Vector-only Add + RMSNorm lab

This is a lab-only Ascend 910B operator for the B1 FP16 decode shape
`[1, 1, 1024]`. Production presets continue to use CANN
`InplaceAddRmsNorm`.

The kernel follows the CANN 9.0 SingleN execution schedule. It changes one
mechanism:

- CANN SingleN reads reciprocal RMS from Vector into Scalar with `GetValue`,
  then passes the scalar back to Vector for `Muls`.
- This operator keeps reciprocal RMS on Vector, expands it with `Brcb`, and
  applies it with a zero-stride vector `Mul`.

## Build and install

Run on the Blue Zone 910B container:

```sh
cd /workspace/repos/paddle_ocr_vl_npu
source npu-setup
bash 09_persistent_page_engine/custom_ops/vector_add_rms_norm/build.sh

09_persistent_page_engine/custom_ops/vector_add_rms_norm/build_out/\
custom_opp_ubuntu_aarch64.run \
  --quiet \
  --install-path=/workspace/repos/paddle_ocr_vl_npu/.runtime_cache/vector_rmsnorm_opp

source .runtime_cache/vector_rmsnorm_opp/vendors/vector_rmsnorm/bin/set_env.bash
```

Use the lab preset
`combined_apply_prefetch_rope_lut_vector_norm` in
`scripts/text_decode_lab.py`. Use a new TorchAir cache directory after each
kernel rebuild because the graph cache can embed the prior kernel binary.

## Measured 910B result

Shape: B1, KV1024, cache position 768, full 18-layer compiled decoder,
FP16, NZ decode weights, weight prefetch enabled.

| Path | Full step | Throughput | Fused-norm median | Scalar median |
|---|---:|---:|---:|---:|
| CANN `InplaceAddRmsNorm` | 1.1458 ms | 872.8 tok/s | 1.69 us | 0.493 us |
| Vector `Brcb` revision | 1.1530 ms | 867.3 tok/s | 2.19 us | 0.963 us |

The custom graph matched all 16 reference argmax tokens. Mean logit absolute
error was 0.00833; maximum error was 0.11719.

Verdict: the Vector-only broadcast is valid, but it is not faster. It reduces
MTE2 and MTE3 time, while added Scalar issue/control work more than replaces
the saved cross-pipeline transfer. Keep this path as a research result. Do not
promote it to production.
