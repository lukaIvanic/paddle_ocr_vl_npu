# 910B gamma/beta FP32 cache control

Completed: exact outputs and reduced vector work, without a consistent latency
gain. See [measured 910B results](RESULTS_910B.md).

Pinned ops-nn v9.1.0, `ceb4536a2bd6fc99b85aec9d0fdcc0f470376292`.
Only normal FP16 V2 keys 1000/1001/1002 opt into the template's cache. Runtime
allocation/use additionally requires architecture 2201, normalized width/stride
1024, normalized output present, BUFFER_NUM=1, and rowStep 1..13. Other contracts
keep upstream behavior. This is a research package, not the default operator.

Both FP16 parameter copies remain intact. After their MTE2 completion is
explicitly awaited, gamma and beta are cast once per core into separate 4096-byte
FP32 buffers. These live for all row chunks. ApplyGammaBeta reuses them with the
same multiply/add order. No reductions, row buffering, copies, tiling, quantizer,
or bias arithmetic are changed. No 310P changes are included.

For width 1024 the original tiler budgets `18440*r + 4608` bytes (no additional
output). **The installed Ascend910B2.ini reports 196608 bytes (192 KiB), giving
at most 10 rows**, not 13. Runtime logs report 196352 usable bytes (256 bytes
less than the platform file). Actual explicit kernel allocations with the cache are
`14336*r + 12320` bytes for no bias or elementwise bias; broadcast bias adds
2048 bytes. At r=10, broadcast reaches **157728 bytes**, leaving **38624 usable
bytes** (38880 relative to the physical platform capacity).
The cache never aliases row scratch or FP16 parameters. The conservative tiler
stays unchanged. For 256/512/2048 rows, expected blocks are 43/47/48, regular
rows/core 6/11/43, rowStep 6/10/10.

The patch's `rowStep <= 13` guard and 256-KiB example are an additional upper
bound, **not a claim that this 910B2 has 256-KiB UB**. Actual rowStep comes from
the unchanged platform-aware tiler. For any r>=3, its existing byte estimate
already exceeds the candidate's exact broadcast allocation by `4104*r-9760`
bytes. r=1/2 allocations are at most 43040 bytes. No tiler reservation or rowStep
change is needed on this 192-KiB device. The initial 256-KiB assumption was
corrected after reading the installed platform file, before interpreting data.

Identity ledger: existing PyTorch `bge_m3::norm_quant_v2`, GE/API/kernel names
remain AddLayerNormQuantV2/aclnnAddLayerNormQuantV2/add_layer_norm_quant_v2;
**vendor is `bge_v2_gc_nn`, installation and graph caches are independent**.
Per the explicitly authorized same-name exception, baseline/candidate must run
in separate fresh processes with exactly one custom OPP selected. Do not load
both packages together or replace the baseline/global CANN. The graph preparation
retains V2-only dispatch to avoid overriding baseline dependency operators.

Build from the repo after sourcing npu-setup and putting Python 3.12 on PATH:

```sh
bash 26_bge_m3_inference/gamma_cache/build_910b.sh \
  /workspace/operator_sources/bge-ops-nn-v9.1.0 /workspace/operators/bge-gamma-cache-RUN
```

Generated third-party source/build/install trees and raw profiles stay outside
Git. The committed patch is the only upstream source change. Run scripts and
measured evidence are documented with the final results.
