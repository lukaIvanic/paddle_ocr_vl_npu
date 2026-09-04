# 310P masked-GQA IncreFA tile boundary

## Source calculation

The public Huawei CANN source is
[`cann/ops-transformer`](https://gitcode.com/cann/ops-transformer). In
`attention/incre_flash_attention/op_host/incre_flash_attention_tiling.cpp`,
`IFATiling::CalcInnerSize` uses this 310P-only branch:

```cpp
if ((socVersion_ == IfaSocVersion::SOC_ASCEND_310P) &&
    (nNumOfQInOneGroup_ * qSeqSize_ > 1U)) {
    auto bmm1BufferSize = 40U * 1024U;
    sInnerSize_ = std::min(
        bmm1BufferSize / nNumOfQInOneGroup_ / qSeqSize_ /
            static_cast<uint32_t>(sizeof(float)),
        MAX_SPLIT_SIZE);
    sInnerSize_ = (sInnerSize_ / 128U) * 128U;
}
```

The formula is identical in the inspected public revisions:

| Source revision | Commit |
|---|---|
| v9.0.0 | `afe72144f9f2ac8441929035795db88a111b30c5` |
| v9.0.1 | `8038339a99bae113a7ae07f4547306d6d15bbddf` |
| inspected 9.1 branch | `e4800825f805bcce9752fccdfcaff04072767c09` |

For one-token decode, `qSeqSize=1` and
`nNumOfQInOneGroup = num_query_heads / num_kv_heads`.

PaddleOCR-VL has 16 query heads and 2 KV heads:

```text
G = 16 / 2 = 8
floor_to_128(40960 / 8 / 1 / 4) = 1280
```

MinerU2.5-Pro has 14 query heads and 2 KV heads:

```text
G = 14 / 2 = 7
floor_to_128(40960 / 7 / 1 / 4)
= floor_to_128(1462)
= 1408
```

Thus MinerU's exact internal S2 tile boundaries below KV4096 are effective
lengths 1408 and 2816. Effective length 4224 is outside the cache.

## Paddle evidence and workaround

The Experiment-09 310P ladder established the Paddle trigger before applying a
workaround:

- effective length 1280 stopped, while adjacent lengths passed;
- KV4096 and KV3584 both stopped at the same logical length;
- masked GQA stopped in eager and compiled execution;
- no-mask GQA, masked MHA, and B1 masked GQA passed;
- batched masked GQA at the exact 1280 tile stopped.

The production `combined_apply_pse_sentinel` mode keeps one PSE tensor present
for the compiled graph. At each exact tile boundary below cache capacity it:

1. exposes the next otherwise-masked physical key in the Boolean mask;
2. assigns that key FP16 minimum (`-65504`) through `pse_shift`;
3. leaves PSE zero and the Boolean mask unchanged at all other positions.

The key remains logically suppressed after softmax. The public 310P AscendC
implementation consumes `pse_shift` through a runtime flag inside the same
top-level tiling-key specialization. It copies, casts, and adds PSE before
applying the Boolean mask. Therefore, "different top-level kernel" is not an
accurate description for this source revision. The workaround changes the
mask transition and activates the PSE path inside that kernel.

Paddle promoted this mode into the B64 production pipeline. The reported full
OmniDocBench v1.6 result on one 310P3 used concurrency 64 and reported 0.7
pages/s, 95.59 Overall, 94.9 text accuracy, 94.4 table Page-TEDS, and 97.4
formula Page-CDM. The PSE mode was the decode-termination fix, not the only
source of that throughput; B64 scheduling, compiled decode, NZ weights, packed
prefill, compiled PromptFA vision, and staged layout also contributed.

## MinerU implementation and 910B validation

Commit `b57f7418` adds the opt-in
`--local-decode-increfa-length-mode pse_sentinel_310p` mode. It derives 1408
from model head geometry. It does not hard-code the observed failed step. The
normal lane keeps its previous cache path. The PSE graph gets a separate stable
cache key.

The full B32/KV4096 production decode graph was run synchronously at every
cache position 0 through 4095 on one 910B2:

```text
positions completed       4096 / 4096
effective length 1408     4.565 ms
effective length 2816     4.514 ms
effective length 4096     4.556 ms
cold first graph call     38.772 s
steady synchronized p50   4.548 ms
```

An immediate replay reported `cache_was_warm=true`, a 0.000179-second compile
wrapper, all 4096 positions complete, and a 4.383 ms synchronized median.

The real two-page production pipeline then crossed effective length 1408 at
step 13 and completed 2/2 pages and 34/34 request streams. Its generation trace
SHA-256 and both Markdown prediction files were byte-identical to two prior
normal-IncreFA 910B anchors. This proves the source and graph contract on 910B.
It does not prove the 310P workaround until the pull-only work server runs it.
