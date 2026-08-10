#include "paddle_decode_kv_scatter_query_kernel.h"
#include "paddle_decode_kv_scatter_query_tiling.h"

using namespace AscendC;
using namespace paddle_decode_kv_scatter_query;

extern "C" __global__ __aicore__ void paddle_decode_kv_scatter_query_mixed24(
    GM_ADDR query,
    GM_ADDR key,
    GM_ADDR value,
    GM_ADDR cachePosition,
    GM_ADDR keyState,
    GM_ADDR valueState,
    GM_ADDR keyCacheRef,
    GM_ADDR valueCacheRef,
    GM_ADDR orderedQuery,
    GM_ADDR attentionMask,
    GM_ADDR keyCacheOut,
    GM_ADDR valueCacheOut,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    REGISTER_TILING_DEFAULT(PaddleDecodeKvScatterQueryTilingData);
    GET_TILING_DATA(tilingData, tiling);
    (void)key;
    (void)value;
    (void)keyCacheOut;
    (void)valueCacheOut;
    (void)workspace;
    if (g_coreType == AIC) {
        return;
    }
    if (GetBlockIdx() == 0 && tilingData.cacheLength == kCacheLength &&
        tilingData.queryElements == kQueryElements &&
        tilingData.stateElements == kStateElements) {
        TPipe pipe;
        Kernel kernel;
        kernel.Init(
            query, keyCacheRef, valueCacheRef, cachePosition, keyState,
            valueState, orderedQuery, attentionMask, &pipe);
        kernel.Process();
        PipeBarrier<PIPE_ALL>();
        pipe.Destroy();
    }
    // Every one of the 24 declared AIV workers enters this subfunction. Core
    // 0 publishes Q/K/V/mask, then this barrier hands those GM writes to the
    // following attention subfunction without feed-sync-all.
    SyncAll<true>();
}
