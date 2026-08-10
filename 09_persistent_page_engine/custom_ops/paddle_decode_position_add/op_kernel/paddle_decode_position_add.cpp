#include "kernel_operator.h"
#include "paddle_decode_position_add_tiling.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void paddle_decode_position_add_v1(
    GM_ADDR cachePosition,
    GM_ADDR ropeDelta,
    GM_ADDR decodePosition,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(PaddleDecodePositionAddTilingData);
    GET_TILING_DATA(tilingData, tiling);
    if (GetBlockIdx() != 0 || tilingData.elements != 1) {
        return;
    }

    GlobalTensor<int64_t> cachePositionGm;
    GlobalTensor<int64_t> ropeDeltaGm;
    GlobalTensor<int64_t> decodePositionGm;
    cachePositionGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int64_t*>(cachePosition), 1);
    ropeDeltaGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int64_t*>(ropeDelta), 1);
    decodePositionGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int64_t*>(decodePosition), 1);

    DataCacheCleanAndInvalid<
        int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
            cachePositionGm);
    DataCacheCleanAndInvalid<
        int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
            ropeDeltaGm);
    decodePositionGm.SetValue(
        0, cachePositionGm.GetValue(0) + ropeDeltaGm.GetValue(0));
    DataCacheCleanAndInvalid<
        int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
            decodePositionGm);
}
