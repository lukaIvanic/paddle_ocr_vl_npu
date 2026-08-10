#include "kernel_operator.h"
#include "paddle_decode_kv_scatter_query_tiling.h"

using namespace AscendC;

namespace {
constexpr uint32_t kQueryHeads = 16;
constexpr uint32_t kKvHeads = 2;
constexpr uint32_t kCacheLength = 1024;
constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kQueryElements = kQueryHeads * kHeadDim;
constexpr uint32_t kStateElements = kKvHeads * kHeadDim;

class PaddleDecodeKvScatterQueryKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR query,
        GM_ADDR keyCache,
        GM_ADDR valueCache,
        GM_ADDR cachePosition,
        GM_ADDR keyState,
        GM_ADDR valueState,
        GM_ADDR orderedQuery,
        TPipe* pipe)
    {
        queryGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ half*>(query), kQueryElements);
        keyCacheGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ half*>(keyCache),
            kKvHeads * kCacheLength * kHeadDim);
        valueCacheGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ half*>(valueCache),
            kKvHeads * kCacheLength * kHeadDim);
        cachePositionGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ int64_t*>(cachePosition), 1);
        keyStateGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ half*>(keyState), kStateElements);
        valueStateGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ half*>(valueState), kStateElements);
        orderedQueryGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ half*>(orderedQuery), kQueryElements);
        pipe->InitBuffer(queryQueue, 1, kQueryElements * sizeof(half));
        pipe->InitBuffer(keyQueue, 1, kStateElements * sizeof(half));
        pipe->InitBuffer(valueQueue, 1, kStateElements * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        DataCacheCleanAndInvalid<
            int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                cachePositionGm);
        const int64_t position = cachePositionGm.GetValue(0);
        if (position < 0 || position >= static_cast<int64_t>(kCacheLength)) {
            return;
        }

        LocalTensor<half> queryLocal = queryQueue.AllocTensor<half>();
        LocalTensor<half> keyLocal = keyQueue.AllocTensor<half>();
        LocalTensor<half> valueLocal = valueQueue.AllocTensor<half>();
        DataCopy(queryLocal, queryGm, kQueryElements);
        DataCopy(keyLocal, keyStateGm, kStateElements);
        DataCopy(valueLocal, valueStateGm, kStateElements);
        queryQueue.EnQue(queryLocal);
        keyQueue.EnQue(keyLocal);
        valueQueue.EnQue(valueLocal);

        queryLocal = queryQueue.DeQue<half>();
        keyLocal = keyQueue.DeQue<half>();
        valueLocal = valueQueue.DeQue<half>();
        DataCopy(orderedQueryGm, queryLocal, kQueryElements);
        for (uint32_t head = 0; head < kKvHeads; ++head) {
            const uint32_t stateOffset = head * kHeadDim;
            const uint32_t cacheOffset =
                (head * kCacheLength + static_cast<uint32_t>(position)) * kHeadDim;
            DataCopy(keyCacheGm[cacheOffset], keyLocal[stateOffset], kHeadDim);
            DataCopy(valueCacheGm[cacheOffset], valueLocal[stateOffset], kHeadDim);
        }
        queryQueue.FreeTensor(queryLocal);
        keyQueue.FreeTensor(keyLocal);
        valueQueue.FreeTensor(valueLocal);
    }

private:
    GlobalTensor<half> queryGm;
    GlobalTensor<half> keyCacheGm;
    GlobalTensor<half> valueCacheGm;
    GlobalTensor<int64_t> cachePositionGm;
    GlobalTensor<half> keyStateGm;
    GlobalTensor<half> valueStateGm;
    GlobalTensor<half> orderedQueryGm;
    TQue<QuePosition::VECIN, 1> queryQueue;
    TQue<QuePosition::VECIN, 1> keyQueue;
    TQue<QuePosition::VECIN, 1> valueQueue;
};
}

extern "C" __global__ __aicore__ void paddle_decode_kv_scatter_query_v1(
    GM_ADDR query,
    GM_ADDR keyCache,
    GM_ADDR valueCache,
    GM_ADDR cachePosition,
    GM_ADDR keyState,
    GM_ADDR valueState,
    GM_ADDR orderedQuery,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(PaddleDecodeKvScatterQueryTilingData);
    GET_TILING_DATA(tilingData, tiling);
    if (GetBlockIdx() != 0 || tilingData.cacheLength != kCacheLength ||
        tilingData.queryElements != kQueryElements ||
        tilingData.stateElements != kStateElements) {
        return;
    }
    TPipe pipe;
    PaddleDecodeKvScatterQueryKernel kernel;
    kernel.Init(
        query, keyCache, valueCache, cachePosition, keyState, valueState,
        orderedQuery, &pipe);
    kernel.Process();
    pipe.Destroy();
}
