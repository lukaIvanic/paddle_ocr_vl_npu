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
        pipe->InitBuffer(queryInputQueue, 1, kQueryElements * sizeof(half));
        pipe->InitBuffer(keyInputQueue, 1, kStateElements * sizeof(half));
        pipe->InitBuffer(valueInputQueue, 1, kStateElements * sizeof(half));
        pipe->InitBuffer(queryOutputQueue, 1, kQueryElements * sizeof(half));
        pipe->InitBuffer(keyOutputQueue, 1, kStateElements * sizeof(half));
        pipe->InitBuffer(valueOutputQueue, 1, kStateElements * sizeof(half));
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

        LocalTensor<half> queryInput = queryInputQueue.AllocTensor<half>();
        DataCopy(queryInput, queryGm, kQueryElements);
        queryInputQueue.EnQue(queryInput);
        queryInput = queryInputQueue.DeQue<half>();
        LocalTensor<half> queryOutput = queryOutputQueue.AllocTensor<half>();
        Adds(queryOutput, queryInput, static_cast<half>(0.0f), kQueryElements);
        queryOutputQueue.EnQue(queryOutput);
        queryInputQueue.FreeTensor(queryInput);
        queryOutput = queryOutputQueue.DeQue<half>();
        DataCopy(orderedQueryGm, queryOutput, kQueryElements);
        queryOutputQueue.FreeTensor(queryOutput);

        LocalTensor<half> keyInput = keyInputQueue.AllocTensor<half>();
        DataCopy(keyInput, keyStateGm, kStateElements);
        keyInputQueue.EnQue(keyInput);
        keyInput = keyInputQueue.DeQue<half>();
        LocalTensor<half> keyOutput = keyOutputQueue.AllocTensor<half>();
        Adds(keyOutput, keyInput, static_cast<half>(0.0f), kStateElements);
        keyOutputQueue.EnQue(keyOutput);
        keyInputQueue.FreeTensor(keyInput);
        keyOutput = keyOutputQueue.DeQue<half>();
        for (uint32_t head = 0; head < kKvHeads; ++head) {
            const uint32_t stateOffset = head * kHeadDim;
            const uint32_t cacheOffset =
                (head * kCacheLength + static_cast<uint32_t>(position)) * kHeadDim;
            DataCopy(keyCacheGm[cacheOffset], keyOutput[stateOffset], kHeadDim);
        }
        keyOutputQueue.FreeTensor(keyOutput);

        LocalTensor<half> valueInput = valueInputQueue.AllocTensor<half>();
        DataCopy(valueInput, valueStateGm, kStateElements);
        valueInputQueue.EnQue(valueInput);
        valueInput = valueInputQueue.DeQue<half>();
        LocalTensor<half> valueOutput = valueOutputQueue.AllocTensor<half>();
        Adds(valueOutput, valueInput, static_cast<half>(0.0f), kStateElements);
        valueOutputQueue.EnQue(valueOutput);
        valueInputQueue.FreeTensor(valueInput);
        valueOutput = valueOutputQueue.DeQue<half>();
        for (uint32_t head = 0; head < kKvHeads; ++head) {
            const uint32_t stateOffset = head * kHeadDim;
            const uint32_t cacheOffset =
                (head * kCacheLength + static_cast<uint32_t>(position)) * kHeadDim;
            DataCopy(valueCacheGm[cacheOffset], valueOutput[stateOffset], kHeadDim);
        }
        valueOutputQueue.FreeTensor(valueOutput);
    }

private:
    GlobalTensor<half> queryGm;
    GlobalTensor<half> keyCacheGm;
    GlobalTensor<half> valueCacheGm;
    GlobalTensor<int64_t> cachePositionGm;
    GlobalTensor<half> keyStateGm;
    GlobalTensor<half> valueStateGm;
    GlobalTensor<half> orderedQueryGm;
    TQue<QuePosition::VECIN, 1> queryInputQueue;
    TQue<QuePosition::VECIN, 1> keyInputQueue;
    TQue<QuePosition::VECIN, 1> valueInputQueue;
    TQue<QuePosition::VECOUT, 1> queryOutputQueue;
    TQue<QuePosition::VECOUT, 1> keyOutputQueue;
    TQue<QuePosition::VECOUT, 1> valueOutputQueue;
};
}

extern "C" __global__ __aicore__ void paddle_decode_kv_scatter_query_v2(
    GM_ADDR query,
    GM_ADDR keyCache,
    GM_ADDR valueCache,
    GM_ADDR cachePosition,
    GM_ADDR keyState,
    GM_ADDR valueState,
    GM_ADDR orderedQuery,
    GM_ADDR keyCacheOut,
    GM_ADDR valueCacheOut,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(PaddleDecodeKvScatterQueryTilingData);
    GET_TILING_DATA(tilingData, tiling);
    (void)keyCacheOut;
    (void)valueCacheOut;
    (void)workspace;
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
