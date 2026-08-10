#include "kernel_operator.h"
#include "paddle_decode_rope_lookup_tiling.h"

using namespace AscendC;

namespace {
constexpr uint32_t kCacheLength = 1024;
constexpr uint32_t kHeadDim = 128;

class PaddleDecodeRopeLookupKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR factorLut,
        GM_ADDR cachePosition,
        GM_ADDR ropeDelta,
        GM_ADDR cos,
        GM_ADDR sin,
        TPipe* pipe)
    {
        factorLutGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ half*>(factorLut),
            2 * kCacheLength * kHeadDim);
        cachePositionGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ int64_t*>(cachePosition), 1);
        ropeDeltaGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ int64_t*>(ropeDelta), 1);
        cosGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(cos), kHeadDim);
        sinGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(sin), kHeadDim);
        pipe->InitBuffer(cosInputQueue, 1, kHeadDim * sizeof(half));
        pipe->InitBuffer(cosOutputQueue, 1, kHeadDim * sizeof(half));
        pipe->InitBuffer(sinInputQueue, 1, kHeadDim * sizeof(half));
        pipe->InitBuffer(sinOutputQueue, 1, kHeadDim * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        DataCacheCleanAndInvalid<
            int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                cachePositionGm);
        DataCacheCleanAndInvalid<
            int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                ropeDeltaGm);
        const int64_t position =
            cachePositionGm.GetValue(0) + ropeDeltaGm.GetValue(0);
        if (position < 0 || position >= static_cast<int64_t>(kCacheLength)) {
            return;
        }
        const uint32_t cosOffset = static_cast<uint32_t>(position) * kHeadDim;
        const uint32_t sinOffset = kCacheLength * kHeadDim + cosOffset;

        LocalTensor<half> cosInput = cosInputQueue.AllocTensor<half>();
        LocalTensor<half> sinInput = sinInputQueue.AllocTensor<half>();
        DataCopy(cosInput, factorLutGm[cosOffset], kHeadDim);
        DataCopy(sinInput, factorLutGm[sinOffset], kHeadDim);
        cosInputQueue.EnQue(cosInput);
        sinInputQueue.EnQue(sinInput);

        cosInput = cosInputQueue.DeQue<half>();
        sinInput = sinInputQueue.DeQue<half>();
        LocalTensor<half> cosOutput = cosOutputQueue.AllocTensor<half>();
        LocalTensor<half> sinOutput = sinOutputQueue.AllocTensor<half>();
        Adds(cosOutput, cosInput, static_cast<half>(0.0f), kHeadDim);
        Adds(sinOutput, sinInput, static_cast<half>(0.0f), kHeadDim);
        cosOutputQueue.EnQue(cosOutput);
        sinOutputQueue.EnQue(sinOutput);
        cosInputQueue.FreeTensor(cosInput);
        sinInputQueue.FreeTensor(sinInput);

        cosOutput = cosOutputQueue.DeQue<half>();
        sinOutput = sinOutputQueue.DeQue<half>();
        DataCopy(cosGm, cosOutput, kHeadDim);
        DataCopy(sinGm, sinOutput, kHeadDim);
        cosOutputQueue.FreeTensor(cosOutput);
        sinOutputQueue.FreeTensor(sinOutput);
    }

private:
    GlobalTensor<half> factorLutGm;
    GlobalTensor<int64_t> cachePositionGm;
    GlobalTensor<int64_t> ropeDeltaGm;
    GlobalTensor<half> cosGm;
    GlobalTensor<half> sinGm;
    TQue<QuePosition::VECIN, 1> cosInputQueue;
    TQue<QuePosition::VECOUT, 1> cosOutputQueue;
    TQue<QuePosition::VECIN, 1> sinInputQueue;
    TQue<QuePosition::VECOUT, 1> sinOutputQueue;
};
}

extern "C" __global__ __aicore__ void paddle_decode_rope_lookup_v1(
    GM_ADDR factorLut,
    GM_ADDR cachePosition,
    GM_ADDR ropeDelta,
    GM_ADDR cos,
    GM_ADDR sin,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(PaddleDecodeRopeLookupTilingData);
    GET_TILING_DATA(tilingData, tiling);
    if (GetBlockIdx() != 0 || tilingData.cacheLength != kCacheLength ||
        tilingData.headDim != kHeadDim) {
        return;
    }
    TPipe pipe;
    PaddleDecodeRopeLookupKernel kernel;
    kernel.Init(factorLut, cachePosition, ropeDelta, cos, sin, &pipe);
    kernel.Process();
    pipe.Destroy();
}
