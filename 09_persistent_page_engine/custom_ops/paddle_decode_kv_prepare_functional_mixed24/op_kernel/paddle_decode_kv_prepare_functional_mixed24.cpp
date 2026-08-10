#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "paddle_decode_kv_prepare_functional_mixed24_tiling.h"

using namespace AscendC;

namespace {
constexpr uint32_t kAivCores = 24;
constexpr uint32_t kQueryHeads = 16;
constexpr uint32_t kKvHeads = 2;
constexpr uint32_t kCacheLength = 1024;
constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kQueryElements = kQueryHeads * kHeadDim;
constexpr uint32_t kStateElements = kKvHeads * kHeadDim;
constexpr uint32_t kCacheElements = kKvHeads * kCacheLength * kHeadDim;
constexpr uint32_t kCopyElementsPerCore = 10928;
constexpr uint32_t kMaskWords = kCacheLength / sizeof(uint32_t);
constexpr uint32_t kFourTrueBytes = 0x01010101U;

class Kernel {
public:
    __aicore__ inline void Init(
        GM_ADDR query,
        GM_ADDR keyCache,
        GM_ADDR valueCache,
        GM_ADDR cachePosition,
        GM_ADDR keyState,
        GM_ADDR valueState,
        GM_ADDR orderedQuery,
        GM_ADDR attentionMask,
        GM_ADDR keyCacheOut,
        GM_ADDR valueCacheOut,
        TPipe* pipe)
    {
        queryGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ half*>(query), kQueryElements);
        keyCacheGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ half*>(keyCache), kCacheElements);
        valueCacheGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ half*>(valueCache), kCacheElements);
        cachePositionGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ int64_t*>(cachePosition), 1);
        keyStateGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ half*>(keyState), kStateElements);
        valueStateGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ half*>(valueState), kStateElements);
        orderedQueryGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ half*>(orderedQuery), kQueryElements);
        attentionMaskGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ uint8_t*>(attentionMask), kCacheLength);
        keyCacheOutGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ half*>(keyCacheOut), kCacheElements);
        valueCacheOutGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ half*>(valueCacheOut), kCacheElements);
        pipe->InitBuffer(
            copyQueue, 1, kCopyElementsPerCore * sizeof(half));
        pipe->InitBuffer(tokenQueue, 1, kQueryElements * sizeof(half));
        pipe->InitBuffer(maskQueue, 1, kCacheLength * sizeof(uint8_t));
    }

    __aicore__ inline void CopyCaches(uint32_t blockIndex)
    {
        const uint32_t start = blockIndex * kCopyElementsPerCore;
        if (start >= kCacheElements) {
            return;
        }
        const uint32_t remaining = kCacheElements - start;
        const uint32_t count =
            remaining < kCopyElementsPerCore
                ? remaining
                : kCopyElementsPerCore;
        CopySegment(keyCacheGm, keyCacheOutGm, start, count);
        CopySegment(valueCacheGm, valueCacheOutGm, start, count);
    }

    __aicore__ inline void PrepareToken()
    {
        DataCacheCleanAndInvalid<
            int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                cachePositionGm);
        const int64_t position = cachePositionGm.GetValue(0);
        if (position < 0 || position >= static_cast<int64_t>(kCacheLength)) {
            return;
        }

        LocalTensor<uint32_t> maskWords =
            maskQueue.AllocTensor<uint32_t>();
        Duplicate<uint32_t>(maskWords, kFourTrueBytes, kMaskWords);
        const uint32_t prefixBytes = static_cast<uint32_t>(position + 1);
        const uint32_t fullZeroWords = prefixBytes / sizeof(uint32_t);
        const uint32_t remainingZeroBytes = prefixBytes % sizeof(uint32_t);
        if (fullZeroWords > 0) {
            Duplicate<uint32_t>(maskWords, 0, fullZeroWords);
        }
        PipeBarrier<PIPE_V>();
        if (remainingZeroBytes > 0) {
            maskWords.SetValue(
                fullZeroWords,
                kFourTrueBytes << (remainingZeroBytes * 8));
        }
        maskQueue.EnQue(maskWords);
        maskWords = maskQueue.DeQue<uint32_t>();
        DataCopy(
            attentionMaskGm,
            maskWords.ReinterpretCast<uint8_t>(),
            kCacheLength);
        maskQueue.FreeTensor(maskWords);

        CopyToken(queryGm, orderedQueryGm, 0, 0, kQueryElements);
        for (uint32_t head = 0; head < kKvHeads; ++head) {
            const uint32_t stateOffset = head * kHeadDim;
            const uint32_t cacheOffset =
                (head * kCacheLength + static_cast<uint32_t>(position)) *
                kHeadDim;
            CopyToken(
                keyStateGm,
                keyCacheOutGm,
                stateOffset,
                cacheOffset,
                kHeadDim);
            CopyToken(
                valueStateGm,
                valueCacheOutGm,
                stateOffset,
                cacheOffset,
                kHeadDim);
        }
    }

private:
    __aicore__ inline void CopySegment(
        GlobalTensor<half>& source,
        GlobalTensor<half>& destination,
        uint32_t start,
        uint32_t count)
    {
        LocalTensor<half> local = copyQueue.AllocTensor<half>();
        DataCopy(local, source[start], count);
        copyQueue.EnQue(local);
        local = copyQueue.DeQue<half>();
        DataCopy(destination[start], local, count);
        copyQueue.FreeTensor(local);
    }

    __aicore__ inline void CopyToken(
        GlobalTensor<half>& source,
        GlobalTensor<half>& destination,
        uint32_t sourceOffset,
        uint32_t destinationOffset,
        uint32_t count)
    {
        LocalTensor<half> local = tokenQueue.AllocTensor<half>();
        DataCopy(local, source[sourceOffset], count);
        tokenQueue.EnQue(local);
        local = tokenQueue.DeQue<half>();
        DataCopy(destination[destinationOffset], local, count);
        tokenQueue.FreeTensor(local);
    }

    GlobalTensor<half> queryGm;
    GlobalTensor<half> keyCacheGm;
    GlobalTensor<half> valueCacheGm;
    GlobalTensor<int64_t> cachePositionGm;
    GlobalTensor<half> keyStateGm;
    GlobalTensor<half> valueStateGm;
    GlobalTensor<half> orderedQueryGm;
    GlobalTensor<uint8_t> attentionMaskGm;
    GlobalTensor<half> keyCacheOutGm;
    GlobalTensor<half> valueCacheOutGm;
    TQue<QuePosition::VECIN, 1> copyQueue;
    TQue<QuePosition::VECIN, 1> tokenQueue;
    TQue<QuePosition::VECOUT, 1> maskQueue;
};
}

extern "C" __global__ __aicore__ void
paddle_decode_kv_prepare_functional_mixed24(
    GM_ADDR query,
    GM_ADDR keyCache,
    GM_ADDR valueCache,
    GM_ADDR cachePosition,
    GM_ADDR keyState,
    GM_ADDR valueState,
    GM_ADDR orderedQuery,
    GM_ADDR attentionMask,
    GM_ADDR keyCacheOut,
    GM_ADDR valueCacheOut,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    REGISTER_TILING_DEFAULT(PaddleDecodeKvPrepareFunctionalMixed24TilingData);
    GET_TILING_DATA(tilingData, tiling);
    (void)workspace;
    if (g_coreType == AIC || GetBlockIdx() >= kAivCores ||
        tilingData.cacheElements != kCacheElements ||
        tilingData.copyElementsPerCore != kCopyElementsPerCore) {
        return;
    }

    TPipe pipe;
    Kernel kernel;
    kernel.Init(
        query, keyCache, valueCache, cachePosition, keyState, valueState,
        orderedQuery, attentionMask, keyCacheOut, valueCacheOut, &pipe);
    kernel.CopyCaches(GetBlockIdx());
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
    if (GetBlockIdx() == 0) {
        kernel.PrepareToken();
    }
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
    pipe.Destroy();
}
