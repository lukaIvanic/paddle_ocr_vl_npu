#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif

#define PADDLE_DECODE_GQA_PLAIN_KV 1
#include "incre_flash_attention_arch32.h"

using namespace AscendC;

namespace {
constexpr uint32_t kKvHeads = 2;
constexpr uint32_t kCacheLength = 1024;
constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kStateElements = kKvHeads * kHeadDim;
constexpr uint32_t kMaskWords = kCacheLength / sizeof(uint32_t);
constexpr uint32_t kFourTrueBytes = 0x01010101U;

class PaddleDecodeAttentionPrep {
public:
    __aicore__ inline void Init(
        GM_ADDR keyCache,
        GM_ADDR valueCache,
        GM_ADDR attentionMask,
        GM_ADDR cachePosition,
        GM_ADDR keyState,
        GM_ADDR valueState,
        TPipe *pipe)
    {
        keyCacheGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ half *>(keyCache),
            kKvHeads * kCacheLength * kHeadDim);
        valueCacheGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ half *>(valueCache),
            kKvHeads * kCacheLength * kHeadDim);
        attentionMaskGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ uint8_t *>(attentionMask), kCacheLength);
        cachePositionGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ int64_t *>(cachePosition), 1);
        keyStateGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ half *>(keyState), kStateElements);
        valueStateGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ half *>(valueState), kStateElements);
        pipe->InitBuffer(keyInputQueue, 1, kStateElements * sizeof(half));
        pipe->InitBuffer(valueInputQueue, 1, kStateElements * sizeof(half));
        pipe->InitBuffer(keyOutputQueue, 1, kStateElements * sizeof(half));
        pipe->InitBuffer(valueOutputQueue, 1, kStateElements * sizeof(half));
        pipe->InitBuffer(maskOutputQueue, 1, kCacheLength * sizeof(uint8_t));
    }

    __aicore__ inline void Process()
    {
        DataCacheCleanAndInvalid<
            int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                cachePositionGm);
        const int64_t position = cachePositionGm.GetValue(0);

        LocalTensor<uint32_t> maskWords =
            maskOutputQueue.AllocTensor<uint32_t>();
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
        maskOutputQueue.EnQue(maskWords);
        maskWords = maskOutputQueue.DeQue<uint32_t>();
        DataCopy(
            attentionMaskGm,
            maskWords.ReinterpretCast<uint8_t>(),
            kCacheLength);
        maskOutputQueue.FreeTensor(maskWords);

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
    GlobalTensor<half> keyCacheGm;
    GlobalTensor<half> valueCacheGm;
    GlobalTensor<uint8_t> attentionMaskGm;
    GlobalTensor<int64_t> cachePositionGm;
    GlobalTensor<half> keyStateGm;
    GlobalTensor<half> valueStateGm;
    TQue<QuePosition::VECIN, 1> keyInputQueue;
    TQue<QuePosition::VECIN, 1> valueInputQueue;
    TQue<QuePosition::VECOUT, 1> keyOutputQueue;
    TQue<QuePosition::VECOUT, 1> valueOutputQueue;
    TQue<QuePosition::VECOUT, 1> maskOutputQueue;
};
} // namespace

extern "C" __global__ __aicore__ void paddle_decode_gqa_incre_flash_attention_aiv(
    __gm__ uint8_t *query,
    __gm__ uint8_t *key,
    __gm__ uint8_t *value,
    __gm__ uint8_t *attenMask,
    __gm__ uint8_t *cachePosition,
    __gm__ uint8_t *keyState,
    __gm__ uint8_t *valueState,
    __gm__ uint8_t *attentionOut,
    __gm__ uint8_t *keyOut,
    __gm__ uint8_t *valueOut,
    __gm__ uint8_t *maskOut,
    __gm__ uint8_t *workspace,
    __gm__ uint8_t *tiling)
{
    (void)keyOut;
    (void)valueOut;
    (void)maskOut;
    if (GetBlockIdx() == 0) {
        TPipe pipe;
        PaddleDecodeAttentionPrep prep;
        prep.Init(
            key,
            value,
            attenMask,
            cachePosition,
            keyState,
            valueState,
            &pipe);
        prep.Process();
        pipe.Destroy();
    }
    // Every launched AIV reaches this barrier. Core 0's GM writes are visible
    // before any worker starts reading the cache or mask for attention.
    SyncAll();
    incre_flash_attention_FIAS_arch32(
        query,
        key,
        value,
        nullptr,
        attenMask,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        attentionOut,
        nullptr,
        workspace,
        tiling);
}
