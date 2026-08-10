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
constexpr uint32_t kQueryHeads = 16;
constexpr uint32_t kKvHeads = 2;
constexpr uint32_t kCacheLength = 1024;
constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kHalfHeadDim = kHeadDim / 2;
constexpr uint32_t kQueryElements = kQueryHeads * kHeadDim;
constexpr uint32_t kStateElements = kKvHeads * kHeadDim;
constexpr uint32_t kValueOffset = kQueryElements + kStateElements;
constexpr uint32_t kPackedElements = kValueOffset + kStateElements;
constexpr uint32_t kFactorPlaneElements = kCacheLength * kHeadDim;
constexpr uint32_t kMaskWords = kCacheLength / sizeof(uint32_t);
constexpr uint32_t kFourTrueBytes = 0x01010101U;
constexpr uint32_t kAttentionWorkers = 16;

class PaddleDecodePackedQkvRopePrep {
public:
    __aicore__ inline void Init(
        GM_ADDR qkv,
        GM_ADDR keyCache,
        GM_ADDR valueCache,
        GM_ADDR attentionMask,
        GM_ADDR cachePosition,
        GM_ADDR factorLut,
        GM_ADDR ropeDelta,
        TPipe *pipe)
    {
        qkvGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ half *>(qkv), kPackedElements);
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
        factorLutGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ half *>(factorLut),
            2 * kFactorPlaneElements);
        ropeDeltaGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ int64_t *>(ropeDelta), 1);

        pipe->InitBuffer(queryInputQueue, 1, kQueryElements * sizeof(half));
        pipe->InitBuffer(queryOutputQueue, 1, kQueryElements * sizeof(half));
        pipe->InitBuffer(keyInputQueue, 1, kStateElements * sizeof(half));
        pipe->InitBuffer(keyOutputQueue, 1, kStateElements * sizeof(half));
        pipe->InitBuffer(valueInputQueue, 1, kStateElements * sizeof(half));
        pipe->InitBuffer(valueOutputQueue, 1, kStateElements * sizeof(half));
        pipe->InitBuffer(factorInputQueue, 1, 2 * kHeadDim * sizeof(half));
        pipe->InitBuffer(maskOutputQueue, 1, kCacheLength * sizeof(uint8_t));
        pipe->InitBuffer(rotateScratch, kHeadDim * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        DataCacheCleanAndInvalid<
            int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                cachePositionGm);
        DataCacheCleanAndInvalid<
            int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                ropeDeltaGm);
        const int64_t cachePosition = cachePositionGm.GetValue(0);
        const int64_t ropePosition = cachePosition + ropeDeltaGm.GetValue(0);

        LocalTensor<half> queryInput = queryInputQueue.AllocTensor<half>();
        LocalTensor<half> keyInput = keyInputQueue.AllocTensor<half>();
        LocalTensor<half> valueInput = valueInputQueue.AllocTensor<half>();
        LocalTensor<half> factors = factorInputQueue.AllocTensor<half>();
        DataCopy(queryInput, qkvGm, kQueryElements);
        DataCopy(keyInput, qkvGm[kQueryElements], kStateElements);
        DataCopy(valueInput, qkvGm[kValueOffset], kStateElements);
        const uint32_t factorOffset =
            static_cast<uint32_t>(ropePosition) * kHeadDim;
        DataCopy(factors, factorLutGm[factorOffset], kHeadDim);
        DataCopy(
            factors[kHeadDim],
            factorLutGm[kFactorPlaneElements + factorOffset],
            kHeadDim);
        queryInputQueue.EnQue(queryInput);
        keyInputQueue.EnQue(keyInput);
        valueInputQueue.EnQue(valueInput);
        factorInputQueue.EnQue(factors);

        queryInput = queryInputQueue.DeQue<half>();
        keyInput = keyInputQueue.DeQue<half>();
        valueInput = valueInputQueue.DeQue<half>();
        factors = factorInputQueue.DeQue<half>();
        LocalTensor<half> queryOutput = queryOutputQueue.AllocTensor<half>();
        LocalTensor<half> keyOutput = keyOutputQueue.AllocTensor<half>();
        LocalTensor<half> valueOutput = valueOutputQueue.AllocTensor<half>();
        LocalTensor<half> scratch = rotateScratch.Get<half>();
        RotateHalf(
            queryOutput, queryInput, factors, factors[kHeadDim],
            kQueryHeads, scratch);
        RotateHalf(
            keyOutput, keyInput, factors, factors[kHeadDim],
            kKvHeads, scratch);
        Adds(valueOutput, valueInput, static_cast<half>(0.0f), kStateElements);
        queryOutputQueue.EnQue(queryOutput);
        keyOutputQueue.EnQue(keyOutput);
        valueOutputQueue.EnQue(valueOutput);
        queryInputQueue.FreeTensor(queryInput);
        keyInputQueue.FreeTensor(keyInput);
        valueInputQueue.FreeTensor(valueInput);
        factorInputQueue.FreeTensor(factors);

        LocalTensor<uint32_t> maskWords =
            maskOutputQueue.AllocTensor<uint32_t>();
        Duplicate<uint32_t>(maskWords, kFourTrueBytes, kMaskWords);
        const uint32_t prefixBytes = static_cast<uint32_t>(cachePosition + 1);
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

        queryOutput = queryOutputQueue.DeQue<half>();
        keyOutput = keyOutputQueue.DeQue<half>();
        valueOutput = valueOutputQueue.DeQue<half>();
        maskWords = maskOutputQueue.DeQue<uint32_t>();
        DataCopy(qkvGm, queryOutput, kQueryElements);
        DataCopy(qkvGm[kQueryElements], keyOutput, kStateElements);
        DataCopy(
            attentionMaskGm,
            maskWords.ReinterpretCast<uint8_t>(),
            kCacheLength);
        for (uint32_t head = 0; head < kKvHeads; ++head) {
            const uint32_t stateOffset = head * kHeadDim;
            const uint32_t cacheOffset =
                (head * kCacheLength + static_cast<uint32_t>(cachePosition)) *
                kHeadDim;
            DataCopy(
                keyCacheGm[cacheOffset], keyOutput[stateOffset], kHeadDim);
            DataCopy(
                valueCacheGm[cacheOffset], valueOutput[stateOffset], kHeadDim);
        }
        queryOutputQueue.FreeTensor(queryOutput);
        keyOutputQueue.FreeTensor(keyOutput);
        valueOutputQueue.FreeTensor(valueOutput);
        maskOutputQueue.FreeTensor(maskWords);
    }

private:
    __aicore__ inline void RotateHalf(
        LocalTensor<half> output,
        LocalTensor<half> input,
        LocalTensor<half> cosine,
        LocalTensor<half> sine,
        uint32_t heads,
        LocalTensor<half> scratch)
    {
        for (uint32_t head = 0; head < heads; ++head) {
            const uint32_t offset = head * kHeadDim;
            Adds(
                scratch,
                input[offset + kHalfHeadDim],
                static_cast<half>(0.0f),
                kHalfHeadDim);
            PipeBarrier<PIPE_V>();
            Muls(
                scratch,
                scratch,
                static_cast<half>(-1.0f),
                kHalfHeadDim);
            PipeBarrier<PIPE_V>();
            Adds(
                scratch[kHalfHeadDim],
                input[offset],
                static_cast<half>(0.0f),
                kHalfHeadDim);
            PipeBarrier<PIPE_V>();
            Mul(output[offset], input[offset], cosine, kHeadDim);
            PipeBarrier<PIPE_V>();
            Mul(scratch, scratch, sine, kHeadDim);
            PipeBarrier<PIPE_V>();
            Add(output[offset], output[offset], scratch, kHeadDim);
            PipeBarrier<PIPE_V>();
        }
    }

    GlobalTensor<half> qkvGm;
    GlobalTensor<half> keyCacheGm;
    GlobalTensor<half> valueCacheGm;
    GlobalTensor<uint8_t> attentionMaskGm;
    GlobalTensor<int64_t> cachePositionGm;
    GlobalTensor<half> factorLutGm;
    GlobalTensor<int64_t> ropeDeltaGm;
    TQue<QuePosition::VECIN, 1> queryInputQueue;
    TQue<QuePosition::VECOUT, 1> queryOutputQueue;
    TQue<QuePosition::VECIN, 1> keyInputQueue;
    TQue<QuePosition::VECOUT, 1> keyOutputQueue;
    TQue<QuePosition::VECIN, 1> valueInputQueue;
    TQue<QuePosition::VECOUT, 1> valueOutputQueue;
    TQue<QuePosition::VECIN, 1> factorInputQueue;
    TQue<QuePosition::VECOUT, 1> maskOutputQueue;
    TBuf<TPosition::VECCALC> rotateScratch;
};
} // namespace

extern "C" __global__ __aicore__ void paddle_decode_gqa_incre_flash_attention_aiv(
    __gm__ uint8_t *qkv,
    __gm__ uint8_t *key,
    __gm__ uint8_t *value,
    __gm__ uint8_t *attenMask,
    __gm__ uint8_t *cachePosition,
    __gm__ uint8_t *factorLut,
    __gm__ uint8_t *ropeDelta,
    __gm__ uint8_t *attentionOut,
    __gm__ uint8_t *keyOut,
    __gm__ uint8_t *valueOut,
    __gm__ uint8_t *maskOut,
    __gm__ uint8_t *qkvOut,
    __gm__ uint8_t *workspace,
    __gm__ uint8_t *tiling)
{
    (void)keyOut;
    (void)valueOut;
    (void)maskOut;
    (void)qkvOut;

    if (g_coreType == AIC) {
        return;
    }

    TPipe fusedPipe;
    if (GetBlockIdx() == 0) {
        PaddleDecodePackedQkvRopePrep prep;
        prep.Init(
            qkv,
            key,
            value,
            attenMask,
            cachePosition,
            factorLut,
            ropeDelta,
            &fusedPipe);
        prep.Process();
    }

    // The hardware barrier covers all 24 AIV workers in the Cube-first launch.
    // It replaces both the old QKV subkernel boundary and the GM/UB software
    // barrier. The first 16 workers then execute the inherited attention body.
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
    if (GetBlockIdx() >= kAttentionWorkers) {
        return;
    }
    PipeBarrier<PIPE_ALL>();
    fusedPipe.Reset();

    incre_flash_attention_FIAS_arch32(
        qkv,
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
        tiling,
        &fusedPipe);
}
