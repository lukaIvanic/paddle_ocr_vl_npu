/*
 * The execution schedule in this lab kernel follows CANN 9.0's
 * AddRmsNorm SingleN FP16 implementation.  The experiment changes only the
 * reciprocal-RMS application: keep the scalar result on the Vector pipeline
 * with Brcb instead of crossing Vector -> Scalar -> Vector through GetValue.
 */
#include "kernel_operator.h"
#include "vector_add_rms_norm_tiling.h"

using namespace AscendC;

namespace {
constexpr uint32_t kHiddenSize = 1024;
constexpr uint32_t kFp16Bytes = kHiddenSize * sizeof(half);
constexpr uint32_t kVectorMaskFp32 = 64;
constexpr uint32_t kVectorRepeatsFp32 = kHiddenSize / kVectorMaskFp32;
constexpr uint32_t kUbBytes = 4 * kHiddenSize * sizeof(float);
constexpr float kEpsilon = 1.0e-5f;
constexpr float kMeanScale = 1.0f / static_cast<float>(kHiddenSize);

class VectorAddRmsNormKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR x1,
        GM_ADDR x2,
        GM_ADDR gamma,
        GM_ADDR y,
        GM_ADDR rstd,
        GM_ADDR x,
        TPipe* pipe)
    {
        x1Gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x1), kHiddenSize);
        x2Gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x2), kHiddenSize);
        gammaGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(gamma), kHiddenSize);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(y), kHiddenSize);
        rstdGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(rstd), 1);
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x), kHiddenSize);
        pipe->InitBuffer(unitBuffer, kUbBytes);
    }

    __aicore__ inline void Process()
    {
        LocalTensor<float> ub = unitBuffer.Get<float>();
        LocalTensor<half> ubFp16 = ub.ReinterpretCast<half>();
        LocalTensor<half> xLocal = ubFp16;
        LocalTensor<half> auxiliaryFp16 = ubFp16[kHiddenSize];
        LocalTensor<float> xFp32 = ub[kHiddenSize];
        LocalTensor<float> square = ub[2 * kHiddenSize];
        LocalTensor<float> work = ub[3 * kHiddenSize];

        DataCopy(xLocal, x1Gm, kHiddenSize);
        event_t x1Ready = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(x1Ready);
        DataCopy(auxiliaryFp16, x2Gm, kHiddenSize);
        event_t x2Ready = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(x2Ready);
        WaitFlag<HardEvent::MTE2_V>(x1Ready);
        WaitFlag<HardEvent::MTE2_V>(x2Ready);

        Add(xLocal, xLocal, auxiliaryFp16, kHiddenSize);
        PipeBarrier<PIPE_V>();

        // Reuse auxiliaryFp16 for gamma while Vector computes the norm.
        event_t vectorAllowsGammaLoad = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(vectorAllowsGammaLoad);
        WaitFlag<HardEvent::V_MTE2>(vectorAllowsGammaLoad);
        DataCopy(auxiliaryFp16, gammaGm, kHiddenSize);
        SetFlag<HardEvent::MTE2_V>(x2Ready);

        // Store the fused residual in parallel with the remaining Vector work.
        event_t vectorAllowsStore = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(vectorAllowsStore);
        WaitFlag<HardEvent::V_MTE3>(vectorAllowsStore);
        DataCopy(xGm, xLocal, kHiddenSize);
        event_t residualStored = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(residualStored);

        Cast(xFp32, xLocal, RoundMode::CAST_NONE, kHiddenSize);
        PipeBarrier<PIPE_V>();
        Mul(square, xFp32, xFp32, kHiddenSize);
        PipeBarrier<PIPE_V>();
        Muls(square, square, kMeanScale, kHiddenSize);
        PipeBarrier<PIPE_V>();
        ReduceSum(square, square, work);
        Adds(square, square, kEpsilon, 1);
        PipeBarrier<PIPE_V>();
        Sqrt(square, square, 1);
        Duplicate(work, 1.0f, 1);
        PipeBarrier<PIPE_V>();
        Div(square, work, square, 1);
        PipeBarrier<PIPE_V>();

        // rstd is a real graph output even though the decoder ignores it.
        SetFlag<HardEvent::V_MTE3>(vectorAllowsStore);
        WaitFlag<HardEvent::V_MTE3>(vectorAllowsStore);
        DataCopyParams rstdParams;
        rstdParams.blockCount = 1;
        rstdParams.blockLen = sizeof(float);
        rstdParams.srcStride = 0;
        rstdParams.dstStride = 0;
        DataCopyPad(rstdGm, square, rstdParams);

        // Experimental change from stock SingleN: no Vector -> Scalar
        // GetValue and no Scalar -> Vector Muls parameter.
        Brcb(work, square, 1, {1, 8});
        PipeBarrier<PIPE_V>();
        MultiplyBroadcast(xFp32, work);

        WaitFlag<HardEvent::MTE3_V>(residualStored);
        Cast(xLocal, xFp32, RoundMode::CAST_NONE, kHiddenSize);
        PipeBarrier<PIPE_V>();
        WaitFlag<HardEvent::MTE2_V>(x2Ready);
        Mul(xLocal, xLocal, auxiliaryFp16, kHiddenSize);
        SetFlag<HardEvent::V_MTE3>(vectorAllowsStore);
        WaitFlag<HardEvent::V_MTE3>(vectorAllowsStore);
        DataCopy(yGm, xLocal, kHiddenSize);
    }

private:
    __aicore__ inline void ReduceSum(
        const LocalTensor<float>& dst,
        const LocalTensor<float>& src,
        const LocalTensor<float>& work)
    {
        BinaryRepeatParams params;
        params.src0RepStride = 8;
        params.src0BlkStride = 1;
        params.src1RepStride = 0;
        params.src1BlkStride = 1;
        params.dstRepStride = 0;
        params.dstBlkStride = 1;
        Duplicate(work, 0.0f, kVectorMaskFp32);
        PipeBarrier<PIPE_V>();
        Add(
            work,
            src,
            work,
            static_cast<uint64_t>(kVectorMaskFp32),
            kVectorRepeatsFp32,
            params);
        PipeBarrier<PIPE_V>();
        AscendCUtils::SetMask<float>(kVectorMaskFp32);
        WholeReduceSum<float, false>(
            dst,
            work,
            MASK_PLACEHOLDER,
            1,
            0,
            1,
            0);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void MultiplyBroadcast(
        const LocalTensor<float>& x,
        const LocalTensor<float>& broadcast)
    {
        BinaryRepeatParams params;
        params.dstBlkStride = 1;
        params.src0BlkStride = 1;
        params.src1BlkStride = 0;
        params.dstRepStride = 8;
        params.src0RepStride = 8;
        params.src1RepStride = 0;
        Mul(
            x,
            x,
            broadcast,
            static_cast<uint64_t>(kVectorMaskFp32),
            kVectorRepeatsFp32,
            params);
        PipeBarrier<PIPE_V>();
    }

    GlobalTensor<half> x1Gm;
    GlobalTensor<half> x2Gm;
    GlobalTensor<half> gammaGm;
    GlobalTensor<half> yGm;
    GlobalTensor<float> rstdGm;
    GlobalTensor<half> xGm;
    TBuf<TPosition::VECCALC> unitBuffer;
};
}

extern "C" __global__ __aicore__ void vector_add_rms_norm(
    GM_ADDR x1,
    GM_ADDR x2,
    GM_ADDR gamma,
    GM_ADDR y,
    GM_ADDR rstd,
    GM_ADDR x,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(VectorAddRmsNormTilingData);
    GET_TILING_DATA(tilingData, tiling);
    if (tilingData.size != kHiddenSize || GetBlockIdx() != 0) {
        return;
    }
    TPipe pipe;
    VectorAddRmsNormKernel kernel;
    kernel.Init(x1, x2, gamma, y, rstd, x, &pipe);
    kernel.Process();
}
