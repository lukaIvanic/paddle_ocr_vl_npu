#include "kernel_operator.h"
#include "vector_add_rms_norm_tiling.h"

using namespace AscendC;

namespace {
constexpr uint32_t kHiddenSize = 1024;
constexpr uint32_t kFp16Bytes = kHiddenSize * sizeof(half);
constexpr uint32_t kFp32Bytes = kHiddenSize * sizeof(float);
constexpr uint32_t kVectorMaskFp32 = 64;
constexpr uint32_t kVectorRepeatsFp32 = kHiddenSize / kVectorMaskFp32;
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

        pipe->InitBuffer(x1Queue, 1, kFp16Bytes);
        pipe->InitBuffer(x2Queue, 1, kFp16Bytes);
        pipe->InitBuffer(gammaQueue, 1, kFp16Bytes);
        pipe->InitBuffer(yQueue, 1, kFp16Bytes);
        pipe->InitBuffer(xQueue, 1, kFp16Bytes);
        pipe->InitBuffer(xFp32Buffer, kFp32Bytes);
        pipe->InitBuffer(squareBuffer, kFp32Bytes);
        pipe->InitBuffer(workBuffer, kFp32Bytes);
        pipe->InitBuffer(rstdBuffer, 32);
    }

    __aicore__ inline void Process()
    {
        CopyInputs();

        LocalTensor<half> x1Local = x1Queue.DeQue<half>();
        LocalTensor<half> x2Local = x2Queue.DeQue<half>();
        LocalTensor<half> gammaLocal = gammaQueue.DeQue<half>();
        LocalTensor<half> xLocal = xQueue.AllocTensor<half>();
        LocalTensor<half> yLocal = yQueue.AllocTensor<half>();
        LocalTensor<float> xFp32 = xFp32Buffer.Get<float>();
        LocalTensor<float> square = squareBuffer.Get<float>();
        LocalTensor<float> work = workBuffer.Get<float>();
        LocalTensor<float> rstdLocal = rstdBuffer.Get<float>();

        Add(xLocal, x1Local, x2Local, kHiddenSize);
        PipeBarrier<PIPE_V>();
        Cast(xFp32, xLocal, RoundMode::CAST_NONE, kHiddenSize);
        PipeBarrier<PIPE_V>();
        Mul(square, xFp32, xFp32, kHiddenSize);
        PipeBarrier<PIPE_V>();
        Muls(square, square, kMeanScale, kHiddenSize);
        PipeBarrier<PIPE_V>();
        ReduceSum(rstdLocal, square, work);
        Adds(rstdLocal, rstdLocal, kEpsilon, 1);
        PipeBarrier<PIPE_V>();
        Sqrt(rstdLocal, rstdLocal, 1);
        Duplicate(work, 1.0f, 1);
        PipeBarrier<PIPE_V>();
        Div(rstdLocal, work, rstdLocal, 1);
        PipeBarrier<PIPE_V>();

        // Keep the reciprocal RMS on the Vector pipeline.  The stock B1
        // kernel crosses Vector -> Scalar with GetValue(), then Scalar ->
        // Vector for Muls().  Brcb plus a zero-stride vector operand applies
        // the same one-element value without either cross-pipeline wait.
        Brcb(work, rstdLocal, 1, {1, 8});
        PipeBarrier<PIPE_V>();
        MultiplyBroadcast(xFp32, work);
        Cast(yLocal, xFp32, RoundMode::CAST_NONE, kHiddenSize);
        PipeBarrier<PIPE_V>();
        Mul(yLocal, yLocal, gammaLocal, kHiddenSize);
        PipeBarrier<PIPE_V>();

        xQueue.EnQue(xLocal);
        yQueue.EnQue(yLocal);
        CopyOutputs(rstdLocal);

        x1Queue.FreeTensor(x1Local);
        x2Queue.FreeTensor(x2Local);
        gammaQueue.FreeTensor(gammaLocal);
    }

private:
    __aicore__ inline void CopyInputs()
    {
        LocalTensor<half> x1Local = x1Queue.AllocTensor<half>();
        LocalTensor<half> x2Local = x2Queue.AllocTensor<half>();
        LocalTensor<half> gammaLocal = gammaQueue.AllocTensor<half>();
        DataCopy(x1Local, x1Gm, kHiddenSize);
        DataCopy(x2Local, x2Gm, kHiddenSize);
        DataCopy(gammaLocal, gammaGm, kHiddenSize);
        x1Queue.EnQue(x1Local);
        x2Queue.EnQue(x2Local);
        gammaQueue.EnQue(gammaLocal);
    }

    __aicore__ inline void ReduceSum(
        const LocalTensor<float>& dst,
        const LocalTensor<float>& src,
        const LocalTensor<float>& work)
    {
        BinaryRepeatParams params;
        params.dstBlkStride = 1;
        params.src0BlkStride = 1;
        params.src1BlkStride = 1;
        params.dstRepStride = 0;
        params.src0RepStride = 8;
        params.src1RepStride = 0;
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

    __aicore__ inline void CopyOutputs(const LocalTensor<float>& rstdLocal)
    {
        LocalTensor<half> xLocal = xQueue.DeQue<half>();
        LocalTensor<half> yLocal = yQueue.DeQue<half>();
        DataCopy(xGm, xLocal, kHiddenSize);
        DataCopy(yGm, yLocal, kHiddenSize);
        DataCopyParams rstdParams;
        rstdParams.blockCount = 1;
        rstdParams.blockLen = sizeof(float);
        rstdParams.srcStride = 0;
        rstdParams.dstStride = 0;
        DataCopyPad(rstdGm, rstdLocal, rstdParams);
        xQueue.FreeTensor(xLocal);
        yQueue.FreeTensor(yLocal);
    }

    GlobalTensor<half> x1Gm;
    GlobalTensor<half> x2Gm;
    GlobalTensor<half> gammaGm;
    GlobalTensor<half> yGm;
    GlobalTensor<float> rstdGm;
    GlobalTensor<half> xGm;

    TQue<TPosition::VECIN, 1> x1Queue;
    TQue<TPosition::VECIN, 1> x2Queue;
    TQue<TPosition::VECIN, 1> gammaQueue;
    TQue<TPosition::VECOUT, 1> yQueue;
    TQue<TPosition::VECOUT, 1> xQueue;
    TBuf<TPosition::VECCALC> xFp32Buffer;
    TBuf<TPosition::VECCALC> squareBuffer;
    TBuf<TPosition::VECCALC> workBuffer;
    TBuf<TPosition::VECCALC> rstdBuffer;
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
