// The filename follows CANN's PaddleDecodeSwiGluV1 -> paddle_decode_swi_glu_v1 mapping.
#include "kernel_operator.h"
#include "paddle_decode_swiglu_tiling.h"

using namespace AscendC;

namespace {
constexpr uint32_t kElements = 3072;
constexpr uint32_t kUbBytes = 4 * kElements * sizeof(float);

class PaddleDecodeSwiGluKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR gate,
        GM_ADDR up,
        GM_ADDR output,
        TPipe* pipe)
    {
        gateGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(gate), kElements);
        upGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(up), kElements);
        outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(output), kElements);
        pipe->InitBuffer(unitBuffer, kUbBytes);
    }

    __aicore__ inline void Process()
    {
        LocalTensor<float> ub = unitBuffer.Get<float>();
        LocalTensor<half> ubHalf = ub.ReinterpretCast<half>();
        LocalTensor<half> gateHalf = ubHalf;
        LocalTensor<half> upHalf = ubHalf[kElements];
        LocalTensor<float> gateFloat = ub[kElements];
        LocalTensor<float> upFloat = ub[2 * kElements];
        LocalTensor<float> work = ub[3 * kElements];

        DataCopy(gateHalf, gateGm, kElements);
        event_t gateReady = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(gateReady);
        DataCopy(upHalf, upGm, kElements);
        event_t upReady = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(upReady);
        WaitFlag<HardEvent::MTE2_V>(gateReady);
        WaitFlag<HardEvent::MTE2_V>(upReady);

        Cast(gateFloat, gateHalf, RoundMode::CAST_NONE, kElements);
        Cast(upFloat, upHalf, RoundMode::CAST_NONE, kElements);
        PipeBarrier<PIPE_V>();
        Muls(work, gateFloat, -1.0f, kElements);
        PipeBarrier<PIPE_V>();
        Exp(work, work, kElements);
        PipeBarrier<PIPE_V>();
        Adds(work, work, 1.0f, kElements);
        PipeBarrier<PIPE_V>();
        Div(gateFloat, gateFloat, work, kElements);
        PipeBarrier<PIPE_V>();
        Mul(gateFloat, gateFloat, upFloat, kElements);
        PipeBarrier<PIPE_V>();
        Cast(gateHalf, gateFloat, RoundMode::CAST_NONE, kElements);
        PipeBarrier<PIPE_V>();

        event_t outputReady = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(outputReady);
        WaitFlag<HardEvent::V_MTE3>(outputReady);
        DataCopy(outputGm, gateHalf, kElements);
    }

private:
    GlobalTensor<half> gateGm;
    GlobalTensor<half> upGm;
    GlobalTensor<half> outputGm;
    TBuf<TPosition::VECCALC> unitBuffer;
};
}

extern "C" __global__ __aicore__ void paddle_decode_swi_glu_v1(
    GM_ADDR gate,
    GM_ADDR up,
    GM_ADDR output,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(PaddleDecodeSwiGluTilingData);
    GET_TILING_DATA(tilingData, tiling);
    (void)workspace;
    if (GetBlockIdx() != 0 || tilingData.elements != kElements) {
        return;
    }
    TPipe pipe;
    PaddleDecodeSwiGluKernel kernel;
    kernel.Init(gate, up, output, &pipe);
    kernel.Process();
    // SuperKernel compilation removes TPipe::Destroy()'s final barrier.  The
    // down projection must not consume output while its MTE3 copy is active.
    PipeBarrier<PIPE_ALL>();
    pipe.Destroy();
}
