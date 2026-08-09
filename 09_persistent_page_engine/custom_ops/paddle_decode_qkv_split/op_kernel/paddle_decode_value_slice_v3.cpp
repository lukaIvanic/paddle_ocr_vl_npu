#include "kernel_operator.h"
#include "paddle_decode_qkv_split_tiling.h"

using namespace AscendC;

namespace {
constexpr uint32_t kInputOffset = 2304;
constexpr uint32_t kElements = 256;

class PaddleDecodeValueSliceKernel {
public:
    __aicore__ inline void Init(GM_ADDR qkv, GM_ADDR output, TPipe* pipe)
    {
        __gm__ half* input = reinterpret_cast<__gm__ half*>(qkv);
        inputGm.SetGlobalBuffer(input + kInputOffset, kElements);
        outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(output), kElements);
        pipe->InitBuffer(inputQueue, 1, kElements * sizeof(half));
        pipe->InitBuffer(outputQueue, 1, kElements * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<half> inputLocal = inputQueue.AllocTensor<half>();
        DataCopy(inputLocal, inputGm, kElements);
        inputQueue.EnQue(inputLocal);
        inputLocal = inputQueue.DeQue<half>();

        LocalTensor<half> outputLocal = outputQueue.AllocTensor<half>();
        DataCopy(outputLocal, inputLocal, kElements);
        inputQueue.FreeTensor(inputLocal);
        outputQueue.EnQue(outputLocal);
        outputLocal = outputQueue.DeQue<half>();
        DataCopy(outputGm, outputLocal, kElements);
        outputQueue.FreeTensor(outputLocal);
    }

private:
    GlobalTensor<half> inputGm;
    GlobalTensor<half> outputGm;
    TQue<QuePosition::VECIN, 1> inputQueue;
    TQue<QuePosition::VECOUT, 1> outputQueue;
};
}

extern "C" __global__ __aicore__ void paddle_decode_value_slice_v3(
    GM_ADDR qkv,
    GM_ADDR output,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(PaddleDecodeQkvSplitTilingData);
    GET_TILING_DATA(tilingData, tiling);
    if (GetBlockIdx() != 0 || tilingData.queryElements != 2048 ||
        tilingData.keyValueElements != kElements) {
        return;
    }
    TPipe pipe;
    PaddleDecodeValueSliceKernel kernel;
    kernel.Init(qkv, output, &pipe);
    kernel.Process();
}
