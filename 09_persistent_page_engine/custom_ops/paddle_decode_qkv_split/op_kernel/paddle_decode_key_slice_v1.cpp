#include "kernel_operator.h"
#include "paddle_decode_qkv_split_tiling.h"

using namespace AscendC;

namespace {
constexpr uint32_t kInputOffset = 2048;
constexpr uint32_t kElements = 256;

class PaddleDecodeKeySliceKernel {
public:
    __aicore__ inline void Init(GM_ADDR qkv, GM_ADDR output, TPipe* pipe)
    {
        __gm__ half* input = reinterpret_cast<__gm__ half*>(qkv);
        inputGm.SetGlobalBuffer(input + kInputOffset, kElements);
        outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(output), kElements);
        pipe->InitBuffer(copyQueue, 1, kElements * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<half> local = copyQueue.AllocTensor<half>();
        DataCopy(local, inputGm, kElements);
        copyQueue.EnQue(local);
        local = copyQueue.DeQue<half>();
        DataCopy(outputGm, local, kElements);
        copyQueue.FreeTensor(local);
    }

private:
    GlobalTensor<half> inputGm;
    GlobalTensor<half> outputGm;
    TQue<QuePosition::VECIN, 1> copyQueue;
};
}

extern "C" __global__ __aicore__ void paddle_decode_key_slice_v1(
    GM_ADDR qkv,
    GM_ADDR output,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(PaddleDecodeQkvSplitTilingData);
    TPipe pipe;
    PaddleDecodeKeySliceKernel kernel;
    kernel.Init(qkv, output, &pipe);
    kernel.Process();
}
