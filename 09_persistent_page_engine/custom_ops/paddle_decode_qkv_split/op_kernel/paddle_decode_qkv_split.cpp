#include "kernel_operator.h"
#include "paddle_decode_qkv_split_tiling.h"

using namespace AscendC;

namespace {
constexpr uint32_t kQueryElements = 2048;
constexpr uint32_t kKeyValueElements = 256;
constexpr uint32_t kTotalElements = 2560;

class PaddleDecodeQkvSplitKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR qkv,
        GM_ADDR query,
        GM_ADDR key,
        GM_ADDR value,
        TPipe* pipe)
    {
        qkvGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ half*>(qkv), kTotalElements);
        queryGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ half*>(query), kQueryElements);
        keyGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ half*>(key), kKeyValueElements);
        valueGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ half*>(value), kKeyValueElements);
        pipe->InitBuffer(inputQueue, 1, kQueryElements * sizeof(half));
        pipe->InitBuffer(outputQueue, 1, kQueryElements * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        CopySegment(qkvGm, queryGm, kQueryElements);
        CopySegment(
            qkvGm[kQueryElements], keyGm, kKeyValueElements);
        CopySegment(
            qkvGm[kQueryElements + kKeyValueElements],
            valueGm,
            kKeyValueElements);
    }

private:
    __aicore__ inline void CopySegment(
        const GlobalTensor<half>& source,
        const GlobalTensor<half>& destination,
        uint32_t elements)
    {
        LocalTensor<half> input = inputQueue.AllocTensor<half>();
        DataCopy(input, source, elements);
        inputQueue.EnQue(input);
        input = inputQueue.DeQue<half>();

        LocalTensor<half> output = outputQueue.AllocTensor<half>();
        Adds(output, input, static_cast<half>(0.0f), elements);
        outputQueue.EnQue(output);
        inputQueue.FreeTensor(input);

        output = outputQueue.DeQue<half>();
        DataCopy(destination, output, elements);
        outputQueue.FreeTensor(output);
    }

    GlobalTensor<half> qkvGm;
    GlobalTensor<half> queryGm;
    GlobalTensor<half> keyGm;
    GlobalTensor<half> valueGm;
    TQue<QuePosition::VECIN, 1> inputQueue;
    TQue<QuePosition::VECOUT, 1> outputQueue;
};
}

extern "C" __global__ __aicore__ void paddle_decode_qkv_split(
    GM_ADDR qkv,
    GM_ADDR query,
    GM_ADDR key,
    GM_ADDR value,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(PaddleDecodeQkvSplitTilingData);
    TPipe pipe;
    PaddleDecodeQkvSplitKernel kernel;
    kernel.Init(qkv, query, key, value, &pipe);
    kernel.Process();
}
