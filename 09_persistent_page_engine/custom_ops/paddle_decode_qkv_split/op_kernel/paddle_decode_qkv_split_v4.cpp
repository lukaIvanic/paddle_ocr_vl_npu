#include "kernel_operator.h"
#include "paddle_decode_qkv_split_tiling.h"

using namespace AscendC;

namespace {
constexpr uint32_t kQueryElements = 2048;
constexpr uint32_t kKeyValueElements = 256;

class PaddleDecodeQkvSplitKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR qkv,
        GM_ADDR query,
        GM_ADDR key,
        GM_ADDR value,
        TPipe* pipe)
    {
        __gm__ half* qkvBase = reinterpret_cast<__gm__ half*>(qkv);
        querySourceGm.SetGlobalBuffer(qkvBase, kQueryElements);
        keySourceGm.SetGlobalBuffer(
            qkvBase + kQueryElements, kKeyValueElements);
        valueSourceGm.SetGlobalBuffer(
            qkvBase + kQueryElements + kKeyValueElements,
            kKeyValueElements);
        queryGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ half*>(query), kQueryElements);
        keyGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ half*>(key), kKeyValueElements);
        valueGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ half*>(value), kKeyValueElements);
        pipe->InitBuffer(queryInputQueue, 1, kQueryElements * sizeof(half));
        pipe->InitBuffer(queryOutputQueue, 1, kQueryElements * sizeof(half));
        pipe->InitBuffer(keyInputQueue, 1, kKeyValueElements * sizeof(half));
        pipe->InitBuffer(keyOutputQueue, 1, kKeyValueElements * sizeof(half));
        pipe->InitBuffer(valueInputQueue, 1, kKeyValueElements * sizeof(half));
        pipe->InitBuffer(valueOutputQueue, 1, kKeyValueElements * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<half> queryInput = queryInputQueue.AllocTensor<half>();
        LocalTensor<half> keyInput = keyInputQueue.AllocTensor<half>();
        LocalTensor<half> valueInput = valueInputQueue.AllocTensor<half>();
        DataCopy(queryInput, querySourceGm, kQueryElements);
        DataCopy(keyInput, keySourceGm, kKeyValueElements);
        DataCopy(valueInput, valueSourceGm, kKeyValueElements);
        queryInputQueue.EnQue(queryInput);
        keyInputQueue.EnQue(keyInput);
        valueInputQueue.EnQue(valueInput);

        queryInput = queryInputQueue.DeQue<half>();
        keyInput = keyInputQueue.DeQue<half>();
        valueInput = valueInputQueue.DeQue<half>();
        LocalTensor<half> queryOutput = queryOutputQueue.AllocTensor<half>();
        LocalTensor<half> keyOutput = keyOutputQueue.AllocTensor<half>();
        LocalTensor<half> valueOutput = valueOutputQueue.AllocTensor<half>();
        Adds(queryOutput, queryInput, static_cast<half>(0.0f), kQueryElements);
        Adds(keyOutput, keyInput, static_cast<half>(0.0f), kKeyValueElements);
        Adds(valueOutput, valueInput, static_cast<half>(0.0f), kKeyValueElements);
        queryOutputQueue.EnQue(queryOutput);
        keyOutputQueue.EnQue(keyOutput);
        valueOutputQueue.EnQue(valueOutput);
        queryInputQueue.FreeTensor(queryInput);
        keyInputQueue.FreeTensor(keyInput);
        valueInputQueue.FreeTensor(valueInput);

        queryOutput = queryOutputQueue.DeQue<half>();
        keyOutput = keyOutputQueue.DeQue<half>();
        valueOutput = valueOutputQueue.DeQue<half>();
        DataCopy(queryGm, queryOutput, kQueryElements);
        DataCopy(keyGm, keyOutput, kKeyValueElements);
        DataCopy(valueGm, valueOutput, kKeyValueElements);
        queryOutputQueue.FreeTensor(queryOutput);
        keyOutputQueue.FreeTensor(keyOutput);
        valueOutputQueue.FreeTensor(valueOutput);
    }

private:
    GlobalTensor<half> querySourceGm;
    GlobalTensor<half> keySourceGm;
    GlobalTensor<half> valueSourceGm;
    GlobalTensor<half> queryGm;
    GlobalTensor<half> keyGm;
    GlobalTensor<half> valueGm;
    TQue<QuePosition::VECIN, 1> queryInputQueue;
    TQue<QuePosition::VECOUT, 1> queryOutputQueue;
    TQue<QuePosition::VECIN, 1> keyInputQueue;
    TQue<QuePosition::VECOUT, 1> keyOutputQueue;
    TQue<QuePosition::VECIN, 1> valueInputQueue;
    TQue<QuePosition::VECOUT, 1> valueOutputQueue;
};
}

extern "C" __global__ __aicore__ void paddle_decode_qkv_split_v4(
    GM_ADDR qkv,
    GM_ADDR query,
    GM_ADDR key,
    GM_ADDR value,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(PaddleDecodeQkvSplitTilingData);
    GET_TILING_DATA(tilingData, tiling);
    if (GetBlockIdx() != 0 || tilingData.queryElements != kQueryElements ||
        tilingData.keyValueElements != kKeyValueElements) {
        return;
    }
    TPipe pipe;
    PaddleDecodeQkvSplitKernel kernel;
    kernel.Init(qkv, query, key, value, &pipe);
    kernel.Process();
    // TPipe::Destroy() deliberately omits its final PIPE_ALL when CANN
    // recompiles this function into a SuperKernel. Complete the three
    // UB-to-GM outputs before the next fused subfunction reuses the global
    // vector-pipe state or reads Q/K/V on another worker.
    PipeBarrier<PIPE_ALL>();
    pipe.Destroy();
}
