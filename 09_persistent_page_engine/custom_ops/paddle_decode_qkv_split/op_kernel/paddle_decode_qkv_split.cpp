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
        pipe->InitBuffer(copyQueue, 1, kQueryElements * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        {
            LocalTensor<half> local = copyQueue.AllocTensor<half>();
            DataCopy(local, qkvGm, kQueryElements);
            copyQueue.EnQue(local);
            local = copyQueue.DeQue<half>();
            DataCopy(queryGm, local, kQueryElements);
            PipeBarrier<PIPE_MTE3>();
            copyQueue.FreeTensor(local);
        }
        {
            LocalTensor<half> local = copyQueue.AllocTensor<half>();
            DataCopy(local, qkvGm[kQueryElements], kKeyValueElements);
            copyQueue.EnQue(local);
            local = copyQueue.DeQue<half>();
            DataCopy(keyGm, local, kKeyValueElements);
            PipeBarrier<PIPE_MTE3>();
            copyQueue.FreeTensor(local);
        }
        {
            LocalTensor<half> local = copyQueue.AllocTensor<half>();
            DataCopy(
                local,
                qkvGm[kQueryElements + kKeyValueElements],
                kKeyValueElements);
            copyQueue.EnQue(local);
            local = copyQueue.DeQue<half>();
            DataCopy(valueGm, local, kKeyValueElements);
            PipeBarrier<PIPE_MTE3>();
            copyQueue.FreeTensor(local);
        }
    }

private:
    GlobalTensor<half> qkvGm;
    GlobalTensor<half> queryGm;
    GlobalTensor<half> keyGm;
    GlobalTensor<half> valueGm;
    TQue<QuePosition::VECIN, 1> copyQueue;
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
