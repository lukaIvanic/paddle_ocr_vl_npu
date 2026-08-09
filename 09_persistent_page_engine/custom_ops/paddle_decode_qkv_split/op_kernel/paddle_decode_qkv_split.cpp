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
        pipe->InitBuffer(copyBuffer, kQueryElements * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<half> local = copyBuffer.Get<half>();

        DataCopy(local, qkvGm, kQueryElements);
        const event_t queryInputReady = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
        SetFlag<HardEvent::MTE2_MTE3>(queryInputReady);
        WaitFlag<HardEvent::MTE2_MTE3>(queryInputReady);
        DataCopy(queryGm, local, kQueryElements);
        const event_t queryStored = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(queryStored);
        WaitFlag<HardEvent::MTE3_MTE2>(queryStored);

        DataCopy(local, qkvGm[kQueryElements], kKeyValueElements);
        const event_t keyInputReady = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
        SetFlag<HardEvent::MTE2_MTE3>(keyInputReady);
        WaitFlag<HardEvent::MTE2_MTE3>(keyInputReady);
        DataCopy(keyGm, local, kKeyValueElements);
        const event_t keyStored = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(keyStored);
        WaitFlag<HardEvent::MTE3_MTE2>(keyStored);

        DataCopy(
            local,
            qkvGm[kQueryElements + kKeyValueElements],
            kKeyValueElements);
        const event_t valueInputReady = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
        SetFlag<HardEvent::MTE2_MTE3>(valueInputReady);
        WaitFlag<HardEvent::MTE2_MTE3>(valueInputReady);
        DataCopy(valueGm, local, kKeyValueElements);
        const event_t valueStored = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(valueStored);
        WaitFlag<HardEvent::MTE3_MTE2>(valueStored);
    }

private:
    GlobalTensor<half> qkvGm;
    GlobalTensor<half> queryGm;
    GlobalTensor<half> keyGm;
    GlobalTensor<half> valueGm;
    TBuf<TPosition::VECCALC> copyBuffer;
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
