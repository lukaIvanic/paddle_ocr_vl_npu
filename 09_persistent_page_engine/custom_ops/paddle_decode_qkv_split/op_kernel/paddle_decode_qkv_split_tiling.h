#ifndef PADDLE_DECODE_QKV_SPLIT_TILING_H
#define PADDLE_DECODE_QKV_SPLIT_TILING_H

#include <cstdint>

struct PaddleDecodeQkvSplitTilingData {
    uint32_t queryElements;
    uint32_t keyValueElements;
};

#endif
