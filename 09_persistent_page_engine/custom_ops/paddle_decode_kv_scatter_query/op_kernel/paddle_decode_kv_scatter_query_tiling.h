#ifndef PADDLE_DECODE_KV_SCATTER_QUERY_TILING_H
#define PADDLE_DECODE_KV_SCATTER_QUERY_TILING_H

#include <cstdint>

struct PaddleDecodeKvScatterQueryTilingData {
    uint32_t cacheLength;
    uint32_t queryElements;
    uint32_t stateElements;
};

#endif
