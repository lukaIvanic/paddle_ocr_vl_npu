/*
 * Minimal Python-ctypes bridge to CANN's public MSTX header implementation.
 *
 * The public header resolves the msprof-provided injection library through
 * MSTX_INJECTION_PATH on first use. No private CANN symbols are linked here.
 */
#include <stdint.h>

#include <mstx/ms_tools_ext.h>

#if defined(__GNUC__)
#define VISION_MSTX_EXPORT __attribute__((visibility("default")))
#else
#define VISION_MSTX_EXPORT
#endif

VISION_MSTX_EXPORT uint64_t vision_mstx_range_start(
    const char *message,
    void *stream)
{
    return (uint64_t)mstxRangeStartA(message, (aclrtStream)stream);
}

VISION_MSTX_EXPORT void vision_mstx_range_end(uint64_t range_id)
{
    mstxRangeEnd((mstxRangeId)range_id);
}
