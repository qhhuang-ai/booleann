#include <stddef.h>
#include <stdint.h>

/* first value: little-endian u32; remaining values: positive u16 deltas */
int delta16le_decode_v1(
    const uint8_t *src,
    size_t src_bytes,
    int32_t *dst,
    size_t dst_capacity,
    size_t *decoded_count
) {
    if (
        src == NULL || dst == NULL || decoded_count == NULL
        || src_bytes < 4u || (src_bytes - 4u) % 2u != 0u
    ) {
        return 1;
    }
    const size_t count = 1u + (src_bytes - 4u) / 2u;
    if (count > dst_capacity) {
        return 2;
    }
    uint32_t value =
        ((uint32_t)src[0])
        | ((uint32_t)src[1] << 8u)
        | ((uint32_t)src[2] << 16u)
        | ((uint32_t)src[3] << 24u);
    if (value > 999999u) {
        return 3;
    }
    dst[0] = (int32_t)value;
    for (size_t index = 1; index < count; ++index) {
        const size_t offset = 4u + 2u * (index - 1u);
        const uint32_t delta =
            ((uint32_t)src[offset])
            | ((uint32_t)src[offset + 1u] << 8u);
        if (delta == 0u || value + delta > 999999u) {
            return 4;
        }
        value += delta;
        dst[index] = (int32_t)value;
    }
    *decoded_count = count;
    return 0;
}
