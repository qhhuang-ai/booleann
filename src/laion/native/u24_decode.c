#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>

static _Atomic int active_calls = 0;
static _Atomic int maximum_active_calls = 0;

static void observe_concurrency_enter(void) {
    int current = atomic_fetch_add_explicit(
        &active_calls, 1, memory_order_relaxed
    ) + 1;
    int observed = atomic_load_explicit(
        &maximum_active_calls, memory_order_relaxed
    );
    while (
        current > observed
        && !atomic_compare_exchange_weak_explicit(
            &maximum_active_calls,
            &observed,
            current,
            memory_order_relaxed,
            memory_order_relaxed
        )
    ) {
    }
}

static void observe_concurrency_leave(void) {
    atomic_fetch_sub_explicit(&active_calls, 1, memory_order_relaxed);
}

int u24le_decode_v1(
    const uint8_t *src,
    size_t src_bytes,
    int32_t *dst,
    size_t dst_capacity,
    size_t *decoded_count
) {
    if (
        src == NULL || dst == NULL || decoded_count == NULL
        || src_bytes % 3u != 0u
    ) {
        return 1;
    }
    const size_t count = src_bytes / 3u;
    if (count > dst_capacity) {
        return 2;
    }
    for (size_t index = 0; index < count; ++index) {
        const size_t offset = 3u * index;
        dst[index] = (int32_t)(
            ((uint32_t)src[offset])
            | ((uint32_t)src[offset + 1u] << 8u)
            | ((uint32_t)src[offset + 2u] << 16u)
        );
    }
    *decoded_count = count;
    return 0;
}

int u24le_decode_repeat_v1(
    const uint8_t *src,
    size_t src_bytes,
    int32_t *dst,
    size_t dst_capacity,
    unsigned repetitions,
    uint64_t *checksum
) {
    if (checksum == NULL || repetitions == 0u) {
        return 3;
    }
    observe_concurrency_enter();
    uint64_t total = 0u;
    for (unsigned repetition = 0; repetition < repetitions; ++repetition) {
        if (
            src == NULL || dst == NULL || src_bytes % 3u != 0u
            || src_bytes / 3u > dst_capacity
        ) {
            observe_concurrency_leave();
            return 1;
        }
        const size_t count = src_bytes / 3u;
        for (size_t index = 0; index < count; ++index) {
            const size_t offset = 3u * index;
            const int32_t value = (int32_t)(
                ((uint32_t)src[offset])
                | ((uint32_t)src[offset + 1u] << 8u)
                | ((uint32_t)src[offset + 2u] << 16u)
            );
            dst[index] = value;
            total += (uint32_t)value + (uint64_t)repetition;
        }
    }
    observe_concurrency_leave();
    *checksum = total;
    return 0;
}

void u24le_reset_concurrency_v1(void) {
    atomic_store_explicit(&active_calls, 0, memory_order_relaxed);
    atomic_store_explicit(&maximum_active_calls, 0, memory_order_relaxed);
}

int u24le_max_concurrency_v1(void) {
    return atomic_load_explicit(
        &maximum_active_calls, memory_order_relaxed
    );
}
