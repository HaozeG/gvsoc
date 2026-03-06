// flex_dma_api.h
#ifndef FLEX_DMA_API_H
#define FLEX_DMA_API_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    COLLECTIVE_REDADD_UINT_16,
    COLLECTIVE_REDADD_INT_16,
    COLLECTIVE_REDADD_FP_16,
    COLLECTIVE_REDMAX_UINT_16,
    COLLECTIVE_REDMAX_INT_16,
    COLLECTIVE_REDMAX_FP_16
} collective_compute_format_t;

void flex_dma_async_1d(uint64_t dst_addr, uint64_t src_addr, size_t transfer_size);
void flex_dma_async_broadcast(uint64_t dst_offset, uint64_t src_offset, size_t transfer_size,
                              uint16_t row_mask, uint16_t col_mask);
void flex_dma_async_reduction(uint64_t dst_offset, uint64_t src_offset, size_t transfer_size,
                              collective_compute_format_t fmt, uint16_t row_mask,
                              uint16_t col_mask);
void flex_dma_async_wait_all(void);

void flex_dma_sync_2d(uint64_t dst_addr, uint64_t src_addr, size_t transfer_size,
                      size_t dst_stride, size_t src_stride, size_t repeat);

void flex_dma_async_Load_HBM_1d(uint32_t local_offset, uint32_t hbm_offset, size_t transfer_size);
void flex_dma_async_Store_HBM_1d(uint32_t local_offset, uint32_t hbm_offset, size_t transfer_size);

void flex_dma_async_pattern_round_shift_right(uint32_t local_offset, uint32_t remote_offset,
                                              size_t transfer_size);
void flex_dma_async_pattern_round_shift_left(uint32_t local_offset, uint32_t remote_offset,
                                             size_t transfer_size);
void flex_dma_async_pattern_round_shift_up(uint32_t local_offset, uint32_t remote_offset,
                                           size_t transfer_size);
void flex_dma_async_pattern_all_to_one(uint32_t local_offset, uint32_t remote_offset,
                                       size_t transfer_size);
void flex_dma_async_pattern_dialog_to_dialog(uint32_t local_offset, uint32_t remote_offset,
                                             size_t transfer_size);
void flex_dma_async_pattern_access_west_hbm(uint32_t local_offset, uint32_t remote_offset,
                                            size_t transfer_size);
void flex_dma_async_pattern_access_south_hbm(uint32_t local_offset, uint32_t remote_offset,
                                             size_t transfer_size);

void flex_dma_pattern_round_shift_right(uint32_t local_offset, uint32_t remote_offset,
                                        size_t transfer_size);
void flex_dma_pattern_round_shift_left(uint32_t local_offset, uint32_t remote_offset,
                                       size_t transfer_size);
void flex_dma_pattern_round_shift_up(uint32_t local_offset, uint32_t remote_offset,
                                     size_t transfer_size);
void flex_dma_pattern_all_to_one(uint32_t local_offset, uint32_t remote_offset,
                                 size_t transfer_size);
void flex_dma_pattern_dialog_to_dialog(uint32_t local_offset, uint32_t remote_offset,
                                       size_t transfer_size);
void flex_dma_pattern_access_west_hbm(uint32_t local_offset, uint32_t remote_offset,
                                      size_t transfer_size);
void flex_dma_pattern_systolic_shift_west_south(uint32_t local_offset,
                                                uint32_t remote_offset,
                                                size_t transfer_size);

#endif
