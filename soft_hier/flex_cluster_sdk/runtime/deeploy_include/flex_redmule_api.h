// flex_redmule_api.h
#ifndef FLEX_REDMULE_API_H
#define FLEX_REDMULE_API_H

#include <stdint.h>
#include "flex_cluster_arch.h"
#include "flex_runtime_api.h"

typedef enum {
    REDMULE_NONE_16,
    REDMULE_UINT_16,
    REDMULE_INT_16,
    REDMULE_FP_16,
    REDMULE_UINT_8,
    REDMULE_INT_8,
    REDMULE_FP_8
} redmule_compute_format_t;

// Redmule function declarations
void flex_redmule_config(uint16_t m_size, uint16_t n_size, uint16_t k_size);
void flex_redmule_trigger(uint32_t x_addr, uint32_t w_addr, uint32_t y_addr, redmule_compute_format_t format);
uint32_t flex_redmule_wait();

#endif
