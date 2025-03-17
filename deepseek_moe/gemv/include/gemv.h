#ifndef MOE_H
#define MOE_H

// #include <math.h>
#include "flex_runtime.h"
#include "flex_printf.h"
#include "flex_dma_pattern.h"
#include "flex_group_barrier.h"
#include "flex_libfp16.h"
#include "flex_redmule.h"

// use float16 as data type
#define DATA_SIZE_BYTES 2
// Parameters for GEMM
// TILE_WIDTH * num_cluster_x = BLOCK_WIDTH
#define BLOCK_WIDTH 256
#define TILE_WIDTH 64
#define OPAND_SIZE TILE_WIDTH * TILE_WIDTH * DATA_SIZE_BYTES
// Parameter for element-wise functions
#define ELEMENT_WISE_TILE_WIDTH 2

// 2.5 in float
fp16 route_scale = (fp16)0x4100;

void gemv(const uint32_t A, const uint32_t x, const uint32_t y, const uint32_t K, const uint32_t N, const uint32_t bias_addr);
void compute_gemv(uint32_t in_token_addr, uint16_t n_token, uint16_t dim, uint16_t inter_dim, uint16_t n_routed_experts, uint16_t n_shared_experts, uint16_t n_activated_experts, uint32_t gate_weights_addr, uint32_t expert_w1_weights_addr, uint32_t expert_w1_bias_addr, uint32_t expert_w2_weights_addr, uint32_t expert_w2_bias_addr, uint32_t expert_w3_weights_addr, uint32_t expert_w3_bias_addr, uint32_t actual_out_addr);


void gemv(const uint32_t A, const uint32_t x, const uint32_t y, const uint32_t K, const uint32_t N, const uint32_t bias_addr) {
    if (0 == N || 0 == K) {
        return;
    }
    flex_global_barrier_xy();
    uint32_t cluster_id = flex_get_cluster_id();
    uint32_t core_id = flex_get_core_id();

    uint32_t accumulator, local_A, local_x;
    accumulator = 0;
    local_A = OPAND_SIZE;
    local_x = local_A + OPAND_SIZE;

    int n_remaining, m_tile, n_tile, k_tile, bK;

    int gj = get_pos(cluster_id).y; 

    for (int j = 0; j < (N - 1) / BLOCK_WIDTH + 1; j++) {
        n_remaining = (j < N / BLOCK_WIDTH) ? BLOCK_WIDTH : (N % BLOCK_WIDTH);

        if (n_remaining <= gj * TILE_WIDTH) {
            continue; 
        }
        
        m_tile = 1;
        n_tile = fmin(TILE_WIDTH, n_remaining - gj * TILE_WIDTH);

        if (flex_is_dm_core()) {
            if (bias_addr == zomem(0)) {
                flex_dma_async_1d(local(accumulator), zomem(0), n_tile * DATA_SIZE_BYTES);
                flex_dma_async_wait_all();
            } else {
                flex_dma_async_1d(local(accumulator), bias_addr + (gj * TILE_WIDTH + j * BLOCK_WIDTH) * DATA_SIZE_BYTES, n_tile * DATA_SIZE_BYTES);
                flex_dma_async_wait_all();
            }
        }
        flex_intra_cluster_sync();

        if (flex_is_first_core()) {
            flex_redmule_config(m_tile, TILE_WIDTH, n_tile);
        }

        for (bK = 0; bK < K; bK += TILE_WIDTH) {
            k_tile = fmin(TILE_WIDTH, K - bK);

            if (flex_is_dm_core()) {
                flex_dma_async_1d(local(local_A), hbm_addr(A + ((K * bK) + gj * TILE_WIDTH + j * BLOCK_WIDTH) * DATA_SIZE_BYTES), k_tile * DATA_SIZE_BYTES);
                flex_dma_async_wait_all();
            }

            if (flex_is_dm_core()) {
                flex_dma_async_1d(local(local_x), hbm_addr(x + bK * DATA_SIZE_BYTES), k_tile * DATA_SIZE_BYTES);
                flex_dma_async_wait_all();
            }
            flex_intra_cluster_sync();

            if (flex_is_first_core()) {
                if (k_tile != TILE_WIDTH) {
                    flex_redmule_config(m_tile, k_tile, n_tile);
                }
                flex_redmule_trigger(local_A, local_x, accumulator, REDMULE_FP_16);
                flex_redmule_wait();
            }
        }
        flex_intra_cluster_sync();

        if (flex_is_dm_core()) {
            flex_dma_async_1d(hbm_addr(y + (gj * TILE_WIDTH + j * BLOCK_WIDTH) * DATA_SIZE_BYTES), local(accumulator), n_tile * DATA_SIZE_BYTES);
            flex_dma_async_wait_all();
        }
    }
    flex_global_barrier_xy();
}

void compute_gemv(uint32_t in_token_addr, uint16_t n_token, uint16_t dim, uint16_t inter_dim, uint16_t n_routed_experts, uint16_t n_shared_experts, uint16_t n_activated_experts, uint32_t gate_weights_addr, uint32_t expert_w1_weights_addr, uint32_t expert_w1_bias_addr, uint32_t expert_w2_weights_addr, uint32_t expert_w2_bias_addr, uint32_t expert_w3_weights_addr, uint32_t expert_w3_bias_addr, uint32_t actual_out_addr) {
    flex_global_barrier_xy();
    uint32_t top_k_weights_addr, top_k_indices_addr;
    uint32_t temp_token_0, temp_token_1;
    top_k_weights_addr = actual_out_addr + n_token * dim * DATA_SIZE_BYTES;
    top_k_indices_addr = top_k_weights_addr + n_token * n_activated_experts * DATA_SIZE_BYTES;
    temp_token_0 = top_k_indices_addr + n_token * n_activated_experts * DATA_SIZE_BYTES;
    temp_token_1 = temp_token_0 + n_token * dim * DATA_SIZE_BYTES;

    // Gate 
    gemv(in_token_addr, gate_weights_addr, temp_token_0, dim, n_routed_experts, zomem(0));
    
    // Routed experts
    // self.w2.forward(silu(self.w1.forward(x)) * self.w3.forward(x))
    int i = 0;
    uint16_t i_expert;
    fp16 w_expert;
    while (i < n_activated_experts) {
        i_expert = ((uint16_t *)hbm_addr(top_k_indices_addr))[i];
        w_expert = ((fp16 *)hbm_addr(top_k_weights_addr))[i];
        
        // w1.forward(x)
        gemv(in_token_addr, expert_w1_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES), temp_token_0, dim, inter_dim, hbm_addr(expert_w1_bias_addr + (inter_dim * i_expert * DATA_SIZE_BYTES)));
        
        // w3.forward(x)
        gemv(in_token_addr, expert_w3_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES), temp_token_1, dim, inter_dim, hbm_addr(expert_w3_bias_addr + (inter_dim * i_expert * DATA_SIZE_BYTES)));
        
        // silu(w1.forward(x)) * w3.forward(x)
        // dot_product(temp_token_0, temp_token_1, temp_token_0, inter_dim, n_token);
        
        // w2.forward(silu(w1.forward(x)) * w3.forward(x))
        gemv(temp_token_0, expert_w2_weights_addr + (inter_dim * dim * i_expert * DATA_SIZE_BYTES), temp_token_0, inter_dim, dim, hbm_addr(expert_w2_bias_addr + (dim * i_expert * DATA_SIZE_BYTES)));

        i++;
    }
        
    // Shared experts
    // self.w2.forward(silu(self.w1.forward(x)) * self.w3.forward(x))
    for (int i_expert = n_routed_experts; i_expert < (n_routed_experts + n_shared_experts); i_expert++) {
        gemv(in_token_addr, expert_w1_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES), temp_token_0, dim, inter_dim, hbm_addr(expert_w1_bias_addr + (inter_dim * i_expert * DATA_SIZE_BYTES)));
        gemv(in_token_addr, expert_w3_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES), temp_token_1, dim, inter_dim, hbm_addr(expert_w3_bias_addr + (inter_dim * i_expert * DATA_SIZE_BYTES)));

        gemv(temp_token_0, expert_w2_weights_addr + (inter_dim * dim * i_expert * DATA_SIZE_BYTES), temp_token_0, inter_dim, dim, hbm_addr(expert_w2_bias_addr + (dim * i_expert * DATA_SIZE_BYTES)));
    }
    flex_global_barrier_xy();
}

#endif