#ifndef MATMUL_H
#define MATMUL_H

// #include <math.h>
#include "flex_runtime.h"
#include "flex_printf.h"
#include "flex_dma_pattern.h"
#include "flex_group_barrier.h"
#include "flex_libfp16.h"

// use float16 as data type
#define DATA_SIZE_BYTES 2
// Parameters for GEMM
#define BLOCK_WIDTH 256
#define TILE_WIDTH 64
#define OPAND_SIZE TILE_WIDTH * TILE_WIDTH * DATA_SIZE_BYTES


void gemm(const uint32_t A, const uint32_t B, const uint32_t C, const uint32_t K, const uint32_t M, const uint32_t N, const uint32_t bias_addr);
void top_k(const uint32_t in_addr, const uint32_t out_addr, const uint32_t k, const uint32_t dim, const uint32_t n_token);
void compute_moe(uint32_t in_token_addr, uint16_t n_token, uint16_t dim, uint16_t inter_dim, uint16_t n_routed_experts, uint16_t n_shared_experts, uint16_t n_activated_experts, uint32_t gate_weights_addr, uint32_t expert_w1_weights_addr, uint32_t expert_w1_bias_addr, uint32_t expert_w2_weights_addr, uint32_t expert_w2_bias_addr, uint32_t expert_w3_weights_addr, uint32_t expert_w3_bias_addr, uint32_t actual_out_addr);

/**
 * @brief Select top k values from each row of the input matrix and write the result to the output matrix along with the corresponding indices.
 * 
 * @param in_addr 
 * @param out_addr 
 * @param k top k values to select (less than 64)
 * @param dim number of candidate values in each row of the input matrix
 * @param n_token number of rows in the input matrix
 */
void top_k(const uint32_t in_addr, const uint32_t out_addr, const uint32_t k, const uint32_t dim, const uint32_t n_token) {
    if (0 == k || 0 == dim || 0 == n_token || k > TILE_WIDTH) {
        return;
    }
    flex_global_barrier_xy();
    uint32_t local_out_value, local_out_indicies, local_in;
    local_out_value = 0;
    local_out_indicies = local_out_value + TILE_WIDTH * DATA_SIZE_BYTES;
    local_in = local_out_indicies + TILE_WIDTH * DATA_SIZE_BYTES;

    uint32_t cluster_id = flex_get_cluster_id();
    uint32_t core_id = flex_get_core_id();





    flex_global_barrier_xy();
}

/**
 * @brief GEMM operation A * B = C for input matrices A (MxK), B (KxN) and output matrix C (MxN). Assumes that K, M and N are multiples of TILE_WIDTH.
 * 
 * @param A offset of matrix A wrt. hbm_addr
 * @param B offset of matrix B wrt. hbm_addr
 * @param C offset of matrix C wrt. hbm_addr
 * @param K shared dimension of A and B
 * @param M 
 * @param N 
 * @param bias_addr address of bias matrix. shaped as [1, N], added to each row of the output matrix
 */
// TODO: For inferencing, the input matrix shape is [1, dim], which is not efficient enough for dividing output tasks to different clusters.
void gemm(const uint32_t A, const uint32_t B, const uint32_t C, const uint32_t K, const uint32_t M, const uint32_t N, const uint32_t bias_addr) {
    
    if (0 == M || 0 == N || 0 == K) {
        return;
    }
    flex_global_barrier_xy();
    uint32_t cluster_id = flex_get_cluster_id();
    uint32_t core_id = flex_get_core_id();

    uint32_t accumulator, local_A, local_B;
    accumulator = 0;
    local_A = OPAND_SIZE;
    local_B = local_A + OPAND_SIZE;

    int m_remaining, n_remaining, m_tile, n_tile, k_tile, bK;

    // each cluster compute a tile of the output matrix
    int gi = get_pos(cluster_id).x;
    int gj = get_pos(cluster_id).y; 

    // if (flex_is_first_core() && gi == 0 && gj == 0) {
    //     printf("[GEMM CONFIG] M: %d, K: %d, N: %d\n", M, K, N);
    // }
    // flex_intra_cluster_sync();

    for (int i = 0; i < (M - 1) / BLOCK_WIDTH + 1; i++) {
        for (int j = 0; j < (N - 1) / BLOCK_WIDTH + 1; j++) {
            m_remaining = (i < M / BLOCK_WIDTH) ? BLOCK_WIDTH : (M % BLOCK_WIDTH);
            n_remaining = (j < N / BLOCK_WIDTH) ? BLOCK_WIDTH : (N % BLOCK_WIDTH);
            
            // Skip clusters that aren't needed for this block
            if (m_remaining <= gi * TILE_WIDTH || n_remaining <= gj * TILE_WIDTH) {
                continue; 
            }
            
            m_tile = fmin(TILE_WIDTH, m_remaining - gi * TILE_WIDTH);
            n_tile = fmin(TILE_WIDTH, n_remaining - gj * TILE_WIDTH);
            
            if(flex_is_dm_core()) {
                if (bias_addr == zomem(0)) {
                    flex_dma_async_1d(local(accumulator), zomem(0), m_tile * n_tile * DATA_SIZE_BYTES);
                    flex_dma_async_wait_all();
                } else {
                    flex_dma_sync_2d(local(accumulator), bias_addr + (gj * TILE_WIDTH + j * BLOCK_WIDTH) * DATA_SIZE_BYTES, n_tile*DATA_SIZE_BYTES, n_tile*DATA_SIZE_BYTES, 0, m_tile);
                }
            }
            flex_intra_cluster_sync();
            if(flex_is_first_core()) {
                // printf("[REDMULE CONFIG] m: %d, n: %d, k: %d\n", m_tile, TILE_WIDTH, n_tile);
                // Redmule Config: [m_size, n_size] * [n_size, k_size] = [m_size, k_size]
                flex_redmule_config(m_tile, TILE_WIDTH, n_tile);
            }
            
            // bK: inner loop tile computing partial sums
            for (bK = 0; bK < K; bK += TILE_WIDTH) {
                k_tile = fmin(TILE_WIDTH, K - bK);

                // SoftHier_HBM -> SoftHier_TCDM 2D
                if(flex_is_dm_core()) {
                    flex_dma_sync_2d(local(local_A), hbm_addr(A + ((K * (TILE_WIDTH * gi + i * BLOCK_WIDTH)) + bK) * DATA_SIZE_BYTES), k_tile*DATA_SIZE_BYTES, k_tile*DATA_SIZE_BYTES, K*DATA_SIZE_BYTES, m_tile);
                }
                // flex_intra_cluster_sync();

                // SoftHier_HBM -> SoftHier_TCDM 2D
                if(flex_is_dm_core()) {
                    flex_dma_sync_2d(local(local_B), hbm_addr(B + (N * bK + TILE_WIDTH * gj + j * BLOCK_WIDTH) * DATA_SIZE_BYTES), n_tile*DATA_SIZE_BYTES,  n_tile*DATA_SIZE_BYTES, N*DATA_SIZE_BYTES, k_tile);
                }
                
                
                // make sure data is ready
                flex_intra_cluster_sync();

                // check operands
                // if(flex_is_first_core()) {
                //     printf("[OPERANDS]\n");
                //     printf("A:  \n");
                //     for (int i = 0; i < m_tile; i++) {
                //         for (int j = 0; j < k_tile; j++) {
                //             printf(" 0x%04x ", 
                //                     ((uint16_t *)local(local_A))[i * k_tile + j]);
                //         }
                //         printf("\n");
                //     }
                //     printf("B:  \n");
                //     for (int i = 0; i < k_tile; i++) {
                //         for (int j = 0; j < n_tile; j++) {
                //             printf(" 0x%04x ", 
                //                     ((uint16_t *)local(local_B))[i * n_tile + j]);
                //         }
                //         printf("\n");
                //     }
                // }

                if (flex_is_first_core()) {
                    // change configuration if the tile in K dimension is not full
                    if (k_tile != TILE_WIDTH) {
                        // printf("[REDMULE CONFIG] m: %d, n: %d, k: %d\n", m_tile, k_tile, n_tile);
                        // Redmule Config: [m_size, n_size] * [n_size, k_size] = [m_size, k_size]
                        flex_redmule_config(m_tile, k_tile, n_tile);
                    }

                    uint32_t _in_local_a = local_A;
                    uint32_t _in_local_b = local_B;
                    uint32_t _in_local_sum = accumulator;

                    ///////////////////
                    // multiply and accumulate
                    flex_redmule_trigger(_in_local_a, _in_local_b, _in_local_sum, REDMULE_FP_16);
                    flex_redmule_wait();
                    ///////////////////
                    // matmul_fp16((fp16 *)_in_local_sum, (fp16 *)_in_local_sum, (fp16 *)_in_local_a, (fp16 *)_in_local_b, m_tile, k_tile, n_tile);

                }
                flex_intra_cluster_sync();

                // if(flex_is_first_core()) {
                    // printf("[ACCUMULATOR]\n    C: 0x%x\n", ((uint16_t *)local(accumulator))[1]);
                // }

                // if (flex_is_dm_core())
                // {
                //     flex_dma_async_1d_reduction(local(accumulator), local(local_sum), m_tile * n_tile * DATA_SIZE_BYTES, COLLECTIVE_REDADD_FP_16);
                //     flex_dma_async_wait_all();
                // }

                // add local_sum to accumulator
                // for (uint32_t i = 0; i < m_tile * n_tile; i += ARCH_NUM_CORE_PER_CLUSTER) {
                //     if (i + core_id < m_tile * n_tile) {
                //         uint32_t local_sum_offset = (i + core_id) * DATA_SIZE_BYTES;
                //         uint32_t accumulator_offset = (i + core_id) * DATA_SIZE_BYTES;
                //         float16_t local_sum_val = ((float16_t *)local(local_sum_offset))[0];
                //         float16_t accumulator_val = ((float16_t *)local(accumulator_offset))[0];
                //     }
                // }

            }

            flex_intra_cluster_sync();
            // SoftHier_TCDM -> SoftHier_HBM
            if(flex_is_dm_core()) {
                flex_dma_sync_2d(hbm_addr(C + ((TILE_WIDTH * gi + i * BLOCK_WIDTH) * N + TILE_WIDTH * gj + j * BLOCK_WIDTH) * DATA_SIZE_BYTES), local(accumulator), n_tile*DATA_SIZE_BYTES, N*DATA_SIZE_BYTES, n_tile*DATA_SIZE_BYTES, m_tile);
            }
            // flex_intra_cluster_sync();
        }
        // flex_intra_cluster_sync();
    }
}

void compute_moe(uint32_t in_token_addr, uint16_t n_token, uint16_t dim, uint16_t inter_dim, uint16_t n_routed_experts, uint16_t n_shared_experts, uint16_t n_activated_experts, uint32_t gate_weights_addr, uint32_t expert_w1_weights_addr, uint32_t expert_w1_bias_addr, uint32_t expert_w2_weights_addr, uint32_t expert_w2_bias_addr, uint32_t expert_w3_weights_addr, uint32_t expert_w3_bias_addr, uint32_t actual_out_addr) {
    flex_global_barrier_xy();
    uint32_t cluster_id = flex_get_cluster_id();
    uint32_t core_id = flex_get_core_id();

    // Gate 
    gemm(in_token_addr, gate_weights_addr, actual_out_addr, dim, n_token, n_routed_experts, zomem(0));
    // matmul_fp16((fp16 *)hbm_addr(actual_out_addr), (fp16 *)hbm_addr(actual_out_addr), (fp16 *)hbm_addr(in_token_addr), (fp16 *)hbm_addr(gate_weights_addr), n_token, dim, n_routed_experts);

    // Routed experts
    // self.w2.forward(silu(self.w1.forward(x)) * self.w3.forward(x))
    for (int i_expert = 0; i_expert < (0); i_expert++) {
        gemm(in_token_addr, expert_w1_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES), actual_out_addr, dim, n_token, inter_dim, hbm_addr(expert_w1_bias_addr + (inter_dim * i_expert * DATA_SIZE_BYTES)));
        // asm_fp16_sigmoid((const fp16 *)hbm_addr(actual_out_addr), (const fp16 *)hbm_addr(actual_out_addr));
        gemm(in_token_addr, expert_w3_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES), actual_out_addr, dim, n_token, inter_dim, hbm_addr(expert_w3_bias_addr + (inter_dim * i_expert * DATA_SIZE_BYTES)));
        
        
        gemm(actual_out_addr, expert_w2_weights_addr + (inter_dim * dim * i_expert * DATA_SIZE_BYTES), actual_out_addr, inter_dim, n_token, dim, hbm_addr(expert_w2_bias_addr + (dim * i_expert * DATA_SIZE_BYTES)));
    }
        
    /**
    // Shared experts
    // self.w2.forward(silu(self.w1.forward(x)) * self.w3.forward(x))
    for (int i_expert = n_routed_experts; i_expert < (n_routed_experts + n_shared_experts); i_expert++) {
        gemm(in_token_addr, expert_w1_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES), actual_out_addr, dim, n_token, inter_dim, hbm_addr(expert_w1_bias_addr + (inter_dim * i_expert * DATA_SIZE_BYTES)));
        // silu_approx(actual_out_addr, n_token, inter_dim);
        gemm(in_token_addr, expert_w3_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES), actual_out_addr, dim, n_token, inter_dim, hbm_addr(expert_w3_bias_addr + (inter_dim * i_expert * DATA_SIZE_BYTES)));

        gemm(actual_out_addr, expert_w2_weights_addr + (inter_dim * dim * i_expert * DATA_SIZE_BYTES), actual_out_addr, inter_dim, n_token, dim, hbm_addr(expert_w2_bias_addr + (dim * i_expert * DATA_SIZE_BYTES)));
    }
    **/
    flex_global_barrier_xy();
}

#endif