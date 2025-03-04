#ifndef MATMUL_H
#define MATMUL_H

// #include <math.h>
#include "flex_runtime.h"
#include "flex_printf.h"
#include "flex_redmule.h"
#include "flex_dma_pattern.h"
#include "flex_group_barrier.h"

// use float16 as data type
#define DATA_BYTE_SIZE 2
// Parameters for GEMM
#define BLOCK_WIDTH 256
#define TILE_WIDTH 64
#define OPAND_SIZE TILE_WIDTH * TILE_WIDTH * DATA_BYTE_SIZE


void gemm(const uint32_t A, const uint32_t B, const uint32_t C, const uint32_t K, const uint32_t M, const uint32_t N, const uint32_t bias_addr);
void silu_approx(uint32_t input, uint32_t M, uint32_t N);
void compute_moe(uint32_t in_token_addr, uint16_t n_token, uint16_t dim, uint16_t inter_dim, uint16_t n_routed_experts, uint16_t n_shared_experts, uint16_t n_activated_experts, uint32_t gate_weights_addr, uint32_t expert_w1_weights_addr, uint32_t expert_w1_bias_addr, uint32_t expert_w2_weights_addr, uint32_t expert_w2_bias_addr, uint32_t expert_w3_weights_addr, uint32_t expert_w3_bias_addr, uint32_t actual_out_addr);

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
void gemm(const uint32_t A, const uint32_t B, const uint32_t C, const uint32_t K, const uint32_t M, const uint32_t N, const uint32_t bias_addr) {
    
    flex_global_barrier_xy();
    uint32_t cluster_id = flex_get_cluster_id();
    uint32_t core_id = flex_get_core_id();
    if (0 == M | 0 == N | 0 == K) {
        return;
    }

    uint32_t accumulator;
    accumulator = 0;
    uint32_t local_A;
    local_A = OPAND_SIZE;
    uint32_t local_B;
    local_B = local_A + OPAND_SIZE;
    int k_tile = TILE_WIDTH;
    for (int i = 0; i < (M - 1) / BLOCK_WIDTH + 1; i++) {
        for (int j = 0; j < (N - 1) / BLOCK_WIDTH + 1; j++) {
            {
                // each cluster compute a tile of the output matrix
                int gi = get_pos(cluster_id).x;
                int gj = get_pos(cluster_id).y; 

                int m_remaining = (i < M / BLOCK_WIDTH) ? BLOCK_WIDTH : (M % BLOCK_WIDTH);
                int n_remaining = (j < N / BLOCK_WIDTH) ? BLOCK_WIDTH : (N % BLOCK_WIDTH);
                
                // Skip clusters that aren't needed for this block
                if (m_remaining <= gi * TILE_WIDTH || n_remaining <= gj * TILE_WIDTH) {
                    continue; 
                }
                
                int m_tile = fmin(TILE_WIDTH, m_remaining - gi * TILE_WIDTH);
                int n_tile = fmin(TILE_WIDTH, n_remaining - gj * TILE_WIDTH);
                
                if(flex_is_dm_core()) {
                    if (bias_addr == zomem(0)) {
                        flex_dma_async_1d(local(accumulator), zomem(0), m_tile * n_tile * DATA_BYTE_SIZE);
                        // flex_dma_async_wait_all();
                    } else {
                        flex_dma_sync_2d(local(accumulator), bias_addr + (gj * TILE_WIDTH + j * BLOCK_WIDTH) * DATA_BYTE_SIZE, n_tile*DATA_BYTE_SIZE, n_tile*DATA_BYTE_SIZE, 0, m_tile);
                    }
                }
                flex_intra_cluster_sync();
                if(flex_is_first_core()) {
                    // printf("[REDMULE CONFIG] m: %d, n: %d, k: %d\n", m_tile, TILE_WIDTH, n_tile);
                    // [M, N] * [N, K] = [M, K]
                    flex_redmule_config(m_tile, TILE_WIDTH, n_tile);
                }

                // bK: inner loop tile computing partial sums
                for (int bK = 0; bK < K; bK += TILE_WIDTH) {
                    k_tile = fmin(TILE_WIDTH, K - bK);

                    // SoftHier_HBM -> SoftHier_TCDM 2D
                    if(flex_is_dm_core()) {
                        flex_dma_sync_2d(local(local_A), hbm_addr(A + ((K * (TILE_WIDTH * gi + i * BLOCK_WIDTH)) + bK) * DATA_BYTE_SIZE), k_tile*DATA_BYTE_SIZE, k_tile*DATA_BYTE_SIZE, K*DATA_BYTE_SIZE, m_tile);
                    }
                    // flex_intra_cluster_sync();

                    // SoftHier_HBM -> SoftHier_TCDM 2D
                    if(flex_is_dm_core()) {
                        flex_dma_sync_2d(local(local_B), hbm_addr(B + (N * bK + TILE_WIDTH * gj + j * BLOCK_WIDTH) * DATA_BYTE_SIZE), n_tile*DATA_BYTE_SIZE,  n_tile*DATA_BYTE_SIZE, N*DATA_BYTE_SIZE, k_tile);
                    }
                    
                    // check operands
                    // if(flex_is_first_core()) {
                        //     printf("[OPERANDS]\n    A: 0x%x, B: 0x%x\n", 
                        //            ((uint16_t *)local(local_A))[0], 
                        //             ((uint16_t *)local(local_B))[0], 
                        // }
                        
                    // make sure data is ready
                    flex_intra_cluster_sync();

                    if (flex_is_first_core()) {
                        if (k_tile != TILE_WIDTH) {
                            // printf("[REDMULE CONFIG] m: %d, n: %d, k: %d\n", m_tile, k_tile, n_tile);
                            // [M, N] * [N, K] = [M, K]
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

                    }
                    flex_intra_cluster_sync();

                    // if (flex_is_dm_core())
                    // {
                    //     // TODO: to see how is reduction implemented
                    //     flex_dma_async_1d_reduction(local(accumulator), local(local_sum), m_tile * n_tile * DATA_BYTE_SIZE, COLLECTIVE_REDADD_FP_16);
                    //     flex_dma_async_wait_all();
                    // }

                    // add local_sum to accumulator
                    // for (uint32_t i = 0; i < m_tile * n_tile; i += ARCH_NUM_CORE_PER_CLUSTER) {
                    //     if (i + core_id < m_tile * n_tile) {
                    //         uint32_t local_sum_offset = (i + core_id) * DATA_BYTE_SIZE;
                    //         uint32_t accumulator_offset = (i + core_id) * DATA_BYTE_SIZE;
                    //         float16_t local_sum_val = ((float16_t *)local(local_sum_offset))[0];
                    //         float16_t accumulator_val = ((float16_t *)local(accumulator_offset))[0];
                    //     }
                    // }

                }

                // SoftHier_TCDM -> SoftHier_HBM
                if(flex_is_dm_core()) {
                    flex_dma_sync_2d(hbm_addr(C + ((TILE_WIDTH * gi + i * BLOCK_WIDTH) * N + TILE_WIDTH * gj + j * BLOCK_WIDTH) * DATA_BYTE_SIZE), local(accumulator), n_tile*DATA_BYTE_SIZE, N*DATA_BYTE_SIZE, n_tile*DATA_BYTE_SIZE, m_tile);
                }
                flex_intra_cluster_sync();
            }
        }
        flex_intra_cluster_sync();
    }
}

void compute_moe(uint32_t in_token_addr, uint16_t n_token, uint16_t dim, uint16_t inter_dim, uint16_t n_routed_experts, uint16_t n_shared_experts, uint16_t n_activated_experts, uint32_t gate_weights_addr, uint32_t expert_w1_weights_addr, uint32_t expert_w1_bias_addr, uint32_t expert_w2_weights_addr, uint32_t expert_w2_bias_addr, uint32_t expert_w3_weights_addr, uint32_t expert_w3_bias_addr, uint32_t actual_out_addr) {
    flex_global_barrier_xy();
    uint32_t cluster_id = flex_get_cluster_id();
    uint32_t core_id = flex_get_core_id();
    // compute gate weights as an example
    gemm(in_token_addr, gate_weights_addr, actual_out_addr, dim, n_token, n_routed_experts, zomem(0));
    flex_global_barrier_xy();
}

#endif