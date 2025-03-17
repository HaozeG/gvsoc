#ifndef MOE_DECODE_H
#define MOE_DECODE_H

#include "flex_runtime.h"
#include "flex_printf.h"
#include "flex_dma_pattern.h"
#include "flex_group_barrier.h"
#include "flex_libfp16.h"

// use float16 as data type
#define DATA_SIZE_BYTES 2
// Parameters for GEMM
// TILE_WIDTH * num_cluster_x = BLOCK_WIDTH
#define BLOCK_WIDTH 256
#define TILE_WIDTH 32
#define OPAND_SIZE TILE_WIDTH * TILE_WIDTH * DATA_SIZE_BYTES
// Parameter for element-wise functions
#define ELEMENT_WISE_TILE_WIDTH 2

// designed for NUM_CLUSTER = 16
typedef uint16_t cluster_map_t;
// 2.5 in float
fp16 route_scale = (fp16)0x4100;

void gemv(const uint32_t A, const uint32_t B, const uint32_t C, const uint32_t K, const uint32_t M, const uint32_t N, const uint32_t bias_addr, cluster_map_t cluster_map);

/**
 * @brief GEMV operation A * B = C for input vector A (1xK), matrix B (KxN) and output matrix C (1xN).
 * 
 * @param A address of vector A
 * @param B address of matrix B
 * @param C address of vector C
 * @param K shared dimension of A and B
 * @param M == 1
 * @param N 
 * @param bias_addr address of bias matrix. shaped as [1, N], added to each row of the output 
 * @param cluster_map activated cluster map for computation 
 */
void gemv(const uint32_t A, const uint32_t B, const uint32_t C, const uint32_t K, const uint32_t M, const uint32_t N, const uint32_t bias_addr, cluster_map_t cluster_map) {
    if (0 == M || 0 == N || 0 == K) {
        return;
    }
    flex_global_barrier_xy();
    uint32_t cluster_id = flex_get_cluster_id();
    // cluster_id among activated clusters
    uint32_t local_cluster_id = ARCH_NUM_CLUSTER;
    uint32_t n_cluster_activated = 0;
    if ((cluster_map & (0x01 << cluster_id)) != 0) {
        for (int i = 0; i < ARCH_NUM_CLUSTER; i++) {
            if ((cluster_map & (0x01 << i)) != 0) {
                if (cluster_id == i) {
                    local_cluster_id = n_cluster_activated;
                }
                n_cluster_activated += 1;
            }
        }

        uint32_t accumulator, local_A_0, local_A_1, local_B_0, local_B_1;
        accumulator = ARCH_CLUSTER_TCDM_SIZE - OPAND_SIZE;
        local_A_0 = accumulator - OPAND_SIZE;
        local_A_1 = local_A_0 - OPAND_SIZE;
        local_B_0 = local_A_1 - OPAND_SIZE;
        local_B_1 = local_B_0 - OPAND_SIZE;

        int m_remaining, n_remaining, m_tile, n_tile, k_tile, bK;
        // tile refer to tile of output a single cluster process on
        int tile_width = TILE_WIDTH;
        // block refer to block of output that all clusters process on at the same time
        int block_width_j = tile_width * n_cluster_activated;
        int block_width_i = tile_width;
        // For double buffering: indicates the suffix of buffer to use
        bool is_odd = 1;

        // each cluster compute a tile of the output matrix
        int gi = 0;
        int gj = local_cluster_id;

        // i: row number
        for (int i = 0; i < (M - 1) / block_width_i + 1; i++) {
            // j: colomn number
            for (int j = 0; j < (N - 1) / block_width_j + 1; j++) {
                m_remaining = (i < M / block_width_i) ? block_width_i : (M % block_width_i);
                n_remaining = (j < N / block_width_j) ? block_width_j : (N % block_width_j);
                
                // Skip clusters that aren't needed for this block
                if (m_remaining <= gi * tile_width || n_remaining <= gj * tile_width) {
                    continue;
                }
                
                m_tile = fmin(tile_width, m_remaining - gi * tile_width);
                n_tile = fmin(tile_width, n_remaining - gj * tile_width);
                
                if(flex_is_dm_core()) {
                    if (bias_addr == zomem(0)) {
                        flex_dma_async_1d(local(accumulator), zomem(0), m_tile * n_tile * DATA_SIZE_BYTES);
                        flex_dma_async_wait_all();
                    } else {
                        flex_dma_sync_2d(local(accumulator), bias_addr + (gj * tile_width + j * block_width_j) * DATA_SIZE_BYTES, n_tile*DATA_SIZE_BYTES, n_tile*DATA_SIZE_BYTES, 0, m_tile);
                    }
                }
                if(flex_is_first_core()) {
                    // flex_redmule_config() usage: [m_size, n_size] * [n_size, k_size] = [m_size, k_size]
                    flex_redmule_config(m_tile, tile_width, n_tile);
                }
                
                // bK: inner loop tile computing partial sums
                for (bK = 0; bK < K; bK += tile_width) {
                    k_tile = fmin(tile_width, K - bK);

                    // SoftHier_HBM -> SoftHier_TCDM 2D
                    if(flex_is_dm_core()) {
                        uint32_t load_dest_A, load_dest_B;
                        if (is_odd) {
                            load_dest_A = local_A_0;
                            load_dest_B = local_B_0;
                        } else {
                            load_dest_A = local_A_1;
                            load_dest_B = local_B_1;
                        }
                        flex_dma_sync_2d(local(load_dest_A), A + ((K * (tile_width * gi + i * block_width_i)) + bK) * DATA_SIZE_BYTES, k_tile*DATA_SIZE_BYTES, k_tile*DATA_SIZE_BYTES, K*DATA_SIZE_BYTES, m_tile);
                        flex_dma_sync_2d(local(load_dest_B), B + (N * bK + tile_width * gj + j * block_width_j) * DATA_SIZE_BYTES, n_tile*DATA_SIZE_BYTES,  n_tile*DATA_SIZE_BYTES, N*DATA_SIZE_BYTES, k_tile);
                    }
                    
                    // make sure data is ready
                    flex_intra_cluster_sync();

                    if (flex_is_first_core()) {
                        // change configuration if the tile in K dimension is not full
                        if (k_tile != tile_width) {
                            // flex_redmule_config() usage: [m_size, n_size] * [n_size, k_size] = [m_size, k_size]
                            flex_redmule_config(m_tile, k_tile, n_tile);
                        }

                        uint32_t _in_local_a, _in_local_b;
                        if (is_odd) {
                            _in_local_a = local_A_0;
                            _in_local_b = local_B_0;
                        } else {
                            _in_local_a = local_A_1;
                            _in_local_b = local_B_1;
                        }
                        uint32_t _in_local_sum = accumulator;

                        ///////////////////
                        // multiply and accumulate
                        flex_redmule_trigger(_in_local_a, _in_local_b, _in_local_sum, REDMULE_FP_16);
                        flex_redmule_wait();
                        ///////////////////
                    }
                    is_odd = 1 - is_odd;
                }

                flex_intra_cluster_sync();
                // SoftHier_TCDM -> SoftHier_HBM
                if(flex_is_dm_core()) {
                    flex_dma_sync_2d(C + ((tile_width * gi + i * block_width_i) * N + tile_width * gj + j * block_width_j) * DATA_SIZE_BYTES, local(accumulator), n_tile*DATA_SIZE_BYTES, N*DATA_SIZE_BYTES, n_tile*DATA_SIZE_BYTES, m_tile);
                }
            }
        }
    }
    flex_global_barrier_xy();
}


#endif