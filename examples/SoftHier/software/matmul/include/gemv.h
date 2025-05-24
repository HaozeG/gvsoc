#ifndef GEMV_H
#define GEMV_H

#include "utils_decode.h"

void gemv(const uint64_t A, const uint64_t B, const uint64_t C, const uint16_t K, const uint16_t M, const uint16_t N, const uint64_t bias_addr, cluster_map_t cluster_map, uint16_t tile_width);

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

void gemv(const uint64_t A, const uint64_t B, const uint64_t C, const uint16_t K, const uint16_t M, const uint16_t N, const uint64_t bias_addr, cluster_map_t cluster_map, uint16_t tile_width) {
    if (0 == M || 0 == N || 0 == K) {
        return;
    }
    // flex_global_barrier_xy();
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
        uint32_t opand_size = tile_width * tile_width * DATA_SIZE_BYTES;
        accumulator = ARCH_CLUSTER_TCDM_SIZE - opand_size;
        local_A_0 = accumulator - opand_size;
        local_A_1 = local_A_0 - opand_size;
        local_B_0 = local_A_1 - opand_size;
        local_B_1 = local_B_0 - opand_size;

        int m_remaining, n_remaining, m_tile, n_tile, k_tile, bK;
        // block refer to block of output that all clusters process on at the same time
        int block_width_j = tile_width * n_cluster_activated;
        int block_width_i = tile_width;
        // For double buffering: indicates the suffix of buffer to use
        bool is_odd = 1;

        // each cluster compute a tile of the output matrix
        int gi = 0;
        int gj = local_cluster_id;

        // Define regular access pattern for HBM nodes
        uint64_t cluster_offset;
        uint32_t cluster_id_x = cluster_id % ARCH_NUM_CLUSTER_X;
        uint32_t cluster_id_y = cluster_id / ARCH_NUM_CLUSTER_X;
        uint32_t tile_offset_per_node;
        uint16_t addr_shift;
        #ifndef CENTRALIZED_MOE
        if (tile_width == TILE_WIDTH_GATE && bias_addr == zomem(0)) {
            addr_shift = 3;
        } else if (tile_width == TILE_WIDTH_EXPERT_0) {
            addr_shift = 2;
        } else if (tile_width == TILE_WIDTH_EXPERT_1) {
            addr_shift = 3;
        } else {
            addr_shift = 0;
        }
        if (1 == ((cluster_id_x % 2) ^ (cluster_id_y % 2))) {
            // cluster_id_x, cluster_id_y has different parity: access south HBM nodes
            cluster_offset = (2 * ARCH_NUM_CLUSTER_Y + ARCH_NUM_CLUSTER_X) * (uint64_t)ARCH_HBM_NODE_ADDR_SPACE + (cluster_id % ARCH_NUM_CLUSTER_X) * (uint64_t)ARCH_HBM_NODE_ADDR_SPACE;
            tile_offset_per_node = (cluster_id_y >> 1);
        } else {
            // cluster_id_x, cluster_id_y has the same parity: access west HBM nodes
            cluster_offset = (cluster_id / ARCH_NUM_CLUSTER_X) * (uint64_t)ARCH_HBM_NODE_ADDR_SPACE;
            tile_offset_per_node = (cluster_id_x >> 1);
        }
        #else
        cluster_offset = 0;
        tile_offset_per_node = 0;
        addr_shift = 0;
        #endif

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
                
                // m_tile = min(tile_width, m_remaining - gi * tile_width);
                m_tile = 1;
                n_tile = min(tile_width, n_remaining - gj * tile_width);

                if(flex_is_dm_core()) {
                    if (bias_addr == zomem(0)) {
                        flex_dma_async_1d(local(accumulator), zomem(0), m_tile * n_tile * DATA_SIZE_BYTES);
                        flex_dma_async_wait_all();
                    } else {
                        // TODO: the way to calculate bias offset may not be correct
                        // flex_dma_sync_2d(local(accumulator), bias_addr + (gj * tile_width + j * block_width_j) * DATA_SIZE_BYTES + cluster_offset, n_tile*DATA_SIZE_BYTES, n_tile*DATA_SIZE_BYTES, 0, m_tile);
                        // for gemv: m_tile = 1
                        flex_dma_async_1d(local(accumulator), bias_addr + (tile_offset_per_node * tile_width + j * (block_width_j >> addr_shift)) * DATA_SIZE_BYTES + cluster_offset, n_tile*DATA_SIZE_BYTES);
                        flex_dma_async_wait_all();
                    }
                }

                if (flex_is_first_core()) {
                    // flex_redmule_config() usage: [m_size, n_size] * [n_size, k_size] = [m_size, k_size]
                    flex_redmule_config(m_tile, tile_width, n_tile);
                }

               
                // bK: inner loop tile computing partial sums
                for (bK = 0; bK < K; bK += tile_width) {
                    k_tile = min(tile_width, K - bK);

                    // SoftHier_HBM -> SoftHier_TCDM 2D
                    if(flex_is_dm_core()) {
                        uint64_t load_dest_A, load_dest_B;
                        if (is_odd) {
                            load_dest_A = local_A_0;
                            load_dest_B = local_B_0;
                        } else {
                            load_dest_A = local_A_1;
                            load_dest_B = local_B_1;
                        }
                        flex_dma_sync_2d(local(load_dest_A), A + ((K * (tile_width * gi + i * block_width_i)) + bK) * DATA_SIZE_BYTES, k_tile*DATA_SIZE_BYTES, k_tile*DATA_SIZE_BYTES, K*DATA_SIZE_BYTES, m_tile);
                        // for gemv: m_tile = 1
                        // flex_dma_async_1d(local(load_dest_A), A + ((K * (tile_width * gi + i * block_width_i)) + bK) * DATA_SIZE_BYTES, k_tile*DATA_SIZE_BYTES);
                        // flex_dma_sync_2d(local(load_dest_B), B + (N * bK + tile_width * gj + j * block_width_j) * DATA_SIZE_BYTES + cluster_offset, n_tile*DATA_SIZE_BYTES,  n_tile*DATA_SIZE_BYTES, N*DATA_SIZE_BYTES, k_tile);
                        flex_dma_async_1d(local(load_dest_B), B + ((N >> addr_shift) * bK + tile_offset_per_node * n_tile * k_tile + j * (block_width_j >> addr_shift)) * DATA_SIZE_BYTES + cluster_offset, n_tile * k_tile * DATA_SIZE_BYTES);
                        flex_dma_async_wait_all();
                    }

                    if (flex_is_first_core()) {
                        // change configuration if the tile in K dimension is not full
                        if (k_tile != tile_width) {
                            // flex_redmule_config() usage: [m_size, n_size] * [n_size, k_size] = [m_size, k_size]
                            flex_redmule_config(m_tile, k_tile, n_tile);
                        }
                    }
                    
                    // make sure data is ready
                    flex_intra_cluster_sync();
                    if (flex_is_first_core()) {
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
    // flex_global_barrier_xy();
}

#endif