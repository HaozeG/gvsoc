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
#define NUM_CLUSTER_X 4
#define NUM_CLUSTER_Y 4
#define HBM_NODE_SIZE 0x04000000 // 64MB

// designed for NUM_CLUSTER = 16
typedef uint16_t cluster_map_t;
// 2.5 in float
fp16 route_scale = (fp16)0x4100;

void gemv(const uint32_t x, const uint32_t A, const uint32_t y, const uint32_t K, const uint32_t N, const uint32_t bias_addr, cluster_map_t cluster_map, uint32_t wb_cluster);
// void gemv(const uint32_t A, const uint32_t B, const uint32_t C, const uint32_t K, const uint32_t M, const uint32_t N, const uint32_t bias_addr, cluster_map_t cluster_map);
void compute_gemv(uint32_t in_token_addr, uint16_t n_token, uint16_t dim, uint16_t inter_dim, uint16_t n_routed_experts, uint16_t n_shared_experts, uint16_t n_activated_experts, uint32_t gate_weights_addr, uint32_t expert_w1_weights_addr, uint32_t expert_w1_bias_addr, uint32_t expert_w2_weights_addr, uint32_t expert_w2_bias_addr, uint32_t expert_w3_weights_addr, uint32_t expert_w3_bias_addr, uint32_t actual_out_addr);
void broadcast_to_all_clusters(uint32_t dst_addr, uint32_t src_addr, uint32_t size);
void gemv_reduction(uint32_t dst_addr, uint32_t src_addr, uint32_t size);

/**
 * @brief GEMV with cluster map. Adapted from Haoze's version to 1-D transfer.
 * TODO: NOT FUNCTIONALLY VERIFIED YET!
 * TODO: external synchronization required
 * 
 * @param A 
 * @param x 
 * @param y 
 * @param K 
 * @param N 
 * @param bias_addr 
 */
void gemv(const uint32_t x, const uint32_t B, const uint32_t y, const uint32_t K, const uint32_t N, const uint32_t bias_addr, cluster_map_t cluster_map, uint32_t wb_cluster) {
    if (0 == N || 0 == K) {
        return;
    }
    // flex_global_barrier_xy();
    uint32_t cluster_id = flex_get_cluster_id();
    uint32_t local_cluster_id = ARCH_NUM_CLUSTER;
    uint32_t n_cluster_activated = 0;

    // uint32_t core_id = flex_get_core_id();

    if ((cluster_map & (0x01 << cluster_id)) != 0) {
        for (int i = 0; i < ARCH_NUM_CLUSTER; i++) {
            if ((cluster_map & (0x01 << i)) != 0) {
                if (cluster_id == i) {
                    local_cluster_id = n_cluster_activated;
                }
                n_cluster_activated += 1;
            }
        }

        // uint32_t accumulator, local_A_0, local_x;
        // accumulator = 0;
        // local_A = OPAND_SIZE;
        // local_x = local_A + OPAND_SIZE;

        // Double buffering
        uint32_t accumulator, local_x_0, local_x_1, local_B_0, local_B_1;
        accumulator = ARCH_CLUSTER_TCDM_SIZE - OPAND_SIZE;
        local_x_0 = accumulator - OPAND_SIZE;
        local_x_1 = local_x_0 - OPAND_SIZE;
        local_B_0 = local_x_1 - OPAND_SIZE;
        local_B_1 = local_B_0 - OPAND_SIZE;

        int n_remaining, m_tile, n_tile, k_tile, bK;
        int block_width_j = TILE_WIDTH * n_cluster_activated;
        // For double buffering: indicates the suffix of buffer to use
        bool is_odd = 1;

        // int gj = get_pos(cluster_id).y; 
        int gj = local_cluster_id;

        for (int j = 0; j < (N - 1) / block_width_j + 1; j++) {
            n_remaining = (j < N / block_width_j) ? block_width_j : (N % block_width_j);

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
                    flex_dma_async_1d(local(accumulator), bias_addr + (gj * TILE_WIDTH + j * block_width_j) * DATA_SIZE_BYTES, n_tile * DATA_SIZE_BYTES);
                    flex_dma_async_wait_all();
                }
            }
            flex_intra_cluster_sync();

            if (flex_is_first_core()) {
                flex_redmule_config(m_tile, TILE_WIDTH, n_tile);
            }

            for (bK = 0; bK < K; bK += TILE_WIDTH) {
                k_tile = fmin(TILE_WIDTH, K - bK);

                // if (flex_is_dm_core()) {
                //     flex_dma_async_1d(local(local_A), hbm_addr(A + ((K * bK) + gj * TILE_WIDTH + j * BLOCK_WIDTH) * DATA_SIZE_BYTES), k_tile * DATA_SIZE_BYTES);
                //     flex_dma_async_wait_all();
                // }

                // if (flex_is_dm_core()) {
                //     flex_dma_async_1d(local(local_x), hbm_addr(x + bK * DATA_SIZE_BYTES), k_tile * DATA_SIZE_BYTES);
                //     flex_dma_async_wait_all();
                // }

                if(flex_is_dm_core()) {
                    uint32_t load_dest_x, load_dest_B;
                    if (is_odd) {
                        load_dest_x = local_x_0;
                        load_dest_B= local_B_0;
                    } else {
                        load_dest_x = local_x_1;
                        load_dest_B = local_B_1;
                    }
                    // flex_dma_async_1d(local(load_dest_x), hbm_addr(x + bK * DATA_SIZE_BYTES), k_tile * DATA_SIZE_BYTES);   // When token is stored in the HBM
                    flex_dma_async_1d(local(load_dest_x), local(x + bK * DATA_SIZE_BYTES), k_tile * DATA_SIZE_BYTES);   // When token is stored in the local TCDM
                    flex_dma_async_wait_all();

                    flex_dma_async_1d(local(load_dest_B), hbm_addr(B + ((N * bK) + gj * TILE_WIDTH + j * block_width_j) * DATA_SIZE_BYTES), k_tile * DATA_SIZE_BYTES);
                    flex_dma_async_wait_all();
                    
                }

                flex_intra_cluster_sync();

                if (flex_is_first_core()) {
                    if (k_tile != TILE_WIDTH) {
                        flex_redmule_config(m_tile, k_tile, n_tile);
                    }

                    uint32_t _in_local_x, _in_local_b;
                    if (is_odd) {
                        _in_local_x = local_x_0;
                        _in_local_b = local_B_0;
                    } else {
                        _in_local_x = local_x_1;
                        _in_local_b = local_B_1;
                    }
                    uint32_t _in_local_sum = accumulator;
                                        
                    flex_redmule_trigger(_in_local_x, _in_local_b, accumulator, REDMULE_FP_16);
                    flex_redmule_wait();
                }
                is_odd = 1 - is_odd;
            }
            flex_intra_cluster_sync();

            if (flex_is_dm_core()) {
                // flex_dma_async_1d(hbm_addr(y + (gj * TILE_WIDTH + j * block_width_j) * DATA_SIZE_BYTES), local(accumulator), n_tile * DATA_SIZE_BYTES); // When writing back to HBM
                flex_dma_async_1d(remote_cid(wb_cluster, y + (gj * TILE_WIDTH + j * block_width_j) * DATA_SIZE_BYTES), local(accumulator), n_tile * DATA_SIZE_BYTES);   // When writing back to local TCDM
                flex_dma_async_wait_all();
            }
        }
    }
    // flex_global_barrier_xy();
}



/**
 * @brief 
 * 
 * @param dst_addr OFFSET in the TCDM of each cluster to store the broadcasted data
 * @param src_addr Source address of the data to be broadcasted
 * @param size 
 */
void broadcast_to_all_clusters(uint32_t dst_addr, uint32_t src_addr, uint32_t size) {
    flex_global_barrier_xy();//Global barrier
    FlexPosition pos = get_pos(flex_get_cluster_id());
    
    //do row-wise broadcast from cluster 0
    if (flex_is_dm_core() && flex_get_cluster_id() == 0)
    {
        flex_dma_async_1d_broadcast(remote_pos(left_pos(pos), dst_addr), src_addr, size);
        flex_dma_async_wait_all();
    }

    flex_global_barrier_xy();//Global barrier

    // Do column-wise broadcast to all clusters
    for (int cid = 0; cid < NUM_CLUSTER_X; ++cid)
    {
        if (flex_is_dm_core() && flex_get_cluster_id() == cid)
        {   
            // Broadcast the data in the local TCDM to the entire column
            flex_dma_async_1d_broadcast(remote_pos(bottom_pos(pos), dst_addr), local(dst_addr), size);
        }
        flex_global_barrier_xy();
    }
    flex_global_barrier_xy();
}

/**
 * @brief 
 * TODO: does the reduction api includes the current cluster?
 * 
 * @param dst_addr Destination address to store the data in the HBM
 * @param src_addr OFFSET in the TCDM of each cluster to store the reduced data
 * @param size 
 */
void gemv_reduction(uint32_t dst_addr, uint32_t src_addr, uint32_t size) {
    flex_global_barrier_xy();//Global barrier
    FlexPosition pos = get_pos(flex_get_cluster_id());
    
    if (flex_is_dm_core() && flex_get_cluster_id() == 0)
    {
        // reduce partial sum of cluster 1, 2, 3 to cluster 0
        flex_dma_async_1d_reduction(local(src_addr), remote_pos(left_pos(pos), src_addr), size, COLLECTIVE_REDADD_FP_16);
        flex_dma_async_wait_all();

        // reduce partial sum of cluster 4, 8, 12 to cluster 0
        flex_dma_async_1d_reduction(local(src_addr), remote_pos(bottom_pos(pos), src_addr), size, COLLECTIVE_REDADD_FP_16);
        flex_dma_async_wait_all();

        // Write the reduced data to the HBM
        flex_dma_async_1d(hbm_addr(dst_addr), local(src_addr), size);
        flex_dma_async_wait_all();
    }
    flex_global_barrier_xy();
}

// expert_w1_weights_addr: offset of w1 weights in EACH HBM CHANNEL (duplicated)
// expert_w1_bias_addr: offset of w1 bias in EACH HBM CHANNEL (duplicated)
// expert_w2_weights_addr: offset of w2 weights in EACH HBM CHANNEL (duplicated)
// expert_w2_bias_addr: offset of w2 bias in EACH HBM CHANNEL (duplicated)
// expert_w3_weights_addr: offset of w3 weights in EACH HBM CHANNEL (duplicated)
// expert_w3_bias_addr: offset of w3 bias in EACH HBM CHANNEL (duplicated)
void compute_gemv(uint32_t in_token_addr, uint16_t n_token, uint16_t dim, uint16_t inter_dim, uint16_t n_routed_experts, uint16_t n_shared_experts, uint16_t n_activated_experts, uint32_t gate_weights_addr, uint32_t expert_w1_weights_addr, uint32_t expert_w1_bias_addr, uint32_t expert_w2_weights_addr, uint32_t expert_w2_bias_addr, uint32_t expert_w3_weights_addr, uint32_t expert_w3_bias_addr, uint32_t actual_out_addr) {
    flex_global_barrier_xy();
    uint32_t top_k_weights_addr, top_k_indices_addr;
    uint32_t temp_token_0, temp_token_1;
    top_k_weights_addr = actual_out_addr + n_token * dim * DATA_SIZE_BYTES;
    top_k_indices_addr = top_k_weights_addr + n_token * n_activated_experts * DATA_SIZE_BYTES;
    temp_token_0 = top_k_indices_addr + n_token * n_activated_experts * DATA_SIZE_BYTES;
    temp_token_1 = temp_token_0 + n_token * dim * DATA_SIZE_BYTES;

    // temp_out_0 stored after the golden output
    uint32_t temp_out_0 = actual_out_addr + 2 * (n_token * dim * DATA_SIZE_BYTES);
    uint32_t temp_out_1 = temp_out_0 + n_token * dim * DATA_SIZE_BYTES;

    // HBM channel address map
    uint32_t hbm_ch0_offset = 0;
    uint32_t hbm_ch1_offset = hbm_ch0_offset + HBM_NODE_SIZE;
    uint32_t hbm_ch2_offset = hbm_ch0_offset + HBM_NODE_SIZE * 2;
    uint32_t hbm_ch3_offset = hbm_ch0_offset + HBM_NODE_SIZE * 3;

    uint32_t hbm_south_offset = hbm_ch0_offset + HBM_NODE_SIZE * (2 * NUM_CLUSTER_Y + NUM_CLUSTER_X);   // channel 4
    uint32_t hbm_ch4_offset = hbm_south_offset;
    uint32_t hbm_ch5_offset = hbm_south_offset + HBM_NODE_SIZE; // channel 5
    uint32_t hbm_ch6_offset = hbm_south_offset + HBM_NODE_SIZE * 2; // channel 6
    uint32_t hbm_ch7_offset = hbm_south_offset + HBM_NODE_SIZE * 3; // channel 7

    /**
     * INSIDE TCDM
     *      0: input token
     *      after token: partial sum
     *      after partial sum: gate output
     */

    // Broadcast the input token to all clusters
    uint32_t tcdm_in_token_offset = 0;
    uint32_t tcdm_token_addr = local(tcdm_in_token_offset);
    uint32_t in_token_size = n_token * dim * DATA_SIZE_BYTES;
    broadcast_to_all_clusters(tcdm_token_addr, hbm_addr(in_token_addr), in_token_size);

    // Address in the TCDM to store the partial sum
    uint32_t tcdm_partial_sum_offset = tcdm_in_token_offset + in_token_size;
    // uint32_t tcdm_partial_sum_addr = local(tcdm_partial_sum_offset);

    // Gate
    cluster_map_t gate_cluster_map = 0xFFFF;
    uint32_t gate_output_offset = tcdm_partial_sum_offset + in_token_size;
    // gemv(in_token_addr, gate_weights_addr, temp_token_0, dim, n_routed_experts, zomem(0));
    // gemv(in_token_addr, gate_weights_addr, temp_token_0, dim, n_routed_experts, zomem(0), cluster_map);
    flex_global_barrier_xy();
    gemv(tcdm_token_addr, gate_weights_addr, gate_output_offset, dim, n_routed_experts, zomem(0), gate_cluster_map, 0);     // Store the gate output in the TCDM of cluster 0
    flex_global_barrier_xy();
    
    // Routed experts
    // self.w2.forward(silu(self.w1.forward(x)) * self.w3.forward(x))
    int i = 0;
    uint16_t i_expert;
    fp16 w_expert;
    while (i < n_activated_experts) {
        // i_expert = ((uint16_t *)hbm_addr(top_k_indices_addr))[i];
        i_expert = 0;
        // w_expert = ((fp16 *)hbm_addr(top_k_weights_addr))[i];
        w_expert = 0;
        
        // w1.forward(x)
        // gemv(in_token_addr, expert_w1_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES), temp_token_0, dim, inter_dim, hbm_addr(expert_w1_bias_addr + (inter_dim * i_expert * DATA_SIZE_BYTES)));
        // gemv(in_token_addr, expert_w1_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES), temp_token_0, dim, inter_dim, hbm_addr(expert_w1_bias_addr + (inter_dim * i_expert * DATA_SIZE_BYTES)), cluster_map);

        // TODO: include bias, cleaner HBM preload, redmule cover idma, parse address as arguments
        // FIXME: use different tcdm token regions for correctness
        flex_global_barrier_xy();
        // [0:63], fetch data from HBM1, compute by cluster 4, store result on cluster 4
        gemv(tcdm_token_addr, hbm_ch1_offset + expert_w1_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES), tcdm_partial_sum_offset, 64, inter_dim, zomem(0), 0x0010, 4);
        // [64:191], fetch data from HBM2, compute by cluster 8+9, store result on cluster 8
        gemv(tcdm_token_addr, hbm_ch2_offset + expert_w1_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES) + 64 * DATA_SIZE_BYTES, tcdm_partial_sum_offset, 128, inter_dim, zomem(0), 0x0300, 8);
        // [192:383] fetch data from HBM3, compute by cluster 12+13+14, store result on cluster 12
        gemv(tcdm_token_addr, hbm_ch3_offset + expert_w1_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES) + 192 * DATA_SIZE_BYTES, tcdm_partial_sum_offset, 192, inter_dim, zomem(0), 0x7000, 12);
        // [384:447] fetch data from HBM4, compute by cluster 0, store result on cluster 0
        gemv(tcdm_token_addr, hbm_ch4_offset + expert_w1_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES) + 384 * DATA_SIZE_BYTES, tcdm_partial_sum_offset, 64, inter_dim, zomem(0), 0x0001, 0);
        // [448:575] fetch data from HBM5, compute by cluster 1+5, store result on cluster 1
        gemv(tcdm_token_addr, hbm_ch5_offset + expert_w1_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES) + 448 * DATA_SIZE_BYTES, tcdm_partial_sum_offset, 128, inter_dim, zomem(0), 0x0022, 1);
        // [576:767] fetch data from HBM6, compute by cluster 2+6+10, store result on cluster 2
        gemv(tcdm_token_addr, hbm_ch6_offset + expert_w1_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES) + 576 * DATA_SIZE_BYTES, tcdm_partial_sum_offset, 192, inter_dim, zomem(0), 0x0444, 2);
        // [768:1023] fetch data from HBM7, compute by cluster 3+7+11+15, store result on cluster 3
        gemv(tcdm_token_addr, hbm_ch7_offset + expert_w1_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES) + 768 * DATA_SIZE_BYTES, tcdm_partial_sum_offset, 256, inter_dim, zomem(0), 0x8888, 3);
        flex_global_barrier_xy();
        // Perform reduction on partial sums to get the final sum
        gemv_reduction(temp_out_0, tcdm_partial_sum_offset, n_token * inter_dim * DATA_SIZE_BYTES);

        
        // w3.forward(x)
        // gemv(in_token_addr, expert_w3_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES), temp_token_1, dim, inter_dim, hbm_addr(expert_w3_bias_addr + (inter_dim * i_expert * DATA_SIZE_BYTES)));
        // gemv(in_token_addr, expert_w3_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES), temp_token_1, dim, inter_dim, hbm_addr(expert_w3_bias_addr + (inter_dim * i_expert * DATA_SIZE_BYTES)), cluster_map);
        flex_global_barrier_xy();
        // [0:63], fetch data from HBM1, compute by cluster 4, store result on cluster 4
        gemv(tcdm_token_addr, hbm_ch1_offset + expert_w3_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES), tcdm_partial_sum_offset, 64, inter_dim, zomem(0), 0x0010, 4);
        // [64:191], fetch data from HBM2, compute by cluster 8+9, store result on cluster 8
        gemv(tcdm_token_addr, hbm_ch2_offset + expert_w3_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES) + 64 * DATA_SIZE_BYTES, tcdm_partial_sum_offset, 128, inter_dim, zomem(0), 0x0300, 8);
        // [192:383] fetch data from HBM3, compute by cluster 12+13+14, store result on cluster 12
        gemv(tcdm_token_addr, hbm_ch3_offset + expert_w3_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES) + 192 * DATA_SIZE_BYTES, tcdm_partial_sum_offset, 192, inter_dim, zomem(0), 0x7000, 12);
        // [384:447] fetch data from HBM4, compute by cluster 0, store result on cluster 0
        gemv(tcdm_token_addr, hbm_ch4_offset + expert_w3_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES) + 384 * DATA_SIZE_BYTES, tcdm_partial_sum_offset, 64, inter_dim, zomem(0), 0x0001, 0);
        // [448:575] fetch data from HBM5, compute by cluster 1+5, store result on cluster 1
        gemv(tcdm_token_addr, hbm_ch5_offset + expert_w3_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES) + 448 * DATA_SIZE_BYTES, tcdm_partial_sum_offset, 128, inter_dim, zomem(0), 0x0022, 1);
        // [576:767] fetch data from HBM6, compute by cluster 2+6+10, store result on cluster 2
        gemv(tcdm_token_addr, hbm_ch6_offset + expert_w3_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES) + 576 * DATA_SIZE_BYTES, tcdm_partial_sum_offset, 192, inter_dim, zomem(0), 0x0444, 2);
        // [768:1023] fetch data from HBM7, compute by cluster 3+7+11+15, store result on cluster 3
        gemv(tcdm_token_addr, hbm_ch7_offset + expert_w3_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES) + 768 * DATA_SIZE_BYTES, tcdm_partial_sum_offset, 256, inter_dim, zomem(0), 0x8888, 3);
        flex_global_barrier_xy();
        // Perform reduction on partial sums to get the final sum
        gemv_reduction(temp_out_1, tcdm_partial_sum_offset, n_token * inter_dim * DATA_SIZE_BYTES);
        
        // silu(w1.forward(x)) * w3.forward(x)
        // dot_product(temp_token_0, temp_token_1, temp_token_0, inter_dim, n_token);
        
        // w2.forward(silu(w1.forward(x)) * w3.forward(x))
        // gemv(temp_token_0, expert_w2_weights_addr + (inter_dim * dim * i_expert * DATA_SIZE_BYTES), temp_token_0, inter_dim, dim, hbm_addr(expert_w2_bias_addr + (dim * i_expert * DATA_SIZE_BYTES)));
        // gemv(temp_token_0, expert_w2_weights_addr + (inter_dim * dim * i_expert * DATA_SIZE_BYTES), temp_token_0, inter_dim, dim, hbm_addr(expert_w2_bias_addr + (dim * i_expert * DATA_SIZE_BYTES)), cluster_map);
        flex_global_barrier_xy();
        // [0:31], fetch data from HBM1, compute by cluster 4, store result on cluster 4
        gemv(temp_out_0, hbm_ch1_offset + expert_w2_weights_addr + (inter_dim * dim * i_expert * DATA_SIZE_BYTES), tcdm_partial_sum_offset, 32, dim, zomem(0), 0x0010, 4);
        // [32:95], fetch data from HBM2, compute by cluster 8+9, store result on cluster 8
        gemv(temp_out_0, hbm_ch2_offset + expert_w2_weights_addr + (inter_dim * dim * i_expert * DATA_SIZE_BYTES) + 32 * DATA_SIZE_BYTES, tcdm_partial_sum_offset, 64, dim, zomem(0), 0x0300, 8);
        // [96:191] fetch data from HBM3, compute by cluster 12+13+14, store result on cluster 12
        gemv(temp_out_0, hbm_ch3_offset + expert_w2_weights_addr + (inter_dim * dim * i_expert * DATA_SIZE_BYTES) + 96 * DATA_SIZE_BYTES, tcdm_partial_sum_offset, 96, dim, zomem(0), 0x7000, 12);
        // [192:223] fetch data from HBM4, compute by cluster 0, store result on cluster 0
        gemv(temp_out_0, hbm_ch4_offset + expert_w2_weights_addr + (inter_dim * dim * i_expert * DATA_SIZE_BYTES) + 192 * DATA_SIZE_BYTES, tcdm_partial_sum_offset, 32, dim, zomem(0), 0x0001, 0);
        // [224:287] fetch data from HBM5, compute by cluster 1+5, store result on cluster 1
        gemv(temp_out_0, hbm_ch5_offset + expert_w2_weights_addr + (inter_dim * dim * i_expert * DATA_SIZE_BYTES) + 224 * DATA_SIZE_BYTES, tcdm_partial_sum_offset, 64, dim, zomem(0), 0x0022, 1);
        // [288:383] fetch data from HBM6, compute by cluster 2+6+10, store result on cluster 2
        gemv(temp_out_0, hbm_ch6_offset + expert_w2_weights_addr + (inter_dim * dim * i_expert * DATA_SIZE_BYTES) + 288 * DATA_SIZE_BYTES, tcdm_partial_sum_offset, 96, dim, zomem(0), 0x0444, 2);
        // [384:511] fetch data from HBM7, compute by cluster 3+7+11+15, store result on cluster 3
        gemv(temp_out_0, hbm_ch7_offset + expert_w2_weights_addr + (inter_dim * dim * i_expert * DATA_SIZE_BYTES) + 384 * DATA_SIZE_BYTES, tcdm_partial_sum_offset, 128, dim, zomem(0), 0x8888, 3);
        flex_global_barrier_xy();
        // Perform reduction on partial sums to get the final sum
        gemv_reduction(temp_out_0, tcdm_partial_sum_offset, n_token * dim * DATA_SIZE_BYTES);

        i++;
    }
        
    // Shared experts
    // self.w2.forward(silu(self.w1.forward(x)) * self.w3.forward(x))
    for (int i_expert = n_routed_experts; i_expert < (n_routed_experts + n_shared_experts); i_expert++) {
        //     // gemv(in_token_addr, expert_w1_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES), temp_token_0, dim, inter_dim, hbm_addr(expert_w1_bias_addr + (inter_dim * i_expert * DATA_SIZE_BYTES)));
        //     gemv(in_token_addr, expert_w1_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES), temp_token_0, dim, inter_dim, hbm_addr(expert_w1_bias_addr + (inter_dim * i_expert * DATA_SIZE_BYTES)), cluster_map);
        // [0:63], fetch data from HBM1, compute by cluster 4, store result on cluster 4
        gemv(tcdm_token_addr, hbm_ch1_offset + expert_w1_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES), tcdm_partial_sum_offset, 64, inter_dim, zomem(0), 0x0010, 4);
        // [64:191], fetch data from HBM2, compute by cluster 8+9, store result on cluster 8
        gemv(tcdm_token_addr, hbm_ch2_offset + expert_w1_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES) + 64 * DATA_SIZE_BYTES, tcdm_partial_sum_offset, 128, inter_dim, zomem(0), 0x0300, 8);
        // [192:383] fetch data from HBM3, compute by cluster 12+13+14, store result on cluster 12
        gemv(tcdm_token_addr, hbm_ch3_offset + expert_w1_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES) + 192 * DATA_SIZE_BYTES, tcdm_partial_sum_offset, 192, inter_dim, zomem(0), 0x7000, 12);
        // [384:447] fetch data from HBM4, compute by cluster 0, store result on cluster 0
        gemv(tcdm_token_addr, hbm_ch4_offset + expert_w1_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES) + 384 * DATA_SIZE_BYTES, tcdm_partial_sum_offset, 64, inter_dim, zomem(0), 0x0001, 0);
        // [448:575] fetch data from HBM5, compute by cluster 1+5, store result on cluster 1
        gemv(tcdm_token_addr, hbm_ch5_offset + expert_w1_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES) + 448 * DATA_SIZE_BYTES, tcdm_partial_sum_offset, 128, inter_dim, zomem(0), 0x0022, 1);
        // [576:767] fetch data from HBM6, compute by cluster 2+6+10, store result on cluster 2
        gemv(tcdm_token_addr, hbm_ch6_offset + expert_w1_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES) + 576 * DATA_SIZE_BYTES, tcdm_partial_sum_offset, 192, inter_dim, zomem(0), 0x0444, 2);
        // [768:1023] fetch data from HBM7, compute by cluster 3+7+11+15, store result on cluster 3
        gemv(tcdm_token_addr, hbm_ch7_offset + expert_w1_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES) + 768 * DATA_SIZE_BYTES, tcdm_partial_sum_offset, 256, inter_dim, zomem(0), 0x8888, 3);
        flex_global_barrier_xy();
        // Perform reduction on partial sums to get the final sum
        gemv_reduction(temp_out_0, tcdm_partial_sum_offset, n_token * inter_dim * DATA_SIZE_BYTES);


    //     // gemv(in_token_addr, expert_w3_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES), temp_token_1, dim, inter_dim, hbm_addr(expert_w3_bias_addr + (inter_dim * i_expert * DATA_SIZE_BYTES)));
    //     gemv(in_token_addr, expert_w3_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES), temp_token_1, dim, inter_dim, hbm_addr(expert_w3_bias_addr + (inter_dim * i_expert * DATA_SIZE_BYTES)), cluster_map);
        flex_global_barrier_xy();
        // [0:63], fetch data from HBM1, compute by cluster 4, store result on cluster 4
        gemv(tcdm_token_addr, hbm_ch1_offset + expert_w3_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES), tcdm_partial_sum_offset, 64, inter_dim, zomem(0), 0x0010, 4);
        // [64:191], fetch data from HBM2, compute by cluster 8+9, store result on cluster 8
        gemv(tcdm_token_addr, hbm_ch2_offset + expert_w3_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES) + 64 * DATA_SIZE_BYTES, tcdm_partial_sum_offset, 128, inter_dim, zomem(0), 0x0300, 8);
        // [192:383] fetch data from HBM3, compute by cluster 12+13+14, store result on cluster 12
        gemv(tcdm_token_addr, hbm_ch3_offset + expert_w3_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES) + 192 * DATA_SIZE_BYTES, tcdm_partial_sum_offset, 192, inter_dim, zomem(0), 0x7000, 12);
        // [384:447] fetch data from HBM4, compute by cluster 0, store result on cluster 0
        gemv(tcdm_token_addr, hbm_ch4_offset + expert_w3_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES) + 384 * DATA_SIZE_BYTES, tcdm_partial_sum_offset, 64, inter_dim, zomem(0), 0x0001, 0);
        // [448:575] fetch data from HBM5, compute by cluster 1+5, store result on cluster 1
        gemv(tcdm_token_addr, hbm_ch5_offset + expert_w3_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES) + 448 * DATA_SIZE_BYTES, tcdm_partial_sum_offset, 128, inter_dim, zomem(0), 0x0022, 1);
        // [576:767] fetch data from HBM6, compute by cluster 2+6+10, store result on cluster 2
        gemv(tcdm_token_addr, hbm_ch6_offset + expert_w3_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES) + 576 * DATA_SIZE_BYTES, tcdm_partial_sum_offset, 192, inter_dim, zomem(0), 0x0444, 2);
        // [768:1023] fetch data from HBM7, compute by cluster 3+7+11+15, store result on cluster 3
        gemv(tcdm_token_addr, hbm_ch7_offset + expert_w3_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES) + 768 * DATA_SIZE_BYTES, tcdm_partial_sum_offset, 256, inter_dim, zomem(0), 0x8888, 3);
        flex_global_barrier_xy();
        // Perform reduction on partial sums to get the final sum
        gemv_reduction(temp_out_1, tcdm_partial_sum_offset, n_token * inter_dim * DATA_SIZE_BYTES);

    //     // gemv(temp_token_0, expert_w2_weights_addr + (inter_dim * dim * i_expert * DATA_SIZE_BYTES), temp_token_0, inter_dim, dim, hbm_addr(expert_w2_bias_addr + (dim * i_expert * DATA_SIZE_BYTES)));
    //     gemv(temp_token_0, expert_w2_weights_addr + (inter_dim * dim * i_expert * DATA_SIZE_BYTES), temp_token_0, inter_dim, dim, hbm_addr(expert_w2_bias_addr + (dim * i_expert * DATA_SIZE_BYTES)), cluster_map);
        flex_global_barrier_xy();
        // [0:31], fetch data from HBM1, compute by cluster 4, store result on cluster 4
        gemv(temp_out_0, hbm_ch1_offset + expert_w2_weights_addr + (inter_dim * dim * i_expert * DATA_SIZE_BYTES), tcdm_partial_sum_offset, 32, dim, zomem(0), 0x0010, 4);
        // [32:95], fetch data from HBM2, compute by cluster 8+9, store result on cluster 8
        gemv(temp_out_0, hbm_ch2_offset + expert_w2_weights_addr + (inter_dim * dim * i_expert * DATA_SIZE_BYTES) + 32 * DATA_SIZE_BYTES, tcdm_partial_sum_offset, 64, dim, zomem(0), 0x0300, 8);
        // [96:191] fetch data from HBM3, compute by cluster 12+13+14, store result on cluster 12
        gemv(temp_out_0, hbm_ch3_offset + expert_w2_weights_addr + (inter_dim * dim * i_expert * DATA_SIZE_BYTES) + 96 * DATA_SIZE_BYTES, tcdm_partial_sum_offset, 96, dim, zomem(0), 0x7000, 12);
        // [192:223] fetch data from HBM4, compute by cluster 0, store result on cluster 0
        gemv(temp_out_0, hbm_ch4_offset + expert_w2_weights_addr + (inter_dim * dim * i_expert * DATA_SIZE_BYTES) + 192 * DATA_SIZE_BYTES, tcdm_partial_sum_offset, 32, dim, zomem(0), 0x0001, 0);
        // [224:287] fetch data from HBM5, compute by cluster 1+5, store result on cluster 1
        gemv(temp_out_0, hbm_ch5_offset + expert_w2_weights_addr + (inter_dim * dim * i_expert * DATA_SIZE_BYTES) + 224 * DATA_SIZE_BYTES, tcdm_partial_sum_offset, 64, dim, zomem(0), 0x0022, 1);
        // [288:383] fetch data from HBM6, compute by cluster 2+6+10, store result on cluster 2
        gemv(temp_out_0, hbm_ch6_offset + expert_w2_weights_addr + (inter_dim * dim * i_expert * DATA_SIZE_BYTES) + 288 * DATA_SIZE_BYTES, tcdm_partial_sum_offset, 96, dim, zomem(0), 0x0444, 2);
        // [384:511] fetch data from HBM7, compute by cluster 3+7+11+15, store result on cluster 3
        gemv(temp_out_0, hbm_ch7_offset + expert_w2_weights_addr + (inter_dim * dim * i_expert * DATA_SIZE_BYTES) + 384 * DATA_SIZE_BYTES, tcdm_partial_sum_offset, 128, dim, zomem(0), 0x8888, 3);
        flex_global_barrier_xy();
        // Perform reduction on partial sums to get the final sum
        gemv_reduction(temp_out_0, tcdm_partial_sum_offset, n_token * dim * DATA_SIZE_BYTES);
    }
    
    flex_global_barrier_xy();
}

#endif