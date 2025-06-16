// Author: Haoze Gao <gaohao@student.ethz.ch>

#ifndef MOE_DECODE_H
#define MOE_DECODE_H

#include "utils_decode.h"
#include "element_wise.h"
#include "gemv.h"
#include "top_k.h"
#include "normalize.h"

void compute_moe(uint64_t in_token_addr, uint64_t n_token, uint64_t dim, uint64_t inter_dim, uint64_t n_routed_experts, uint64_t n_shared_experts, uint64_t n_activated_experts, uint64_t gate_weights_addr, uint64_t expert_w1_weights_addr, uint64_t expert_w1_bias_addr, uint64_t expert_w2_weights_addr, uint64_t expert_w2_bias_addr, uint64_t expert_w3_weights_addr, uint64_t expert_w3_bias_addr, uint64_t actual_out_addr) {
    cluster_map_t cluster_coloring_0, cluster_coloring_1, cluster_all;
    cluster_coloring_0 = 0x5A5A;    // 0101101001011010: 1 3 4 6 9 11 12 14
    cluster_coloring_1 = 0xA5A5;    // 1010010110100101: 0 2 5 7 8 10 13 15
    cluster_all = 0xFFFF;

    uint32_t top_k_weights_tcdm, top_k_indices_tcdm;
    top_k_weights_tcdm = 0;
    top_k_indices_tcdm = top_k_weights_tcdm + n_token * n_activated_experts * DATA_SIZE_BYTES;
    // Temporary write-back locations
    uint64_t top_k_weights_addr, top_k_indices_addr;
    uint64_t temp_token_0, temp_token_1;
    top_k_weights_addr = actual_out_addr + 2 * (n_token * dim * DATA_SIZE_BYTES);
    top_k_indices_addr = top_k_weights_addr + n_token * n_activated_experts * DATA_SIZE_BYTES;
    // stores intermediate results of all experts
    temp_token_0 = top_k_indices_addr + n_token * n_activated_experts * DATA_SIZE_BYTES;
    temp_token_1 = temp_token_0 + n_token * inter_dim * (n_routed_experts + n_shared_experts) * DATA_SIZE_BYTES;
    
    // if (0 == flex_get_cluster_id() && flex_is_first_core()) {
    //     printf("top_k_weights_addr: %08x%08x\n", (uint32_t)(top_k_weights_addr >> 32), (uint32_t)(top_k_weights_addr & 0xFFFFFFFF));
    //     printf("top_k_indices_addr: %08x%08x\n", (uint32_t)(top_k_indices_addr >> 32), (uint32_t)(top_k_indices_addr & 0xFFFFFFFF));
    //     printf("temp_token_0: %08x%08x\n", (uint32_t)(temp_token_0 >> 32), (uint32_t)(temp_token_0 & 0xFFFFFFFF));
    //     printf("temp_token_1: %08x%08x\n", (uint32_t)(temp_token_1 >> 32), (uint32_t)(temp_token_1 & 0xFFFFFFFF));
    // }
    // flex_global_barrier_xy();
    // Gate 
    gemv(hbm_addr(in_token_addr), hbm_addr(gate_weights_addr), hbm_addr(temp_token_0), dim, n_token, n_routed_experts, zomem(0), cluster_all, TILE_WIDTH_GATE);
    flex_global_barrier_xy();
    top_k(hbm_addr(temp_token_0), hbm_addr(top_k_weights_addr), hbm_addr(top_k_indices_addr), n_activated_experts, n_routed_experts, n_token, cluster_all);
    flex_global_barrier_xy();
    // sigmoid
    sigmoid(hbm_addr(top_k_weights_addr), hbm_addr(top_k_weights_addr), n_activated_experts, n_token, cluster_all);
    flex_global_barrier_xy();
    // normalize
    normalize(hbm_addr(top_k_weights_addr), hbm_addr(top_k_weights_addr), n_activated_experts, n_token, cluster_all);
    flex_global_barrier_xy();
    // load computed top k weights and indices into TCDM
    if (flex_is_dm_core()) {
        flex_dma_async_1d(local(top_k_weights_tcdm), hbm_addr(top_k_weights_addr), n_token * n_activated_experts * DATA_SIZE_BYTES);
        flex_dma_async_1d(local(top_k_indices_tcdm), hbm_addr(top_k_indices_addr), n_token * n_activated_experts * DATA_SIZE_BYTES);
        flex_dma_async_wait_all();
    }
    flex_global_barrier_xy();
    // Routed experts
    // self.w2.forward(silu(self.w1.forward(x)) * self.w3.forward(x))
    uint16_t i_expert;
    fp16 w_expert;
    int i = 0;
    #ifdef SYNC_REDUCE
    uint64_t temp_token_offset;
    while (i < n_activated_experts) {
        // load expert weights and indices 
        i_expert = ((uint16_t *)local(top_k_indices_tcdm))[i];
        w_expert = ((fp16 *)local(top_k_weights_tcdm))[i];

        temp_token_offset = i_expert * inter_dim * DATA_SIZE_BYTES;
        // if (0 == flex_get_cluster_id() && flex_is_first_core()) {
        //     printf("[ROUTED EXPERTS] expert_id = %d, expert_weight = 0x%04x\n", i_expert, w_expert);
        // }
        // flex_global_barrier_xy();
        mul_op(&w_expert, &route_scale, &w_expert);
        // w1.forward(x)
        gemv(hbm_addr(in_token_addr), hbm_addr(expert_w1_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES) / 4), hbm_addr(temp_token_0 + temp_token_offset), dim, n_token, inter_dim, hbm_addr(expert_w1_bias_addr + (inter_dim * i_expert * DATA_SIZE_BYTES) / 4), cluster_coloring_0, TILE_WIDTH_EXPERT_0);
        // w3.forward(x)
        gemv(hbm_addr(in_token_addr), hbm_addr(expert_w3_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES) / 4), hbm_addr(temp_token_1 + temp_token_offset), dim, n_token, inter_dim, hbm_addr(expert_w3_bias_addr + (inter_dim * i_expert * DATA_SIZE_BYTES) / 4), cluster_coloring_1, TILE_WIDTH_EXPERT_0);
        i++;   
    }
    i = 0;
    while (i < n_shared_experts) {
        // load expert weights and indices 
        i_expert = i + n_routed_experts;

        temp_token_offset = i_expert * inter_dim * DATA_SIZE_BYTES;
        // if (0 == flex_get_cluster_id() && flex_is_first_core()) {
        //     printf("[ROUTED EXPERTS] expert_id = %d, expert_weight = 0x%04x\n", i_expert, w_expert);
        // }
        // flex_global_barrier_xy();
        mul_op(&w_expert, &route_scale, &w_expert);
        // w1.forward(x)
        gemv(hbm_addr(in_token_addr), hbm_addr(expert_w1_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES) / 4), hbm_addr(temp_token_0 + temp_token_offset), dim, n_token, inter_dim, hbm_addr(expert_w1_bias_addr + (inter_dim * i_expert * DATA_SIZE_BYTES) / 4), cluster_coloring_0, TILE_WIDTH_EXPERT_0);
        // w3.forward(x)
        gemv(hbm_addr(in_token_addr), hbm_addr(expert_w3_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES) / 4), hbm_addr(temp_token_1 + temp_token_offset), dim, n_token, inter_dim, hbm_addr(expert_w3_bias_addr + (inter_dim * i_expert * DATA_SIZE_BYTES) / 4), cluster_coloring_1, TILE_WIDTH_EXPERT_0);
        i++;   
    }
    i = 0;
    flex_global_barrier_xy();
    while (i < n_activated_experts) {
        // load expert weights and indices 
        i_expert = ((uint16_t *)local(top_k_indices_tcdm))[i];
        w_expert = ((fp16 *)local(top_k_weights_tcdm))[i];
        
        temp_token_offset = i_expert * inter_dim * DATA_SIZE_BYTES;
        // silu(w1.forward(x))
        silu(hbm_addr(temp_token_0 + temp_token_offset), hbm_addr(temp_token_0 + temp_token_offset), inter_dim, n_token, cluster_all);
        // silu(w1.forward(x)) * w3.forward(x)
        dot_product(hbm_addr(temp_token_0 + temp_token_offset), hbm_addr(temp_token_1 + temp_token_offset), hbm_addr(temp_token_0 + temp_token_offset), inter_dim, n_token, cluster_all);
        i++;
    }
    i = 0;
    while (i < n_shared_experts) {
        // load expert weights and indices 
        i_expert = i + n_routed_experts;

        temp_token_offset = i_expert * inter_dim * DATA_SIZE_BYTES;
        // silu(w1.forward(x))
        silu(hbm_addr(temp_token_0 + temp_token_offset), hbm_addr(temp_token_0 + temp_token_offset), inter_dim, n_token, cluster_all);
        // silu(w1.forward(x)) * w3.forward(x)
        dot_product(hbm_addr(temp_token_0 + temp_token_offset), hbm_addr(temp_token_1 + temp_token_offset), hbm_addr(temp_token_0 + temp_token_offset), inter_dim, n_token, cluster_all);
        i++;
    }
    flex_global_barrier_xy();
    i = 0;
    while (i < n_activated_experts) {
        // load expert weights and indices 
        i_expert = ((uint16_t *)local(top_k_indices_tcdm))[i];
        w_expert = ((fp16 *)local(top_k_weights_tcdm))[i];

        temp_token_offset = i_expert * inter_dim * DATA_SIZE_BYTES;
        // w2.forward(silu(w1.forward(x)) * w3.forward(x))
        gemv(hbm_addr(temp_token_0 + temp_token_offset), hbm_addr(expert_w2_weights_addr + (inter_dim * dim * i_expert * DATA_SIZE_BYTES) / 8), hbm_addr(temp_token_0 + temp_token_offset), inter_dim, n_token, dim, hbm_addr(expert_w2_bias_addr + (dim * i_expert * DATA_SIZE_BYTES) / 8), cluster_all, TILE_WIDTH_EXPERT_1);
        
        // NOTE: regulate the data traffic? this global barrier increases utilization
        // flex_global_barrier_xy();
        // multiply by gate weight and add to the output
        dot_product_const(hbm_addr(temp_token_0 + temp_token_offset), w_expert, hbm_addr(temp_token_0 + temp_token_offset), dim, n_token, cluster_all);
        add(hbm_addr(temp_token_0 + temp_token_offset), hbm_addr(actual_out_addr), hbm_addr(actual_out_addr), dim, n_token, cluster_all);
        
        i++;
    }
    i = 0;
    while (i < n_shared_experts) {
        // load expert weights and indices 
        i_expert = i + n_routed_experts;

        temp_token_offset = i_expert * inter_dim * DATA_SIZE_BYTES;
        // w2.forward(silu(w1.forward(x)) * w3.forward(x))
        gemv(hbm_addr(temp_token_0 + temp_token_offset), hbm_addr(expert_w2_weights_addr + (inter_dim * dim * i_expert * DATA_SIZE_BYTES) / 8), hbm_addr(temp_token_0 + temp_token_offset), inter_dim, n_token, dim, hbm_addr(expert_w2_bias_addr + (dim * i_expert * DATA_SIZE_BYTES) / 8), cluster_all, TILE_WIDTH_EXPERT_1);
        
        // flex_global_barrier_xy();
        // add to the output
        add(hbm_addr(temp_token_0 + temp_token_offset), hbm_addr(actual_out_addr), hbm_addr(actual_out_addr), dim, n_token, cluster_all);
        
        i++;
    }
    #else
    while (i < n_activated_experts) {
        i_expert = ((uint16_t *)local(top_k_indices_tcdm))[i];
        w_expert = ((fp16 *)local(top_k_weights_tcdm))[i];
        // if (0 == flex_get_cluster_id() && flex_is_first_core()) {
        //     printf("[ROUTED EXPERTS] expert_id = %d, expert_weight = 0x%04x\n", i_expert, w_expert);
        // }
        // flex_global_barrier_xy();
        mul_op(&w_expert, &route_scale, &w_expert);
        // w1.forward(x)
        gemv(hbm_addr(in_token_addr), hbm_addr(expert_w1_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES) / 4), hbm_addr(temp_token_0), dim, n_token, inter_dim, hbm_addr(expert_w1_bias_addr + (inter_dim * i_expert * DATA_SIZE_BYTES) / 4), cluster_coloring_0, TILE_WIDTH_EXPERT_0);
        // w3.forward(x)
        gemv(hbm_addr(in_token_addr), hbm_addr(expert_w3_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES) / 4), hbm_addr(temp_token_1), dim, n_token, inter_dim, hbm_addr(expert_w3_bias_addr + (inter_dim * i_expert * DATA_SIZE_BYTES) / 4), cluster_coloring_1, TILE_WIDTH_EXPERT_0);
        flex_global_barrier_xy();
        
        // silu(w1.forward(x))
        silu(hbm_addr(temp_token_0), hbm_addr(temp_token_0), inter_dim, n_token, cluster_all);
        // flex_global_barrier_xy();
        // silu(w1.forward(x)) * w3.forward(x)
        dot_product(hbm_addr(temp_token_0), hbm_addr(temp_token_1), hbm_addr(temp_token_0), inter_dim, n_token, cluster_all);

        flex_global_barrier_xy();
        // w2.forward(silu(w1.forward(x)) * w3.forward(x))
        gemv(hbm_addr(temp_token_0), hbm_addr(expert_w2_weights_addr + (inter_dim * dim * i_expert * DATA_SIZE_BYTES) / 8), hbm_addr(temp_token_0), inter_dim, n_token, dim, hbm_addr(expert_w2_bias_addr + (dim * i_expert * DATA_SIZE_BYTES) / 8), cluster_all, TILE_WIDTH_EXPERT_1);
        
        flex_global_barrier_xy();
        // multiply by gate weight and add to the output
        dot_product_const(hbm_addr(temp_token_0), w_expert, hbm_addr(temp_token_0), dim, n_token, cluster_all);
        add(hbm_addr(temp_token_0), hbm_addr(actual_out_addr), hbm_addr(actual_out_addr), dim, n_token, cluster_all);
        flex_global_barrier_xy();
        i++;
    }
        
    // Shared experts
    // self.w2.forward(silu(self.w1.forward(x)) * self.w3.forward(x))
    for (int i_expert = n_routed_experts; i_expert < (n_routed_experts + n_shared_experts); i_expert++) {
        // w1.forward(x)
        gemv(hbm_addr(in_token_addr), hbm_addr(expert_w1_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES) / 4), hbm_addr(temp_token_0), dim, n_token, inter_dim, hbm_addr(expert_w1_bias_addr + (inter_dim * i_expert * DATA_SIZE_BYTES) / 4), cluster_coloring_0, TILE_WIDTH_EXPERT_0);
        
        // w3.forward(x)
        gemv(hbm_addr(in_token_addr), hbm_addr(expert_w3_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES) / 4), hbm_addr(temp_token_1), dim, n_token, inter_dim, hbm_addr(expert_w3_bias_addr + (inter_dim * i_expert * DATA_SIZE_BYTES) / 4), cluster_coloring_1, TILE_WIDTH_EXPERT_0);
        flex_global_barrier_xy();
        // silu(w1.forward(x))
        silu(hbm_addr(temp_token_0), hbm_addr(temp_token_0), inter_dim, n_token, cluster_all);
        // flex_global_barrier_xy();
        // silu(w1.forward(x)) * w3.forward(x)
        dot_product(hbm_addr(temp_token_0), hbm_addr(temp_token_1), hbm_addr(temp_token_0), inter_dim, n_token, cluster_all);
        
        flex_global_barrier_xy();
        // w2.forward(silu(w1.forward(x)) * w3.forward(x))
        gemv(hbm_addr(temp_token_0), hbm_addr(expert_w2_weights_addr + (inter_dim * dim * i_expert * DATA_SIZE_BYTES) / 8), hbm_addr(temp_token_0), inter_dim, n_token, dim, hbm_addr(expert_w2_bias_addr + (dim * i_expert * DATA_SIZE_BYTES) / 8), cluster_all, TILE_WIDTH_EXPERT_1);
        
        flex_global_barrier_xy();
        // add to the output
        add(hbm_addr(temp_token_0), hbm_addr(actual_out_addr), hbm_addr(actual_out_addr), dim, n_token, cluster_all);
        // can be ignored because n_shared_experts is 1
        flex_global_barrier_xy();
    }
    #endif
}

#endif