#include <math.h>
#include "flex_runtime.h"
#include "flex_redmule.h"
#include "flex_printf.h"
#include "flex_cluster_arch.h"
#include "flex_dma_pattern.h"
// #include "moe.h"
// #include "moe_decode_centralized.h"
#include "moe_prefill.h"
// #define PRINT_DEBUG 0

int main();
int main(){
    flex_barrier_xy_init();
    flex_global_barrier_xy();

    // Parameters below follows the configuration of MoE model used in preload
    uint64_t n_token = 64;
    uint16_t dim = 1024;
    // uint64_t dim = 7168;
    uint16_t inter_dim = 512;
    // uint64_t inter_dim = 2048;
    // uint64_t n_routed_experts = 256;
    uint64_t n_routed_experts = 16;
    uint64_t n_shared_experts = 1;
    uint64_t n_activated_experts = 8;
    // uint16_t n_activated_experts = 4;
    
    /**
     * Shape of matrix:
     * in_token, actual_out, golden_out: [n_token, dim]
     * 
     * Gate
     *   weight: [dim, n_routed_experts]
     * Expert [n_routed_experts + n_shared_experts, ...]
     *   w1
     *     weight: [dim, inter_dim]
     *     bias: [1, inter_dim]
     *   w2
     *     weight: [inter_dim, dim]
     *     bias: [1, dim]
     *   w3
     *     weight: [dim, inter_dim]
     *     bias: [1, inter_dim]
     */

     uint64_t in_token_offset =            0;
     uint64_t gate_weights_offset =        in_token_offset + n_token * dim * DATA_SIZE_BYTES;
     uint64_t expert_w1_weights_offset =   gate_weights_offset + dim * n_routed_experts * DATA_SIZE_BYTES;
     uint64_t expert_w1_bias_offset =      expert_w1_weights_offset + dim * inter_dim * (n_routed_experts + n_shared_experts) * DATA_SIZE_BYTES;
     uint64_t expert_w2_weights_offset =   expert_w1_bias_offset + inter_dim * (n_routed_experts + n_shared_experts) * DATA_SIZE_BYTES;
     uint64_t expert_w2_bias_offset =      expert_w2_weights_offset + inter_dim * dim * (n_routed_experts + n_shared_experts) * DATA_SIZE_BYTES;
     uint64_t expert_w3_weights_offset =   expert_w2_bias_offset + dim * (n_routed_experts + n_shared_experts) * DATA_SIZE_BYTES;
     uint64_t expert_w3_bias_offset =      expert_w3_weights_offset + dim * (n_routed_experts + n_shared_experts) * inter_dim * DATA_SIZE_BYTES;
     uint64_t actual_out_offset =          expert_w3_bias_offset + inter_dim * (n_routed_experts + n_shared_experts) * DATA_SIZE_BYTES;
     uint64_t golden_out_offset =          actual_out_offset + n_token * dim * DATA_SIZE_BYTES;



#ifdef PRINT_DEBUG
    if (flex_is_first_core() && (0 == flex_get_cluster_id()))
    {
        printf("HBM_NODE_ADDR_SPACE: 0x%08x%08x\n", (uint32_t)(ARCH_HBM_NODE_ADDR_SPACE >> 32), (uint32_t)(ARCH_HBM_NODE_ADDR_SPACE & 0xFFFFFFFF));
        printf("[Check Preload] Addresses:\n");
        printf("in_token offset: 0x%08x%08x\n", (uint32_t)((in_token_offset) >> 32), (uint32_t)((in_token_offset) & 0xFFFFFFFF));
        printf("in_token addr: 0x%16x\n", (uint64_t)((hbm_addr(in_token_offset))));
        printf("gate_weights offset: 0x%08x%08x\n", (uint32_t)((gate_weights_offset) >> 32), (uint32_t)((gate_weights_offset) & 0xFFFFFFFF));
        printf("expert_w1_weights offset: 0x%08x%08x\n", (uint32_t)((expert_w1_weights_offset) >> 32), (uint32_t)((expert_w1_weights_offset) & 0xFFFFFFFF));
        printf("expert_w1_weights addr: 0x%08x%08x\n", (uint32_t)(hbm_south(0, expert_w1_weights_offset) >> 32), (uint32_t)(hbm_south(0, expert_w1_weights_offset) & 0xFFFFFFFF));
        printf("expert_w1_bias offset: 0x%08x%08x\n", (uint32_t)((expert_w1_bias_offset) >> 32), (uint32_t)((expert_w1_bias_offset) & 0xFFFFFFFF));
        printf("expert_w1_bias addr: 0x%08x%08x\n", (uint32_t)(hbm_south(0, expert_w1_bias_offset) >> 32), (uint32_t)(hbm_south(0, expert_w1_bias_offset) & 0xFFFFFFFF));
        printf("expert_w3_weights offset: 0x%08x%08x\n", (uint32_t)((expert_w3_weights_offset) >> 32), (uint32_t)((expert_w3_weights_offset) & 0xFFFFFFFF));
        printf("expert_w3_bias offset: 0x%08x%08x\n", (uint32_t)((expert_w3_bias_offset) >> 32), (uint32_t)((expert_w3_bias_offset) & 0xFFFFFFFF));
        printf("expert_w2_weights offset: 0x%08x%08x\n", (uint32_t)((expert_w2_weights_offset) >> 32), (uint32_t)((expert_w2_weights_offset) & 0xFFFFFFFF));
        printf("expert_w2_bias offset: 0x%08x%08x\n", (uint32_t)((expert_w2_bias_offset) >> 32), (uint32_t)((expert_w2_bias_offset) & 0xFFFFFFFF));
        printf("actual_out offset: 0x%08x%08x\n", (uint32_t)((actual_out_offset) >> 32), (uint32_t)((actual_out_offset) & 0xFFFFFFFF));

        printf("n_token: %d\n", (uint32_t)n_token);
        printf("dim: %d\n", (uint32_t)dim);
        printf("inter_dim: %d\n", (uint32_t)inter_dim);
        printf("n_routed_experts: %d\n", (uint32_t)n_routed_experts);
    }
    
    // First element of each matrix
    if (flex_is_first_core() && (flex_get_cluster_id()==0))
    {
        // ((uint16_t *)(hbm_addr(in_token_offset)))[0 + 0 * dim] = (uint16_t)0x3c00;
        // ((uint16_t *)(hbm_addr(in_token_offset)))[1 + 0 * dim] = (uint16_t)0x3c00;
        // ((uint16_t *)(hbm_addr(gate_weights_offset)))[0 + 0 * dim] = (uint16_t)0x4000;
        // ((uint16_t *)(hbm_addr(gate_weights_offset)))[1 + 0 * dim] = (uint16_t)0x3c00;
        // ((uint16_t *)(hbm_addr(gate_weights_offset)))[2 + 0 * dim] = (uint16_t)0x4000;
        // ((uint16_t *)(hbm_addr(gate_weights_offset)))[0 + 1 * n_routed_experts] = (uint16_t)0x3c00;

        printf("[Check Preload] with first 8 elements of each row of the input\n");
        printf("in_token:\n");
        for (int i = 0; i < n_token; i++) {
            printf("    ");
            // for (int j = 0; j < dim; j++) {
            for (int j = 0; j < 16; j++) {
                printf("0x%04x ", (uint16_t)(hbm_addr(in_token_offset) + (j + i * dim) * DATA_SIZE_BYTES));
            }
            printf("\n");
        }
        printf("gate_weights:\n");
        for (int i = 0; i < 2; i++) {
            printf("    ");
            // for (int j = 0; j < dim; j++) {
            for (int j = 0; j < n_routed_experts; j++) {
                printf("0x%04x ", (uint16_t)(hbm_addr(gate_weights_offset) + (j + i * n_routed_experts) * DATA_SIZE_BYTES));
            }
            printf("\n");
        }
        printf("expert_w1_weights:\n");
        for (int i = 0; i < 1; i++) {
            printf("    ");
            for (int j = 0; j < 16; j++) {
            // for (int j = 0; j < inter_dim; j++) {
                #ifdef MOE_DECODE_H
                    printf("0x%04x ", (uint16_t)(hbm_addr(expert_w1_weights_offset) + (j + i * inter_dim) * DATA_SIZE_BYTES));
                    // printf("0x%04x ", ((uint16_t *)(hbm_addr(expert_w1_weights_offset + ARCH_HBM_NODE_ADDR_SPACE * (2 * ARCH_NUM_CLUSTER_Y + ARCH_NUM_CLUSTER_X))))[j + i * inter_dim]);
                #endif
                #ifdef MOE_DECODE_CENTRALIZED_H
                    printf("0x%04x ", (uint16_t)(hbm_addr(expert_w1_weights_offset) + (j + i * inter_dim) * DATA_SIZE_BYTES));
                    // printf("0x%04x ", ((uint16_t *)(hbm_addr(expert_w1_weights_offset)))[j + i * inter_dim]);
                #endif
            }
            printf("\n");
        }
        printf("expert_w1_bias:\n");
        printf("    ");
        // for (int j = 0; j < inter_dim; j++) {
        for (int j = 0; j < 16; j++) {
            #ifdef MOE_DECODE_H
                // printf("0x%04x ", ((uint16_t *)(hbm_addr(expert_w1_bias_offset + ARCH_HBM_NODE_ADDR_SPACE * (2 * ARCH_NUM_CLUSTER_Y + ARCH_NUM_CLUSTER_X))))[j]);
                printf("0x%04x ", (uint16_t)(hbm_addr(expert_w1_bias_offset) + j * DATA_SIZE_BYTES));
            #endif
            #ifdef MOE_DECODE_CENTRALIZED_H
                // printf("0x%04x ", ((uint16_t *)(hbm_addr(expert_w1_bias_offset)))[j]);
                printf("0x%04x ", (uint16_t)(hbm_addr(expert_w1_bias_offset) + j * DATA_SIZE_BYTES));
            #endif
        }
        printf("\n");
        printf("expert_w3_weights:\n");
        for (int i = 0; i < n_token; i++) {
            printf("    ");
            // for (int j = 0; j < inter_dim; j++) {
            for (int j = 0; j < 16; j++) {
                // printf("0x%04x ", ((uint16_t *)(hbm_addr(expert_w3_weights_offset)))[j + i * inter_dim]);
                printf("0x%04x ", (uint16_t)(hbm_addr(expert_w3_weights_offset) + (j + i * inter_dim) * DATA_SIZE_BYTES));
            }
            printf("\n");
        }
        printf("actual_out:\n");
        for (int i = 0; i < n_token; i++) {
            printf("    ");
            // for (int j = 0; j < dim; j++) {
            for (int j = 0; j < 32; j++) {
                // printf("0x%04x ", ((uint16_t *)(hbm_addr(actual_out_offset)))[j + i * dim]);
                printf("0x%04x ", (uint16_t)(hbm_addr(actual_out_offset) + (j + i * dim) * DATA_SIZE_BYTES));
            }
            printf("\n");
        }
        printf("golden_out:\n");
        for (int i = 0; i < n_token; i++) {
            printf("    ");
            // for (int j = 0; j < dim; j++) {
            for (int j = 0; j < 16; j++) {
                // printf("0x%04x ", ((uint16_t *)(hbm_addr(golden_out_offset)))[j + i * dim]);
                printf("0x%04x ", (uint16_t)(hbm_addr(golden_out_offset) + (j + i * dim) * DATA_SIZE_BYTES));
            }
            printf("\n");
        }        
    }
#endif

    uint32_t eoc_val = 0;
    flex_global_barrier_xy();
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0) {
        printf("[Start MoE Computation]\n");
        flex_timer_start();
    }
    #ifdef MOE_H
    // if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0) {
    //     printf("[Start MoE Computation]\n");
    // }
    // compute_moe(in_token_offset, n_token, dim, inter_dim, n_routed_experts, n_shared_experts, n_activated_experts, gate_weights_offset, expert_w1_weights_offset, expert_w1_bias_offset, expert_w2_weights_offset, expert_w2_bias_offset, expert_w3_weights_offset, expert_w3_bias_offset, actual_out_offset);
    top_k((in_token_offset), (actual_out_offset), (actual_out_offset + DATA_SIZE_BYTES * n_activated_experts), n_activated_experts, n_routed_experts, n_token);
#endif
    
#ifdef MOE_PREFILL_H
    // compute_moe(in_token_offset, n_token, dim, inter_dim, n_routed_experts, n_shared_experts, n_activated_experts, gate_weights_offset, expert_w1_weights_offset, expert_w1_bias_offset, expert_w2_weights_offset, expert_w2_bias_offset, expert_w3_weights_offset, expert_w3_bias_offset, actual_out_offset);

    // void gemm_systolic_wise(const uint64_t A, const uint64_t B, const uint64_t C, uint32_t K_size, uint32_t M_size, uint32_t N_size, uint32_t elem_size, uint32_t tile_dimension_K, uint32_t tile_dimension_M, uint32_t tile_dimension_N, const uint64_t bias_addr, cluster_map_t cluster_map);
    cluster_map_t cluster_coloring_0, cluster_coloring_1, cluster_all;
    cluster_coloring_0 = 0x5A5A;    // 0101101001011010: 1 3 4 6 9 11 12 14
    cluster_coloring_1 = 0xA5A5;    // 1010010110100101: 0 2 5 7 8 10 13 15
    cluster_all = 0xFFFF;
    
    gemm_systolic_wise(hbm_addr(in_token_offset), hbm_addr(expert_w1_weights_offset), hbm_addr(actual_out_offset), dim, n_token, inter_dim, DATA_SIZE_BYTES, TILE_WIDTH_EXPERT_0, TILE_WIDTH_TOKENS, TILE_WIDTH_EXPERT_0, zomem(0), cluster_all);

#endif
    flex_global_barrier_xy();
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0) {
        flex_timer_end();
    }
    
#ifdef PRINT_DEBUG
    // get the output
    if (flex_is_first_core() && (flex_get_cluster_id()==0))
    {
        printf("[Check Results]\n");

        printf("actual_out:\n");
        for (int i = 0; i < n_token; i++) {
            printf("    ");
            // for (int j = 0; j < dim; j++) {
            for (int j = 0; j < 32; j++) {
                // printf("0x%04x ", ((uint16_t *)(hbm_addr(actual_out_offset)))[j + i * dim]);
                printf("0x%04x ", (uint16_t)(hbm_addr(actual_out_offset) + (j + i * dim) * DATA_SIZE_BYTES));
            }
            printf("\n");
        }
        
        printf("golden:\n");
        for (int i = 0; i < n_token; i++) {
            printf("    ");
            // for (int j = 0; j < dim; j++) {
            for (int j = 0; j < 32; j++) {
                // printf("0x%04x ", ((uint16_t *)(hbm_addr(golden_out_offset)))[j + i * dim]);
                printf("0x%04x ", (uint16_t)(hbm_addr(golden_out_offset) + (j + i * dim) * DATA_SIZE_BYTES));
            }
            printf("\n");
        }
    }
#endif
    flex_global_barrier_xy();
    flex_eoc(eoc_val);
    return 0;
}
