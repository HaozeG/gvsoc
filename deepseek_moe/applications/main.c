#include <math.h>
#include "flex_runtime.h"
#include "flex_redmule.h"
#include "flex_printf.h"
#include "flex_cluster_arch.h"
#include "flex_dma_pattern.h"
#include "moe.h"
// #define PRINT_DEBUG

#define HBM_NODE_SIZE 0x04000000 // 64MB
#define NUM_CLUSTER_X 4
#define NUM_CLUSTER_Y 4

int main();
int main(){
    flex_barrier_xy_init();
    flex_global_barrier_xy();

    // Parameters below follows the configuration of MoE model used in preload
    uint16_t n_token = 1;
    uint16_t dim = 1024;
    uint16_t inter_dim = 512;
    uint16_t n_routed_experts = 8;
    uint16_t n_shared_experts = 1;
    uint16_t n_activated_experts = 4;
    
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
    // Read the value preloaded into HBM
    // uint32_t in_token_offset =            0;
    // uint32_t gate_weights_offset =        in_token_offset + n_token * dim * DATA_SIZE_BYTES;
    // uint32_t expert_w1_weights_offset =   gate_weights_offset + dim * n_routed_experts * DATA_SIZE_BYTES;
    // uint32_t expert_w1_bias_offset =      expert_w1_weights_offset + dim * inter_dim * (n_routed_experts + n_shared_experts) * DATA_SIZE_BYTES;
    // uint32_t expert_w2_weights_offset =   expert_w1_bias_offset + inter_dim * (n_routed_experts + n_shared_experts) * DATA_SIZE_BYTES;
    // uint32_t expert_w2_bias_offset =      expert_w2_weights_offset + inter_dim * dim * (n_routed_experts + n_shared_experts) * DATA_SIZE_BYTES;
    // uint32_t expert_w3_weights_offset =   expert_w2_bias_offset + dim * (n_routed_experts + n_shared_experts) * DATA_SIZE_BYTES;
    // uint32_t expert_w3_bias_offset =      expert_w3_weights_offset + dim * (n_routed_experts + n_shared_experts) * inter_dim * DATA_SIZE_BYTES;
    // uint32_t actual_out_offset =          expert_w3_bias_offset + inter_dim * (n_routed_experts + n_shared_experts) * DATA_SIZE_BYTES;
    // uint32_t golden_out_offset =          actual_out_offset + n_token * dim * DATA_SIZE_BYTES;

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
     * HBM data placement version 1
     */
    uint32_t in_token_offset = hbm_ch0_offset;   // channel 0
    uint32_t gate_weights_offset = in_token_offset + n_token * dim * DATA_SIZE_BYTES; // channel 0
    uint32_t expert_w1_weights_offset = hbm_ch2_offset; //channel 2
    uint32_t expert_w2_weights_offset = hbm_ch1_offset; // channel 1
    uint32_t expert_w3_weights_offset = hbm_ch5_offset; // channel 5

    // All bias matrices are placed in the same HBM channel (channel 4)
    uint32_t expert_w1_bias_offset = hbm_ch4_offset;  // channel 4
    uint32_t expert_w2_bias_offset = expert_w1_bias_offset + inter_dim * dim * (n_routed_experts + n_shared_experts) * DATA_SIZE_BYTES;
    uint32_t expert_w3_bias_offset = expert_w2_bias_offset + dim * (n_routed_experts + n_shared_experts) * inter_dim * DATA_SIZE_BYTES;

    // uint32_t gate_weights_offset = hbm_south_offset + HBM_NODE_SIZE; // channel 5
    uint32_t actual_out_offset = hbm_ch6_offset; // channel 5
    uint32_t golden_out_offset = actual_out_offset + n_token * dim * DATA_SIZE_BYTES; // channel 5

    /** 
     * HBM data placement version 2
     */
    // Read the value preloaded into HBM (Updated for split matrices)
    // uint32_t in_token_offset = 0;
    // uint32_t expert_w1_weights_offset_1 = in_token_offset + HBM_NODE_SIZE; // First 4.5MB of W1
    // uint32_t expert_w1_weights_offset_2 = in_token_offset + HBM_NODE_SIZE * 2; // Second 4.5MB of W1

    // uint32_t expert_w2_weights_offset_1 = (in_token_offset + HBM_NODE_SIZE * 2) + 4.5 * 1024 * 1024; // First 4.5MB of W2
    // uint32_t expert_w2_weights_offset_2 = in_token_offset + HBM_NODE_SIZE * 3; // Second 4.5MB of W2

    // uint32_t expert_w3_weights_offset_1 = (in_token_offset + HBM_NODE_SIZE * 3) + 4.5 * 1024 * 1024; // First 4.5MB of W3
    // uint32_t expert_w3_weights_offset_2 = in_token_offset + HBM_NODE_SIZE * 4; // Second 4.5MB of W3

    


#ifdef PRINT_DEBUG
    if (flex_is_first_core() && (flex_get_cluster_id()==0))
    {
        printf("[Check Preload] Addresses\n");
        printf("in_token: 0x%x\n", hbm_addr(in_token_offset));
        printf("gate_weight: 0x%x\n", hbm_addr(gate_weights_offset));
        printf("expert_w1_weight: 0x%x\n", hbm_addr(expert_w1_weights_offset));
        printf("expert_w1_bias: 0x%x\n", hbm_addr(expert_w1_bias_offset));
        printf("expert_w2_weight: 0x%x\n", hbm_addr(expert_w2_weights_offset));
        printf("expert_w2_bias: 0x%x\n", hbm_addr(expert_w2_bias_offset));
        printf("expert_w3_weight: 0x%x\n", hbm_addr(expert_w3_weights_offset));
        printf("expert_w3_bias: 0x%x\n", hbm_addr(expert_w3_bias_offset));
        printf("actual: 0x%x\n", hbm_addr(actual_out_offset));
        printf("golden: 0x%x\n", hbm_addr(golden_out_offset));
        printf("n_token: %x\n", n_token);
        printf("dim: %d\n", dim);
        printf("inter_dim: %d\n", inter_dim);
        printf("n_routed_experts: %d\n",n_routed_experts);
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

        // printf("[Check Preload] Data\n");
        printf("[Check Preload] with first 8 elements of each row of the input\n");
        printf("in_token:\n");
        for (int i = 0; i < n_token; i++) {
            printf("    ");
            // for (int j = 0; j < dim; j++) {
            for (int j = 0; j < 16; j++) {
                printf("0x%04x ", ((uint16_t *)(hbm_addr(in_token_offset)))[j + i * dim]);
            }
            printf("\n");
        }
        printf("gate_weights:\n");
        for (int i = 0; i < 2; i++) {
            printf("    ");
            // for (int j = 0; j < dim; j++) {
            for (int j = 0; j < n_routed_experts; j++) {
                printf("0x%04x ", ((uint16_t *)(hbm_addr(gate_weights_offset)))[j + i * n_routed_experts]);
            }
            printf("\n");
        }
        printf("expert_w1_weights:\n");
        for (int i = 0; i < 1; i++) {
            printf("    ");
            for (int j = 0; j < 16; j++) {
            // for (int j = 0; j < inter_dim; j++) {
                printf("0x%04x ", ((uint16_t *)(hbm_addr(expert_w1_weights_offset)))[j + i * inter_dim]);
            }
            printf("\n");
        }
        printf("expert_w1_bias:\n");
        for (int i = 0; i < n_token; i++) {
            printf("    ");
            // for (int j = 0; j < dim; j++) {
            for (int j = 0; j < 16; j++) {
                printf("0x%04x ", ((uint16_t *)(hbm_addr(expert_w1_bias_offset)))[j + i * dim]);
            }
            printf("\n");
        }
        printf("expert_w3_weights:\n");
        for (int i = 0; i < n_token; i++) {
            printf("    ");
            // for (int j = 0; j < dim; j++) {
            for (int j = 0; j < 16; j++) {
                printf("0x%04x ", ((uint16_t *)(hbm_addr(expert_w3_weights_offset)))[j + i * dim]);
            }
            printf("\n");
        }
        printf("actual_out:\n");
        for (int i = 0; i < n_token; i++) {
            printf("    ");
            // for (int j = 0; j < dim; j++) {
            for (int j = 0; j < 16; j++) {
                printf("0x%04x ", ((uint16_t *)(hbm_addr(actual_out_offset)))[j + i * dim]);
            }
            printf("\n");
        }
        printf("golden_out:\n");
        for (int i = 0; i < n_token; i++) {
            printf("    ");
            // for (int j = 0; j < dim; j++) {
            for (int j = 0; j < 16; j++) {
                printf("0x%04x ", ((uint16_t *)(hbm_addr(golden_out_offset)))[j + i * dim]);
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
    compute_moe(in_token_offset, n_token, dim, inter_dim, 
                n_routed_experts, n_shared_experts, n_activated_experts, 
                gate_weights_offset, expert_w1_weights_offset, expert_w1_bias_offset, 
                expert_w2_weights_offset, expert_w2_bias_offset, 
                expert_w3_weights_offset, expert_w3_bias_offset, 
                actual_out_offset);
    flex_global_barrier_xy();
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0) {
        flex_timer_end();
    }
    
#ifdef PRINT_DEBUG
    // get the output
    if (flex_is_first_core() && (flex_get_cluster_id()==0))
    {
        printf("[Check Results]\n");
        // printf("golden_out:\n");
        // for (int i = 0; i < n_token; i++) {
        //     printf("    ");
        //     // for (int j = 0; j < dim; j++) {
        //     for (int j = 0; j < 16; j++) {
        //         printf("0x%04x ", ((uint16_t *)(hbm_addr(golden_out_offset)))[j + i * dim]);
        //     }
        //     printf("\n");
        // }        
        printf("actual_out:\n");
        for (int i = 0; i < n_token; i++) {
            printf("    ");
            // for (int j = 0; j < dim; j++) {
            for (int j = 0; j < 16; j++) {
                printf("0x%04x ", ((uint16_t *)(hbm_addr(actual_out_offset)))[j + i * dim]);
            }
            printf("\n");
        }
    }
#endif
    flex_global_barrier_xy();
    flex_eoc(eoc_val);
    return 0;
}
