#include <math.h>
#include "flex_runtime.h"
#include "flex_redmule.h"
#include "flex_printf.h"
#include "flex_cluster_arch.h"
#include "flex_dma_pattern.h"
// #include "moe.h"
// #include "moe_decode_centralized.h"
#include "moe_decode.h"
#define PRINT_DEBUG 0

int main();
int main(){
    flex_barrier_xy_init();
    flex_global_barrier_xy();

    // Parameters below follows the configuration of MoE model used in preload
    uint16_t n_token = 1;
    uint16_t dim = 1024;
    // uint16_t dim = 1536;
    uint16_t inter_dim = 512;
    // uint16_t inter_dim = 768;
    uint16_t n_routed_experts = 16;
    uint16_t n_shared_experts = 1;
    uint16_t n_activated_experts = 8;
    
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

    /** HBM placement version 3 */
    // uint32_t expert_w1_weights_offset = 0;
    // uint32_t expert_w1_bias_offset = expert_w1_weights_offset + dim * inter_dim * (n_routed_experts + n_shared_experts) * DATA_SIZE_BYTES;
    // uint32_t expert_w2_weights_offset = expert_w1_bias_offset + inter_dim * (n_routed_experts + n_shared_experts) * DATA_SIZE_BYTES;
    // uint32_t expert_w2_bias_offset = expert_w2_weights_offset + inter_dim * dim * (n_routed_experts + n_shared_experts) * DATA_SIZE_BYTES;
    // uint32_t expert_w3_weights_offset = expert_w2_bias_offset + dim * (n_routed_experts + n_shared_experts) * DATA_SIZE_BYTES;
    // uint32_t expert_w3_bias_offset = expert_w3_weights_offset + dim * (n_routed_experts + n_shared_experts) * inter_dim * DATA_SIZE_BYTES;
    // uint32_t gate_weights_offset = expert_w3_bias_offset + inter_dim * (n_routed_experts + n_shared_experts) * DATA_SIZE_BYTES;
    // uint32_t in_token_offset = gate_weights_offset + dim * n_routed_experts * DATA_SIZE_BYTES;    
    // uint32_t actual_out_offset = in_token_offset + n_token * dim * DATA_SIZE_BYTES;
    // uint32_t golden_out_offset = actual_out_offset + n_token * dim * DATA_SIZE_BYTES;

    /** HBM placement version 4 */
    // W1 stored at the beginning of channel 4, 5, 6, 7
    // uint32_t expert_w1_weights_offset = 0;
    // uint32_t expert_w1_bias_offset = expert_w1_weights_offset + dim * inter_dim * (n_routed_experts + n_shared_experts) * DATA_SIZE_BYTES;

    // // W3 stored at the beginning of channel 0, 1, 2, 3
    // uint32_t expert_w3_weights_offset = 0;
    // uint32_t expert_w3_bias_offset = expert_w3_weights_offset + dim * (n_routed_experts + n_shared_experts) * inter_dim * DATA_SIZE_BYTES;

    // // W2 stored in all channels following the W1/W3
    // uint32_t expert_w2_weights_offset = expert_w1_bias_offset + inter_dim * (n_routed_experts + n_shared_experts) * DATA_SIZE_BYTES;
    // uint32_t expert_w2_bias_offset = expert_w2_weights_offset + inter_dim * dim * (n_routed_experts + n_shared_experts) * DATA_SIZE_BYTES;

    // // Gate weights stored in all channels following the W2 bias
    // uint32_t gate_weights_offset = expert_w2_bias_offset + dim * (n_routed_experts + n_shared_experts) * DATA_SIZE_BYTES;

    // // All the other data stored in channel 0, following the gate weights
    // uint32_t in_token_offset = gate_weights_offset + dim * n_routed_experts * DATA_SIZE_BYTES;
    // uint32_t actual_out_offset = in_token_offset + n_token * dim * DATA_SIZE_BYTES;
    // uint32_t golden_out_offset = actual_out_offset + n_token * dim * DATA_SIZE_BYTES;

    /** HBM placement optimized */
    // Size of the tiles in each HBM channel (for 2 clusters)
    uint32_t w1_w3_tile_size_partitioned = 2 * TILE_WIDTH_EXPERT_0 * dim * (n_routed_experts + n_shared_experts) * DATA_SIZE_BYTES;
    uint32_t w2_tile_size_partitioned = 2 * TILE_WIDTH_EXPERT_1 * inter_dim  * (n_routed_experts + n_shared_experts) * DATA_SIZE_BYTES;

    // W1 stored separately at the beginning of channel 4, 5, 6, 7
    uint32_t expert_w1_weights_offset = 0;
    uint32_t expert_w1_bias_offset = expert_w1_weights_offset + w1_w3_tile_size_partitioned;

    // W3 stored separately at the beginning of channel 0, 1, 2, 3
    uint32_t expert_w3_weights_offset = 0;
    uint32_t expert_w3_bias_offset = expert_w3_weights_offset + w1_w3_tile_size_partitioned;

    // W2 stored in all channels following the W1/W3
    uint32_t expert_w2_weights_offset = expert_w1_bias_offset + inter_dim * (n_routed_experts + n_shared_experts) * DATA_SIZE_BYTES;
    uint32_t expert_w2_bias_offset = expert_w2_weights_offset + w2_tile_size_partitioned;
    
    // Gate weights stored in all channels following the W2 bias
    uint32_t gate_weights_offset = expert_w2_bias_offset + dim * (n_routed_experts + n_shared_experts) * DATA_SIZE_BYTES;

    // All the other data stored in channel 0, following the gate weights
    uint32_t in_token_offset = gate_weights_offset + dim * n_routed_experts * DATA_SIZE_BYTES;
    uint32_t actual_out_offset = in_token_offset + n_token * dim * DATA_SIZE_BYTES;
    uint32_t golden_out_offset = actual_out_offset + n_token * dim * DATA_SIZE_BYTES;

#ifdef PRINT_DEBUG
    if (flex_is_first_core() && (flex_get_cluster_id()==0))
    {
        printf("[Check Preload] Addresses\n");
        printf("in_token: 0x%x\n", hbm_addr(in_token_offset));
        printf("gate_weight: 0x%x\n", hbm_addr(gate_weights_offset));
        // printf("expert_w1_weight: 0x%x\n", hbm_addr(expert_w1_weights_offset));
        printf("expert_w1_weight: 0x%x\n", hbm_south(0, expert_w1_weights_offset));    // For distributed version
        // printf("expert_w1_bias: 0x%x\n", hbm_addr(expert_w1_bias_offset));
        printf("expert_w1_bias: 0x%x\n", hbm_south(0, expert_w1_bias_offset));         // For distributed version
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

        // printf("Offset Values:\n");
        // printf("expert_w1_weights_offset: 0x%x\n", expert_w1_weights_offset);
        // printf("expert_w1_bias_offset: 0x%x\n", expert_w1_bias_offset);
        // printf("expert_w3_weights_offset: 0x%x\n", expert_w3_weights_offset);
        // printf("expert_w3_bias_offset: 0x%x\n", expert_w3_bias_offset);
        // printf("expert_w2_weights_offset: 0x%x\n", expert_w2_weights_offset);
        // printf("expert_w2_bias_offset: 0x%x\n", expert_w2_bias_offset);
        // printf("gate_weights_offset: 0x%x\n", gate_weights_offset);
        // printf("in_token_offset: 0x%x\n", in_token_offset);
        // printf("actual_out_offset: 0x%x\n", actual_out_offset);
        // printf("golden_out_offset: 0x%x\n", golden_out_offset);
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
                printf("0x%04x ", ((uint16_t *)(hbm_addr(in_token_offset)))[j + i * dim]);  // For centralized version
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
                // printf("0x%04x ", ((uint16_t *)(hbm_addr(expert_w1_weights_offset)))[j + i * inter_dim]);    // For centralzied version
                printf("0x%04x ", ((uint16_t *)(hbm_south(0, expert_w1_weights_offset)))[j + i * inter_dim]);   // For distributed version
            }
            printf("\n");
        }
        printf("expert_w1_bias:\n");
        for (int i = 0; i < n_token; i++) {
            printf("    ");
            // for (int j = 0; j < dim; j++) {
            for (int j = 0; j < 16; j++) {
                // printf("0x%04x ", ((uint16_t *)(hbm_addr(expert_w1_bias_offset)))[j + i * dim]);        // For centralized version
                printf("0x%04x ", ((uint16_t *)(hbm_south(0, expert_w1_bias_offset)))[j + i * dim]);    // For distributed version
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
            for (int j = 0; j < 32; j++) {
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
#ifdef MOE_H
    // if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0) {
    //     printf("[Start MoE Computation]\n");
    // }
    // compute_moe(in_token_offset, n_token, dim, inter_dim, n_routed_experts, n_shared_experts, n_activated_experts, gate_weights_offset, expert_w1_weights_offset, expert_w1_bias_offset, expert_w2_weights_offset, expert_w2_bias_offset, expert_w3_weights_offset, expert_w3_bias_offset, actual_out_offset);
    top_k((in_token_offset), (actual_out_offset), (actual_out_offset + DATA_SIZE_BYTES * n_activated_experts), n_activated_experts, n_routed_experts, n_token);
#endif

#ifdef MOE_DECODE_H
    // cluster_map_t activated_cluster = 0xFFFF;
    // gemv(hbm_addr(in_token_offset), hbm_addr(gate_weights_offset), hbm_addr(actual_out_offset), dim, 1, n_routed_experts, zomem(0), activated_cluster);
    // gemv(hbm_addr(in_token_offset), hbm_addr(expert_w1_weights_offset), hbm_addr(actual_out_offset), dim, 1, inter_dim, hbm_addr(expert_w1_bias_offset), activated_cluster);
    // silu(hbm_addr(in_token_offset), hbm_addr(actual_out_offset), dim, 1, activated_cluster);
    // dot_product(hbm_addr(in_token_offset), hbm_addr(gate_weights_offset), hbm_addr(actual_out_offset), dim, 1, activated_cluster);
    // fp16 in_const = 0x4000;
    // dot_product_const(hbm_addr(in_token_offset), in_const, hbm_addr(actual_out_offset), dim, 1, activated_cluster);
    // add(hbm_addr(in_token_offset), hbm_addr(gate_weights_offset), hbm_addr(actual_out_offset), dim, 1, activated_cluster);
    // normalize(hbm_addr(in_token_offset), hbm_addr(actual_out_offset), n_activated_experts, 100, activated_cluster);
    // top_k(hbm_addr(in_token_offset), hbm_addr(actual_out_offset+1024), hbm_addr(actual_out_offset), 8, 201, 1, activated_cluster);
    // top_k(hbm_addr(in_token_offset), hbm_addr(actual_out_offset), hbm_addr(actual_out_offset + 1024), 8, 200, 1, activated_cluster);
    // fp16 in1 = 0x9633;
    // fp16 in2 = 0x15a4;
    // fp16 out;
    // mul_op(&in1, &in2, &out);
    // if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0) {
    //     printf("0x%04x * 0x%04x = 0x%04x\n", in1, in2, out);
    // }

    compute_moe(in_token_offset, n_token, dim, inter_dim, n_routed_experts, n_shared_experts, n_activated_experts, gate_weights_offset, expert_w1_weights_offset, expert_w1_bias_offset, expert_w2_weights_offset, expert_w2_bias_offset, expert_w3_weights_offset, expert_w3_bias_offset, actual_out_offset);

    // cluster_map_t cluster_coloring_0, cluster_coloring_1, cluster_all;
    // cluster_coloring_0 = 0x5A5A;    // 0101101001011010: 1 3 4 6 9 11 12 14
    // cluster_coloring_1 = 0xA5A5;    // 1010010110100101: 0 2 5 7 8 10 13 15
    // cluster_all = 0xFFFF;
    // uint32_t top_k_weights_addr, top_k_indices_addr;
    // uint32_t temp_token_0, temp_token_1;
    // top_k_weights_addr = actual_out_offset + n_token * dim * DATA_SIZE_BYTES;
    // top_k_indices_addr = top_k_weights_addr + n_token * n_activated_experts * DATA_SIZE_BYTES;
    // temp_token_0 = top_k_indices_addr + n_token * n_activated_experts * DATA_SIZE_BYTES;
    // temp_token_1 = temp_token_0 + n_token * dim * DATA_SIZE_BYTES;
    
    // Gate 
    // flex_global_barrier_xy();
    // gemv(hbm_addr(in_token_offset), hbm_addr(gate_weights_offset), hbm_addr(temp_token_0), dim, n_token, n_routed_experts, zomem(0), cluster_all);
    // flex_global_barrier_xy();

    // uint16_t i_expert;
    // i_expert = 0;
    // fp16 w_expert;
    // int i = 0;
    // while (i < n_activated_experts) {
    //     // w1.forward(x)
    //     flex_global_barrier_xy();
    //     gemv(hbm_addr(in_token_offset), hbm_addr(expert_w1_weights_offset + (dim * inter_dim * i_expert * DATA_SIZE_BYTES)), hbm_addr(temp_token_0), dim, n_token, inter_dim, hbm_addr(expert_w1_bias_offset + (inter_dim * i_expert * DATA_SIZE_BYTES)), cluster_coloring_0);
        
    //     // w3.forward(x)
    //     gemv(hbm_addr(in_token_offset), hbm_addr(expert_w3_weights_offset + (dim * inter_dim * i_expert * DATA_SIZE_BYTES)), hbm_addr(temp_token_1), dim, n_token, inter_dim, hbm_addr(expert_w3_bias_offset + (inter_dim * i_expert * DATA_SIZE_BYTES)), cluster_coloring_1);
    //     flex_global_barrier_xy();
        
    //     gemv(hbm_addr(temp_token_0), hbm_addr(expert_w2_weights_offset + (inter_dim * dim * i_expert * DATA_SIZE_BYTES)), hbm_addr(temp_token_0), inter_dim, n_token, dim, hbm_addr(expert_w2_bias_offset + (dim * i_expert * DATA_SIZE_BYTES)), cluster_all);
    //     flex_global_barrier_xy();
        
    //     // TEST
    //     // i_expert++;

    //     i++;
    // }
        
    // // Shared experts
    // for (int i_expert = n_routed_experts; i_expert < (n_routed_experts + n_shared_experts); i_expert++) {
    //     // w1.forward(x)
    //     flex_global_barrier_xy();
    //     gemv(hbm_addr(in_token_offset), hbm_addr(expert_w1_weights_offset + (dim * inter_dim * i_expert * DATA_SIZE_BYTES)), hbm_addr(temp_token_0), dim, n_token, inter_dim, hbm_addr(expert_w1_bias_offset + (inter_dim * i_expert * DATA_SIZE_BYTES)), cluster_coloring_0);
        
    //     // w3.forward(x)
    //     gemv(hbm_addr(in_token_offset), hbm_addr(expert_w3_weights_offset + (dim * inter_dim * i_expert * DATA_SIZE_BYTES)), hbm_addr(temp_token_1), dim, n_token, inter_dim, hbm_addr(expert_w3_bias_offset + (inter_dim * i_expert * DATA_SIZE_BYTES)), cluster_coloring_1);
    //     flex_global_barrier_xy();
        
    //     flex_global_barrier_xy();
    //     // w2.forward(silu(w1.forward(x)) * w3.forward(x))
    //     gemv(hbm_addr(temp_token_0), hbm_addr(expert_w2_weights_offset + (inter_dim * dim * i_expert * DATA_SIZE_BYTES)), hbm_addr(temp_token_0), inter_dim, n_token, dim, hbm_addr(expert_w2_bias_offset + (dim * i_expert * DATA_SIZE_BYTES)), cluster_all);
    //     flex_global_barrier_xy();
    // }
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
                printf("0x%04x ", ((uint16_t *)(hbm_addr(actual_out_offset)))[j + i * dim]);
            }
            printf("\n");
        }
        
        printf("golden:\n");
        for (int i = 0; i < n_token; i++) {
            printf("    ");
            // for (int j = 0; j < dim; j++) {
            for (int j = 0; j < 32; j++) {
                printf("0x%04x ", ((uint16_t *)(hbm_addr(golden_out_offset)))[j + i * dim]);
            }
            printf("\n");
        }
    }
#endif
    flex_global_barrier_xy();
    flex_eoc(eoc_val);
    return 0;
}
