#include <math.h>
#include <inttypes.h>
#include "flex_runtime.h"
#include "flex_redmule.h"
#include "flex_printf.h"
#include "flex_cluster_arch.h"
#include "flex_dma_pattern.h"
// #include "moe.h"
// #include "moe_decode_centralized.h"
// #include "moe_decode.h"
#include "moe_prefill.h"
// #define PRINT_DEBUG 0

int main();
int main(){
    flex_barrier_xy_init();
    flex_global_barrier_xy();

    // Parameters below follows the configuration of MoE model used in preload
    uint64_t n_token = 512;
    // uint16_t dim = 1024;
    uint64_t dim = 7168;
    // uint16_t inter_dim = 512;
    uint64_t inter_dim = 2048;
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
    // Read the value preloaded into HBM
#ifdef MOE_DECODE_CENTRALIZED_H
    uint32_t in_token_offset =            0;
    uint32_t gate_weights_offset =        in_token_offset + n_token * dim * DATA_SIZE_BYTES;
    uint32_t expert_w1_weights_offset =   gate_weights_offset + dim * n_routed_experts * DATA_SIZE_BYTES;
    uint32_t expert_w1_bias_offset =      expert_w1_weights_offset + dim * inter_dim * (n_routed_experts + n_shared_experts) * DATA_SIZE_BYTES;
    uint32_t expert_w2_weights_offset =   expert_w1_bias_offset + inter_dim * (n_routed_experts + n_shared_experts) * DATA_SIZE_BYTES;
    uint32_t expert_w2_bias_offset =      expert_w2_weights_offset + inter_dim * dim * (n_routed_experts + n_shared_experts) * DATA_SIZE_BYTES;
    uint32_t expert_w3_weights_offset =   expert_w2_bias_offset + dim * (n_routed_experts + n_shared_experts) * DATA_SIZE_BYTES;
    uint32_t expert_w3_bias_offset =      expert_w3_weights_offset + dim * (n_routed_experts + n_shared_experts) * inter_dim * DATA_SIZE_BYTES;
    uint32_t actual_out_offset =          expert_w3_bias_offset + inter_dim * (n_routed_experts + n_shared_experts) * DATA_SIZE_BYTES;
    uint32_t golden_out_offset =          actual_out_offset + n_token * dim * DATA_SIZE_BYTES;
#endif

#ifdef MOE_DECODE_H
    // // W1 stored at the beginning of channel 4, 5, 6, 7
    // uint64_t expert_w1_weights_offset = 0;
    // uint64_t expert_w1_bias_offset = expert_w1_weights_offset + dim * inter_dim * (n_routed_experts + n_shared_experts) * DATA_SIZE_BYTES;

    // // W3 stored at the beginning of channel 0, 1, 2, 3
    // uint64_t expert_w3_weights_offset = 0;
    // uint64_t expert_w3_bias_offset = expert_w3_weights_offset + dim * inter_dim * (n_routed_experts + n_shared_experts) * DATA_SIZE_BYTES;

    // // W2 stored in all channels following the W1/W3
    // uint64_t expert_w2_weights_offset = expert_w1_bias_offset + inter_dim * (n_routed_experts + n_shared_experts) * DATA_SIZE_BYTES;
    // uint64_t expert_w2_bias_offset = expert_w2_weights_offset + inter_dim * dim * (n_routed_experts + n_shared_experts) * DATA_SIZE_BYTES;

    // // Gate weights stored in all channels following the W2 bias
    // uint64_t gate_weights_offset = expert_w2_bias_offset + dim * (n_routed_experts + n_shared_experts) * DATA_SIZE_BYTES;

    // // All the other data stored in channel 0, following the gate weights
    // uint64_t in_token_offset = gate_weights_offset + dim * n_routed_experts * DATA_SIZE_BYTES;
    // uint64_t actual_out_offset = in_token_offset + n_token * dim * DATA_SIZE_BYTES;
    // uint64_t golden_out_offset = actual_out_offset + n_token * dim * DATA_SIZE_BYTES;

    /* --- */
    uint64_t expert_w1_weights_offset = 0;
    uint64_t expert_w1_bias_offset = expert_w1_weights_offset + (dim * inter_dim * (n_routed_experts + n_shared_experts) * DATA_SIZE_BYTES / ARCH_NUM_CLUSTER_Y);
    uint64_t expert_w3_weights_offset = 0;
    uint64_t expert_w3_bias_offset = expert_w3_weights_offset + dim * inter_dim * (n_routed_experts + n_shared_experts) * DATA_SIZE_BYTES / ARCH_NUM_CLUSTER_X;

    uint64_t expert_w2_weights_offset = expert_w1_bias_offset + inter_dim * (n_routed_experts + n_shared_experts) * DATA_SIZE_BYTES / ARCH_NUM_CLUSTER_Y;
    uint64_t expert_w2_bias_offset = expert_w2_weights_offset + inter_dim * dim * (n_routed_experts + n_shared_experts) * DATA_SIZE_BYTES / ARCH_NUM_CLUSTER_Y;
    uint64_t gate_weights_offset = expert_w2_bias_offset + dim * (n_routed_experts + n_shared_experts) * DATA_SIZE_BYTES / ARCH_NUM_CLUSTER_Y;
    uint64_t in_token_offset = gate_weights_offset + dim * n_routed_experts * DATA_SIZE_BYTES;
    uint64_t actual_out_offset = in_token_offset + n_token * dim * DATA_SIZE_BYTES;
    uint64_t golden_out_offset = actual_out_offset + n_token * dim * DATA_SIZE_BYTES;
    #endif
#ifdef MOE_PREFILL_H
    uint64_t expert_w1_weights_offset = 0;
    uint64_t expert_w1_bias_offset = expert_w1_weights_offset + (dim * inter_dim * (n_routed_experts + n_shared_experts) * DATA_SIZE_BYTES / ARCH_NUM_CLUSTER_Y);
    uint64_t expert_w3_weights_offset = 0;
    uint64_t expert_w3_bias_offset = expert_w3_weights_offset + dim * inter_dim * (n_routed_experts + n_shared_experts) * DATA_SIZE_BYTES / ARCH_NUM_CLUSTER_X;

    uint64_t expert_w2_weights_offset = expert_w1_bias_offset + inter_dim * (n_routed_experts + n_shared_experts) * DATA_SIZE_BYTES / ARCH_NUM_CLUSTER_Y;
    uint64_t expert_w2_bias_offset = expert_w2_weights_offset + inter_dim * dim * (n_routed_experts + n_shared_experts) * DATA_SIZE_BYTES / ARCH_NUM_CLUSTER_Y;
    uint64_t gate_weights_offset = expert_w2_bias_offset + dim * (n_routed_experts + n_shared_experts) * DATA_SIZE_BYTES / ARCH_NUM_CLUSTER_Y;
    // in_token, actual_out, golden_out distributed among west HBM nodes
    uint64_t in_token_offset = gate_weights_offset + dim * n_routed_experts * DATA_SIZE_BYTES / ARCH_NUM_CLUSTER_Y;
    uint64_t actual_out_offset = in_token_offset + n_token * dim * DATA_SIZE_BYTES / ARCH_NUM_CLUSTER_Y;
    uint64_t golden_out_offset = actual_out_offset + n_token * dim * DATA_SIZE_BYTES / ARCH_NUM_CLUSTER_Y;
#endif

#ifdef PRINT_DEBUG
    if (flex_is_first_core() && (0 == flex_get_cluster_id()))
    {
        printf("HBM_NODE_ADDR_SPACE: 0x%08x%08x\n", (uint32_t)(ARCH_HBM_NODE_ADDR_SPACE >> 32), (uint32_t)(ARCH_HBM_NODE_ADDR_SPACE & 0xFFFFFFFF));
        printf("[Check Preload] Addresses:\n");
        printf("in_token: 0x%08x%08x\n", (uint32_t)((in_token_offset) >> 32), (uint32_t)((in_token_offset) & 0xFFFFFFFF));
        printf("gate_weights: 0x%08x%08x\n", (uint32_t)((gate_weights_offset) >> 32), (uint32_t)((gate_weights_offset) & 0xFFFFFFFF));
        printf("expert_w1_weights: 0x%08x%08x\n", (uint32_t)((expert_w1_weights_offset) >> 32), (uint32_t)((expert_w1_weights_offset) & 0xFFFFFFFF));
        printf("expert_w1_bias: 0x%08x%08x\n", (uint32_t)((expert_w1_bias_offset) >> 32), (uint32_t)((expert_w1_bias_offset) & 0xFFFFFFFF));
        printf("expert_w2_weights: 0x%08x%08x\n", (uint32_t)((expert_w2_weights_offset) >> 32), (uint32_t)((expert_w2_weights_offset) & 0xFFFFFFFF));
        printf("expert_w2_bias: 0x%08x%08x\n", (uint32_t)((expert_w2_bias_offset) >> 32), (uint32_t)((expert_w2_bias_offset) & 0xFFFFFFFF));
        printf("actual_out: 0x%08x%08x\n", (uint32_t)((actual_out_offset) >> 32), (uint32_t)((actual_out_offset) & 0xFFFFFFFF));

        printf("n_token: %d\n", (uint32_t)n_token);
        printf("dim: %d\n", (uint32_t)dim);
        printf("inter_dim: %d\n", (uint32_t)inter_dim);
        printf("n_routed_experts: %d\n", (uint32_t)n_routed_experts);
    }
    // if (flex_is_dm_core() && flex_get_cluster_id() == 0){
    //     uint64_t offset = (uint64_t)ARCH_HBM_NODE_ADDR_SPACE * (2 * ARCH_NUM_CLUSTER_Y + ARCH_NUM_CLUSTER_X + 1);
    //     uint32_t expert_w1_weights_tcdm = 0;
    //     for (int i = 0; i < 16; i++) {
    //         for (int j = 0; j < inter_dim / 4; j++) {
    //             ((uint16_t *)(local(expert_w1_weights_tcdm)))[j + i * inter_dim / 4] = (uint16_t)0x3c00 + j * 4;
    //         }
    //     }
    //     flex_dma_async_1d( hbm_addr(expert_w1_weights_offset) + offset, local(expert_w1_weights_tcdm), dim * inter_dim * DATA_SIZE_BYTES / 4);
    //     flex_dma_async_wait_all();
    // }
    // First element of each matrix
    if (flex_is_first_core() && (flex_get_cluster_id()==0))
    {
        ((uint16_t *)(hbm_addr(in_token_offset)))[1 + 0 * dim] = (uint16_t)0x3c00;
        ((uint16_t *)(hbm_addr(in_token_offset)))[10 + 0 * dim] = (uint16_t)0x3c00;
        ((uint16_t *)(hbm_addr(gate_weights_offset)))[0 + 0 * dim] = (uint16_t)0x4000;
        ((uint16_t *)(hbm_addr(gate_weights_offset)))[1 + 0 * dim] = (uint16_t)0x3c00;
        ((uint16_t *)(hbm_addr(gate_weights_offset)))[2 + 0 * dim] = (uint16_t)0x4000;
        ((uint16_t *)(hbm_addr(gate_weights_offset)))[0 + 1 * n_routed_experts] = (uint16_t)0x3c00;
        // for (int i = 0; i < dim; i++) {
        for (int i = 0; i < 128; i++) {
            for (int j = 0; j < 16; j++) {
                ((uint16_t *)(hbm_addr(expert_w1_weights_offset)))[j + i * dim] = (uint16_t)0x3c00;
            }
        }
        ((uint16_t *)(hbm_addr(in_token_offset)))[3 + 0 * dim] = (uint16_t)0x3c00;
        ((uint16_t *)(hbm_addr(in_token_offset)))[32 + 0 * dim] = (uint16_t)0x3c00;
        ((uint16_t *)(hbm_addr(in_token_offset)))[0 + 2 * dim] = (uint16_t)0x3c00;
        // for (int i = 0; i < n_routed_experts; i++) {
        //     for (int j = 0; j < dim/8; j++) {
        //         ((uint16_t *)(hbm_addr(gate_weights_offset)))[j + i * dim] = (uint16_t)0x4000;
        //     }
        // }

        // for (int i = 0; i < dim; i++) {
        //     for (int j = 0; j < inter_dim/4; j++) {
        //         ((uint16_t *)(hbm_addr(expert_w3_weights_offset)))[j + i * inter_dim] = (uint16_t)0x3c00;
        //     }
        // }
        // for (int i = 0; i < inter_dim; i++) {
        //     for (int j = 0; j < dim/8; j++) {
        //         ((uint16_t *)(hbm_addr(expert_w2_weights_offset)))[j + i * inter_dim] = (uint16_t)0x3c00;
        //     }
        // }

        // printf("[Check Preload] Data\n");
        printf("[Check Preload] with first 8 elements of each row of the input\n");
        printf("in_token:\n");
        for (int i = 0; i < n_token; i++) {
            printf("    ");
            // for (int j = 0; j < dim; j++) {
            for (int j = 0; j < 16; j++) {
                uint16_t * in_token_ptr = (uint16_t *)(hbm_addr(in_token_offset));
                printf("0x%04x ", (uint16_t)*(uint64_t *)(hbm_addr(in_token_offset) + (j + i * dim) * DATA_SIZE_BYTES));
            }
            printf("\n");
        }
        printf("gate_weights:\n");
        for (int i = 0; i < 2; i++) {
            printf("    ");
            // for (int j = 0; j < dim; j++) {
            for (int j = 0; j < n_routed_experts; j++) {
                printf("0x%04x ", (uint16_t)*(uint64_t *)(hbm_addr(gate_weights_offset) + (j + i * n_routed_experts) * DATA_SIZE_BYTES));
            }
            printf("\n");
        }
        printf("expert_w1_weights:\n");
        for (int i = 0; i < 1; i++) {
            printf("    ");
            for (int j = 0; j < 16; j++) {
            // for (int j = 0; j < inter_dim; j++) {
                #ifdef MOE_DECODE_H
                    printf("0x%04x ", (uint16_t)*(uint64_t *)(hbm_addr(expert_w1_weights_offset) + (j + i * inter_dim) * DATA_SIZE_BYTES));
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
                printf("0x%04x ", (uint16_t)*(uint64_t *)(hbm_addr(expert_w1_bias_offset) + j * DATA_SIZE_BYTES));
            #endif
            #ifdef MOE_DECODE_CENTRALIZED_H
                // printf("0x%04x ", ((uint16_t *)(hbm_addr(expert_w1_bias_offset)))[j]);
                printf("0x%04x ", (uint16_t)(hbm_addr(expert_w1_bias_offset) + j * DATA_SIZE_BYTES));
            #endif
        }
        printf("\n");
        // printf("expert_w3_weights:\n");
        // for (int i = 0; i < n_token; i++) {
        //     printf("    ");
        //     // for (int j = 0; j < inter_dim; j++) {
        //     for (int j = 0; j < 16; j++) {
        //         // printf("0x%04x ", ((uint16_t *)(hbm_addr(expert_w3_weights_offset)))[j + i * inter_dim]);
        //         printf("0x%04x ", (uint16_t)*(uint64_t *)(hbm_addr(expert_w3_weights_offset) + (j + i * inter_dim) * DATA_SIZE_BYTES));
        //     }
        //     printf("\n");
        // }
        // printf("actual_out:\n");
        // for (int i = 0; i < n_token; i++) {
        //     printf("    ");
        //     // for (int j = 0; j < dim; j++) {
        //     for (int j = 0; j < 32; j++) {
        //         // printf("0x%04x ", ((uint16_t *)(hbm_addr(actual_out_offset)))[j + i * dim]);
        //         printf("0x%04x ", (uint16_t)*(uint64_t *)(hbm_addr(actual_out_offset) + (j + i * dim) * DATA_SIZE_BYTES));
        //     }
        //     printf("\n");
        // }
        // printf("golden_out:\n");
        // for (int i = 0; i < n_token; i++) {
        //     printf("    ");
        //     // for (int j = 0; j < dim; j++) {
        //     for (int j = 0; j < 16; j++) {
        //         // printf("0x%04x ", ((uint16_t *)(hbm_addr(golden_out_offset)))[j + i * dim]);
        //         printf("0x%04x ", (uint16_t)*(uint64_t *)(hbm_addr(golden_out_offset) + (j + i * dim) * DATA_SIZE_BYTES));
        //     }
        //     printf("\n");
        // }        
    }
#endif

    uint32_t eoc_val = 0;
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0) {
        printf("[Start MoE Computation]\n");
        flex_timer_start();
    }
    flex_global_barrier_xy();
#ifdef MOE_H
    // if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0) {
    //     printf("[Start MoE Computation]\n");
    // }
    // compute_moe(in_token_offset, n_token, dim, inter_dim, n_routed_experts, n_shared_experts, n_activated_experts, gate_weights_offset, expert_w1_weights_offset, expert_w1_bias_offset, expert_w2_weights_offset, expert_w2_bias_offset, expert_w3_weights_offset, expert_w3_bias_offset, actual_out_offset);
    top_k((in_token_offset), (actual_out_offset), (actual_out_offset + DATA_SIZE_BYTES * n_activated_experts), n_activated_experts, n_routed_experts, n_token);
#endif

#ifdef MOE_DECODE_CENTRALIZED_H
    compute_moe(in_token_offset, n_token, dim, inter_dim, n_routed_experts, n_shared_experts, n_activated_experts, gate_weights_offset, expert_w1_weights_offset, expert_w1_bias_offset, expert_w2_weights_offset, expert_w2_bias_offset, expert_w3_weights_offset, expert_w3_bias_offset, actual_out_offset);
    cluster_map_t activated_cluster;
    // activated_cluster = 0x5A5A;
    // gemv(hbm_addr(in_token_offset), hbm_addr(expert_w1_weights_offset), hbm_addr(actual_out_offset), dim, 1, inter_dim, hbm_addr(expert_w1_bias_offset), activated_cluster);
    // activated_cluster = 0xA5A5;
    // gemv(hbm_addr(in_token_offset), hbm_addr(expert_w3_weights_offset), hbm_addr(actual_out_offset), dim, 1, inter_dim, hbm_addr(expert_w3_bias_offset), activated_cluster);
    // activated_cluster = 0xFFFF;
    // gemv(hbm_addr(in_token_offset), hbm_addr(expert_w2_weights_offset), hbm_addr(actual_out_offset), inter_dim, 1, dim, hbm_addr(expert_w2_bias_offset), activated_cluster);
#endif
    
#ifdef MOE_DECODE_H
    compute_moe(in_token_offset, n_token, dim, inter_dim, n_routed_experts, n_shared_experts, n_activated_experts, gate_weights_offset, expert_w1_weights_offset, expert_w1_bias_offset, expert_w2_weights_offset, expert_w2_bias_offset, expert_w3_weights_offset, expert_w3_bias_offset, actual_out_offset);
    // cluster_map_t activated_cluster;
    // activated_cluster = 0x5A5A;
    // gemv(hbm_addr(in_token_offset), hbm_addr(expert_w1_weights_offset), hbm_addr(actual_out_offset), dim, 1, inter_dim, hbm_addr(expert_w1_bias_offset), activated_cluster, TILE_WIDTH_EXPERT_0);
    // activated_cluster = 0xA5A5;
    // gemv(hbm_addr(in_token_offset), hbm_addr(expert_w3_weights_offset), hbm_addr(actual_out_offset), dim, 1, inter_dim, hbm_addr(expert_w3_bias_offset), activated_cluster, TILE_WIDTH_EXPERT_0);
    // activated_cluster = 0xFFFF;
    // gemv(hbm_addr(in_token_offset), hbm_addr(expert_w2_weights_offset), hbm_addr(actual_out_offset), inter_dim, 1, dim, hbm_addr(expert_w2_bias_offset), activated_cluster);
    // activated_cluster = 0xFFFF;
    // top_k(hbm_addr(in_token_offset), hbm_addr(actual_out_offset), hbm_addr(actual_out_offset + 128), n_activated_experts, n_routed_experts, n_token, activated_cluster);
    // gemv(hbm_addr(in_token_offset), hbm_addr(gate_weights_offset), hbm_addr(actual_out_offset), dim, n_token, n_routed_experts, zomem(0), activated_cluster);
    // flex_global_barrier_xy();
    // gemv(hbm_addr(in_token_offset), hbm_addr(expert_w1_weights_offset), hbm_addr(actual_out_offset), 7168, 1, 2048, hbm_addr(expert_w1_bias_offset), activated_cluster);
    // activated_cluster = 0x0033;
    // gemv(hbm_addr(in_token_offset), hbm_addr(expert_w1_weights_offset), hbm_addr(actual_out_offset), 7168, 1, 2048, hbm_addr(expert_w2_bias_offset), activated_cluster);
    // gemv(hbm_addr(in_token_offset), hbm_addr(expert_w1_weights_offset), hbm_addr(actual_out_offset), 2048, 1, 7168, hbm_addr(expert_w2_bias_offset), activated_cluster);
    // silu(hbm_addr(in_token_offset), hbm_addr(actual_out_offset), 8, 1, activated_cluster);
    // sigmoid(hbm_addr(in_token_offset), hbm_addr(actual_out_offset), 8, 1, activated_cluster);
    // dot_product(hbm_addr(in_token_offset), hbm_addr(in_token_offset), hbm_addr(actual_out_offset), 2048, 1, activated_cluster);
    // add(hbm_addr(in_token_offset), hbm_addr(in_token_offset), hbm_addr(actual_out_offset), inter_dim, 1, activated_cluster);
    // fp16 in_const = 0x3c00;
    // dot_product_const(hbm_addr(in_token_offset), in_const, hbm_addr(actual_out_offset), 8, 1, activated_cluster);
    // add(hbm_addr(in_token_offset), hbm_addr(gate_weights_offset), hbm_addr(actual_out_offset), 7168, 1, activated_cluster);
    // normalize(hbm_addr(in_token_offset), hbm_addr(actual_out_offset), n_activated_experts, 2, activated_cluster);
    // top_k(hbm_addr(in_token_offset), hbm_addr(actual_out_offset+1024), hbm_addr(actual_out_offset), 8, 256, 1, activated_cluster);
    // top_k(hbm_addr(in_token_offset), hbm_addr(actual_out_offset), hbm_addr(actual_out_offset + 1024), 8, 200, 1, activated_cluster);
    // fp16 in1 = 0x9633;
    // fp16 in2 = 0x15a4;
    // fp16 out;
    // mul_op(&in1, &in2, &out);
    // if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0) {
    //     printf("0x%04x * 0x%04x = 0x%04x\n", in1, in2, out);
    // }
#endif
#ifdef MOE_PREFILL_H
    compute_moe(in_token_offset, n_token, dim, inter_dim, n_routed_experts, n_shared_experts, n_activated_experts, gate_weights_offset, expert_w1_weights_offset, expert_w1_bias_offset, expert_w2_weights_offset, expert_w2_bias_offset, expert_w3_weights_offset, expert_w3_bias_offset, actual_out_offset);
#endif
    flex_global_barrier_xy();
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0) {
        flex_timer_end();
    }
    
#ifdef PRINT_DEBUG
    // get the output
    uint64_t out_buffer = 0;
    if (flex_is_dm_core() && (0 == flex_get_cluster_id())) {
        flex_dma_async_1d(local(out_buffer), hbm_west((uint64_t)3, actual_out_offset), n_routed_experts * 128* DATA_SIZE_BYTES);
        flex_dma_async_wait_all();
    }
    flex_intra_cluster_sync();
    if (flex_is_first_core() && (flex_get_cluster_id()==0))
    {
        printf("[Check Results]\n");
        // printf("actual_out:\n");
        // // for (int i = 0; i < (n_token >> 2)*n_routed_experts; i++) {
        // for (int i = 0; i < 3; i++) {
        //     printf("    ");
        //     // for (int j = 0; j < dim; j++) {
        //     for (int j = 0; j < 32; j++) {
        //         // printf("0x%04x ", ((uint16_t *)(hbm_addr(actual_out_offset)))[j + i * dim]);
        //         printf("0x%04x ", (uint16_t)*(uint64_t *)(hbm_addr(actual_out_offset) + (j + i * dim) * DATA_SIZE_BYTES));
        //     }
        //     printf("\n");
        // }
        printf("out_index:\n");
        for (int i = 0; i < 16; i++) {
            printf("    ");
            // for (int j = 0; j < dim; j++) {
            for (int j = 0; j < 128; j++) {
                printf("0x%04x ", ((uint16_t *)(out_buffer))[j + i * 128]);
            }
            printf("\n");
        }
    }
#endif
    flex_global_barrier_xy();
    flex_eoc(eoc_val);
    return 0;
}
