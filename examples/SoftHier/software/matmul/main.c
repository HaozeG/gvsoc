#include <math.h>
#include "flex_runtime.h"
#include "flex_redmule.h"
#include "flex_printf.h"
#include "flex_cluster_arch.h"
#include "flex_dma_pattern.h"
#include "moe.h"

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
     * Expert
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
    uint32_t in_token_offset =            0;
    uint32_t gate_weights_offset =        in_token_offset + n_token * dim * 2;
    uint32_t expert_w1_weights_offset =   gate_weights_offset + dim * n_routed_experts * 2;
    uint32_t expert_w1_bias_offset =      expert_w1_weights_offset + dim * inter_dim * (n_routed_experts + n_shared_experts) * 2;
    uint32_t expert_w2_weights_offset =   expert_w1_bias_offset + inter_dim * (n_routed_experts + n_shared_experts) * 2;
    uint32_t expert_w2_bias_offset =      expert_w2_weights_offset + inter_dim * dim * (n_routed_experts + n_shared_experts) * 2;
    uint32_t expert_w3_weights_offset =   expert_w2_bias_offset + dim * (n_routed_experts + n_shared_experts) * 2;
    uint32_t expert_w3_bias_offset =      expert_w3_weights_offset + dim * (n_routed_experts + n_shared_experts) * inter_dim * 2;
    uint32_t actual_out_offset =          expert_w3_bias_offset + inter_dim * (n_routed_experts + n_shared_experts) * 2;
    uint32_t golden_out_offset =          actual_out_offset + n_token * dim * 2;

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
        for (int i = 0; i < n_token; i++) {
            printf("    ");
            // for (int j = 0; j < dim; j++) {
            for (int j = 0; j < 16; j++) {
                printf("0x%04x ", ((uint16_t *)(hbm_addr(gate_weights_offset)))[j + i * dim]);
            }
            printf("\n");
        }
        printf("expert_w1_weights:\n");
        for (int i = 0; i < n_token; i++) {
            printf("    ");
            // for (int j = 0; j < dim; j++) {
            for (int j = 0; j < 16; j++) {
                printf("0x%04x ", ((uint16_t *)(hbm_addr(expert_w1_weights_offset)))[j + i * dim]);
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
        // printf("0x%04x\n", ((uint16_t *)(hbm_addr(in_token_addr)))[0]);
        // printf("0x%04x\n", ((uint16_t *)(hbm_addr(gate_weights_addr)))[0]);
        // printf("0x%04x\n", ((uint16_t *)(hbm_addr(expert_w1_weights_addr)))[0]);
        // printf("0x%04x\n", ((uint16_t *)(hbm_addr(expert_w1_bias_addr)))[0]);
        // printf("0x%04x\n", ((uint16_t *)(hbm_addr(expert_w2_weights_addr)))[0]);
        // printf("0x%04x\n", ((uint16_t *)(hbm_addr(expert_w2_bias_addr)))[0]);
        // printf("0x%04x\n", ((uint16_t *)(hbm_addr(expert_w3_weights_addr)))[0]);
        // printf("0x%04x\n", ((uint16_t *)(hbm_addr(expert_w3_bias_addr)))[0]);
        // printf("0x%04x\n", ((uint16_t *)(hbm_addr(actual_out_addr)))[0]);
        // printf("0x%04x\n", ((uint16_t *)(hbm_addr(golden_out_addr)))[0]);
        
    }

    uint32_t L1_offset = 0;
    if (flex_is_dm_core() && (0 == flex_get_cluster_id())) {
        // volatile uint16_t * local_ptr = (volatile uint16_t *)local(L1_offset);
        // flex_dma_async_1d(local(L1_offset), hbm_addr(in_token_addr), 4);
        // flex_dma_async_wait_all();
        // printf("0x%04x\n", local_ptr[1]);
        // flex_dma_async_1d(local(L1_offset), hbm_addr(actual_out_addr), 4);
        // flex_dma_async_wait_all();
        // printf("0x%04x\n", local_ptr[1]);
    }

    uint32_t eoc_val = 0;
    flex_global_barrier_xy();
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0) {
        printf("[Start MoE Computation]\n");
        flex_timer_start();
    }
    compute_moe(in_token_offset, n_token, dim, inter_dim, n_routed_experts, n_shared_experts, n_activated_experts, gate_weights_offset, expert_w1_weights_offset, expert_w1_bias_offset, expert_w2_weights_offset, expert_w2_bias_offset, expert_w3_weights_offset, expert_w3_bias_offset, actual_out_offset);
    flex_global_barrier_xy();
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0) {
        flex_timer_end();
    }
    
    // get the output
    if (flex_is_first_core() && (flex_get_cluster_id()==0))
    {
        printf("[Check Results] with first 8 elements of each row of the output\n");
        for (int i = 0; i < n_token; i++) {
            printf("    ");
            // for (int j = 0; j < dim; j++) {
            for (int j = 0; j < 8; j++) {
                printf("0x%04x ", ((uint16_t *)(hbm_addr(actual_out_offset)))[j + i * dim]);
            }
            printf("\n");
        }
        // printf("[Check Results] First 3 Row Elements:\n");
        // for (auto C_i = 0; C_i < 3; C_i++) {
        //     for (auto C_j = 0; C_j < M; C_j++) {
        //         show_progress_animated(C_j + C_i * M,M*3);
                // if (((uint16_t *)(hbm_addr(C)))[C_j + C_i * M] != ((uint16_t *)(hbm_addr(G)))[C_j + C_i * M])
        //         {
        //             eoc_val = 1;
        //             printf("\n[Check Error] at (%0d, %0d): expected: 0x%x, but got: 0x%x\n",
        //                 C_i,
        //                 C_j,
        //                 ((uint16_t *)(hbm_addr(G)))[C_j + C_i * M],
        //                 ((uint16_t *)(hbm_addr(C)))[C_j + C_i * M]);
        //             break;
        //         }
        //     }
        //     if (eoc_val == 1)
        //     {
        //         break;
        //     }
        // }

        // if (eoc_val == 0)
        // {
        //     printf("[Check Passed] GEMM is Numerically Correct\n");
        // }
    }
    flex_global_barrier_xy();
    flex_eoc(eoc_val);
    return 0;
}
