#ifndef MOE_PREFILL_H
#define MOE_PREFILL_H

#include "utils_decode.h"
#include "element_wise.h"
#include "gemv.h"
#include "top_k.h"
#include "normalize.h"
#include "gemm_systolic_wise.h"

#define PRINT_DEBUG_CLUSTER_ID 0
// use float16 as data type
#define DTYPE fp16
#define DATA_SIZE_BYTES 2
#define FP16_ZERO 0x0000
fp16 INDICE_NOT_USED = 0xffff;

// Parameters for GEMM
#define GEMM_TILE_WIDTH_M 128
#define GEMM_TILE_WIDTH_N 256
#define GEMM_TILE_WIDTH_K 64
#define SEQ_LEN GEMM_TILE_WIDTH_M * ARCH_NUM_CLUSTER_Y

typedef struct {
    uint64_t curr_HBM_node[256];
    uint32_t token_cnt[256];
    uint64_t value_offset[256];
    uint64_t index_offset[256];
} ExpertOffset;

void compute_moe(uint64_t in_token_offset, uint64_t n_token, uint64_t dim, uint64_t inter_dim, uint64_t n_routed_experts, uint64_t n_shared_experts, uint64_t n_activated_experts, uint64_t gate_weights_addr, uint64_t expert_w1_weights_addr, uint64_t expert_w1_bias_addr, uint64_t expert_w2_weights_addr, uint64_t expert_w2_bias_addr, uint64_t expert_w3_weights_addr, uint64_t expert_w3_bias_addr, uint64_t actual_out_addr);

void top_k_weighted_permute(const uint16_t dim, const uint16_t k, const uint16_t n_routed_expert, const uint16_t n_token_per_cluster, const uint64_t in_weight_addr, const uint64_t in_token_offset, const uint64_t base_out_token_offset, const uint64_t base_out_index_offset, ExpertOffset* expert_offset, cluster_map_t cluster_map);
void unpermute_sum(const uint16_t dim, const uint16_t n_token_per_cluster, const uint16_t n_expert, uint64_t out_offset, uint64_t base_token_offset, uint64_t base_index_offset, ExpertOffset *expert_offset, cluster_map_t cluster_map);


/**
 * @brief Select top k values from each row of the input matrix, normalize along each token, and gather the resulting weighted input token to the output matrix along with the corresponding indices. Input matrix is already in TCDM.
 * 
 * @param dim dimension of the token
 * @param k top k values to select (less than 256)
 * @param n_routed_expert number of candidate values in each row of the input matrix
 * @param n_token_per_cluster number of rows to process for each activated cluster
 * @param in_weight_addr in TCDM
 * @param in_token_offset in HBM
 * @param base_out_token_offset in HBM
 * @param base_out_index_offset in HBM
 * @param expert_offset offset of tokens for each expert in HBM
 * @param cluster_map requires activated clusters to be the whole colomn of clusters on west edge
 */
void top_k_weighted_permute(const uint16_t dim, const uint16_t k, const uint16_t n_routed_expert, const uint16_t n_token_per_cluster, const uint64_t in_weight_addr, const uint64_t in_token_offset, const uint64_t base_out_token_offset, const uint64_t base_out_index_offset, ExpertOffset* expert_offset, cluster_map_t cluster_map) {
    if (0 == k || 0 == n_routed_expert || 0 == n_token_per_cluster || k > 256) {
        return;
    }
    // flex_global_barrier_xy();
    
    uint32_t cluster_id = flex_get_cluster_id();
    uint32_t core_id = flex_get_core_id();
    // cluster_id among activated clusters
    uint32_t local_cluster_id = ARCH_NUM_CLUSTER;
    uint32_t n_cluster_activated = 0;
    uint32_t i_row_cluster;
    
    uint32_t local_out_value, local_out_indices, local_in_weight, local_in_token, local_in_token_weighted, local_token_index, local_token_count, local_sum, n_tokens_per_expert, local_const_1;
    local_out_value = ARCH_CLUSTER_TCDM_SIZE - n_token_per_cluster * k * DATA_SIZE_BYTES;
    local_out_indices = local_out_value - n_token_per_cluster * k * DATA_SIZE_BYTES;
    local_token_index = local_out_indices - DATA_SIZE_BYTES;
    local_token_count = local_token_index - sizeof(ExpertOffset);
    local_in_token = local_token_count - n_token_per_cluster * dim * DATA_SIZE_BYTES;
    local_in_token_weighted = local_in_token - dim * DATA_SIZE_BYTES;
    local_sum =  local_in_token_weighted - DATA_SIZE_BYTES;
    n_tokens_per_expert = local_sum - n_routed_expert * DATA_SIZE_BYTES;
    local_const_1 = n_tokens_per_expert - SPATZ_VL * DATA_SIZE_BYTES;

    uint16_t * n_tokens_per_expert_ptr = (uint16_t *)n_tokens_per_expert;
    // curr_token_count records the information of destination offsets in HBM when processing current tokens in this cluster
    ExpertOffset *curr_token_count = (ExpertOffset *)local(local_token_count);
    local_in_weight = in_weight_addr;

    // check TCDM offsets and inputs
    // if (cluster_id == 0 && core_id == 0) {
    //     printf("[TOP_K] n_tokens_per_expert_ptr = 0x%x\n", n_tokens_per_expert_ptr);
    //     printf("[TOP_K] local_out_value = 0x%x\n", local_out_value);
    //     printf("[TOP_K] local_out_indices = 0x%x\n", local_out_indices);
    //     printf("[TOP_K] local_in_weight = 0x%x\n", local_in_weight);
    //     printf("[TOP_K] local_in_token = 0x%x\n", local_in_token);
    //     printf("[TOP_K] local_token_index = 0x%x\n", local_token_index);
    //     printf("[TOP_K] local_token_count = 0x%x\n", local_token_count);
    //     printf("[TOP_K] local_sum = 0x%x\n", local_sum);
    //     printf("[TOP_K] n_routed_expert = %d\n", n_routed_expert);
    //     printf("[TOP_K] n_token_per_cluster = %d\n", n_token_per_cluster);
    //     printf("[TOP_K] dim = %d\n", dim);
    //     printf("[TOP_K] local_const_1 = 0x%x\n", local_const_1);
    // }
    // flex_global_barrier_xy();

    // update n_cluster_activated information to all clusters for later global barrier sync
    for (int i = 0; i < ARCH_NUM_CLUSTER; i++) {
        if ((cluster_map & (0x01 << i)) != 0) {
            n_cluster_activated += 1;
        }
    }
    if ((cluster_map & (0x01 << cluster_id)) != 0) {
        local_cluster_id = 0;
        for (int i = 0; i < cluster_id; i++) {
            if ((cluster_map & (0x01 << i)) != 0) {
                local_cluster_id++;
            }
        }
        
        i_row_cluster = core_id;
        // Allocate an array for indices
        uint16_t indices[256]; // 256 should be large enough for n_routed_expert
        while (i_row_cluster < n_token_per_cluster) {
            // if (core_id == 0 && cluster_id == 0) {
            //     printf("[TOP_K] i_row_cluster = %d\n", i_row_cluster);
            // }
            // Initialize index array
            for (int i = 0; i < n_routed_expert; i++) {
                indices[i] = i;
            }
            
            // Quickselect to partition the array so that the k largest elements are at the beginning
            quickselect((DTYPE *)local(local_in_weight + i_row_cluster * n_routed_expert * DATA_SIZE_BYTES), indices, 0, n_routed_expert - 1, k - 1);
        
            // Copy to buffers
            for (int i = 0; i < k; i++) {
                ((DTYPE *)local(local_out_value + i_row_cluster * k * DATA_SIZE_BYTES))[i] = ((DTYPE *)local(local_in_weight + i_row_cluster * n_routed_expert * DATA_SIZE_BYTES))[i];
                ((uint16_t *)local(local_out_indices + i_row_cluster * k * DATA_SIZE_BYTES))[i] = indices[i];
            }
            i_row_cluster += ARCH_NUM_CORE_PER_CLUSTER;
        }
        flex_intra_cluster_sync();
        // check quick sort results
        // if (0 == local_cluster_id && 0 == core_id) {
        //     for (int i = 0; i < 8; i++) {
        //         for (int j = 0; j < k; j++) {
        //             printf("[TOP_K] local_out_value[%d][%d] = 0x%x\n", i, j, ((fp16 *)local(local_out_value))[i * k + j]);
        //             printf("[TOP_K] local_out_indices[%d][%d] = 0x%x\n", i, j, ((fp16 *)local(local_out_indices))[i * k + j]);
        //         }
        //     }
        // }

        // Sigmoid the output values
        #ifdef SPATZ_ENABLE
        if (0 == core_id) {
            uint16_t * local_in_ptr = (uint16_t *)local(local_out_value);
            uint16_t * local_const_1_ptr = (uint16_t *)local(local_const_1);
            uint16_t * local_out_ptr = (uint16_t *)local(local_out_value);
            for (int i = 0; i < SPATZ_VL; i++) {
                local_const_1_ptr[i] = 0x3c00;
            }
            
            // compute element-wise operation with spatz core
            uint16_t vl;
            uint16_t i_element = 0;
            asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(SPATZ_VL));
            asm volatile("vle16.v v2, (%0)" : : "r"(local_const_1_ptr));
            while ((n_token_per_cluster * k) > i_element) {
                asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(min(SPATZ_VL, n_token_per_cluster * k - i_element)));
                asm volatile("vle16.v v0, (%0)" : : "r"(local_in_ptr));

                // sigmoid: 1 / (1 + exp(-x))
                asm volatile("vfneg.v v3, v0");
                asm_rvv_exp(3,4);
                asm volatile("vfadd.vv v3, v2, v4");
                asm volatile("vfdiv.vv v8, v2, v3");

                asm volatile("vse16.v v8, (%0)" : : "r"(local_out_ptr));

                local_in_ptr += vl;
                local_out_ptr += vl;
                i_element += vl;
            }
        }
        #endif
        flex_intra_cluster_sync();
        // check sigmoid results
        // if (0 == local_cluster_id && 0 == core_id) {
        //     for (int i = 0; i < 8; i++) {
        //         for (int j = 0; j < k; j++) {
        //             printf("[TOP_K] local_out_value[%d][%d] = 0x%x\n", i, j, ((fp16 *)local(local_out_value))[i * k + j]);
        //         }
        //     }
        // }
            
        // Normalize the output values
        // process one row per cluster at a time
        i_row_cluster = 0;
        while (i_row_cluster < n_token_per_cluster) {            
            if (0 == core_id) {
                float sum = 0;
                for (int i = 0; i < k; i++) {
                    // printf("[NORMALIZE] 0x%x\n", ((fp16 *)local(local_out_value))[i]);
                    sum += fp16_to_float(((fp16 *)local(local_out_value + i_row_cluster * k * DATA_SIZE_BYTES))[i]);
                }
                ((fp16 *)local(local_sum))[0] = float_to_fp16(sum);
            }
            flex_intra_cluster_sync();

            // prevent divide by zero
            if (FP16_ZERO != ((fp16 *)local(local_sum))[0]) {
                #ifdef SPATZ_ENABLE
                if (0 == core_id) {
                        DTYPE * local_in_ptr = (DTYPE *)local(local_out_value + i_row_cluster * k * DATA_SIZE_BYTES);
                        DTYPE * local_out_ptr = (DTYPE *)local(local_out_value + i_row_cluster * k * DATA_SIZE_BYTES);
                        
                        asm volatile("vsetvli zero, %0, e16, m8, ta, ma" : : "r"(dim));
                        asm volatile("fmv.h.x ft0, %0" : : "r"(((fp16 *)local(local_sum))[0]));
                        // compute element-wise operation with spatz core
                        uint16_t vl;
                        uint16_t i_element = 0;
                        while (k > i_element) {
                            asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(min(SPATZ_VL, dim - i_element)));
                            asm volatile("vle16.v v0, (%0)" : : "r"(local_in_ptr));
                            asm volatile("vfdiv.vf v1, v0, ft0");
                            asm volatile("vse16.v v1, (%0)" : : "r"(local_out_ptr));
                            
                            local_in_ptr += vl;
                            local_out_ptr += vl;
                            i_element += vl;
                        }
                    }
                #else
                // TODO: now not supported without spatz core
                // uint32_t n_element_per_core = (dim - 1) / ARCH_NUM_CORE_PER_CLUSTER + 1;
                // for (int i = 0; i < n_element_per_core; i++) {
                //     if (i + core_id * n_element_per_core < dim) {
                //         fp16 a = ((fp16 *)local(local_out))[i + core_id * n_element_per_core];
                //         fp16 *b_ptr = (fp16 *)local(local_sum);
                //         fp16 *c_ptr = &((fp16 *)local(local_out))[i + core_id * n_element_per_core];
                //         // if (0 == core_id) {
                //         //     printf("[NORMALIZE] a = 0x%x sum = 0x%x\n", a, *b_ptr);
                //         // }
                //         asm_fp16_div(&a, b_ptr, c_ptr);
                //     }
                // }
                #endif
            }
            i_row_cluster += 1;
        }
        // LOCAL COUNTING: to know how many tokens each expert gets in this cluster
        if (0 == core_id) {
            // initialize to 0
            for (int i = 0; i < n_routed_expert; i++) {
                n_tokens_per_expert_ptr[i] = 0;
            }
            // count the number of tokens for each expert in this cluster
            for (int i = 0; i < n_token_per_cluster; i++) {
                for (int j = 0; j < k; j++) {
                    uint16_t index = ((uint16_t *)local(local_out_indices))[i * k + j];
                    n_tokens_per_expert_ptr[index] += 1;
                }
            }
            // if (0 == cluster_id) {
            //     for (int i = 0; i < n_routed_expert; i++) {
            //         printf("[TOP_K] n_tokens_per_expert_ptr[%d] = %d\n", i, n_tokens_per_expert_ptr[i]);
            //     }
            // }
        }
        // TODO: if atomic operation is supported, we can use atomic operation to do parallel counting
        // uint32_t n_tokens_per_core = (n_token_per_cluster - 1) / ARCH_NUM_CORE_PER_CLUSTER + 1;
        // for (int i = core_id * n_tokens_per_core; i < (core_id + 1) * n_tokens_per_core && i < n_token_per_cluster; i++) {
        //     for (int j = 0; j < k; j++) {
        //         uint16_t index = ((uint16_t *)local(local_out_indices))[i * k + j];
        //         n_tokens_per_expert[index] += 1;
        //     }
        // }
        flex_intra_cluster_sync();
        // Check local counting result
        // if (0 == local_cluster_id && 0 == core_id) {
        //     for (int i = 0; i < n_routed_expert; i++) {
        //         printf("[TOP_K] n_tokens_per_expert_ptr[%d] = %d\n", i, n_tokens_per_expert_ptr[i]);
        //     }
        // }
    }
    flex_global_barrier_xy();

    // GLOBAL COUNTING: to know how many tokens each expert gets in all clusters
    // reduce the result to cluster 0
    if (0 == local_cluster_id && flex_is_dm_core()) {
        flex_dma_async_reduction(n_tokens_per_expert, n_tokens_per_expert, n_routed_expert * sizeof(uint16_t), COLLECTIVE_REDADD_INT_16, 0b11, 0b00);
        flex_dma_async_wait_all();
    }
    flex_global_barrier_xy();
    // broadcast the result to all clusters
    if (0 == local_cluster_id && flex_is_dm_core()) {
        flex_dma_async_broadcast(n_tokens_per_expert, n_tokens_per_expert, n_routed_expert * sizeof(uint16_t), 0b11, 0b00);
        flex_dma_async_wait_all();
    }
    flex_global_barrier_xy();

    // Check global counting result
    // for (int i_cluster = 0; i_cluster < n_cluster_activated; i_cluster++) {
    //     if (local_cluster_id == i_cluster && 0 == core_id) {
    //         printf("[TOP_K] cluster %d\n", cluster_id);
    //         for (int i = 0; i < n_routed_expert; i++) {
    //             printf("[TOP_K] n_tokens_per_expert_ptr[%d] = %d\n", i, n_tokens_per_expert_ptr[i]);
    //         }
    //     }
    //     flex_global_barrier_xy();
    // }

    if (local_cluster_id < n_cluster_activated) {
        if (0 == core_id) {
            for (int i = 0; i < n_routed_expert; i++) {
                curr_token_count->curr_HBM_node[i] = 0;
                curr_token_count->token_cnt[i] = 0;
                curr_token_count->value_offset[i] = 0;
                curr_token_count->index_offset[i] = 0;
            }
        }
        // calculate HBM offset of each expert's token buffer
        // tokens evenly distributed to all west HBM nodes
        if (1 == core_id) {
            expert_offset->value_offset[0] = base_out_token_offset;
            expert_offset->index_offset[0] = base_out_index_offset;
            for (int i = 1; i < n_routed_expert; i++) {
                if (n_tokens_per_expert_ptr[i - 1] != 0) {
                    expert_offset->token_cnt[i - 1] = (n_tokens_per_expert_ptr[i - 1] - 1) / n_cluster_activated + 1;
                    expert_offset->value_offset[i] = expert_offset->value_offset[i - 1] + ((n_tokens_per_expert_ptr[i - 1] - 1) / n_cluster_activated + 1) * dim * DATA_SIZE_BYTES;
                    expert_offset->index_offset[i] = expert_offset->index_offset[i - 1] + ((n_tokens_per_expert_ptr[i - 1] - 1) / n_cluster_activated + 1) * DATA_SIZE_BYTES;
                } else {
                    expert_offset->token_cnt[i - 1] = 0;
                    expert_offset->value_offset[i] = expert_offset->value_offset[i - 1];
                    expert_offset->index_offset[i] = expert_offset->index_offset[i - 1];
                }
            }
            if (n_tokens_per_expert_ptr[n_routed_expert - 1] != 0) {
                expert_offset->token_cnt[n_routed_expert - 1] = (n_tokens_per_expert_ptr[n_routed_expert - 1] - 1) / n_cluster_activated + 1;
            } else {
                expert_offset->token_cnt[n_routed_expert - 1] = 0;
            }
        }
        // load n_token_per_cluster tokens to TCDM
        if (flex_is_dm_core()) {
            flex_dma_async_1d(local(local_in_token), hbm_west((uint64_t)local_cluster_id, in_token_offset), n_token_per_cluster * dim * DATA_SIZE_BYTES);
            flex_dma_async_wait_all();
        }
        flex_intra_cluster_sync();
        // Check expert offset result
        // if (0 == cluster_id && 0 == core_id) {
        //     for (int i = 0; i < n_routed_expert; i++) {
        //         printf("[TOP_K] expert_offset->value_offset[%d] = 0x%x\n", i, expert_offset->value_offset[i]);
        //         printf("[TOP_K] expert_offset->index_offset[%d] = 0x%x\n", i, expert_offset->index_offset[i]);
        //         print64(expert_offset->value_offset[i]);
        //         print64(expert_offset->index_offset[i]);
        //     }
        // }   
    }
    flex_global_barrier_xy();
    
    // Check local token result
    // if (0 == local_cluster_id && 0 == core_id) {
    //     for (int i = 0; i < 8; i++) {
    //         for (int j = 0; j < 8; j++) {
    //             printf("[TOP_K] local_in_token[%d][%d] = 0x%x\n", i, j, ((fp16 *)local(local_in_token))[i * dim + j]);
    //         }
    //     }
    // }

    // GATHER top k values and indices to HBM
    // iterate over tokens in activated clusters, not parallelized
    for (uint16_t i_cluster = 0; i_cluster < n_cluster_activated; i_cluster++) {
        if (local_cluster_id == i_cluster) {
            // if (0 == core_id) {
            //     printf("[TOP_K] cluster %d\n", cluster_id);
            //     printf("[TOP_K] local_cluster_id = %d\n", local_cluster_id);
            //     printf("[TOP_K] n_cluster_activated = %d\n", n_cluster_activated);
            // }
            for (uint32_t i_token = 0; i_token < n_token_per_cluster; i_token++) {
                // if (0 == core_id) {
                //     printf("[TOP_K] i_token = %d\n", i_token);
                // }
                for (uint16_t i_k = 0; i_k < k; i_k++) {
                    uint16_t i_expert = ((uint16_t *)local(local_out_indices))[i_token * k + i_k];
                    uint64_t value_dest_addr = hbm_west(curr_token_count->curr_HBM_node[i_expert], expert_offset->value_offset[i_expert] + curr_token_count->value_offset[i_expert] * dim * DATA_SIZE_BYTES);
                    uint64_t index_dest_addr = hbm_west(curr_token_count->curr_HBM_node[i_expert], expert_offset->index_offset[i_expert] + curr_token_count->index_offset[i_expert] * DATA_SIZE_BYTES);
                    // apply weight to token using spatz core
                    // NOTE: require spatz unit attached to first core
                    if (0 == core_id) {
                        DTYPE w_expert = ((fp16 *)local(local_out_value))[i_token * k + i_k];
                        DTYPE * local_in_ptr = (DTYPE *)local(local_in_token + i_token * dim * DATA_SIZE_BYTES);
                        DTYPE * local_out_ptr = (DTYPE *)local(local_in_token_weighted);
                        uint16_t vl;
                        uint16_t i_element = 0;
                        asm volatile("fmv.h.x ft0, %0" : : "r"(w_expert));
                        while (dim > i_element) {
                            asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(min(SPATZ_VL, dim - i_element)));
                            asm volatile("vle16.v v0, (%0)" : : "r"(local_in_ptr));
                            asm volatile("vfmul.vf v1, v0, ft0");
                            asm volatile("vse16.v v1, (%0)" : : "r"(local_out_ptr));

                            local_in_ptr += vl;
                            local_out_ptr += vl;
                            i_element += vl;
                        }
                    }
                    if (1 == core_id) {
                        // printf("[TOP_K] i_token = %d\n", i_cluster * n_token_per_cluster + i_token);
                        // printf("[TOP_K] curr_token_count->curr_HBM_node[%d] = %d\n", i_expert, curr_token_count->curr_HBM_node[i_expert]);
                        // printf("[TOP_K] curr_token_count->value_offset[%d] = %d\n", i_expert, curr_token_count->value_offset[i_expert]);
                        // printf("[TOP_K] curr_token_count->index_offset[%d] = %d\n", i_expert, curr_token_count->index_offset[i_expert]);
                        // print64(expert_offset->value_offset[i_expert]);
                        // printf("[TOP_K] i_expert = %d\n", i_expert);
                        *(uint16_t *)local(local_token_index) = (uint16_t)(i_cluster * n_token_per_cluster + i_token);
                        // update the curr_token_count information
                        // TODO: can be a smaller data structure
                        curr_token_count->value_offset[i_expert] += 1;
                        curr_token_count->index_offset[i_expert] += 1;
                        // compare with token buffer size of i_expert
                        if ((0 == n_tokens_per_expert_ptr[i_expert]) || (curr_token_count->value_offset[i_expert] >= ((n_tokens_per_expert_ptr[i_expert] - 1) / n_cluster_activated + 1))) {
                            curr_token_count->value_offset[i_expert] = 0;
                            curr_token_count->index_offset[i_expert] = 0;
                            // move to next HBM node if token buffer for current HBM node is full
                            curr_token_count->curr_HBM_node[i_expert] += 1;
                            // printf("[TOP_K] curr_token_count->curr_HBM_node[%d] = %d\n", i_expert, curr_token_count->curr_HBM_node[i_expert]);
                            // printf("[TOP_K] curr_token_count->value_offset[%d] = %d\n", i_expert, curr_token_count->value_offset[i_expert]);
                            // printf("[TOP_K] curr_token_count->index_offset[%d] = %d\n", i_expert, curr_token_count->index_offset[i_expert]);
                            // printf("buffer size = %d\n", ((n_tokens_per_expert_ptr[i_expert] - 1) / n_cluster_activated + 1));
                        }
                    }
                    flex_intra_cluster_sync();
                    // transfer the weighted token to HBM
                    if (flex_is_dm_core()) {
                        flex_dma_async_1d(value_dest_addr, local(local_in_token_weighted), dim * DATA_SIZE_BYTES);
                        flex_dma_async_1d(index_dest_addr, local(local_token_index), DATA_SIZE_BYTES);
                        flex_dma_async_wait_all();
                    }
                    flex_intra_cluster_sync();
                }
            }
            // update the curr_token_count information to next cluster
            if (flex_is_dm_core()) {
                flex_dma_async_1d(remote_pos(top_pos(get_pos(cluster_id)), local_token_count), local_token_count, sizeof(ExpertOffset));
                flex_dma_async_wait_all();
            }
        }
        flex_global_barrier_xy();
    }

    // fill the remaining token buffer with 0 as values and INDICE_NOT_USED as indices
    if (local_cluster_id == (n_cluster_activated - 1)) {
        // if (0 == core_id) {
        //     printf("[TOP_K] cluster %d\n", cluster_id);
        //     printf("[TOP_K] local_cluster_id = %d\n", local_cluster_id);
        // }
        uint16_t *local_token_index_ptr = (uint16_t *)local(local_token_index);
        *local_token_index_ptr = INDICE_NOT_USED;
        for (int i_expert = 0; i_expert < n_routed_expert; i_expert++) {
            // if (0 == core_id) {
            //     printf("[TOP_K] i_expert = %d\n", i_expert);
            //     printf("[TOP_K] num of tokens = %d\n", n_tokens_per_expert_ptr[i_expert]);
            // }
            // check if the token buffer is full
            while ((0 != n_tokens_per_expert_ptr[i_expert]) && (curr_token_count->curr_HBM_node[i_expert] != n_cluster_activated)) {
                uint64_t value_dest_addr = hbm_west(curr_token_count->curr_HBM_node[i_expert], expert_offset->value_offset[i_expert] + curr_token_count->value_offset[i_expert] * dim * DATA_SIZE_BYTES);
                uint64_t index_dest_addr = hbm_west(curr_token_count->curr_HBM_node[i_expert], expert_offset->index_offset[i_expert] + curr_token_count->index_offset[i_expert] * DATA_SIZE_BYTES);
                if (flex_is_dm_core()) {
                    // TODO: value transfer can be ignored
                    flex_dma_async_1d(value_dest_addr, zomem(0), dim * DATA_SIZE_BYTES);
                    flex_dma_async_1d(index_dest_addr, local(local_token_index), DATA_SIZE_BYTES);
                    flex_dma_async_wait_all();
                }
                if (0 == core_id) {
                    // printf("[TOP_K] filling i_expert = %d at node %d\n", i_expert, curr_token_count->curr_HBM_node[i_expert]);
                    curr_token_count->value_offset[i_expert] += 1;
                    curr_token_count->index_offset[i_expert] += 1;
                    // compare with token buffer size of i_expert
                    if ((curr_token_count->value_offset[i_expert] >= expert_offset->token_cnt[i_expert])) {
                        curr_token_count->value_offset[i_expert] = 0;
                        curr_token_count->index_offset[i_expert] = 0;
                        // move to next HBM node
                        curr_token_count->curr_HBM_node[i_expert] += 1;
                        // printf("[TOP_K] curr_token_count->curr_HBM_node[%d] = %d\n", i_expert, curr_token_count->curr_HBM_node[i_expert]);
                        // printf("[TOP_K] curr_token_count->value_offset[%d] = %d\n", i_expert, curr_token_count->value_offset[i_expert]);
                        // printf("[TOP_K] curr_token_count->index_offset[%d] = %d\n", i_expert, curr_token_count->index_offset[i_expert]);
                        // if (0 == n_tokens_per_expert_ptr[i_expert]) {
                        //     printf("buffer size = 0\n");
                        // }
                        // else {
                        //     printf("buffer size = %d\n", ((n_tokens_per_expert_ptr[i_expert] - 1) / n_cluster_activated + 1));
                        // }
                    }
                }
                flex_intra_cluster_sync();
            }
        }
    }
    flex_global_barrier_xy();
}

/**
 * @brief Unpermute tokens assigned to different experts and adds unpermute result to get final prefill result
 * @param dim dimension of one token
 * @param n_expert 
 * @param n_token_per_cluster 
 * @param out_offset store final results
 * @param base_token_offset base offset in HBM that stores permuted tokens
 * @param base_index_offset base offset in HBM that stores permuted tokens' original indexes
 * @param expert_offset offsets pointing to different experts from the base offset of permuted tokens 
 * @param cluster_map activated clusters
 */
void unpermute_sum(const uint16_t dim, const uint16_t n_expert, const uint16_t n_token_per_cluster, uint64_t out_offset, uint64_t base_token_offset, uint64_t base_index_offset, ExpertOffset* expert_offset, cluster_map_t cluster_map) {
    if (0 == expert_offset) {
        return;
    }
    uint32_t cluster_id = flex_get_cluster_id();
    uint32_t core_id = flex_get_core_id();
    // cluster_id among activated clusters
    uint32_t local_cluster_id = ARCH_NUM_CLUSTER;
    uint32_t n_cluster_activated = 0;

    uint64_t local_token_0, local_token_1, local_index_0, local_index_1, local_out;
    local_token_0 = ARCH_CLUSTER_TCDM_SIZE - dim * DATA_SIZE_BYTES;
    local_token_1 = local_token_0 - dim * DATA_SIZE_BYTES;
    local_index_0 = local_token_1 - DATA_SIZE_BYTES;
    local_index_1 = local_index_0 - DATA_SIZE_BYTES;
    local_out = local_index_1 - n_token_per_cluster * dim * DATA_SIZE_BYTES;
    
    // check input parameters
    // if ((0 == core_id) && (0 == cluster_id)) {
    //     printf("[UNPERMUTE_SUM] dim = %d\n", dim);
    //     printf("[UNPERMUTE_SUM] n_expert = %d\n", n_expert);
    //     printf("[UNPERMUTE_SUM] n_token_per_cluster = %d\n", n_token_per_cluster);
    //     printf("[UNPERMUTE_SUM] out_offset = 0x%x\n", out_offset);
    //     printf("[UNPERMUTE_SUM] base_token_offset = 0x%x\n", base_token_offset);
    //     printf("[UNPERMUTE_SUM] base_index_offset = 0x%x\n", base_index_offset);
    //     // expert_offset
    //     for (int i = 0; i < n_expert; i++) {
    //         printf("[UNPERMUTE_SUM] expert_offset->curr_HBM_node[%d] = 0x%x\n", i, expert_offset->curr_HBM_node[i]);
    //         printf("[UNPERMUTE_SUM] expert_offset->token_cnt[%d] = 0x%x\n", i, expert_offset->token_cnt[i]);
    //         printf("[UNPERMUTE_SUM] expert_offset->value_offset[%d] = 0x%x\n", i, expert_offset->value_offset[i]);
    //         printf("[UNPERMUTE_SUM] expert_offset->index_offset[%d] = 0x%x\n", i, expert_offset->index_offset[i]);
    //     }
    //     printf("[UNPERMUTE_SUM] cluster_map = 0x%x\n", cluster_map);
    // }
    // flex_global_barrier_xy();
    for (int i = 0; i < ARCH_NUM_CLUSTER; i++) {
        if ((cluster_map & (0x01 << i)) != 0) {
            n_cluster_activated += 1;
        }
    }
    if ((cluster_map & (0x01 << cluster_id)) != 0) {
        local_cluster_id = 0;
        for (int i = 0; i < cluster_id; i++) {
            if ((cluster_map & (0x01 << i)) != 0) {
                local_cluster_id++;
            }
        }

        // clear local_out
        if (flex_is_dm_core()) {
            for (int i = 0; i < (n_token_per_cluster * dim * DATA_SIZE_BYTES - 1) / ARCH_CLUSTER_ZOMEM_SIZE + 1; i++) {
                // consider the size limit of zomem
                flex_dma_async_1d(local(local_out + i * ARCH_CLUSTER_ZOMEM_SIZE), zomem(0), min(ARCH_CLUSTER_ZOMEM_SIZE, (n_token_per_cluster * dim * DATA_SIZE_BYTES - i * ARCH_CLUSTER_ZOMEM_SIZE)));
                flex_dma_async_wait_all();
            }
        }
        flex_intra_cluster_sync();
        uint16_t i_expert, i_token, index;
        uint64_t local_token, local_index;
        bool is_buffer_0 = 1;
        bool is_local_token = 0;
        // loop over experts
        i_expert = 0;
        while (i_expert < n_expert) {
            expert_offset->curr_HBM_node[i_expert] = 0;
            i_token = 0;
            // if (0 == core_id && PRINT_DEBUG_CLUSTER_ID == local_cluster_id) {
            //     printf("[UNPERMUTE_SUM] i_expert = %d\n", i_expert);
            // }
            while ((0 != expert_offset->token_cnt[i_expert]) && (expert_offset->curr_HBM_node[i_expert] != n_cluster_activated)) {
                if (1 == is_buffer_0) {
                    local_token = local_token_0;
                    local_index = local_index_0;
                } else {
                    local_token = local_token_1;
                    local_index = local_index_1;
                }
                // load in index
                if (flex_is_dm_core()) {
                    flex_dma_async_1d(local(local_index), hbm_west(expert_offset->curr_HBM_node[i_expert], expert_offset->index_offset[i_expert] + i_token * DATA_SIZE_BYTES), DATA_SIZE_BYTES);
                    flex_dma_async_wait_all();
                }
                flex_intra_cluster_sync();
                // check if token belong to local token of this cluster
                index = *(uint16_t *)local_index;
                // if (0 == core_id && PRINT_DEBUG_CLUSTER_ID == local_cluster_id) {
                //     printf("[UNPERMUTE_SUM] index = %d\n", index);
                //     printf("[UNPERMUTE_SUM] i_token = %d\n", i_token);
                //     printf("[UNPERMUTE_SUM] expert_offset->curr_HBM_node[%d] = %d\n", i_expert, expert_offset->curr_HBM_node[i_expert]);
                // }
                if ((index != INDICE_NOT_USED) && (index >= (local_cluster_id * n_token_per_cluster)) && (index < ((local_cluster_id + 1) * n_token_per_cluster))) {
                    // if (0 == core_id && PRINT_DEBUG_CLUSTER_ID == local_cluster_id) {
                    //     printf("[UNPERMUTE_SUM] index = %d is local token\n", index);
                    // }
                    is_local_token = 1;
                } else {
                    is_local_token = 0;
                }
                if (flex_is_dm_core()) {
                    if (1 == is_local_token) {
                        // load in token
                        uint64_t src_token_offset = expert_offset->value_offset[i_expert] + i_token * dim * DATA_SIZE_BYTES;
                        flex_dma_async_1d(local(local_token), hbm_west(expert_offset->curr_HBM_node[i_expert], src_token_offset), dim * DATA_SIZE_BYTES);
                    } else {
                        flex_dma_async_1d(local(local_token), zomem(0), dim * DATA_SIZE_BYTES);
                    }
                    flex_dma_async_wait_all();
                }
                flex_intra_cluster_sync();
    
                // accumulate with previous value
                if ((0 == core_id) && (1 == is_local_token)) {
                    // if (index == 0) {
                    //     printf("[UNPERMUTE_SUM] index = 0###\n");
                    // }
                    uint16_t index_local = index - local_cluster_id * n_token_per_cluster;
                    uint16_t * local_in1_ptr = (uint16_t *)local(local_token);
                    uint16_t * local_in2_ptr = (uint16_t *)local(local_out + index_local * dim * DATA_SIZE_BYTES);
                    uint16_t * local_out_ptr = (uint16_t *)local(local_out + index_local * dim * DATA_SIZE_BYTES);
                    
                    // compute element-wise operation with spatz core
                    uint16_t vl;
                    uint16_t i_element = 0;
                    while (dim > i_element) {
                        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(min(SPATZ_VL, dim - i_element)));
                        asm volatile("vle16.v v0, (%0)" : : "r"(local_in1_ptr));
                        asm volatile("vle16.v v1, (%0)" : : "r"(local_in2_ptr));
                        
                        asm volatile("vfadd.vv v8, v0, v1");

                        asm volatile("vse16.v v8, (%0)" : : "r"(local_out_ptr));

                        local_in1_ptr += vl;
                        local_in2_ptr += vl;
                        local_out_ptr += vl;
                        i_element += vl;
                    }
                }
                       
                is_buffer_0 = ~is_buffer_0;
                i_token += 1;
                // only update with one core
                if (flex_is_dm_core()) {
                    if (i_token >= expert_offset->token_cnt[i_expert]) {
                        i_token = 0;
                        expert_offset->curr_HBM_node[i_expert] += 1;
                    }
                }
            }
            flex_intra_cluster_sync();
            i_expert++;
        }
        // store back
        flex_intra_cluster_sync();
        if (flex_is_dm_core()) {
            // if (PRINT_DEBUG_CLUSTER_ID == local_cluster_id) {
            //     // print
            //     for (int i = 0; i < 5; i++) {
            //         for (int j = 0; j < 8; j++) {
            //             printf("[UNPERMUTE_SUM] local_out[%d][%d] = 0x%x\n", i, j, ((fp16 *)local(local_out))[i * dim + j]);
            //         }
            //     }
            // }
            flex_dma_async_1d(hbm_west(local_cluster_id, out_offset), local(local_out), n_token_per_cluster * dim * DATA_SIZE_BYTES);
            flex_dma_async_wait_all();
        }
        flex_intra_cluster_sync();
    }
}

void compute_moe(uint64_t in_token_offset, uint64_t n_token, uint64_t dim, uint64_t inter_dim, uint64_t n_routed_experts, uint64_t n_shared_experts, uint64_t n_activated_experts, uint64_t gate_weights_addr, uint64_t expert_w1_weights_addr, uint64_t expert_w1_bias_addr, uint64_t expert_w2_weights_addr, uint64_t expert_w2_bias_addr, uint64_t expert_w3_weights_addr, uint64_t expert_w3_bias_addr, uint64_t actual_out_offset) {
    cluster_map_t cluster_coloring_0, cluster_coloring_1, cluster_all, cluster_west_edge;
    cluster_coloring_0 = 0x5A5A;    // 0101101001011010: 1 3 4 6 9 11 12 14
    cluster_coloring_1 = 0xA5A5;    // 1010010110100101: 0 2 5 7 8 10 13 15
    cluster_all = 0xFFFF;
    cluster_west_edge = 0x1111; // 0001000100010001: 0 4 8 12
    uint64_t local_token_offset, local_expert_offset;
    local_token_offset = 0;
    local_expert_offset = local_token_offset + n_token * n_routed_experts * DATA_SIZE_BYTES;
    uint64_t *local_token_offset_ptr = &local_token_offset;
    uint64_t hbm_token_offset, hbm_index_offset;
    hbm_index_offset = actual_out_offset + ((n_token - 1) / ARCH_NUM_CLUSTER_Y + 1) * dim * DATA_SIZE_BYTES;
    hbm_token_offset = hbm_index_offset + ((n_token - 1) / ARCH_NUM_CLUSTER_Y + 1) * DATA_SIZE_BYTES;
    // ExpertOffset expert_offset;

    uint16_t n_token_per_cluster = GEMM_TILE_WIDTH_M;
    // gate processes SEQ_LEN tokens at one time 
    for (uint64_t i_token = 0; i_token < n_token; i_token += SEQ_LEN) {
        // TODO: no write back, result stored in clusters' TCDM
        // the computation of all clusters should cover the whole output matrix at once? Otherwise, require store back
        // gemm_systolic_wise(SEQ_LEN, dim, n_routed_experts, DATA_SIZE_BYTES, GEMM_TILE_WIDTH_M, GEMM_TILE_WIDTH_N, GEMM_TILE_WIDTH_K, local_token_offset_ptr);
        if (0 == flex_get_cluster_id() && flex_is_dm_core()) {  
            flex_dma_async_1d(local(local_token_offset), hbm_addr(in_token_offset + i_token * n_activated_experts * DATA_SIZE_BYTES), SEQ_LEN * n_activated_experts * DATA_SIZE_BYTES);
            flex_dma_async_wait_all();
        }
        flex_global_barrier_xy();
        top_k_weighted_permute(dim, n_activated_experts, n_routed_experts, n_token_per_cluster, local(local_token_offset), in_token_offset, hbm_token_offset, hbm_index_offset, (ExpertOffset *)local_expert_offset, cluster_west_edge);
        flex_global_barrier_xy();
        unpermute_sum(dim, n_routed_experts, n_token_per_cluster, actual_out_offset + (i_token / ARCH_NUM_CLUSTER_Y) * dim * DATA_SIZE_BYTES, hbm_token_offset, hbm_index_offset, (ExpertOffset *)local_expert_offset, cluster_west_edge);
        flex_global_barrier_xy();
    }
}

#endif