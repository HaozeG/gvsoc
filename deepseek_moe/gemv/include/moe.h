#ifndef MOE_H
#define MOE_H

// #include <math.h>
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
#define TILE_WIDTH 64
#define OPAND_SIZE TILE_WIDTH * TILE_WIDTH * DATA_SIZE_BYTES
// Parameter for element-wise functions
#define ELEMENT_WISE_TILE_WIDTH 2

// 2.5 in float
fp16 route_scale = (fp16)0x4100;

typedef void (*element_wise_op_1_in_t)(const fp16* input, fp16* output);
typedef void (*element_wise_op_2_in_t)(const fp16* input1, const fp16* input2, fp16* output);
typedef void (*element_wise_op_2_in_const_t)(const fp16* input1, const fp16* in_const, fp16* output);

void gemm(const uint32_t A, const uint32_t B, const uint32_t C, const uint32_t K, const uint32_t M, const uint32_t N, const uint32_t bias_addr);
void top_k(const uint32_t in_addr, const uint32_t out_value_addr, const uint32_t out_index_addr, const uint32_t k, const uint32_t dim, const uint32_t n_token);
void silu(const uint32_t in_addr, const uint32_t out_addr, const uint32_t dim, const uint32_t n_token);
void sigmoid(const uint32_t in_addr, const uint32_t out_addr, const uint32_t dim, const uint32_t n_token);
void normalize(const uint32_t in_addr, const uint32_t out_addr, const uint32_t dim, const uint32_t n_token);
void dot_product(const uint32_t in_addr_1, const uint32_t in_addr_2, const uint32_t out_addr, const uint32_t dim, const uint32_t n_token);
void add(const uint32_t in_addr_1, const uint32_t in_addr_2, const uint32_t out_addr, const uint32_t dim, const uint32_t n_token);
void compute_moe(uint32_t in_token_addr, uint16_t n_token, uint16_t dim, uint16_t inter_dim, uint16_t n_routed_experts, uint16_t n_shared_experts, uint16_t n_activated_experts, uint32_t gate_weights_addr, uint32_t expert_w1_weights_addr, uint32_t expert_w1_bias_addr, uint32_t expert_w2_weights_addr, uint32_t expert_w2_bias_addr, uint32_t expert_w3_weights_addr, uint32_t expert_w3_bias_addr, uint32_t actual_out_addr);
void apply_element_wise_1_in(const uint32_t in_addr, const uint32_t out_addr, const uint32_t dim, const uint32_t n_token, element_wise_op_1_in_t op);
void apply_element_wise_2_in_const(const uint32_t in_addr, const fp16 in_const, const uint32_t out_addr, const uint32_t dim, const uint32_t n_token, element_wise_op_2_in_const_t op);
void apply_element_wise_2_in(const uint32_t in_addr1, const uint32_t in_addr2, const uint32_t out_addr, const uint32_t dim, const uint32_t n_token, element_wise_op_2_in_t op);
void silu_op(const fp16* input, fp16* output);
void sigmoid_op(const fp16* input, fp16* output);
void mul_op(const fp16* input1, const fp16* input2, fp16* output);
void add_op(const fp16* input1, const fp16* input2, fp16* output);

/**
 * @brief Select top k values from each row of the input matrix and write the result to the output matrix along with the corresponding indices.
 * 
 * @param in_addr 
 * @param out_value_addr 
 * @param out_index_addr 
 * @param k top k values to select (less than 64)
 * @param n_routed_expert number of candidate values in each row of the input matrix
 * @param n_token number of rows in the input matrix
 */
// TODO: consider using quick select algorithm for better performance
void top_k(const uint32_t in_addr, const uint32_t out_value_addr, const uint32_t out_index_addr, const uint32_t k, const uint32_t n_routed_expert, const uint32_t n_token) {
    if (0 == k || 0 == n_routed_expert || 0 == n_token || k > TILE_WIDTH) {
        return;
    }
    flex_global_barrier_xy();
    
    uint32_t cluster_id = flex_get_cluster_id();
    uint32_t core_id = ARCH_NUM_CORE_PER_CLUSTER - flex_get_core_id() - 1;  // reverse the core id to make the dm core the first core in the cluster
    uint32_t i_row_cluster = cluster_id * ARCH_NUM_CORE_PER_CLUSTER;
    uint16_t transfer_rows;
    uint16_t max, max_idx;

    uint32_t local_out_value, local_out_indicies, local_in;
    local_out_value = ARCH_CLUSTER_TCDM_SIZE - ARCH_NUM_CORE_PER_CLUSTER * k * DATA_SIZE_BYTES;
    local_out_indicies = local_out_value - ARCH_NUM_CORE_PER_CLUSTER * k * DATA_SIZE_BYTES;
    local_in = local_out_indicies - ARCH_NUM_CORE_PER_CLUSTER * n_routed_expert * DATA_SIZE_BYTES;
    local_in += core_id * n_routed_expert * DATA_SIZE_BYTES;

    // NOTE: cannot use i_row_core < n_token here. Excluding certain cores from computation involving intra-cluster synchronization could cause synchronization issues!!!
    while (i_row_cluster < n_token) {
        // one dma transfer per cluster
        transfer_rows = fmin(ARCH_NUM_CORE_PER_CLUSTER, n_token - i_row_cluster);
        // printf("[TOP_K] i_row: %d, transfer_rows: %d\n", i_row, transfer_rows);

        if (flex_is_dm_core()) {
            flex_dma_async_1d(local(local_in), hbm_addr(in_addr + i_row_cluster * n_routed_expert * DATA_SIZE_BYTES), transfer_rows * n_routed_expert * DATA_SIZE_BYTES);
            flex_dma_async_wait_all();
        }
        flex_intra_cluster_sync();

        if (i_row_cluster + core_id < n_token) {
            // iterate each row k times to find the top k values
            for (int i = 0; i < k; i++) {
                max = ((uint16_t *)local(local_in))[i];
                max_idx = i;
                for (int j = i + 1; j < n_routed_expert; j++) {
                    if (asm_fp16_compare((const fp16 *)local(local_in + j * DATA_SIZE_BYTES), (const fp16 *)&max) == 1) {
                        max = ((uint16_t *)local(local_in))[j];
                        max_idx = j;
                        // printf("[TOP_K] update max: 0x%04x, idx: %d\n", max, max_idx);
                    }
                }
                // printf("[TOP_K] top %d: 0x%04x, idx: %d\n", i, max, max_idx);
                // Keep the top k values at top k positions of input matrix
                uint16_t temp;
                temp = ((uint16_t *)local(local_in))[i];
                ((uint16_t *)local(local_in))[i] = ((uint16_t *)local(local_in))[max_idx];
                ((uint16_t *)local(local_in))[max_idx] = temp;

                // Keep the top k indices at top k positions of output matrix
                ((uint16_t *)local(local_out_value))[i] = max;
                ((uint16_t *)local(local_out_indicies))[i] = max_idx;
            }
        }
        flex_intra_cluster_sync();
        // transfer the top k values and indices to HBM
        if (flex_is_dm_core()) {
            // flex_dma_async_1d(hbm_addr(in_addr + i_row * k * DATA_SIZE_BYTES), local(local_in), transfer_rows * n_routed_expert * DATA_SIZE_BYTES);
            flex_dma_async_1d(hbm_addr(out_value_addr + i_row_cluster * k * DATA_SIZE_BYTES), local(local_out_value), transfer_rows * k * DATA_SIZE_BYTES);
            flex_dma_async_1d(hbm_addr(out_index_addr + i_row_cluster * k * DATA_SIZE_BYTES), local(local_out_indicies), transfer_rows * k * DATA_SIZE_BYTES);
            flex_dma_async_wait_all();
        }
        i_row_cluster += ARCH_NUM_CORE_PER_CLUSTER * ARCH_NUM_CLUSTER;
    }
    flex_global_barrier_xy();
}

/**
 * @brief normalize the input matrix along rows. Consider dim here to be small, just use the simplist way to implement it.
 * 
 * @param in_addr 
 * @param out_addr 
 * @param dim colomn dimension of the input matrix
 * @param n_token row dimension of the input matrix
 */
void normalize(const uint32_t in_addr, const uint32_t out_addr, const uint32_t dim, const uint32_t n_token) {
    if (0 == dim || 0 == n_token) {
        return;
    }
    flex_global_barrier_xy();
    
    uint32_t cluster_id = flex_get_cluster_id();
    uint32_t core_id = ARCH_NUM_CORE_PER_CLUSTER - flex_get_core_id() - 1;  // reverse the core id to make the dm core the first core in the cluster
    uint32_t i_row_cluster = cluster_id * ARCH_NUM_CORE_PER_CLUSTER;
    uint16_t transfer_rows;

    uint32_t local_out, local_sum;
    local_out = ARCH_CLUSTER_TCDM_SIZE - dim * DATA_SIZE_BYTES;
    local_sum = local_out - DATA_SIZE_BYTES;

    // NOTE: cannot use i_row_core < n_token here. Excluding certain cores from computation involving intra-cluster synchronization could cause synchronization issues!!!
    while (i_row_cluster < n_token) {
        // Transfer one row per cluster
        if (flex_is_dm_core()) {
            // flex_dma_async_1d_reduction(local(local_sum), hbm_addr(in_addr + i_row_cluster * dim * DATA_SIZE_BYTES), dim * DATA_SIZE_BYTES, COLLECTIVE_REDADD_FP_16);
            flex_dma_async_1d(local(local_out), hbm_addr(in_addr + i_row_cluster * dim * DATA_SIZE_BYTES), dim * DATA_SIZE_BYTES);
            flex_dma_async_wait_all();
        }
        flex_intra_cluster_sync();
        
        if (0 == core_id) {
            float sum = 0;
            for (int i = 0; i < dim; i++) {
                // printf("[NORMALIZE] 0x%x\n", ((fp16 *)local(local_out))[i]);
                sum += fp16_to_float(((fp16 *)local(local_out))[i]);
            }
            ((fp16 *)local(local_sum))[0] = float_to_fp16(sum);
        }
        flex_intra_cluster_sync();

        uint32_t n_element_per_core = (dim - 1) / ARCH_NUM_CORE_PER_CLUSTER + 1;
        for (int i = 0; i < n_element_per_core; i++) {
            if (i + core_id * n_element_per_core < dim) {
                fp16 a = ((fp16 *)local(local_out))[i + core_id * n_element_per_core];
                fp16 *b_ptr = (fp16 *)local(local_sum);
                fp16 *c_ptr = &((fp16 *)local(local_out))[i + core_id * n_element_per_core];
                // if (0 == core_id) {
                //     printf("[NORMALIZE] a = 0x%x sum = 0x%x\n", a, *b_ptr);
                // }
                asm_fp16_div(&a, b_ptr, c_ptr);
            }
        }
        flex_intra_cluster_sync();
        // transfer the top k values and indices to HBM
        if (flex_is_dm_core()) {
            flex_dma_async_1d(hbm_addr(out_addr + i_row_cluster * dim * DATA_SIZE_BYTES), local(local_out), dim * DATA_SIZE_BYTES);
            flex_dma_async_wait_all();
        }
        i_row_cluster += ARCH_NUM_CORE_PER_CLUSTER * ARCH_NUM_CLUSTER;
    }
    flex_global_barrier_xy();
}

void sigmoid_op(const fp16* input, fp16* output) {
    asm_fp16_sigmoid(input, output);
}

void add_op(const fp16* input1, const fp16* input2, fp16* output) {
    float fa = fp16_to_float(*input1);
    float fb = fp16_to_float(*input2);
    *output = float_to_fp16(fa + fb);
}

void mul_op(const fp16* input1, const fp16* input2, fp16* output) {
    float fa = fp16_to_float(*input1);
    float fb = fp16_to_float(*input2);
    *output = float_to_fp16(fa * fb);
}

void silu_op(const fp16* input, fp16* output) {
    asm_fp16_sigmoid(input, output);
    // *output = fp16_fma(*output, *input, (fp16)0);
    float fa = fp16_to_float(*output);
    float fb = fp16_to_float(*input);
    *output = float_to_fp16(fa * fb);
}


/**
 * @brief apply element-wise operation on ONE INPUT matrix. Each core processes ELEMENT_WISE_TILE_WIDTH elements at a time.
 * 
 * @param in_addr 
 * @param out_addr 
 * @param dim 
 * @param n_token 
 * @param op element_wise_op_t function pointer
 */
void apply_element_wise_1_in(const uint32_t in_addr, const uint32_t out_addr, const uint32_t dim, const uint32_t n_token, element_wise_op_1_in_t op) {
    if (0 == dim || 0 == n_token || NULL == op) {
        return;
    }
    flex_global_barrier_xy();
    uint32_t cluster_id = flex_get_cluster_id();
    uint32_t core_id = ARCH_NUM_CORE_PER_CLUSTER - flex_get_core_id() - 1;  // reverse the core id to make the dm core the first core in the cluster
    uint32_t i_element_cluster = cluster_id * ARCH_NUM_CORE_PER_CLUSTER * ELEMENT_WISE_TILE_WIDTH;
    uint32_t n_element_per_cluster, n_element_per_core;

    uint32_t local_in, local_out;
    local_in = ARCH_CLUSTER_TCDM_SIZE - ARCH_NUM_CORE_PER_CLUSTER * ELEMENT_WISE_TILE_WIDTH * DATA_SIZE_BYTES;
    local_out = local_in - ARCH_NUM_CORE_PER_CLUSTER * ELEMENT_WISE_TILE_WIDTH * DATA_SIZE_BYTES;

    while (i_element_cluster < n_token * dim) {
        // load data: one dma transfer per cluster
        n_element_per_cluster = fmin(ARCH_NUM_CORE_PER_CLUSTER * ELEMENT_WISE_TILE_WIDTH, dim * n_token - cluster_id * ARCH_NUM_CORE_PER_CLUSTER * ELEMENT_WISE_TILE_WIDTH);
        if (flex_is_dm_core()) {
            // printf("[APPLY_ELEMENT_WISE] Start loading data\n");
            flex_dma_async_1d(local(local_in), hbm_addr(in_addr + i_element_cluster * DATA_SIZE_BYTES), n_element_per_cluster * DATA_SIZE_BYTES);
            flex_dma_async_wait_all();
            // printf("[APPLY_ELEMENT_WISE] local_in: 0x%x, in_addr: 0x%x, n_element_per_cluster: %d\n", local(local_in), hbm_addr(in_addr + i_element * DATA_SIZE_BYTES), n_element_per_cluster);
        }
        flex_intra_cluster_sync();

        // compute element-wise operation
        n_element_per_core = fmin(ELEMENT_WISE_TILE_WIDTH, n_element_per_cluster - core_id * ELEMENT_WISE_TILE_WIDTH);
        // printf("[APPLY_ELEMENT_WISE] Start computing element-wise operation\n");
        // printf("[APPLY_ELEMENT_WISE] block_id: %d, i_element: %d, n_element_per_cluster: %d, n_element_per_core: %d, core_id: %d, cluster_id: %d\n", block_id, i_element, n_element_per_cluster, n_element_per_core, core_id, cluster_id);
        for (int i = 0; i < n_element_per_core; i++) {
            int idx = i + core_id * ELEMENT_WISE_TILE_WIDTH;
            // printf("[APPLY_ELEMENT_WISE] addr: %x\n", local_in + idx * DATA_SIZE_BYTES);
            // printf("[APPLY_ELEMENT_WISE] input: 0x%04x\n", ((fp16*)local(local_in + idx * DATA_SIZE_BYTES))[0]);
            op((const fp16*)local(local_in + idx * DATA_SIZE_BYTES), 
                     (fp16*)local(local_out + idx * DATA_SIZE_BYTES));
        }
        flex_intra_cluster_sync();

        // store data: one dma transfer per cluster
        if (flex_is_dm_core()) {
            flex_dma_async_1d(hbm_addr(out_addr + i_element_cluster * DATA_SIZE_BYTES), local(local_out), n_element_per_cluster * DATA_SIZE_BYTES);
            flex_dma_async_wait_all();
        }
        i_element_cluster += ARCH_NUM_CORE_PER_CLUSTER * ARCH_NUM_CLUSTER * ELEMENT_WISE_TILE_WIDTH;
    }
    flex_global_barrier_xy();
}

/**
 * @brief apply element-wise operation on ONE INPUT matrix using another input constant. Each core processes ELEMENT_WISE_TILE_WIDTH elements at a time.
 * 
 * @param in_addr 
 * @param in_const
 * @param out_addr 
 * @param dim 
 * @param n_token 
 * @param op element_wise_op_t function pointer
 */
void apply_element_wise_2_in_const(const uint32_t in_addr, const fp16 in_const, const uint32_t out_addr, const uint32_t dim, const uint32_t n_token, element_wise_op_2_in_const_t op) {
    if (0 == dim || 0 == n_token || NULL == op) {
        return;
    }
    flex_global_barrier_xy();
    uint32_t cluster_id = flex_get_cluster_id();
    uint32_t core_id = ARCH_NUM_CORE_PER_CLUSTER - flex_get_core_id() - 1;  // reverse the core id to make the dm core the first core in the cluster
    uint32_t i_element_cluster = cluster_id * ARCH_NUM_CORE_PER_CLUSTER * ELEMENT_WISE_TILE_WIDTH;
    uint32_t n_element_per_cluster, n_element_per_core;

    uint32_t local_in, local_out;
    local_in = ARCH_CLUSTER_TCDM_SIZE - ARCH_NUM_CORE_PER_CLUSTER * ELEMENT_WISE_TILE_WIDTH * DATA_SIZE_BYTES;
    local_out = local_in - ARCH_NUM_CORE_PER_CLUSTER * ELEMENT_WISE_TILE_WIDTH * DATA_SIZE_BYTES;

    while (i_element_cluster < n_token * dim) {
        // load data: one dma transfer per cluster
        n_element_per_cluster = fmin(ARCH_NUM_CORE_PER_CLUSTER * ELEMENT_WISE_TILE_WIDTH, dim * n_token - cluster_id * ARCH_NUM_CORE_PER_CLUSTER * ELEMENT_WISE_TILE_WIDTH);
        if (flex_is_dm_core()) {
            flex_dma_async_1d(local(local_in), hbm_addr(in_addr + i_element_cluster * DATA_SIZE_BYTES), n_element_per_cluster * DATA_SIZE_BYTES);
            flex_dma_async_wait_all();
        }
        flex_intra_cluster_sync();

        // compute element-wise operation
        n_element_per_core = fmin(ELEMENT_WISE_TILE_WIDTH, n_element_per_cluster - core_id * ELEMENT_WISE_TILE_WIDTH);
        for (int i = 0; i < n_element_per_core; i++) {
            int idx = i + core_id * ELEMENT_WISE_TILE_WIDTH;
            op((const fp16*)local(local_in + idx * DATA_SIZE_BYTES), (fp16*) &in_const, (fp16*)local(local_out + idx * DATA_SIZE_BYTES));
        }
        flex_intra_cluster_sync();

        // store data: one dma transfer per cluster
        if (flex_is_dm_core()) {
            flex_dma_async_1d(hbm_addr(out_addr + i_element_cluster * DATA_SIZE_BYTES), local(local_out), n_element_per_cluster * DATA_SIZE_BYTES);
            flex_dma_async_wait_all();
        }
        i_element_cluster += ARCH_NUM_CORE_PER_CLUSTER * ARCH_NUM_CLUSTER * ELEMENT_WISE_TILE_WIDTH;
    }
    flex_global_barrier_xy();
}

/**
 * @brief apply element-wise operation on TWO INPUT matrix. Each core processes ELEMENT_WISE_TILE_WIDTH elements at a time.
 * 
 * @param in_addr1
 * @param in_addr2
 * @param out_addr 
 * @param dim 
 * @param n_token 
 * @param op element_wise_op_t function pointer
 */
void apply_element_wise_2_in(const uint32_t in_addr1, const uint32_t in_addr2, const uint32_t out_addr, const uint32_t dim, const uint32_t n_token, element_wise_op_2_in_t op) {
    if (0 == dim || 0 == n_token || NULL == op) {
        return;
    }
    flex_global_barrier_xy();
    uint32_t cluster_id = flex_get_cluster_id();
    uint32_t core_id = ARCH_NUM_CORE_PER_CLUSTER - flex_get_core_id() - 1;  // reverse the core id to make the dm core the first core in the cluster
    uint32_t i_element_cluster = cluster_id * ARCH_NUM_CORE_PER_CLUSTER * ELEMENT_WISE_TILE_WIDTH;
    uint32_t n_element_per_cluster, n_element_per_core;

    uint32_t local_in1, local_in2, local_out;
    local_in1 = ARCH_CLUSTER_TCDM_SIZE - ARCH_NUM_CORE_PER_CLUSTER * ELEMENT_WISE_TILE_WIDTH * DATA_SIZE_BYTES;
    local_in2 = local_in1 - ARCH_NUM_CORE_PER_CLUSTER * ELEMENT_WISE_TILE_WIDTH * DATA_SIZE_BYTES;
    local_out = local_in2 - ARCH_NUM_CORE_PER_CLUSTER * ELEMENT_WISE_TILE_WIDTH * DATA_SIZE_BYTES;

    while (i_element_cluster < n_token * dim) {
        // load data: one dma transfer per cluster
        n_element_per_cluster = fmin(ARCH_NUM_CORE_PER_CLUSTER * ELEMENT_WISE_TILE_WIDTH, dim * n_token - cluster_id * ARCH_NUM_CORE_PER_CLUSTER * ELEMENT_WISE_TILE_WIDTH);
        if (flex_is_dm_core()) {
            // printf("[APPLY_ELEMENT_WISE] Start loading data\n");
            flex_dma_async_1d(local(local_in1), hbm_addr(in_addr1 + i_element_cluster * DATA_SIZE_BYTES), n_element_per_cluster * DATA_SIZE_BYTES);
            flex_dma_async_1d(local(local_in2), hbm_addr(in_addr2 + i_element_cluster * DATA_SIZE_BYTES), n_element_per_cluster * DATA_SIZE_BYTES);
            flex_dma_async_wait_all();
            // printf("[APPLY_ELEMENT_WISE] local_in: 0x%x, in_addr: 0x%x, n_element_per_cluster: %d\n", local(local_in), hbm_addr(in_addr + i_element * DATA_SIZE_BYTES), n_element_per_cluster);
        }
        flex_intra_cluster_sync();

        // compute element-wise operation
        n_element_per_core = fmin(ELEMENT_WISE_TILE_WIDTH, n_element_per_cluster - core_id * ELEMENT_WISE_TILE_WIDTH);
        // printf("[APPLY_ELEMENT_WISE] Start computing element-wise operation\n");
        // printf("[APPLY_ELEMENT_WISE] block_id: %d, i_element: %d, n_element_per_cluster: %d, n_element_per_core: %d, core_id: %d, cluster_id: %d\n", block_id, i_element, n_element_per_cluster, n_element_per_core, core_id, cluster_id);
        for (int i = 0; i < n_element_per_core; i++) {
            int idx = i + core_id * ELEMENT_WISE_TILE_WIDTH;
            // printf("[APPLY_ELEMENT_WISE] addr: %x\n", local_in + idx * DATA_SIZE_BYTES);
            // printf("[APPLY_ELEMENT_WISE] input: 0x%04x\n", ((fp16*)local(local_in + idx * DATA_SIZE_BYTES))[0]);
            op((const fp16*)local(local_in1 + idx * DATA_SIZE_BYTES), (const fp16*)local(local_in1 + idx * DATA_SIZE_BYTES), (fp16*)local(local_out + idx * DATA_SIZE_BYTES));
        }
        flex_intra_cluster_sync();

        // store data: one dma transfer per cluster
        if (flex_is_dm_core()) {
            flex_dma_async_1d(hbm_addr(out_addr + i_element_cluster * DATA_SIZE_BYTES), local(local_out), n_element_per_cluster * DATA_SIZE_BYTES);
            flex_dma_async_wait_all();
        }
        i_element_cluster += ARCH_NUM_CORE_PER_CLUSTER * ARCH_NUM_CLUSTER * ELEMENT_WISE_TILE_WIDTH;
    }
    flex_global_barrier_xy();
}

/**
 * @brief element-wise silu activation function
 * 
 * @param in_addr 
 * @param out_addr 
 * @param dim 
 * @param n_token 
 */
void silu(const uint32_t in_addr, const uint32_t out_addr, const uint32_t dim, const uint32_t n_token) {
    apply_element_wise_1_in(in_addr, out_addr, dim, n_token, silu_op); 
}

/**
 * @brief element-wise sigmoid function
 * 
 * @param in_addr 
 * @param out_addr 
 * @param dim 
 * @param n_token 
 */
void sigmoid(const uint32_t in_addr, const uint32_t out_addr, const uint32_t dim, const uint32_t n_token) {
    apply_element_wise_1_in(in_addr, out_addr, dim, n_token, sigmoid_op); 
}

/**
 * @brief element-wise dot-product function
 * 
 * @param in_addr_1 
 * @param in_addr_2 
 * @param out_addr 
 * @param dim 
 * @param n_token 
 */
void dot_product(const uint32_t in_addr_1, const uint32_t in_addr_2, const uint32_t out_addr, const uint32_t dim, const uint32_t n_token) {
    apply_element_wise_2_in(in_addr_1, in_addr_2, out_addr, dim, n_token, mul_op);
}

/**
 * @brief element-wise add function
 * 
 * @param in_addr_1 
 * @param in_addr_2 
 * @param out_addr 
 * @param dim 
 * @param n_token 
 */
void add(const uint32_t in_addr_1, const uint32_t in_addr_2, const uint32_t out_addr, const uint32_t dim, const uint32_t n_token) {
    apply_element_wise_2_in(in_addr_1, in_addr_2, out_addr, dim, n_token, add_op);
}

// TODO: For inferencing, the input matrix shape is [1, dim], which is not efficient enough for dividing output tasks to different clusters.
/**
 * @brief GEMM operation A * B = C for input matrices A (MxK), B (KxN) and output matrix C (MxN). Assumes that K, M and N are multiples of TILE_WIDTH.
 * 
 * @param A offset of matrix A wrt. hbm_addr
 * @param B offset of matrix B wrt. hbm_addr
 * @param C offset of matrix C wrt. hbm_addr
 * @param K shared dimension of A and B
 * @param M 
 * @param N 
 * @param bias_addr address of bias matrix. shaped as [1, N], added to each row of the output matrix
 */
void gemm(const uint32_t A, const uint32_t B, const uint32_t C, const uint32_t K, const uint32_t M, const uint32_t N, const uint32_t bias_addr) {
    if (0 == M || 0 == N || 0 == K) {
        return;
    }
    flex_global_barrier_xy();
    uint32_t cluster_id = flex_get_cluster_id();
    uint32_t core_id = flex_get_core_id();

    uint32_t accumulator, local_A, local_B;
    accumulator = 0;
    local_A = OPAND_SIZE;
    local_B = local_A + OPAND_SIZE;

    int m_remaining, n_remaining, m_tile, n_tile, k_tile, bK;

    // each cluster compute a tile of the output matrix
    int gi = get_pos(cluster_id).x;
    int gj = get_pos(cluster_id).y; 

    // if (flex_is_first_core() && gi == 0 && gj == 0) {
    //     printf("[GEMM CONFIG] M: %d, K: %d, N: %d\n", M, K, N);
    // }
    // flex_intra_cluster_sync();

    for (int i = 0; i < (M - 1) / BLOCK_WIDTH + 1; i++) {
        for (int j = 0; j < (N - 1) / BLOCK_WIDTH + 1; j++) {
            m_remaining = (i < M / BLOCK_WIDTH) ? BLOCK_WIDTH : (M % BLOCK_WIDTH);
            n_remaining = (j < N / BLOCK_WIDTH) ? BLOCK_WIDTH : (N % BLOCK_WIDTH);
            
            // Skip clusters that aren't needed for this block
            if (m_remaining <= gi * TILE_WIDTH || n_remaining <= gj * TILE_WIDTH) {
                continue; 
            }
            
            m_tile = fmin(TILE_WIDTH, m_remaining - gi * TILE_WIDTH);
            n_tile = fmin(TILE_WIDTH, n_remaining - gj * TILE_WIDTH);
            
            if(flex_is_dm_core()) {
                if (bias_addr == zomem(0)) {
                    flex_dma_async_1d(local(accumulator), zomem(0), m_tile * n_tile * DATA_SIZE_BYTES);
                    flex_dma_async_wait_all();
                } else {
                    flex_dma_sync_2d(local(accumulator), bias_addr + (gj * TILE_WIDTH + j * BLOCK_WIDTH) * DATA_SIZE_BYTES, n_tile*DATA_SIZE_BYTES, n_tile*DATA_SIZE_BYTES, 0, m_tile);
                }
            }
            flex_intra_cluster_sync();
            if(flex_is_first_core()) {
                // printf("[REDMULE CONFIG] m: %d, n: %d, k: %d\n", m_tile, TILE_WIDTH, n_tile);
                // Redmule Config: [m_size, n_size] * [n_size, k_size] = [m_size, k_size]
                flex_redmule_config(m_tile, TILE_WIDTH, n_tile);
            }
            
            // bK: inner loop tile computing partial sums
            for (bK = 0; bK < K; bK += TILE_WIDTH) {
                k_tile = fmin(TILE_WIDTH, K - bK);

                // SoftHier_HBM -> SoftHier_TCDM 2D
                if(flex_is_dm_core()) {
                    flex_dma_sync_2d(local(local_A), hbm_addr(A + ((K * (TILE_WIDTH * gi + i * BLOCK_WIDTH)) + bK) * DATA_SIZE_BYTES), k_tile*DATA_SIZE_BYTES, k_tile*DATA_SIZE_BYTES, K*DATA_SIZE_BYTES, m_tile);
                }
                // flex_intra_cluster_sync();

                // SoftHier_HBM -> SoftHier_TCDM 2D
                if(flex_is_dm_core()) {
                    flex_dma_sync_2d(local(local_B), hbm_addr(B + (N * bK + TILE_WIDTH * gj + j * BLOCK_WIDTH) * DATA_SIZE_BYTES), n_tile*DATA_SIZE_BYTES,  n_tile*DATA_SIZE_BYTES, N*DATA_SIZE_BYTES, k_tile);
                }
                
                
                // make sure data is ready
                flex_intra_cluster_sync();

                // check operands
                // if(flex_is_first_core()) {
                //     printf("[OPERANDS]\n");
                //     printf("A:  \n");
                //     for (int i = 0; i < m_tile; i++) {
                //         for (int j = 0; j < k_tile; j++) {
                //             printf(" 0x%04x ", 
                //                     ((uint16_t *)local(local_A))[i * k_tile + j]);
                //         }
                //         printf("\n");
                //     }
                //     printf("B:  \n");
                //     for (int i = 0; i < k_tile; i++) {
                //         for (int j = 0; j < n_tile; j++) {
                //             printf(" 0x%04x ", 
                //                     ((uint16_t *)local(local_B))[i * n_tile + j]);
                //         }
                //         printf("\n");
                //     }
                // }

                if (flex_is_first_core()) {
                    // change configuration if the tile in K dimension is not full
                    if (k_tile != TILE_WIDTH) {
                        // printf("[REDMULE CONFIG] m: %d, n: %d, k: %d\n", m_tile, k_tile, n_tile);
                        // Redmule Config: [m_size, n_size] * [n_size, k_size] = [m_size, k_size]
                        flex_redmule_config(m_tile, k_tile, n_tile);
                    }

                    uint32_t _in_local_a = local_A;
                    uint32_t _in_local_b = local_B;
                    uint32_t _in_local_sum = accumulator;

                    ///////////////////
                    // multiply and accumulate
                    flex_redmule_trigger(_in_local_a, _in_local_b, _in_local_sum, REDMULE_FP_16);
                    flex_redmule_wait();
                    ///////////////////
                    // matmul_fp16((fp16 *)_in_local_sum, (fp16 *)_in_local_sum, (fp16 *)_in_local_a, (fp16 *)_in_local_b, m_tile, k_tile, n_tile);

                }
                // TODO: consider double buffering in_local_a and in_local_b to remove this line safely
                // flex_intra_cluster_sync();

                // if(flex_is_first_core()) {
                    // printf("[ACCUMULATOR]\n    C: 0x%x\n", ((uint16_t *)local(accumulator))[1]);
                // }

                // if (flex_is_dm_core())
                // {
                //     flex_dma_async_1d_reduction(local(accumulator), local(local_sum), m_tile * n_tile * DATA_SIZE_BYTES, COLLECTIVE_REDADD_FP_16);
                //     flex_dma_async_wait_all();
                // }

                // add local_sum to accumulator
                // for (uint32_t i = 0; i < m_tile * n_tile; i += ARCH_NUM_CORE_PER_CLUSTER) {
                //     if (i + core_id < m_tile * n_tile) {
                //         uint32_t local_sum_offset = (i + core_id) * DATA_SIZE_BYTES;
                //         uint32_t accumulator_offset = (i + core_id) * DATA_SIZE_BYTES;
                //         float16_t local_sum_val = ((float16_t *)local(local_sum_offset))[0];
                //         float16_t accumulator_val = ((float16_t *)local(accumulator_offset))[0];
                //     }
                // }

            }

            flex_intra_cluster_sync();
            // SoftHier_TCDM -> SoftHier_HBM
            if(flex_is_dm_core()) {
                flex_dma_sync_2d(hbm_addr(C + ((TILE_WIDTH * gi + i * BLOCK_WIDTH) * N + TILE_WIDTH * gj + j * BLOCK_WIDTH) * DATA_SIZE_BYTES), local(accumulator), n_tile*DATA_SIZE_BYTES, N*DATA_SIZE_BYTES, n_tile*DATA_SIZE_BYTES, m_tile);
            }
            // flex_intra_cluster_sync();
        }
        // flex_intra_cluster_sync();
    }
    flex_global_barrier_xy();
}

void compute_moe(uint32_t in_token_addr, uint16_t n_token, uint16_t dim, uint16_t inter_dim, uint16_t n_routed_experts, uint16_t n_shared_experts, uint16_t n_activated_experts, uint32_t gate_weights_addr, uint32_t expert_w1_weights_addr, uint32_t expert_w1_bias_addr, uint32_t expert_w2_weights_addr, uint32_t expert_w2_bias_addr, uint32_t expert_w3_weights_addr, uint32_t expert_w3_bias_addr, uint32_t actual_out_addr) {
    flex_global_barrier_xy();
    uint32_t top_k_weights_addr, top_k_indices_addr;
    uint32_t temp_token_0, temp_token_1;
    top_k_weights_addr = actual_out_addr + n_token * dim * DATA_SIZE_BYTES;
    top_k_indices_addr = top_k_weights_addr + n_token * n_activated_experts * DATA_SIZE_BYTES;
    temp_token_0 = top_k_indices_addr + n_token * n_activated_experts * DATA_SIZE_BYTES;
    temp_token_1 = temp_token_0 + n_token * dim * DATA_SIZE_BYTES;

    // Gate 
    gemm(in_token_addr, gate_weights_addr, temp_token_0, dim, n_token, n_routed_experts, zomem(0));
    top_k(temp_token_0, top_k_weights_addr, top_k_indices_addr, n_activated_experts, n_routed_experts, n_token);
    // sigmoid
    sigmoid(top_k_weights_addr, top_k_weights_addr, n_activated_experts, n_token);
    // normalize
    normalize(top_k_weights_addr, top_k_weights_addr, n_activated_experts, n_token);
    
    // Routed experts
    // self.w2.forward(silu(self.w1.forward(x)) * self.w3.forward(x))
    int i = 0;
    uint16_t i_expert;
    fp16 w_expert;
    while (i < n_activated_experts) {
        // TODO: check i, w
        i_expert = ((uint16_t *)hbm_addr(top_k_indices_addr))[i];
        w_expert = ((fp16 *)hbm_addr(top_k_weights_addr))[i];
        mul_op(&w_expert, &route_scale, &w_expert);
        
        // w1.forward(x)
        gemm(in_token_addr, expert_w1_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES), temp_token_0, dim, n_token, inter_dim, hbm_addr(expert_w1_bias_addr + (inter_dim * i_expert * DATA_SIZE_BYTES)));
        // silu(w1.forward(x))
        silu(temp_token_0, temp_token_0, inter_dim, n_token);
        
        // w3.forward(x)
        gemm(in_token_addr, expert_w3_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES), temp_token_1, dim, n_token, inter_dim, hbm_addr(expert_w3_bias_addr + (inter_dim * i_expert * DATA_SIZE_BYTES)));
        
        // silu(w1.forward(x)) * w3.forward(x)
        dot_product(temp_token_0, temp_token_1, temp_token_0, inter_dim, n_token);
        
        // w2.forward(silu(w1.forward(x)) * w3.forward(x))
        gemm(temp_token_0, expert_w2_weights_addr + (inter_dim * dim * i_expert * DATA_SIZE_BYTES), temp_token_0, inter_dim, n_token, dim, hbm_addr(expert_w2_bias_addr + (dim * i_expert * DATA_SIZE_BYTES)));
        
        // multiply by gate weight and add to the output
        apply_element_wise_2_in_const(temp_token_0, w_expert, temp_token_0, dim, n_token, mul_op);
        apply_element_wise_2_in(temp_token_0, actual_out_addr, actual_out_addr, dim, n_token, add_op);

        i++;
    }
        
    // Shared experts
    // self.w2.forward(silu(self.w1.forward(x)) * self.w3.forward(x))
    for (int i_expert = n_routed_experts; i_expert < (n_routed_experts + n_shared_experts); i_expert++) {
        gemm(in_token_addr, expert_w1_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES), temp_token_0, dim, n_token, inter_dim, hbm_addr(expert_w1_bias_addr + (inter_dim * i_expert * DATA_SIZE_BYTES)));
        silu(temp_token_0, temp_token_0, inter_dim, n_token);
        gemm(in_token_addr, expert_w3_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES), temp_token_1, dim, n_token, inter_dim, hbm_addr(expert_w3_bias_addr + (inter_dim * i_expert * DATA_SIZE_BYTES)));

        dot_product(temp_token_0, temp_token_1, temp_token_0, inter_dim, n_token);

        gemm(temp_token_0, expert_w2_weights_addr + (inter_dim * dim * i_expert * DATA_SIZE_BYTES), temp_token_0, inter_dim, n_token, dim, hbm_addr(expert_w2_bias_addr + (dim * i_expert * DATA_SIZE_BYTES)));
        // Add shared experts output with weighted routed experts output
        apply_element_wise_2_in(temp_token_0, actual_out_addr, actual_out_addr, dim, n_token, add_op);
    }
    flex_global_barrier_xy();
}

#endif