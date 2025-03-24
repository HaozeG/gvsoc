#ifndef MOE_DECODE_H
#define MOE_DECODE_H

#include "flex_runtime.h"
#include "flex_printf.h"
#include "flex_dma_pattern.h"
#include "flex_group_barrier.h"
#include "flex_libfp16.h"

// use float16 as data type
#define DATA_SIZE_BYTES 2
// Parameters for GEMV
#define TILE_WIDTH 64
#define OPAND_SIZE TILE_WIDTH * TILE_WIDTH * DATA_SIZE_BYTES
// Parameter for element-wise functions
#define ELEMENT_WISE_TILE_WIDTH 8
#define NUM_CLUSTER_X 4
#define NUM_CLUSTER_Y 4

typedef void (*element_wise_op_1_in_t)(const fp16* input, fp16* output);
typedef void (*element_wise_op_2_in_t)(const fp16* input1, const fp16* input2, fp16* output);
typedef void (*element_wise_op_2_in_const_t)(const fp16* input1, const fp16* in_const, fp16* output);
// designed for NUM_CLUSTER = 16
typedef uint16_t cluster_map_t;
// 2.5 in float
fp16 route_scale = (fp16)0x4100;

void compute_moe(uint32_t in_token_addr, uint16_t n_token, uint16_t dim, uint16_t inter_dim, uint16_t n_routed_experts, uint16_t n_shared_experts, uint16_t n_activated_experts, uint32_t gate_weights_addr, uint32_t expert_w1_weights_addr, uint32_t expert_w1_bias_addr, uint32_t expert_w2_weights_addr, uint32_t expert_w2_bias_addr, uint32_t expert_w3_weights_addr, uint32_t expert_w3_bias_addr, uint32_t actual_out_addr);
void gemv(const uint32_t A, const uint32_t B, const uint32_t C, const uint32_t K, const uint32_t M, const uint32_t N, const uint32_t bias_addr, cluster_map_t cluster_map);
void top_k(const uint32_t in_addr, const uint32_t out_value_addr, const uint32_t out_index_addr, const uint32_t k, const uint32_t dim, const uint32_t n_token, cluster_map_t cluster_map);
void normalize(const uint32_t in_addr, const uint32_t out_addr, const uint32_t dim, const uint32_t n_token, cluster_map_t cluster_map);
void silu(const uint32_t in_addr, const uint32_t out_addr, const uint32_t dim, const uint32_t n_token, cluster_map_t cluster_map);
void sigmoid(const uint32_t in_addr, const uint32_t out_addr, const uint32_t dim, const uint32_t n_token, cluster_map_t cluster_map);
void add(const uint32_t in_addr_1, const uint32_t in_addr_2, const uint32_t out_addr, const uint32_t dim, const uint32_t n_token, cluster_map_t cluster_map);
void dot_product(const uint32_t in_addr_1, const uint32_t in_addr_2, const uint32_t out_addr, const uint32_t dim, const uint32_t n_token, cluster_map_t cluster_map);
void dot_product_const(const uint32_t in_addr, const fp16 in_addr_const, const uint32_t out_addr, const uint32_t dim, const uint32_t n_token, cluster_map_t cluster_map);

void apply_element_wise_1_in(const uint32_t in_addr, const uint32_t out_addr, const uint32_t dim, const uint32_t n_token, element_wise_op_1_in_t op, cluster_map_t cluster_map);
void apply_element_wise_2_in_const(const uint32_t in_addr, const fp16 in_const, const uint32_t out_addr, const uint32_t dim, const uint32_t n_token, element_wise_op_2_in_const_t op, cluster_map_t cluster_map);
void apply_element_wise_2_in(const uint32_t in_addr1, const uint32_t in_addr2, const uint32_t out_addr, const uint32_t dim, const uint32_t n_token, element_wise_op_2_in_t op, cluster_map_t cluster_map);
void silu_op(const fp16* input, fp16* output);
void sigmoid_op(const fp16* input, fp16* output);
void mul_op(const fp16* input1, const fp16* input2, fp16* output);
void add_op(const fp16* input1, const fp16* input2, fp16* output);

void broadcast_to_all_clusters(uint32_t dst_addr, uint32_t src_addr, uint32_t size);

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
void gemv(const uint32_t A, const uint32_t B, const uint32_t C, const uint32_t K, const uint32_t M, const uint32_t N, const uint32_t bias_addr, cluster_map_t cluster_map) {
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
        accumulator = ARCH_CLUSTER_TCDM_SIZE - OPAND_SIZE;
        local_A_0 = accumulator - OPAND_SIZE;
        local_A_1 = local_A_0 - OPAND_SIZE;
        local_B_0 = local_A_1 - OPAND_SIZE;
        local_B_1 = local_B_0 - OPAND_SIZE;

        int m_remaining, n_remaining, m_tile, n_tile, k_tile, bK;
        // tile refer to tile of output a single cluster process on
        int tile_width = TILE_WIDTH;
        // block refer to block of output that all clusters process on at the same time
        int block_width_j = tile_width * n_cluster_activated;
        int block_width_i = tile_width;
        // For double buffering: indicates the suffix of buffer to use
        bool is_odd = 1;

        // each cluster compute a tile of the output matrix
        int gi = 0;
        int gj = local_cluster_id;

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
                
                m_tile = fmin(tile_width, m_remaining - gi * tile_width);
                n_tile = fmin(tile_width, n_remaining - gj * tile_width);
                
                if(flex_is_dm_core()) {
                    if (bias_addr == zomem(0)) {
                        flex_dma_async_1d(local(accumulator), zomem(0), m_tile * n_tile * DATA_SIZE_BYTES);
                        flex_dma_async_wait_all();
                    } else {
                        flex_dma_sync_2d(local(accumulator), bias_addr + (gj * tile_width + j * block_width_j) * DATA_SIZE_BYTES, n_tile*DATA_SIZE_BYTES, n_tile*DATA_SIZE_BYTES, 0, m_tile);
                    }
                }
                if(flex_is_first_core()) {
                    // flex_redmule_config() usage: [m_size, n_size] * [n_size, k_size] = [m_size, k_size]
                    flex_redmule_config(m_tile, tile_width, n_tile);
                }
                
                uint32_t cluster_offset;
                // cluster_offset = cluster_id / 4 * ARCH_HBM_NODE_ADDR_SPACE;

                // Define regular access pattern for HBM nodes
                // TODO: Remove hard-coded values for scalability
                cluster_offset = 0;
                switch (cluster_id) {
                    case 0:
                    case 2:
                        cluster_offset = 0 * ARCH_HBM_NODE_ADDR_SPACE;
                        break;
                    
                    case 5:
                    case 7:
                        cluster_offset = 1 * ARCH_HBM_NODE_ADDR_SPACE;
                        break;
                    
                    case 8:
                    case 10:
                        cluster_offset = 2 * ARCH_HBM_NODE_ADDR_SPACE;
                        break;
                    
                    case 13:
                    case 15:
                        cluster_offset = 3 * ARCH_HBM_NODE_ADDR_SPACE;
                        break;
                    
                    case 4:
                    case 12:
                        cluster_offset = ARCH_HBM_NODE_ADDR_SPACE * (2 * NUM_CLUSTER_Y + NUM_CLUSTER_X);
                        break;
                    
                    case 1:
                    case 9:
                        cluster_offset = ARCH_HBM_NODE_ADDR_SPACE * (2 * NUM_CLUSTER_Y + NUM_CLUSTER_X) + ARCH_HBM_NODE_ADDR_SPACE;
                        break;
                    
                    case 6:
                    case 14:
                        cluster_offset = ARCH_HBM_NODE_ADDR_SPACE * (2 * NUM_CLUSTER_Y + NUM_CLUSTER_X) + 2 * ARCH_HBM_NODE_ADDR_SPACE;
                        break;
                    
                    case 3:
                    case 11:
                        cluster_offset = ARCH_HBM_NODE_ADDR_SPACE * (2 * NUM_CLUSTER_Y + NUM_CLUSTER_X) + 3 * ARCH_HBM_NODE_ADDR_SPACE;
                        break;

                    default:
                        cluster_offset = 0;
                        break;
                }
                

                // bK: inner loop tile computing partial sums
                for (bK = 0; bK < K; bK += tile_width) {
                    k_tile = fmin(tile_width, K - bK);

                    // SoftHier_HBM -> SoftHier_TCDM 2D
                    if(flex_is_dm_core()) {
                        uint32_t load_dest_A, load_dest_B;
                        if (is_odd) {
                            load_dest_A = local_A_0;
                            load_dest_B = local_B_0;
                        } else {
                            load_dest_A = local_A_1;
                            load_dest_B = local_B_1;
                        }
                        flex_dma_sync_2d(local(load_dest_A), A + ((K * (tile_width * gi + i * block_width_i)) + bK) * DATA_SIZE_BYTES, k_tile*DATA_SIZE_BYTES, k_tile*DATA_SIZE_BYTES, K*DATA_SIZE_BYTES, m_tile);
                        flex_dma_sync_2d(local(load_dest_B), B + (N * bK + tile_width * gj + j * block_width_j) * DATA_SIZE_BYTES + cluster_offset, n_tile*DATA_SIZE_BYTES,  n_tile*DATA_SIZE_BYTES, N*DATA_SIZE_BYTES, k_tile);
                    }
                    
                    // make sure data is ready
                    flex_intra_cluster_sync();

                    if (flex_is_first_core()) {
                        // change configuration if the tile in K dimension is not full
                        if (k_tile != tile_width) {
                            // flex_redmule_config() usage: [m_size, n_size] * [n_size, k_size] = [m_size, k_size]
                            flex_redmule_config(m_tile, k_tile, n_tile);
                        }

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


// Helper function for quickselect partitioning
int partition(uint16_t* arr, uint16_t* indices, int left, int right) {
    // Choose rightmost element as pivot
    uint16_t pivot = arr[right];
    int i = left - 1;
    
    for (int j = left; j < right; j++) {
        // Use fp16 comparison to maintain fp16 ordering
        if (asm_fp16_compare((const fp16 *)(arr + j), (const fp16 *)&pivot) >= 0) {
            i++;
            // Swap values
            uint16_t temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
            
            // Swap indices if tracking
            if (indices != NULL) {
                temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
            }
        }
    }
    
    // Swap pivot to its final position
    uint16_t temp = arr[i + 1];
    arr[i + 1] = arr[right];
    arr[right] = temp;
    
    // Swap indices if tracking
    if (indices != NULL) {
        temp = indices[i + 1];
        indices[i + 1] = indices[right];
        indices[right] = temp;
    }
    
    return i + 1;
}

// Helper function to implement quickselect for top-k
void quickselect(uint16_t* arr, uint16_t* indices, int left, int right, int k) {
    if (left < right) {
        int pivot_idx = partition(arr, indices, left, right);
        
        // If pivot is at k, we're done
        if (pivot_idx == k) {
            return;
        }
        // If pivot index is greater than k, search left subarray
        else if (pivot_idx > k) {
            quickselect(arr, indices, left, pivot_idx - 1, k);
        }
        // If pivot index is less than k, search right subarray
        else {
            quickselect(arr, indices, pivot_idx + 1, right, k);
        }
    }
}

/**
 * @brief Select top k values from each row of the input matrix and write the result to the output matrix along with the corresponding indices.
 * 
 * @param in_addr 
 * @param out_value_addr 
 * @param out_index_addr 
 * @param k top k values to select (less than 64)
 * @param n_routed_expert number of candidate values in each row of the input matrix
 * @param n_token number of rows in the input matrix
 * @param cluster_map
 */
void top_k(const uint32_t in_addr, const uint32_t out_value_addr, const uint32_t out_index_addr, const uint32_t k, const uint32_t n_routed_expert, const uint32_t n_token, cluster_map_t cluster_map) {
    if (0 == k || 0 == n_routed_expert || 0 == n_token || k > 16) {
        return;
    }
    flex_global_barrier_xy();
    
    uint32_t cluster_id = flex_get_cluster_id();
    uint32_t core_id = ARCH_NUM_CORE_PER_CLUSTER - flex_get_core_id() - 1;  // reverse the core id to make the dm core the first core in the cluster
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
        uint32_t i_row_cluster = local_cluster_id * ARCH_NUM_CORE_PER_CLUSTER;
        uint16_t transfer_rows;

        uint32_t local_out_value, local_out_indices, local_in;
        local_out_value = ARCH_CLUSTER_TCDM_SIZE - ARCH_NUM_CORE_PER_CLUSTER * k * DATA_SIZE_BYTES;
        local_out_indices = local_out_value - ARCH_NUM_CORE_PER_CLUSTER * k * DATA_SIZE_BYTES;
        local_in = local_out_indices - ARCH_NUM_CORE_PER_CLUSTER * n_routed_expert * DATA_SIZE_BYTES;
        local_in += core_id * n_routed_expert * DATA_SIZE_BYTES;
        local_out_indices += core_id * k * DATA_SIZE_BYTES;
        local_out_value += core_id * k * DATA_SIZE_BYTES;

        // Allocate an array for indices
        uint16_t indices[256]; // Should be large enough for n_routed_expert

        while (i_row_cluster < n_token) {
            // one dma transfer per cluster
            transfer_rows = fmin(ARCH_NUM_CORE_PER_CLUSTER, n_token - i_row_cluster);

            if (flex_is_dm_core()) {
                flex_dma_async_1d(local(local_in), 
                                 (in_addr + i_row_cluster * n_routed_expert * DATA_SIZE_BYTES), 
                                 transfer_rows * n_routed_expert * DATA_SIZE_BYTES);
                flex_dma_async_wait_all();
            }
            flex_intra_cluster_sync();

            if ((i_row_cluster + core_id) < n_token) {
                // Initialize index array
                for (int i = 0; i < n_routed_expert; i++) {
                    indices[i] = i;
                }
                
                // Quickselect to partition the array so that the k largest elements are at the beginning
                quickselect((uint16_t*)local(local_in), indices, 0, n_routed_expert - 1, k - 1);
                
                // Sort the top k elements (if needed for fully sorted output)
                // Simple insertion sort for the small k values
                for (int i = 1; i < k; i++) {
                    uint16_t key_val = ((uint16_t *)local(local_in))[i];
                    uint16_t key_idx = indices[i];
                    int j = i - 1;
                    
                    while (j >= 0 && asm_fp16_compare((const fp16 *)&key_val, (const fp16 *)&(((uint16_t *)local(local_in))[j])) == 1) {
                        ((uint16_t *)local(local_in))[j + 1] = ((uint16_t *)local(local_in))[j];
                        indices[j + 1] = indices[j];
                        j--;
                    }
                    
                    ((uint16_t *)local(local_in))[j + 1] = key_val;
                    indices[j + 1] = key_idx;
                }
                
                // Copy to output buffers
                for (int i = 0; i < k; i++) {
                    ((uint16_t *)local(local_out_value))[i] = ((uint16_t *)local(local_in))[i];
                    ((uint16_t *)local(local_out_indices))[i] = indices[i];
                }
            }
            
            flex_intra_cluster_sync();
            // transfer the top k values and indices to HBM
            if (flex_is_dm_core()) {
                flex_dma_async_1d((out_value_addr + i_row_cluster * k * DATA_SIZE_BYTES), 
                local(local_out_value), transfer_rows * k * DATA_SIZE_BYTES);
                flex_dma_async_1d((out_index_addr + i_row_cluster * k * DATA_SIZE_BYTES), 
                local(local_out_indices), transfer_rows * k * DATA_SIZE_BYTES);
                flex_dma_async_wait_all();
            }
            
            i_row_cluster += ARCH_NUM_CORE_PER_CLUSTER * n_cluster_activated;
        }
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
 * @param cluster_map
 */
void normalize(const uint32_t in_addr, const uint32_t out_addr, const uint32_t dim, const uint32_t n_token, cluster_map_t cluster_map) {
    if (0 == dim || 0 == n_token) {
        return;
    }
    flex_global_barrier_xy();
    
    uint32_t cluster_id = flex_get_cluster_id();
    uint32_t core_id = ARCH_NUM_CORE_PER_CLUSTER - flex_get_core_id() - 1;  // reverse the core id to make the dm core the first core in the cluster
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
        uint32_t i_row_cluster = local_cluster_id;
        uint16_t transfer_rows;

        uint32_t local_out, local_sum;
        local_out = ARCH_CLUSTER_TCDM_SIZE - dim * DATA_SIZE_BYTES;
        local_sum = local_out - DATA_SIZE_BYTES;

        while (i_row_cluster < n_token) {
            // Transfer one row per cluster
            if (flex_is_dm_core()) {
                flex_dma_async_1d(local(local_out), (in_addr + i_row_cluster * dim * DATA_SIZE_BYTES), dim * DATA_SIZE_BYTES);
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
                flex_dma_async_1d((out_addr + i_row_cluster * dim * DATA_SIZE_BYTES), local(local_out), dim * DATA_SIZE_BYTES);
                flex_dma_async_wait_all();
            }
            i_row_cluster += n_cluster_activated;
        }
    }
    flex_global_barrier_xy();
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
void apply_element_wise_1_in(const uint32_t in_addr, const uint32_t out_addr, const uint32_t dim, const uint32_t n_token, element_wise_op_1_in_t op, cluster_map_t cluster_map) {
    if (0 == dim || 0 == n_token || NULL == op) {
        return;
    }
    flex_global_barrier_xy();
    
    uint32_t cluster_id = flex_get_cluster_id();
    uint32_t core_id = ARCH_NUM_CORE_PER_CLUSTER - flex_get_core_id() - 1;  // reverse the core id to make the dm core the first core in the cluster
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
        int n_element_per_cluster, n_element_per_core, exec_cycle_per_cluster;
        exec_cycle_per_cluster = (dim * n_token - 1) / (n_cluster_activated * ARCH_NUM_CORE_PER_CLUSTER * ELEMENT_WISE_TILE_WIDTH) + 1;
        // index of the first element to be processed by current cluster
        uint32_t i_element_cluster = local_cluster_id * exec_cycle_per_cluster * ARCH_NUM_CORE_PER_CLUSTER * ELEMENT_WISE_TILE_WIDTH;
        n_element_per_cluster = fmin(exec_cycle_per_cluster * ARCH_NUM_CORE_PER_CLUSTER * ELEMENT_WISE_TILE_WIDTH, dim * n_token - i_element_cluster);
        if (n_element_per_cluster > 0) {
            uint32_t local_in_0, local_in_1, local_out_0, local_out_1;
            uint32_t load_dest, store_src;
            local_in_0 = ARCH_CLUSTER_TCDM_SIZE - n_element_per_cluster * DATA_SIZE_BYTES;
            local_in_1 = local_in_0 - n_element_per_cluster * DATA_SIZE_BYTES;
            local_out_0 = local_in_1 - n_element_per_cluster * DATA_SIZE_BYTES;
            local_out_1 = local_out_0 - n_element_per_cluster * DATA_SIZE_BYTES;
            
            load_dest = local_in_0;
            store_src = local_out_0;
            // load data: one dma transfer per cluster
            if (flex_is_dm_core()) {
                flex_dma_async_1d(local(load_dest), in_addr + i_element_cluster * DATA_SIZE_BYTES, n_element_per_cluster * DATA_SIZE_BYTES);
                flex_dma_async_wait_all();
            }
            
            flex_intra_cluster_sync();
            // index in local memory
            int i_element_core = core_id * ELEMENT_WISE_TILE_WIDTH;
            while (i_element_core < n_element_per_cluster) {
                // compute element-wise operation
                n_element_per_core = fmin(ELEMENT_WISE_TILE_WIDTH, n_element_per_cluster - i_element_core);
                for (int i = 0; i < n_element_per_core; i++) {
                    op((const fp16*)local(load_dest + (i + i_element_core) * DATA_SIZE_BYTES), 
                            (fp16*)local(store_src + (i + i_element_core) * DATA_SIZE_BYTES));
                }    
                i_element_core += ELEMENT_WISE_TILE_WIDTH * ARCH_NUM_CORE_PER_CLUSTER;
            }
            flex_intra_cluster_sync();
            // store data: one dma transfer per cluster
            if (flex_is_dm_core()) {
                flex_dma_async_1d(out_addr + i_element_cluster * DATA_SIZE_BYTES, local(store_src), n_element_per_cluster * DATA_SIZE_BYTES);
                flex_dma_async_wait_all();
            }
        }
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
 * @param cluster_map
 */
void apply_element_wise_2_in(const uint32_t in_addr1, const uint32_t in_addr2, const uint32_t out_addr, const uint32_t dim, const uint32_t n_token, element_wise_op_2_in_t op, cluster_map_t cluster_map) {
    if (0 == dim || 0 == n_token || NULL == op) {
        return;
    }
    flex_global_barrier_xy();
    uint32_t local_in1, local_in2, local_out;
    local_in1 = ARCH_CLUSTER_TCDM_SIZE - ARCH_NUM_CORE_PER_CLUSTER * ELEMENT_WISE_TILE_WIDTH * DATA_SIZE_BYTES;
    local_in2 = local_in1 - ARCH_NUM_CORE_PER_CLUSTER * ELEMENT_WISE_TILE_WIDTH * DATA_SIZE_BYTES;
    local_out = local_in2 - ARCH_NUM_CORE_PER_CLUSTER * ELEMENT_WISE_TILE_WIDTH * DATA_SIZE_BYTES;

    uint32_t cluster_id = flex_get_cluster_id();
    uint32_t core_id = ARCH_NUM_CORE_PER_CLUSTER - flex_get_core_id() - 1;  // reverse the core id to make the dm core the first core in the cluster
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
        // index of the first element to be processed by current cluster
        uint32_t i_element_cluster = local_cluster_id * ARCH_NUM_CORE_PER_CLUSTER * ELEMENT_WISE_TILE_WIDTH;
        uint32_t n_element_per_cluster, n_element_per_core;

        while (i_element_cluster < n_token * dim) {
            // load data: one dma transfer per cluster
            n_element_per_cluster = fmin(ARCH_NUM_CORE_PER_CLUSTER * ELEMENT_WISE_TILE_WIDTH, dim * n_token - i_element_cluster);
            if (flex_is_dm_core()) {
                flex_dma_async_1d(local(local_in1), in_addr1 + i_element_cluster * DATA_SIZE_BYTES, n_element_per_cluster * DATA_SIZE_BYTES);
                flex_dma_async_1d(local(local_in2), in_addr2 + i_element_cluster * DATA_SIZE_BYTES, n_element_per_cluster * DATA_SIZE_BYTES);
                flex_dma_async_wait_all();
            }
            
            // compute element-wise operation
            n_element_per_core = fmin(ELEMENT_WISE_TILE_WIDTH, n_element_per_cluster - core_id * ELEMENT_WISE_TILE_WIDTH);
            flex_intra_cluster_sync();
            for (int i = 0; i < n_element_per_core; i++) {
                int idx = i + core_id * ELEMENT_WISE_TILE_WIDTH;
                op((const fp16*)local(local_in1 + idx * DATA_SIZE_BYTES), (const fp16*)local(local_in2 + idx * DATA_SIZE_BYTES), (fp16*)local(local_out + idx * DATA_SIZE_BYTES));
            }
            flex_intra_cluster_sync();

            // store data: one dma transfer per cluster
            if (flex_is_dm_core()) {
                flex_dma_async_1d(out_addr + i_element_cluster * DATA_SIZE_BYTES, local(local_out), n_element_per_cluster * DATA_SIZE_BYTES);
                flex_dma_async_wait_all();
            }
            i_element_cluster += ARCH_NUM_CORE_PER_CLUSTER * n_cluster_activated * ELEMENT_WISE_TILE_WIDTH;
        }
    }
    flex_global_barrier_xy();
}

/**
 * @brief apply element-wise operation on TWO INPUT matrix. Each core processes ELEMENT_WISE_TILE_WIDTH elements at a time.
 * 
 * @param in_addr
 * @param in_const constant value to be multiplied
 * @param out_addr 
 * @param dim 
 * @param n_token 
 * @param op element_wise_op_t function pointer
 * @param cluster_map
 */
void apply_element_wise_2_in_const(const uint32_t in_addr, const fp16 in_const, const uint32_t out_addr, const uint32_t dim, const uint32_t n_token, element_wise_op_2_in_const_t op, cluster_map_t cluster_map) {
    if (0 == dim || 0 == n_token || NULL == op) {
        return;
    }
    flex_global_barrier_xy();
    uint32_t local_in, local_out;
    local_in = ARCH_CLUSTER_TCDM_SIZE - ARCH_NUM_CORE_PER_CLUSTER * ELEMENT_WISE_TILE_WIDTH * DATA_SIZE_BYTES;
    local_out = local_in - ARCH_NUM_CORE_PER_CLUSTER * ELEMENT_WISE_TILE_WIDTH * DATA_SIZE_BYTES;

    uint32_t cluster_id = flex_get_cluster_id();
    uint32_t core_id = ARCH_NUM_CORE_PER_CLUSTER - flex_get_core_id() - 1;  // reverse the core id to make the dm core the first core in the cluster
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
        // index of the first element to be processed by current cluster
        uint32_t i_element_cluster = local_cluster_id * ARCH_NUM_CORE_PER_CLUSTER * ELEMENT_WISE_TILE_WIDTH;
        uint32_t n_element_per_cluster, n_element_per_core;

        while (i_element_cluster < n_token * dim) {
            // load data: one dma transfer per cluster
            n_element_per_cluster = fmin(ARCH_NUM_CORE_PER_CLUSTER * ELEMENT_WISE_TILE_WIDTH, dim * n_token - i_element_cluster);
            if (flex_is_dm_core()) {
                flex_dma_async_1d(local(local_in), in_addr + i_element_cluster * DATA_SIZE_BYTES, n_element_per_cluster * DATA_SIZE_BYTES);
                flex_dma_async_wait_all();
            }
            
            // compute element-wise operation
            n_element_per_core = fmin(ELEMENT_WISE_TILE_WIDTH, n_element_per_cluster - core_id * ELEMENT_WISE_TILE_WIDTH);
            flex_intra_cluster_sync();
            for (int i = 0; i < n_element_per_core; i++) {
                int idx = i + core_id * ELEMENT_WISE_TILE_WIDTH;
                op((const fp16*)local(local_in + idx * DATA_SIZE_BYTES), (fp16*) &in_const, (fp16*)local(local_out + idx * DATA_SIZE_BYTES));
            }
            flex_intra_cluster_sync();

            // store data: one dma transfer per cluster
            if (flex_is_dm_core()) {
                flex_dma_async_1d(out_addr + i_element_cluster * DATA_SIZE_BYTES, local(local_out), n_element_per_cluster * DATA_SIZE_BYTES);
                flex_dma_async_wait_all();
            }
            i_element_cluster += ARCH_NUM_CORE_PER_CLUSTER * n_cluster_activated * ELEMENT_WISE_TILE_WIDTH;
        }
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
 * @param cluster_map
 */
void silu(const uint32_t in_addr, const uint32_t out_addr, const uint32_t dim, const uint32_t n_token, cluster_map_t cluster_map) {
    apply_element_wise_1_in(in_addr, out_addr, dim, n_token, silu_op, cluster_map); 
}

/**
 * @brief element-wise sigmoid function
 * 
 * @param in_addr 
 * @param out_addr 
 * @param dim 
 * @param n_token 
 * @param cluster_map
 */
void sigmoid(const uint32_t in_addr, const uint32_t out_addr, const uint32_t dim, const uint32_t n_token, cluster_map_t cluster_map) {
    apply_element_wise_1_in(in_addr, out_addr, dim, n_token, sigmoid_op, cluster_map); 
}

/**
 * @brief element-wise dot-product function
 * 
 * @param in_addr_1 
 * @param in_addr_2 
 * @param out_addr 
 * @param dim 
 * @param n_token 
 * @param cluster_map
 */
void dot_product(const uint32_t in_addr_1, const uint32_t in_addr_2, const uint32_t out_addr, const uint32_t dim, const uint32_t n_token, cluster_map_t cluster_map) {
    apply_element_wise_2_in(in_addr_1, in_addr_2, out_addr, dim, n_token, mul_op, cluster_map);
}

/**
 * @brief dot-product with constant function
 * 
 * @param in_addr 
 * @param in_const fp16 constant value to be multiplied
 * @param out_addr 
 * @param dim 
 * @param n_token 
 * @param cluster_map
 */
void dot_product_const(const uint32_t in_addr, const fp16 in_const, const uint32_t out_addr, const uint32_t dim, const uint32_t n_token, cluster_map_t cluster_map) {
    apply_element_wise_2_in_const(in_addr, in_const, out_addr, dim, n_token, mul_op, cluster_map);
}

/**
 * @brief element-wise add function
 * 
 * @param in_addr_1 
 * @param in_addr_2 
 * @param out_addr 
 * @param dim 
 * @param n_token 
 * @param cluster_map
 */
void add(const uint32_t in_addr_1, const uint32_t in_addr_2, const uint32_t out_addr, const uint32_t dim, const uint32_t n_token, cluster_map_t cluster_map) {
    apply_element_wise_2_in(in_addr_1, in_addr_2, out_addr, dim, n_token, add_op, cluster_map);
}

void silu_op(const fp16* input, fp16* output) {
    asm_fp16_sigmoid(input, output);
    float fa = fp16_to_float(*output);
    float fb = fp16_to_float(*input);
    *output = float_to_fp16(fa * fb);
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
    for (int cid = 0; cid < ARCH_NUM_CLUSTER_X; ++cid)
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

void compute_moe(uint32_t in_token_addr, uint16_t n_token, uint16_t dim, uint16_t inter_dim, uint16_t n_routed_experts, uint16_t n_shared_experts, uint16_t n_activated_experts, uint32_t gate_weights_addr, uint32_t expert_w1_weights_addr, uint32_t expert_w1_bias_addr, uint32_t expert_w2_weights_addr, uint32_t expert_w2_bias_addr, uint32_t expert_w3_weights_addr, uint32_t expert_w3_bias_addr, uint32_t actual_out_addr) {
    cluster_map_t cluster_coloring_0, cluster_coloring_1, cluster_all;
    cluster_coloring_0 = 0x5A5A;    // 0101101001011010: 1 3 4 6 9 11 12 14
    cluster_coloring_1 = 0xA5A5;    // 1010010110100101: 0 2 5 7 8 10 13 15
    cluster_all = 0xFFFF;
    uint32_t top_k_weights_addr, top_k_indices_addr;
    uint32_t temp_token_0, temp_token_1;
    top_k_weights_addr = actual_out_addr + n_token * dim * DATA_SIZE_BYTES;
    top_k_indices_addr = top_k_weights_addr + n_token * n_activated_experts * DATA_SIZE_BYTES;
    temp_token_0 = top_k_indices_addr + n_token * n_activated_experts * DATA_SIZE_BYTES;
    temp_token_1 = temp_token_0 + n_token * dim * DATA_SIZE_BYTES;
    
    // Gate 
    flex_global_barrier_xy();
    gemv(hbm_addr(in_token_addr), hbm_addr(gate_weights_addr), hbm_addr(temp_token_0), dim, n_token, n_routed_experts, zomem(0), cluster_all);
    // flex_global_barrier_xy();
    // return;
    // top_k(hbm_addr(temp_token_0), hbm_addr(top_k_weights_addr), hbm_addr(top_k_indices_addr), n_activated_experts, n_routed_experts, n_token, cluster_all);
    // flex_global_barrier_xy();
    // sigmoid
    // sigmoid(hbm_addr(top_k_weights_addr), hbm_addr(top_k_weights_addr), n_activated_experts, n_token, cluster_all);
    // flex_global_barrier_xy();
    // normalize
    // normalize(hbm_addr(top_k_weights_addr), hbm_addr(top_k_weights_addr), n_activated_experts, n_token, cluster_all);
    // flex_global_barrier_xy();

    // Routed experts
    // self.w2.forward(silu(self.w1.forward(x)) * self.w3.forward(x))
    uint16_t i_expert;
    i_expert = 0;
    fp16 w_expert;
    int i = 0;
    while (i < n_activated_experts) {
        // TODO: check i, w
        // i_expert = ((uint16_t *)hbm_addr(top_k_indices_addr))[i];
        // w_expert = ((fp16 *)hbm_addr(top_k_weights_addr))[i];
        // mul_op(&w_expert, &route_scale, &w_expert);
        
        // w1.forward(x)
        flex_global_barrier_xy();
        gemv(hbm_addr(in_token_addr), hbm_addr(expert_w1_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES)), hbm_addr(temp_token_0), dim, n_token, inter_dim, hbm_addr(expert_w1_bias_addr + (inter_dim * i_expert * DATA_SIZE_BYTES)), cluster_coloring_0);
        
        // w3.forward(x)
        gemv(hbm_addr(in_token_addr), hbm_addr(expert_w3_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES)), hbm_addr(temp_token_1), dim, n_token, inter_dim, hbm_addr(expert_w3_bias_addr + (inter_dim * i_expert * DATA_SIZE_BYTES)), cluster_coloring_1);
        flex_global_barrier_xy();
        
        // flex_global_barrier_xy();
        // silu(w1.forward(x))
        // silu(hbm_addr(temp_token_0), hbm_addr(temp_token_0), inter_dim, n_token, cluster_all);
        // flex_global_barrier_xy();
        // silu(w1.forward(x)) * w3.forward(x)
        // dot_product(hbm_addr(temp_token_0), hbm_addr(temp_token_1), hbm_addr(temp_token_0), inter_dim, n_token, cluster_all);
        
        // flex_global_barrier_xy();
        // w2.forward(silu(w1.forward(x)) * w3.forward(x))
        gemv(hbm_addr(temp_token_0), hbm_addr(expert_w2_weights_addr + (inter_dim * dim * i_expert * DATA_SIZE_BYTES)), hbm_addr(temp_token_0), inter_dim, n_token, dim, hbm_addr(expert_w2_bias_addr + (dim * i_expert * DATA_SIZE_BYTES)), cluster_all);
        
        // multiply by gate weight and add to the output
        // dot_product_const(hbm_addr(temp_token_0), w_expert, hbm_addr(temp_token_0), dim, n_token, cluster_all);
        // apply_element_wise_2_in_const(temp_token_0, w_expert, temp_token_0, dim, n_token, mul_op);
        // add(hbm_addr(temp_token_0), hbm_addr(actual_out_addr), hbm_addr(actual_out_addr), dim, n_token, cluster_all);
        // apply_element_wise_2_in(temp_token_0, actual_out_addr, actual_out_addr, dim, n_token, add_op);
        // flex_global_barrier_xy();
        
        // TEST
        i_expert++;
        // w_expert++;

        i++;
    }
        
    // Shared experts
    // self.w2.forward(silu(self.w1.forward(x)) * self.w3.forward(x))
    for (int i_expert = n_routed_experts; i_expert < (n_routed_experts + n_shared_experts); i_expert++) {
        // w1.forward(x)
        flex_global_barrier_xy();
        gemv(hbm_addr(in_token_addr), hbm_addr(expert_w1_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES)), hbm_addr(temp_token_0), dim, n_token, inter_dim, hbm_addr(expert_w1_bias_addr + (inter_dim * i_expert * DATA_SIZE_BYTES)), cluster_coloring_0);
        
        // w3.forward(x)
        gemv(hbm_addr(in_token_addr), hbm_addr(expert_w3_weights_addr + (dim * inter_dim * i_expert * DATA_SIZE_BYTES)), hbm_addr(temp_token_1), dim, n_token, inter_dim, hbm_addr(expert_w3_bias_addr + (inter_dim * i_expert * DATA_SIZE_BYTES)), cluster_coloring_1);
        flex_global_barrier_xy();
        
        // flex_global_barrier_xy();
        // silu(w1.forward(x))
        // silu(hbm_addr(temp_token_0), hbm_addr(temp_token_0), inter_dim, n_token, cluster_all);
        // flex_global_barrier_xy();
        // silu(w1.forward(x)) * w3.forward(x)
        // dot_product(hbm_addr(temp_token_0), hbm_addr(temp_token_1), hbm_addr(temp_token_0), inter_dim, n_token, cluster_all);
        
        // flex_global_barrier_xy();
        // w2.forward(silu(w1.forward(x)) * w3.forward(x))
        gemv(hbm_addr(temp_token_0), hbm_addr(expert_w2_weights_addr + (inter_dim * dim * i_expert * DATA_SIZE_BYTES)), hbm_addr(temp_token_0), inter_dim, n_token, dim, hbm_addr(expert_w2_bias_addr + (dim * i_expert * DATA_SIZE_BYTES)), cluster_all);
        
        // add to the output
        // add(hbm_addr(temp_token_0), hbm_addr(actual_out_addr), hbm_addr(actual_out_addr), dim, n_token, cluster_all);
        // flex_global_barrier_xy();
    }
}

#endif