#ifndef TOP_K_H
#define TOP_K_H

#include "utils_decode.h"

void top_k(const uint64_t in_addr, const uint64_t out_value_addr, const uint64_t out_index_addr, const uint16_t k, const uint16_t dim, const uint16_t n_token, cluster_map_t cluster_map);

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
 * @param in_addr in HBM
 * @param out_value_addr can be in TCDM or HBM
 * @param out_index_addr can be in TCDM or HBM
 * @param k top k values to select (less than 64)
 * @param n_routed_expert number of candidate values in each row of the input matrix
 * @param n_token number of rows in the input matrix
 * @param cluster_map
 */
void top_k(const uint64_t in_addr, const uint64_t out_value_addr, const uint64_t out_index_addr, const uint16_t k, const uint16_t n_routed_expert, const uint16_t n_token, cluster_map_t cluster_map) {
    if (0 == k || 0 == n_routed_expert || 0 == n_token || k > 16) {
        return;
    }
    // flex_global_barrier_xy();
    
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
        uint32_t transfer_rows;

        uint32_t local_out_value, local_out_indices, local_in;
        local_out_value = ARCH_CLUSTER_TCDM_SIZE - ARCH_NUM_CORE_PER_CLUSTER * k * DATA_SIZE_BYTES;
        local_out_indices = local_out_value - ARCH_NUM_CORE_PER_CLUSTER * k * DATA_SIZE_BYTES;
        local_in = local_out_indices - ARCH_NUM_CORE_PER_CLUSTER * n_routed_expert * DATA_SIZE_BYTES;
        local_in += core_id * n_routed_expert * DATA_SIZE_BYTES;
        local_out_indices += core_id * k * DATA_SIZE_BYTES;
        local_out_value += core_id * k * DATA_SIZE_BYTES;

        while (i_row_cluster < n_token) {
            // Allocate an array for indices
            uint16_t indices[256]; // Should be large enough for n_routed_expert
            // one dma transfer per cluster
            transfer_rows = min(ARCH_NUM_CORE_PER_CLUSTER, n_token - i_row_cluster);

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
                    // printf("");
                    indices[i] = i;
                }
                
                // Quickselect to partition the array so that the k largest elements are at the beginning
                quickselect((uint16_t*)local(local_in), indices, 0, n_routed_expert - 1, k - 1);
                
                // // Sort the top k elements (if needed for fully sorted output)
            //     // Simple insertion sort for the small k values
            //     for (int i = 1; i < k; i++) {
            //         uint16_t key_val = ((uint16_t *)local(local_in))[i];
            //         uint16_t key_idx = indices[i];
            //         int j = i - 1;
                    
            //         while (j >= 0 && asm_fp16_compare((const fp16 *)&key_val, (const fp16 *)&(((uint16_t *)local(local_in))[j])) == 1) {
            //             ((uint16_t *)local(local_in))[j + 1] = ((uint16_t *)local(local_in))[j];
            //             indices[j + 1] = indices[j];
            //             j--;
            //         }
                    
            //         ((uint16_t *)local(local_in))[j + 1] = key_val;
            //         indices[j + 1] = key_idx;
            //     }
                // Copy to output buffers
                for (int i = 0; i < k; i++) {
                    // printf("");
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
    // flex_global_barrier_xy();
}


#endif