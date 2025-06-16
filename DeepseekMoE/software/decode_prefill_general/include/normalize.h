// Author: Haoze Gao <gaohao@ethz.ch>

#ifndef NORMALIZE_H
#define NORMALIZE_H

#include "utils_decode.h"
#include "element_wise.h"

void normalize(const uint64_t in_addr, const uint64_t out_addr, const uint16_t dim, const uint16_t n_token, cluster_map_t cluster_map);

/**
 * @brief normalize the input matrix along rows. Consider dim here to be small, just use the simplist way to implement it.
 * 
 * @param in_addr can be in TCDM or HBM
 * @param out_addr can be in TCDM or HBM
 * @param dim colomn dimension of the input matrix
 * @param n_token row dimension of the input matrix
 * @param cluster_map
 */
void normalize(const uint64_t in_addr, const uint64_t out_addr, const uint16_t dim, const uint16_t n_token, cluster_map_t cluster_map) {
    if (0 == dim || 0 == n_token) {
        return;
    }
    // flex_global_barrier_xy();
    
    uint32_t cluster_id = flex_get_cluster_id();
    // uint32_t core_id = ARCH_NUM_CORE_PER_CLUSTER - flex_get_core_id() - 1;  // reverse the core id to make the dm core the first core in the cluster
    uint32_t core_id = flex_get_core_id();
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
        uint64_t i_row_cluster = local_cluster_id;
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

            #ifdef SPATZ_ENABLE
            if (0 == core_id) {
                uint16_t * local_in_ptr = (uint16_t *)local(local_out);
                uint16_t * local_out_ptr = (uint16_t *)local(local_out);
                
                asm volatile("vsetvli zero, %0, e16, m8, ta, ma" : : "r"(dim));
                asm volatile("fmv.h.x ft0, %0" : : "r"(((fp16 *)local(local_sum))[0]));
                // compute element-wise operation with spatz core
                uint16_t vl;
                uint16_t i_element = 0;
                while (dim > i_element) {
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
            #endif
            flex_intra_cluster_sync();
            // transfer the top k values and indices to HBM
            if (flex_is_dm_core()) {
                flex_dma_async_1d((out_addr + i_row_cluster * dim * DATA_SIZE_BYTES), local(local_out), dim * DATA_SIZE_BYTES);
                flex_dma_async_wait_all();
            }
            i_row_cluster += n_cluster_activated;
        }
    }
    // flex_global_barrier_xy();
}

#endif 