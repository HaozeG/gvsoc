// Author: Haoze Gao <gaohao@student.ethz.ch>

#ifndef ELEMENT_WISE_H
#define ELEMENT_WISE_H

#include "utils_decode.h"

#include "flex_runtime.h"
#include "flex_printf.h"
#include "flex_redmule.h"
#include "flex_dma_pattern.h"
#include "flex_group_barrier.h"
#include "flex_libfp16.h"

void silu(const uint64_t in_addr, const uint64_t out_addr, const uint16_t dim, const uint16_t n_token, cluster_map_t cluster_map);
void sigmoid(const uint64_t in_addr, const uint64_t out_addr, const uint16_t dim, const uint16_t n_token, cluster_map_t cluster_map);
void add(const uint64_t in_addr_1, const uint64_t in_addr_2, const uint64_t out_addr, const uint16_t dim, const uint16_t n_token, cluster_map_t cluster_map);
void dot_product(const uint64_t in_addr_1, const uint64_t in_addr_2, const uint64_t out_addr, const uint16_t dim, const uint16_t n_token, cluster_map_t cluster_map);
void dot_product_const(const uint64_t in_addr, const fp16 in_const, const uint64_t out_addr, const uint16_t dim, const uint16_t n_token, cluster_map_t cluster_map);

void apply_element_wise_1_in(const uint64_t in_addr, const uint64_t out_addr, const uint16_t dim, const uint16_t n_token, element_wise_op_1_in_t op, cluster_map_t cluster_map);
void apply_element_wise_2_in_const(const uint64_t in_addr, const fp16 in_const, const uint64_t out_addr, const uint16_t dim, const uint16_t n_token, element_wise_op_2_in_const_t op, cluster_map_t cluster_map);
void apply_element_wise_2_in(const uint64_t in_addr1, const uint64_t in_addr2, const uint64_t out_addr, const uint16_t dim, const uint16_t n_token, element_wise_op_2_in_t op, cluster_map_t cluster_map);
void silu_op(const fp16* input, fp16* output);
void sigmoid_op(const fp16* input, fp16* output);
void mul_op(const fp16* input1, const fp16* input2, fp16* output);
void add_op(const fp16* input1, const fp16* input2, fp16* output);

/**
 * @brief apply element-wise operation on ONE INPUT matrix. Each core processes ELEMENT_WISE_TILE_WIDTH elements at a time.
 * 
 * @param in_addr can be in TCDM or HBM
 * @param out_addr can be in TCDM or HBM
 * @param dim 
 * @param n_token 
 * @param op element_wise_op_t function pointer
 */
void apply_element_wise_1_in(const uint64_t in_addr, const uint64_t out_addr, const uint16_t dim, const uint16_t n_token, element_wise_op_1_in_t op, cluster_map_t cluster_map) {
    if (0 == dim || 0 == n_token || NULL == op) {
        return;
    }
    // flex_global_barrier_xy();
    uint32_t cluster_id = flex_get_cluster_id();
    #ifdef SPATZ_ENABLE
    uint32_t core_id = flex_get_core_id();  // reverse the core id to make the dm core the first core in the cluster
    #else
    uint32_t core_id = ARCH_NUM_CORE_PER_CLUSTER - flex_get_core_id() - 1;  // reverse the core id to make the dm core the first core in the cluster
    #endif
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
        uint32_t local_in, local_out, local_const_0, local_const_1;
        
        uint32_t n_element_per_cluster;
        #ifdef SPATZ_ENABLE
            // equally divide the elements among activated clusters
            n_element_per_cluster = (dim - 1) / n_cluster_activated + 1;
            // make sure each cluster has enough elements to process
            if (n_element_per_cluster < SPATZ_VL_MIN) {
                n_element_per_cluster = SPATZ_VL_MIN;
            }
            // index of the first element to be processed by current cluster
            uint64_t i_element_cluster = local_cluster_id * n_element_per_cluster;
            
            if (i_element_cluster >= dim * n_token) {
                return;
            }
            // load data: one dma transfer per cluster
            n_element_per_cluster = min(n_element_per_cluster, dim * n_token - i_element_cluster);
            local_in = ARCH_CLUSTER_TCDM_SIZE - n_element_per_cluster * DATA_SIZE_BYTES;
            local_out = local_in - n_element_per_cluster * DATA_SIZE_BYTES;
            local_const_0 = local_out - SPATZ_VL * DATA_SIZE_BYTES;
            local_const_1 = local_const_0 - SPATZ_VL * DATA_SIZE_BYTES;
            // load input values
            if (flex_is_dm_core()) {
                flex_dma_async_1d(local(local_in), in_addr + i_element_cluster * DATA_SIZE_BYTES, n_element_per_cluster * DATA_SIZE_BYTES);
                flex_dma_async_wait_all();
            }
            
            // make sure data is ready
            flex_intra_cluster_sync();
            // prepare for constant values
            // NOTE: require spatz attached to the first core in the cluster
            if (0 == core_id) {
                uint16_t * local_in_ptr = (uint16_t *)local(local_in);
                uint16_t * local_const_1_ptr = (uint16_t *)local(local_const_1);
                uint16_t * local_out_ptr = (uint16_t *)local(local_out);
                for (int i = 0; i < SPATZ_VL; i++) {
                    local_const_1_ptr[i] = 0x3c00;
                }
                
                // compute element-wise operation with spatz core
                uint16_t vl;
                uint16_t i_element = 0;
                // fp16 const_1 = 0x3c00; // 1.0
                // asm volatile("fmv.h.x ft0, %0" : : "r"(const_1));
                // asm volatile("vfmv.v.f v2, ft0");
                asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(SPATZ_VL));
                asm volatile("vle16.v v2, (%0)" : : "r"(local_const_1_ptr));
                while (n_element_per_cluster > i_element) {
                    asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(min(SPATZ_VL, n_element_per_cluster - i_element)));
                    asm volatile("vle16.v v0, (%0)" : : "r"(local_in_ptr));

                    if (op == silu_op) {
                        // silu: x * sigmoid(x)
                        asm volatile("vfneg.v v3, v0");
                        asm_rvv_exp(3,4);
                        asm volatile("vfadd.vv v3, v2, v4");
                        asm volatile("vfdiv.vv v8, v2, v3");
                        asm volatile("vfmul.vv v8, v0, v8");
                    } else if (op == sigmoid_op) {
                        // sigmoid: 1 / (1 + exp(-x))
                        asm volatile("vfneg.v v3, v0");
                        asm_rvv_exp(3,4);
                        asm volatile("vfadd.vv v3, v2, v4");
                        asm volatile("vfdiv.vv v8, v2, v3");
                    } else {
                        // Unsupported operation
                    }
                    asm volatile("vse16.v v8, (%0)" : : "r"(local_out_ptr));

                    local_in_ptr += vl;
                    local_out_ptr += vl;
                    i_element += vl;
                }
            }
            
            // make sure data is ready
            flex_intra_cluster_sync();
            // store data: one dma transfer per cluster
            if (flex_is_dm_core()) {
                flex_dma_async_1d(out_addr + i_element_cluster * DATA_SIZE_BYTES, local(local_out), n_element_per_cluster * DATA_SIZE_BYTES);
                flex_dma_async_wait_all();
            }
            #else
            local_in = ARCH_CLUSTER_TCDM_SIZE - ARCH_NUM_CORE_PER_CLUSTER * ELEMENT_WISE_TILE_WIDTH * DATA_SIZE_BYTES;
            local_out = local_in - ARCH_NUM_CORE_PER_CLUSTER * ELEMENT_WISE_TILE_WIDTH * DATA_SIZE_BYTES;

            // index of the first element to be processed by current cluster
            uint64_t i_element_cluster = local_cluster_id * ARCH_NUM_CORE_PER_CLUSTER * ELEMENT_WISE_TILE_WIDTH;

            while (i_element_cluster < n_token * dim) {
                // load data: one dma transfer per cluster
                uint32_t max_cluster_capacity = ARCH_NUM_CORE_PER_CLUSTER * ELEMENT_WISE_TILE_WIDTH;
                uint32_t remaining_elements = dim * n_token - i_element_cluster;
                n_element_per_cluster = (remaining_elements < max_cluster_capacity) ? 
                                        remaining_elements : max_cluster_capacity;
                if (flex_is_dm_core()) {
                    flex_dma_async_1d(local(local_in), in_addr + i_element_cluster * DATA_SIZE_BYTES, n_element_per_cluster * DATA_SIZE_BYTES);
                    flex_dma_async_wait_all();
                }

                // compute element-wise operation
                int idx = core_id * ELEMENT_WISE_TILE_WIDTH;
                int n_element_per_core = min(ELEMENT_WISE_TILE_WIDTH, n_element_per_cluster - idx);
                if (n_element_per_core <= 0) {
                    n_element_per_core = 0;
                }
                flex_intra_cluster_sync();
                for (int i = 0; i < n_element_per_core; i++, idx++) {
                    op((const fp16*)local(local_in + idx * DATA_SIZE_BYTES), 
                            (fp16*)local(local_out + idx * DATA_SIZE_BYTES));
                }
                flex_intra_cluster_sync();

                // store data: one dma transfer per cluster
                if (flex_is_dm_core()) {
                    flex_dma_async_1d(out_addr + i_element_cluster * DATA_SIZE_BYTES, local(local_out), n_element_per_cluster * DATA_SIZE_BYTES);
                    flex_dma_async_wait_all();
                }
                i_element_cluster += ARCH_NUM_CORE_PER_CLUSTER * n_cluster_activated * ELEMENT_WISE_TILE_WIDTH;
            }
        #endif
    }
    
    // flex_global_barrier_xy();
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
void apply_element_wise_2_in(const uint64_t in_addr1, const uint64_t in_addr2, const uint64_t out_addr, const uint16_t dim, const uint16_t n_token, element_wise_op_2_in_t op, cluster_map_t cluster_map) {
    if (0 == dim || 0 == n_token || NULL == op) {
        return;
    }
    // flex_global_barrier_xy();
    uint32_t cluster_id = flex_get_cluster_id();
    #ifdef SPATZ_ENABLE
    uint32_t core_id = flex_get_core_id();  // reverse the core id to make the dm core the first core in the cluster
    #else
    uint32_t core_id = ARCH_NUM_CORE_PER_CLUSTER - flex_get_core_id() - 1;  // reverse the core id to make the dm core the first core in the cluster
    #endif
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
        uint32_t local_in1, local_in2, local_out;
        
        uint32_t n_element_per_cluster;
        #ifdef SPATZ_ENABLE
            n_element_per_cluster = (dim - 1) / n_cluster_activated + 1;
            // make sure each cluster has enough elements to process
            if (n_element_per_cluster < SPATZ_VL_MIN) {
                n_element_per_cluster = SPATZ_VL_MIN;
            }
            // index of the first element to be processed by current cluster
            uint64_t i_element_cluster = local_cluster_id * n_element_per_cluster;
            
            if (i_element_cluster >= dim * n_token) {
                return;
            }

            // load data: one dma transfer per cluster
            n_element_per_cluster = min(n_element_per_cluster, dim * n_token - i_element_cluster);
            local_in1 = ARCH_CLUSTER_TCDM_SIZE - n_element_per_cluster * DATA_SIZE_BYTES;
            local_in2 = local_in1 - n_element_per_cluster * DATA_SIZE_BYTES;
            local_out = local_in2 - n_element_per_cluster * DATA_SIZE_BYTES;
            // load input values
            if (flex_is_dm_core()) {
                flex_dma_async_1d(local(local_in1), in_addr1 + i_element_cluster * DATA_SIZE_BYTES, n_element_per_cluster * DATA_SIZE_BYTES);
                flex_dma_async_1d(local(local_in2), in_addr2 + i_element_cluster * DATA_SIZE_BYTES, n_element_per_cluster * DATA_SIZE_BYTES);
                flex_dma_async_wait_all();
            }

            // make sure data is ready
            flex_intra_cluster_sync();

            // NOTE: require spatz attached to the first core in the cluster
            if (0 == core_id) {
                uint16_t * local_in1_ptr = (uint16_t *)local(local_in1);
                uint16_t * local_in2_ptr = (uint16_t *)local(local_in2);
                uint16_t * local_out_ptr = (uint16_t *)local(local_out);
                
                // compute element-wise operation with spatz core
                uint16_t vl;
                uint16_t i_element = 0;
                while (n_element_per_cluster > i_element) {
                    asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(min(SPATZ_VL, n_element_per_cluster - i_element)));
                    asm volatile("vle16.v v0, (%0)" : : "r"(local_in1_ptr));
                    asm volatile("vle16.v v1, (%0)" : : "r"(local_in2_ptr));
                    
                    if (op == add_op) {
                        asm volatile("vfadd.vv v8, v0, v1");
                    } else if (op == mul_op) {
                        asm volatile("vfmul.vv v8, v0, v1");
                    } else {
                        // Unsupported operation
                    }
                    asm volatile("vse16.v v8, (%0)" : : "r"(local_out_ptr));

                    local_in1_ptr += vl;
                    local_in2_ptr += vl;
                    local_out_ptr += vl;
                    i_element += vl;
                }
            }
            
            // make sure data is ready
            flex_intra_cluster_sync();
            // store data: one dma transfer per cluster
            if (flex_is_dm_core()) {
                flex_dma_async_1d(out_addr + i_element_cluster * DATA_SIZE_BYTES, local(local_out), n_element_per_cluster * DATA_SIZE_BYTES);
                flex_dma_async_wait_all();
            }
            #else
            local_in1 = ARCH_CLUSTER_TCDM_SIZE - ARCH_NUM_CORE_PER_CLUSTER * ELEMENT_WISE_TILE_WIDTH * DATA_SIZE_BYTES;
            local_in2 = local_in1 - ARCH_NUM_CORE_PER_CLUSTER * ELEMENT_WISE_TILE_WIDTH * DATA_SIZE_BYTES;
            local_out = local_in2 - ARCH_NUM_CORE_PER_CLUSTER * ELEMENT_WISE_TILE_WIDTH * DATA_SIZE_BYTES;
            // index of the first element to be processed by current cluster
            uint32_t i_element_cluster = local_cluster_id * ARCH_NUM_CORE_PER_CLUSTER * ELEMENT_WISE_TILE_WIDTH;
    
            while (i_element_cluster < n_token * dim) {
                // load data: one dma transfer per cluster
                n_element_per_cluster = min(ARCH_NUM_CORE_PER_CLUSTER * ELEMENT_WISE_TILE_WIDTH, dim * n_token - i_element_cluster);
                if (flex_is_dm_core()) {
                    flex_dma_async_1d(local(local_in1), in_addr1 + i_element_cluster * DATA_SIZE_BYTES, n_element_per_cluster * DATA_SIZE_BYTES);
                    flex_dma_async_1d(local(local_in2), in_addr2 + i_element_cluster * DATA_SIZE_BYTES, n_element_per_cluster * DATA_SIZE_BYTES);
                    flex_dma_async_wait_all();
                }
                
                // compute element-wise operation
                int n_element_per_core = min(ELEMENT_WISE_TILE_WIDTH, n_element_per_cluster - core_id * ELEMENT_WISE_TILE_WIDTH);
                if (n_element_per_core <= 0) {
                    n_element_per_core = 0;
                }
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
        #endif
    }
    // flex_global_barrier_xy();
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
void apply_element_wise_2_in_const(const uint64_t in_addr, const fp16 in_const, const uint64_t out_addr, const uint16_t dim, const uint16_t n_token, element_wise_op_2_in_const_t op, cluster_map_t cluster_map) {
    if (0 == dim || 0 == n_token || NULL == op) {
        return;
    }
    // flex_global_barrier_xy();
    uint32_t cluster_id = flex_get_cluster_id();
    #ifdef SPATZ_ENABLE
    uint32_t core_id = flex_get_core_id();  // reverse the core id to make the dm core the first core in the cluster
    #else
    uint32_t core_id = ARCH_NUM_CORE_PER_CLUSTER - flex_get_core_id() - 1;  // reverse the core id to make the dm core the first core in the cluster
    #endif
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
        uint32_t local_in, local_out;
        
        uint32_t n_element_per_cluster;
        #ifdef SPATZ_ENABLE
            n_element_per_cluster = (dim - 1) / n_cluster_activated + 1;
            // make sure each cluster has enough elements to process
            if (n_element_per_cluster < SPATZ_VL_MIN) {
                n_element_per_cluster = SPATZ_VL_MIN;
            }
            // index of the first element to be processed by current cluster
            uint64_t i_element_cluster = local_cluster_id * n_element_per_cluster;
            
            if (i_element_cluster >= dim * n_token) {
                return;
            }

            // load data: one dma transfer per cluster
            n_element_per_cluster = min(n_element_per_cluster, dim * n_token - i_element_cluster);
            local_in = ARCH_CLUSTER_TCDM_SIZE - n_element_per_cluster * DATA_SIZE_BYTES;
            local_out = local_in - n_element_per_cluster * DATA_SIZE_BYTES;
            // load input value
            if (flex_is_dm_core()) {
                flex_dma_async_1d(local(local_in), in_addr + i_element_cluster * DATA_SIZE_BYTES, n_element_per_cluster * DATA_SIZE_BYTES);
                flex_dma_async_wait_all();
            }
            
            // make sure data is ready
            flex_intra_cluster_sync();
            
            // NOTE: require spatz attached to the first core in the cluster
            if (0 == core_id) {
                uint16_t * local_in_ptr = (uint16_t *)local(local_in);
                uint16_t * local_out_ptr = (uint16_t *)local(local_out);
                // compute element-wise operation with spatz core
                uint16_t vl;
                uint16_t i_element = 0;
                // Load FP16 value into a floating-point register
                asm volatile("fmv.h.x ft0, %0" : : "r"(in_const));
                while (n_element_per_cluster > i_element) {
                    // printf("");
                    asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(min(SPATZ_VL, n_element_per_cluster - i_element)));
                    asm volatile("vle16.v v0, (%0)" : : "r"(local_in_ptr));
                    
                    if (op == mul_op) {
                        // Perform vector-scalar multiplication
                        asm volatile("vfmul.vf v8, v0, ft0");
                    } else {
                        // Unsupported operation
                    }
                    asm volatile("vse16.v v8, (%0)" : : "r"(local_out_ptr));

                    local_in_ptr += vl;
                    local_out_ptr += vl;
                    i_element += vl;
                }
            }
            
            // make sure data is ready
            flex_intra_cluster_sync();
            // store data: one dma transfer per cluster
            if (flex_is_dm_core()) {
                flex_dma_async_1d(out_addr + i_element_cluster * DATA_SIZE_BYTES, local(local_out), n_element_per_cluster * DATA_SIZE_BYTES);
                flex_dma_async_wait_all();
            }
            #else
            if (0 == dim || 0 == n_token || NULL == op) {
                return;
            }
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
                int n_element_per_cluster, n_element_per_core;
        
                while (i_element_cluster < n_token * dim) {
                    // load data: one dma transfer per cluster
                    n_element_per_cluster = min(ARCH_NUM_CORE_PER_CLUSTER * ELEMENT_WISE_TILE_WIDTH, dim * n_token - i_element_cluster);
                    if (flex_is_dm_core()) {
                        flex_dma_async_1d(local(local_in), in_addr + i_element_cluster * DATA_SIZE_BYTES, n_element_per_cluster * DATA_SIZE_BYTES);
                        flex_dma_async_wait_all();
                    }
                    
                    // compute element-wise operation
                    n_element_per_core = min(ELEMENT_WISE_TILE_WIDTH, n_element_per_cluster - core_id * ELEMENT_WISE_TILE_WIDTH);
                    if (n_element_per_core <= 0) {
                        n_element_per_core = 0;
                    }
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
        #endif
    }
    // flex_global_barrier_xy();
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
void silu(const uint64_t in_addr, const uint64_t out_addr, const uint16_t dim, const uint16_t n_token, cluster_map_t cluster_map) {
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
void sigmoid(const uint64_t in_addr, const uint64_t out_addr, const uint16_t dim, const uint16_t n_token, cluster_map_t cluster_map) {
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
void dot_product(const uint64_t in_addr_1, const uint64_t in_addr_2, const uint64_t out_addr, const uint16_t dim, const uint16_t n_token, cluster_map_t cluster_map) {
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
void dot_product_const(const uint64_t in_addr, const fp16 in_const, const uint64_t out_addr, const uint16_t dim, const uint16_t n_token, cluster_map_t cluster_map) {
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
void add(const uint64_t in_addr_1, const uint64_t in_addr_2, const uint64_t out_addr, const uint16_t dim, const uint16_t n_token, cluster_map_t cluster_map) {
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


#endif