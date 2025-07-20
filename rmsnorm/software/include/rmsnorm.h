#ifndef RMSNORM_H
#define RMSNORM_H

#define HW_SYNC

#include "element_wise.h"

#include "flex_runtime.h"
#include "flex_redmule.h"
#include "flex_printf.h"
#include "flex_cluster_arch.h"
#include "flex_dma_pattern.h"


void compute_rmsnorm(
    uint64_t in_token_offset, 
    uint32_t n_token, 
    fp16 dim_fp16,
    uint32_t dim
);

/**
 * @brief compute RMSNorm: RMSNorm(x) = x / RMS(x), RMS(x) = sqrt(mean(x^2)).
 * 
 * @param in_offset inplace computation, output store at the same location as input
 * @param n_token number of tokens, i.e., number of rows in the input matrix
 * @param dim_fp16 feature dimension in fp16, i.e., number of columns in the input matrix
 * @param dim feature dimension, i.e., number of columns in the input matrix
 */
void compute_rmsnorm(
    uint64_t in_offset, 
    uint32_t n_token, 
    fp16 dim_fp16,
    uint32_t dim
) {
    // check parameters
    // TODO: upper limit of token considering TCMD size is not checked
    if (0 == dim || 0 == n_token) {
        return;
    }

    uint32_t cluster_id = flex_get_cluster_id();
    uint32_t core_id = flex_get_core_id();
    // determine work for each cluster
    // suppose dim is large enough, only consider parallel processing along dim, not n_token
    uint32_t tile_size_dim = (dim - 1) / ARCH_NUM_CLUSTER + 1; // ideal tile size along dim
    RMSNormInfo info;
    info.in_offset = in_offset;
    info.tile_size_dim = max(min(tile_size_dim, dim - cluster_id * tile_size_dim), 0);  // actual tile size along dim
    info.tile_size_n_token = n_token;
    
    uint32_t local_sum = ARCH_CLUSTER_TCDM_SIZE - info.tile_size_n_token * DATA_SIZE_BYTES;
    uint32_t local_square = local_sum - info.tile_size_dim * info.tile_size_n_token * DATA_SIZE_BYTES; 
    uint32_t local_temp = local_square - info.tile_size_n_token * DATA_SIZE_BYTES; // local_temp is used for software reduction

    // print local info
    #ifdef PRINT_DEBUG
    if (flex_is_first_core() && (cluster_id == 0))
    {
        printf("[RMSNorm Info]\n");
        printf("    dim_fp16: 0x%04x\n", dim_fp16);
        printf("    dim: %d\n", dim);
        printf("    n_token: %d\n", n_token);

        printf("    tile_size_n_token: %d\n", info.tile_size_n_token);
        printf("    tile_size_dim: %d\n", info.tile_size_dim);
        printf("    in_offset: 0x%08x\n", info.in_offset);
        printf("    local_sum: 0x%08x\n", local_sum);
        printf("    local_square: 0x%08x\n", local_square);
    }
    flex_global_barrier_xy();
    #endif

    // local square
    if (0 == core_id) {
        uint16_t * local_in_ptr = (uint16_t *)local(in_offset);
        uint16_t * local_out_ptr = (uint16_t *)local(local_square);
        
        // compute element-wise operation with spatz core
        uint16_t vl;
        uint16_t i_element = 0;
        while (info.tile_size_dim * info.tile_size_n_token > i_element) {
            asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(min(SPATZ_VL, info.tile_size_dim * info.tile_size_n_token - i_element)));
            asm volatile("vle16.v v0, (%0)" : : "r"(local_in_ptr));
            asm volatile("vle16.v v1, (%0)" : : "r"(local_in_ptr));
            asm volatile("vfmul.vv v8, v0, v1");
            asm volatile("vse16.v v8, (%0)" : : "r"(local_out_ptr));

            local_in_ptr += vl;
            local_out_ptr += vl;
            i_element += vl;
        }
    }

    // RMS
    // local sum
    // init with local sums with zeros
    if (flex_is_dm_core()) {
        flex_dma_async_1d(local(local_sum), zomem(0), info.tile_size_n_token * DATA_SIZE_BYTES);
        flex_dma_async_wait_all();
    }
    flex_intra_cluster_sync();
    if (0 == core_id) {
        uint16_t * local_in_ptr = (uint16_t *)local(local_square);
        uint16_t * local_out_ptr = (uint16_t *)local(local_sum);
        
        // perform reduced sum with spatz core
        uint16_t vl;
        uint16_t i_element = 0;
        uint16_t i_token = 0;
        while (info.tile_size_n_token > i_token) {
            while (info.tile_size_dim > i_element) {
                asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(min(SPATZ_VL, info.tile_size_dim - i_element)));
                asm volatile("vle16.v v0, (%0)" : : "r"(local_in_ptr));
                asm volatile("vle16.v v1, (%0)" : : "r"(local_out_ptr));
                asm volatile("vfredusum.vs v1, v0, v1");
                asm volatile("vse16.v v1, (%0)" : : "r"(local_out_ptr));
                local_in_ptr += vl;
                i_element += vl;
            }
            i_element = 0; // reset for next token
            local_out_ptr++;
            i_token++;
        }
    }
    flex_global_barrier_xy();
    
    // print local sum
    #ifdef PRINT_DEBUG
    if (flex_is_first_core() && (cluster_id == 0))
    {
        printf("[Local Sum]\n");
        uint16_t * local_out_ptr = (uint16_t *)local(local_sum);
        for (int i = 0; i < info.tile_size_n_token; i++) {
            printf("    ");
            printf("0x%04x \n", local_out_ptr[i]);
        }
    }
    #endif

    // global sum
    flex_global_barrier_xy();
    #ifdef HW_SYNC
        // use hardware reduction
        if (flex_is_dm_core() && cluster_id == 0)
        {
            flex_dma_async_reduction(local_sum, local_sum, info.tile_size_n_token * DATA_SIZE_BYTES, COLLECTIVE_REDADD_FP_16, 0b00, 0b00);
            flex_dma_async_wait_all();
        }
        flex_global_barrier_xy();
    #else
        // use software reduction
    
    #endif

    // print global sum
    #ifdef PRINT_DEBUG
    if (flex_is_first_core() && (cluster_id == 0))
    {
        printf("[Global Sum]\n");
        uint16_t * local_out_ptr = (uint16_t *)local(local_sum);
        for (int i = 0; i < info.tile_size_n_token; i++) {
            printf("    ");
            printf("0x%04x \n", local_out_ptr[i]);
        }
    }
    flex_global_barrier_xy();
    #endif
    
    // root & mean on sum
    if (0 == core_id && cluster_id == 0) {
        uint16_t * local_in_ptr = (uint16_t *)local(local_sum);
        uint16_t * local_out_ptr = (uint16_t *)local(local_sum);
        
        uint16_t vl;
        uint16_t i_token = 0;
        asm volatile("fmv.h.x ft0, %0" : : "r"(dim_fp16));
        while (info.tile_size_n_token > i_token) {
            asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(min(SPATZ_VL, info.tile_size_n_token - i_token)));
            asm volatile("vle16.v v0, (%0)" : : "r"(local_in_ptr));
            // mean: v1 = v0 / dim
            asm volatile("vfdiv.vf v1, v0, ft0");
            // root: v1 = sqrt(v0 / dim)
            // TODO: vfsqrt not implemented? Ignore for now 
            // asm volatile("vfsqrt.v v1, v1");
            asm volatile("vse16.v v1, (%0)" : : "r"(local_out_ptr));
            local_in_ptr += vl;
            local_out_ptr += vl;
            i_token += vl;
        }
    }
    flex_global_barrier_xy();

    #ifdef PRINT_DEBUG
    if (flex_is_first_core() && (cluster_id == 0))
    {
        printf("[RMS]\n");
        uint16_t * local_out_ptr = (uint16_t *)local(local_sum);
        for (int i = 0; i < info.tile_size_n_token; i++) {
            printf("    ");
            printf("0x%04x \n", local_out_ptr[i]);
        }
    }
    flex_global_barrier_xy();
    #endif
    
    // broadcast RMS
    #ifdef HW_SYNC
        if (flex_is_dm_core() && cluster_id == 0)
        {
            flex_dma_async_broadcast(local_sum, local_sum, info.tile_size_n_token * DATA_SIZE_BYTES, 0b00, 0b00);
            flex_dma_async_wait_all();
        }
    #else

    #endif

    // local division
    flex_intra_cluster_sync();
    if (0 == core_id) {
        uint16_t * local_in_ptr = (uint16_t *)local(in_offset);
        uint16_t * local_out_ptr = (uint16_t *)local(in_offset);
        uint16_t vl;
        uint16_t i_element = 0;
        uint16_t i_token = 0;
        
        while (info.tile_size_n_token > i_token) {
            asm volatile("fmv.h.x ft0, %0" : : "r"(((fp16 *)local(local_sum))[i_token]));
            while (info.tile_size_dim > i_element) {
                asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(min(SPATZ_VL, info.tile_size_dim - i_element)));
                asm volatile("vle16.v v0, (%0)" : : "r"(local_in_ptr));
                asm volatile("vfdiv.vf v1, v0, ft0");
                asm volatile("vse16.v v1, (%0)" : : "r"(local_out_ptr));
    
                local_in_ptr += vl;
                local_out_ptr += vl;
                i_element += vl;
            }
            i_element = 0; // reset for next token
            i_token++;
        }
    }
    flex_intra_cluster_sync();
}


#endif 