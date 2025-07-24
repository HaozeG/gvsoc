#ifndef RMSNORM_H
#define RMSNORM_H

// #define HW_SYNC

#include "element_wise.h"

#include "flex_runtime.h"
#include "flex_redmule.h"
#include "flex_printf.h"
#include "flex_cluster_arch.h"
#include "flex_dma_pattern.h"

// for software reduction
void reduction_init(RMSNormInfo *info);
void reduction_get_next_hop(RMSNormInfo *info);
void reduction_update_state(RMSNormInfo *info);

// for software broadcast
void broadcast_init(RMSNormInfo *info);
void broadcast_get_next_hop(RMSNormInfo *info);
void broadcast_update_state(RMSNormInfo *info);

void compute_rmsnorm(
    uint64_t in_token_offset, 
    uint32_t n_token, 
    uint32_t dim
);

/**
 * @brief compute RMSNorm: RMSNorm(x) = x / RMS(x), RMS(x) = sqrt(mean(x^2)).
 * 
 * @param in_offset inplace computation, output store at the same location as input
 * @param n_token number of tokens, i.e., number of rows in the input matrix
 * @param dim feature dimension, i.e., number of columns in the input matrix
 */
void compute_rmsnorm(
    uint64_t in_offset, 
    uint32_t n_token, 
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
    uint32_t tile_size_dim = dim / ARCH_NUM_CLUSTER; // ideal tile size along dim
    FlexPosition pos = get_pos(cluster_id);
    RMSNormInfo info;
    info.cluster_id_x = pos.x;
    info.cluster_id_y = pos.y;
    info.in_offset = in_offset;
    info.tile_size_dim = (cluster_id < (dim % ARCH_NUM_CLUSTER)) ? (tile_size_dim + 1) : tile_size_dim; // actual tile size along dim
    info.tile_size_n_token = n_token;
    
    uint32_t local_sum = ARCH_CLUSTER_TCDM_SIZE - info.tile_size_n_token * DATA_SIZE_BYTES;
    uint32_t local_square = local_sum - info.tile_size_dim * info.tile_size_n_token * DATA_SIZE_BYTES; 
    uint32_t local_temp = local_square - info.tile_size_n_token * DATA_SIZE_BYTES; // local_temp is used for software reduction

    // print local info
    #ifdef PRINT_DEBUG
    if (flex_is_first_core() && (cluster_id == PRINT_DEBUG_CLUSTER_ID))
    {
        printf("[RMSNorm Info]\n");
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
        fp16 * local_in_ptr = (fp16 *)local(in_offset);
        fp16 * local_out_ptr = (fp16 *)local(local_square);

        // compute element-wise operation with spatz core
        uint16_t vl;
        uint16_t i_element = 0;
        while (info.tile_size_dim * info.tile_size_n_token > i_element) {
            asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(min(SPATZ_VL, info.tile_size_dim * info.tile_size_n_token - i_element)));
            asm volatile("vle16.v v0, (%0)" : : "r"(local_in_ptr));
            asm volatile("vfmul.vv v8, v0, v0");
            asm volatile("vse16.v v8, (%0)" : : "r"(local_out_ptr));

            local_in_ptr += vl;
            local_out_ptr += vl;
            i_element += vl;
        }
    }

    // print local square
    #ifdef PRINT_DEBUG
    flex_global_barrier_xy();
    if (flex_is_first_core() && (cluster_id == PRINT_DEBUG_CLUSTER_ID))
    {
        printf("[Local Square]\n");
        uint16_t * local_out_ptr = (uint16_t *)local(local_square);
        for (int i = 0; i < info.tile_size_n_token; i++) {
            printf("    ");
            for (int j = 0; j < info.tile_size_dim; j++) {
                printf("%f ", fp16_to_float(local_out_ptr[j + i * info.tile_size_dim]));
            }
            printf("\n");
        }
    }
    #endif

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
    
    // print local sum
    #ifdef PRINT_DEBUG
    flex_global_barrier_xy();
    if (flex_is_first_core() && (cluster_id == PRINT_DEBUG_CLUSTER_ID))
    {
        printf("[Local Sum]\n");
        uint16_t * local_out_ptr = (uint16_t *)local(local_sum);
        for (int i = 0; i < info.tile_size_n_token; i++) {
            printf("    ");
            printf("%f \n", fp16_to_float(local_out_ptr[i]));
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
        // reduced sum is stored at local_sum of cluster 0
        reduction_init(&info);
        flex_intra_cluster_sync();
        while (!info.is_finished) {
            reduction_get_next_hop(&info);
            // transfer partial sum to next hop
            if (info.is_operate) {
                if (flex_is_dm_core()) {
                    FlexPosition src_pos = pos;
                    if (info.next_hop_direction == 0) {
                        // left 
                        src_pos.x = min(src_pos.x + info.next_hop_offset, ARCH_NUM_CLUSTER_X - 1);
                        flex_dma_async_1d(local_temp, remote_pos(src_pos, local_sum), info.tile_size_n_token * DATA_SIZE_BYTES);
                    } else if (info.next_hop_direction == 3) {
                        // down 
                        src_pos.y = min(src_pos.y + info.next_hop_offset, ARCH_NUM_CLUSTER_Y - 1);
                        flex_dma_async_1d(local_temp, remote_pos(src_pos, local_sum), info.tile_size_n_token * DATA_SIZE_BYTES);
                    } else {
                        // not used 
                    }
                    #ifdef PRINT_DEBUG
                    if (cluster_id == PRINT_DEBUG_CLUSTER_ID) {
                        printf("[Reduction] Transfer from Cluster at Position (%d, %d) to Cluster at Position (%d, %d)\n", 
                            info.next_hop_direction == 0 ? (pos.x + info.next_hop_offset) : pos.x,
                            info.next_hop_direction == 3 ? (pos.y + info.next_hop_offset) : pos.y, pos.x, pos.y);
                    }
                    #endif
                    flex_dma_async_wait_all();
                }
                // make sure data is ready
                flex_intra_cluster_sync();
                // do local reduction
                if (0 == core_id && info.is_operate) {
                    uint16_t * local_in_ptr = (uint16_t *)local(local_temp);
                    uint16_t * local_out_ptr = (uint16_t *)local(local_sum);
                    
                    // perform reduced sum with spatz core
                    uint16_t vl;
                    uint16_t i_token = 0;
                    while (info.tile_size_n_token > i_token) {      
                        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(min(SPATZ_VL, info.tile_size_n_token - i_token)));
                        asm volatile("vle16.v v0, (%0)" : : "r"(local_in_ptr));
                        asm volatile("vle16.v v1, (%0)" : : "r"(local_out_ptr));
                        asm volatile("vfadd.vv v1, v0, v1");
                        asm volatile("vse16.v v1, (%0)" : : "r"(local_out_ptr));
                        local_in_ptr += vl;
                        local_out_ptr += vl;
                        i_token += vl;
                    }
                }
            }
            reduction_update_state(&info);
            #ifdef PRINT_DEBUG
            if (cluster_id == PRINT_DEBUG_CLUSTER_ID && core_id == 0 && info.is_finished == 0) {
                printf("[Reduction] Updated: Iteration %d, Next Hop Direction %d\n", 
                    info.iteration, info.next_hop_direction);
            }
            #endif
            flex_global_barrier_xy();
        }
    #endif

    // print global sum
    #ifdef PRINT_DEBUG
    if (flex_is_first_core() && (0 == cluster_id))
    {
        printf("[Global Sum]\n");
        uint16_t * local_out_ptr = (uint16_t *)local(local_sum);
        for (int i = 0; i < info.tile_size_n_token; i++) {
            printf("    ");
            printf("%f \n", fp16_to_float(local_out_ptr[i]));
        }
    }
    flex_global_barrier_xy();
    #endif
    
    // root & mean on sum
    // use cluster 0 
    if (0 == core_id && 0 == cluster_id) {
        uint16_t * local_in_ptr = (uint16_t *)local(local_sum);
        uint16_t * local_out_ptr = (uint16_t *)local(local_sum);
        
        uint16_t vl;
        uint16_t i_token = 0;
        asm volatile("fmv.h.x ft0, %0" : : "r"(float_to_fp16((float)dim)));
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
    
    // broadcast RMS
    #ifdef HW_SYNC
        if (flex_is_dm_core() && cluster_id == 0)
        {
            flex_dma_async_broadcast(local_sum, local_sum, info.tile_size_n_token * DATA_SIZE_BYTES, 0b00, 0b00);
            flex_dma_async_wait_all();
        }
    #else
        // use software broadcast
        broadcast_init(&info);
        flex_intra_cluster_sync();
        while (!info.is_finished) {
            broadcast_get_next_hop(&info);
            // transfer RMS to next hop
            if (info.is_operate) {
                if (flex_is_dm_core()) {
                    FlexPosition dest_pos = pos;
                    if (info.next_hop_direction == 1) {
                        // right
                        dest_pos.x = min(dest_pos.x + info.next_hop_offset, ARCH_NUM_CLUSTER_X - 1);
                        flex_dma_async_1d(remote_pos(dest_pos, local_sum), local_sum, info.tile_size_n_token * DATA_SIZE_BYTES);
                    } else if (info.next_hop_direction == 2) {
                        // up
                        dest_pos.y = min(dest_pos.y + info.next_hop_offset, ARCH_NUM_CLUSTER_Y - 1);
                        flex_dma_async_1d(remote_pos(dest_pos, local_sum), local_sum, info.tile_size_n_token * DATA_SIZE_BYTES);
                    } else {
                        // not used
                    }
                    #ifdef PRINT_DEBUG
                    if (cluster_id == PRINT_DEBUG_CLUSTER_ID) {
                        printf("[Broadcast] Transfer from Cluster at Position (%d, %d) to Cluster at Position (%d, %d)\n", pos.x, pos.y,
                            info.next_hop_direction == 1 ? (pos.x + info.next_hop_offset) : pos.x,
                            info.next_hop_direction == 2 ? (pos.y + info.next_hop_offset) : pos.y);
                    }
                    #endif
                    flex_dma_async_wait_all();
                }
            }
            broadcast_update_state(&info);
            #ifdef PRINT_DEBUG
            flex_intra_cluster_sync();
            if (cluster_id == PRINT_DEBUG_CLUSTER_ID && core_id == 0 && info.is_finished == 0) {
                printf("[Broadcast] Updated: Iteration %d, Next Hop Direction %d\n", 
                    info.iteration, info.next_hop_direction);
            }
            #endif
            flex_global_barrier_xy();
        }
    #endif 

    // print RMS
    #ifdef PRINT_DEBUG
    if (flex_is_first_core() && (cluster_id == PRINT_DEBUG_CLUSTER_ID))
    {
        printf("[RMS]\n");
        uint16_t * local_out_ptr = (uint16_t *)local(local_sum);
        for (int i = 0; i < info.tile_size_n_token; i++) {
            printf("    ");
            printf("%f \n", fp16_to_float(local_out_ptr[i]));
        }
    }
    flex_global_barrier_xy();
    #endif

    #ifdef PRINT_DEBUG
    if (flex_is_first_core() && (cluster_id == PRINT_DEBUG_CLUSTER_ID))
    {
        printf("[Input]\n");
        uint16_t * local_out_ptr = (uint16_t *)local(in_offset);
        for (int i = 0; i < info.tile_size_n_token; i++) {
            printf("    ");
            printf("%f \n", fp16_to_float(local_out_ptr[i]));
        }
    }
    flex_global_barrier_xy();
    #endif

    // local division
    flex_intra_cluster_sync();
    if (0 == core_id) {
        fp16 * local_in_ptr = (fp16 *)local(in_offset);
        fp16 * local_out_ptr = (fp16 *)local(in_offset);
        uint16_t vl;
        uint16_t i_element = 0;
        uint16_t i_token = 0;

        while (info.tile_size_n_token > i_token) {
            asm volatile("fmv.h.x f0, %0" : : "r"(((fp16 *)local(local_sum))[i_token]));
            #ifdef PRINT_DEBUG
            if (flex_get_cluster_id() == PRINT_DEBUG_CLUSTER_ID) {
                printf("[RMSNorm] RMS for Token %d: 0x%04x\n", i_token, ((fp16 *)local(local_sum))[i_token]);
            }
            #endif
            while (info.tile_size_dim > i_element) {
                asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(min(SPATZ_VL, info.tile_size_dim - i_element)));
                asm volatile("vle16.v v0, (%0)" : : "r"(local_in_ptr));
                asm volatile("vfdiv.vf v0, v0, f0");
                asm volatile("vse16.v v0, (%0)" : : "r"(local_out_ptr));
    
                local_in_ptr += vl;
                local_out_ptr += vl;
                i_element += vl;

                #ifdef PRINT_DEBUG
                if (flex_get_cluster_id() == PRINT_DEBUG_CLUSTER_ID) {
                    printf("vl = %d, i_element = %d, i_token = %d\n", vl, i_element, i_token);
                }
                #endif
            }
            #ifdef PRINT_DEBUG
                if ((cluster_id == PRINT_DEBUG_CLUSTER_ID))
                {
                    printf("[Output]\n");
                    for (int i = 0; i < info.tile_size_n_token; i++) {
                        printf("    ");
                        printf("%f \n", fp16_to_float(((fp16 *)local(in_offset))[i]));
                    }
                }
            #endif
            i_element = 0; // reset for next token
            i_token++;
        }
    }
    flex_intra_cluster_sync();
}

/*
    The following functions are used for software reduction.
*/
void reduction_init(RMSNormInfo *info) {
    info->is_finished = 0;
    info->iteration = 0;
    info->next_hop_direction = 0; // start with left
    info->next_hop_offset = 0;
    info->is_operate = 0;

    #ifdef PRINT_DEBUG
    if (flex_is_first_core() && (flex_get_cluster_id() == PRINT_DEBUG_CLUSTER_ID))
    {
        printf("[Reduction] Initialized: Iteration %d, Next Hop Direction %d\n", 
            info->iteration, info->next_hop_direction);
    }
    #endif
}

void reduction_update_state(RMSNormInfo *info) {
    info->iteration++;
    if (info->next_hop_direction == 0) {
        if (ARCH_NUM_CLUSTER_X <= (0b01 << info->iteration)) {
            info->num_cluster_log2_x = info->iteration - 1;
            info->next_hop_direction = 3; // down
            info->iteration = 0; // reset iteration
        }
    } else if (info->next_hop_direction == 3) {
        if (ARCH_NUM_CLUSTER_Y <= (0b01 << info->iteration)) {
            info->num_cluster_log2_y = info->iteration - 1;
            info->is_finished = 1; // finished
            info->iteration = 0; // reset iteration
        }
    } else {
        // not used in reduction
    }
}

void reduction_get_next_hop(RMSNormInfo *info) {
    if (info->is_finished) {
        return;
    }
    // check if requires transfer
    uint32_t pos;
    if (info->next_hop_direction == 0) {
        pos = info->cluster_id_x;
    } else if (info->next_hop_direction == 3) {
        pos = info->cluster_id_y;
    } else {
        // not used in reduction
    }
    // only the even position requires transfer
    if (((pos & (0b01 << info->iteration)) == 0) && ((pos & 0b01) == 0)) {
        // transfer required
        info->next_hop_offset = (0b01 << info->iteration);
        info->is_operate = 1;
        // check if next hop is out of bound
        if (info->next_hop_direction == 0) {
            if (info->next_hop_offset + pos >= ARCH_NUM_CLUSTER_X) {
                info->is_operate = 0; // no compute required
            }
        } else if (info->next_hop_direction == 3) {
            if (info->next_hop_offset + pos >= ARCH_NUM_CLUSTER_Y) {
                info->is_operate = 0; // no compute required
            }
            // check if not the first colomn
            if (info->cluster_id_x != 0) {
                info->is_operate = 0; // no compute required
            }
        } else {
            // not used in reduction
        }
    } else {
        // no transfer required
        info->next_hop_offset = 0;
        info->is_operate = 0;
    }
}

/*
    The following functions are used for software broadcast.
*/
void broadcast_init(RMSNormInfo *info) {
    info->is_finished = 0;
    info->iteration = 0;
    info->next_hop_direction = 1; // start with right
    info->next_hop_offset = 0;
    info->is_operate = 0;

    #ifdef PRINT_DEBUG
    if (flex_is_first_core() && (flex_get_cluster_id() == PRINT_DEBUG_CLUSTER_ID))
    {        printf("[Broadcast] Initialized: Iteration %d, Next Hop Direction %d\n", 
            info->iteration, info->next_hop_direction);
    }
    #endif
}

void broadcast_update_state(RMSNormInfo *info) {
    info->iteration++;
    if (info->next_hop_direction == 1) {
        if (ARCH_NUM_CLUSTER_X <= (0b01 << info->iteration)) {
            info->next_hop_direction = 2; // up
            info->iteration = 0; // reset iteration
        }
    } else if (info->next_hop_direction == 2) {
        if (ARCH_NUM_CLUSTER_Y <= (0b01 << info->iteration)) {
            info->is_finished = 1; // finished
            info->iteration = 0; // reset iteration
        }
    } else {
        // not used in broadcast
    }
}

void broadcast_get_next_hop(RMSNormInfo *info) {
    if (info->is_finished) {
        return;
    }
    // check if requires transfer
    uint32_t pos, num_cluster_log2;
    if (info->next_hop_direction == 1) {
        pos = info->cluster_id_x;
        num_cluster_log2 = info->num_cluster_log2_x;
    } else if (info->next_hop_direction == 2) {
        pos = info->cluster_id_y;
        num_cluster_log2 = info->num_cluster_log2_y;
    } else {
        // not used in broadcast
    }
    // only the even position requires transfer
    if (((pos & (0b01 << (num_cluster_log2 - info->iteration))) == 0) && ((pos & 0b01) == 0)) {
        // transfer required
        info->next_hop_offset = (0b01 << (num_cluster_log2 - info->iteration));
        info->is_operate = 1;
        // check if next hop is out of bound
        if (info->next_hop_direction == 1) {
            if (info->next_hop_offset + pos >= ARCH_NUM_CLUSTER_X) {
                info->is_operate = 0; // no transfer required
            }
            // check if not the first row 
            if (info->cluster_id_y != 0) {
                info->is_operate = 0; // no transfer required
            }
        } else if (info->next_hop_direction == 2) {
            if (info->next_hop_offset + pos >= ARCH_NUM_CLUSTER_Y) {
                info->is_operate = 0; // no transfer required
            }
        } else {
            // not used in broadcast
        }
    } else {
        // no transfer required
        info->next_hop_offset = 0;
        info->is_operate = 0;
    }
}

#endif 
