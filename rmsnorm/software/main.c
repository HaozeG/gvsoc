// Author: Haoze Gao <gaohao@student.ethz.ch>, Ho Tin Hung <hohung@student.ethz.ch>

#include <math.h>
#include <inttypes.h>
#include "flex_runtime.h"
#include "flex_redmule.h"
#include "flex_printf.h"
#include "flex_cluster_arch.h"
#include "flex_dma_pattern.h"

#include "rmsnorm.h"

int main();
int main(){
    flex_barrier_xy_init();
    flex_global_barrier_xy();

    uint32_t eoc_val = 0;
    // TODO: read parameters from command line or configuration file
    uint32_t n_token = 5;
    // uint16_t dim = 1024;
    uint16_t dim = 140; // feature dimension
    uint32_t input_offset = 0x00000000; // TCDM offset for input data

    // TODO: consider better division strategy
    uint32_t tile_size_dim = dim / ARCH_NUM_CLUSTER; // ideal tile size along dim
    uint32_t actual_tile_size_dim = (flex_get_cluster_id() < (dim % ARCH_NUM_CLUSTER)) ? (tile_size_dim + 1) : tile_size_dim; // actual tile size along dim
    // assign initial values to input data
    if (flex_is_dm_core()) {
        fp16 * local_in_ptr = (fp16 *)local(input_offset);
        #ifdef PRINT_DEBUG
        if (flex_get_cluster_id() == PRINT_DEBUG_CLUSTER_ID) {
            printf("[Input Data Initialization]\n");
        }
        #endif
        for (int i = 0; i < n_token; i++) {
            #ifdef PRINT_DEBUG
            if (flex_get_cluster_id() == PRINT_DEBUG_CLUSTER_ID) {
                printf("    ");
            }
            #endif
            for (int j = 0; j < actual_tile_size_dim; j++) {
                // TODO: initial values to be determined
                local_in_ptr[j + i * actual_tile_size_dim] = 0x3c00;
                #ifdef PRINT_DEBUG
                if (flex_get_cluster_id() == PRINT_DEBUG_CLUSTER_ID) {
                    printf("%f ", fp16_to_float(local_in_ptr[j + i * actual_tile_size_dim]));
                }
                #endif
            }
            #ifdef PRINT_DEBUG
            if (flex_get_cluster_id() == PRINT_DEBUG_CLUSTER_ID) {
                printf("\n");
            }
            #endif
        }
    }
    flex_global_barrier_xy();

    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0) {
        printf("[Start RMSNorm Computation]\n");
        flex_timer_start();
    }
    flex_global_barrier_xy();

    // call compute function
    compute_rmsnorm(input_offset, n_token, dim);

    flex_global_barrier_xy();
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0) {
        flex_timer_end();
    }

    // print results
    #ifdef PRINT_DEBUG
    if (flex_is_dm_core()) {
        fp16 * local_in_ptr = (fp16 *)local(input_offset);
        if (flex_get_cluster_id() == PRINT_DEBUG_CLUSTER_ID) {
            printf("[Check Output]\n");
        }

        for (int i = 0; i < n_token; i++) {
            if (flex_get_cluster_id() == PRINT_DEBUG_CLUSTER_ID) {
                printf("    ");
            }
            for (int j = 0; j < actual_tile_size_dim; j++) {
                if (flex_get_cluster_id() == PRINT_DEBUG_CLUSTER_ID) {
                    printf("%f ", fp16_to_float(local_in_ptr[j + i * actual_tile_size_dim]));
                }
            }
            if (flex_get_cluster_id() == PRINT_DEBUG_CLUSTER_ID) {
                printf("\n");
            }
        }
    }
    #endif
    
    flex_global_barrier_xy();
    flex_eoc(eoc_val);
    return 0;
}
