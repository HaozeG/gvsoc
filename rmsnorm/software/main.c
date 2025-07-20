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
    uint32_t n_token = 4;
    // uint16_t dim = 1024;
    uint16_t dim = 13; // feature dimension
    fp16 dim_fp16 = float_to_fp16((float)dim);
    uint32_t input_offset = 0x00000000; // TCDM offset for input data

    uint32_t tile_size_dim = (dim - 1) / ARCH_NUM_CLUSTER + 1; // ideal tile size along dim
    uint32_t actual_tile_size_dim = max(min(tile_size_dim, dim - flex_get_cluster_id() * tile_size_dim), 0);
    // assign initial values to input data
    if (flex_is_dm_core()) {
        uint16_t * local_in_ptr = (uint16_t *)local(input_offset);
        #ifdef PRINT_DEBUG
        if (flex_get_cluster_id() == 0) {
            printf("[Input Data Initialization]\n");
        }
        #endif
        for (int i = 0; i < n_token; i++) {
            #ifdef PRINT_DEBUG
            if (flex_get_cluster_id() == 0) {
                printf("    ");
            }
            #endif
            for (int j = 0; j < actual_tile_size_dim; j++) {
                // TODO: initial values to be determined
                local_in_ptr[j + i * actual_tile_size_dim] = 0x3c00;
                #ifdef PRINT_DEBUG
                if (flex_get_cluster_id() == 0) {
                    printf("0x%04x ", local_in_ptr[j + i * actual_tile_size_dim]);
                }
                #endif
            }
            #ifdef PRINT_DEBUG
            if (flex_get_cluster_id() == 0) {
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
    compute_rmsnorm(input_offset, n_token, dim_fp16, dim);

    flex_global_barrier_xy();
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0) {
        flex_timer_end();
    }
    
#ifdef PRINT_DEBUG
    // get the output data
    if (flex_is_first_core() && (flex_get_cluster_id() == 0))
    {
        printf("[Check Results]\n");
        for (int i = 0; i < n_token; i++) {
            printf("    ");
            // for (int j = 0; j < actual_tile_size_dim; j++) {
            for (int j = 0; j < min(32, actual_tile_size_dim); j++) {
                printf("0x%04x ", ((uint16_t *)(input_offset))[j + i * actual_tile_size_dim]);
            }
            printf("\n");
        }
    }
#endif
    flex_global_barrier_xy();
    flex_eoc(eoc_val);
    return 0;
}
