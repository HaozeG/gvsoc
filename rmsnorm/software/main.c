// Author: Haoze Gao <gaohao@student.ethz.ch>, Ho Tin Hung <hohung@student.ethz.ch>

#include <math.h>
#include <inttypes.h>
#include "flex_runtime.h"
#include "flex_redmule.h"
#include "flex_printf.h"
#include "flex_cluster_arch.h"
#include "flex_dma_pattern.h"

#include "rmsnorm.h"
// #define PRINT_DEBUG 0

int main();
int main(){
    flex_barrier_xy_init();
    flex_global_barrier_xy();

    // TODO: read parameters from command line or configuration file
    uint32_t n_token = 16;
    // uint16_t dim = 1024;
    uint64_t dim = 7168;
    uint32_t input_offset = 0x00000000; // TCDM offset for input data

    uint32_t eoc_val = 0;
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
    
#ifdef PRINT_DEBUG
    // get the output
    uint64_t out_buffer = 0;
    if (flex_is_dm_core() && (0 == flex_get_cluster_id())) {
        flex_dma_async_1d(local(out_buffer), hbm_west((uint64_t)0, actual_out_offset), n_token / ARCH_NUM_CLUSTER_Y * dim * DATA_SIZE_BYTES);
        flex_dma_async_wait_all();
    }
    flex_intra_cluster_sync();
    if (flex_is_first_core() && (flex_get_cluster_id()==0))
    {
        printf("[Check Results]\n");
        // printf("actual_out:\n");
        // // for (int i = 0; i < (n_token >> 2)*n_routed_experts; i++) {
        // for (int i = 0; i < 3; i++) {
        //     printf("    ");
        //     // for (int j = 0; j < dim; j++) {
        //     for (int j = 0; j < 32; j++) {
        //         // printf("0x%04x ", ((uint16_t *)(hbm_addr(actual_out_offset)))[j + i * dim]);
        //         printf("0x%04x ", (uint16_t)*(uint64_t *)(hbm_addr(actual_out_offset) + (j + i * dim) * DATA_SIZE_BYTES));
        //     }
        //     printf("\n");
        // }
        printf("actual_out:\n");
        for (int i = 0; i < 16; i++) {
            printf("    ");
            // for (int j = 0; j < dim; j++) {
            for (int j = 0; j < 32; j++) {
                printf("0x%04x ", ((uint16_t *)(out_buffer))[j + i * dim]);
            }
            printf("\n");
        }
    }
#endif
    flex_global_barrier_xy();
    flex_eoc(eoc_val);
    return 0;
}
