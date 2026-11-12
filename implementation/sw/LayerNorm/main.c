// LayerNorm on the Spatz cores: timed run, then an output check against the
// preloaded golden (see include/LayerNorm.h and scripts/kernels/lnorm_preload.py).

#include "flex_runtime.h"
#include "flex_dump.h"
#include "lnorm.h"
#include "preload.h"
#include "LayerNorm.h"
#include "../common/fp16_check.h"

// The preload replays the device's fp32 summation order exactly while a row
// fits in one strip (n <= 4096 on the default arch); outputs are then
// bit-exact. 2 ULP covers longer rows, where the order differs.
#ifndef LNORM_CHECK_TOL_ULP
#define LNORM_CHECK_TOL_ULP 2
#endif

int main()
{
    uint32_t eoc_val = 0;
    flex_barrier_xy_init();
    flex_global_barrier_xy();
    /**************************************/
    /*  Program Execution Region -- Start */
    /**************************************/

    LayerNormInfo info = LayerNormAnalyze(
        LNORM_M_SIZE/*num_total_token*/,
        LNORM_N_SIZE/*token_embedded_length*/,
        I_ADDR/*input_address*/,
        O_EADDR/*output_address*/,
        G_ADDR/*gamma_address*/,
        B_ADDR/*beta_address*/);
    flex_global_barrier_xy();

    //execute LayerNorm
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0) flex_timer_start();
    flex_global_barrier_xy();
    LayerNormRun(&info);
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0) flex_timer_end();
    flex_global_barrier_xy();

    //check output (outside the timed region)
    eoc_val = fp16_check_rows(O_EADDR, O_GADDR, LNORM_M_SIZE, LNORM_N_SIZE, LNORM_CHECK_TOL_ULP, info.L1_Area);
    flex_global_barrier_xy();

    /**************************************/
    /*  Program Execution Region -- Stop  */
    /**************************************/
    flex_global_barrier_xy();
    flex_eoc(eoc_val);
    return 0;
}
