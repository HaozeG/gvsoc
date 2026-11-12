// Elementwise row operators (add, bias, scale, axpy, rope) on Spatz, with an output check
// against the preloaded fp16 replay golden.

#include "flex_runtime.h"
#include "flex_dump.h"
#include "elt.h"
#include "preload.h"
#include "Eltwise.h"
#include "../common/fp16_check.h"

#ifndef ELT_CHECK_TOL_ULP
#define ELT_CHECK_TOL_ULP 0
#endif

int main()
{
    uint32_t eoc_val = 0;
    flex_barrier_xy_init();
    flex_global_barrier_xy();

    EltwiseInfo info = EltwiseAnalyze(ELT_M_SIZE, ELT_N_SIZE, I_ADDR, Z_ADDR, O_EADDR);
    flex_global_barrier_xy();

    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0) flex_timer_start();
    flex_global_barrier_xy();
    EltwiseRun(&info);
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0) flex_timer_end();
    flex_global_barrier_xy();

    // outside the timed region
    eoc_val = fp16_check_rows(O_EADDR, O_GADDR, ELT_M_SIZE, ELT_N_SIZE, ELT_CHECK_TOL_ULP, info.L1_Area);
    flex_global_barrier_xy();
    flex_eoc(eoc_val);
    return 0;
}
