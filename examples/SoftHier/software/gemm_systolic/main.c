#include "flex_runtime.h"
#include "gemm_systolic_wise.h"
#include <math.h>

int main()
{
    uint32_t eoc_val = 0;
    flex_barrier_xy_init();
    flex_global_barrier_xy();
    /**************************************/
    /*  Program Execution Region -- Start */
    /**************************************/

    // gemm_systolic_wise(4096, 4096, 4096, 2, 256, 256, 256);
    gemm_systolic_wise(64, 2048, 7168, 2, 16, 128, 448);

    /**************************************/
    /*  Program Execution Region -- Stop  */
    /**************************************/
    flex_global_barrier_xy();
    flex_eoc(eoc_val);
    return 0;
}