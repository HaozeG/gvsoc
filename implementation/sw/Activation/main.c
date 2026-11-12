// Copyright 2025 ETH Zurich and 
// University of Bologna

// Solderpad Hardware License
// Version 0.51, see LICENSE for details.

// SPDX-License-Identifier: SHL-0.51

// Author: Chi Zhang <chizhang@iis.ee.ethz.ch>, ETH Zurich
// Date: 1.Oct.2025

#include "flex_runtime.h"
#include "flex_dump.h"
#include "acti.h"
#include "preload.h"
#include "Activation.h"
#include "../common/fp16_check.h"

#ifndef ACTI_CHECK_TOL_ULP
#define ACTI_CHECK_TOL_ULP 1
#endif

int main()
{
    uint32_t eoc_val = 0;
    flex_barrier_xy_init();
    flex_global_barrier_xy();
    /**************************************/
    /*  Program Execution Region -- Start */
    /**************************************/

    ActivationInfo info = ActivationAnaylze(
        ACTI_M_SIZE/*num_total_token*/,
        ACTI_N_SIZE/*token_embedded_length*/,
        ACTI_GATE_ENABLE/*gate_enable*/,
        ACTI_BIAS_ENABLE/*bias_enable*/,
        I_ADDR/*input_address*/,
        O_EADDR/*output_address*/,
        G_ADDR/*gate_address*/,
        B_ADDR/*bias_address*/);
    flex_global_barrier_xy();

    //execute Activation
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0) flex_timer_start();
    flex_global_barrier_xy();
    ActivationRun(&info);
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0) flex_timer_end();
    flex_global_barrier_xy();

    //check output against the preloaded fp16 replay golden (outside the timed region)
#if defined(ACTI_FP16)
    eoc_val = fp16_check_rows(O_EADDR, O_GADDR, ACTI_M_SIZE, ACTI_N_SIZE, ACTI_CHECK_TOL_ULP, info.L1_Area);
    flex_global_barrier_xy();
#endif

    /**************************************/
    /*  Program Execution Region -- Stop  */
    /**************************************/
    flex_global_barrier_xy();
    flex_eoc(eoc_val);
    return 0;
}