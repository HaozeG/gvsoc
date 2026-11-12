#ifndef _FP16_CHECK_H_
#define _FP16_CHECK_H_

/*
 * On-device comparison of an fp16 output matrix against a preloaded golden.
 *
 * Runs after the timed region. Cluster c checks rows c, c + ARCH_NUM_CLUSTER, ...
 * The DM core copies one output row and one golden row into L1 at l1_base;
 * core 0 compares them element by element.
 *
 * An element is bad when its distance in fp16 units in the last place (ULP) is
 * above tol_ulp and its absolute difference is above 2^-14. NaN in the output
 * where the golden is not NaN counts as bad.
 *
 * Each cluster prints one line, in cluster order. Must be called by every core
 * of every cluster (it uses intra-cluster syncs and global barriers). Returns
 * the cluster's bad count (only meaningful on core 0).
 */

#include "flex_runtime.h"
#include "flex_printf.h"
#include "flex_dma_pattern.h"
#include "flex_libfp16.h"

static inline int32_t fp16_ordered(uint16_t u)
{
    int32_t mag = (int32_t)(u & 0x7fff);
    return (u & 0x8000) ? -mag : mag;
}

static inline uint32_t fp16_is_nan(uint16_t u)
{
    return ((u & 0x7c00) == 0x7c00) && (u & 0x03ff);
}

uint32_t fp16_check_rows(
    uint64_t out_addr,
    uint64_t gold_addr,
    uint32_t num_rows,
    uint32_t row_elems,
    uint32_t tol_ulp,
    uint32_t l1_base)
{
    uint32_t row_bytes = 2 * row_elems;
    uint32_t l1_out    = l1_base;
    uint32_t l1_gold   = l1_base + row_bytes;
    uint32_t bad = 0, nan = 0, max_ulp = 0, rows = 0;

    for (uint32_t r = flex_get_cluster_id(); r < num_rows; r += ARCH_NUM_CLUSTER)
    {
        if (flex_is_dm_core())
        {
            flex_dma_async_1d(l1_out,  out_addr  + (uint64_t)r * row_bytes, row_bytes);
            flex_dma_async_1d(l1_gold, gold_addr + (uint64_t)r * row_bytes, row_bytes);
            flex_dma_async_wait_all();
        }
        flex_intra_cluster_sync();
        if (flex_is_first_core())
        {
            uint16_t * o = (uint16_t *)l1_out;
            uint16_t * g = (uint16_t *)l1_gold;
            for (uint32_t i = 0; i < row_elems; ++i)
            {
                if (fp16_is_nan(g[i])) continue;
                if (fp16_is_nan(o[i])) { nan++; bad++; continue; }
                int32_t d = fp16_ordered(o[i]) - fp16_ordered(g[i]);
                uint32_t ulp = (uint32_t)(d < 0 ? -d : d);
                if (ulp > max_ulp) max_ulp = ulp;
                if (ulp > tol_ulp)
                {
                    float diff = fp16_to_float(o[i]) - fp16_to_float(g[i]);
                    if (diff < 0) diff = -diff;
                    if (diff > 6.103515625e-05f)
                    {
#ifdef FP16_CHECK_VERBOSE
                        if (bad < 8) printf("[check] c%d row %d col %d: out 0x%x gold 0x%x\n",
                                            flex_get_cluster_id(), r, i, o[i], g[i]);
#endif
                        bad++;
                    }
                }
            }
            rows++;
        }
        flex_intra_cluster_sync();
    }
    // print one cluster at a time; concurrent printf output interleaves per character
    for (uint32_t c = 0; c < ARCH_NUM_CLUSTER; ++c)
    {
        if (flex_get_cluster_id() == c && flex_is_first_core())
        {
            printf("[check] cluster %d: rows %d, bad %d (nan %d), max_ulp %d, tol_ulp %d\n",
                   c, rows, bad, nan, max_ulp, tol_ulp);
        }
        flex_global_barrier_xy();
    }
    return bad;
}

#endif
