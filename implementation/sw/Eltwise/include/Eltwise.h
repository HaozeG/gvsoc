// Elementwise row operators on Spatz: add, bias, scale, axpy, rope.
//
// Same scheme as Activation: rows are dealt round-robin to the clusters; per row the DM core
// loads the row and its second operand into L1, the Spatz cores each process a slice, and the
// DM core stores the row back.
//
//   add:   y = x + z            (residual add, position-embedding add)
//   bias:  y = x + b            (b: one row of N, loaded once per cluster and broadcast)
//   scale: y = alpha x
//   axpy:  y = x + alpha z      (vfmacc, one rounding; the flow-matching Euler update)
//   rope:  per head of ELT_HEAD_DIM columns, with (c | s) the row's table entry (cos[dh/2] then
//          sin[dh/2], precomputed for the row's position): y1 = x1 c - x2 s, y2 = x2 c + x1 s
//          (HF rotate-half convention, as softhier-mlir sh_rope); Spatz cores split the heads.
//   pixel_shuffle: SmolVLM connector, one output row = the ELT_PIX_SCALE x ELT_PIX_SCALE patch block
//          of an ELT_GRID x ELT_GRID image, row-major: one 2D DMA (ELT_PIX_SCALE chunks of
//          ELT_PIX_SCALE consecutive patch rows) into L1, then the row is stored. No Spatz work.
//
// Scalars are static NaN-boxed constants loaded with "m" operands (see Activation.h); no
// vfmv.v.f (wrong in the GVSoC model) and none of the untimed vfwadd.wv / vfwsub.vf / vmv.*.

#ifndef _ELTWISE_H_
#define _ELTWISE_H_

#include "flex_runtime.h"
#include "flex_printf.h"
#include "flex_dma_pattern.h"
#include "elt.h"

#if !defined(ELT_FP16)
#error "Eltwise supports fp16 only"
#endif

typedef struct EltwiseInfo
{
    uint64_t        Input;
    uint64_t        Second;         /* z (add, axpy), b (bias), table (rope) */
    uint64_t        Output;
    uint32_t        num_rows;
    uint32_t        row_elems;
    uint32_t        row_bytes;
    uint32_t        second_bytes;   /* bytes of the second operand per row */
    uint32_t        start_token;
    uint32_t        L1_I;
    uint32_t        L1_Z;
    uint32_t        L1_Area;
    uint32_t        spatz_attached;
    uint32_t        spatz_sid;
} EltwiseInfo;

EltwiseInfo EltwiseAnalyze(
    uint32_t num_rows,
    uint32_t row_elems,
    uint64_t input_address,
    uint64_t second_address,
    uint64_t output_address)
{
    EltwiseInfo info;
    info.Input          = input_address;
    info.Second         = second_address;
    info.Output         = output_address;
    info.num_rows       = num_rows;
    info.row_elems      = row_elems;
    info.row_bytes      = row_elems * DATA_TYPE_BYTE;
#if defined(ELT_OP_ROPE)
    info.second_bytes   = ELT_HEAD_DIM * DATA_TYPE_BYTE;
#elif defined(ELT_OP_SCALE) || defined(ELT_OP_PIXEL_SHUFFLE)
    info.second_bytes   = 0;
#else
    info.second_bytes   = info.row_bytes;
#endif
    FlexPosition pos    = get_pos(flex_get_cluster_id());
    info.start_token    = ((pos.x + pos.y) % ARCH_NUM_CLUSTER_X) * ARCH_NUM_CLUSTER_Y + pos.y;
    info.L1_I           = local(0);
    info.L1_Z           = info.L1_I + info.row_bytes;
    info.L1_Area        = info.L1_Z + info.row_bytes;

    /* static: a local initializer becomes a copy onto the stack (out-of-bound TCDM access, as in LayerNorm) */
    static const uint32_t check[ARCH_NUM_CORE_PER_CLUSTER] = ARCH_SPATZ_ATTACED_CHECK_LIST;
    static const uint32_t sid[ARCH_NUM_CORE_PER_CLUSTER]   = ARCH_SPATZ_ATTACED_SID_LIST;
    info.spatz_attached = check[flex_get_core_id()];
    info.spatz_sid      = sid[flex_get_core_id()];
    return info;
}

/* Two-operand ops on one slice of vlen elements (LMUL 8, strip-mined). */
static inline void vector_lib_binary(uint32_t x_addr, uint32_t z_addr, uint32_t vlen)
{
    uint32_t avl;
#if defined(ELT_OP_SCALE) || defined(ELT_OP_AXPY)
    static const uint64_t c_alpha = 0xffffffffffff0000ull | (uint64_t)ELT_ALPHA_H;
    asm volatile("fld fa5, %0" :: "m"(c_alpha) : "fa5");
#endif
    while (vlen > 0) {
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(avl) : "r"(vlen));
        asm volatile("vle16.v v8, (%0)" :: "r"(x_addr));                 // x
#if defined(ELT_OP_SCALE)
        asm volatile("vfmul.vf v8, v8, fa5");                            // alpha x
#else
        asm volatile("vle16.v v0, (%0)" :: "r"(z_addr));                 // z or b
#if defined(ELT_OP_AXPY)
        asm volatile("vfmacc.vf v8, fa5, v0");                           // x + alpha z
#else
        asm volatile("vfadd.vv v8, v8, v0");                             // x + z
#endif
#endif
        asm volatile("vse16.v v8, (%0)" :: "r"(x_addr));
        vlen   -= avl;
        x_addr += DATA_TYPE_BYTE * avl;
        z_addr += DATA_TYPE_BYTE * avl;
    }
}

#if defined(ELT_OP_ROPE)
/* RoPE on the heads h = sid, sid + spatz_num, ... of one row in place (LMUL 1: dh/2 <= 1024). */
static inline void vector_lib_rope(uint32_t row_addr, uint32_t table_addr, uint32_t sid, uint32_t num_heads)
{
    const uint32_t half = ELT_HEAD_DIM / 2;
    uint32_t avl;
    asm volatile("vsetvli %0, %1, e16, m1, ta, ma" : "=r"(avl) : "r"(half));
    asm volatile("vle16.v v2, (%0)" :: "r"(table_addr));                         // c
    asm volatile("vle16.v v3, (%0)" :: "r"(table_addr + half * DATA_TYPE_BYTE)); // s
    for (uint32_t h = sid; h < num_heads; h += ARCH_SPATZ_ATTACED_CORES) {
        uint32_t x1 = row_addr + h * ELT_HEAD_DIM * DATA_TYPE_BYTE;
        uint32_t x2 = x1 + half * DATA_TYPE_BYTE;
        asm volatile("vle16.v v4, (%0)" :: "r"(x1));
        asm volatile("vle16.v v5, (%0)" :: "r"(x2));
        asm volatile("vfmul.vv v6, v4, v2");    // x1 c
        asm volatile("vfmul.vv v7, v5, v3");    // x2 s
        asm volatile("vfsub.vv v6, v6, v7");    // y1 = x1 c - x2 s
        asm volatile("vfmul.vv v8, v5, v2");    // x2 c
        asm volatile("vfmul.vv v9, v4, v3");    // x1 s
        asm volatile("vfadd.vv v8, v8, v9");    // y2 = x2 c + x1 s
        asm volatile("vse16.v v6, (%0)" :: "r"(x1));
        asm volatile("vse16.v v8, (%0)" :: "r"(x2));
    }
}
#endif

void EltwiseRun(EltwiseInfo *info)
{
#if defined(ELT_OP_BIAS)
    /* the bias row is the same for every row: load it once */
    if (flex_is_dm_core()) {
        flex_dma_async_1d(info->L1_Z, info->Second, info->row_bytes);
        flex_dma_async_wait_all();
    }
    flex_intra_cluster_sync();
#endif
    while (info->start_token < info->num_rows)
    {
        if (flex_is_dm_core()) {
#if defined(ELT_OP_PIXEL_SHUFFLE)
            {
                const uint32_t g     = ELT_GRID / ELT_PIX_SCALE;          /* output tokens per image side */
                const uint32_t d     = ELT_N_SIZE / (ELT_PIX_SCALE * ELT_PIX_SCALE);
                const uint32_t t     = info->start_token;
                const uint32_t img   = t / (g * g), gr = (t % (g * g)) / g, gb = t % g;
                const uint64_t src   = info->Input + ((uint64_t)img * ELT_GRID * ELT_GRID
                                       + (gr * ELT_PIX_SCALE) * ELT_GRID + gb * ELT_PIX_SCALE) * d * DATA_TYPE_BYTE;
                const uint32_t chunk = ELT_PIX_SCALE * d * DATA_TYPE_BYTE;  /* scale consecutive patches */
                flex_dma_async_2d(info->L1_I, src, chunk, chunk, ELT_GRID * d * DATA_TYPE_BYTE, ELT_PIX_SCALE);
            }
#else
            flex_dma_async_1d(info->L1_I, info->Input + info->start_token * info->row_bytes, info->row_bytes);
#endif
#if defined(ELT_OP_ADD) || defined(ELT_OP_AXPY) || defined(ELT_OP_ROPE)
            flex_dma_async_1d(info->L1_Z, info->Second + info->start_token * info->second_bytes, info->second_bytes);
#endif
            flex_dma_async_wait_all();
        }
        flex_intra_cluster_sync();

        if (info->spatz_attached) {
#if defined(ELT_OP_PIXEL_SHUFFLE)
            /* data movement only */
#elif defined(ELT_OP_ROPE)
            vector_lib_rope(info->L1_I, info->L1_Z, info->spatz_sid, info->row_elems / ELT_HEAD_DIM);
#else
            uint32_t slice = info->row_elems / ARCH_SPATZ_ATTACED_CORES;
            uint32_t off   = info->spatz_sid * slice * DATA_TYPE_BYTE;
            vector_lib_binary(info->L1_I + off, info->L1_Z + off, slice);
#endif
        }
        flex_intra_cluster_sync();

        if (flex_is_dm_core()) {
            flex_dma_async_1d(info->Output + info->start_token * info->row_bytes, info->L1_I, info->row_bytes);
            flex_dma_async_wait_all();
        }
        info->start_token += ARCH_NUM_CLUSTER;
    }
}

#endif
