#ifndef _LAYERNORM_H_
#define _LAYERNORM_H_

/*
 * LayerNorm over the last dimension, fp16 in and out, on the Spatz cores:
 *   y = (x - mean) / sqrt(var + eps) * gamma + beta
 *
 * Mapping
 * - Each cluster takes blocks of ARCH_SPATZ_ATTACED_CORES consecutive tokens;
 *   one block is one DMA in and one DMA out. Each Spatz core normalizes one
 *   row of the block, so no reduction crosses cores.
 * - gamma and beta (one row each) are loaded into L1 once.
 *
 * Precision (per row)
 * 1. sum(x) is accumulated in fp32 lanes (vfwadd.wv) and reduced in fp32.
 * 2. sum((x - m)^2) with m = fp16(mean), in fp32 (vfwsub.vf, vfmacc.vv).
 *    var = sum/n - (mean - m)^2 corrects for m != mean.
 * 3. var + eps and 1/sqrt in scalar fp32 (eps is below fp16 resolution).
 * 4. y = (((x - m) - m2) * r) * gamma + beta in fp16, with
 *    m2 = fp16(mean - m) and r = fp16(1/sqrt(var + eps)).
 * fp16 accumulation would overflow once the row RMS exceeds about 9 (n = 768).
 */

#include "flex_runtime.h"
#include "flex_printf.h"
#include "flex_dma_pattern.h"
#include "flex_libfp16.h"
#include "lnorm.h"

typedef struct LayerNormInfo
{
    uint64_t        Input;
    uint64_t        Output;
    uint64_t        Gamma;
    uint64_t        Beta;
    uint32_t        num_total_token;
    uint32_t        token_embedded_length;
    uint32_t        first_token;    // first token of this cluster's first block
    uint32_t        block_tokens;   // tokens per block = Spatz cores per cluster
    uint32_t        row_bytes;
    uint32_t        L1_G;           // gamma row
    uint32_t        L1_B;           // beta row
    uint32_t        L1_I;           // block_tokens rows, normalized in place
    uint32_t        L1_Area;        // first free L1 address

    //Spatz information
    uint32_t        spatz_attached;
    uint32_t        spatz_sid;
}LayerNormInfo;

LayerNormInfo LayerNormAnalyze(
    uint32_t num_total_token,
    uint32_t token_embedded_length,
    uint64_t input_address,
    uint64_t output_address,
    uint64_t gamma_address,
    uint64_t beta_address)
{
    LayerNormInfo info;

    info.Input                  = input_address;
    info.Output                 = output_address;
    info.Gamma                  = gamma_address;
    info.Beta                   = beta_address;
    info.num_total_token        = num_total_token;
    info.token_embedded_length  = token_embedded_length;
    info.block_tokens           = ARCH_SPATZ_ATTACED_CORES;
    info.row_bytes              = DATA_TYPE_BYTE * token_embedded_length;
    FlexPosition pos            = get_pos(flex_get_cluster_id());
    uint32_t cluster_rank       = ((pos.x + pos.y) % ARCH_NUM_CLUSTER_X) * ARCH_NUM_CLUSTER_Y + pos.y;
    info.first_token            = cluster_rank * info.block_tokens;
    info.L1_G                   = local(0);
    info.L1_B                   = info.L1_G + info.row_bytes;
    info.L1_I                   = info.L1_B + info.row_bytes;
    info.L1_Area                = info.L1_I + info.block_tokens * info.row_bytes;

    // static const and indexed: a local array initializer is compiled into a
    // vector memcpy onto the stack, which Spatz cannot reach (only TCDM)
    static const uint32_t check[ARCH_NUM_CORE_PER_CLUSTER] = ARCH_SPATZ_ATTACED_CHECK_LIST;
    static const uint32_t sid[ARCH_NUM_CORE_PER_CLUSTER]   = ARCH_SPATZ_ATTACED_SID_LIST;
    info.spatz_attached         = check[flex_get_core_id()];
    info.spatz_sid              = sid[flex_get_core_id()];

    return info;
}

/* fp32 -> fp16, round to nearest even (flex_libfp16.h's float_to_fp16
 * truncates and mishandles subnormals) */
static inline uint16_t lnorm_f32_to_f16(float f)
{
    union { float f; uint32_t u; } v = { f };
    uint32_t u    = v.u;
    uint32_t sign = (u >> 16) & 0x8000;
    u &= 0x7fffffff;
    if (u >= 0x7f800000) return sign | (u > 0x7f800000 ? 0x7e00 : 0x7c00);
    if (u >= 0x477ff000) return sign | 0x7c00;          /* >= 65520 rounds to inf */
    if (u <  0x38800000)                                /* below 2^-14: subnormal */
    {
        if (u < 0x33000000) return sign;                /* below 2^-25 */
        uint32_t e     = u >> 23;
        uint32_t m     = (u & 0x7fffff) | 0x800000;
        uint32_t shift = 126 - e;
        uint32_t r     = m >> shift;
        uint32_t rem   = m & ((1u << shift) - 1);
        uint32_t half  = 1u << (shift - 1);
        if (rem > half || (rem == half && (r & 1))) r++;
        return sign | r;
    }
    uint32_t r   = (u >> 13) - (112u << 10);
    uint32_t rem = u & 0x1fff;
    if (rem > 0x1000 || (rem == 0x1000 && (r & 1))) r++;
    return sign | r;
}

/* fp16 -> fp32, exact (subnormals included) */
static inline float lnorm_f16_to_f32(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t e    = (h >> 10) & 0x1f;
    uint32_t m    = h & 0x3ff;
    union { uint32_t u; float f; } v;
    if (e == 0)
    {
        float f = (float)m * 5.9604644775390625e-08f;   /* m * 2^-24 */
        return sign ? -f : f;
    }
    v.u = sign | (e == 31 ? 0x7f800000 : (e + 112) << 23) | (m << 13);
    return v.f;
}

void vector_lib_layernorm_row(uint32_t x_addr, uint32_t g_addr, uint32_t b_addr, uint32_t n);

void LayerNormRun(LayerNormInfo * info)
{
    //gamma and beta, once
    if (flex_is_dm_core())
    {
        flex_dma_async_1d(info->L1_G, info->Gamma, info->row_bytes);
        flex_dma_async_1d(info->L1_B, info->Beta,  info->row_bytes);
        flex_dma_async_wait_all();
    }
    flex_intra_cluster_sync();

    uint32_t stride = ARCH_NUM_CLUSTER * info->block_tokens;
    for (uint32_t t = info->first_token; t < info->num_total_token; t += stride)
    {
        uint32_t rows = info->num_total_token - t;
        if (rows > info->block_tokens) rows = info->block_tokens;

        //Load block to L1
        if (flex_is_dm_core())
        {
            flex_dma_async_1d(info->L1_I, info->Input + (uint64_t)t * info->row_bytes, rows * info->row_bytes);
            flex_dma_async_wait_all();
        }
        flex_intra_cluster_sync();

        //One row per Spatz core
        if (info->spatz_attached && info->spatz_sid < rows)
        {
            vector_lib_layernorm_row(
                info->L1_I + info->spatz_sid * info->row_bytes,
                info->L1_G,
                info->L1_B,
                info->token_embedded_length);
        }
        flex_intra_cluster_sync();

        //Store back to HBM
        if (flex_is_dm_core())
        {
            flex_dma_async_1d(info->Output + (uint64_t)t * info->row_bytes, info->L1_I, rows * info->row_bytes);
            flex_dma_async_wait_all();
        }
    }
}

/* sum over the first vl_max fp32 lanes of v16, returned through an integer
 * register (vmv.x.s). Storing it to L1 with vse32 and reading it back with a
 * scalar load returned the old value: the load overtook the vector store. */
static inline float vector_lib_reduce_f32(uint32_t vl_max)
{
    uint32_t avl;
    union { uint32_t u; float f; } r;
    asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(avl) : "r"(vl_max));
    asm volatile("vmv.s.x v24, zero");
    asm volatile("vfredusum.vs v24, v16, v24");
    asm volatile("vmv.x.s %0, v24" : "=r"(r.u));
    return r.f;
}

/* fp16 x, fp32 lanes: zero the accumulator v16 over vl_max lanes */
static inline uint32_t vector_lib_zero_acc(uint32_t n)
{
    uint32_t vl_max;
    asm volatile("vsetvli %0, %1, e16, m4, ta, ma" : "=r"(vl_max) : "r"(n));
    asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl_max) : "r"(vl_max));
    asm volatile("vmv.v.i v16, 0");
    return vl_max;
}

void vector_lib_layernorm_row(
    uint32_t x_addr,
    uint32_t g_addr,
    uint32_t b_addr,
    uint32_t n)
{
    uint32_t avl, vlen, addr;

    //1. sum(x) in fp32
    uint32_t vl_max = vector_lib_zero_acc(n);
    for (vlen = n, addr = x_addr; vlen > 0; vlen -= avl, addr += 2 * avl)
    {
        asm volatile("vsetvli %0, %1, e16, m4, tu, ma" : "=r"(avl) : "r"(vlen));
        asm volatile("vle16.v v4, (%0)" ::"r"(addr));
        asm volatile("vfwadd.wv v16, v16, v4");
    }
    float s1   = vector_lib_reduce_f32(vl_max);
    float mean = s1 / (float)n;
    uint16_t m_h = lnorm_f32_to_f16(mean);
    uint64_t m_box = 0xffffffffffff0000ull | m_h;
    asm volatile("fld fa5, %0" :: "m"(m_box) : "fa5");

    //2. sum((x - m)^2) in fp32
    vector_lib_zero_acc(n);
    for (vlen = n, addr = x_addr; vlen > 0; vlen -= avl, addr += 2 * avl)
    {
        asm volatile("vsetvli %0, %1, e16, m4, tu, ma" : "=r"(avl) : "r"(vlen));
        asm volatile("vle16.v v4, (%0)" ::"r"(addr));
        asm volatile("vfwsub.vf v8, v4, fa5");
        asm volatile("vsetvli %0, %1, e32, m8, tu, ma" : "=r"(avl) : "r"(avl));
        asm volatile("vfmacc.vv v16, v8, v8");
    }
    float s2    = vector_lib_reduce_f32(vl_max);
    float delta = mean - lnorm_f16_to_f32(m_h);
    // var = s2/n - delta^2 with an explicit fnmsub.s. GVSoC workaround: in the
    // Snitch FP subsystem model fmv.w.x and fdiv.s write fp32 results without
    // NaN-boxing (upper 32 bits 0) while flw/fsub.s box them; the fnmsub.s that
    // GCC fuses here (-ffast-math) then read the unboxed s2/n as NaN and the
    // clamp below turned it into var = 0. Root cause in the simulator: open.
    float s2n   = s2 / (float)n;
    float var;
    asm volatile("fnmsub.s %0, %1, %1, %2" : "=f"(var) : "f"(delta), "f"(s2n));
    if (var < 0.0f) var = 0.0f;
    float rstd  = 1.0f / __builtin_sqrtf(var + LNORM_EPS_F);
    uint64_t r_box  = 0xffffffffffff0000ull | lnorm_f32_to_f16(rstd);
    // m2 = fp16(mean - m): without it the output is off by (mean - m) * r,
    // up to half an fp16 step of the mean divided by sigma (0.04 at mean 10,
    // sigma 0.1). delta is exact in fp32 (mean and m are within one fp16 step).
    uint64_t m2_box = 0xffffffffffff0000ull | lnorm_f32_to_f16(delta);
    // "m" operands: with "r"(&var) the compiler does not know the asm reads
    // var, and drops the store to the stack as dead.
    // Reload the scalars: the float code above may have used fa5.
    asm volatile("fld fa5, %0" :: "m"(m_box) : "fa5");
    asm volatile("fld fa6, %0" :: "m"(r_box) : "fa6");
    asm volatile("fld fa4, %0" :: "m"(m2_box) : "fa4");

    //3. y = (((x - m) - m2) * r) * gamma + beta in fp16, in place
    for (vlen = n, addr = x_addr; vlen > 0; vlen -= avl, addr += 2 * avl, g_addr += 2 * avl, b_addr += 2 * avl)
    {
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(avl) : "r"(vlen));
        asm volatile("vle16.v v8, (%0)" ::"r"(addr));
        asm volatile("vfsub.vf v8, v8, fa5");
        asm volatile("vfsub.vf v8, v8, fa4");
        asm volatile("vfmul.vf v8, v8, fa6");
        asm volatile("vle16.v v16, (%0)" ::"r"(g_addr));
        asm volatile("vfmul.vv v8, v8, v16");
        asm volatile("vle16.v v16, (%0)" ::"r"(b_addr));
        asm volatile("vfadd.vv v8, v8, v16");
        asm volatile("vse16.v v8, (%0)" ::"r"(addr) : "memory");
    }
}

#endif
