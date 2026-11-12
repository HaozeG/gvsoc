#ifndef _ACTI_H_
#define _ACTI_H_

#include "flex_runtime.h"
#include "flex_printf.h"
#include "flex_redmule.h"
#include "flex_dma_pattern.h"
#include "flex_group_barrier.h"
#include "flex_libfp8.h"
#include "flex_libfp16.h"
#include "acti.h"

typedef struct ActivationInfo
{
    uint64_t        Input;
    uint64_t        Output;
    uint64_t        Gate;
    uint64_t        Bias;
    uint32_t        num_total_token;
    uint32_t        token_embedded_length;
    uint32_t        gate_enable;
    uint32_t        bias_enable;
    uint32_t        start_token;
    uint32_t        L1_I;
    uint32_t        L1_G;
    uint32_t        L1_B;
    uint32_t        L1_Area;
    uint32_t        L1_I_size;

    //Spatz Information
    uint32_t        spatz_num;
    uint32_t        spatz_check_list[ARCH_NUM_CORE_PER_CLUSTER];
    uint32_t        spatz_sid_list[ARCH_NUM_CORE_PER_CLUSTER];
    uint32_t        spatz_attached;
    uint32_t        spatz_sid;

    //L1 location for spatz partition
    uint32_t        L1SP_I;
    uint32_t        L1SP_G;
    uint32_t        L1SP_B;
}ActivationInfo;

ActivationInfo ActivationAnaylze(
    uint32_t num_total_token,
    uint32_t token_embedded_length,
    uint32_t gate_enable,
    uint32_t bias_enable,
    uint64_t input_address,
    uint64_t output_address,
    uint64_t gate_address,
    uint64_t bias_address)
{
    ActivationInfo info;

    info.Input                  = input_address;
    info.Output                 = output_address;
    info.Gate                   = gate_address;
    info.Bias                   = bias_address;
    info.token_embedded_length  = token_embedded_length;
    info.num_total_token        = num_total_token;
    info.gate_enable            = gate_enable;
    info.bias_enable            = bias_enable;
    FlexPosition pos            = get_pos(flex_get_cluster_id());
    info.start_token            = ((pos.x + pos.y) % ARCH_NUM_CLUSTER_X) * ARCH_NUM_CLUSTER_Y + pos.y;
    info.L1_I                   = local(0);
    info.L1_G                   = info.L1_I + DATA_TYPE_BYTE * info.token_embedded_length;
    info.L1_B                   = info.L1_G + DATA_TYPE_BYTE * info.token_embedded_length;
    info.L1_Area                = info.L1_B + DATA_TYPE_BYTE * info.token_embedded_length;
    info.L1_I_size              = DATA_TYPE_BYTE * info.token_embedded_length;

    //Spatz information
    info.spatz_num              = ARCH_SPATZ_ATTACED_CORES;
    uint32_t temp[ARCH_NUM_CORE_PER_CLUSTER] = ARCH_SPATZ_ATTACED_CHECK_LIST;
    for (int i = 0; i < ARCH_NUM_CORE_PER_CLUSTER; ++i){info.spatz_check_list[i] = temp[i];}
    uint32_t sidt[ARCH_NUM_CORE_PER_CLUSTER] = ARCH_SPATZ_ATTACED_SID_LIST;
    for (int i = 0; i < ARCH_NUM_CORE_PER_CLUSTER; ++i){info.spatz_sid_list[i] = sidt[i];}
    info.spatz_attached         = info.spatz_check_list[flex_get_core_id()];
    info.spatz_sid              = info.spatz_sid_list[flex_get_core_id()];

    //L1 location for spatz partition
    info.L1SP_I                 = info.L1_I  + info.spatz_sid * (info.L1_I_size  / ARCH_SPATZ_ATTACED_CORES);
    info.L1SP_G                 = info.L1_G  + info.spatz_sid * (info.L1_I_size  / ARCH_SPATZ_ATTACED_CORES);
    info.L1SP_B                 = info.L1_B  + info.spatz_sid * (info.L1_I_size  / ARCH_SPATZ_ATTACED_CORES);

    return info;
}

inline void vector_lib_sigmoid(uint32_t i_addr, uint32_t o_addr, uint32_t vlen);
inline void vector_lib_relu(uint32_t i_addr, uint32_t o_addr, uint32_t vlen);
inline void vector_lib_silu(uint32_t i_addr, uint32_t o_addr, uint32_t vlen);
inline void vector_lib_gelu(uint32_t i_addr, uint32_t o_addr, uint32_t vlen);
inline void vector_lib_gate(uint32_t i_addr, uint32_t o_addr, uint32_t vlen);
inline void vector_lib_bias(uint32_t i_addr, uint32_t o_addr, uint32_t vlen);

void ActivationRun(ActivationInfo * info)
{
    while (info->start_token < info->num_total_token)
    {
        //Load token to L1
        if (flex_is_dm_core())
        {
            flex_dma_async_1d(
                info->L1_I, /*destination*/
                info->Input + info->start_token * info->L1_I_size, /*source*/
                info->L1_I_size/*transfer size*/); //Start 2D iDMA
            flex_dma_async_wait_all(); // Wait for iDMA Finishing
            if (info->gate_enable)
            {
                flex_dma_async_1d(
                    info->L1_G, /*destination*/
                    info->Gate + info->start_token * info->L1_I_size, /*source*/
                    info->L1_I_size/*transfer size*/); //Start 2D iDMA
                flex_dma_async_wait_all(); // Wait for iDMA Finishing
            }
            if (info->bias_enable)
            {
                flex_dma_async_1d(
                    info->L1_B, /*destination*/
                    info->Bias + info->start_token * info->L1_I_size, /*source*/
                    info->L1_I_size/*transfer size*/); //Start 2D iDMA
                flex_dma_async_wait_all(); // Wait for iDMA Finishing
            }
        }
        flex_intra_cluster_sync();

        //Do Activation
        if (info->spatz_attached)
        {
#if defined(ACTI_ALGO_SIGMOID)
            vector_lib_sigmoid(
                info->L1SP_I/*i_addr*/, 
                info->L1SP_I/*o_addr*/, 
                info->token_embedded_length / ARCH_SPATZ_ATTACED_CORES/*vlen*/);
#elif defined(ACTI_ALGO_RELU)
            vector_lib_relu(
                info->L1SP_I/*i_addr*/, 
                info->L1SP_I/*o_addr*/, 
                info->token_embedded_length / ARCH_SPATZ_ATTACED_CORES/*vlen*/);
#elif defined(ACTI_ALGO_SILU)
            vector_lib_silu(
                info->L1SP_I/*i_addr*/, 
                info->L1SP_I/*o_addr*/, 
                info->token_embedded_length / ARCH_SPATZ_ATTACED_CORES/*vlen*/);
#elif defined(ACTI_ALGO_GELU)
            vector_lib_gelu(
                info->L1SP_I/*i_addr*/,
                info->L1SP_I/*o_addr*/,
                info->token_embedded_length / ARCH_SPATZ_ATTACED_CORES/*vlen*/);
#endif
            if (info->gate_enable)
            {
                vector_lib_gate(
                    info->L1SP_G/*i_addr*/, 
                    info->L1SP_I/*o_addr*/, 
                    info->token_embedded_length / ARCH_SPATZ_ATTACED_CORES/*vlen*/);
            }
            if (info->bias_enable)
            {
                vector_lib_bias(
                    info->L1SP_B/*i_addr*/, 
                    info->L1SP_I/*o_addr*/, 
                    info->token_embedded_length / ARCH_SPATZ_ATTACED_CORES/*vlen*/);
            }
        }
        flex_intra_cluster_sync();

        //Store back to HBM
        if (flex_is_dm_core())
        {
            flex_dma_async_1d(
                info->Output + info->start_token * info->L1_I_size, /*destination*/
                info->L1_I, /*source*/
                info->L1_I_size/*transfer size*/); //Start 2D iDMA
            flex_dma_async_wait_all(); // Wait for iDMA Finishing
        }

        //Update counter
        info->start_token += ARCH_NUM_CLUSTER;
    }
}

/*Sigmoid*/
inline void vector_lib_sigmoid(
    uint32_t i_addr,
    uint32_t o_addr,
    uint32_t vlen)
{
    uint32_t avl;
    /* Constants are static (initialized data): with a stack local and
     * "r"(&var), GCC may drop the store the fld depends on.
     * The 1 comes from a scalar, not from e/e: e/e is inf/inf = NaN once
     * e^-x overflows (x < about -11 in fp16).
     * The vector of ones is (-x) - (-x) + 1, exact for finite x;
     * vfmv.v.f did not produce 1.0 in the GVSoC model. */
#if defined(ACTI_FP16)
    static const uint64_t c_mone = 0xffffffffffffbc00ull; /* -1.0 */
    static const uint64_t c_one  = 0xffffffffffff3c00ull; /*  1.0 */
#elif defined(ACTI_FP8)
    static const uint64_t c_mone = 0xffffffffffffffbcull;
    static const uint64_t c_one  = 0xffffffffffffff3cull;
#endif
    asm volatile("fld fa6, %0" :: "m"(c_mone) : "fa6");
    asm volatile("fld fa4, %0" :: "m"(c_one) : "fa4");
    while(vlen > 0){
        asm volatile("vsetvli %0, %1, e" XSTR(DATA_TYPE_WIDTH) ", m8, ta, ma" : "=r"(avl) : "r"(vlen));
        asm volatile("vle" XSTR(DATA_TYPE_WIDTH) ".v v8,  (%0)" ::"r"(i_addr)); // x
        asm volatile("vfmul.vf v8, v8, fa6"); // -x
        asm volatile(".word %0\n"::"i"(0x32041857)) /*vfexp.vv v16, v8*/; // e^-x
        asm volatile("vfsub.vv v0, v8, v8"); // 0 (x finite)
        asm volatile("vfadd.vf v0, v0, fa4"); // 1
        asm volatile("vfadd.vv v16, v16, v0"); // 1 + e^-x
        asm volatile("vfdiv.vv v8, v0, v16"); // 1/(1 + e^-x)
        asm volatile("vse" XSTR(DATA_TYPE_WIDTH) ".v v8,  (%0)" ::"r"(o_addr));
        vlen -= avl;
        i_addr += DATA_TYPE_BYTE*avl;
        o_addr += DATA_TYPE_BYTE*avl;
    }
}

/*ReLU*/
inline void vector_lib_relu(
    uint32_t i_addr,
    uint32_t o_addr,
    uint32_t vlen)
{
    uint32_t avl;
    static const uint64_t c_zero = 0xffffffffffff0000ull; /* +0.0, NaN-boxed */
    asm volatile("fld fa5, %0" :: "m"(c_zero) : "fa5");
    while(vlen > 0){
        asm volatile("vsetvli %0, %1, e" XSTR(DATA_TYPE_WIDTH) ", m8, ta, ma" : "=r"(avl) : "r"(vlen));
        asm volatile("vle" XSTR(DATA_TYPE_WIDTH) ".v v8,  (%0)" ::"r"(i_addr));
        asm volatile("vfmax.vf v8, v8, fa5");
        asm volatile("vse" XSTR(DATA_TYPE_WIDTH) ".v v8,  (%0)" ::"r"(o_addr));
        vlen -= avl;
        i_addr += DATA_TYPE_BYTE*avl;
        o_addr += DATA_TYPE_BYTE*avl;
    }
}

/*SiLU*/
inline void vector_lib_silu(
    uint32_t i_addr,
    uint32_t o_addr,
    uint32_t vlen)
{
    uint32_t avl;
    /* static constants and the 1 from a scalar: see vector_lib_sigmoid */
#if defined(ACTI_FP16)
    static const uint64_t c_mone = 0xffffffffffffbc00ull; /* -1.0 */
    static const uint64_t c_one  = 0xffffffffffff3c00ull; /*  1.0 */
#elif defined(ACTI_FP8)
    static const uint64_t c_mone = 0xffffffffffffffbcull;
    static const uint64_t c_one  = 0xffffffffffffff3cull;
#endif
    asm volatile("fld fa6, %0" :: "m"(c_mone) : "fa6");
    asm volatile("fld fa4, %0" :: "m"(c_one) : "fa4");
    while(vlen > 0){
        asm volatile("vsetvli %0, %1, e" XSTR(DATA_TYPE_WIDTH) ", m8, ta, ma" : "=r"(avl) : "r"(vlen));
        asm volatile("vle" XSTR(DATA_TYPE_WIDTH) ".v v24,  (%0)" ::"r"(i_addr)); // x
        asm volatile("vfmul.vf v8, v24, fa6"); // -x
        asm volatile(".word %0\n"::"i"(0x32041857)) /*vfexp.vv v16, v8*/; // e^-x
        asm volatile("vfadd.vf v16, v16, fa4"); // 1 + e^-x
        asm volatile("vfdiv.vv v8, v24, v16"); // x/(1 + e^-x)
        asm volatile("vse" XSTR(DATA_TYPE_WIDTH) ".v v8,  (%0)" ::"r"(o_addr));
        vlen -= avl;
        i_addr += DATA_TYPE_BYTE*avl;
        o_addr += DATA_TYPE_BYTE*avl;
    }
}

/*GELU, tanh form (gelu_pytorch_tanh, as used by the SigLIP vision encoder)
 *   gelu(x) = 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))
 *           = x / (1 + exp(-2 sqrt(2/pi) x (1 + 0.044715 x^2)))
 * All steps in the data type (fp16), one rounding per instruction.
 * The constant 1 comes from a scalar register, not from e/e, so that exp
 * overflow (x < about -4) gives x/inf = -0 and underflow gives x/1 = x.
 * x (1 + c x^2) overflows only when x^2 does (|x| > 256), and then still
 * gives the right limit (inf or -0).
 * vfexp is the custom encoding 0x32041857: vd = v16, vs1 = v8.
 * Scalars are NaN-boxed 64-bit values so that fld leaves a valid fp16
 * in the low bits whatever the register file does with the upper bits.
 */
inline void vector_lib_gelu(
    uint32_t i_addr,
    uint32_t o_addr,
    uint32_t vlen)
{
    uint32_t avl;
#if defined(ACTI_FP16)
    static const uint64_t c_cube  = 0xffffffffffff29b9ull; /* 0.044715   */
    static const uint64_t c_one   = 0xffffffffffff3c00ull; /* 1.0        */
    static const uint64_t c_mtwok = 0xffffffffffffbe62ull; /* -1.5957691 = -2 sqrt(2/pi) */
#elif defined(ACTI_FP8)
    static const uint64_t c_cube  = 0xffffffffffffff29ull;
    static const uint64_t c_one   = 0xffffffffffffff3cull;
    static const uint64_t c_mtwok = 0xffffffffffffffbeull;
#endif
    asm volatile("fld fa3, %0" :: "m"(c_cube) : "fa3");
    asm volatile("fld fa4, %0" :: "m"(c_one) : "fa4");
    asm volatile("fld fa7, %0" :: "m"(c_mtwok) : "fa7");
    while(vlen > 0){
        asm volatile("vsetvli %0, %1, e" XSTR(DATA_TYPE_WIDTH) ", m8, ta, ma" : "=r"(avl) : "r"(vlen));
        asm volatile("vle" XSTR(DATA_TYPE_WIDTH) ".v v24,  (%0)" ::"r"(i_addr)); // x
        asm volatile("vfmul.vv v8, v24, v24");   // x^2
        asm volatile("vfmul.vf v8, v8, fa3");    // c x^2
        asm volatile("vfadd.vf v8, v8, fa4");    // 1 + c x^2
        asm volatile("vfmul.vv v8, v8, v24");    // u = x (1 + c x^2)
        asm volatile("vfmul.vf v8, v8, fa7");    // -2k u
        asm volatile(".word %0\n"::"i"(0x32041857)) /*vfexp.vv v16, v8*/; // e^(-2k u)
        asm volatile("vfadd.vf v16, v16, fa4");  // 1 + e^(-2k u)
        asm volatile("vfdiv.vv v8, v24, v16");   // x / (1 + e^(-2k u))
        asm volatile("vse" XSTR(DATA_TYPE_WIDTH) ".v v8,  (%0)" ::"r"(o_addr));
        vlen -= avl;
        i_addr += DATA_TYPE_BYTE*avl;
        o_addr += DATA_TYPE_BYTE*avl;
    }
}

/*Gate*/
inline void vector_lib_gate(
    uint32_t i_addr,
    uint32_t o_addr,
    uint32_t vlen)
{
    uint32_t avl;
    while(vlen > 0){
        asm volatile("vsetvli %0, %1, e" XSTR(DATA_TYPE_WIDTH) ", m8, ta, ma" : "=r"(avl) : "r"(vlen));
        asm volatile("vle" XSTR(DATA_TYPE_WIDTH) ".v v8,  (%0)" ::"r"(i_addr));
        asm volatile("vle" XSTR(DATA_TYPE_WIDTH) ".v v0,  (%0)" ::"r"(o_addr));
        asm volatile("vfmul.vv v8, v8, v0");
        asm volatile("vse" XSTR(DATA_TYPE_WIDTH) ".v v8,  (%0)" ::"r"(o_addr));
        vlen -= avl;
        i_addr += DATA_TYPE_BYTE*avl;
        o_addr += DATA_TYPE_BYTE*avl;
    }
}

/*Bias*/
inline void vector_lib_bias(
    uint32_t i_addr,
    uint32_t o_addr,
    uint32_t vlen)
{
    uint32_t avl;
    while(vlen > 0){
        asm volatile("vsetvli %0, %1, e" XSTR(DATA_TYPE_WIDTH) ", m8, ta, ma" : "=r"(avl) : "r"(vlen));
        asm volatile("vle" XSTR(DATA_TYPE_WIDTH) ".v v8,  (%0)" ::"r"(i_addr));
        asm volatile("vle" XSTR(DATA_TYPE_WIDTH) ".v v0,  (%0)" ::"r"(o_addr));
        asm volatile("vfadd.vv v8, v8, v0");
        asm volatile("vse" XSTR(DATA_TYPE_WIDTH) ".v v8,  (%0)" ::"r"(o_addr));
        vlen -= avl;
        i_addr += DATA_TYPE_BYTE*avl;
        o_addr += DATA_TYPE_BYTE*avl;
    }
}

#endif