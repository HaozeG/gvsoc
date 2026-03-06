#ifndef _FLEX_LIBfp16_H_
#define _FLEX_LIBfp16_H_

#include "flex_types.h"
#include "flex_runtime_api.h"

// Convert float to fp16 (half-precision)
fp16 float_to_fp16(float value) {
    FloatBits floatBits;
    floatBits.f = value;

    uint16_t sign = floatBits.parts.sign << 15;
    int32_t exponent = floatBits.parts.exponent - 127 + 15; // adjust bias from 127 to 15
    uint32_t mantissa = floatBits.parts.mantissa >> 13;     // reduce to 10 bits

    if (exponent <= 0) {
        if (exponent < -10) return sign;   // too small
        mantissa = (floatBits.parts.mantissa | 0x800000) >> (1 - exponent);
        return sign | mantissa;
    } else if (exponent >= 0x1F) {
        return sign | 0x7C00;  // overflow to infinity
    }
    return sign | (exponent << 10) | mantissa;
}

// Convert fp16 to float
float fp16_to_float(fp16 value) {
    FloatBits floatBits;
    floatBits.parts.sign = (value >> 15) & 0x1;
    int32_t exponent = (value >> 10) & 0x1F;
    floatBits.parts.exponent = (exponent == 0) ? 0 : exponent + 127 - 15;
    floatBits.parts.mantissa = (value & 0x3FF) << 13;
    return floatBits.f;
}

// Fused multiply-add for fp16
fp16 fp16_fma(fp16 a, fp16 b, fp16 c) {
    float fa = fp16_to_float(a);
    float fb = fp16_to_float(b);
    float fc = fp16_to_float(c);
    float result = (fa * fb) + fc;
    return float_to_fp16(result);
}

void matmul_fp16(fp16 * z, fp16 * y, fp16 * x, fp16 * w, uint16_t m_size, uint16_t n_size, uint16_t k_size){
    for (int i = 0; i < m_size; ++i)
    {
        for (int j = 0; j < k_size; ++j)
        {
            z[i * k_size + j] = y[i * k_size + j];
            for (int k = 0; k < n_size; ++k)
            {
                z[i * k_size + j] = fp16_fma(x[i * n_size + k], w[k * k_size + j], z[i * k_size + j]);
            }
        }
    }
}




/*
 * asm_fp16_compare:
 *   Returns -1 if (*a < *b), 0 if (*a == *b), +1 if (*a > *b).
 *   Purely uses half-precision RISC-V instructions for the compare.
 */
int asm_fp16_compare(const fp16 *a, const fp16 *b)
{
    // Copy inputs into local half-precision variables.
    fp16 av = *a;
    fp16 bv = *b;

    // We'll compute diff = (av - bv) in half precision.
    // Then move the bits of diff into a 16-bit integer.
    uint16_t diff_bits;

    asm volatile(
        // Move av, bv into FP registers
        "fmv.h.x    ft0, %[av]         \n"
        "fmv.h.x    ft1, %[bv]         \n"
        // diff = av - bv
        "fsub.h     ft2, ft0, ft1      \n"
        // Move bits of diff from ft2 into diff_bits
        "fmv.x.h    %[diff_bits], ft2  \n"
        : [diff_bits] "=r"(diff_bits)
        : [av] "r"(av), [bv] "r"(bv)
        : "ft0", "ft1", "ft2"
    );

    // Now interpret diff_bits as a 16-bit float:
    // Sign bit is bit 15: if set => negative => (av < bv)
    // If entire bits == 0 => exact zero => av == bv
    // Else => av > bv

    if (diff_bits == 0) {
        return 0;    // av == bv
    } else if (diff_bits & 0x8000) {
        return -1;   // negative sign => av < bv
    } else {
        return 1;    // positive => av > bv
    }
}


#endif