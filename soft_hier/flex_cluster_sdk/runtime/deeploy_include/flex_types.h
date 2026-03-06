#ifndef FLEX_TYPES_H
#define FLEX_TYPES_H

#include <stdint.h>

typedef union {
    float f;
    struct {
        uint32_t mantissa : 23;
        uint32_t exponent : 8;
        uint32_t sign : 1;
    } parts;
} FloatBits;

typedef uint16_t fp16;

fp16 float_to_fp16(float value);
float fp16_to_float(fp16 value);
fp16 fp16_fma(fp16 a, fp16 b, fp16 c);

inline void asm_fp16_div(const fp16 *a, const fp16 *b, fp16 *c) {
    // Load the half-precision values from memory into local variables
    // Operation: cv = av / bv
    fp16 av = *a;
    fp16 bv = *b;
    fp16 cv;

    // Perform the division using RISC-V half-precision instructions
    asm volatile(
        "fmv.h.x ft0, %[av]\n"   // move half value 'av' into ft0
        "fmv.h.x ft1, %[bv]\n"   // move half value 'bv' into ft1
        "fdiv.h   ft2, ft0, ft1\n"   // ft2 = ft0 / ft1 (half-precision)
        "fmv.x.h  %[cv], ft2\n"  // move the result from ft2 to our local variable 'cv'
        : [cv] "=r"(cv)
        : [av] "r"(av), [bv] "r"(bv)
        : "ft0", "ft1", "ft2"    // clobbered registers
    );

    // Store the result back to memory
    *c = cv;
}


inline void asm_fp16_exp(const fp16 *inp, fp16 *oup)
{
    // We'll store the input and output in local half variables
    fp16 x = *inp;
    fp16 r; // Will hold the final result

    // Some half-precision constants for the series coefficients.
    // These are approximate but good enough for a small demonstration.
    //   1.0   = 0x3C00 in half
    //   0.5   = 0x3800
    //   1/6 ≈ 0.1667  -> ~ 0.166015625 in half (0x3555)
    //   1/24 ≈ 0.0417 -> ~ 0.0416564941 in half (0x2AAA)
    static const fp16 c1      = 0x3C00;     // 1
    static const fp16 cHalf   = 0x3800;     // 1/2
    static const fp16 cOneSix = 0x3555;  // 1/6
    static const fp16 cOne24  = 0x2AAA;  // 1/24

    //
    // We will do all arithmetic in half-precision registers
    // using inline assembly with RISC-V Zfh instructions:
    //   fmv.h.x   (move half bits from int register to FP register)
    //   fmul.h    (half-precision multiply)
    //   fadd.h    (half-precision add)
    //   fmv.x.h   (move half bits from FP register to int register)
    //
    // Register usage (ft0–ft3 are temporary floating registers):
    //   ft0 = x
    //   ft1 = accumulated sum
    //   ft2, ft3 = scratch for multiplications and constants
    //

    asm volatile(
        // --- Load x into ft0 ---
        "fmv.h.x   ft0, %[x]       \n"

        // --- Start with ft1 = 1.0 ---
        "fmv.h.x   ft1, %[c1]      \n"

        // ft1 = ft1 + x   => 1 + x
        "fadd.h    ft1, ft1, ft0   \n"

        // --- Compute x^2/2 ---
        // ft2 = x^2
        "fmul.h    ft2, ft0, ft0   \n"
        // ft3 = 0.5
        "fmv.h.x   ft3, %[cHalf]   \n"
        // ft2 = ft2 * ft3 => x^2/2
        "fmul.h    ft2, ft2, ft3   \n"
        // ft1 = ft1 + ft2 => (1 + x) + x^2/2
        "fadd.h    ft1, ft1, ft2   \n"

        // --- Compute x^3/6 ---
        // ft2 = (x^2/2) * x = x^3/2
        "fmul.h    ft2, ft2, ft0   \n"
        // ft3 = 1/6
        "fmv.h.x   ft3, %[cOneSix] \n"
        // ft2 = ft2 * (1/6) => x^3/6
        "fmul.h    ft2, ft2, ft3   \n"
        // ft1 += x^3/6
        "fadd.h    ft1, ft1, ft2   \n"

        // --- Compute x^4/24 ---
        // ft2 = (x^3/6) * x = x^4/6
        "fmul.h    ft2, ft2, ft0   \n"
        // ft3 = 1/24
        "fmv.h.x   ft3, %[cOne24]  \n"
        // ft2 = ft2 * (1/24) => x^4/24
        "fmul.h    ft2, ft2, ft3   \n"
        // ft1 += x^4/24
        "fadd.h    ft1, ft1, ft2   \n"

        // --- Move the final sum (ft1) to r ---
        "fmv.x.h   %[r], ft1       \n"

        : [r] "=r"(r)          // Output
        : [x] "r"(x), 
          [c1] "r"(c1), 
          [cHalf] "r"(cHalf), 
          [cOneSix] "r"(cOneSix), 
          [cOne24] "r"(cOne24)
        : "ft0", "ft1", "ft2", "ft3"
    );

    // Store result back to memory
    *oup = r;
}


inline void asm_fp16_sigmoid(const fp16 *inp, fp16 *out)
{
    // We'll need a few temporaries in half precision.
    fp16 x      = *inp;   // input
    fp16 negX;            // will hold -x
    fp16 eNegX;           // will hold exp(-x)
    fp16 sum;             // will hold (1 + eNegX)

    // A half-precision constant "1.0"
    static const fp16 ONE = 0x3C00;

    //----------------------------------------------------------------
    // 1) negX = -x
    //
    //   RISC-V Zfh does not always have a “fneg.h” mnemonic, but we
    //   can use fsgnjn.h  (sign-injection) to flip the sign of x:
    //----------------------------------------------------------------
    asm volatile(
        "fmv.h.x    ft0, %[x]          \n"  // ft0 = x
        "fsgnjn.h   ft0, ft0, ft0      \n"  // ft0 = -ft0
        "fmv.x.h    %[negX], ft0       \n"  // negX = -x
        : [negX] "=r" (negX)
        : [x] "r" (x)
        : "ft0"
    );

    //----------------------------------------------------------------
    // 2) eNegX = exp(negX)
    //    (Use your existing half-precision exponent routine)
    //----------------------------------------------------------------
    asm_fp16_exp(&negX, &eNegX);

    //----------------------------------------------------------------
    // 3) sum = 1 + eNegX
    //    We'll do half-precision addition via inline assembly.
    //----------------------------------------------------------------
    asm volatile(
        "fmv.h.x    ft1, %[one]       \n"  // ft1 = 1.0
        "fmv.h.x    ft2, %[eNegX]     \n"  // ft2 = eNegX
        "fadd.h     ft1, ft1, ft2     \n"  // ft1 = 1 + eNegX
        "fmv.x.h    %[sum], ft1       \n"  // sum = (1 + eNegX)
        : [sum] "=r" (sum)
        : [one] "r" (ONE),
          [eNegX] "r" (eNegX)
        : "ft1", "ft2"
    );

    //----------------------------------------------------------------
    // 4) *out = 1 / sum = 1 / (1 + e^{-x})
    //    Use your existing half-precision division function:
    //----------------------------------------------------------------
    asm_fp16_div(&ONE, &sum, out);

    // done:  *out is the sigmoid of *inp
}

#define VFR_TYPE_ENCODE(funct6, m, vs2, vs1, funct3, vd, opcode)                    \
    ((funct6 << 26) | (m << 25) | (vs2 << 20) | (vs1 << 15) | (funct3 << 12) | (vd << 7) | \
     (opcode))

inline void asm_rvv_exp(uint32_t vs1_num, uint32_t vd_num){
    asm volatile(".word %0\n"::"i"(VFR_TYPE_ENCODE(0b001100, 0b1, 0b00000, vs1_num, 0b001, vd_num, 0b1010111)));
}

#endif // FLEX_TYPES_H