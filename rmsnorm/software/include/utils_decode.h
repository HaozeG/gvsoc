// Author: Haoze Gao <gaohao@ethz.ch>

#ifndef UTILS_DECODE_H
#define UTILS_DECODE_H

/*
NAIVE IMPLEMENTATION: 
enable CENTRALIZED_MOE, disable all others
#include "moe_decode_centralized.h" in main.c

CLUSTER COLORING+DATA REDISTRIBUTION: 
disable all
#include "moe_decode.h" in main.c

+SPATZ VECTORIZATION: 
enable SPATZ_ENABLE, disable all others
#include "moe_decode.h" in main.c

+SYNCHRONIZATION REDUCTION: 
enable SPATZ_ENABLE, SYNC_REDUCE, disable all others
#include "moe_decode.h" in main.c
*/

// #define CENTRALIZED_MOE
#define SPATZ_ENABLE
#define SYNC_REDUCE

#include "flex_libfp16.h"
#include "flex_runtime.h"
#include "flex_printf.h"
#include "flex_redmule.h"
#include "flex_dma_pattern.h"
#include "flex_group_barrier.h"

// use float16 as data type
#define DTYPE fp16
#define DATA_SIZE_BYTES 2
// Parameters for GEMV
// NOTE: designed such that TILE_WIDTH * 8 = width of output matrix
// This is for dedicated preload data distribution to enable 1d DMA
// NOTE: the following TILE_WIDTH should be different to each other, otherwise change the addr_shift value assignment in gemv()
// #define TILE_WIDTH 256
#define TILE_WIDTH_GATE 256
// used for w1 and w3
#define TILE_WIDTH_EXPERT_0 256
// #define TILE_WIDTH_EXPERT_0 64
// used for w2
#define TILE_WIDTH_EXPERT_1 448
// #define TILE_WIDTH_EXPERT_1 64
// Parameter for element-wise functions
#define ELEMENT_WISE_TILE_WIDTH 16
#define SPATZ_VL 256
#define SPATZ_VL_MIN 8

typedef void (*element_wise_op_1_in_t)(const fp16* input, fp16* output);
typedef void (*element_wise_op_2_in_t)(const fp16* input1, const fp16* input2, fp16* output);
typedef void (*element_wise_op_2_in_const_t)(const fp16* input1, const fp16* in_const, fp16* output);
// NOTE: designed for NUM_CLUSTER <= 16
typedef uint16_t cluster_map_t;
// 2.5 in float
fp16 route_scale = (fp16)0x4100;

int32_t min(int32_t a, int32_t b) {
    return (a < b) ? a : b;
}

void print64(uint64_t input) {
    printf(" ##0x%08x%08x## ", (uint32_t)(input >> 32), (uint32_t)input);
}

// TODO:
typedef struct RMSNormInfo {
    uint32_t tile_size_n_token; // number of tokens in each tile
    uint32_t tile_size_dim;     // number of dimensions in each tile
    uint32_t in_offset;         // offset of input data in TCDM
    uint32_t out_offset;        // offset of output data in TCDM
} RMSNormInfo;

#endif