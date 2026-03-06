// flex_runtime_api.h
#ifndef FLEX_RUNTIME_API_H
#define FLEX_RUNTIME_API_H

#include <stdint.h>
#include "flex_cluster_arch.h"
#include "flex_printf_api.h"

/* from flex_runtime.h */
#define ARCH_NUM_CLUSTER            (ARCH_NUM_CLUSTER_X*ARCH_NUM_CLUSTER_Y)
#define ARCH_SYNC_SIZE              (ARCH_SYNC_INTERLEAVE + ARCH_SYNC_SPECIAL_MEM)
#define cluster_index(x,y)          ((y)*ARCH_NUM_CLUSTER_X+(x))
#define local(offset)               (ARCH_CLUSTER_TCDM_BASE+offset)
#define zomem(offset)               (ARCH_CLUSTER_ZOMEM_BASE+offset)
#define remote_cid(cid,offset)      (ARCH_CLUSTER_TCDM_REMOTE+cid*ARCH_CLUSTER_TCDM_SIZE+offset)
#define remote_xy(x,y,offset)       (ARCH_CLUSTER_TCDM_REMOTE+cluster_index(x,y)*ARCH_CLUSTER_TCDM_SIZE+offset)
#define remote_pos(pos,offset)      (ARCH_CLUSTER_TCDM_REMOTE+cluster_index(pos.x,pos.y)*ARCH_CLUSTER_TCDM_SIZE+offset)
#define hbm_addr(offset)            ((uint64_t)ARCH_HBM_START_BASE+offset)
#define hbm_west(nid,offset)        ((uint64_t)ARCH_HBM_START_BASE+(nid)*ARCH_HBM_NODE_ADDR_SPACE+offset)
#define hbm_north(nid,offset)       ((uint64_t)ARCH_HBM_START_BASE+(nid)*ARCH_HBM_NODE_ADDR_SPACE+ARCH_HBM_NODE_ADDR_SPACE*ARCH_NUM_CLUSTER_Y+offset)
#define hbm_east(nid,offset)        ((uint64_t)ARCH_HBM_START_BASE+(nid)*ARCH_HBM_NODE_ADDR_SPACE+ARCH_HBM_NODE_ADDR_SPACE*(ARCH_NUM_CLUSTER_Y+ARCH_NUM_CLUSTER_X)+offset)
#define hbm_south(nid,offset)       ((uint64_t)ARCH_HBM_START_BASE+(nid)*ARCH_HBM_NODE_ADDR_SPACE+ARCH_HBM_NODE_ADDR_SPACE*2*ARCH_NUM_CLUSTER_Y+ARCH_HBM_NODE_ADDR_SPACE*ARCH_NUM_CLUSTER_X+offset)
#define is_hbm_region(addr)         (addr >= ARCH_HBM_START_BASE)
#define is_power_of_two(value)      (((value) & ((value) - 1)) == 0)

/*******************
* Cluster Position *
*******************/

typedef struct FlexPosition
{
    uint32_t x;
    uint32_t y;
} FlexPosition;

FlexPosition get_pos(uint32_t cluster_id);
FlexPosition right_pos(FlexPosition pos);
FlexPosition left_pos(FlexPosition pos);
FlexPosition top_pos(FlexPosition pos);
FlexPosition bottom_pos(FlexPosition pos);

/*******************
*  Core Position   *
*******************/

uint32_t flex_get_cluster_id();
uint32_t flex_get_core_id();
uint32_t flex_is_dm_core();
uint32_t flex_is_first_core();

/*******************
*  Global Barrier  *
*******************/

uint32_t flex_get_enable_value();
uint32_t flex_get_disable_value();
uint32_t flex_get_barrier_num_cluster();
uint32_t flex_get_barrier_num_cluster_x();
uint32_t flex_get_barrier_num_cluster_y();
void flex_annotate_barrier(uint32_t type);
void flex_reset_barrier(uint32_t* barrier);
uint32_t flex_amo_fetch_add(uint32_t* barrier);
void flex_intra_cluster_sync();
inline void flex_wakeup_clusters(uint8_t row_mask, uint8_t col_mask)
{
    volatile uint32_t * wakeup_reg   = (volatile uint32_t *) (ARCH_CLUSTER_REG_BASE+28);
    uint32_t value = (((uint32_t)col_mask) << 16) | row_mask;
    *wakeup_reg = value;
}

inline void flex_wakeup_all_clusters()
{
    volatile uint32_t * wakeup_reg   = (volatile uint32_t *) (ARCH_CLUSTER_REG_BASE+28);
    uint32_t value = flex_get_disable_value();
    *wakeup_reg = value;
}
void flex_barrier_init();
void flex_global_barrier();
void flex_global_barrier_polling();
void flex_barrier_xy_init();
void flex_global_barrier_xy();
void flex_global_barrier_xy_polling();

/*******************
*        EoC       *
*******************/

void flex_eoc(uint32_t val);

/*******************
*   Perf Counter   *
*******************/

void flex_timer_start();
void flex_timer_end();

/*******************
*      Logging     *
*******************/

void flex_log_char(char c);
void flex_print(char * str);
void flex_print_int(uint32_t data);

/************************
*      Stop at Time     *
************************/

void flex_sat(uint32_t val);


/* from flex_alloc.h */
/********************
*  Data Allocation  *
********************/

/*
Desc: Free-memory-block indicator
@var: (unit32_t)               size -- capacity of the free memory block (in bytes)
@var: (struct alloc_block_s *) next -- pointer to the next free memory block
*/
typedef struct alloc_block_s {
  uint32_t size;
  struct alloc_block_s *next;
} alloc_block_t;

/*
Desc: Allocator data structure
@var: (alloc_block_t *) first block -- pointer to the first free memory block 
*/
typedef struct {
  alloc_block_t *first_block;
} alloc_t;

/********************
*  Initialization   *
********************/

// Initialize the first free-memory-block indicator, and set up the pointer in the allocator
void flex_cluster_alloc_init(alloc_t *alloc, void *base, const uint32_t size);

/***************
*  Allocation  *
***************/

// Memory alllocation with programmer-specified allocator
void *domain_malloc(alloc_t *alloc, const uint32_t size);

// Memory allocation with default l1 heap allocator
void *flex_l1_malloc(const uint32_t size);
void *flex_hbm_malloc(const uint32_t size);


/******************
*  De-allocation  *
******************/

// De-allocation with programmer-specified allocator
void domain_free(alloc_t *alloc, void *const ptr);

// De-allocation with default l1 heap allocator
void flex_l1_free(void *const ptr);
void flex_hbm_free(void *const ptr);

/*********************
*  Helper functions  *
*********************/

// Return the address of the default l1 heap allocator
alloc_t *flex_get_allocator_l1();
alloc_t *flex_get_allocator_hbm();

// [debug] print all free-memory-blocks in l1 heap
void flex_dump_heap();

void flex_alloc_init();

#endif
