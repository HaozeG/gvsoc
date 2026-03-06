// flex_group_barrier_api.h
#ifndef FLEX_GROUP_BARRIER_API_H
#define FLEX_GROUP_BARRIER_API_H

#include <stdint.h>

/*****************************************
*  Grid Group Synchronization functions  *
*****************************************/

typedef struct GridSyncGroupInfo
{
    //General information
    uint32_t valid_grid;
    uint32_t grid_x_dim;
    uint32_t grid_y_dim;
    uint32_t grid_x_num;
    uint32_t grid_y_num;

    //Local information
    uint32_t this_grid_id;
    uint32_t this_grid_id_x;
    uint32_t this_grid_id_y;
    uint32_t this_grid_left_most;
    uint32_t this_grid_right_most;
    uint32_t this_grid_top_most;
    uint32_t this_grid_bottom_most;
    uint32_t this_grid_cluster_num;
    uint32_t this_grid_cluster_num_x;
    uint32_t this_grid_cluster_num_y;

    //Sync information
    uint8_t  wakeup_row_mask;
    uint8_t  wakeup_col_mask;
    uint32_t sync_x_cluster;
    uint32_t sync_y_cluster;
    volatile uint32_t * sync_x_point;
    volatile uint32_t * sync_x_piter;
    volatile uint32_t * sync_y_point;
    volatile uint32_t * sync_y_piter;
} GridSyncGroupInfo;

GridSyncGroupInfo grid_sync_group_init(uint32_t grid_x_dim, uint32_t grid_y_dim);
void grid_sync_group_barrier_xy(GridSyncGroupInfo * info);
void grid_sync_group_barrier_xy_polling(GridSyncGroupInfo * info);

#endif
