/**
 * Defines the Grooves communication interface: a message passing substrate
 * for the Totem graph-processing framework. The layer is tightly coupled with
 * the partition data types and interface, which resulted in a cyclic dependency
 * between the two modules.
 *
 *  Created on: 2011-01-05
 *  Author: Abdullah Gharaibeh
 */
#ifndef TOTEM_GROOVES_H
#define TOTEM_GROOVES_H

// totem includes
#include "totem_hash_table.h"

// forward declaration to break the cyclic dependency between the grooves and
// the partition modules
typedef struct partition_set_s partition_set_t;

/**
 * The communication direction. "PUSH" configures the engine to push data from 
 * the source vertex of a boundary edge to the destination vertex. This is used,
 * for example, in BFS to allow a vertex to set the cost of a neighbouring 
 * remote vertex. "PULL" configures the engine to allow a boundary edge's source
 * vertex to pull the state of the corresponding destination vertex. This is
 * used, for example, in backward propagation procedures in centrality 
 * algorithms. 
 */
typedef enum {
  GROOVES_PUSH = 0,
  GROOVES_PULL
} grooves_direction_t;

/**
 * Defines the basic data type that is used as communication stubs between
 * partitions. It maintains the state of remote neighbors to a partition.
 * TODO(abdullah): change the name to grooves_ghost_vertices_t
 */
typedef struct grooves_box_table_s {
  vid_t* rmt_nbrs;      /**< remote neighbors' ids. */
  void*  push_values;   /**< data pushed to remote vertices */
  void*  pull_values;   /**< data pulled from remote vertices */
  vid_t  count;         /**< number of remote neighbors */
} grooves_box_table_t;

/**
 * Initializes the grooves layer: constructs the communication stubs for
 * each partition on its corresponding processor.
 * @param[in] pset the partition set to operate on
 * @return generic success or failure
 */
error_t grooves_initialize(partition_set_t* pset);

/**
 * Clears state allocated by groovs layer.
 */
error_t grooves_finalize(partition_set_t* pset);

/**
 * Launches communication between partitions. Depending on the direction, data 
 * is either pushed from the source (local) to the destination (remote), or 
 * pulled from the source (remote) to the destination (local). This is done by 
 * launching asynchronous transfers of the values arrays of a partition's inbox
 * and outbox buffers.
 * @param[in] pset the partition set to operate on
 * @param[in] direction the direction of communication, PULL or PUSH
 * @return generic success or failure
 */
error_t grooves_launch_communications(partition_set_t* pset, 
                                      grooves_direction_t direction);

/**
 * Blocks until all data transfers initiated by grooves_launch_communications
 * have finished.
 * @param[in] pset the partition set to operate on
 * @return generic success or failure
 */
error_t grooves_synchronize(partition_set_t* pset);

#endif  // TOTEM_GROOVES_H
