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
 * Defines the basic data type that is used as communication stubs between
 * partitions. In particular, it maintains the state of remote neighbors to a
 * partition. Vertex ids (including the partition id) of the remote neighbors
 * represent the keys in the hash table. Since the hash table allows storing
 * only integer values, the corresponding value of each key in the hash table
 * is used as an index in the values array. This design enables maintaining
 * any type as a state (e.g., float for ranks in PageRank), and allows for
 * atomic update of the value by two different threads.
 */
typedef struct grooves_box_table_s {
  id_t*         rmt_nbrs; /**< a table of the remote neighbors' ids. */
  void*         values;   /**< the actual state of each vertex */
  uint32_t      count;    /**< number of neighbors */
} grooves_box_table_t;

/**
 * Initializes the grooves layer: constructs the communication stubs for
 * each partition on its corresponding processor.
 */
error_t grooves_initialize(partition_set_t* pset);

/**
 * Clears state allocated by groovs layer.
 */
error_t grooves_finalize(partition_set_t* pset);

/**
 * Launches outbox-->inbox communications across partitions. This is done by
 * launching asynchronous transfers of the values array of a partition's outbox
 * to its corresponding inbox in a target partition.
 */
error_t grooves_launch_communications(partition_set_t* pset);

/**
 * Blocks until all data transfers initiated by grooves_launch_communications
 * have finished.
 */
error_t grooves_synchronize(partition_set_t* pset);

#endif  // TOTEM_GROOVES_H
