/**
 * Defines the high-level interface of Totem framework. It offers an interface 
 * to the user of a totem-based algorithm to initialize/finalize the framework's
 * algorithm-agnostic state, and query profiling data recorded during a previous
 * execution. This is basically a wrapper to the engine interface.
 *
 *  Created on: 2012-07-03
 *  Author: Abdullah Gharaibeh
 */

#ifndef TOTEM_H
#define TOTEM_H

#include "totem_comdef.h"
#include "totem_graph.h"
#include "totem_attributes.h"
#include "totem_partition.h"

/**
 * Defines the set of timers measured internally by Totem
 */
typedef struct totem_timing_s {
  double engine_init;  /**< Engine initialization  */
  double engine_par;   /**< Partitioning (included in engine_init) */
  double alg_exec;     /**< Algorithm execution alg_(comp + comm) */
  double alg_comp;     /**< Compute phase */
  double alg_comm;     /**< Communication phase (inlcudes scatter/gather) */
  double alg_aggr;     /**< Final result aggregation */
  double alg_scatter;  /**< The scatter step in communication (push mode) */
  double alg_gather;   /**< The gather step in communication (pull mode) */
  double alg_gpu_comp; /**< Computation time of the slowest GPU
                            (included in alg_comp) */
  double alg_gpu_total_comp; /**< Sum of computation time of all GPUs */
  double alg_cpu_comp; /**< CPU computation (included in alg_comp) */
  double alg_init;     /**< Algorithm initialization */
  double alg_finalize; /**< Algorithm finalization */
} totem_timing_t;


/**
 * Initializes the state required for hybrid CPU-GPU processing. It creates a
 * set of partitions equal to the number of GPUs plus one for the CPU. Note that
 * this function initializes algorithm-agnostic state only. This function
 * corresponds to Kernel 1 (the graph construction kernel) of the Graph500 
 * benchmark specification.
 * @param[in] graph  the input graph
 * @param[in] attr   attributes to setup the engine
 */
error_t totem_init(graph_t* graph, totem_attr_t* attr);

/**
 * Clears the state allocated by the engine via the totem_init function.
 */
void totem_finalize();

/**
 * Returns a reference to the set of timers measured internally by the engine
 */
const totem_timing_t* totem_timing();

/**
 * Resets the timers that measure the internals of the engine
 */
void totem_timing_reset();

/**
 * Returns the number of partitions
 */
uint32_t totem_partition_count();

/**
 * Returns the number of vertices in a specific partition
 */
vid_t totem_par_vertex_count(uint32_t pid);

/**
 * Returns the number of edges in a specific partition
 */
eid_t totem_par_edge_count(uint32_t pid);

/**
 * Returns the number of remote vertices in a specific partition
 */
vid_t totem_par_rmt_vertex_count(uint32_t pid);

/**
 * Returns the number of remote edges in a specific partition
 */
eid_t totem_par_rmt_edge_count(uint32_t pid);

#endif  // TOTEM_H
