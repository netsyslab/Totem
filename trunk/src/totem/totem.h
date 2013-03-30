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

/**
 * Execution platform options
 */
typedef enum {
  PLATFORM_CPU,       // execute on the CPU only
  PLATFORM_GPU,       // execute on GPUs only
  PLATFORM_HYBRID,    // execute on the CPU and the GPUs
  PLATFORM_MAX        // indicates the number of platform options
} platform_t;

/**
 * Partitioning algorithm type
 */
typedef enum {
  PAR_RANDOM = 0,
  PAR_SORTED_ASC,
  PAR_SORTED_DSC,
  PAR_MAX
} partition_algorithm_t;

/**
 * Defines the attributes used to initialize a Totem
 */
typedef struct totem_attr_s {
  partition_algorithm_t par_algo;      /**< partitioning algorithm */
  platform_t            platform;      /**< the execution platform */
  uint32_t              gpu_count;     /**< number of GPUs to use  */
  bool                  mapped;        /**< whether the vertices array of GPU 
                                            partitions is allocated as memory
                                            mapped buffer on the host or on the
                                            GPU memory */
  float                 cpu_par_share; /**< the percentage of edges assigned
                                            to the CPU partition. Note that this
                                            value is relevant only in hybrid 
                                            platforms. The GPUs will be assigned
                                            equal shares after deducting the CPU
                                            share. If this is set to zero, then
                                            the graph is divided among all 
                                            processors equally. */
  size_t                push_msg_size; /**< push comm. message size in bits*/
  size_t                pull_msg_size; /**< pull comm. message size in bits*/
} totem_attr_t;

// default attributes: hybrid (one GPU + CPU) platform, random 50-50 
// partitioning, no mapped memory, push message size is word and zero
// pull message size
#define TOTEM_DEFAULT_ATTR {PAR_RANDOM, PLATFORM_HYBRID, 1, false, 0.5,  \
      MSG_SIZE_WORD, MSG_SIZE_ZERO}

/**
 * Defines the set of timers measured internally by Totem
 */
typedef struct totem_timing_s {
  double engine_init;  /**< Engine initialization  */
  double engine_par;   /**< Partitioning (included in engine_init) */
  double alg_exec;     /**< Algorithm execution alg_(comp + comm + aggr) */
  double alg_comp;     /**< Compute phase */
  double alg_comm;     /**< Communication phase (inlcudes scatter/gather) */
  double alg_aggr;     /**< Final result aggregation */
  double alg_scatter;  /**< The scatter step in communication (push mode) */
  double alg_gather;   /**< The gather step in communication (pull mode) */
  double alg_gpu_comp; /**< GPU computation (included in alg_comp) */
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
