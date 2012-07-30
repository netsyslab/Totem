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
 * Execution platform options. The engine will create a partition per processor.
 * Note that if the system has one GPU only, then ENGINE_PLATFORM_GPU and
 * ENGINE_PLATFORM_MULTI_GPU will be the same, as well as ENGINE_PLATFORM_HYBRID
 * and ENGINE_PLATFORM_ALL.
 */
typedef enum {
  PLATFORM_CPU,       // execute on the CPU only
  PLATFORM_GPU,       // execute on one GPU only
  PLATFORM_MULTI_GPU, // execute on all available GPUs
  PLATFORM_HYBRID,    // execute on the CPU and one GPU
  PLATFORM_ALL,       // execute on all processors (CPU and all GPUs)
  PLATFORM_MAX        // indicates number of platform options
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
  float                 cpu_par_share; /**< the percentage of edges
                                            assigned to the CPU
                                            partition. Note that the
                                            value of this member is
                                            relevant only in platforms
                                            that include a CPU with at
                                            least one GPU. The GPUs
                                            will be assigned equal
                                            shares after deducting
                                            the CPU share. If this
                                            is assigned to zero, then
                                            the graph is divided among
                                            all processors equally. */
  size_t                msg_size;      /**< comm. element size in bytes*/
} totem_attr_t;

// default attributes
#define TOTEM_DEFAULT_ATTR {PAR_RANDOM, PLATFORM_ALL, 0.5, sizeof(int)}

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
 * Returns the total time spent on initializing the state (includes
 * partitioning and building the state)
 * TODO(abdullah): instead of having many function to query the internal
 * timers, we can have just one function that returns a copy of an instance
 * of timer logs data type (e.g., totem_timers_t)
 */
double totem_time_initialization();

/**
 * Returns the time spent on partitioning the graph
 */
double totem_time_partitioning();

/**
 * Returns the overall time spent on executing all the supersteps
 */
double totem_time_execution();

/**
 * Returns the total time spent on the computation phase
 */
double totem_time_computation();

/**
 * Returns the total time spent computing on the GPU
 */
double totem_time_gpu_computation();

/**
 * Returns the total time spent on the communication phase
 */
double totem_time_communication();

/**
 * Returns the time spent on scattering the data received in the inbox buffers
 * to the local state of the destination vertices during the communication phase
 */
double totem_time_scatter();

/**
 * Returns the time spent on aggregating the final result
 */
double totem_time_aggregation();

/**
 * Returns the number of partitions
 */
uint32_t totem_partition_count();

/**
 * Returns the number of vertices in a specific partition
 */
uint64_t totem_par_vertex_count(uint32_t pid);

/**
 * Returns the number of edges in a specific partition
 */
uint64_t totem_par_edge_count(uint32_t pid);

/**
 * Returns the number of remote vertices in a specific partition
 */
uint64_t totem_par_rmt_vertex_count(uint32_t pid);

/**
 * Returns the number of remote edges in a specific partition
 */
uint64_t totem_par_rmt_edge_count(uint32_t pid);

#endif  // TOTEM_H
