/**
 * Defines the interface that drives the totem framework. It offers the main
 * interface the developer is supposed to use to implement graph algorithms on 
 * multi-GPU and CPU platform.
 *
 * In summary, an algorithm must offer a number of callback functions that 
 * configure the engine. The following is a pseudocode of an algorithm that 
 * uses this interface:
 *
 *    T* output_g; // allocated and filled by algo_aggr callback function
 *    graph_algo(graph_t* graph, partition_algorithm_t par_algo, T** output) {
 *      engine_config_t config = {
 *        graph,
 *        par_algo, 
 *        sizeof(T), 
 *        algo_ss_kernel,
 *        algo_par_kernel, 
 *        algo_par_scatter,
 *        algo_par_init,
 *        algo_par_finalize,
 *        algo_par_aggr,
 *      };
 *      engine_init(&config);
 *      engine_execute();
 *      *output = output_g;
 *     }
 *
 *  Created on: 2012-02-02
 *  Author: Abdullah Gharaibeh
 */

#ifndef TOTEM_ENGINE_CUH
#define TOTEM_ENGINE_CUH

#include "totem_comkernel.cuh"
#include "totem_partition.h"

/**
 * Callback function at the beginning of a superstep. Note that this callback 
 * is invoked once and is NOT per partition. It enables setting-up an 
 * algorithm-specific global state at the beginning of each superstep
 */
typedef void(*engine_ss_kernel_func_t)();

/**
 * Callback function on a partition at a superstep's compute phase. For GPU
 * partitions, this function is supposed to asynchronously invoke the GPU kernel
 * using the compute "stream" available for each partition. Note that the client
 * is responsible to check if the partition is CPU or GPU resident, and invoke
 * the processor specific kernel accordingly. The engine, however, guarantees 
 * that for GPU partitions the correct device is set (i.e., kernel invocations
 * is guaranteed to be launched on the correct GPU)
 */
typedef void(*engine_par_kernel_func_t)(partition_t*);

/**
 * Callback function to scatter inbox data to partition-specific state. 
 * This callback function allows the client to call one of the scatter 
 * functions (defined in totem_engine.cuh).
 * The purpose of this callback is to enable algorithm-specific distribution
 * of the messages received at the inbox table into the algorithm's state 
 * variables.
 * For example, PageRank has a "rank" array that represents the rank of each
 * vertex. The rank of each vertex is computed by summing the ranks of the 
 * neighboring vertices. In each superstep, the ranks of remote neighbors of 
 * a vertex are communicated into the inbox table of the partition. To this end,
 * a scatter function simply aggregates the "rank" of the remote neighbor with
 * the rank of the destination vertex (the aggregation is "add" in this case).
 */
typedef void(*engine_par_scatter_func_t)(partition_t*);

/**
 * Callback function on a partition to enable algorithm-specific per-partition
 * state initialization.
 */
typedef void(*engine_par_init_func_t)(partition_t*);

/**
 * Callback function on a partition to enable algorithm-specific per-partition
 * state finalization.
 */
typedef void(*engine_par_finalize_func_t)(partition_t*);

/**
 * Callback function on a partition to enable aggregating the final result. 
 * This is called after receiving the termination signal from all partitions.
 */
typedef void(*engine_par_aggr_func_t)(partition_t*);

/**
 * Engine configuration type. Algorithms use an instance of this type to
 * configure the execution engine.
 * TODO(abdullah) regarding the cpu_par_share member, ideally, we would like
 * to split the graph among processors with different shares (e.g., imagine 
 * a system with GPUs with different memory capacities).
 */
typedef struct engine_config_s {
  graph_t*                     graph;            /**< the input graph */
  platform_t                   platform;         /**< the execution platform */
  partition_algorithm_t        par_algo;         /**< partitioning algorithm */
  float                        cpu_par_share;    /**< the percentage of edges 
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
  size_t                       msg_size;         /**< comm. element size */
  engine_ss_kernel_func_t      ss_kernel_func;   /**< per superstep init func */
  engine_par_kernel_func_t     par_kernel_func;  /**< per par. comp. func */
  engine_par_scatter_func_t    par_scatter_func; /**< per par. scatter func */
  engine_par_init_func_t       par_init_func;    /**< per par. init function */
  engine_par_finalize_func_t   par_finalize_func;/**< per par. finalize func */
  engine_par_aggr_func_t       par_aggr_func;    /**< per partition results 
                                                      aggregation func */
} engine_config_t;

/**
 * Default configuration
 */
#define ENGINE_DEFAULT_CONFIG {NULL, PLATFORM_ALL, PAR_RANDOM, 0, \
      sizeof(int), NULL, NULL, NULL, NULL, NULL, NULL}

/**
 * Returns the address of a neighbor's state. If remote, it returns a reference
 * to its state in the outbox table. If local, it returns a reference to its
 * state in the array pstate
 */
#define ENGINE_FETCH_DST(_pid, _nbr, _outbox, _pstate, _pcount, _dst, _type) \
  do {                                                                  \
    int nbr_pid = GET_PARTITION_ID((_nbr));                             \
    if (nbr_pid != (_pid)) {                                            \
      int box_id = GROOVES_BOX_INDEX(nbr_pid, (_pid), (_pcount));       \
      _type * values = (_type *)(_outbox)[box_id].values;               \
      (_dst) = &values[GET_VERTEX_ID((_nbr))];                          \
    } else {                                                            \
      (_dst) = &(_pstate)[GET_VERTEX_ID((_nbr))];                       \
    }                                                                   \
  } while(0)

/**
 * Sets up the state required for hybrid CPU-GPU processing. It creats a set
 * of partitions equal to the number of GPUs plus one on the CPU.
 * @param[in] config   attributes to configure the engine
 */
error_t engine_init(engine_config_t* config);

/**
 * Performs the computation-->communication-->synchronization execution cycle.
 * It returns only after all partitions have sent a "finished" signal in the
 * same superstep via engine_report_finished.
 */
error_t engine_execute();

/**
 * Allows a partition to report that it has finished computing. Note that if all
 * partitions reported finish status, then the engine terminates
 */
void engine_report_finished(uint32_t pid);

/**
 * Returns the number of partitions
 */
uint32_t engine_partition_count();

/**
 * Returns the current superstep number
 */
uint32_t engine_superstep();

/**
 * Returns the total number of vertices in the graph
 */
uint32_t engine_vertex_count();

/**
 * Returns the total number of edges in the graph
 */
uint32_t engine_edge_count();

/**
 * Returns the number of vertices of the largest GPU partition
 */
uint64_t engine_largest_gpu_partition();

/**
 * Returns a reference to a map that maps a vertex id to its new id in the 
 * corresponding partition
 */
id_t* engine_vertex_id_in_partition();

/**
 * Returns a vertex's new id in the corresponding partition
 */
id_t engine_vertex_id_in_partition(id_t);

/**
 * Returns the total time spent on initializing the state (includes 
 * partitioning and building the state)
 */
double engine_time_initialization();

/**
 * Returns the time spent on partitioning the graph
 */
double engine_time_partitioning();

/**
 * Returns the time spent on executing all the supersteps
 */
double engine_time_execution();

/**
 * Returns the total time spent on the computation phase
 */
double engine_time_computation();
double engine_time_gpu_computation();

/**
 * Returns the total time spent on the communication phase
 */
double engine_time_communication();

/**
 * Scatters the messages in the inbox table to the corresponding vertices. The 
 * assumption is that each vertex in the partition has a position in the array 
 * "dst". The message to a vertex in the inbox is added to the vertex's state 
 * in dst.
 * @param[in] pid the input partition
 * @param[in] dst the destination array where the messages will be sent
 */
template<typename T>
void engine_scatter_inbox_add(uint32_t pid, T* dst);

/**
 * Scatters the messages in the inbox table using min reduction.
 * @param[in] pid the input partition
 * @param[in] dst the destination array where the messages will be sent
 */
template<typename T>
void engine_scatter_inbox_min(uint32_t pid, T* dst);

/**
 * Scatters the messages in the inbox table using max reduction.
 * @param[in] pid the input partition
 * @param[in] dst the destination array where the messages will be sent
 */
template<typename T>
void engine_scatter_inbox_max(uint32_t pid, T* dst);

/**
 * Sets all entries in the outbox's values array to value
 * @param[in] pid the input partition
 * @param[in] value value to be set
 */
template<typename T>
void engine_set_outbox(uint32_t pid, T value);

// This header file includes implementations of the templatized functions 
// defined in this interface. Must be placed at the bottom to solve some 
// dependencies.
#include "totem_engine_internal.cuh"

#endif  // TOTEM_ENGINE_CUH
