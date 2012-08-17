/**
 * Defines the partitioning interface.
 *
 *  Created on: 2011-12-29
 *  Author: Abdullah Gharaibeh
 */
#ifndef TOTEM_PARTITION_H
#define TOTEM_PARTITION_H

// totem includes
#include "totem_graph.h"
#include "totem_grooves.h"

/**
 * Log (base 2) of the maximum number of partitions. Practically, it specifies
 * the number of bits allocated for the partition identifier when encoded in the
 * vertex identifier in a partition's edges array
 * TODO(abdullah): change the macros to constant variables and inline functions
 */
#define MAX_LOG_PARTITION_COUNT  2

/**
 * Maximum number of partitions supported per graph
 */
#define MAX_PARTITION_COUNT      (1 << (MAX_LOG_PARTITION_COUNT))

/**
 * Log (base 2) of the maximum number of vertices in a partition.
 */
#define MAX_LOG_VERTEX_COUNT     ((sizeof(vid_t) * 8) - MAX_LOG_PARTITION_COUNT)

/**
 * A mask used to identify the vertex id bits by clearing out the partition id
 * bits which are assumed to be in the higher order bits
 */
#define VERTEX_ID_MASK           (((vid_t)-1) >> MAX_LOG_PARTITION_COUNT)

/**
 * Decodes the partition id, which are placed in the higher order bits
 */
#define GET_PARTITION_ID(_vid)   ((_vid) >> (MAX_LOG_VERTEX_COUNT))

/**
 * Decodes the vertex id, which are placed in the lower order bits
 */
#define GET_VERTEX_ID(_vid)      ((_vid) & VERTEX_ID_MASK)

/**
 * Returns a new vertex id which encodes the correponding partition id in the
 * higher order bits.
 */
#define SET_PARTITION_ID(_vid, _pid) \
  ((_vid) | (((vid_t)(_pid)) << MAX_LOG_VERTEX_COUNT))

/**
 * A graph partition type based on adjacency list representation. The vertex ids
 * in the edges list have the partition id encoded in the most significant bits.
 * This allows for a vertex to have a neighbor in another partition.
 *
 * The outbox and inbox parameters represent the communication stubs of Grooves.
 * Outbox is a list of tables where each table stores the state to be
 * communicated with another partition; therefore, the length of this array is
 * partition_count - 1, where partition_count is the number of partitions
 * in the partition set this partition belongs to.
 * Similarly, inbox is a list of tables where each table stores the state
 * communicated by each remote partition to this partition.
 */
typedef struct partition_s {
  uint32_t             id;           /**< partition id, it is equal to the
                                        partition's index in a partition_set */
  graph_t              subgraph;     /**< the subgraph this partition
                                        represents */
  vid_t*               map;          /**< maps the a vertex id in a partition
                                         to its original id in the graph. used
                                         when aggregating the final results */
  grooves_box_table_t* outbox;       /**< table of messages to be sent to
                                        remote nbrs */
  grooves_box_table_t* inbox;        /**< table of messages received from
                                        other partitions */
  processor_t          processor;    /**< the processor this partition will be
                                        processed on. */
  void*                algo_state;   /**< algorithm-specific state (allocated
                                        and finalized by algorithm-specific
                                        callback functions). */
  grooves_box_table_t* outbox_d;     /**< a mirror of the outbox table on the
                                        GPU for GPU-resident partitions. Note
                                        that this just maintains the references
                                        to the gpu-allocated state (i.e., the
                                        state itself is not mirrored on both
                                        the host and the GPU). This is needed
                                        to allow easy management of the state,
                                        which sometimes is managed by the host
                                        (e.g., to initiate transfers where
                                        in/outbox references should be used)
                                        and sometimes by the GPU (e.g., during
                                        actual processing where in/outbox_d
                                        should be used) */
  grooves_box_table_t* inbox_d;      /**< a mirror of the inbox table */
  cudaStream_t         streams[2];   /**< used in GPU-resident partitions to
                                        enable overlapped operations (e.g.,
                                        communication among all devices or with
                                        computation). The first stream is used
                                        to launch communication operations,
                                        while the second one is used to launch
                                        kernel calls (computation) */
  cudaEvent_t         event_start;   /**< used to record the start kernel
                                        execution event for GPU partitions. to
                                        measure the kernel's execution time */
  cudaEvent_t         event_end;     /**< used to record the end kernel
                                        execution event for GPU partitions.
                                        Together with event_start, it is used to
                                        measure the kernel's execution time */
  eid_t            rmt_edge_count;   /**< the number of remote edges (edges
                                        that start in this partition and ends
                                        in another one) */
  vid_t            rmt_vertex_count; /**< the number of remote vertices
                                        (vertices that are the destination
                                        of edges that start in this partition
                                        and end in another one) */
} partition_t;

/**
 * Defines a set of partitions. Note that the vertex id in the original graph
 * is mapped to a new id in its corresponding partition such that the vertex
 * ids of a partition are contiguous from 0 to partition->vertex_count - 1.
 */
typedef struct partition_set_s {
  graph_t*     graph;           /**< the graph this partition set belongs to */
  bool         weighted;        /**< indicates if edges have weights. */
  partition_t* partitions;      /**< the array of partitions */
  int          partition_count; /**< number of partitions in the set */
  size_t       msg_size;        /**< the size of a communication message */
  vid_t*       id_in_partition; /**< maps a vertex id in the graph to its
                                   new id in its designated partition */
} partition_set_t;

/**
 * Computes the modularity of the specified graph partitioning configuration.
 * Modularity measures the fraction of edges that falls within a given partition
 * minus the expected number of such edges if the graph was randomly generated.
 * More formally, Q = (\sum_i=1^m e_{ii} - a_i^2, where m is the number of
 * partitions, e_{ii} is the fraction of edges (from the entire network) that
 * have their sources and destination inside the same partition, and a_i^2 is
 * the expected fraction of edges if the graph was randomly generated. This
 * implementation follows the specification from Newman-Girvan [Physical Review
 * E 69, 026113, 2004].
 *
 * @param[in] graph The graph data structure with the network information
 * @param[in] partition_set the partitioning information
 * @param[out] modularity the result modularity value for the given partition
 *             \in [0,1].
 */
error_t partition_modularity(graph_t* graph, partition_set_t* partition_set,
                             double* modularity);

/**
 * Split the graph randomly into the specified number of partitions with the
 * specified fractional distribution for each partition.
 *
 * @param[in] graph the input graph
 * @param[in] partition_count the number of partitions the vertices should be
 *                            assigned to
 * @param[in] partition_fraction an array with the fraction of the graph to be
 *                               assigned for each of the partitions. If set,
 *                               the sum of this array should be exactly 1. if
 *                               NULL, the partitions will be assigned equal
 *                               fractions.
 * @param[out] partition_labels an array with a partition id for each vertex as
 *                              identified by the array position. It is set to
 *                              NULL in case of failure.
 * @return SUCCESS if the partitions are assigned, FAILURE otherwise.
 */
error_t partition_random(graph_t* graph, int partition_count, 
                         double* partition_fraction, vid_t** partition_labels);

/**
 * Split the graph after sorting the vertices by edge degree into the specified 
 * number of partitions with the specified fractional distribution for each 
 * partition.
 *
 * @param[in] graph the input graph
 * @param[in] partition_count the number of partitions the vertices should be
 *                            assigned to
 * @param[in] partition_fraction an array with the fraction of the graph to be
 *                               assigned for each of the partitions. If set,
 *                               the sum of this array should be exactly 1. if
 *                               NULL, the partitions will be assigned equal
 *                               fractions.
 * @param[out] partition_labels an array with a partition id for each vertex as
 *                              identified by the array position. It is set to
 *                              NULL in case of failure.
 * @return SUCCESS if the partitions are assigned, FAILURE otherwise.
 */
error_t partition_by_asc_sorted_degree(graph_t* graph, int partition_count,
                                       double* partition_fraction, 
                                       vid_t** partition_labels);
error_t partition_by_dsc_sorted_degree(graph_t* graph, int partition_count,
                                       double* partition_fraction,
                                       vid_t** partition_labels);

/**
 * The following defines the signature of a partitioning algorithm function. The
 * PARTITION_FUNC array offers a simple way to invoke a partitioning algorithm
 * given a partition_algorithm_t (enumeration defined in totem.h) variable. Note
 * that the order of the functions here must be the same as their corresponding 
 * entry in the enumeration.
 */
typedef error_t(*partition_func_t)(graph_t*, int, double*, vid_t**);
PRIVATE const partition_func_t PARTITION_FUNC[] = {
  partition_random,
  partition_by_asc_sorted_degree,
  partition_by_dsc_sorted_degree
};

/**
 * Creates the a partition set based on the vertex to partition assignment
 * specified in the lables array
 *
 * @param[in] graph the input graph
 * @param[in] partition_labels an array with a partition id for each vertex as
 *                   identified by the array position
 * @param[in] partition_count the number of partitions
 * @param[in] msg_size  the size of a communication element
 * @param[out] partition_set the set of resulting graphs
 * @return SUCCESS if the partitions are assigned, FAILURE otherwise.
 */
error_t partition_set_initialize(graph_t* graph, vid_t* partition_labels,
                                 processor_t* partition_processor,
                                 int partition_count, size_t msg_size,
                                 partition_set_t** partition_set);

/**
 * De-allocates a partition set object
 * @param[in] partition_set a reference to partition set type to be de-allocated
 * @return generic success or failure
 */
error_t partition_set_finalize(partition_set_t* partition_set);

#endif  // TOTEM_PARTITION_H
