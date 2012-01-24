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
 */
#define MAX_LOG_PARTITION_COUNT  2

/**
 * Maximum number of partitions supported per graph
 */
#define MAX_PARTITION_COUNT (1 << (MAX_LOG_PARTITION_COUNT))

/**
 * Log (base 2) of the maximum number of vertices in a partition.
 */
#define MAX_LOG_VERTEX_COUNT     ((sizeof(id_t) * 8) - MAX_LOG_PARTITION_COUNT)

/**
 * A mask used to identify the vertex id bits by clearing out the partition id 
 * bits which are assumed to be in the higher order bits
 */
#define VERTEX_ID_MASK           (((id_t)-1) >> MAX_LOG_PARTITION_COUNT)

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
  ((_vid) | ((_pid) << MAX_LOG_VERTEX_COUNT))

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
  graph_t              subgraph;   /**< the subgraph this partition 
                                      represents */
  grooves_box_table_t* outbox;     /**< table of messages to be sent to 
                                      remote nbrs */
  grooves_box_table_t* inbox;      /**< table of messages received from 
                                      other partitions */
  processor_t          processor;  /**< the processor this partition will be
                                      processed on. */
} partition_t;

/**
 * Defines a set of partitions. Note that the vertex id in the original graph 
 * is mapped to a new id in its corresponding partition such that the vertex
 * ids of a partition are contiguous from 0 to partition->vertex_count - 1.
 */
typedef struct partition_set_s {
  graph_t*     graph;           /**< the graph this partition set belongs to */
  bool         weighted;        /**< indicates if edges have weights. */
  partition_t* partitions;      /**< the partitions list */
  int          partition_count; /**< number of partitions in the set */
  size_t       value_size;      /**< the size of a communication element */
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
 * Split the graph into the specified number of partitions by randomly assigning
 * vertices to each partition.
 *
 * @param[in] graph the input graph
 * @param[in] partition_count the number of partitions the vertices should be 
 *                            assigned to
 * @param[in] seed a number to seed the pseudorandom number generator
 * @param[out] partition_labels an array with a partition id for each vertex as
 *                              identified by the array position. It is set to
 *                              NULL in case of failure.
 * @return SUCCESS if the partitions are assigned, FAILURE otherwise.
 */
error_t partition_random(graph_t* graph, int partition_count,
                         unsigned int seed, id_t** partition_labels);

/**
 * Creates the a partition set based on the vertex to partition assignment 
 * specified in the lables array
 *
 * @param[in] graph the input graph
 * @param[in] partition_labels an array with a partition id for each vertex as
 *                   identified by the array position
 * @param[in] partition_count the number of partitions
 * @param[in] value_size  the size of a communication element
 * @param[out] partition_set the set of resulting graphs
 * @return SUCCESS if the partitions are assigned, FAILURE otherwise.
 */
error_t partition_set_initialize(graph_t* graph, id_t* partition_labels, 
                                 processor_t* partition_processor,
                                 int partition_count, 
                                 size_t value_size,
                                 partition_set_t** partition_set);

/**
 * De-allocates a partition set object
 * @param[in] partition_set a reference to partition set type to be de-allocated
 * @return generic success or failure
 */
error_t partition_set_finalize(partition_set_t* partition_set);

#endif  // TOTEM_PARTITION_H
