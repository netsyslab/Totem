/**
 * Defines structures related to the options or configurations related to
 * Totem. The intention of this file is to separate the configurations from
 * the runtime files.
 *
 *  Created on: 2014-17-07
 *  Author: Abdullah Gharaibeh
 */

#ifndef TOTEM_ATTRIBUTES_H
#define TOTEM_ATTRIBUTES_H

#include "totem_comdef.h"
#include "totem_graph.h"

// Partitioning algorithm type.
typedef enum {
  PAR_RANDOM = 0,
  PAR_SORTED_ASC,
  PAR_SORTED_DSC,
  PAR_MAX
} partition_algorithm_t;

// Callback function on a partition to enable algorithm-specific per-partition
// state allocation/finalization.
typedef struct partition_s partition_t;
typedef void(*totem_cb_func_t)(partition_t*);

// Execution platform options.
typedef enum {
  PLATFORM_CPU,     // Execute on CPU only.
  PLATFORM_GPU,     // Execute on GPUs only.
  PLATFORM_HYBRID,  // Execute on the CPU and the GPUs.
  PLATFORM_MAX      // Indicates the number of platform options.
} platform_t;

// Defines the attributes used to initialize a Totem.
typedef struct totem_attr_s {
  partition_algorithm_t par_algo;      // CPU-GPU partitioning strateg.
  platform_t            platform;      // The execution platform.
  uint32_t              gpu_count;     // Number of GPUs to use.
  gpu_graph_mem_t       gpu_graph_mem;  // Determines the type of memory used
                                        // to place the graph data structure of
                                        // GPU partitions.
  bool                  gpu_par_randomized;  // Whether the placement of
                                             // vertices across GPUs is random
                                             // or according to par_algo.
  bool                  sorted;  // Maps the vertex ids in sorted order by
                                 // edge degree during the partitioning phase.
  bool                  edge_sort_dsc;  // Makes the direction of edge sorting
                                        // descending instead of ascending.
  bool                  edge_sort_by_degree;  // Sorts the neighbours by degree
                                              // instead of by id.
  float                 cpu_par_share;  // The percentage of edges assigned
                                        // to the CPU partition. Note that this
                                        // value is relevant only in hybrid
                                        // platforms. The GPUs will be assigned
                                        // equal shares after deducting the CPU
                                        // share. If this is set to zero, then
                                        // the graph is divided among all
                                        // processors equally.
  size_t                push_msg_size;  // Push comm. message size in bits.
  size_t                pull_msg_size;  // Pull comm. message size in bits.
  totem_cb_func_t       alloc_func;     // Callback function to allocate
                                        // application-specific state.
  totem_cb_func_t       free_func;      // Callback function to free
                                        // application-specific state.
} totem_attr_t;

// Default attributes: hybrid (one GPU + CPU) platform, random 50-50
// partitioning, push message size is word and zero pull message size.
#define TOTEM_DEFAULT_ATTR {PAR_RANDOM, PLATFORM_HYBRID, 1, \
        GPU_GRAPH_MEM_DEVICE, false, false, false, false, 0.5, MSG_SIZE_WORD, \
        MSG_SIZE_ZERO, NULL, NULL}

#endif  // TOTEM_ATTRIBUTES_H
