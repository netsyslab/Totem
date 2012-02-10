/**
 * This is an internal header file that is included only by the engine itself. 
 * This header file helps separating some internal functionality of the engine 
 * that must be placed in a .h file (e.g., templatized interfaces).
 * 
 * Currently it includes inbox scatter functions. These functions allow for 
 * distributing the data received at the inbox table into the algorithm's 
 * state variables. The assumption is that they will be invoked from inside the 
 * engine_scatter_func callback function.
 *
 * For example, PageRank has a "rank" array that represents the rank of each
 * vertex. The rank of each vertex is computed by summing the ranks of the 
 * neighboring vertices. In each superstep, the ranks of remote neighbors of 
 * a vertex are communicated into the inbox table of the partition. To this end,
 * a scatter function simply aggregates the "rank" of the remote neighbor with
 * the rank of the destination vertex (the aggregation is "add" in this case).
 */

#ifndef TOTEM_ENGINE_INTERNAL_CUH
#define TOTEM_ENGINE_INTERNAL_CUH

#include "totem_comkernel.cuh"
#include "totem_partition.h"

/**
 * defines the execution context of the engine
 */
typedef struct engine_context_s {
  bool             initialized;
  partition_set_t* pset;
  id_t*            par_labels;
  uint32_t         superstep;
  engine_config_t  config;
  bool*            finished;
  uint64_t         largest_gpu_par;
} engine_context_t;

extern engine_context_t context;

/**
 * Fetches the vertex id and value of an entry in a grooves box
 */
#define _FETCH_ENTRY(_box, _index, _vid, _value)    \
  {                                                 \
    uint64_t entry = (_box)->ht.entries[_index];    \
    id_t key = HT_GET_KEY(entry);                   \
    if (key == HT_KEY_EMPTY) break;                 \
    T* values = (T*)(_box)->values;                 \
    _value = values[HT_GET_VALUE(entry)];           \
    _vid = GET_VERTEX_ID(key);                      \
  }

/**
 * The following macros encapsulate processor-agnostic element
 * reduction operations.
 */
#define _REDUCE_ENTRY_ADD(_box, _index, _dst)         \
  do {                                                \
    id_t vid; T value;                                \
    _FETCH_ENTRY((_box), (_index), vid, value);       \
    (_dst)[vid] += value;                             \
  } while(0)

#define _REDUCE_ENTRY_MIN(_box, _index, _dst)                 \
  do {                                                        \
    id_t vid; T value;                                        \
    _FETCH_ENTRY((_box), _index, vid, value);                 \
    (_dst)[vid] = value < (_dst)[vid] ? value : (_dst)[vid];  \
  } while(0)

#define _REDUCE_ENTRY_MAX(_box, _index, _dst)                   \
  do {                                                          \
    id_t vid; T value;                                          \
    _FETCH_ENTRY((_box), _index, vid, value);                   \
    (_dst)[vid] = value > (_dst)[vid] ? value : (_dst)[vid];    \
  } while(0)

template<typename T> 
__global__ void scatter_add(grooves_box_table_t inbox, T* dst) {
  uint32_t index = THREAD_GLOBAL_INDEX;
  if (index >= inbox.ht.size) return;
  _REDUCE_ENTRY_ADD(&inbox, index, dst);
}

template<typename T> 
__global__ void scatter_min(grooves_box_table_t inbox, T* dst) {
  uint32_t index = THREAD_GLOBAL_INDEX;
  if (index >= inbox.ht.size) return;
  _REDUCE_ENTRY_MIN(&inbox, index, dst);
}

template<typename T> 
__global__ void scatter_max(grooves_box_table_t inbox, T* dst) {
  uint32_t index = THREAD_GLOBAL_INDEX;
  if (index >= inbox.ht.size) return;
  _REDUCE_ENTRY_MAX(&inbox, index, dst);
}

// TODO(abdullah): test parallelize the cpu scatter
template<typename T>
void engine_scatter_inbox_add(uint32_t pid, T* dst) {
  assert(pid < context.pset->partition_count);
  partition_t* par = &context.pset->partitions[pid];
  for (int bid = 0; bid < context.pset->partition_count - 1; bid++) {
    grooves_box_table_t* inbox = &par->inbox[bid];
    if (!inbox->count) continue;
    if (par->processor.type == PROCESSOR_GPU) {
      dim3 blocks, threads;
      KERNEL_CONFIGURE(inbox->ht.size, blocks, threads);
      scatter_add<<<blocks, threads, 1, par->streams[1]>>>(*inbox, dst);
      CALL_CU_SAFE(cudaGetLastError());
    } else {
      assert(par->processor.type == PROCESSOR_CPU);
      for (int index = 0; index < inbox->ht.size; index++) {
        _REDUCE_ENTRY_ADD(inbox, index, dst);
      }
    }
  }
}

template<typename T>
void engine_scatter_inbox_min(uint32_t pid, T* dst) {
  assert(pid < context.pset->partition_count);
  partition_t* par = &context.pset->partitions[pid];
  for (int bid = 0; bid < context.pset->partition_count - 1; bid++) {
    grooves_box_table_t* inbox = &par->inbox[bid];
    if (!inbox->count) continue;
    if (par->processor.type == PROCESSOR_GPU) {
      dim3 blocks, threads;
      KERNEL_CONFIGURE(inbox->ht.size, blocks, threads);
      scatter_min<<<blocks, threads, 1, par->streams[1]>>>(*inbox, dst);
      CALL_CU_SAFE(cudaGetLastError());
    } else {
      assert(par->processor.type == PROCESSOR_CPU);
      for (int index = 0; index < inbox->ht.size; index++) {
        _REDUCE_ENTRY_MIN(inbox, index, dst);
      }
    }
  }
}

template<typename T>
void engine_scatter_inbox_max(uint32_t pid, T* dst) {
  assert(pid < context.pset->partition_count);
  partition_t* par = &context.pset->partitions[pid];
  for (int bid = 0; bid < context.pset->partition_count - 1; bid++) {
    grooves_box_table_t* inbox = &par->inbox[bid];
    if (!inbox->count) continue;
    if (par->processor.type == PROCESSOR_GPU) {
      dim3 blocks, threads;
      KERNEL_CONFIGURE(inbox->ht.size, blocks, threads);
      scatter_max<<<blocks, threads, 1, par->streams[1]>>>(*inbox, dst);
      CALL_CU_SAFE(cudaGetLastError());
    } else {
      assert(par->processor.type == PROCESSOR_CPU);
      for (int index = 0; index < inbox->ht.size; index++) {
        _REDUCE_ENTRY_MAX(inbox, index, dst);
      }
    }
  }
}

template<typename T>
void engine_set_outbox(uint32_t pid, T value) {
  assert(pid < context.pset->partition_count);
  partition_t* par = &context.pset->partitions[pid];
  for (int bid = 0; bid < context.pset->partition_count - 1; bid++) {
    grooves_box_table_t* outbox =  &par->outbox[bid];
    if (!outbox->count) continue;
    T* values = (T*)outbox->values;
    if (par->processor.type == PROCESSOR_GPU) {
      dim3 blocks, threads;
      KERNEL_CONFIGURE(outbox->count, blocks, threads);
      memset_device<<<blocks, threads, 1, par->streams[1]>>>(values, value, 
                                                             outbox->count);
      CALL_CU_SAFE(cudaGetLastError());
    } else {
      assert(par->processor.type == PROCESSOR_CPU);
      for (int i = 0; i < outbox->count; i++) values[i] = value;
    }
  }
}

inline uint32_t engine_partition_count() {
  assert(context.pset);
  return context.pset->partition_count;
}

inline uint32_t engine_superstep() {
  assert(context.pset);
  return context.superstep;
}

inline uint32_t engine_vertex_count() {
  assert(context.pset);
  return context.pset->graph->vertex_count;
}

inline uint32_t engine_edge_count() {
  assert(context.pset);
  return context.pset->graph->edge_count;
}

inline uint64_t engine_largest_gpu_partition() {
  return context.largest_gpu_par;
}

inline void engine_report_finished(uint32_t pid) {
  assert(pid < context.pset->partition_count);
  context.finished[pid] = true;
}

#endif  // TOTEM_ENGINE_INTERNAL_CUH
