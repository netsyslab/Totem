/**
 * This is an internal header file that is included only by the engine itself.
 * This header file helps separating some internal functionality of the engine
 * that must be placed in a .h file (e.g., templatized interfaces).
 *
 * Currently it includes inbox scatter/reduce functions. These functions allow
 * for distributing the data received at the inbox table into the algorithm's
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
  uint32_t         superstep;
  engine_config_t  config;
  bool*            finished;
  uint64_t         largest_gpu_par;
  double           time_init;
  double           time_par;
  double           time_exec;
  double           time_comm;
  double           time_comp;
  double           time_gpu_comp;
  int              partition_count;
  uint64_t         vertex_count[MAX_PARTITION_COUNT];
  uint64_t         edge_count[MAX_PARTITION_COUNT];
  uint64_t         rmt_vertex_count[MAX_PARTITION_COUNT];
  uint64_t         rmt_edge_count[MAX_PARTITION_COUNT];
} engine_context_t;

extern engine_context_t context;

/**
 * Fetches the vertex id and value of an entry in a grooves box table
 */
#define _FETCH_ENTRY(_box, _index, _vid, _value)    \
  do {                                              \
    T* values = (T*)(_box)->values;                 \
    _value = values[(_index)];                      \
    _vid = (_box)->rmt_nbrs[(_index)];              \
  } while(0)

/**
 * The following macros encapsulate processor-agnostic element
 * reduction operations
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
  if (index >= inbox.count) return;
  _REDUCE_ENTRY_ADD(&inbox, index, dst);
}

template<typename T>
__global__ void scatter_min(grooves_box_table_t inbox, T* dst) {
  uint32_t index = THREAD_GLOBAL_INDEX;
  if (index >= inbox.count) return;
  _REDUCE_ENTRY_MIN(&inbox, index, dst);
}

template<typename T>
__global__ void scatter_max(grooves_box_table_t inbox, T* dst) {
  uint32_t index = THREAD_GLOBAL_INDEX;
  if (index >= inbox.count) return;
  _REDUCE_ENTRY_MAX(&inbox, index, dst);
}

template<typename T>
void engine_scatter_inbox_add(uint32_t pid, T* dst) {
  assert(pid < context.pset->partition_count);
  partition_t* par = &context.pset->partitions[pid];
  for (int bid = 0; bid < context.pset->partition_count - 1; bid++) {
    grooves_box_table_t* inbox = &par->inbox[bid];
    if (!inbox->count) continue;
    if (par->processor.type == PROCESSOR_GPU) {
      dim3 blocks, threads;
      KERNEL_CONFIGURE(inbox->count, blocks, threads);
      scatter_add<<<blocks, threads, 0, par->streams[1]>>>(*inbox, dst);
      CALL_CU_SAFE(cudaGetLastError());
    } else {
      assert(par->processor.type == PROCESSOR_CPU);
      #ifdef _OPENMP
      #pragma omp parallel for
      #endif
      for (int index = 0; index < inbox->count; index++) {
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
      KERNEL_CONFIGURE(inbox->count, blocks, threads);
      scatter_min<<<blocks, threads, 0, par->streams[1]>>>(*inbox, dst);
      CALL_CU_SAFE(cudaGetLastError());
    } else {
      assert(par->processor.type == PROCESSOR_CPU);
      #ifdef _OPENMP
      #pragma omp parallel for
      #endif
      for (int index = 0; index < inbox->count; index++) {
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
      KERNEL_CONFIGURE(inbox->count, blocks, threads);
      scatter_max<<<blocks, threads, 0, par->streams[1]>>>(*inbox, dst);
      CALL_CU_SAFE(cudaGetLastError());
    } else {
      assert(par->processor.type == PROCESSOR_CPU);
      #ifdef _OPENMP
      #pragma omp parallel for
      #endif
      for (int index = 0; index < inbox->count; index++) {
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
      memset_device<<<blocks, threads, 0, par->streams[1]>>>(values, value,
                                                             outbox->count);
      CALL_CU_SAFE(cudaGetLastError());
    } else {
      assert(par->processor.type == PROCESSOR_CPU);
      #ifdef _OPENMP
      #pragma omp parallel for
      #endif
      for (int i = 0; i < outbox->count; i++) values[i] = value;
    }
  }
}

inline uint32_t engine_partition_count() {
  return context.partition_count;
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

inline id_t* engine_vertex_id_in_partition() {
  return context.pset->id_in_partition;
}

inline id_t engine_vertex_id_in_partition(id_t v) {
  return context.pset->id_in_partition[v];
}

inline double engine_time_initialization() {
  return context.time_init;
}

inline double engine_time_partitioning() {
  return context.time_par;
}

inline double engine_time_execution() {
  return context.time_exec;
}

inline double engine_time_computation() {
  return context.time_comp;
}

inline double engine_time_gpu_computation() {
  return context.time_gpu_comp;
}

inline double engine_time_communication() {
  return context.time_comm;
}

inline double engine_par_rmt_vertex_count(uint32_t pid) {
  return context.rmt_vertex_count[pid];
}

inline double engine_par_rmt_edge_count(uint32_t pid) {
  return context.rmt_edge_count[pid];
}

inline double engine_par_vertex_count(uint32_t pid) {
  return context.vertex_count[pid];
}

inline double engine_par_edge_count(uint32_t pid) {
  return context.edge_count[pid];
}


#endif  // TOTEM_ENGINE_INTERNAL_CUH
