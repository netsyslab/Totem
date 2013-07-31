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
#include "totem_mem.h"

/**
 * defines the execution context of the engine
 */
typedef struct engine_context_s {
  bool             initialized;
  graph_t*         graph;
  partition_set_t* pset;
  uint32_t         superstep;
  engine_config_t  config;
  totem_attr_t     attr;
  vid_t            largest_gpu_par;
  uint32_t         partition_count;
  totem_timing_t   timing;
  bool*            finished;
  vid_t            vertex_count[MAX_PARTITION_COUNT];
  eid_t            edge_count[MAX_PARTITION_COUNT];
  vid_t            rmt_vertex_count[MAX_PARTITION_COUNT];
  eid_t            rmt_edge_count[MAX_PARTITION_COUNT];
} engine_context_t;

/**
 * Default context values
 */
#define ENGINE_DEFAULT_CONTEXT {false, NULL, NULL, 0, ENGINE_DEFAULT_CONFIG, \
                                TOTEM_DEFAULT_ATTR, 0, 0};

extern engine_context_t context;

/**
 * Fetches the vertex id and value of an entry in a grooves box table
 */
#define _FETCH_ENTRY(_box, _index, _vid, _value)    \
  do {                                              \
    T* values = (T*)(_box)->push_values;            \
    _value = values[(_index)];                      \
    _vid = (_box)->rmt_nbrs[(_index)];              \
  } while(0)

/**
 * The following macros encapsulate processor-agnostic element
 * reduction operations
 */
#define _REDUCE_ENTRY_ADD(_box, _index, _dst)         \
  do {                                                \
    vid_t vid; T value;                                \
    _FETCH_ENTRY((_box), (_index), vid, value);       \
    (_dst)[vid] += value;                             \
  } while(0)

#define _REDUCE_ENTRY_MIN(_box, _index, _dst)                 \
  do {                                                        \
    vid_t vid; T value;                                        \
    _FETCH_ENTRY((_box), _index, vid, value);                 \
    (_dst)[vid] = value < (_dst)[vid] ? value : (_dst)[vid];  \
  } while(0)

#define _REDUCE_ENTRY_MAX(_box, _index, _dst)                   \
  do {                                                          \
    vid_t vid; T value;                                          \
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

// TODO (abdullah): There is a lot of repeated control code in the following 
// template functions. We could have a single private template function that 
// receives the operation as parameter and the other functions min, max, add, 
// etc would invoke it.
template<typename T>
void engine_scatter_inbox_add(uint32_t pid, T* dst) {
  assert(pid < context.pset->partition_count);
  partition_t* par = &context.pset->partitions[pid];
  for (int rmt_pid = 0; rmt_pid < context.pset->partition_count; rmt_pid++) {
    if (rmt_pid == pid) continue;
    grooves_box_table_t* inbox = &par->inbox[rmt_pid];
    if (!inbox->count) continue;
    if (par->processor.type == PROCESSOR_GPU) {
      dim3 blocks, threads;
      KERNEL_CONFIGURE(inbox->count, blocks, threads);
      scatter_add<<<blocks, threads, 0, par->streams[1]>>>(*inbox, dst);
      CALL_CU_SAFE(cudaGetLastError());
    } else {
      assert(par->processor.type == PROCESSOR_CPU);
      OMP(omp parallel for)
      for (uint32_t index = 0; index < inbox->count; index++) {
        _REDUCE_ENTRY_ADD(inbox, index, dst);
      }
    }
  }
}

template<typename T>
void engine_scatter_inbox_min(uint32_t pid, T* dst) {
  assert(pid < context.pset->partition_count);
  partition_t* par = &context.pset->partitions[pid];
  for (int rmt_pid = 0; rmt_pid < context.pset->partition_count; rmt_pid++) {
    if (rmt_pid == pid) continue;
    grooves_box_table_t* inbox = &par->inbox[rmt_pid];
    if (!inbox->count) continue;
    if (par->processor.type == PROCESSOR_GPU) {
      dim3 blocks, threads;
      KERNEL_CONFIGURE(inbox->count, blocks, threads);
      scatter_min<<<blocks, threads, 0, par->streams[1]>>>(*inbox, dst);
      CALL_CU_SAFE(cudaGetLastError());
    } else {
      assert(par->processor.type == PROCESSOR_CPU);
      OMP(omp parallel for)
      for (uint32_t index = 0; index < inbox->count; index++) {
        _REDUCE_ENTRY_MIN(inbox, index, dst);
      }
    }
  }
}

template<typename T>
void engine_scatter_inbox_max(uint32_t pid, T* dst) {
  assert(pid < context.pset->partition_count);
  partition_t* par = &context.pset->partitions[pid];
  for (int rmt_pid = 0; rmt_pid < context.pset->partition_count; rmt_pid++) {
    if (rmt_pid == pid) continue;
    grooves_box_table_t* inbox = &par->inbox[rmt_pid];
    if (!inbox->count) continue;
    if (par->processor.type == PROCESSOR_GPU) {
      dim3 blocks, threads;
      KERNEL_CONFIGURE(inbox->count, blocks, threads);
      scatter_max<<<blocks, threads, 0, par->streams[1]>>>(*inbox, dst);
      CALL_CU_SAFE(cudaGetLastError());
    } else {
      assert(par->processor.type == PROCESSOR_CPU);
      OMP(omp parallel for)
      for (uint32_t index = 0; index < inbox->count; index++) {
        _REDUCE_ENTRY_MAX(inbox, index, dst);
      }
    }
  }
}

template<typename T>
__global__ void gather(grooves_box_table_t inbox, T* src) {
  uint32_t index = THREAD_GLOBAL_INDEX;
  if (index >= inbox.count) return;
  ((T*)(inbox.pull_values))[index] = src[inbox.rmt_nbrs[index]];
}

template<typename T>
void engine_gather_inbox(uint32_t pid, T* src) {
  assert(pid < context.pset->partition_count);
  partition_t* par = &context.pset->partitions[pid];
  for (int rmt_pid = 0; rmt_pid < context.pset->partition_count; rmt_pid++) {
    if (rmt_pid == pid) continue;
    grooves_box_table_t* inbox = &par->inbox[rmt_pid];
    if (!inbox->count) continue;
    if (par->processor.type == PROCESSOR_GPU) {
      dim3 blocks, threads;
      KERNEL_CONFIGURE(inbox->count, blocks, threads);
      gather<<<blocks, threads, 0, par->streams[1]>>>(*inbox, src);
      CALL_CU_SAFE(cudaGetLastError());
    } else {
      assert(par->processor.type == PROCESSOR_CPU);
      OMP(omp parallel for)
      for (uint32_t index = 0; index < inbox->count; index++) {
        ((T*)(inbox->pull_values))[index] = src[inbox->rmt_nbrs[index]];
      }
    }
  }
}

template<typename T>
void engine_set_outbox(uint32_t pid, T value) {
  assert(pid < context.pset->partition_count);
  partition_t* par = &context.pset->partitions[pid];
  for (int rmt_pid = 0; rmt_pid < context.pset->partition_count; rmt_pid++) {
    if (rmt_pid == pid) continue;
    grooves_box_table_t* outbox =  &par->outbox[rmt_pid];
    if (!outbox->count) continue;
    T* values = (T*)outbox->push_values;
    if (par->processor.type == PROCESSOR_GPU) {
      CALL_SAFE(totem_memset(values, value, outbox->count, TOTEM_MEM_DEVICE,
                             par->streams[1]));
    } else {
      assert(par->processor.type == PROCESSOR_CPU);
      OMP(omp parallel for)
      for (uint32_t i = 0; i < outbox->count; i++) values[i] = value;
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

inline vid_t engine_vertex_count() {
  assert(context.pset);
  return context.pset->graph->vertex_count;
}

inline eid_t engine_edge_count() {
  assert(context.pset);
  return context.pset->graph->edge_count;
}

inline vid_t engine_largest_gpu_partition() {
  return context.largest_gpu_par;
}

inline void engine_report_not_finished() {
  *context.finished = false;
}

inline bool* engine_get_finished_ptr() {
  return context.finished;
}

inline bool* engine_get_finished_ptr(int pid) {
  bool* finished = context.finished;
  if (context.pset->partitions[pid].processor.type == PROCESSOR_GPU) {
    CALL_CU_SAFE(cudaHostGetDevicePointer((void **)&(finished), 
                                          (void *)context.finished, 0));
  }
  return finished;
}

inline vid_t* engine_vertex_id_in_partition() {
  return context.pset->id_in_partition;
}

inline vid_t engine_vertex_id_in_partition(vid_t v) {
  return context.pset->id_in_partition[v];
}

inline vid_t engine_vertex_id_local_to_global(vid_t v) {
  int pid = GET_PARTITION_ID(v);
  vid_t vid = GET_VERTEX_ID(v);
  assert(pid < context.pset->partition_count);
  assert(vid < context.pset->partitions[pid].subgraph.vertex_count);
  return context.pset->partitions[pid].map[vid];
}

inline const graph_t* engine_get_graph() {
  return context.graph;
}

inline partition_algorithm_t engine_partition_algorithm() {
  return context.attr.par_algo;
}

inline bool engine_sorted() {
  return context.attr.sorted;
}

#endif  // TOTEM_ENGINE_INTERNAL_CUH
