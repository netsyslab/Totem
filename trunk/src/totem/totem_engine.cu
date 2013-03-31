/**
 * Implements the core execution engine of Totem
 *
 *  Created on: 2012-02-02
 *  Author: Abdullah Gharaibeh
 */

#include "totem_engine.cuh"
#include "totem_util.h"

engine_context_t context = ENGINE_DEFAULT_CONTEXT;

inline PRIVATE void set_processor(partition_t* par) {
  if (par->processor.type == PROCESSOR_GPU) {
      CALL_CU_SAFE(cudaSetDevice(par->processor.id));
  }
}

/**
 * Blocks until all kernels initiated by the client have finished.
 */
inline PRIVATE void superstep_compute_synchronize() {
  float max_gpu_time = 0;
  for (int pid = 0; pid < context.pset->partition_count; pid++) {
    partition_t* par = &context.pset->partitions[pid];
    if (par->processor.type == PROCESSOR_CPU) continue;
    CALL_CU_SAFE(cudaStreamSynchronize(par->streams[1]));
    float time;
    cudaEventElapsedTime(&time, par->event_start, par->event_end);
    // in a multi-gpu setup, we time the slowest one
    max_gpu_time = time > max_gpu_time ? time : max_gpu_time;
  }
  context.timing.alg_gpu_comp += max_gpu_time;
}

/**
 * Launches the compute kernel on each partition
 */
inline PRIVATE void superstep_compute() {
  stopwatch_t stopwatch;
  stopwatch_start(&stopwatch);
  // invoke the per superstep callback function
  if (context.config.ss_kernel_func) {
    context.config.ss_kernel_func();
  }
  for (int pid = 0; pid < context.pset->partition_count; pid++) {
    // The kernel for GPU partitions is supposed not to block. The client is
    // supposedly invoking the GPU kernel asynchronously, and using the compute
    // "stream" available for each partition
    partition_t* par = &context.pset->partitions[pid];
    stopwatch_t stopwatch_cpu;
    if (par->processor.type == PROCESSOR_GPU) {
      CALL_CU_SAFE(cudaEventRecord(par->event_start, par->streams[1]));
      set_processor(par);
    } else {
      stopwatch_start(&stopwatch_cpu);
    }
    context.config.par_kernel_func(par);
    if (par->processor.type == PROCESSOR_GPU) {
      CALL_CU_SAFE(cudaEventRecord(par->event_end, par->streams[1]));
    } else {
      context.timing.alg_cpu_comp += stopwatch_elapsed(&stopwatch_cpu);
    }
  }
  superstep_compute_synchronize();
  context.timing.alg_comp += stopwatch_elapsed(&stopwatch);
}

/**
 * Triggers grooves to synchronize state across partitions
 */
inline PRIVATE void superstep_communicate() {
  stopwatch_t stopwatch;
  stopwatch_start(&stopwatch);
  if (context.config.par_gather_func) {
    for (int pid = 0; pid < context.pset->partition_count; pid++) {
      set_processor(&context.pset->partitions[pid]);
      context.config.par_gather_func(&context.pset->partitions[pid]);
    }
  }
  context.timing.alg_gather += stopwatch_elapsed(&stopwatch);
  grooves_launch_communications(context.pset, context.config.direction);
  grooves_synchronize(context.pset);
  stopwatch_t stopwatch_scatter;
  stopwatch_start(&stopwatch_scatter);
  if (context.config.par_scatter_func) {
    for (int pid = 0; pid < context.pset->partition_count; pid++) {
      set_processor(&context.pset->partitions[pid]);
      context.config.par_scatter_func(&context.pset->partitions[pid]);
    }
  }
  context.timing.alg_scatter += stopwatch_elapsed(&stopwatch_scatter);
  context.timing.alg_comm += stopwatch_elapsed(&stopwatch);
}

/**
 * Prepares state for the next superstep
 */
inline PRIVATE void superstep_next() {
  context.superstep++;
  *context.finished = true;
}

PRIVATE void engine_aggregate() {
  stopwatch_t stopwatch;
  stopwatch_start(&stopwatch);
  if (context.config.par_aggr_func) {
    for (int pid = 0; pid < context.pset->partition_count; pid++) {
      set_processor(&context.pset->partitions[pid]);
      context.config.par_aggr_func(&context.pset->partitions[pid]);
    }
  }
  context.timing.alg_aggr += stopwatch_elapsed(&stopwatch); 
}

error_t engine_execute() {
  stopwatch_t stopwatch;
  stopwatch_start(&stopwatch);
  while (true) {
    superstep_next();             // prepare state for the next round
    superstep_compute();          // compute phase
    if (*context.finished) break; // check for termination
    superstep_communicate();      // communication/synchronize phase
    if (*context.finished) break; // check for termination
  }
  engine_aggregate();
  context.timing.alg_exec += stopwatch_elapsed(&stopwatch);
  stopwatch_start(&stopwatch);
  if (context.config.par_finalize_func) {
    for (int pid = 0; pid < context.pset->partition_count; pid++) {
      set_processor(&context.pset->partitions[pid]);
      context.config.par_finalize_func(&context.pset->partitions[pid]);
    }
  }
  context.timing.alg_finalize += stopwatch_elapsed(&stopwatch); 
  return SUCCESS;
}

PRIVATE error_t init_check(graph_t* graph, totem_attr_t* attr) {
  if (context.initialized || !graph->vertex_count || 
      attr->gpu_count > get_gpu_count()) {
    return FAILURE;
  }
  return SUCCESS;
}

PRIVATE void init_get_platform(int &pcount, int &gpu_count, bool &use_cpu) {
  // identify the execution platform
  gpu_count = context.attr.gpu_count;
  use_cpu = true;
  switch(context.attr.platform) {
    case PLATFORM_CPU:
      gpu_count = 0;
      break;
    case PLATFORM_GPU:
      use_cpu = false;
      break;
    case PLATFORM_HYBRID:
      break;
    default:
      assert(false);
  }
  pcount = use_cpu ? gpu_count + 1 : gpu_count;
}

PRIVATE void init_get_shares(int pcount, int gpu_count, bool use_cpu, 
                             double* shares) {
  assert(shares);
  // identify the share of each partition
  // TODO(abdullah): ideally, we would like to split the graph among processors
  // with different shares (e.g., a system with GPUs with different
  // memory capacities).
  if (context.attr.platform == PLATFORM_HYBRID) {
    shares[pcount - 1] = context.attr.cpu_par_share;
    // the rest is divided equally among the GPUs
    double gpu_par_share = 
      (1.0 - context.attr.cpu_par_share) / (double)gpu_count;
    double total_share = context.attr.cpu_par_share;
    for (int gpu_id = 0; gpu_id < gpu_count - 1; gpu_id++) {
      shares[gpu_id] = gpu_par_share;
      total_share += gpu_par_share;
    }
    shares[gpu_count - 1] = 1.0 - total_share;
  }
}

PRIVATE void init_get_processors(int pcount, int gpu_count, bool use_cpu,
                                 processor_t* processors) {
  // setup the processors' types and ids
  assert(processors);
  for (int gpu_id = 0; gpu_id < gpu_count; gpu_id++) {
    processors[gpu_id].type = PROCESSOR_GPU;
    processors[gpu_id].id = gpu_id;
  }
  if (use_cpu) {
    processors[pcount - 1].type = PROCESSOR_CPU;
  }
}

PRIVATE void init_partition(int pcount, double* shares, 
                            processor_t* processors) {
  stopwatch_t stopwatch_par;
  stopwatch_start(&stopwatch_par);
  vid_t* par_labels;
  assert(context.attr.par_algo < PAR_MAX);
  CALL_SAFE(PARTITION_FUNC[context.attr.par_algo](context.graph, pcount, 
                                                  context.attr.platform == 
                                                  PLATFORM_HYBRID ?
                                                  shares : NULL, &par_labels));
  context.timing.engine_par = stopwatch_elapsed(&stopwatch_par);
  CALL_SAFE(partition_set_initialize(context.graph, par_labels,
                                     processors, pcount, context.attr.mapped,
                                     context.attr.push_msg_size,
                                     context.attr.pull_msg_size,
                                     &context.pset));
  free(par_labels);
}

PRIVATE void init_context(graph_t* graph, totem_attr_t* attr) {
  memset(&context, 0, sizeof(engine_context_t));
  context.graph = graph;
  context.attr = *attr;
  // The global finish flag is allocated on the host using the
  // cudaHostAllocMapped option which allows GPU kernels to access it directly
  // from within the GPU. This flag is initialized to true at the beginning of 
  // each superstep before the kernel callback. Any of the partitions set this
  // flag to false if it still has work to do.
  CALL_CU_SAFE(cudaHostAlloc((void **)&context.finished, sizeof(bool),
                             cudaHostAllocPortable | cudaHostAllocMapped));
  *context.finished = true;
  context.initialized = true;
}

PRIVATE void init_context_partitions_state() {
  // get largest gpu partition and initialize the stat information
  // stored in the context
  context.partition_count = context.pset->partition_count;
  context.largest_gpu_par = 0;
  for (int pid = 0; pid < context.pset->partition_count; pid++) {
    partition_t* par = &context.pset->partitions[pid];
    context.vertex_count[pid]     = par->subgraph.vertex_count;
    context.edge_count[pid]       = par->subgraph.edge_count;
    context.rmt_vertex_count[pid] = par->rmt_vertex_count;
    context.rmt_edge_count[pid]   = par->rmt_edge_count;
    if (par->processor.type == PROCESSOR_CPU) continue;
    uint64_t vcount = context.pset->partitions[pid].subgraph.vertex_count;
    context.largest_gpu_par = vcount > context.largest_gpu_par ? vcount : 
      context.largest_gpu_par;
  }
}

error_t engine_init(graph_t* graph, totem_attr_t* attr) {
  stopwatch_t stopwatch;
  stopwatch_start(&stopwatch);
  if (init_check(graph, attr) == FAILURE) return FAILURE;

  init_context(graph, attr);
  int pcount = 0;
  int gpu_count = 0;
  bool use_cpu = true;
  init_get_platform(pcount, gpu_count, use_cpu);
  double shares[MAX_PARTITION_COUNT];
  init_get_shares(pcount, gpu_count, use_cpu, shares);
  processor_t processors[MAX_PARTITION_COUNT];
  init_get_processors(pcount, gpu_count, use_cpu, processors);
  init_partition(pcount, shares, processors);
  init_context_partitions_state();

  context.timing.engine_init = stopwatch_elapsed(&stopwatch);
  return SUCCESS;
}

error_t engine_config(engine_config_t* config) {
  if (!context.initialized || !config->par_kernel_func) return FAILURE;
  context.config = *config;
  stopwatch_t stopwatch;
  stopwatch_start(&stopwatch);
  context.superstep = 0;
  *context.finished = false;
  // callback the per-partition initialization function
  if (context.config.par_init_func) {
    for (int pid = 0; pid < context.pset->partition_count; pid++) {
      set_processor(&context.pset->partitions[pid]);
      context.config.par_init_func(&context.pset->partitions[pid]);
    }
  }
  context.timing.alg_init += stopwatch_elapsed(&stopwatch);
  return SUCCESS;
}

void engine_finalize() {
  assert(context.initialized);
  context.initialized = false;
  CALL_CU_SAFE(cudaFreeHost(context.finished));
  CALL_SAFE(partition_set_finalize(context.pset));
}

void engine_reset_bsp_timers() {
  context.timing.alg_exec     = 0;
  context.timing.alg_comp     = 0;
  context.timing.alg_comm     = 0;
  context.timing.alg_aggr     = 0;
  context.timing.alg_scatter  = 0;
  context.timing.alg_gather   = 0;
  context.timing.alg_gpu_comp = 0;
  context.timing.alg_cpu_comp = 0;
  context.timing.alg_init     = 0;
  context.timing.alg_finalize = 0;
}
