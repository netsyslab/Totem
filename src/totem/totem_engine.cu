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

inline PRIVATE void reset_exec_timers() {
    context.time_exec     = 0;
    context.time_comm     = 0;
    context.time_scatter  = 0;
    context.time_comp     = 0;
    context.time_gpu_comp = 0;
    context.time_aggr     = 0;
}

/**
 * Returns true if all partitions reported a finished state
 */
inline PRIVATE bool superstep_check_finished() {
  bool finished = true;
  for (int pid = 0; pid < context.pset->partition_count; pid++) {
    finished &= context.finished[pid];
  }
  return finished;
}

/**
 * Blocks until all kernels initiated by the client have finished.
 */
inline PRIVATE void superstep_compute_synchronize() {
  for (int pid = 0; pid < context.pset->partition_count; pid++) {
    partition_t* par = &context.pset->partitions[pid];
    if (par->processor.type == PROCESSOR_CPU) continue;
    CALL_CU_SAFE(cudaStreamSynchronize(par->streams[1]));
    float time;
    cudaEventElapsedTime(&time, par->event_start, par->event_end);
    context.time_gpu_comp += time;
  }
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
    if (par->processor.type == PROCESSOR_GPU) {
      CALL_CU_SAFE(cudaEventRecord(par->event_start, par->streams[1]));
      set_processor(par);
    }
    context.config.par_kernel_func(par);
    if (par->processor.type == PROCESSOR_GPU) {
      CALL_CU_SAFE(cudaEventRecord(par->event_end, par->streams[1]));
    }
  }
  superstep_compute_synchronize();
  context.time_comp += stopwatch_elapsed(&stopwatch);
}

/**
 * Triggers grooves to synchronize state across partitions
 */
inline PRIVATE void superstep_communicate() {
  stopwatch_t stopwatch;
  stopwatch_start(&stopwatch);
  grooves_launch_communications(context.pset);
  grooves_synchronize(context.pset);
  stopwatch_t stopwatch_aggr;
  stopwatch_start(&stopwatch_aggr);
  if (!context.config.par_scatter_func) return;
  for (int pid = 0; pid < context.pset->partition_count; pid++) {
    set_processor(&context.pset->partitions[pid]);
    context.config.par_scatter_func(&context.pset->partitions[pid]);
  }
  context.time_scatter += stopwatch_elapsed(&stopwatch_aggr);
  context.time_comm += stopwatch_elapsed(&stopwatch);
}

/**
 * Prepares state for the next superstep
 */
inline PRIVATE void superstep_next() {
  context.superstep++;
  memset(context.finished, 0, context.pset->partition_count * sizeof(bool));
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
  context.time_aggr = stopwatch_elapsed(&stopwatch); 
}

error_t engine_execute() {
  stopwatch_t stopwatch;
  stopwatch_start(&stopwatch);
  while (true) {
    superstep_next();                      // prepare state for the next round
    superstep_compute();                   // compute phase
    if (superstep_check_finished()) break; // check for termination
    superstep_communicate();               // communication/synchronize phase
    if (superstep_check_finished()) break; // check for termination
  }
  engine_aggregate();
  context.time_exec = stopwatch_elapsed(&stopwatch); 
  if (context.config.par_finalize_func) {
    for (int pid = 0; pid < context.pset->partition_count; pid++) {
      set_processor(&context.pset->partitions[pid]);
      context.config.par_finalize_func(&context.pset->partitions[pid]);
    }
  }
  return SUCCESS;
}

error_t engine_init(graph_t* graph, totem_attr_t* attr) {
  stopwatch_t stopwatch;
  stopwatch_start(&stopwatch);
  if (context.initialized || !graph->vertex_count) return FAILURE;
  memset(&context, 0, sizeof(engine_context_t));
  context.graph = graph;
  context.attr = *attr;

  // identify the execution platform
  int gpu_count = attr->gpu_count;
  bool use_cpu = true;
  switch(context.attr.platform) {
    case PLATFORM_CPU:
      gpu_count = 0;
      break;
    case PLATFORM_GPU:
      use_cpu = false;
    case PLATFORM_HYBRID:
      if (gpu_count > get_gpu_count()) {
        return FAILURE;
      }
      break;
    default:
      assert(false);
  }

  // identify the share of each partition
  // TODO(abdullah): ideally, we would like to split the graph among processors
  // with different shares (e.g., a system with GPUs with different
  // memory capacities).
  int pcount = use_cpu ? gpu_count + 1 : gpu_count;
  double* par_share = NULL;
  if (context.attr.platform == PLATFORM_HYBRID) {
    par_share = (double*)calloc(pcount, sizeof(double));
    par_share[pcount - 1] = context.attr.cpu_par_share;
    // the rest is divided equally among the GPUs
    double gpu_par_share = 
      (1.0 - context.attr.cpu_par_share) / (double)gpu_count;
    double total_share = context.attr.cpu_par_share;
    for (int gpu_id = 0; gpu_id < gpu_count - 1; gpu_id++) {
      par_share[gpu_id] = gpu_par_share;
      total_share += gpu_par_share;
    }
    par_share[gpu_count - 1] = 1.0 - total_share;
  }

  // setup the processors' types and ids
  processor_t* processors = (processor_t*)calloc(pcount, sizeof(processor_t));
  assert(processors);
  for (int gpu_id = 0; gpu_id < gpu_count; gpu_id++) {
    processors[gpu_id].type = PROCESSOR_GPU;
    processors[gpu_id].id = gpu_id;
  }
  if (use_cpu) {
    processors[pcount - 1].type = PROCESSOR_CPU;
  }

  // partition the graph
  stopwatch_t stopwatch_par;
  stopwatch_start(&stopwatch_par);
  id_t* par_labels;
  assert(context.attr.par_algo < PAR_MAX);
  CALL_SAFE(PARTITION_FUNC[context.attr.par_algo](context.graph, pcount, 
                                                  par_share, &par_labels));
  context.time_par = stopwatch_elapsed(&stopwatch_par);
  CALL_SAFE(partition_set_initialize(context.graph, par_labels,
                                     processors, pcount,
                                     context.attr.msg_size,
                                     &context.pset));
  free(processors);
  free(par_labels);
  if (par_share) free(par_share);

  // get largest gpu partition and initialize the stat information
  // stored in the context
  context.partition_count = context.pset->partition_count;
  uint64_t largest = 0;
  for (int pid = 0; pid < context.pset->partition_count; pid++) {
    partition_t* par = &context.pset->partitions[pid];
    context.vertex_count[pid]     = par->subgraph.vertex_count;
    context.edge_count[pid]       = par->subgraph.edge_count;
    context.rmt_vertex_count[pid] = par->rmt_vertex_count;
    context.rmt_edge_count[pid]   = par->rmt_edge_count;
    if (par->processor.type == PROCESSOR_CPU) continue;
    uint64_t vcount = context.pset->partitions[pid].subgraph.vertex_count;
    largest = vcount > largest ? vcount : largest;
  }
  context.largest_gpu_par = largest;

  context.finished = (bool*)calloc(pcount, sizeof(bool));
  context.initialized = true;
  context.time_init = stopwatch_elapsed(&stopwatch);
  return SUCCESS;
}

error_t engine_config(engine_config_t* config) {
  if (!context.initialized || !config->par_kernel_func) return FAILURE;

  context.config = *config;
  reset_exec_timers();
  context.superstep = 0;
  memset(context.finished, 0, context.pset->partition_count * sizeof(bool));
  // callback the per-partition initialization function
  if (context.config.par_init_func) {
    for (int pid = 0; pid < context.pset->partition_count; pid++) {
      set_processor(&context.pset->partitions[pid]);
      context.config.par_init_func(&context.pset->partitions[pid]);
    }
  }
  return SUCCESS;
}

void engine_finalize() {
  assert(context.initialized);
  context.initialized = false;
  CALL_SAFE(partition_set_finalize(context.pset));
  free(context.finished);
}
