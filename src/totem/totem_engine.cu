/**
 * Implements the core execution engine of Totem
 *
 *  Created on: 2012-02-02
 *  Author: Abdullah Gharaibeh
 */

#include "totem_engine.cuh"

engine_context_t context = {false, NULL, 0, ENGINE_DEFAULT_CONFIG, 
                            0, 0, 0, 0};

#define SET_PROCESSOR(_par)                                     \
  do {                                                          \
    if ((_par)->processor.type == PROCESSOR_GPU) {              \
      CALL_CU_SAFE(cudaSetDevice((_par)->processor.id));        \
    }                                                           \
  } while(0)

/**
 * Clears allocated state
 */
PRIVATE void engine_finalize() {
  assert(context.pset);
  if (context.config.par_finalize_func) {
    for (int pid = 0; pid < context.pset->partition_count; pid++) {
      SET_PROCESSOR(&context.pset->partitions[pid]);
      context.config.par_finalize_func(&context.pset->partitions[pid]);
    }
  }
  CALL_SAFE(partition_set_finalize(context.pset));
  free(context.finished);
  context.initialized = false;
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
      SET_PROCESSOR(par);
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
  if (!context.config.par_scatter_func) return;
  // The assumption is that the first partition is the CPU one, and the
  // rest are GPU ones. This is guaranteed by engine_init.
  for (int pid = 0; pid < context.pset->partition_count; pid++) {
    SET_PROCESSOR(&context.pset->partitions[pid]);
    context.config.par_scatter_func(&context.pset->partitions[pid]);
  }
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
  if (context.config.par_aggr_func) {
    for (int pid = 0; pid < context.pset->partition_count; pid++) {
      SET_PROCESSOR(&context.pset->partitions[pid]);
      context.config.par_aggr_func(&context.pset->partitions[pid]);
    }
  }
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

  engine_finalize();
  return SUCCESS;
}

error_t engine_init(engine_config_t* config) {
  stopwatch_t stopwatch;
  stopwatch_start(&stopwatch);
  if (context.initialized || !config->par_kernel_func) return FAILURE;
  memset(&context, 0, sizeof(engine_context_t));
  context.config = *config;

  // identify the execution platform
  int gpu_count;
  bool use_cpu = true;
  CALL_CU_SAFE(cudaGetDeviceCount(&gpu_count));
  switch(context.config.platform) {
    case PLATFORM_CPU:
      gpu_count = 0;
      context.config.cpu_par_share = 0;
      break;
    case PLATFORM_GPU:
      gpu_count = 1;
      use_cpu = false;
      break;
    case PLATFORM_MULTI_GPU:
      use_cpu = false;
      break;
    case PLATFORM_HYBRID:
      gpu_count = 1;
      break;
    case PLATFORM_ALL:
      break;
    default:
      assert(false);        
  }
  int pcount = use_cpu ? gpu_count + 1 : gpu_count;
  processor_t* processors = (processor_t*)calloc(pcount, sizeof(processor_t));
  assert(processors);

  // identify the share of each partition  
  // TODO(abdullah): ideally, we would like to split the graph among processors
  // with different shares (e.g., a system with GPUs with different 
  // memory capacities).
  float* par_share = NULL;
  if (context.config.cpu_par_share && use_cpu) {
    par_share = (float*)calloc(pcount, sizeof(float));
    par_share[pcount - 1] = context.config.cpu_par_share;
    float gpu_par_share = 
      (1.0 - context.config.cpu_par_share) / (float)gpu_count;
    float total_share = context.config.cpu_par_share;
    for (int gpu_id = 0; gpu_id < gpu_count - 1; gpu_id++) {
      par_share[gpu_id] = gpu_par_share;
      total_share += gpu_par_share;
    }
    par_share[gpu_count - 1] = 1.0 - total_share;
  }
  
  // setup the processors' types and ids
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
  switch (context.config.par_algo) {
    case PAR_RANDOM:
      CALL_SAFE(partition_random(context.config.graph, 
                                 (uint32_t)pcount, par_share,
                                 13, &par_labels));
      break;
    default:
      // TODO(abdullah): Use Lauro's logging library.
      printf("ERROR: Undefined partition algorithm.\n"); fflush(stdout);
      assert(false);
  }
  context.time_par = stopwatch_elapsed(&stopwatch_par);
  CALL_SAFE(partition_set_initialize(context.config.graph, par_labels,
                                     processors, pcount, 
                                     context.config.msg_size, 
                                     &context.pset));
  free(processors);
  free(par_labels);
  if (par_share) free(par_share);

  // callback the per-partition initialization function
  if (context.config.par_init_func) {
    for (int pid = 0; pid < context.pset->partition_count; pid++) {
      SET_PROCESSOR(&context.pset->partitions[pid]);
      context.config.par_init_func(&context.pset->partitions[pid]);
    }
  }

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
