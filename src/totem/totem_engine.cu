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
    SET_PROCESSOR(&context.pset->partitions[pid]);
    context.config.par_kernel_func(&context.pset->partitions[pid]);
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
  if (context.initialized) return FAILURE;
  if (!config->par_kernel_func) return FAILURE;
  memset(&context, 0, sizeof(engine_context_t));
  context.config = *config;

  int pcount;
  CALL_CU_SAFE(cudaGetDeviceCount(&pcount));
  pcount += 1;
  processor_t* processors = (processor_t*)calloc(pcount, sizeof(processor_t));
  assert(processors);
  for (int gpu_id = 0; gpu_id < pcount; gpu_id++) {
    processors[gpu_id].type = PROCESSOR_GPU;
    processors[gpu_id].id = gpu_id;
  }
  processors[pcount - 1].type = PROCESSOR_CPU;

  // partition the graph
  stopwatch_t stopwatch_par;  
  stopwatch_start(&stopwatch_par);
  id_t* par_labels;
  switch (config->par_algo) {
    case PAR_RANDOM:
      CALL_SAFE(partition_random(config->graph, (uint32_t)pcount, 
                                 13, &(par_labels)));
      break;
    default:
      // TODO(abdullah): Use Lauro's logging library.
      printf("ERROR: Undefined partition algorithm.\n"); fflush(stdout);
      assert(false);
  }
  context.time_par = stopwatch_elapsed(&stopwatch_par);
  CALL_SAFE(partition_set_initialize(config->graph, par_labels,
                                     processors, pcount, config->msg_size, 
                                     &context.pset));
  free(processors);
  free(par_labels);

  // callback the per-partition initialization function
  if (context.config.par_init_func) {
    for (int pid = 0; pid < context.pset->partition_count; pid++) {
      SET_PROCESSOR(&context.pset->partitions[pid]);
      context.config.par_init_func(&context.pset->partitions[pid]);
    }
  }

  // get largest gpu partition
  uint64_t largest = 0;
  for (int pid = 0; pid < context.pset->partition_count; pid++) {
    partition_t* par = &context.pset->partitions[pid];
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
