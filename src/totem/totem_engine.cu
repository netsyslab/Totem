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
inline PRIVATE double superstep_compute_synchronize() {
  double max_gpu_time = 0;
  for (int pid = 0; pid < context.pset->partition_count; pid++) {
    partition_t* par = &context.pset->partitions[pid];
    if (par->processor.type == PROCESSOR_CPU) continue;
    float time;
    cudaEventElapsedTime(&time, par->event_start, par->event_end);
    // log the total time spent computing on GPUs
    context.timing.alg_gpu_total_comp += time;
    max_gpu_time = time > max_gpu_time ? time : max_gpu_time;
    // TODO(abdullah): use a logging mechanism instead of ifdef
#ifdef FEATURE_VERBOSE_TIMING
    printf("\tGPU%d: %0.2f", par->processor.id, time);
#endif
  }
  // log the time of the slowest gpu
  context.timing.alg_gpu_comp += max_gpu_time;
  return max_gpu_time;
}

PRIVATE void superstep_launch_gpu(partition_t* par) {
  // The kernel for GPU partitions is supposed not to block. The client is
  // supposedly invoking the GPU kernel asynchronously, and using the compute
  // "stream" available for each partition
  set_processor(par);
  CALL_CU_SAFE(cudaEventRecord(par->event_start, par->streams[1]));
  if (par->subgraph.vertex_count != 0) {
    context.config.par_kernel_func(par);
  }
  CALL_CU_SAFE(cudaEventRecord(par->event_end, par->streams[1]));
}

PRIVATE void superstep_launch_cpu(partition_t* par, double& cpu_time,
                                  double& cpu_scatter_time) {
  cpu_time = 0;
  cpu_scatter_time = 0;
  if (par->subgraph.vertex_count != 0) {
    stopwatch_t stopwatch;
    stopwatch_start(&stopwatch);
    if ((context.config.direction == GROOVES_PUSH) &&
        (context.config.par_scatter_func != NULL) &&
        ((context.superstep > 1))) {
      context.config.par_scatter_func(par);
    }
    cpu_scatter_time = stopwatch_elapsed(&stopwatch);

    // Make sure that data sent/received from a GPU partition to
    // the CPU partition is available
    for (int rmt_pid = 0; rmt_pid < context.pset->partition_count;
         rmt_pid++) {
      if (rmt_pid == par->id) continue;
      partition_t* rmt_par = &context.pset->partitions[rmt_pid];
      CALL_CU_SAFE(cudaStreamSynchronize(rmt_par->streams[0]));
    }

    stopwatch_start(&stopwatch);
    context.config.par_kernel_func(par);
    cpu_time = stopwatch_elapsed(&stopwatch);
    context.timing.alg_cpu_comp += cpu_time;
  }
#ifdef FEATURE_VERBOSE_TIMING
  printf("#\tCPU: %0.2f", cpu_time);
#endif
}

/**
 * Launches the compute kernel on each partition
 */
inline PRIVATE void superstep_execute() {
  stopwatch_t stopwatch;
  stopwatch_start(&stopwatch);

  // invoke the per superstep callback function
  if (context.config.ss_kernel_func) {
    context.config.ss_kernel_func();
  }

  bool* tmp = context.comm_curr;
  context.comm_curr = context.comm_prev;
  context.comm_prev = tmp;
  memset(context.comm_curr, true, MAX_PARTITION_COUNT);

  if ((context.superstep > 1)) {
    for (int pid = 0; pid < engine_partition_count(); pid++) {
      partition_t* par = &context.pset->partitions[pid];
      if (par->subgraph.vertex_count == 0) continue;
      if (context.config.direction == GROOVES_PUSH) {
        if ((context.config.par_scatter_func != NULL) &&
            (par->processor.type == PROCESSOR_GPU)) {
          set_processor(par);
          context.config.par_scatter_func(par);
        }
      } else if (context.config.direction == GROOVES_PULL) {
        grooves_launch_communications(context.pset, pid, GROOVES_PULL);
      } else {
        assert(false);
        fprintf(stderr, "Unsupported communication type %d\n",
                context.config.direction); fflush(stderr);
      }
    }
  }

  double cpu_time = 0;
  double cpu_gather_time = 0;
  double cpu_scatter_time = 0;
  for (int pid = 0; pid < context.pset->partition_count; pid++) {
    partition_t* par = &context.pset->partitions[pid];
    if (par->processor.type == PROCESSOR_GPU) {
      superstep_launch_gpu(par);
    } else if (par->processor.type == PROCESSOR_CPU) {
      superstep_launch_cpu(par, cpu_time, cpu_scatter_time);
    } else {
      fprintf(stderr, "Unsupported processor type %d\n",
              context.config.direction); fflush(stderr);
      assert(false);
    }
    if (par->subgraph.vertex_count == 0) continue;
    // If push-based, launch communication; if pull-based, launch gather kernels
    if (context.comm_curr[par->id]) {
      if (context.config.direction == GROOVES_PUSH) {
        // communication will be launched in the context of the source stream it
        // will start only after the kernel in the source partition finished
        // execution
        grooves_launch_communications(context.pset, par->id, GROOVES_PUSH);
      }
    }
  }
  if ((context.config.direction == GROOVES_PULL) &&
      (context.config.par_gather_func != NULL)) {
    for (int pid = 0; pid < engine_partition_count(); pid++) {
      partition_t* par = &context.pset->partitions[pid];
      set_processor(par);
      stopwatch_t gather;
      stopwatch_start(&gather);
      context.config.par_gather_func(par);
      cpu_gather_time += stopwatch_elapsed(&gather);
    }
  }
  double launch_time = stopwatch_elapsed(&stopwatch);

  // Synchronize the streams (data transfers and kernels) and swap the buffers
  // used for double buffering to overlap communication with computation
  grooves_synchronize(context.pset, context.config.direction);
  double gpu_time = superstep_compute_synchronize();

  // Record the time spent on computation and communication
  double total_time = stopwatch_elapsed(&stopwatch);
  double comp_time = cpu_time > gpu_time ? cpu_time : gpu_time;
  context.timing.alg_comp += comp_time;
  double comm_time = total_time - comp_time;
  context.timing.alg_comm += comm_time;
  context.timing.alg_scatter += cpu_scatter_time;
  context.timing.alg_gather += cpu_gather_time;
#ifdef FEATURE_VERBOSE_TIMING
  printf("\tComp: %0.2f\tComm: %0.2f\tLaunch: %0.2f\tCPUScatter: %0.2f"
         "\tCPUGather: %0.2f\t\tTotal: %0.2f\n",
         comp_time, comm_time, launch_time, cpu_scatter_time, cpu_gather_time,
         total_time);
#endif
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
  for (int pid = 0; pid < context.pset->partition_count; pid++) {
    partition_t* par = &context.pset->partitions[pid];
    if (par->processor.type == PROCESSOR_CPU) { continue; }
    CALL_CU_SAFE(cudaStreamSynchronize(par->streams[1]));
  }
  context.timing.alg_aggr += stopwatch_elapsed(&stopwatch);
}

error_t engine_execute() {
  stopwatch_t stopwatch;
  stopwatch_start(&stopwatch);
  while (true) {
    superstep_next();             // prepare state for the next round
    superstep_execute();          // compute phase
    if (*context.finished) break; // check for termination
  }

  context.timing.alg_exec += stopwatch_elapsed(&stopwatch);
  engine_aggregate();
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

PRIVATE void init_partition(int pcount, int gpu_count, double* shares,
                            processor_t* processors) {
  stopwatch_t stopwatch_par;
  stopwatch_start(&stopwatch_par);
  vid_t* par_labels;
  assert(context.attr.par_algo < PAR_MAX);
  CALL_SAFE(PARTITION_FUNC[context.attr.par_algo](context.graph, pcount,
                                                  context.attr.platform ==
                                                  PLATFORM_HYBRID ?
                                                  shares : NULL, &par_labels,
                                                  &context.attr));

  context.timing.engine_par = stopwatch_elapsed(&stopwatch_par);
  CALL_SAFE(partition_set_initialize(context.graph, par_labels,
                                     processors, pcount, &context.attr,
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

PRIVATE error_t init_check_space(graph_t* graph, totem_attr_t* attr, int pcount,
                                 double* shares, processor_t* processors) {
  if (attr->gpu_graph_mem != GPU_GRAPH_MEM_DEVICE) return SUCCESS;
  for (int pid = 0; pid < pcount; pid++) {
    if (processors[pid].type == PROCESSOR_GPU) {
      size_t needed = (((double)graph->vertex_count +
                        (double)graph->edge_count) *
                       shares[pid]) * sizeof(vid_t);
      CALL_CU_SAFE(cudaSetDevice(processors[pid].id));
      size_t available = 0; size_t total = 0;
      CALL_CU_SAFE(cudaMemGetInfo(&available, &total));
      // Reserve at least GPU_MIN_ALG_STATE of the space for algorithm state
      available = (double)available * (1 - GPU_MIN_ALG_STATE);
      if (needed > available) {
        fprintf(stderr,
                "Error: GPU out of memory. Needed:%dMB, Available:%dMB\n",
                needed/(1024*1024), available/(1024*1024));
        return FAILURE;
      }
    }
  }
  return SUCCESS;
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
  if (init_check_space(graph, attr, pcount, shares, processors) == FAILURE) {
    return FAILURE;
  }
  init_partition(pcount, gpu_count, shares, processors);
  init_context_partitions_state();

  // allocate application specific state
  if (attr->alloc_func) {
    for (int pid = 0; pid < context.pset->partition_count; pid++) {
      set_processor(&context.pset->partitions[pid]);
      attr->alloc_func(&context.pset->partitions[pid]);
    }
  }

  context.comm_curr = (bool*)malloc(MAX_PARTITION_COUNT);
  context.comm_prev = (bool*)malloc(MAX_PARTITION_COUNT);

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
  memset(context.comm_curr, true, MAX_PARTITION_COUNT);
  memset(context.comm_prev, true, MAX_PARTITION_COUNT);
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
  free(context.comm_curr);
  free(context.comm_prev);
  // free application-specific state
  if (context.attr.free_func) {
    for (int pid = 0; pid < context.pset->partition_count; pid++) {
      set_processor(&context.pset->partitions[pid]);
      context.attr.free_func(&context.pset->partitions[pid]);
    }
  }
  assert(context.initialized);
  context.initialized = false;
  CALL_CU_SAFE(cudaFreeHost(context.finished));
  CALL_SAFE(partition_set_finalize(context.pset));
}

error_t engine_update_msg_size(grooves_direction_t dir, size_t msg_size) {
  if (((dir == GROOVES_PUSH) && (msg_size > context.attr.push_msg_size)) ||
      ((dir == GROOVES_PULL) && (msg_size > context.attr.pull_msg_size))) {
    return FAILURE;
  }
  partition_set_update_msg_size(context.pset, dir, msg_size);
  return SUCCESS;
}

void engine_reset_msg_size(grooves_direction_t dir) {
  size_t msg_size = 0;
  if (dir == GROOVES_PUSH) {
    msg_size = context.attr.push_msg_size;
  } else if(dir == GROOVES_PULL) {
    msg_size = context.attr.pull_msg_size;
  } else {
    fprintf(stderr, "Unsupported communication direction type\n");
    assert(false);
  }
  partition_set_update_msg_size(context.pset, dir, msg_size);
}

void engine_reset_bsp_timers() {
  context.timing.alg_exec           = 0;
  context.timing.alg_comp           = 0;
  context.timing.alg_comm           = 0;
  context.timing.alg_aggr           = 0;
  context.timing.alg_scatter        = 0;
  context.timing.alg_gather         = 0;
  context.timing.alg_gpu_comp       = 0;
  context.timing.alg_gpu_total_comp = 0;
  context.timing.alg_cpu_comp       = 0;
  context.timing.alg_init           = 0;
  context.timing.alg_finalize       = 0;
}
