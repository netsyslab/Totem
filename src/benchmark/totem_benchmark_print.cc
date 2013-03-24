/**
 * Benchmark printing functions
 *
 *  Created on: 2013-03-11
 *  Author: Abdullah Gharaibeh
 */

// totem includes
#include "totem_benchmark.h"

PRIVATE const char* PLATFORM_STR[] = {"CPU", "GPU", "HYBRID"};
PRIVATE const char* PAR_ALGO_STR[] = {"RANDOM", "HIGH", "LOW"};
PRIVATE const char* OMP_SCHEDULE_STR[] = {"", "STATIC", "DYNAMIC", "GUIDED", 
                                          "RUNTIME"};

/**
 * Prints partitioning characteristics
 */
PRIVATE void print_header_partitions(graph_t* graph) {
  uint64_t rv = 0; uint64_t re = 0;
  for (uint32_t pid = 0; pid < totem_partition_count(); pid++) {
    rv += totem_par_rmt_vertex_count(pid);
    re += totem_par_rmt_edge_count(pid);
  }
  // Print the total percentage of remote vertices/edges
  printf("rmt_vertex:%0.0f\trmt_edge:%0.0f\tbeta:%0.0f\t", 
         100.0*(double)((double)rv/(double)graph->vertex_count),
         100.0*(double)((double)re/(double)graph->edge_count),
         100.0*(double)((double)rv/(double)graph->edge_count));
  
  // For each partition, print partition id, % of vertices, % of edges, 
  // % of remote vertices, % of remote edges
  for (uint32_t pid = 0; pid < totem_partition_count(); pid++) {
    printf("pid%d:%0.0f,%0.0f,%0.0f,%0.0f\t", pid, 
           100.0 * ((double)totem_par_vertex_count(pid) / 
                    (double)graph->vertex_count), 
           100.0 * ((double)totem_par_edge_count(pid) / 
                    (double)graph->edge_count),
           100.0 * ((double)totem_par_rmt_vertex_count(pid) / 
                    (double)graph->vertex_count),
           100.0 * ((double)totem_par_rmt_edge_count(pid) / 
                    (double)graph->edge_count));
   
  }
}

/**
 * Prints out the configuration parameters of this benchmark run
 */
void print_header(graph_t* graph, benchmark_options_t* options, 
                  const char* benchmark_name, bool totem_based) {
  const char* OMP_PROC_BIND = getenv("OMP_PROC_BIND");
  omp_sched_t sched; int sched_modifier;
  omp_get_schedule(&sched, &sched_modifier);
  printf("file:%s\tbenchmark:%s\tvertices:%llu\tedges:%llu\tpartitioning:%s\t"
         "platform:%s\talpha:%d\trepeat:%d\tgpu_count:%d\t"
         "thread_count:%d\tthread_sched:%s\tthread_bind:%s", 
         options->graph_file, benchmark_name, 
         (uint64_t)graph->vertex_count, (uint64_t)graph->edge_count, 
         PAR_ALGO_STR[options->par_algo], PLATFORM_STR[options->platform], 
         options->alpha, options->repeat, options->gpu_count, 
         omp_get_max_threads(), OMP_SCHEDULE_STR[sched],
         OMP_PROC_BIND == NULL ? "false" : OMP_PROC_BIND);
  if (totem_based) {
    // print the time spent on initializing Totem and partitioning the graph
    const totem_timing_t* timers = totem_timing();
    printf("\ttime_init:%0.2f\ttime_par:%0.2f\t",
           timers->engine_init, timers->engine_par);
    print_header_partitions(graph);
  }
  printf("\ntotal\texec\tinit\tcomp\tcomm\tfinalize\tgpu_comp\tscatter\t"
         "gather\taggr\ttrv_edges\texec_rate\n"); 
  fflush(stdout);
}

/**
 * Prints out detailed timing of a single run
 */
void print_timing(graph_t* graph, double time_total, uint64_t trv_edges, 
                  bool totem_based) {
  const totem_timing_t* timers = totem_timing();
  printf("%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t"
         "%0.2f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\t%llu\t%0.4f\n",
         time_total,
         totem_based ? timers->alg_exec : time_total,
         totem_based ? timers->alg_init : 0,
         totem_based ? timers->alg_comp : time_total,
         totem_based ? timers->alg_comm : 0,
         totem_based ? timers->alg_finalize : 0,
         totem_based ? timers->alg_gpu_comp : 0,
         totem_based ? timers->alg_scatter : 0,
         totem_based ? timers->alg_gather : 0,
         totem_based ? timers->alg_aggr : 0,
         trv_edges,
         (trv_edges / (totem_based ? timers->alg_exec : time_total))/1000000);
  fflush(stdout);
}
