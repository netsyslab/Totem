/**
 * Main entry of the benchmark
 *
 *  Created on: 2011-02-28
 *  Author: Abdullah Gharaibeh
 */

// totem includes
#include "totem_benchmark.h"


// Defines attributes of the algorithms available for benchmarking
PRIVATE void benchmark_bfs(graph_t*, void*, totem_attr_t*);
PRIVATE void benchmark_pagerank(graph_t*, void*, totem_attr_t*);
PRIVATE void benchmark_dijkstra(graph_t*, void*, totem_attr_t*);
PRIVATE void benchmark_betweenness(graph_t*, void*, totem_attr_t*);
const benchmark_attr_t BENCHMARKS[] = {
  {
    benchmark_bfs, 
   "BFS", 
   1, 
   MSG_SIZE_ZERO, 
   sizeof(cost_t)
  },
  {
    benchmark_pagerank, 
    "PAGERANK", 
    MSG_SIZE_ZERO, 
    sizeof(rank_t) * BITS_PER_BYTE, 
    sizeof(rank_t)
  },
  {
    benchmark_dijkstra, 
    "DIJKSTRA", 
    sizeof(weight_t) * BITS_PER_BYTE, 
   MSG_SIZE_ZERO, 
    sizeof(weight_t)
  },
  {
    benchmark_betweenness, 
    "BETWEENNESS", 
    MSG_SIZE_ZERO, 
    MSG_SIZE_ZERO, 
   sizeof(weight_t)
  }
};

// A reference to the options used to configure the benchmark
PRIVATE benchmark_options_t* options = NULL;

PRIVATE const char* PLATFORM_STR[] = {"CPU", "GPU", "HYBRID"};
PRIVATE const char* PAR_ALGO_STR[] = {"RANDOM", "HIGH", "LOW"};
PRIVATE const int SEED = 1985;

/**
 * Prints out detailed timing of a single run
 */
PRIVATE void print_timing(graph_t* graph, double time_total, eid_t trv_edges) {
  bool totem_based = options->platform != PLATFORM_CPU;
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
PRIVATE void print_header(graph_t* graph) {
  const char* OMP_PROC_BIND = getenv("OMP_PROC_BIND");
  printf("file:%s\tbenchmark:%s\tvertices:%llu\tedges:%llu\tpartitioning:%s\t"
         "platform:%s\talpha:%d\trepeat:%d\t"
         "thread_count:%d\tthread_bind:%s", options->graph_file, 
         BENCHMARKS[options->benchmark].str, (uint64_t)graph->vertex_count, 
         (uint64_t)graph->edge_count, PAR_ALGO_STR[options->par_algo], 
         PLATFORM_STR[options->platform], options->alpha, options->repeat, 
         omp_get_max_threads(), 
         OMP_PROC_BIND == NULL ? "NotSet" : OMP_PROC_BIND);
  if (options->platform != PLATFORM_CPU) {
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
 * Returns the number of traversed edges used in computing the processing rate
 */
PRIVATE eid_t get_traversed_edges(graph_t* graph, void* benchmark_output) {
  eid_t trv_edges = 0;
  switch(options->benchmark) {
    case BENCHMARK_BFS:
      OMP(omp parallel for reduction(+ : trv_edges))
      for (vid_t vid = 0; vid < graph->vertex_count; vid++) {
        cost_t* cost = (cost_t*)benchmark_output;
        if (cost[vid] != INF_COST) {
          trv_edges += (graph->vertices[vid + 1] - graph->vertices[vid]);
        }
      }
      break;
    case BENCHMARK_DIJKSTRA:
      OMP(omp parallel for reduction(+ : trv_edges))
      for (vid_t vid = 0; vid < graph->vertex_count; vid++) {
        weight_t* distance = (weight_t*)benchmark_output;
        if (distance[vid] != WEIGHT_MAX) {
          trv_edges += (graph->vertices[vid + 1] - graph->vertices[vid]);
        }
      }
      break;
    case BENCHMARK_PAGERANK:
      trv_edges = graph->edge_count * PAGE_RANK_ROUNDS;
      break;
    case BENCHMARK_BETWEENNESS:
      // The two is for the two phases: forward and backward
      trv_edges = 2 * graph->edge_count * graph->vertex_count;
      break;
    default:
      trv_edges = graph->edge_count;
      break;
  }
  return trv_edges;
}

/**
 * Randomly picks a random source vertex, and ensures that it is connected to at
 * least one other vertex.
 */
PRIVATE vid_t get_random_src(graph_t* graph) {
  vid_t src = rand() % graph->vertex_count;
  while ((graph->vertices[src + 1] - graph->vertices[src] == 0)) {
    src = rand() % graph->vertex_count;
  }
  return src;
}

/**
 * Runs BFS benchmark according to Graph500 specification
 */
PRIVATE void benchmark_bfs(graph_t* graph, void* cost, totem_attr_t* attr) {
  if (options->platform == PLATFORM_CPU) {
    bfs_cpu(graph, get_random_src(graph), (cost_t*)cost);
  } else {
    CALL_SAFE(bfs_hybrid(get_random_src(graph), (cost_t*)cost));
  }
}

/**
 * Runs PageRank benchmark
 */
PRIVATE void benchmark_pagerank(graph_t* graph, void* rank, 
                                totem_attr_t* attr) {
  if (options->platform == PLATFORM_CPU) {
    page_rank_incoming_cpu(graph, NULL, (rank_t*)rank);
  } else {
    CALL_SAFE(page_rank_incoming_hybrid(NULL, (rank_t*)rank));
  }
}

/**
 * Runs Dijkstra benchmark
 */
PRIVATE void benchmark_dijkstra(graph_t* graph, void* distance,
                                totem_attr_t* attr) {
  if (options->platform == PLATFORM_CPU) {
    dijkstra_cpu(graph, get_random_src(graph), (weight_t*)distance);
  } else {
    assert(false);
  }
  mem_free(distance);
}

/**
 * Runs Betweenness Centrality benchmark
 */
PRIVATE void benchmark_betweenness(graph_t* graph, void* betweenness_score, 
                                   totem_attr_t* attr) {
  if (options->platform == PLATFORM_CPU) {
    betweenness_cpu(graph, (weight_t*)betweenness_score);
  } else {
      assert(false);
  }
  mem_free(betweenness_score);
}

/**
 * The main execution loop of the benchmark
 */
PRIVATE void benchmark_run() {
  assert(options);
  graph_t* graph = NULL;
  CALL_SAFE(graph_initialize(options->graph_file, 
                             (options->benchmark == BENCHMARK_DIJKSTRA),
                             &graph));
  srand(SEED);
  void* benchmark_state = mem_alloc(graph->vertex_count * 
                                    BENCHMARKS[options->benchmark].output_size);
  assert(benchmark_state || (BENCHMARKS[options->benchmark].output_size == 0));
  totem_attr_t attr = TOTEM_DEFAULT_ATTR;
  if (options->platform != PLATFORM_CPU) {
    attr.par_algo = options->par_algo;
    attr.cpu_par_share = (float)options->alpha / 100.0;
    attr.platform = options->platform;
    attr.gpu_count = options->gpu_count;
    attr.push_msg_size = BENCHMARKS[options->benchmark].push_msg_size;
    attr.pull_msg_size = BENCHMARKS[options->benchmark].pull_msg_size;
    CALL_SAFE(totem_init(graph, &attr));
  }
  print_header(graph);

  for (int s = 0; s < options->repeat; s++) {
    stopwatch_t stopwatch;
    stopwatch_start(&stopwatch);
    BENCHMARKS[options->benchmark].func(graph, benchmark_state, &attr);
    print_timing(graph, stopwatch_elapsed(&stopwatch), 
                 get_traversed_edges(graph, benchmark_state));
  }

  if (options->platform != PLATFORM_CPU) {
    totem_finalize();
  }
  mem_free(benchmark_state);
  CALL_SAFE(graph_finalize(graph));
}

/**
 * The main entry of the program
 */
int main(int argc, char** argv) {
  CALL_SAFE(check_cuda_version());
  options = benchmark_cmdline_parse(argc, argv);
  benchmark_run();
  return 0;
}
