/**
 * Main entry of the program. Parses command line options as well.
 *
 *  Created on: 2011-02-28
 *  Author: Abdullah Gharaibeh
 */

// totem includes
#include "totem.h"
#include "totem_mem.h"
#include "totem_util.h"

// Available benchmarks
typedef enum {
  BENCHMARK_BFS = 0,
  BENCHMARK_PAGERANK,
  BENCHMARK_DIJKSTRA,
  BENCHMARK_BETWEENNESS,
  BENCHMARK_MAX
} benchmark_t;
PRIVATE const char* BENCHMARK_STR[] = {
  "BFS",
  "PAGERANK",
  "DIJKSTRA",
  "BETWEENNESS"
};
// forward declaration of benchmark functions
typedef void(*benchmark_func_t)(graph_t*, totem_attr_t*);
PRIVATE void benchmark_bfs(graph_t* graph, totem_attr_t* attr);
PRIVATE void benchmark_pagerank(graph_t* graph, totem_attr_t* attr);
PRIVATE void benchmark_dijkstra(graph_t* graph, totem_attr_t* attr);
PRIVATE void benchmark_betweenness(graph_t* graph, totem_attr_t* attr);
PRIVATE const benchmark_func_t BENCHMARK_FUNC[] = {
  benchmark_bfs,
  benchmark_pagerank,
  benchmark_dijkstra,
  benchmark_betweenness
};
PRIVATE const size_t BENCHMARK_PUSH_MSG_SIZE[] = {
  1,
  MSG_SIZE_ZERO,
  sizeof(weight_t) * BITS_PER_BYTE
};
PRIVATE const size_t BENCHMARK_PULL_MSG_SIZE[] = {
  MSG_SIZE_ZERO,
  sizeof(float)    * BITS_PER_BYTE,
  MSG_SIZE_ZERO
};

//Command line options
typedef struct options_s {
  char*                 graph_file; // The file to run the benchmark on
  benchmark_t           benchmark;  // Benchmark to run
  platform_t            platform;   // Execution platform
  int                   gpu_count;  // Number of GPUs to use for hybrid and
                                    // GPU-only platforms
  int                   repeat;     // Number of times to repeat an execution
                                    // (for traversal algorithms, number of 
                                    // sources used)
  bool                  verify;     // Verify the results
  int                   alpha;      // Percentage of edges left on the CPU for 
                                    // hybrid platforms
  partition_algorithm_t par_algo;   // Partitioning algorithm
} options_t;

// A global options_t instance that is used to configure the benchmark
PRIVATE options_t options = {
  NULL,           // graph_file
  BENCHMARK_BFS,  // benchmark 
  PLATFORM_CPU,   // platform
  1,              // number of GPUs
  5,              // repeat
  false,          // verify
  50,             // alpha
  PAR_RANDOM      // partitioning algorithm
};

// Misc global variables
PRIVATE int max_gpu_count = 0;
PRIVATE const int REPEAT_MAX = 100;
PRIVATE const int SEED = 1985;
PRIVATE const char* PLATFORM_STR[] = {
  "CPU",
  "GPU",
  "HYBRID"
};

PRIVATE const char* PAR_ALGO_STR[] = {
  "RANDOM",
  "HIGH",
  "LOW"
};

/**
 * Displays the help message of the program.
 * @param[in] exe_name name of the executable
 * @param[in] exit_err exist error
 */
PRIVATE void display_help(char* exe_name, int exit_err) {
  printf("Usage: %s [options] graph_file\n"
         "Options\n"
         "  -aNUM [0-100] Percentage of edges in CPU partition (default 50\%)\n"
         "  -bNUM Benchmark\n"
         "     %d: BFS (default)\n"
         "     %d: PageRank\n"
         "     %d: Dijkstra\n"
         "     %d: Betweenness\n"
         "  -pNUM Platform\n"
         "     %d: Execute on CPU only (default)\n"
         "     %d: Execute on GPUs only\n"
         "     %d: Execute on the CPU and on the GPUs\n"
         "  -gNUM [0-%d] Number of GPUs to use. This is applicable for GPU\n"
         "         and Hybrid platforms only (default 1).\n"
         "  -tNUM Partitioning Algorithm\n"
         "     %d: Random (default)\n"
         "     %d: High degree nodes on CPU\n"
         "     %d: Low degree nodes on CPU\n"
         "  -rNUM [1-%d] Number of times an experiment is repeated or sources\n"
         "         used to benchmark a traversal algorithm (default 5)\n"
         "  -v Verify benchmark result\n"
         "  -h Print this help message\n",
         exe_name, BENCHMARK_BFS, BENCHMARK_PAGERANK, BENCHMARK_DIJKSTRA, 
         BENCHMARK_BETWEENNESS, PLATFORM_CPU, PLATFORM_GPU, PLATFORM_HYBRID,
         max_gpu_count, PAR_RANDOM, PAR_SORTED_ASC, PAR_SORTED_DSC, REPEAT_MAX);
  exit(exit_err);
}

/**
 * Parses command line options.
 * @param[in] argc number of arguments
 * @param[in] argv argument array
 */
PRIVATE void parse_command_line(int argc, char** argv) {
  optarg = NULL;
  int ch, benchmark, platform, par_algo;
  while(((ch = getopt(argc, argv, "ha:b:t:p:g:r:v")) != EOF)) {
    switch (ch) {
      case 'a':
        options.alpha = atoi(optarg);
        if (options.alpha > 100 || options.alpha < 0) {
          fprintf(stderr, "Invalid alpha value\n");
          display_help(argv[0], -1);
        }
        break;
      case 'b':
        benchmark = atoi(optarg);
        if (benchmark >= BENCHMARK_MAX || benchmark < 0) {
          fprintf(stderr, "Invalid benchmark\n");
          display_help(argv[0], -1);
        }
        options.benchmark = (benchmark_t)benchmark;
        break;
      case 't':
        par_algo = atoi(optarg);
        if (par_algo >= PAR_MAX || par_algo < 0) {
          fprintf(stderr, "Invalid partitoining algorithm\n");
          display_help(argv[0], -1);
        }
        options.par_algo = (partition_algorithm_t)par_algo;
        break;
      case 'p':
        platform = atoi(optarg);
        if (platform >= PLATFORM_MAX || platform < 0) {
          fprintf(stderr, "Invalid platform\n");
          display_help(argv[0], -1);
        }
        options.platform = (platform_t)platform;
        break;
      case 'g':
        options.gpu_count = atoi(optarg);
        if (options.gpu_count > max_gpu_count || options.gpu_count < 0) {
          fprintf(stderr, "Invalid number of GPUs %d\n", options.gpu_count);
          display_help(argv[0], -1);
        }
        break;
      case 'r':
        options.repeat = atoi(optarg);
        if (options.repeat > REPEAT_MAX || options.repeat <= 0) {
          fprintf(stderr, "Invalid repeat argument\n");
          display_help(argv[0], -1);
        }
        break;
      case 'v':
        options.verify = true;
        break;
      case 'h':
        display_help(argv[0], 0);
        break;
      default: 
        display_help(argv[0], -1);
    };
  }
  if ((optind != argc - 1)) {
    fprintf(stderr, "Missing arguments!\n");
    display_help(argv[0], -1);
  }
  options.graph_file = argv[optind++];
}

void print_timing(graph_t* graph, double time_total, bool totem_based, 
                  const char* str) {
  const totem_timing_t* timers = totem_timing();
  printf("time_total:%0.2f\t"
         "time_exec:%0.2f\t"
         "time_comp:%0.2f\t"
         "time_gpu_comp:%0.2f\t"
         "time_comm:%0.2f\t"
         "time_scatter:%0.2f\t"
         "time_aggr:%0.2f",
         time_total,
         totem_based ? timers->alg_exec : time_total,
         totem_based ? timers->alg_comp : time_total,
         totem_based ? timers->alg_gpu_comp : 0,
         totem_based ? timers->alg_comm : 0,
         totem_based ? timers->alg_scatter : 0,
         totem_based ? timers->alg_aggr : 0);
  if (str) printf("\t%s", str);
  printf("\n");
  fflush(stdout);
}

void print_header(graph_t* graph, bool totem_based) {
  const char* OMP_PROC_BIND = getenv("OMP_PROC_BIND");
  printf("file:%s\tbenchmark:%s\tvertices:%llu\tedges:%llu\tpartitioning:%s\t"
         "platform:%s\talpha:%d\trepeat:%d\tverify:%s\t"
         "thread_count:%d\tthread_bind:%s", options.graph_file, 
         BENCHMARK_STR[options.benchmark], (uint64_t)graph->vertex_count, 
         (uint64_t)graph->edge_count, PAR_ALGO_STR[options.par_algo], 
         PLATFORM_STR[options.platform], options.alpha, options.repeat, 
         options.verify? "true" : "false", 
         omp_get_max_threads(), 
         OMP_PROC_BIND == NULL ? "NotSet" : OMP_PROC_BIND);
  if (totem_based) {
    printf("\t");
    // print graph and partitioning characteristics
    uint64_t rv = 0; uint64_t re = 0;
    for (uint32_t pid = 0; pid < totem_partition_count(); pid++) {
      double v = 100.0 * ((double)totem_par_vertex_count(pid) / 
                          (double)graph->vertex_count);
      double e = 100.0 * ((double)totem_par_edge_count(pid) / 
                          (double)graph->edge_count);
      rv += totem_par_rmt_vertex_count(pid);
      re += totem_par_rmt_edge_count(pid);
      // print for each partition the number and percentation of 
      // remote vertices and edges
      printf("pid%d:(%0.0f,%0.0f)\trmt%d:(%0.0f,%0.0f)\t", pid, v, e, pid, 
             100.0 * ((double)totem_par_rmt_vertex_count(pid) / 
                      (double)graph->vertex_count),
             100.0 * ((double)totem_par_rmt_edge_count(pid) / 
                      (double)graph->edge_count));
    }
    // print the total percentage of remote edges and vertices
    printf("rmt_vertex:%0.0f\trmt_edge:%0.0f\tbeta:%0.0f\t", 
           100.0*(double)((double)rv/(double)graph->vertex_count),
           100.0*(double)((double)re/(double)graph->edge_count),
           100.0*(double)((double)rv/(double)graph->edge_count));
    // print the time spent on initializing Totem and partitioning the graph
    const totem_timing_t* timers = totem_timing();
    printf("time_init:%0.2f\ttime_par:%0.2f",
           timers->engine_init, timers->engine_par);
  }
  printf("\n"); fflush(stdout);
}

/**
 * Verfies BFS result by comparing it with bfs_cpu (assuming it is correct)
 */
PRIVATE void verify_bfs(graph_t* graph, vid_t src_id, uint32_t* cost) {
  uint32_t* cost_ref = (uint32_t*)malloc(graph->vertex_count * 
                                         sizeof(uint32_t));
  bfs_cpu(graph, src_id, cost_ref);
  for (vid_t v = 0; v < graph->vertex_count; v++) {
    assert(cost[v] == cost_ref[v]);
  }
  free(cost_ref);
}

PRIVATE vid_t get_traversed_edges(graph_t* graph, uint32_t* cost) {
  eid_t trv_edges = 0;
  OMP(omp parallel for reduction(+ : trv_edges))
  for (vid_t vid = 0; vid < graph->vertex_count; vid++) {
    if (cost[vid] != INFINITE) {
      trv_edges += (graph->vertices[vid + 1] - graph->vertices[vid]);
    }
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
PRIVATE void benchmark_bfs(graph_t* graph, totem_attr_t* attr) {
  srand(SEED);
  uint32_t* cost = (uint32_t*)mem_alloc(graph->vertex_count * sizeof(uint32_t));
  assert(cost);
  for (int s = 0; s < options.repeat; s++) {
    vid_t src = get_random_src(graph);
    stopwatch_t stopwatch;
    stopwatch_start(&stopwatch);
    if (options.platform == PLATFORM_CPU) {
      bfs_cpu(graph, src, cost);
    } else {
      CALL_SAFE(bfs_hybrid(src, cost));
    }
    double total_time = stopwatch_elapsed(&stopwatch);
    if (options.verify) verify_bfs(graph, src, cost);
    char buffer[100];
    sprintf(buffer, "seed:%u\ttrv_edges:%llu", src,
            get_traversed_edges(graph, cost));
    print_timing(graph, total_time, options.platform != PLATFORM_CPU, buffer);
  }
  mem_free(cost);
}

/**
 * Runs PageRank benchmark
 */
PRIVATE void benchmark_pagerank(graph_t* graph, totem_attr_t* attr) {
  float* rank = (float*)mem_alloc(graph->vertex_count * sizeof(float));
  assert(rank);
  for (int round = 0; round < options.repeat; round++) {
    stopwatch_t stopwatch;
    stopwatch_start(&stopwatch);
    if (options.platform == PLATFORM_CPU) {
      page_rank_incoming_cpu(graph, NULL, rank);
    } else {
      CALL_SAFE(page_rank_incoming_hybrid(NULL, rank));
    }
    print_timing(graph, stopwatch_elapsed(&stopwatch), 
                 options.platform != PLATFORM_CPU, NULL);  
  }
  mem_free(rank);
}

/**
 * Runs PageRank benchmark
 */
PRIVATE void benchmark_dijkstra(graph_t* graph, totem_attr_t* attr) {
  srand(SEED);
  weight_t* distance = (weight_t*)mem_alloc(graph->vertex_count * sizeof(weight_t));
  assert(distance);
  for (int s = 0; s < options.repeat; s++) {
    vid_t src = get_random_src(graph);
    stopwatch_t stopwatch;
    stopwatch_start(&stopwatch);
    if (options.platform == PLATFORM_CPU) {
      dijkstra_cpu(graph, src, distance);
    } else {
      assert(false);
    }
    print_timing(graph, stopwatch_elapsed(&stopwatch),
                 options.platform != PLATFORM_CPU, NULL);
  }
  mem_free(distance);
}

/**
 * Runs Betweenness Centrality benchmark
 */
PRIVATE void benchmark_betweenness(graph_t* graph, totem_attr_t* attr) {
  for (int s = 0; s < options.repeat; s++) {
    stopwatch_t stopwatch;
    stopwatch_start(&stopwatch);
    weight_t* centrality_score = NULL;
    if (options.platform == PLATFORM_CPU) {
      betweenness_cpu(graph, &centrality_score);
    } else {
      assert(false);
    }
    print_timing(graph, stopwatch_elapsed(&stopwatch),
                 options.platform != PLATFORM_CPU, NULL);
    mem_free(centrality_score);
  }
}

/**
 * The main execution loop of the benchmark
 */
PRIVATE void run_benchmark() {
  graph_t* graph = NULL; 
  CALL_SAFE(graph_initialize(options.graph_file, 
                             (options.benchmark == BENCHMARK_DIJKSTRA),
                             &graph));
  if (options.platform == PLATFORM_CPU) {
    print_header(graph, false);
    BENCHMARK_FUNC[options.benchmark](graph, NULL);    
  } else {
    totem_attr_t attr = TOTEM_DEFAULT_ATTR;
    attr.par_algo = options.par_algo;
    attr.cpu_par_share = (float)options.alpha / 100.0;
    attr.platform = options.platform;
    attr.gpu_count = options.gpu_count;
    attr.push_msg_size = BENCHMARK_PUSH_MSG_SIZE[options.benchmark];
    attr.pull_msg_size = BENCHMARK_PULL_MSG_SIZE[options.benchmark];
    CALL_SAFE(totem_init(graph, &attr));
    print_header(graph, true);
    BENCHMARK_FUNC[options.benchmark](graph, &attr);
    totem_finalize();
  }
  CALL_SAFE(graph_finalize(graph));
}

/**
 * The main entry of the program
 */
int main(int argc, char** argv) {
  CALL_SAFE(check_cuda_version());
  max_gpu_count = get_gpu_count();
  parse_command_line(argc, argv);
  run_benchmark();
  return 0;
}
