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
  BENCHMARK_MAX
} benchmark_t;
PRIVATE const char* BENCHMARK_STR[] = {
  "BFS",
  "PAGERANK",
  "DIJKSTRA"
};
// forward declaration of benchmark functions
typedef void(*benchmark_func_t)(graph_t*, totem_attr_t*);
PRIVATE void benchmark_bfs(graph_t* graph, totem_attr_t* attr);
PRIVATE void benchmark_pagerank(graph_t* graph, totem_attr_t* attr);
PRIVATE void benchmark_dijkstra(graph_t* graph, totem_attr_t* attr);
PRIVATE const benchmark_func_t BENCHMARK_FUNC[] = {
  benchmark_bfs,
  benchmark_pagerank,
  benchmark_dijkstra
};
PRIVATE const size_t BENCHMARK_MSG_SIZE[] = {
  sizeof(uint32_t),
  sizeof(float),
  sizeof(weight_t)
};

//Command line options
typedef struct options_s {
  char*                 graph_file;
  benchmark_t           benchmark;
  platform_t            platform;
  int                   repeat;   
  int                   src_count;
  bool                  verify;
  int                   alpha;
  partition_algorithm_t par_algo;
  // TODO(abdullah): add an option to have the src picked randomely
} options_t;
PRIVATE options_t options = {
  NULL,                         // graph_file
  BENCHMARK_BFS,                // benchmark 
  PLATFORM_CPU,                 // platform
  1,                            // repeat
  2,                            // src_count
  false,                        // verify
  50,                           // alpha
  PAR_RANDOM                    // partitioning algorithm
};

// Misc global constants
PRIVATE const int REPEAT_MAX = 20; 
PRIVATE const char* PLATFORM_STR[] = {
  "CPU",
  "GPU",
  "MULTI_GPU",
  "HYBRID",
  "ALL"
};
PRIVATE const char* PAR_ALGO_STR[] = {
  "RANDOM",
  "SORTED_ASC",
  "SORTED_DSC"
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
         "  -lNUM Partitioning Algorithm\n"
         "     %d: Random (default)\n"
         "     %d: Sorted Ascending\n"
         "     %d: Sorted Dscending\n"
         "  -pNUM Platform\n"
         "     %d: Execute on the CPU only (default)\n"
         "     %d: Execute on the GPU only\n"
         "     %d: Execute on all available GPUs\n"
         "     %d: Execute on the CPU and one GPU\n"
         "     %d: Execute on all available processors (CPU and all GPUs)\n"
         "  -rNUM [1-%d] Number of times an experiment is repeated"
         " (default 1)\n"
         "  -sNUM Number sources used to benchmark a traversal algorithm"
         " (default 2)\n"
         "  -v Verify benchmark result\n"
         "  -h Print this help message\n",
         exe_name, BENCHMARK_BFS, BENCHMARK_PAGERANK, BENCHMARK_DIJKSTRA, 
         PAR_RANDOM, PAR_SORTED_ASC, PAR_SORTED_DSC, PLATFORM_CPU, 
         PLATFORM_GPU, PLATFORM_MULTI_GPU, PLATFORM_HYBRID, PLATFORM_ALL, 
         REPEAT_MAX);
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
  while(((ch = getopt(argc, argv, "ha:b:l:p:r:s:v")) != EOF)) {
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
      case 'l':
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
      case 'r':
        options.repeat = atoi(optarg);
        if (options.repeat > REPEAT_MAX || options.repeat <= 0) {
          fprintf(stderr, "Invalid repeat argument\n");
          display_help(argv[0], -1);
        }
        break;
      case 's':
        options.src_count = atoi(optarg);
        if (options.src_count <= 0) {
          fprintf(stderr, "Invalid number of traversal sources\n");
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

void print_timing(graph_t* graph, double time_total) {
  printf("time_total:%0.2f\t"
         "time_exec:%0.2f\t"
         "time_comp:%0.2f\t"
         "time_gpu_comp:%0.2f\t"
         "time_comm:%0.2f\n",
         time_total,
         totem_time_execution(),
         totem_time_computation(),
         totem_time_gpu_computation(),
         totem_time_communication());
  fflush(stdout);
}

void print_header(graph_t* graph, totem_attr_t* attr) {
  printf("file:%s\tbenchmark:%s\tpartitioning:%s\tplatform:%s\talpha:%d\t"
         "repeat:%d\tsrc_count:%d\tverify:%s\t", 
         options.graph_file, BENCHMARK_STR[options.benchmark],
         PAR_ALGO_STR[options.par_algo], PLATFORM_STR[options.platform],
         options.alpha, options.repeat, options.src_count,
         options.verify? "true" : "false");
  // print graph and partitioning characteristics
  id_t rv = 0; id_t re = 0;
  for (id_t pid = 0; pid < totem_partition_count(); pid++) {
    double v = 100.0 * ((double)totem_par_vertex_count(pid) / 
                       (double)graph->vertex_count);
    double e = 100.0 * ((double)totem_par_edge_count(pid) / 
                       (double)graph->edge_count);
    rv += totem_par_rmt_vertex_count(pid);
    re += totem_par_rmt_edge_count(pid);
    printf("pid%d:(%0.0f,%0.0f)\t", pid, v, e);
  }
  printf("vertices:%lld\tedges:%lld\trmt_vertex:%0.0f\trmt_edge:%0.0f\t"
         "beta:%0.0f\t", graph->vertex_count, graph->edge_count,
         100.0*(double)((double)rv/(double)graph->vertex_count),
         100.0*(double)((double)re/(double)graph->edge_count),
         100.0*(double)((double)rv/(double)graph->edge_count));
  printf("time_init:%0.2f\ttime_par:%0.2f\n",
         totem_time_initialization(), totem_time_partitioning());
}

/**
 * Verfies BFS result by comparing it with bfs_cpu (assuming it is correct)
 */
PRIVATE void verify_bfs(graph_t* graph, id_t src_id, uint32_t* cost) {
  uint32_t* cost_ref;
  bfs_cpu(graph, src_id, &cost_ref);
  for (id_t v = 0; v < graph->vertex_count; v++) {
    assert(cost[v] == cost_ref[v]);
  }
  mem_free(cost_ref);
}

PRIVATE id_t get_random_src(graph_t* graph) {
  srand(time(NULL));
  id_t src = rand() % graph->vertex_count;
  while ((graph->vertices[src + 1] - graph->vertices[src] == 0)) {
    src = rand() % graph->vertex_count;
  }
  return src;
}

/**
 * Runs BFS benchmark according to Graph500 specification
 */
PRIVATE void benchmark_bfs(graph_t* graph, totem_attr_t* attr) {
  // 64 randomly chosen seeds (Graph500 benchmark)
  for (int s = 0; s < options.src_count; s++) {
    // make sure the source id is connected
    id_t src = get_random_src(graph);
    stopwatch_t stopwatch;
    stopwatch_start(&stopwatch);
    uint32_t* cost = NULL;
    CALL_SAFE(bfs_hybrid(src, &cost));
    print_timing(graph, stopwatch_elapsed(&stopwatch));
    if (options.verify) verify_bfs(graph, src, cost);
    mem_free(cost);
  }
}

/**
 * Runs PageRank benchmark
 */
PRIVATE void benchmark_pagerank(graph_t* graph, totem_attr_t* attr) {
  stopwatch_t stopwatch;
  stopwatch_start(&stopwatch);
  float* rank = NULL;
  CALL_SAFE(page_rank_hybrid(NULL, &rank));
  print_timing(graph, stopwatch_elapsed(&stopwatch));
  mem_free(rank);
}

/**
 * Runs PageRank benchmark
 */
PRIVATE void benchmark_dijkstra(graph_t* graph, totem_attr_t* attr) {
  // TODO(abdullah): Implement this
  printf("Dijkstra benchmark not yet implemented!\n");
  exit(-1);
}

/**
 * The main execution loop of the benchmark
 */
PRIVATE void run_benchmark() {
  graph_t* graph = NULL; 
  CALL_SAFE(graph_initialize(options.graph_file, 
                             (options.benchmark == BENCHMARK_DIJKSTRA),
                             &graph));
  totem_attr_t attr = TOTEM_DEFAULT_ATTR;
  attr.par_algo = options.par_algo;
  attr.cpu_par_share = (float)options.alpha / 100.0;
  attr.platform = options.platform;
  attr.msg_size = BENCHMARK_MSG_SIZE[options.benchmark];
  CALL_SAFE(totem_init(graph, &attr));
  print_header(graph, &attr);
  for (int round = 0; round < options.repeat; round++) {
    BENCHMARK_FUNC[options.benchmark](graph, &attr);
  }
  totem_finalize();
  CALL_SAFE(graph_finalize(graph)); 
}

/**
 * The main entry of the program
 */
int main(int argc, char** argv) {
  parse_command_line(argc, argv);
  /* Ensure the minimum CUDA architecture is supported */
  CALL_SAFE(check_cuda_version());
  run_benchmark();
  return 0;
}
