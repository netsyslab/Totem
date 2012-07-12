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

//Command line options
typedef struct options_s {
  char*       graph_file;
  benchmark_t benchmark;
  platform_t  platform;
  int         repeat;   
  int         src_count;
  bool        verify;
  // TODO(abdullah): add an option to set a set of values for alpha
  // TODO(abdullah): add an option to have the src picked randomely
} options_t;
PRIVATE options_t options = {
  NULL,           // graph_file
  BENCHMARK_BFS,  // benchmark 
  PLATFORM_CPU,   // platform
  2,              // repeat
  2,              // src_count
  false,          // verify
};

// Misc global constants
PRIVATE const int REPEAT_MAX = 20; 
PRIVATE const char* PLATFORM_STR[] = {
  "CPU",
  "GPU",
  "MULTI_GPU",
  "HYBRID",
  "ALL",
  "SWEEP"
};
PRIVATE const id_t TRAVERSAL_SRCS[] = {
  17038, 8400, 30763, 32516, 32616, 15288, 24607, 21964, 3210, 26998, 29826, 
  8974, 2910, 31634, 27803, 26243, 23980, 5662, 12912, 5281, 1689, 30258, 3266,
  21457, 25796, 16182, 10634, 29690, 32226, 15196, 24864, 15140, 3424, 28402, 
  28646, 17363, 11764, 14022, 9992, 21697, 602, 7588, 13784, 18309, 12392, 
  14865, 666, 4932, 296, 5700, 27562, 3233, 24072, 6302, 19331, 16594, 29761, 
  1521, 17823, 29479, 10310, 17593, 2763, 23799
};
PRIVATE const int TRAVERSAL_SRC_COUNT = sizeof(TRAVERSAL_SRCS)/sizeof(id_t);

/**
 * Displays the help message of the program.
 * @param[in] exe_name name of the executable
 * @param[in] exit_err exist error
 */
PRIVATE void display_help(char* exe_name, int exit_err) {
  printf("Usage: %s [options] graph_file\n"
         "Options\n"
         "  -bNUM Benchmark\n"
         "     %d: BFS (default)\n"
         "     %d: PageRank\n"
         "     %d: Dijkstra\n"
         "  -pNUM Platform\n"
         "     %d: Execute on the CPU only (default)\n"
         "     %d: Execute on the GPU only\n"
         "     %d: Execute on all available GPUs\n"
         "     %d: Execute on the CPU and one GPU\n"
         "     %d: Execute on all available processors (CPU and all GPUs)\n"
         "     %d: Parameter sweep (run on all platform options)\n"
         "  -rNUM [1-%d] Number of times an experiment is repeated"
         " (default 2)\n"
         "  -sNUM [1-%d] Number sources used to benchmark a traversal algorithm"
         " (default 2)\n"
         "  -v Verify benchmark result\n"
         "  -h Print this help message\n",
         exe_name, BENCHMARK_BFS, BENCHMARK_PAGERANK, BENCHMARK_DIJKSTRA, 
         PLATFORM_CPU, PLATFORM_GPU, PLATFORM_MULTI_GPU, PLATFORM_HYBRID, 
         PLATFORM_ALL, PLATFORM_MAX, REPEAT_MAX, TRAVERSAL_SRC_COUNT);
  exit(exit_err);
}

/**
 * Parses command line options.
 * @param[in] argc number of arguments
 * @param[in] argv argument array
 */
PRIVATE void parse_command_line(int argc, char** argv) {
  optarg = NULL;
  int ch, benchmark, platform;
  while(((ch = getopt(argc, argv, "hb:p:r:s:v")) != EOF)) {
    switch (ch) {
      case 'b':
        benchmark = atoi(optarg);
        if (benchmark >= BENCHMARK_MAX || benchmark < 0) {
          fprintf(stderr, "Invalid benchmark\n");
          display_help(argv[0], -1);
        }
        options.benchmark = (benchmark_t)benchmark;
        break;
      case 'p':
        platform = atoi(optarg);
        if (platform > PLATFORM_MAX || platform < 0) {
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
        if (options.src_count > TRAVERSAL_SRC_COUNT || options.src_count <= 0) {
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

void print_stat(graph_t* graph, totem_attr_t* attr, double time_total) {
  // print attributes
  printf("benchmark:%s\t"
         "platform:%s\t"
         "alpha:%d\t", 
         BENCHMARK_STR[options.benchmark],
         PLATFORM_STR[attr->platform],
         (int)(attr->cpu_par_share * 100.0));

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
  printf("vertices:%lld\t"
         "edges:%lld\t"
         "rmt_vertex:%0.0f\t"
         "rmt_edge:%0.0f\t"
         "beta:%0.0f\t",
         graph->vertex_count,
         graph->edge_count,
         100.0*(double)((double)rv/(double)graph->vertex_count),
         100.0*(double)((double)re/(double)graph->edge_count),
         100.0*(double)((double)rv/(double)graph->edge_count));

  // print timing
  printf("time_total:%0.2f\t"
         "time_init:%0.2f\t"
         "time_par:%0.2f\t"
         "time_exec:%0.2f\t"
         "time_comp:%0.2f\t"
         "time_gpu_comp:%0.2f\t"
         "time_comm:%0.2f\n",
         time_total,
         totem_time_initialization(),
         totem_time_partitioning(),
         totem_time_execution(),
         totem_time_computation(),
         totem_time_gpu_computation(),
         totem_time_communication());
  fflush(stdout);
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

/**
 * Runs BFS benchmark according to Graph500 specification
 */
PRIVATE void benchmark_bfs(graph_t* graph, totem_attr_t* attr) {
  // 64 randomly chosen seeds (Graph500 benchmark)
  for (int s = 0; s < options.src_count; s++) {
    // make sure the source id is valid
    id_t src_id = TRAVERSAL_SRCS[s] % graph->vertex_count;
    stopwatch_t stopwatch;
    stopwatch_start(&stopwatch);
    uint32_t* cost = NULL;
    CALL_SAFE(bfs_hybrid(src_id, &cost));
    print_stat(graph, attr, stopwatch_elapsed(&stopwatch));
    if (options.verify) verify_bfs(graph, src_id, cost);
    mem_free(cost);
  }
}

/**
 * Runs PageRank benchmark
 */
PRIVATE void benchmark_pagerank(graph_t* graph, totem_attr_t* attr) {
  // TODO(abdullah): Implement this
  printf("PageRank benchmark not yet implemented!\n");
  exit(-1);
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
 * Identifies the platforms where the benchmark will be run
 */
PRIVATE void get_platforms(platform_t** platforms, int* platform_count) {
  if (options.platform == PLATFORM_MAX) {
    *platform_count = PLATFORM_MAX;
    *platforms = (platform_t*)calloc(sizeof(platform_t), *platform_count);
    for (int p = PLATFORM_CPU; p < PLATFORM_MAX; p++) {
      (*platforms)[p] = (platform_t)p;
    }
    return;
  }
  *platform_count = 1;
  *platforms = (platform_t*)calloc(sizeof(platform_t), *platform_count);
  (*platforms)[0] = options.platform;
}

/**
 * The main execution loop of the benchmark
 */
PRIVATE void run_benchmark() {
  graph_t* graph = NULL; 
  CALL_SAFE(graph_initialize(options.graph_file, 
                             (options.benchmark == BENCHMARK_DIJKSTRA),
                             &graph));
  platform_t* platforms = NULL;
  int platform_count = 0;
  get_platforms(&platforms, &platform_count);
  assert(platforms && platform_count);
  totem_attr_t attr = TOTEM_DEFAULT_ATTR;
  for (int p = 0; p < platform_count; p++) {
    attr.platform = platforms[p];
    attr.cpu_par_share = .5;
    CALL_SAFE(totem_init(graph, &attr));
    for (int round = 0; round < options.repeat; round++) {
      BENCHMARK_FUNC[options.benchmark](graph, &attr);
    }
    totem_finalize();
  }
  free(platforms);
  CALL_SAFE(graph_finalize(graph));
}

/**
 * The main entry of the program
 */
int main(int argc, char** argv) {
  parse_command_line(argc, argv);
  printf("file: %s\t"
         "benchmark:%s\t"
         "platform:%s\t"
         "repeat:%d\t"
         "src_count:%d\t"
         "verify:%s\n", 
         options.graph_file, 
         BENCHMARK_STR[options.benchmark],
         PLATFORM_STR[options.platform],
         options.repeat,
         options.src_count,
         options.verify? "true" : "false");
  /* Ensure the minimum CUDA architecture is supported */
  CALL_SAFE(check_cuda_version());
  run_benchmark();
  return 0;
}
