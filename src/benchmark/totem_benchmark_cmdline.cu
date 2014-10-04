/**
 * Processes command line arguments
 *
 *  Created on: 2013-02-09
 *  Author: Abdullah Gharaibeh
 */

#include "totem_benchmark.h"

// An options instance that is used to configure the benchmark.
PRIVATE benchmark_options_t options = {
  NULL,                   // Graph file.
  BENCHMARK_BFS,          // Benchmark.
  PLATFORM_CPU,           // Platform.
  1,                      // Number of GPUs.
  omp_get_max_threads(),  // Number of CPU threads.
  omp_sched_guided,       // OMP scheduling.
  1,                      // Repeat.
  50,                     // Alpha.
  PAR_RANDOM,             // Partitioning algorithm.
  GPU_GRAPH_MEM_DEVICE,   // Allocate gpu-based partitions on the device
  false,                  // Do not randomize vertex placement across
                          // GPU partitions.
  false,                  // Vertex ids will not be sorted by edge degree.
  false,                  // Edges will be sorted ascending by default.
};

// A getter for a reference to the benchmark options.
benchmark_options_t* totem_benchmark_get_options() {return &options;}

// Maximum Number of times an experiment is repeated or sources used to
// benchmark a traversal algorithm.
const int REPEAT_MAX = 1000;

/**
 * Displays the help message of the program.
 * @param[in] exe_name name of the executable
 * @param[in] exit_err exist error
 */
PRIVATE void display_help(char* exe_name, int exit_err) {
  printf("Usage: %s [options] graph_file\n"
         "Options\n"
         "  -aNUM [0-100] Percentage of edges allocated to CPU partition "
         "(default 50%%)\n"
         "  -bNUM Benchmark\n"
         "     %d: BFS top-down (default)\n"
         "     %d: PageRank\n"
         "     %d: SSSP\n"
         "     %d: Betweenness\n"
         "     %d: Graph500 top-down\n"
         "     %d: Clustering Coefficient\n"
         "     %d: BFS stepwise\n"
         "     %d: Graph500 stepwise\n"
         "  -e Swaps the direction of edge sorting to be descending order.\n"
         "     (default FALSE)\n"
         "  -gNUM [0-%d] Number of GPUs to use. This is applicable for GPU\n"
         "        and Hybrid platforms only (default 1).\n"
         "  -iNUM Partitioning Algorithm\n"
         "     %d: Random (default)\n"
         "     %d: High degree nodes on CPU\n"
         "     %d: Low degree nodes on CPU\n"
         "  -mNUM Type of memory to use for GPU-based partitions\n"
         "     %d: Device (default)\n"
         "     %d: Host as memory mapped\n"
         "     %d: Only the vertices array on the host\n"
         "     %d: Only the edges array on the host\n"
         "     %d: Edges array partitioned between the device and the host\n"
         "  -o Enables random placement of vertices across GPU partitions\n"
         "     in case of multi-GPU setups (default FALSE)\n"
         "  -pNUM Platform\n"
         "     %d: Execute on CPU only (default)\n"
         "     %d: Execute on GPUs only\n"
         "     %d: Execute on the CPU and on the GPUs\n"
         "  -q Enables mapping by sorted vertex degree during partitioning\n"
         "     (default FALSE)\n"
         "  -rNUM [1-%d] Number of times an experiment is repeated or sources\n"
         "        used to benchmark a traversal algorithm (default 5)\n"
         "  -sNUM OMP scheduling type\n"
         "     %d: static\n"
         "     %d: dynamic\n"
         "     %d: guided (default)\n"
         "  -tNUM [1-%d] Number of CPU threads to use (default %d).\n"
         "  -h Print this help message\n",
         exe_name, BENCHMARK_BFS, BENCHMARK_PAGERANK, BENCHMARK_SSSP,
         BENCHMARK_BETWEENNESS, BENCHMARK_GRAPH500,
         BENCHMARK_CLUSTERING_COEFFICIENT, BENCHMARK_BFS_STEPWISE,
         BENCHMARK_GRAPH500_STEPWISE, get_gpu_count(), PAR_RANDOM,
         PAR_SORTED_ASC, PAR_SORTED_DSC, GPU_GRAPH_MEM_DEVICE,
         GPU_GRAPH_MEM_MAPPED, GPU_GRAPH_MEM_MAPPED_VERTICES,
         GPU_GRAPH_MEM_MAPPED_EDGES, GPU_GRAPH_MEM_PARTITIONED_EDGES,
         PLATFORM_CPU, PLATFORM_GPU, PLATFORM_HYBRID, REPEAT_MAX,
         omp_sched_static, omp_sched_dynamic, omp_sched_guided,
         omp_get_max_threads(), omp_get_max_threads());
  exit(exit_err);
}

/**
 * Parses command line options.
 * @param[in] argc number of arguments
 * @param[in] argv argument array
 */
benchmark_options_t* benchmark_cmdline_parse(int argc, char** argv) {
  optarg = NULL;
  int ch, benchmark, platform, par_algo, gpu_graph_mem;
  while (((ch = getopt(argc, argv, "a:b:eg:i:m:op:qr:s:t:h")) != EOF)) {
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
      case 'e':
        options.edge_sort_dsc = true;
        break;
      case 'g':
        options.gpu_count = atoi(optarg);
        if (options.gpu_count > get_gpu_count() || options.gpu_count < 0) {
          fprintf(stderr, "Invalid number of GPUs %d\n", options.gpu_count);
          display_help(argv[0], -1);
        }
        break;
      case 'i':
        par_algo = atoi(optarg);
        if (par_algo >= PAR_MAX || par_algo < 0) {
          fprintf(stderr, "Invalid partitoining algorithm\n");
          display_help(argv[0], -1);
        }
        options.par_algo = (partition_algorithm_t)par_algo;
        break;
      case 'm':
        gpu_graph_mem = atoi(optarg);
        if (gpu_graph_mem >= GPU_GRAPH_MEM_MAX || gpu_graph_mem < 0) {
          fprintf(stderr, "Invalid GPU graph memory type\n");
          display_help(argv[0], -1);
        }
        options.gpu_graph_mem = (gpu_graph_mem_t)gpu_graph_mem;
        break;
      case 'o':
        options.gpu_par_randomized = true;
        break;
      case 'p':
        platform = atoi(optarg);
        if (platform >= PLATFORM_MAX || platform < 0) {
          fprintf(stderr, "Invalid platform\n");
          display_help(argv[0], -1);
        }
        options.platform = (platform_t)platform;
        break;
      case 'q':
        options.sorted = true;
        break;
      case 'r':
        options.repeat = atoi(optarg);
        if (options.repeat > REPEAT_MAX || options.repeat <= 0) {
          fprintf(stderr, "Invalid repeat argument\n");
          display_help(argv[0], -1);
        }
        break;
      case 's':
        options.omp_sched = (omp_sched_t)atoi(optarg);
        if (options.omp_sched != omp_sched_static &&
            options.omp_sched != omp_sched_dynamic &&
            options.omp_sched != omp_sched_guided) {
          fprintf(stderr, "Invalid OMP scheduling argument %d\n",
                  options.omp_sched);
          display_help(argv[0], -1);
        }
        break;
      case 't':
        options.thread_count = atoi(optarg);
        if (options.thread_count <= 0) {
          fprintf(stderr, "Invalid number of threads\n");
          display_help(argv[0], -1);
        }
        break;
      case 'h':
        display_help(argv[0], 0);
        break;
      default:
        display_help(argv[0], -1);
    }
  }

  if ((optind != argc - 1)) {
    fprintf(stderr, "Missing arguments!\n");
    display_help(argv[0], -1);
  }
  options.graph_file = argv[optind++];

  return &options;
}
