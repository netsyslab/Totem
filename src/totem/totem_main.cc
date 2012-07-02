/**
 * Main entry of the program. Parses command line options as well.
 *
 *  Created on: 2011-02-28
 *  Author: Abdullah Gharaibeh
 */

// totem includes
#include "totem_comdef.h"
#include "totem_graph.h"
#include "totem_mem.h"
#include "totem_util.h"

/**
 * Global variable of program options
 */
PRIVATE options_t options = {
  NULL,  // graph_file
  false  // weighted
};

/**
 * Displays the help message of the program.
 * @param[in] exe_name name of the executable
 * @param[in] exit_err exist error
 */
PRIVATE void display_help(char* exe_name, int exit_err) {
  printf("Usage: %s [options] graph_file\n"
         "\n"
         "Options\n"
         "  -h Print this help message\n"
         "  -w Load edge weights\n",
         exe_name);

  exit(exit_err);
}

/**
 * parses command line options.
 * @param[in] argc number of arguments
 * @param[in] argv argument array
 */
PRIVATE void parse_command_line(int argc, char** argv) {

  optarg = NULL;
  /* template for a new option:
     case 'q': new_option_value = atoi(optarg); break; */
  int ch;
  while(((ch = getopt(argc, argv, "hw")) != EOF)) {
    switch (ch) {
      case 'h':
        display_help(argv[0], 0);
        break;
      case 'w':
        options.weighted = true;
        break;
      case '?':
        fprintf(stderr, "unknown option %c\n", optopt);
        display_help(argv[0], -1);
        break;
      default: assert(0);
    };
  }

  if ((optind != argc - 1)) {
    fprintf(stderr, "missing arguments!\n");
    display_help(argv[0], -1);
  }

  options.graph_file = argv[optind++];
}

/**
 * main entry of the program
 * @param[in] argc number of arguments
 * @param[in] argv argument array
 */
int main(int argc, char** argv) {

  /* Ensure the minimum CUDA architecture is supported */
  if(check_cuda_version() != SUCCESS) {
     exit(EXIT_FAILURE);
  }
  parse_command_line(argc, argv);

  graph_t* graph;
  CALL_SAFE(graph_initialize(options.graph_file, options.weighted,
                             &graph));

  // invoke the graph algorithm here instead e.g., bfs(graph, &options);
  //graph_print(graph);

  error_t err;
  stopwatch_t stopwatch;

  for (int rounds = 1; rounds > 0; rounds--) {
    stopwatch_start(&stopwatch);
    float* rank_cpu = NULL;
    float* rank_cpu_weight = NULL;
    err = page_rank_cpu(graph, rank_cpu_weight, &rank_cpu);
    assert(err == SUCCESS);
    printf("%f\t", stopwatch_elapsed(&stopwatch));

    stopwatch_start(&stopwatch);
    float* rank = NULL;
    float* rank_gpu_weight = NULL;
    err = page_rank_gpu(graph, rank_gpu_weight, &rank);
    assert(err == SUCCESS);
    printf("%f\n", stopwatch_elapsed(&stopwatch));

    for (uint32_t i = 0; i < graph->vertex_count; i++) {
      if (rank[i] != rank_cpu[i]) {
        printf("%d %f %f\n", i, rank[i], rank_cpu[i]);
        assert(0);
      }
    }
    mem_free(rank_cpu);
    mem_free(rank);
  }

  CALL_SAFE(graph_finalize(graph));

  return 0;
}
