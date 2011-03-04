/**
 * Main entry of the program. Parses command line options as well.
 *
 *  Created on: 2011-02-28
 *  Author: Abdullah Gharaibeh
 */

// totem includes
#include "totem_comdef.h"
#include "totem_graph.h"

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
 * prints out a graph in totem format (written to test the parser)
 * @param[in] graph the graph data structure to print out
 */
PRIVATE void print_graph(graph_t* graph) {   
  for (uint32_t vid = 0; vid < graph->vertices_count; vid++) {    
    int neighbors = graph->vertices[vid + 1] - graph->vertices[vid];
    uint32_t* edges = &(graph->edges[graph->vertices[vid]]);
    for (int i = 0; i < neighbors; i++) {
      fprintf(stdout, "%d %d\n", vid, edges[i]);
    }
  }
}

/**
 * main entry of the program
 * @param[in] argc number of arguments
 * @param[in] argv argument array
 */
int main(int argc, char** argv) {

  parse_command_line(argc, argv);

  graph_t* graph;
  CALL_SAFE(graph_initialize(options.graph_file, options.weighted, 
                             &graph));

  // invoke the graph algorithm here instead e.g., bfs(graph, &options);
  print_graph(graph);

  CALL_SAFE(graph_finalize(graph));

  return 0;
}
