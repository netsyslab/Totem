/**
 * Main entry of the graph generation program.
 *
 *  Created on: 2014-07-28
 *  Author: Abdullah Gharaibeh
 */

// system includes
#include <map>
#include <sstream>
#include <string>

// totem includes
#include "totem_comdef.h"
#include "totem_generator.h"
#include "totem_graph.h"

PRIVATE generator_config_t config = {
  kAnalyzeCommand,     // Main command.
  kSummarySubCommand,  // Sub command.
  "",     // Input graph file.
  "",     // Output directory.
  20,     // Scale.
  16,     // Edge factor.
  false,  // Weighted.
  false,  // Do not verify direction.
  false,  // Execute the command rather than showing it's help message.
};

// Defines the signature of the command/sub-command handler function.
typedef void(*command_handler_func_t)(generator_config_t*);

// Forward declarations of all command and sub-command handlers.
PRIVATE void analyze_handler(generator_config_t* config);
PRIVATE void analyze_summary_handler(generator_config_t* config);
PRIVATE void analyze_degree_distribution_handler(generator_config_t* config);
PRIVATE void alter_handler(generator_config_t* config);
PRIVATE void alter_binary_handler(generator_config_t* config);
PRIVATE void alter_permute_handler(generator_config_t* config);
PRIVATE void alter_remove_singletons_handler(generator_config_t* config);
PRIVATE void alter_reverse_handler(generator_config_t* config);
PRIVATE void alter_sort_neighbours_handler(generator_config_t* config);
PRIVATE void alter_sort_vertices_handler(generator_config_t* config);
PRIVATE void alter_undirected_handler(generator_config_t* config);
PRIVATE void create_handler(generator_config_t* config);
PRIVATE void create_rmat_handler(generator_config_t* config);
PRIVATE void create_uniform_handler(generator_config_t* config);

// Maps each command/sub-command with its handler.
const std::map<std::string, command_handler_func_t> dispatch_map = {
  // "Analyze" command and sub-commands handlers.
  {kAnalyzeCommand, analyze_handler},
  {kSummarySubCommand, analyze_summary_handler},
  {kDegreeDistributionSubCommand, analyze_degree_distribution_handler},

  // "Alter" command and sub-commands handlers.
  {kAlterCommand, alter_handler},
  {kBinarySubCommand, alter_binary_handler},
  {kPermuteSubCommand, alter_permute_handler},
  {kRemoveSingletonsSubCommand, alter_remove_singletons_handler},
  {kReverseSubCommand, alter_reverse_handler},
  {kSortNeighboursSubCommand, alter_sort_neighbours_handler},
  {kSortVerticesSubCommand, alter_sort_vertices_handler},
  {kUndirectedSubCommand, alter_undirected_handler},

  // "Create" command and sub-commands handlers.
  {kCreateCommand, create_handler},
  {kRmatSubCommand, create_rmat_handler},
  {kUniformSubCommand, create_uniform_handler}
};

PRIVATE void dispatch(const std::string& command) {
  assert(dispatch_map.find(command) != dispatch_map.end());
  const command_handler_func_t handler = dispatch_map.find(command)->second;
  printf("Invoking %s command handler\n", command.c_str());
  handler(&config);
}

PRIVATE void alter_handler(generator_config_t* config) {
  dispatch(config->sub_command);
}

PRIVATE void write_graph(graph_t* graph, const char* graph_path) {
  printf("Writing graph file %s ", graph_path);
  CALL_SAFE(graph_store_binary(graph, graph_path));
  printf("done\n");
  CALL_SAFE(graph_finalize(graph));
}

PRIVATE void get_output_file_with_extension(generator_config_t* config,
                                            const std::string& ext,
                                            std::string* output_file) {
  if (config->output_directory.empty()) {
    output_file->assign(config->input_graph_file);
  } else {
    const std::string& input_graph_file = config->input_graph_file;
    std::string basename = input_graph_file.rfind("/") != std::string::npos ?
        input_graph_file.substr(input_graph_file.rfind("/") + 1) :
        input_graph_file;
    output_file->assign(config->output_directory);
    output_file->append("/");
    output_file->append(basename);
  }
  output_file->append(ext);
}

PRIVATE void write_graph_with_extension(generator_config_t* config,
                                        graph_t* graph,
                                        const std::string& ext) {
  std::string output_graph_file;
  get_output_file_with_extension(config, ext, &output_graph_file);
  write_graph(graph, output_graph_file.c_str());
}

PRIVATE void alter_binary_handler(generator_config_t* config) {
  graph_t* graph = NULL;
  CALL_SAFE(graph_initialize(config->input_graph_file.c_str(), false, &graph));
  write_graph_with_extension(config, graph, ".tbin");
}

PRIVATE void alter_permute_handler(generator_config_t* config) {
  graph_t* permuted_graph = NULL;
  if (generator_permute(config, &permuted_graph) == SUCCESS) {
    write_graph_with_extension(config, permuted_graph, ".permuted");
  }
}

PRIVATE void alter_remove_singletons_handler(generator_config_t* config) {
  graph_t* graph = NULL;
  CALL_SAFE(graph_initialize(config->input_graph_file.c_str(), false, &graph));
  graph_t* graph_no_singletons = NULL;
  if (graph_remove_singletons(graph, &graph_no_singletons) == SUCCESS) {
    write_graph_with_extension(config, graph_no_singletons, ".noSingletons");
  }
  graph_finalize(graph);
}

PRIVATE void alter_reverse_handler(generator_config_t* config) {
  graph_t* reversed_graph = NULL;
  if (generator_reverse(config, &reversed_graph) == SUCCESS) {
    write_graph_with_extension(config, reversed_graph, ".reversed");
  }
}

PRIVATE void alter_sort_neighbours_handler(generator_config_t* config) {
  graph_t* graph = NULL;
  CALL_SAFE(graph_initialize(config->input_graph_file.c_str(), false, &graph));
  graph_sort_nbrs(graph);
  write_graph_with_extension(config, graph, ".sortedNbrs");
}

PRIVATE void alter_sort_vertices_handler(generator_config_t* config) {
  graph_t* sorted_graph = NULL;
  if (generator_sort_vertices_by_degree(config, &sorted_graph) == SUCCESS) {
    write_graph_with_extension(config, sorted_graph, ".sortedVertices");
  }
}

PRIVATE void alter_undirected_handler(generator_config_t* config) {
  graph_t* undirected_graph = NULL;
  if (generator_undirected(config, &undirected_graph) == SUCCESS) {
    write_graph_with_extension(config, undirected_graph, ".undirected");
  }
}

PRIVATE void analyze_handler(generator_config_t* config) {
  dispatch(config->sub_command);
}

PRIVATE void analyze_summary_handler(generator_config_t* config) {
  printf("Checking graph %s\n", config->input_graph_file.c_str());
  std::string report = "";
  if (generator_check_and_summarize(config, &report) == SUCCESS) {
    printf("Passed\n");
  } else {
    printf("Failed!\n");
  }
  printf("\nSummary Report:\n===============\n%s", report.c_str());
}

PRIVATE void analyze_degree_distribution_handler(generator_config_t* config) {
  eid_t highest_degree = 0;
  eid_t* degree_distribution = NULL;
  generator_degree_distribution(config, &degree_distribution, &highest_degree);
  if (!degree_distribution) { return; }

  std::string degree_file;
  get_output_file_with_extension(config, ".degreeDist", &degree_file);

  printf("Writing file %s ", degree_file.c_str());
  FILE* file_handler = fopen(degree_file.c_str(), "w");
  fprintf(file_handler, "degree\tvertex_count\n");
  for (eid_t degree = 0; degree < highest_degree; degree++) {
    if (degree_distribution[degree]) {
      fprintf(file_handler, "%llu\t%llu\n", (uint64_t)degree,
              (uint64_t)degree_distribution[degree]);
    }
  }
  fclose(file_handler);
  printf("done\n");
  free(degree_distribution);
}

PRIVATE void create_handler(generator_config_t* config) {
  dispatch(config->sub_command);
}

PRIVATE void create_rmat_handler(generator_config_t* config) {
  const double kA = 0.57;
  const double kB = 0.19;
  const double kC = 0.19;
  graph_t* graph;
  if (generator_create_rmat(config, kA, kB, kC, &graph) != SUCCESS) {
    printf("Creating an RMAT graph failed!\n");
    return;
  }
  write_graph(graph, config->input_graph_file.c_str());
}

PRIVATE void create_uniform_handler(generator_config_t* config) {
  graph_t* graph;
  if (generator_create_uniform(config, &graph) != SUCCESS) {
    printf("Creating a uniform graph failed!\n");
    return;
  }
  write_graph(graph, config->input_graph_file.c_str());
}

int main(int argc, char** argv) {
  parse_command_line(argc, argv, &config);
  dispatch(config.command);
  return 0;
}
