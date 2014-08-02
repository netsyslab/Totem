/**
 * Defines the generator's data types and constants.
 *
 *  Created on: 2014-02-28
 *  Author: Abdullah Gharaibeh
 */

#ifndef TOTEM_GENERATOR_H
#define TOTEM_GENERATOR_H

// system includes
#include <string>

// totem includes
#include "totem_comdef.h"
#include "totem_graph.h"

// The tool's configuration parameters.
typedef struct generator_config_s {
  std::string command;
  std::string sub_command;
  std::string input_graph_file;
  std::string output_directory;
  int   scale;
  int   edge_factor;
  bool  weighted;
  bool  check_direction;
  bool  command_help;
} generator_config_t;

// Declarations of the constants that defines the set of commands and
// sub-commands supported by the tool.

// Analyze command and sub-commands.
extern const char* kAnalyzeCommand;
extern const char* kSummarySubCommand;
extern const char* kDegreeDistributionSubCommand;

// Alter command and sub-commands.
extern const char* kAlterCommand;
extern const char* kBinarySubCommand;
extern const char* kPermuteSubCommand;
extern const char* kRemoveSingletonsSubCommand;
extern const char* kReverseSubCommand;
extern const char* kSortNeighboursSubCommand;
extern const char* kSortVerticesSubCommand;
extern const char* kUndirectedSubCommand;

// Create command and sub-commands.
extern const char* kCreateCommand;
extern const char* kRmatSubCommand;
extern const char* kUniformSubCommand;

// Declares the maximum scale the graph generator supports (which translates to
// 2^scale number of vertices).
extern const int kMaxVertexScale;

/**
 * Parses command line options and arguments.
 * @param[in] argc number of arguments
 * @param[in] argv argument array
 */
void parse_command_line(int argc, char** argv, generator_config_t* config);

/**
 * Performs sanity check on the graph and produces summary information regarding
 * its characteristics.
 * @param[in] config specifies the graph to be checked and the checking options
 */
error_t generator_check_and_summarize(generator_config_t* config,
                                      std::string* report);

/**
 * Generates the degree distribution of the graph.
 * @param[in] config specifies the graph to be analyzed
 * @param[in] degree_distribution degree distribution array
 * @param[in] highest_degree the highest degree and the length of the degree
 *                           distribution array
 */
error_t generator_degree_distribution(generator_config_t* config,
                                      eid_t** degree_distribution,
                                      eid_t* highest_degree);

/**
 * Creates an RMAT graph.
 * @param[in] config specifies the size of the graph to be generated
 * @param[in] a,b,c RMAT configuration parameters
 * @param[out] graph the generated graph
 */
error_t generator_create_rmat(generator_config_t* config, double a, double b,
                              double c, graph_t** graph);

/**
 * Creates a graph with uniform edge distribution.
 * @param[in] config specifies the size of the graph to be generated
 * @param[out] graph the generated graph
 */
error_t generator_create_uniform(generator_config_t* config, graph_t** graph);

/**
 * Creates a new graph from an existing one after permuting the ids of its
 * vertices.
 * @param[in] config specifies the source graph
 * @param[out] graph the permuted graph
 */
error_t generator_permute(generator_config_t* config, graph_t** permuted_graph);

/**
 * Creates a new graph from an existing one after reversing the direction of
 * each edge.
 * @param[in] config specifies the source graph
 * @param[out] reversed_graph the reversed graph
 */
error_t generator_reverse(generator_config_t* config, graph_t** reversed_graph);

/**
 * Creates a new graph from an existing one after permuting the vertex ids such
 * that they are sorted by degree.
 * @param[in] config specifies the source graph
 * @param[out] sorted_graph the sorted graph
 */
error_t generator_sort_vertices_by_degree(generator_config_t* config,
                                          graph_t** sorted_graph);

/**
 * Creates a new undirected graph from an existing directed one.
 * @param[in] config specifies the source graph
 * @param[out] reversed_graph the reversed graph
 */
error_t generator_undirected(generator_config_t* config,
                             graph_t** undirected_graph);

#endif  // TOTEM_GENERATOR_H
