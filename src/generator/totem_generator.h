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
extern const char* kRandomWeightsSubCommand;

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
 * ALTER command handler. It invokes the specific ALTER sub-command handler.
 * @param[in] config specifies the options of the specific ALTER operation.
 */
void alter_handler(generator_config_t* config);

/**
 * ANALYZE command handler. It invokes the specific ANALYZE sub-command handler.
 * @param[in] config specifies the options of the specific ANALYZE operation.
 */
void analyze_handler(generator_config_t* config);

/**
 * CREATE command handler. It invokes the specific CREATE sub-command handler.
 * @param[in] config specifies the options of the specific CREATE operation.
 */
void create_handler(generator_config_t* config);

#endif  // TOTEM_GENERATOR_H
