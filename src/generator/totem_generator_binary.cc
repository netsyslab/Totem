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

int main(int argc, char** argv) {
  generator_config_t config = {
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
  parse_command_line(argc, argv, &config);

  // Defines the signature of the command/sub-command handler function.
  typedef void(*command_handler_func_t)(generator_config_t*);

  // Maps each command/sub-command with its handler.
  const std::map<std::string, command_handler_func_t> dispatch_map = {
    {kAnalyzeCommand, analyze_handler},
    {kAlterCommand, alter_handler},
    {kCreateCommand, create_handler},
  };

  const auto& handler = dispatch_map.find(config.command);
  assert(handler != dispatch_map.end());
  printf("Invoking %s command handler.\n", config.command.c_str());
  handler->second(&config);
  return 0;
}
