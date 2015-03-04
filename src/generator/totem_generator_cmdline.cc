/**
 * Parser of the command line options and arguments for the graph generation
 * tool.
 *
 *  Created on: 2014-07-28
 *  Author: Abdullah Gharaibeh
 */

// system includes
#include <set>
#include <string>
#include <map>

// totem includes
#include "totem_comdef.h"
#include "totem_generator.h"
#include "totem_graph.h"

// The set of commands supported by the tool. Each one should be associated
// with a help message.
const char* kAnalyzeCommand = "ANALYZE";
const char* kSummarySubCommand = "SUMMARY";
const char* kDegreeDistributionSubCommand = "DEGREE-DIST";

const char* kAlterCommand = "ALTER";
const char* kBinarySubCommand = "BINARY";
const char* kPermuteSubCommand = "PERMUTE";
const char* kReverseSubCommand = "REVERSE";
const char* kRemoveSingletonsSubCommand = "REMOVE-SINGLETONS";
const char* kSortNeighboursSubCommand = "SORT-NBRS";
const char* kSortVerticesSubCommand = "SORT-VERTICES";
const char* kUndirectedSubCommand = "UNDIRECTED";
const char* kRandomWeightsSubCommand = "RANDOM-WEIGHTS";

const char* kCreateCommand = "CREATE";
const char* kRmatSubCommand = "RMAT";
const char* kUniformSubCommand = "UNIFORM";

// Maps the commands supported by the tool to their associated set of
// sub-commands.
PRIVATE const std::map<std::string, std::set<std::string> > commands = {
  {kAnalyzeCommand, {kSummarySubCommand, kDegreeDistributionSubCommand}},
  {kAlterCommand, {kBinarySubCommand, kPermuteSubCommand, kReverseSubCommand,
                   kRemoveSingletonsSubCommand, kSortNeighboursSubCommand,
                   kSortVerticesSubCommand, kUndirectedSubCommand,
                   kRandomWeightsSubCommand}},
  {kCreateCommand, {kRmatSubCommand, kUniformSubCommand}}
};

// Maps each command/sub-command with its specific help message.
const std::map<std::string, std::string> help_map = {
  // "Analyze" command and sub-commands handlers.
  {
    kSummarySubCommand,
    "\tPerforms sanity check on the graph and prints out a number of\n"
    "\tcharacteristics such as the number of edges and vertices.\n"
  },
  {
    kDegreeDistributionSubCommand,
    "\tGenerates the degree distribution of the graph. The generated file\n"
    "\tcontains for each degree the number of vertices with that degree.\n"
  },

  // "Alter" command and sub-commands handlers.
  {
    kBinarySubCommand,
    "\tStores the given, presumably text-based, graph file in Totem's binary\n"
    "\tgraph format.\n"
  },
  {
    kPermuteSubCommand,
    "\tGenerates a new graph from the given one after randomly permuting the\n"
    "\tids of its vertices.\n"
  },
  {
    kRemoveSingletonsSubCommand,
    "\tGenerates a new graph from the given one after removing all vertices\n"
    "\twith neither incoming nor outgoing edges.\n"
  },
  {
    kReverseSubCommand,
    "\tGenerates a new graph from the given one after reversing the direction\n"
    "\tof each edge.\n"
  },
  {
    kSortNeighboursSubCommand,
    "\tGenerates a new graph from the given one after sorting by id the\n"
    "\tneighbours of each vertex.\n"
  },
  {
    kSortVerticesSubCommand,
    "\tGenerates a new graph from the given one after permuting the vertex\n"
    "\tids such that they are ordered by degree.\n"
  },
  {
    kUndirectedSubCommand,
    "\tGenerates a new undirected graph from the given one.\n"
  },
  {
    kRandomWeightsSubCommand,
    "\tGenerates a new graph with random weights attached to its edgs.\n"
  },

  // "Create" command and sub-commands handlers.
  {
    kRmatSubCommand,
    "\tCreates a new graph using the RMAT graph generation algorithm,\n"
    "\twhich generates random graphs with power-law degree distribution. The\n"
    "\tparameters are fixed at a=0.57 b=0.19 c=0.19 d=.05. The -sNUM and\n"
    "\t-eNUM options allow choosing the scale and the edge factor,\n"
    "\trespectively. The number of vertices will be 2^scale, while the number\n"
    "\tof edges is edge_factor*2^scale.\n"
  },
  {
    kUniformSubCommand,
    "\tCreates a new graph with uniform degree distribution. The -sNUM and\n"
    "\t-eNUM options allow choosing the scale and the edge factor,\n"
    "\trespectively. The number of vertices will be 2^scale, while the number\n"
    "\tof edges is edge_factor * 2^scale.\n"
  }
};

PRIVATE void display_help(char* exe_name, int exit_err,
                          const char* err_msg = NULL) {
  if (err_msg) { fprintf(stderr, "Error: "); fprintf(stderr, err_msg); }
  std::string exe_name_str(exe_name);
  std::string exe_basename = exe_name_str.rfind("/") != std::string::npos ?
      exe_name_str.substr(exe_name_str.rfind("/") + 1) : exe_name_str;
  printf("\nUsage: %s <command> [sub-command] [options] <graph file>\n"
         "\nCommands:\n"
         "ANALYZE {%s | %s} <graph file>\n"
         "ALTER {%s | %s | %s | %s | %s |\n"
         "       %s | %s | %s} <graph file>\n"
         "CREATE {%s | %s} [options] <graph file>\n"
         "\nOptions (the applicable command is indicated between <>):\n"
         "  -sNUM   <CREATE> Scale of the number of vertices (default 20)\n"
         "  -eNUM   <CREATE> Edge factor (default 16)\n"
         "  -w      <CREATE> Generate random edge weights\n"
         "  -d      <ANALYZE SUMMARY> Checks if the graph is undirected by\n"
         "          examining that each edge exists in both directions (this\n"
         "          might be time consuming for large graphs)\n"
         "  -oPATH  <ALTER | ANALYZE DEGREE-DIST> Path to the output\n"
         "          directory for the altered graph or the generated degree\n"
         "          distribution.\n"
         "  -h      Print this help message\n\n"
         "Note 1: \"%s <command> [sub-command] help\" shows command-specific\n"
         "          help message.\n"
         "Note 2: ALTER and ANALYZE DEGREE-DIST commands require write access\n"
         "        to the directory of the source graph if the -o option is\n"
         "        not set.\n",
         exe_basename.c_str(),
         kSummarySubCommand, kDegreeDistributionSubCommand, kBinarySubCommand,
         kPermuteSubCommand, kRemoveSingletonsSubCommand, kReverseSubCommand,
         kSortNeighboursSubCommand, kSortVerticesSubCommand,
         kUndirectedSubCommand, kRandomWeightsSubCommand, kRmatSubCommand,
         kUniformSubCommand, exe_basename.c_str());

  if (exit_err == 0) {
    printf("\n\nDetailed descriptions of commands:\n");
    for (const auto& command : commands) {
      if (command.second.empty()) {
        printf("%s\n%s", command.first.c_str(),
               help_map.find(command.first)->second.c_str());
      } else {
        for (const auto& sub_command : command.second) {
          printf("%s %s\n%s", command.first.c_str(),
                 sub_command.c_str(),
                 help_map.find(sub_command)->second.c_str());
        }
      }
    }
  }

  exit(exit_err);
}

PRIVATE void parse_command_line_arguments(int argc, char** argv,
                                          generator_config_t* config) {
  // Check the command.
  if (optind == argc) { display_help(argv[0], -1, "Missing command!\n"); }
  config->command.assign(argv[optind++]);
  std::transform(config->command.begin(), config->command.end(),
                 config->command.begin(), ::toupper);
  if (commands.find(config->command) == commands.end()) {
    display_help(argv[0], -1, "Invalid command!\n");
  }

  // Check the sub command.
  const auto& sub_commands = commands.find(config->command)->second;
  if (!sub_commands.empty()) {
    if (optind == argc) { display_help(argv[0], -1, "Missing sub-command!\n"); }
    config->sub_command.assign(argv[optind++]);
    std::transform(config->sub_command.begin(), config->sub_command.end(),
                   config->sub_command.begin(), ::toupper);
    if (sub_commands.find(config->sub_command) == sub_commands.end()) {
      display_help(argv[0], -1, "Invalid sub-command!\n");
    }
  }

  // The last argument should be either the path to the graph file, or an
  // indication to print the command's specific help message.
  if (optind == argc) { display_help(argv[0], -1, "Missing arguments!\n"); }
  std::string arg(argv[optind]);
  std::transform(arg.begin(), arg.end(), arg.begin(), ::toupper);
  if (arg == "HELP") { config->command_help = true;
  } else { config->input_graph_file.assign(argv[optind]); }
  optind++;

  // There should be no more arguments.
  if (optind != argc) { display_help(argv[0], -1, "Invalid arguments!\n"); }
}

PRIVATE void parse_command_line_options(int argc, char** argv,
                                        generator_config_t* config) {
  optarg = NULL;
  int ch;
  while (((ch = getopt(argc, argv, "he:s:wdo:")) != EOF)) {
    switch (ch) {
      case 'e':
        config->edge_factor = atoi(optarg);
        if (config->edge_factor < 1) {
          display_help(argv[0], -1, "Invalid edge factor\n");
        }
        break;
      case 's':
        config->scale = atoi(optarg);
        if (config->scale < 1 || config->scale > kMaxVertexScale) {
          display_help(argv[0], -1, "Invalid scale\n");
        }
        break;
      case 'w':
        config->weighted = true;
        break;
      case 'd':
        config->check_direction = true;
        break;
      case 'o':
        config->output_directory.assign(optarg);
        break;
      case 'h':
        display_help(argv[0], 0);
        break;
      default:
        display_help(argv[0], -1);
    }
  }
}

void parse_command_line(int argc, char** argv, generator_config_t* config) {
  parse_command_line_options(argc, argv, config);
  parse_command_line_arguments(argc, argv, config);
  if (config->command_help) {
    printf("%s %s\n", config->command.c_str(),
           config->sub_command.c_str());
    std::string help_msg = config->sub_command.empty() ?
        help_map.find(config->command)->second :
        help_map.find(config->sub_command)->second;
    printf("%s", help_msg.c_str());
    exit(0);
  }
}
