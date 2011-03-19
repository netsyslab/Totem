/* TODO(lauro,abdullah,elizeu): Add license.
 *
 * Contains unit tests for totem helper functions.
 *
 *  Created on: 2011-03-09
 *      Author: Lauro Beltrão Costa
 */

// totem includes
#include "totem_common_unittest.h"

// Tests for initialize helper function.
TEST(GraphHelper, Initialize) {
  graph_t* graph;

  graph_initialize(DATA_FOLDER("single_node.totem"), false, &graph);
  EXPECT_EQ((uint32_t)1, graph->vertex_count);
  EXPECT_EQ((uint32_t)0, graph->edge_count);
  graph_finalize(graph);

  graph_initialize(DATA_FOLDER("single_node_loop.totem"), false, &graph);
  EXPECT_EQ((uint32_t)1, graph->vertex_count);
  EXPECT_EQ((uint32_t)1, graph->edge_count);
  graph_finalize(graph);

  graph_initialize(DATA_FOLDER("chain_1000_nodes.totem"), false, &graph);
  EXPECT_EQ((uint32_t)1000, graph->vertex_count);
  EXPECT_EQ((uint32_t)1998, graph->edge_count);
  graph_finalize(graph);

  graph_initialize(DATA_FOLDER("complete_graph_300_nodes.totem"), false, 
                   &graph);
  EXPECT_EQ((uint32_t)300, graph->vertex_count);
  EXPECT_EQ((uint32_t)89700, graph->edge_count);
  graph_finalize(graph);

  // TODO(lauro, abdullah): Add more cases and test other fields.
}
